// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <openvino/genai/chat_history.hpp>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/json_container.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/runtime/tensor.hpp>

namespace ovserver {

struct TextGenerationSpec;

// Text generation request. `prompt` is the concatenated user text; `images` are
// RGB u8 [H,W,3] tensors attached to the request. All numeric options are
// optional so the pipeline's defaults are preserved unless a client overrides.
struct TextGenerateOptions {
    std::string prompt;
    std::vector<ov::Tensor> images;
    std::string system_message;
    // Full conversation history as OpenAI-style message objects
    // ({role, content, ...}). When non-empty it replaces `prompt` /
    // `system_message` so multi-turn and tool-call exchanges are preserved
    // through the model's chat template.
    std::vector<ov::genai::JsonContainer> chat_messages;
    // OpenAI "tools" definitions (function calling). When set they are injected
    // into the chat template via ChatHistory::set_tools().
    std::optional<ov::genai::JsonContainer> tools;
    // OpenAI "response_format": constrains generation so the output matches a
    // JSON schema / regex / EBNF grammar (structured output, xgrammar backend).
    std::optional<ov::genai::StructuredOutputConfig> structured_output;
    std::optional<std::size_t> max_new_tokens;
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<std::size_t> top_k;
    std::optional<std::size_t> rng_seed;
    // OpenAI "stop": strings that end generation.
    std::set<std::string> stop_strings;
    // OpenAI penalties, each clamped to [-2, 2] by the caller.
    std::optional<float> frequency_penalty;
    std::optional<float> presence_penalty;

    // Streaming: invoked with each decoded text fragment as it is produced.
    // Return false to cancel generation (client abort / finish).
    std::function<bool(std::string)> on_text;
    // Streaming: invoked with each decoded reasoning fragment (extracted from
    // the model's thinking markers) as it is produced. Return false to cancel.
    std::function<bool(std::string)> on_reasoning;
    // Invoked once at the very end with the final full text; always called even
    // in non-streaming mode.
    std::function<void(std::string)> on_done;
};

// One parsed OpenAI function call from the assistant output.
struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;  // JSON object as a string
};

struct TextResult {
    std::string text;                     // visible content (thinking split off)
    std::string reasoning_content;        // text inside thinking markers
    std::vector<ToolCall> tool_calls;     // parsed function calls, if any
    std::string finish_reason;  // "stop" | "length" | "abort" | "tool_calls"
    std::size_t prompt_tokens = 0;
    std::size_t completion_tokens = 0;
};

// Reasoning marker configuration detected for a model at load time.
struct ReasoningMarkers {
    // True when the model is expected to emit thinking markers and we should
    // split them into reasoning_content.
    bool enabled = false;
    // When false, reasoning begins immediately and the open tag is already part
    // of the prompt (e.g. DeepSeek-R1 distill models).
    bool expect_open_tag = true;
    std::string open_tag;
    std::string close_tag;
};

// Wraps an ov::genai::ContinuousBatchingPipeline (e.g. Qwen2.5-VL,
// Qwen3-VL) exported to OpenVINO IR. Requests submit via add_request() from
// their own worker threads while a single background executor thread advances
// the scheduler with step(), which batches concurrent requests into a shared
// KV cache.
class TextGenerationModel {
public:
    TextGenerationModel(const std::string& id,
                        const TextGenerationSpec& spec);

    TextGenerationModel(const TextGenerationModel&) = delete;
    TextGenerationModel& operator=(const TextGenerationModel&) = delete;

    ~TextGenerationModel();

    const std::string& id() const { return m_id; }

    // Detected reasoning markers; empty open_tag means "no reasoning split".
    const ReasoningMarkers& reasoning_markers() const { return m_reasoning; }
    // True when the model's chat template supports <tool_call> function-calling.
    bool tools_supported() const { return m_tools_supported; }

    TextResult generate(const TextGenerateOptions& opts);

private:
    void executor_run();
    std::uint64_t next_request_id();
    void detect_parsers();
    void snapshot_log(std::chrono::steady_clock::time_point now,
                      std::chrono::steady_clock::time_point window_start);

    // Per-request inference metrics. Worker threads accumulate `tokens`; the
    // executor thread samples them once per second to compute aggregate rates.
    struct RequestMetrics {
        std::chrono::steady_clock::time_point start;
        std::chrono::steady_clock::time_point baseline_time;
        std::size_t tokens = 0;           // generated tokens so far
        std::size_t baseline_tokens = 0;  // generated tokens at last snapshot
        std::size_t prompt_tokens = 0;    // encoded prompt length
        bool has_baseline = false;
    };

    std::string m_id;
    std::filesystem::path m_models_path;
    std::string m_device;

    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> m_pipeline;
    ov::genai::Tokenizer m_tokenizer;
    ReasoningMarkers m_reasoning;
    bool m_tools_supported = false;
    // Speculative-decoding strategy active for this pipeline (drives which
    // GenerationConfig fields generate() sets per request).
    bool m_prompt_lookup_active = false;
    bool m_mtp_active = false;
    std::size_t m_num_assistant_tokens = 5;
    std::size_t m_max_ngram_size = 3;

    std::thread m_executor;
    std::atomic<bool> m_stop{false};
    std::mutex m_cv_mutex;
    std::condition_variable m_cv;

    std::mutex m_id_mutex;
    std::uint64_t m_next_id = 0;

    std::mutex m_metrics_mutex;
    std::unordered_map<std::uint64_t, RequestMetrics> m_requests;
    // Prompt tokens whose prefill completed since the last snapshot (credited
    // by worker threads when a request transitions prefill -> decode).
    std::size_t m_prefill_tokens_in_window = 0;
    std::chrono::steady_clock::time_point m_last_snapshot;
};

struct TextGenerationSpec {
    std::filesystem::path path;
    std::string device;
    // GPU compile-time knobs (OVMS defaults). Applied only on GPU devices.
    std::string kv_cache_precision = "u8";
    std::uint64_t dynamic_quant_group_size = 32;
    bool enable_sdpa_optimization = true;
    // Scheduler knob for linear-attention prefix caching (ignored by models
    // without linear attention layers).
    std::size_t cache_interval_multiplier = 64;
    // Retain KV blocks across requests for prompt-prefix reuse (better TTFT).
    bool enable_prefix_caching = true;
    // Prompt-lookup speculative decoding (no extra model needed).
    bool prompt_lookup = false;
    std::size_t num_assistant_tokens = 5;
    std::size_t max_ngram_size = 3;
    // Auto-detect bundled MTP head (openvino_mtp_model.xml) in model dir.
    bool enable_mtp = true;
};

}  // namespace ovserver