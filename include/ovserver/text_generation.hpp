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
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/runtime/tensor.hpp>

namespace ovserver {

// Text generation request. `prompt` is the concatenated user text; `images` are
// RGB u8 [H,W,3] tensors attached to the request. All numeric options are
// optional so the pipeline's defaults are preserved unless a client overrides.
struct TextGenerateOptions {
    std::string prompt;
    std::vector<ov::Tensor> images;
    std::string system_message;
    std::optional<std::size_t> max_new_tokens;
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<std::size_t> top_k;
    std::optional<std::size_t> rng_seed;

    // Streaming: invoked with each decoded text fragment as it is produced.
    // Return false to cancel generation (client abort / finish).
    std::function<bool(std::string)> on_text;
    // Invoked once at the very end with the final full text; always called even
    // in non-streaming mode.
    std::function<void(std::string)> on_done;
};

struct TextResult {
    std::string text;
    std::string finish_reason;  // "stop" | "length" | "abort"
};

// Wraps an ov::genai::ContinuousBatchingPipeline (e.g. Qwen2.5-VL,
// Qwen3-VL) exported to OpenVINO IR. Requests submit via add_request() from
// their own worker threads while a single background executor thread advances
// the scheduler with step(), which batches concurrent requests into a shared
// KV cache.
class TextGenerationModel {
public:
    TextGenerationModel(const std::string& id,
                        const std::filesystem::path& models_path,
                        const std::string& device);

    TextGenerationModel(const TextGenerationModel&) = delete;
    TextGenerationModel& operator=(const TextGenerationModel&) = delete;

    ~TextGenerationModel();

    const std::string& id() const { return m_id; }

    TextResult generate(const TextGenerateOptions& opts);

private:
    void executor_run();
    std::uint64_t next_request_id();
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
};

}  // namespace ovserver