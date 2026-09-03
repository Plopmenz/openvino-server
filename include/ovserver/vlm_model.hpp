// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <openvino/core/any.hpp>
#include <openvino/genai/continuous_batching_pipeline.hpp>
#include <openvino/genai/generation_config.hpp>
#include <openvino/genai/scheduler_config.hpp>
#include <openvino/genai/tokenizer.hpp>
#include <openvino/runtime/tensor.hpp>

namespace ovserver {

// VLM generation request. `prompt` is the concatenated user text; `images` are
// RGB u8 [H,W,3] tensors attached to the request. All numeric options are
// optional so the pipeline's defaults are preserved unless a client overrides.
struct VLMGenerateOptions {
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

struct VLMResult {
    std::string text;
    std::string finish_reason;  // "stop" | "length" | "abort"
};

// Tuning knobs for the continuous-batching scheduler, mirroring OVMS's
// LLMCalculatorOptions proto defaults exactly. cache_size=0 selects dynamic
// (unbounded) KV-cache allocation, which is OVMS's default.
struct VLMSchedulerOptions {
    std::size_t max_num_batched_tokens = 256;
    std::size_t max_num_seqs = 256;
    std::size_t cache_size = 10;       // GB; 0 => dynamic KV cache allocation
    bool dynamic_split_fuse = true;   // required for prompts longer than a batch
    bool enable_prefix_caching = false;
};

// Wraps an ov::genai::ContinuousBatchingPipeline (e.g. Qwen3.6-35B-A3B,
// Qwen2.5-VL) exported to OpenVINO IR. Unlike VLMPipeline, the CB backend
// schedules many concurrent requests into shared KV-cache batches: requests
// submit via add_request() from their own worker threads while a single
// background executor thread advances the scheduler with step(). This gives
// real multi-request batching / concurrency.
class VLMModel {
public:
    VLMModel(const std::string& id,
             const std::filesystem::path& models_path,
             const std::string& device,
             const ov::AnyMap& properties = {},
             const VLMSchedulerOptions& scheduler = {},
             unsigned rest_workers = 4);

    VLMModel(const VLMModel&) = delete;
    VLMModel& operator=(const VLMModel&) = delete;

    ~VLMModel();

    const std::string& id() const { return m_id; }

    VLMResult generate(const VLMGenerateOptions& opts);

private:
    void executor_run();
    std::uint64_t next_request_id();

    std::string m_id;
    std::filesystem::path m_models_path;
    std::string m_device;
    ov::AnyMap m_properties;
    VLMSchedulerOptions m_sched;
    unsigned m_rest_workers = 4;

    std::shared_ptr<ov::genai::ContinuousBatchingPipeline> m_pipeline;
    ov::genai::Tokenizer m_tokenizer;

    std::thread m_executor;
    std::atomic<bool> m_stop{false};
    std::mutex m_cv_mutex;
    std::condition_variable m_cv;

    std::mutex m_id_mutex;
    std::uint64_t m_next_id = 0;
};

struct VLMModelSpec {
    std::filesystem::path path;
    std::string device;
    ov::AnyMap properties;
    VLMSchedulerOptions scheduler;
    unsigned rest_workers = 4;
};

}  // namespace ovserver
