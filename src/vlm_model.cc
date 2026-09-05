// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/vlm_model.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <openvino/genai/scheduler_config.hpp>
#include <openvino/genai/text_streamer.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/core.hpp>

namespace ovserver {

namespace {

// Mirrors OVMS's ovms::applyDefaultCpuProperties: sensible CPU execution
// defaults that are only meaningful on a CPU device.
void apply_default_cpu_properties(ov::AnyMap& props, unsigned core_count) {
    if (props.find(ov::hint::enable_cpu_pinning.name()) == props.end()) {
        props[ov::hint::enable_cpu_pinning.name()] = false;
    }
    if (props.find(ov::inference_num_threads.name()) == props.end() &&
        core_count > 0) {
        props[ov::inference_num_threads.name()] = static_cast<int>(core_count);
    }
}

// Mirrors OVMS's ContinuousBatchingServableInitializer: the tokenizer /
// detokenizer always run on CPU with a THROUGHPUT perf mode and a stream count
// bounded by the REST worker count, passed to the pipeline as the dedicated
// tokenizer plugin config (5th constructor argument).
ov::AnyMap make_tokenizer_props(unsigned rest_workers, unsigned core_count) {
    ov::AnyMap props;
    props[ov::num_streams.name()] = static_cast<int>(
        std::min(rest_workers, core_count));
    props[ov::hint::performance_mode.name()] =
        ov::hint::PerformanceMode::THROUGHPUT;
    apply_default_cpu_properties(props, core_count);
    return props;
}

std::string finish_reason_str(ov::genai::GenerationFinishReason r) {
    switch (r) {
        case ov::genai::GenerationFinishReason::LENGTH:
            return "length";
        case ov::genai::GenerationFinishReason::STOP:
        default:
            return "stop";
    }
}

ov::genai::SchedulerConfig make_scheduler(const VLMSchedulerOptions& o) {
    ov::genai::SchedulerConfig cfg;
    cfg.max_num_batched_tokens = o.max_num_batched_tokens;
    cfg.max_num_seqs = o.max_num_seqs;
    cfg.cache_size = o.cache_size;
    cfg.dynamic_split_fuse = o.dynamic_split_fuse;
    cfg.enable_prefix_caching = o.enable_prefix_caching;
    return cfg;
}

bool is_gpu_device(const std::string& device) {
    return device.find("GPU") != std::string::npos;
}

// Build the plugin properties map for the pipeline.
// For GPU: set ATTENTION_BACKEND=SDPA so the pipeline internally handles
// shape bounding at the GenAI layer.
ov::AnyMap build_plugin_props(const ov::AnyMap& properties,
                              const ov::genai::SchedulerConfig& sched_cfg,
                              const std::string& device) {
    ov::AnyMap props = properties;
    if (is_gpu_device(device)) {
        props["ATTENTION_BACKEND"] = "SDPA";
    }
    // Activate continuous batching mode via the scheduler_config property.
    // This is how OVMS activates CB pipeline through LLMPipeline.
    props[ov::genai::scheduler_config.name()] = sched_cfg;
    return props;
}

}  // namespace

VLMModel::VLMModel(const std::string& id,
                   const std::filesystem::path& models_path,
                   const std::string& device,
                   const ov::AnyMap& properties,
                   const VLMSchedulerOptions& scheduler,
                   unsigned rest_workers)
    : m_id(id),
      m_models_path(models_path),
      m_device(device),
      m_properties(properties),
      m_sched(scheduler),
      m_rest_workers(rest_workers) {

    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    ov::genai::SchedulerConfig sched_cfg = make_scheduler(scheduler);

    std::cerr << "[vlm model '" << id << "'] constructing pipeline"
              << " device=" << device
              << " max_num_batched_tokens=" << sched_cfg.max_num_batched_tokens
              << " num_kv_blocks=" << sched_cfg.num_kv_blocks
              << " cache_size=" << sched_cfg.cache_size
              << " dynamic_split_fuse=" << sched_cfg.dynamic_split_fuse
              << " enable_prefix_caching=" << sched_cfg.enable_prefix_caching
              << std::endl;

        std::cerr << "[vlm model '" << id << "'] CPU mode: CB pipeline"
                  << std::endl;
        m_pipeline = std::make_shared<ov::genai::ContinuousBatchingPipeline>(
            models_path, sched_cfg, device,
            properties,
            make_tokenizer_props(rest_workers, cores));

    m_tokenizer = m_pipeline->get_tokenizer();

    std::cerr << "[vlm model '" << id << "'] loaded successfully" << std::endl;
    m_executor = std::thread([this] { executor_run(); });
}

VLMModel::~VLMModel() {
    m_stop = true;
    {
        std::lock_guard<std::mutex> lock(m_cv_mutex);
        m_cv.notify_all();
    }
    if (m_executor.joinable()) {
        m_executor.join();
    }
}

std::uint64_t VLMModel::next_request_id() {
    std::lock_guard<std::mutex> lock(m_id_mutex);
    return m_next_id++;
}

void VLMModel::executor_run() {
    while (!m_stop) {
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait(lock, [this] {
            return m_stop || m_pipeline->has_non_finished_requests();
        });
        if (m_stop) {
            return;
        }
        lock.unlock();
        try {
            while (!m_stop && m_pipeline->has_non_finished_requests()) {
                m_pipeline->step();
            }
        } catch (const std::exception& e) {
            std::cerr << "[vlm model '" << m_id
                      << "'] scheduler step error: " << e.what() << std::endl;
        }
    }
}

VLMResult VLMModel::generate(const VLMGenerateOptions& opts) {
    ov::genai::GenerationConfig cfg;
    if (opts.max_new_tokens) cfg.max_new_tokens = *opts.max_new_tokens;
    if (opts.temperature) {
        cfg.temperature = *opts.temperature;
        cfg.do_sample = true;
    }
    if (opts.top_p) {
        cfg.top_p = *opts.top_p;
        cfg.do_sample = true;
    }
    if (opts.top_k) {
        cfg.top_k = *opts.top_k;
        cfg.do_sample = true;
    }
    if (opts.rng_seed) cfg.rng_seed = *opts.rng_seed;

    // Build the templated prompt so the model's own chat_template is applied
    ov::genai::ChatHistory history;
    if (!opts.system_message.empty()) {
        history.push_back({{"role", "system"}, {"content", opts.system_message}});
    }
    history.push_back({{"role", "user"}, {"content", opts.prompt}});
    const std::string templated =
        m_tokenizer.apply_chat_template(history, /*add_generation_prompt=*/true);

    const std::uint64_t req_id = next_request_id();
    ov::genai::GenerationHandle handle =
        m_pipeline->add_request(req_id, templated, opts.images, cfg);
    {
        std::lock_guard<std::mutex> lock(m_cv_mutex);
        m_cv.notify_one();
    }

    std::string accumulated;
    ov::genai::GenerationFinishReason finish =
        ov::genai::GenerationFinishReason::NONE;
    auto on_word = [&accumulated, &opts](std::string word)
        -> ov::genai::CallbackTypeVariant {
        accumulated += word;
        if (opts.on_text && !opts.on_text(std::move(word))) {
            return ov::genai::StreamingStatus::CANCEL;
        }
        return ov::genai::StreamingStatus::RUNNING;
    };
    auto streamer =
        std::make_shared<ov::genai::TextStreamer>(m_tokenizer, on_word);

    bool aborted = false;
    while (handle->get_status() == ov::genai::GenerationStatus::RUNNING ||
           handle->can_read()) {
        ov::genai::GenerationOutputs outputs = handle->read();
        for (auto& [rid, out] : outputs) {
            if (out.finish_reason != ov::genai::GenerationFinishReason::NONE) {
                finish = out.finish_reason;
            }
            for (const int64_t tid : out.generated_ids) {
                const ov::genai::StreamingStatus st = streamer->write(tid);
                if (st == ov::genai::StreamingStatus::CANCEL) {
                    handle->cancel();
                    aborted = true;
                    break;
                }
                if (st != ov::genai::StreamingStatus::RUNNING) {
                    handle->stop();
                    break;
                }
            }
            if (aborted) {
                break;
            }
        }
        if (aborted) {
            break;
        }
    }
    streamer->end();

    VLMResult result;
    result.text = std::move(accumulated);
    result.finish_reason = aborted ? "abort" : finish_reason_str(finish);
    if (opts.on_done) {
        opts.on_done(result.text);
    }
    return result;
}

}  // namespace ovserver
