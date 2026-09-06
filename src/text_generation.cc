// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/text_generation.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include <openvino/genai/text_streamer.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

namespace ovserver {

namespace {

std::size_t proc_field_kb(const std::string& field) {
    std::ifstream status("/proc/self/status");
    if (!status) {
        return 0;
    }
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind(field, 0) == 0) {
            const std::size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos) {
                return static_cast<std::size_t>(std::stoull(line.substr(pos)));
            }
        }
    }
    return 0;
}

// Total GPU device memory currently allocated by the plugin, in MiB.
// Returns -1 when not applicable (non-GPU device, plugin unavailable).
double device_gpu_mib(const std::string& device) {
    if (device.find("GPU") == std::string::npos) {
        return -1.0;
    }
    try {
        ov::Core core;
        auto stats = core.get_property(device, ov::intel_gpu::memory_statistics);
        std::uint64_t bytes = 0;
        for (const auto& [name, value] : stats) {
            bytes += value;
        }
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    } catch (const std::exception&) {
        return -1.0;
    }
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

}  // namespace

TextGenerationModel::TextGenerationModel(
    const std::string& id,
    const std::filesystem::path& models_path,
    const std::string& device)
    : m_id(id), m_models_path(models_path), m_device(device) {
    // Plain construction with defaults: no scheduler tuning, no plugin or
    // tokenizer properties.
    const auto t0 = std::chrono::steady_clock::now();
    std::cerr << "[text model '" << id << "'] loading from " << models_path
              << " on " << device << " ..." << std::endl;
    ov::genai::SchedulerConfig sched_cfg;
    try {
        m_pipeline = std::make_shared<ov::genai::ContinuousBatchingPipeline>(
            models_path, sched_cfg, device);
        m_tokenizer = m_pipeline->get_tokenizer();
    } catch (const std::exception& e) {
        std::cerr << "[text model '" << id << "'] loading FAILED: " << e.what()
                  << std::endl;
        throw;
    }
    const auto load_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    std::cerr << "[text model '" << id << "'] loaded in " << load_s
              << " s" << std::endl;
    std::cerr << "[text model '" << id << "'] memory: ";
    const double gpu_mib = device_gpu_mib(device);
    if (gpu_mib >= 0.0) {
        std::cerr << "GPU " << std::fixed << std::setprecision(2)
                  << (gpu_mib / 1024.0) << " GiB" << std::endl;
    } else if (device.find("GPU") != std::string::npos) {
        std::cerr << "GPU ??? GiB" << std::endl;
    } else {
        std::cerr << "CPU " << std::fixed << std::setprecision(2)
                  << (proc_field_kb("VmRSS:") / (1024.0 * 1024.0)) << " GiB"
                  << std::endl;
    }
    m_last_snapshot = std::chrono::steady_clock::now();
    m_executor = std::thread([this] { executor_run(); });
}

TextGenerationModel::~TextGenerationModel() {
    m_stop = true;
    {
        std::lock_guard<std::mutex> lock(m_cv_mutex);
        m_cv.notify_all();
    }
    if (m_executor.joinable()) {
        m_executor.join();
    }
}

std::uint64_t TextGenerationModel::next_request_id() {
    std::lock_guard<std::mutex> lock(m_id_mutex);
    return m_next_id++;
}

void TextGenerationModel::executor_run() {
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
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - m_last_snapshot).count()
                    >= 1.0) {
                    const auto window_start = m_last_snapshot;
                    m_last_snapshot = now;
                    snapshot_log(now, window_start);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[text model '" << m_id
                      << "'] scheduler step error: " << e.what() << std::endl;
        }
    }
}

void TextGenerationModel::snapshot_log(
    const std::chrono::steady_clock::time_point now,
    const std::chrono::steady_clock::time_point window_start) {
    std::lock_guard<std::mutex> lock(m_metrics_mutex);
    if (!m_requests.empty()) {
        std::size_t prefill = 0;
        std::size_t decode = 0;
        std::size_t decode_delta = 0;
        for (auto& [rid, m] : m_requests) {
            if (m.tokens == 0) {
                ++prefill;
            } else {
                ++decode;
            }
            if (!m.has_baseline) {
                // Request not present in the previous snapshot: record a
                // baseline now so its rates show up from the next snapshot.
                m.has_baseline = true;
                m.baseline_tokens = m.tokens;
                m.baseline_time = now;
                continue;
            }
            if (m.tokens > m.baseline_tokens) {
                decode_delta += m.tokens - m.baseline_tokens;
            }
            m.baseline_tokens = m.tokens;
            m.baseline_time = now;
        }

        const double dt =
            std::chrono::duration<double>(now - window_start).count();
        if (dt > 0.0) {
            const double prefill_rate = m_prefill_tokens_in_window / dt;
            const double decode_rate =
                static_cast<double>(decode_delta) / dt;
            std::cerr << "[text model '" << m_id << "'] snapshot: prefill "
                      << prefill_rate << " tok/s (" << prefill
                      << " req) | decode " << decode_rate << " tok/s (" << decode
                      << " req)" << std::endl;
        }
    }
    m_prefill_tokens_in_window = 0;
}

TextResult TextGenerationModel::generate(const TextGenerateOptions& opts) {
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

    // Apply the model's own chat template so role framing matches training.
    ov::genai::ChatHistory history;
    if (!opts.system_message.empty()) {
        history.push_back({{"role", "system"}, {"content", opts.system_message}});
    }
    history.push_back({{"role", "user"}, {"content", opts.prompt}});
    const std::string templated =
        m_tokenizer.apply_chat_template(history, /*add_generation_prompt=*/true);
    // Prompt token count: needed to measure aggregate prefill throughput (the
    // CB pipeline does not expose per-request prefill timings).
    const std::size_t prompt_tokens =
        m_tokenizer.encode(templated).input_ids.get_size();

    const std::uint64_t req_id = next_request_id();
    ov::genai::GenerationHandle handle =
        m_pipeline->add_request(req_id, templated, opts.images, cfg);
    {
        std::lock_guard<std::mutex> lock(m_cv_mutex);
        m_cv.notify_one();
    }
    {
        std::lock_guard<std::mutex> lock(m_metrics_mutex);
        m_requests[req_id] = {std::chrono::steady_clock::now(),
                              std::chrono::steady_clock::now(), 0, 0,
                              prompt_tokens, false};
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
    bool prefill_done = false;
    std::size_t tokens = 0;
    const auto gen_start = std::chrono::steady_clock::now();
    while (handle->get_status() == ov::genai::GenerationStatus::RUNNING ||
           handle->can_read()) {
        ov::genai::GenerationOutputs outputs = handle->read();
        for (auto& [rid, out] : outputs) {
            if (out.finish_reason != ov::genai::GenerationFinishReason::NONE) {
                finish = out.finish_reason;
            }
            for (const int64_t tid : out.generated_ids) {
                if (!prefill_done) {
                    // First generated token: prefill of this request just
                    // completed. Credit its prompt tokens to the prefill
                    // throughput of the current snapshot window, but only if
                    // the request already appeared in a previous snapshot.
                    prefill_done = true;
                    std::lock_guard<std::mutex> lock(m_metrics_mutex);
                    RequestMetrics& m = m_requests.at(req_id);
                    if (m.has_baseline) {
                        m_prefill_tokens_in_window += m.prompt_tokens;
                    }
                }
                ++tokens;
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
        {
            std::lock_guard<std::mutex> lock(m_metrics_mutex);
            m_requests[req_id].tokens = tokens;
        }
        if (aborted) {
            break;
        }
    }
    streamer->end();

    const double gen_s = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - gen_start)
                             .count();
    {
        std::lock_guard<std::mutex> lock(m_metrics_mutex);
        m_requests.erase(req_id);
    }
    std::cerr << "[text model '" << m_id << "'] request " << req_id
              << " finished in " << gen_s << " s (" << tokens << " tokens"
              << (tokens > 0 ? ", avg " + std::to_string(tokens / gen_s)
                             : std::string())
              << " tok/s)" << std::endl;

    TextResult result;
    result.text = std::move(accumulated);
    result.finish_reason = aborted ? "abort" : finish_reason_str(finish);
    if (opts.on_done) {
        opts.on_done(result.text);
    }
    return result;
}

}  // namespace ovserver