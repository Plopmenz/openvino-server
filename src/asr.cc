// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/asr.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <openvino/genai/automatic_speech_recognition/pipeline.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/properties.hpp>
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

void log_memory(const std::string& id, const std::string& device) {
    std::cerr << "[asr model '" << id << "'] memory: ";
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
}

}  // namespace

ASRModel::ASRModel(const std::string& id,
                   const std::filesystem::path& models_path,
                   const std::string& device,
                   const std::string& cache_dir)
    : m_id(id), m_models_path(models_path), m_device(device) {
    const auto t0 = std::chrono::steady_clock::now();
    std::cerr << "[asr model '" << id << "'] loading from " << m_models_path
              << " on " << m_device << " ..." << std::endl;
    try {
        ov::AnyMap properties;
        if (!cache_dir.empty()) {
            properties.emplace(ov::cache_dir(cache_dir));
        }
        m_pipeline = std::make_shared<ov::genai::ASRPipeline>(
            m_models_path.string(), m_device, properties);
    } catch (const std::exception& e) {
        std::cerr << "[asr model '" << id << "'] loading FAILED: " << e.what()
                  << std::endl;
        throw;
    }
    const auto load_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    std::cerr << "[asr model '" << id << "'] loaded in " << load_s
              << " s" << std::endl;
    log_memory(id, m_device);
}

ASRModel::~ASRModel() = default;

ASRResult ASRModel::generate(const ASRGenerateOptions& opts,
                             ASROnText on_text) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ov::genai::ASRGenerationConfig cfg;
    cfg.language = opts.language;
    cfg.task = opts.task;
    cfg.initial_prompt = opts.initial_prompt;
    cfg.context = opts.context;
    cfg.return_timestamps = opts.return_timestamps;
    if (opts.temperature)
        cfg.temperature = *opts.temperature;
    if (opts.max_new_tokens)
        cfg.max_new_tokens = *opts.max_new_tokens;

    std::cerr << "asr start: samples=" << opts.samples.size() << std::endl;

    const auto start = std::chrono::steady_clock::now();
    const ov::genai::AudioInputs input{opts.samples};
    ov::genai::ASRDecodedResults result;
    if (on_text) {
        result = m_pipeline->generate(input, cfg, on_text);
    } else {
        result = m_pipeline->generate(input, cfg);
    }

    ASRResult out;
    if (!result.texts.empty())
        out.text = result.texts[0];
    if (!result.languages.empty())
        out.language = result.languages[0];
    if (result.chunks && !result.chunks->empty()) {
        for (const auto& chunk : (*result.chunks)[0]) {
            ASRSegment seg;
            seg.start = chunk.start_ts;
            seg.end = chunk.end_ts;
            seg.text = chunk.text;
            out.segments.push_back(std::move(seg));
        }
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    std::cerr << "asr done: ms=" << elapsed << std::endl;
    return out;
}

}  // namespace ovserver