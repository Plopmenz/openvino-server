// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/tts.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <openvino/genai/speech_generation/text2speech_pipeline.hpp>
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
    std::cerr << "[tts model '" << id << "'] memory: ";
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

TTSModel::TTSModel(const std::string& id,
                   const std::filesystem::path& models_path,
                   const std::string& device,
                   const std::string& cache_dir)
    : m_id(id), m_models_path(models_path), m_device(device) {
    const auto t0 = std::chrono::steady_clock::now();
    std::cerr << "[tts model '" << id << "'] loading from " << m_models_path
              << " on " << m_device << " ..." << std::endl;
    try {
        ov::AnyMap properties;
        if (!cache_dir.empty()) {
            properties.emplace(ov::cache_dir(cache_dir));
        }
        m_pipeline =
            std::make_shared<ov::genai::Text2SpeechPipeline>(m_models_path.string(),
                                                              m_device, properties);
    } catch (const std::exception& e) {
        std::cerr << "[tts model '" << id << "'] loading FAILED: " << e.what()
                  << std::endl;
        throw;
    }
    const auto load_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    std::cerr << "[tts model '" << id << "'] loaded in " << load_s
              << " s" << std::endl;
    log_memory(id, m_device);
}

TTSModel::~TTSModel() = default;

TTSResult TTSModel::generate(const std::string& text) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::cerr << "tts start: text=" << text << std::endl;
    const ov::genai::Text2SpeechDecodedResults result =
        m_pipeline->generate(std::vector<std::string>{text});
    if (result.speeches.empty())
        throw std::runtime_error("tts: no speech produced");
    const ov::Tensor& speech = result.speeches.front();
    if (speech.get_element_type() != ov::element::f32)
        throw std::runtime_error("tts: unexpected sample element type " +
                                 speech.get_element_type().get_type_name());
    const float* data = speech.data<float>();
    const std::size_t count = static_cast<std::size_t>(speech.get_size());
    TTSResult out;
    out.sample_rate = result.output_sample_rate;
    out.samples.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float v = data[i];
        const float c = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
        out.samples.push_back(
            static_cast<std::int16_t>(c * 32767.0f));
    }
    std::cerr << "tts done: samples=" << count << " rate=" << out.sample_rate
          << std::endl;
    return out;
}

}  // namespace ovserver