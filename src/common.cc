// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/common.hpp"

#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

namespace ovserver {

namespace {

// Reads a sized field in kB from /proc/self/status (e.g. VmRSS).
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
// Returns -1.0 when not applicable (non-GPU device, plugin unavailable).
double device_gpu_mib(const std::string& device) {
    if (device.find("GPU") == std::string::npos) {
        return -1.0;
    }
    try {
        ov::Core core;
        auto stats =
            core.get_property(device, ov::intel_gpu::memory_statistics);
        std::uint64_t bytes = 0;
        for (const auto& [name, value] : stats) {
            bytes += value;
        }
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    } catch (const std::exception&) {
        return -1.0;
    }
}

}  // namespace

ov::AnyMap inference_properties(const std::string& device,
                                const std::string& cache_dir,
                                const GpuTuning& gpu) {
    ov::AnyMap properties;
    if (!cache_dir.empty()) {
        properties.emplace(ov::cache_dir(cache_dir));
    }
    if (device.find("GPU") == std::string::npos) {
        return properties;
    }
    properties.emplace(ov::hint::kv_cache_precision(gpu.kv_cache_precision));
    properties.emplace(
        ov::hint::dynamic_quantization_group_size(
            gpu.dynamic_quantization_group_size));
    properties.emplace(
        ov::intel_gpu::hint::enable_sdpa_optimization(
            gpu.enable_sdpa_optimization));
    return properties;
}

void log_model_memory(const std::string& tag,
                      const std::string& id,
                      const std::string& device) {
    std::cerr << "[" << tag << " model '" << id << "'] memory: ";
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

PipelineLoadLog::PipelineLoadLog(std::string tag,
                                 std::string id,
                                 const std::filesystem::path& path,
                                 std::string device)
    : m_tag(std::move(tag)),
      m_id(std::move(id)),
      m_path(path),
      m_device(std::move(device)),
      m_t0(std::chrono::steady_clock::now()) {
    std::cerr << "[" << m_tag << " model '" << m_id << "'] loading from "
              << m_path << " on " << m_device << " ..." << std::endl;
}

void PipelineLoadLog::completion() {
    const double load_s = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - m_t0)
                              .count();
    std::cerr << "[" << m_tag << " model '" << m_id << "'] loaded in "
              << load_s << " s" << std::endl;
    log_model_memory(m_tag, m_id, m_device);
}

}  // namespace ovserver