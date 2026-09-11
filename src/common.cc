// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/common.hpp"

#include <algorithm>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>

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

// Total NPU DDR memory currently allocated, in MiB.
// Returns -1.0 when not applicable (non-NPU device, plugin unavailable).
double device_npu_mib(const std::string& device) {
    if (device.find("NPU") == std::string::npos) {
        return -1.0;
    }
    try {
        ov::Core core;
        const std::uint64_t bytes =
            core.get_property(device, ov::intel_npu::device_alloc_mem_size);
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
    // Wrap GPU-specific hints in ov::device::properties so they are scoped to
    // the GPU device.  GenAI forwards the full property map to the draft model
    // compile call; without device-scoping the CPU plugin rejects unknown
    // GPU-specific keys.
    ov::AnyMap gpu_props;
    properties.emplace(ov::device::properties("GPU", gpu_props));
    return properties;
}

void log_model_memory(const std::string& tag,
                      const std::string& id,
                      const std::vector<std::string>& devices) {
    std::cerr << "[" << tag << " model '" << id << "'] memory: ";
    std::vector<std::string> seen;
    bool first = true;
    for (const auto& device : devices) {
        if (std::find(seen.begin(), seen.end(), device) != seen.end()) {
            continue;
        }
        seen.push_back(device);

        bool unknown = false;
        double gib = -1.0;
        std::string label;
        const double gpu_mib = device_gpu_mib(device);
        if (gpu_mib >= 0.0) {
            label = "GPU";
            gib = gpu_mib / 1024.0;
        } else if (device.find("GPU") != std::string::npos) {
            label = "GPU";
            unknown = true;
        } else {
            const double npu_mib = device_npu_mib(device);
            if (npu_mib >= 0.0) {
                label = "NPU";
                gib = npu_mib / 1024.0;
            } else if (device.find("NPU") != std::string::npos) {
                label = "NPU";
                unknown = true;
            } else {
                label = "CPU";
                gib = proc_field_kb("VmRSS:") / (1024.0 * 1024.0);
            }
        }

        if (!first) {
            std::cerr << ", ";
        }
        first = false;
        std::cerr << label << " ";
        if (unknown) {
            std::cerr << "??? GiB";
        } else {
            std::cerr << std::fixed << std::setprecision(2) << gib << " GiB";
        }
    }
    std::cerr << std::endl;
}

PipelineLoadLog::PipelineLoadLog(std::string tag,
                                 std::string id,
                                 const std::filesystem::path& path,
                                 std::string device,
                                 std::string second_device)
    : m_tag(std::move(tag)),
      m_id(std::move(id)),
      m_path(path),
      m_device(std::move(device)),
      m_second_device(std::move(second_device)),
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
    std::vector<std::string> devices{m_device};
    if (!m_second_device.empty() && m_second_device != m_device) {
        devices.push_back(m_second_device);
    }
    log_model_memory(m_tag, m_id, devices);
}

}  // namespace ovserver
