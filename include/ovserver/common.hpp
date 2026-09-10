// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>

#include <openvino/openvino.hpp>

namespace ovserver {

// Device-level inference tuning shared by every GenAI pipeline.
struct GpuTuning {
    ov::element::Type kv_cache_precision{ov::element::u8};
    std::uint64_t dynamic_quantization_group_size{32};
    bool enable_sdpa_optimization{true};
};

// Builds the property map forwarded to a GenAI pipeline: the model cache dir
// plus GPU-specific tuning (u8 KV cache, dynamic quantization, SDPA
// optimization) when the target is the Intel GPU plugin. Other devices get the
// cache dir only, since these hints are rejected at compile time by non-GPU
// plugins.
ov::AnyMap inference_properties(const std::string& device,
                                const std::string& cache_dir,
                                const GpuTuning& gpu = {});

// Prints the standard post-load memory line, e.g.
//     [image model 'sdxl'] memory: GPU 3.79 GiB
// GPU allocation stats are queried only when the device targets the GPU
// plugin; otherwise the process RSS is reported.
void log_model_memory(const std::string& tag,
                      const std::string& id,
                      const std::string& device);

// Scope guard emitting the shared startup/loaded log lines of every pipeline
// constructor: "[tag model 'id'] loading from PATH on DEVICE ..." at
// construction and, on completion(), "[tag model 'id'] loaded in Xs" followed
// by the per-device memory line.
class PipelineLoadLog {
public:
    PipelineLoadLog(std::string tag,
                    std::string id,
                    const std::filesystem::path& path,
                    std::string device);

    // Marks the load as finished and prints the "loaded in Xs" + memory lines.
    void completion();

private:
    std::string m_tag;
    std::string m_id;
    std::filesystem::path m_path;
    std::string m_device;
    std::chrono::steady_clock::time_point m_t0;
};

}  // namespace ovserver