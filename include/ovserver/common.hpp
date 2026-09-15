// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

namespace ovserver {

// Parses and stores device-scoped inference properties supplied on the command
// line as JSON, e.g. --device-props '{"gpu":{"KV_CACHE_PRECISION":"u8"}}'.
// Top-level keys are device names (matched case-insensitively against the
// device OpenVINO reports, so "gpu" matches "GPU"); their values are OpenVINO
// property maps forwarded to the GenAI pipeline verbatim — per-device
// construction is the caller's job. Throws std::runtime_error on malformed
// input.
void set_device_props(const std::string& json);

// Builds the property map forwarded to a GenAI pipeline: always the model
// cache dir; when the storage set by set_device_props() carries an entry for
// `device`, those properties are wrapped in ov::device::properties so they are
// scoped to that device. Non-matching devices get the cache dir only.
ov::AnyMap inference_properties(const std::string& device,
                                const std::string& cache_dir);

// Quotes a string for safe inclusion in a single-quoted shell command built
// with /bin/sh: wraps it in double quotes and backslash-escapes the characters
// the shell would otherwise interpret (" \ $ `).
std::string shell_quote(const std::string& value);

// Builds the denoising-step progress callback shared by the image and video
// pipelines. Each invocation logs "[tag model 'id'] request N step X/Y: Ds".
std::function<bool(std::size_t, std::size_t, ov::Tensor&)> step_logger(
    const std::string& tag,
    const std::string& id,
    std::uint64_t req_id,
    std::chrono::steady_clock::time_point start);

// Prints the standard post-load memory line, e.g.
//     [image model 'sdxl'] memory: GPU 3.79 GiB
//     [tts  model 'kokoro'] memory: NPU 0.13 GiB
//     [txt  model 'qwen']   memory: CPU 1.45 GiB
//     [txt  model 'qwen']   memory: GPU 3.79 GiB, NPU 0.13 GiB
// One entry is emitted per distinct device in `devices`, in order. GPU
// allocation stats are queried via the GPU plugin's internal allocator; NPU
// allocation stats use the NPU plugin's device_alloc_mem_size property. Devices
// without a queryable allocator (CPU, AUTO) report the process RSS.
void log_model_memory(const std::string& tag,
                      const std::string& id,
                      const std::vector<std::string>& devices);

// Scope guard emitting the shared startup/loaded log lines of every pipeline
// constructor: "[tag model 'id'] loading from PATH on DEVICE ..." at
// construction and, on completion(), "[tag model 'id'] loaded in Xs" followed
// by the per-device memory line. `second_device` is an additional device the
// pipeline uses (e.g. a speculative-decoding draft model); its memory is
// reported alongside `device` when set and distinct from it. The draft model
// itself has no separately measurable load time, so the single load time is
// kept.
class PipelineLoadLog {
public:
    PipelineLoadLog(std::string tag,
                    std::string id,
                    const std::filesystem::path& path,
                    std::string device,
                    std::string second_device = {});

    // Marks the load as finished and prints the "loaded in Xs" + memory lines.
    void completion();

private:
    std::string m_tag;
    std::string m_id;
    std::filesystem::path m_path;
    std::string m_device;
    std::string m_second_device;
    std::chrono::steady_clock::time_point m_t0;
};

}  // namespace ovserver