// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <openvino/genai/image_generation/text2image_pipeline.hpp>

namespace ovserver {

// All fields are optional so that model-specific defaults from the pipeline
// (e.g. Qwen-Image's guidance=4.0, 50 steps, 1024x1024) are preserved unless a
// client explicitly overrides them.
struct ImageGenerateOptions {
    std::string prompt;
    std::optional<std::string> negative_prompt;
    std::optional<float> guidance_scale;
    std::optional<int64_t> height;
    std::optional<int64_t> width;
    std::size_t num_images = 1;
    std::optional<std::size_t> num_inference_steps;
    std::optional<std::size_t> rng_seed;
};

struct ImageResult {
    int height;
    int width;
    int channels;
    std::vector<std::uint8_t> data;
};

// Wraps an ov::genai::Text2ImagePipeline (e.g. Qwen-Image) exported to
// OpenVINO IR. The pipeline is built on startup exactly like the reference
// Python path -- Text2ImagePipeline(path, device) -- with no shape or plugin
// configuration. Concurrent requests clone() it, which shares the compiled
// models while giving each request its own scheduler.
class ImageGenerationModel {
public:
    ImageGenerationModel(const std::string& id,
                         const std::filesystem::path& models_path,
                         const std::string& device,
                         const std::string& cache_dir);

    ImageGenerationModel(const ImageGenerationModel&) = delete;
    ImageGenerationModel& operator=(const ImageGenerationModel&) = delete;

    ~ImageGenerationModel();

    const std::string& id() const { return m_id; }

    std::vector<ImageResult> generate(const ImageGenerateOptions& opts);

private:
    void snapshot_run();

    // Per-request diffusion timing. The pipeline callback records the duration
    // of the just-completed step; the snapshot thread averages the latest
    // s/step across running requests once per second.
    struct StepStats {
        std::size_t steps = 0;
        double last_step_s = 0.0;
    };

    std::string m_id;
    std::filesystem::path m_models_path;
    std::string m_device;

    std::mutex m_mutex;
    std::shared_ptr<ov::genai::Text2ImagePipeline> m_pipeline;

    std::thread m_snapshot_thread;
    std::atomic<bool> m_stop{false};
    std::atomic<std::uint64_t> m_next_req_id{0};

    std::mutex m_metrics_mutex;
    std::unordered_map<std::uint64_t, StepStats> m_steps;
    std::chrono::steady_clock::time_point m_last_snapshot;
};

struct ImageGenerationSpec {
    std::filesystem::path path;
    std::string device;
    // Directory for OpenVINO compiled-model blobs. Empty disables caching.
    std::string cache_dir;
};

}  // namespace ovserver