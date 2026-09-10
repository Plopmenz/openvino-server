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
#include <vector>

#include <openvino/genai/video_generation/text2video_pipeline.hpp>

namespace ovserver {

// All fields are optional so that model-specific defaults (e.g. LTX-Video's
// 161 frames, 25 fps, 512x512) are preserved unless a client overrides them.
struct VideoGenerateOptions {
    std::string prompt;
    std::optional<std::string> negative_prompt;
    std::optional<float> guidance_scale;
    std::optional<int64_t> height;
    std::optional<int64_t> width;
    std::size_t num_videos = 1;
    std::optional<std::size_t> num_inference_steps;
    std::optional<std::size_t> rng_seed;
    std::optional<std::size_t> num_frames;
    std::optional<float> frame_rate;
};

// Single decoded frame (RGB, one byte per channel).
struct VideoFrame {
    std::size_t height = 0;
    std::size_t width = 0;
    std::vector<std::uint8_t> rgb;  // height * width * 3
};

struct VideoResult {
    std::vector<VideoFrame> frames;
    std::size_t fps = 25;
};

// Wraps an ov::genai::Text2VideoPipeline (e.g. LTX-Video) exported to
// OpenVINO IR. The pipeline is built on startup exactly like the reference
// Python path -- Text2VideoPipeline(path, device). Concurrent requests clone()
// it, which shares the compiled models while giving each request its own
// scheduler.
class VideoGenerationModel {
public:
    VideoGenerationModel(const std::string& id,
                         const std::filesystem::path& models_path,
                         const std::string& device,
                         const std::string& cache_dir);

    VideoGenerationModel(const VideoGenerationModel&) = delete;
    VideoGenerationModel& operator=(const VideoGenerationModel&) = delete;

    ~VideoGenerationModel();

    const std::string& id() const { return m_id; }

    std::vector<VideoResult> generate(const VideoGenerateOptions& opts);

private:
    std::string m_id;
    std::filesystem::path m_models_path;
    std::string m_device;

    std::mutex m_mutex;
    std::shared_ptr<ov::genai::Text2VideoPipeline> m_pipeline;
    std::atomic<std::uint64_t> m_next_req_id{0};
};

struct VideoGenerationSpec {
    std::filesystem::path path;
    std::string device;
    // Directory for OpenVINO compiled-model blobs. Empty disables caching.
    std::string cache_dir;
};

// Encodes a video as H.264/MP4 and writes it to `path` via a pipefed ffmpeg.
// Returns the output file size in bytes, or 0 if ffmpeg failed.
std::uint64_t write_video_mp4(const std::string& path,
                              const VideoResult& video);

}  // namespace ovserver