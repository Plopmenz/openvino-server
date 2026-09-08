// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/video_generation.hpp"

#include <cstdio>
#include <cstring>
#include <exception>

#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/genai/image_generation/generation_config.hpp>
#include <openvino/genai/video_generation/generation_config.hpp>

#include "ovserver/audio.hpp"
#include <iostream>

namespace ovserver {
namespace {

std::string shell_quote(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (char c : value) {
        if (c == '"' || c == '\\' || c == '$' || c == '`') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// Concatenates RGB frames into an MP4 (H.264) file at the given path using a
// pipefed ffmpeg. Returns the output file size in bytes (0 means the encoder
// failed or produced nothing).
std::uint64_t encode_frames_to_mp4(const std::string& out_path,
                                   const std::vector<VideoFrame>& frames,
                                   std::size_t fps) {
    if (frames.empty() || frames.front().rgb.empty())
        return 0;
    const std::size_t width = frames.front().width;
    const std::size_t height = frames.front().height;
    for (const auto& f : frames) {
        if (f.width != width || f.height != height)
            return 0;
    }
    const std::string cmd =
        shell_quote(ffmpeg_path()) +
        " -hide_banner -loglevel error -y "
        "-f rawvideo -pix_fmt rgb24 -s " +
        std::to_string(width) + "x" + std::to_string(height) + " -r " +
        std::to_string(fps) +
        " -i pipe:0 -an -c:v libx264 -pix_fmt yuv420p -crf 20 " +
        shell_quote(out_path);
    FILE* pipe = popen(cmd.c_str(), "w");
    if (pipe == nullptr) {
        std::cerr << "video encode: ffmpeg is not available (" << cmd << ")"
                  << std::endl;
        return 0;
    }
    for (const auto& f : frames) {
        const std::size_t plane_size = f.rgb.size();
        const std::size_t written = fwrite(f.rgb.data(), 1, plane_size, pipe);
        if (written != plane_size) {
            pclose(pipe);
            return 0;
        }
    }
    const int status = pclose(pipe);
    if (status != 0) {
        std::cerr << "video encode: ffmpeg failed with status " << status
                  << " for " << out_path << std::endl;
        return 0;
    }
    std::error_code ec;
    const std::uint64_t size =
        static_cast<std::uint64_t>(std::filesystem::file_size(out_path, ec));
    return ec ? 0 : size;
}

// Copies one video plane (H*W*3) into a VideoFrame, converting to planar RGB
// uint8 for any supported element type. LTX-Video historically emits float
// tensors in [0, 1]; uint8 is handled directly.
VideoFrame copy_plane(const ov::Tensor& tensor,
                      std::size_t frame,
                      std::size_t height,
                      std::size_t width) {
    VideoFrame out;
    out.height = height;
    out.width = width;
    out.rgb.resize(height * width * 3);
    if (tensor.get_element_type() == ov::element::u8) {
        const auto* src =
            tensor.data<std::uint8_t>() + frame * height * width * 3;
        std::memcpy(out.rgb.data(), src, out.rgb.size());
        return out;
    }
    if (tensor.get_element_type() == ov::element::f32) {
        const auto* src =
            tensor.data<float>() + frame * height * width * 3;
        for (std::size_t i = 0; i < out.rgb.size(); ++i) {
            const float v = src[i] < 0.0f ? 0.0f : (src[i] > 1.0f ? 1.0f
                                                                  : src[i]);
            out.rgb[i] = static_cast<std::uint8_t>(v * 255.0f + 0.5f);
        }
        return out;
    }
    if (tensor.get_element_type() == ov::element::f16) {
        const auto* src =
            tensor.data<ov::float16>() + frame * height * width * 3;
        for (std::size_t i = 0; i < out.rgb.size(); ++i) {
            const float v = static_cast<float>(src[i]);
            const float c = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            out.rgb[i] = static_cast<std::uint8_t>(c * 255.0f + 0.5f);
        }
        return out;
    }
    throw std::runtime_error(
        "unsupported video tensor element type " +
        tensor.get_element_type().get_type_name());
}

}  // namespace

std::uint64_t write_video_mp4(const std::string& path,
                              const VideoResult& video) {
    return encode_frames_to_mp4(path, video.frames, video.fps);
}

VideoGenerationModel::VideoGenerationModel(
    const std::string& id,
    const std::filesystem::path& models_path,
    const std::string& device)
    : m_id(id), m_models_path(models_path), m_device(device) {
    std::cerr << "video load: warm start" << std::endl;
    ov::AnyMap properties;
    if (m_device.find("GPU") != std::string::npos) {
        properties[ov::hint::kv_cache_precision.name()] =
            ov::element::u8;
        properties[ov::hint::dynamic_quantization_group_size.name()] =
            std::uint64_t{32};
        properties[ov::intel_gpu::hint::enable_sdpa_optimization.name()] =
            true;
    }
    m_pipeline =
        std::make_shared<ov::genai::Text2VideoPipeline>(m_models_path.string(),
                                                        m_device, properties);
    std::cerr << "video load: done" << std::endl;
}

VideoGenerationModel::~VideoGenerationModel() = default;

std::vector<VideoResult> VideoGenerationModel::generate(
    const VideoGenerateOptions& opts) {
    const std::uint64_t req_id = m_next_req_id.fetch_add(1);
    const auto start = std::chrono::steady_clock::now();

    ov::AnyMap properties;
    properties[ov::genai::num_videos_per_prompt.name()] = opts.num_videos;
    if (opts.negative_prompt)
        properties[ov::genai::negative_prompt.name()] = *opts.negative_prompt;
    if (opts.guidance_scale)
        properties[ov::genai::guidance_scale.name()] = *opts.guidance_scale;
    if (opts.height)
        properties[ov::genai::height.name()] =
            static_cast<std::int64_t>(*opts.height);
    if (opts.width)
        properties[ov::genai::width.name()] =
            static_cast<std::int64_t>(*opts.width);
    if (opts.num_inference_steps)
        properties[ov::genai::num_inference_steps.name()] =
            *opts.num_inference_steps;
    if (opts.num_frames)
        properties[ov::genai::num_frames.name()] = *opts.num_frames;
    if (opts.frame_rate)
        properties[ov::genai::frame_rate.name()] = *opts.frame_rate;
    if (opts.rng_seed) {
        properties[ov::genai::generator.name()] =
            std::make_shared<ov::genai::CppStdGenerator>(
                static_cast<std::uint32_t>(*opts.rng_seed));
    }

    std::cerr << "video gen start: prompt=" << opts.prompt << std::endl;

    const ov::genai::VideoGenerationResult result = [&]() {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_pipeline->generate(opts.prompt, properties);
    }();

    const ov::Tensor& video = result.video;
    const ov::Shape shape = video.get_shape();
    if (shape.size() != 5 || shape[0] == 0 || shape[2] == 0 ||
        shape[3] == 0)
        throw std::runtime_error("video: unexpected output shape");
    const std::size_t num_videos = static_cast<std::size_t>(shape[0]);
    const std::size_t num_frames = static_cast<std::size_t>(shape[1]);
    const std::size_t height = static_cast<std::size_t>(shape[2]);
    const std::size_t width = static_cast<std::size_t>(shape[3]);
    const std::size_t fps =
        opts.frame_rate
            ? static_cast<std::size_t>(*opts.frame_rate)
            : 25;

    std::vector<VideoResult> videos;
    videos.reserve(num_videos);
    for (std::size_t n = 0; n < num_videos; ++n) {
        VideoResult res;
        res.fps = fps;
        res.frames.reserve(num_frames);
        for (std::size_t f = 0; f < num_frames; ++f)
            res.frames.push_back(
                copy_plane(video, n * num_frames + f, height, width));
        videos.push_back(std::move(res));
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    std::cerr << "video gen done: req=" << req_id << " ms=" << elapsed
              << " videos=" << num_videos << " frames=" << num_frames
              << " h=" << height << " w=" << width << std::endl;
    return videos;
}

}  // namespace ovserver