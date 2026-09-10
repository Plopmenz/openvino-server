// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/video_generation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/genai/image_generation/generation_config.hpp>
#include <openvino/genai/video_generation/generation_config.hpp>
#include <openvino/runtime/core.hpp>

#include "ovserver/audio.hpp"

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

// Rounds up to the next multiple of 32, the spatial compression factor of the
// LTX-Video VAE. Sizes that are not multiples of 32 (e.g. 1280x720) are
// generated at the padded resolution and resized down afterwards.
std::int64_t align32(std::int64_t v) {
    if (v <= 0) return v;
    return (v + 31) / 32 * 32;
}

// Bilinear RGB resize of a decoded frame. Produces exactly the requested
// dimensions, whatever resolution the model actually decoded.
VideoFrame resize_frame(const VideoFrame& f,
                        std::size_t out_width,
                        std::size_t out_height) {
    VideoFrame out;
    out.width = out_width;
    out.height = out_height;
    if (out_width == f.width && out_height == f.height) {
        out.rgb = f.rgb;
        return out;
    }
    if (f.width == 0 || f.height == 0) {
        throw std::runtime_error("video: cannot resize an empty frame");
    }
    out.rgb.resize(out_height * out_width * 3);
    const double sx = static_cast<double>(f.width) / out_width;
    const double sy = static_cast<double>(f.height) / out_height;
    const auto* src = f.rgb.data();
    auto* dst = out.rgb.data();
    for (std::size_t y = 0; y < out_height; ++y) {
        const double fy = std::min<double>(f.height - 1, y * sy);
        const std::size_t y0 = static_cast<std::size_t>(fy);
        const std::size_t y1 = std::min(f.height - 1, y0 + 1);
        const double wy = fy - y0;
        for (std::size_t x = 0; x < out_width; ++x) {
            const double fx = std::min<double>(f.width - 1, x * sx);
            const std::size_t x0 = static_cast<std::size_t>(fx);
            const std::size_t x1 = std::min(f.width - 1, x0 + 1);
            const double wx = fx - x0;
            const auto* p00 = src + (y0 * f.width + x0) * 3;
            const auto* p10 = src + (y0 * f.width + x1) * 3;
            const auto* p01 = src + (y1 * f.width + x0) * 3;
            const auto* p11 = src + (y1 * f.width + x1) * 3;
            auto* d = dst + (y * out_width + x) * 3;
            for (int c = 0; c < 3; ++c) {
                const double top = static_cast<double>(p00[c]) +
                                   wx * (p10[c] - p00[c]);
                const double bot = static_cast<double>(p01[c]) +
                                   wx * (p11[c] - p01[c]);
                double v = top + wy * (bot - top);
                if (v < 0.0) v = 0.0;
                if (v > 255.0) v = 255.0;
                d[c] = static_cast<std::uint8_t>(v + 0.5);
            }
        }
    }
    return out;
}

}  // namespace

std::uint64_t write_video_mp4(const std::string& path,
                              const VideoResult& video) {
    return encode_frames_to_mp4(path, video.frames, video.fps);
}

VideoGenerationModel::VideoGenerationModel(
    const std::string& id,
    const std::filesystem::path& models_path,
    const std::string& device,
    const std::string& cache_dir)
    : m_id(id), m_models_path(models_path), m_device(device) {
    const auto t0 = std::chrono::steady_clock::now();
    std::cerr << "[video model '" << id << "'] loading from " << m_models_path
              << " on " << m_device << " ..." << std::endl;
    ov::AnyMap properties;
    if (m_device.find("GPU") != std::string::npos) {
        properties.emplace(ov::hint::kv_cache_precision(ov::element::u8));
        properties.emplace(
            ov::hint::dynamic_quantization_group_size(std::uint64_t{32}));
        properties.emplace(
            ov::intel_gpu::hint::enable_sdpa_optimization(true));
    }
    if (!cache_dir.empty()) {
        properties.emplace(ov::cache_dir(cache_dir));
    }
    try {
        m_pipeline =
            std::make_shared<ov::genai::Text2VideoPipeline>(m_models_path.string(),
                                                            m_device, properties);
    } catch (const std::exception& e) {
        std::cerr << "[video model '" << id << "'] loading FAILED: " << e.what()
                  << std::endl;
        throw;
    }
    const auto load_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    std::cerr << "[video model '" << id << "'] loaded in " << load_s
              << " s" << std::endl;
    std::cerr << "[video model '" << id << "'] memory: ";
    const double gpu_mib = device_gpu_mib(m_device);
    if (gpu_mib >= 0.0) {
        std::cerr << "GPU " << std::fixed << std::setprecision(2)
                  << (gpu_mib / 1024.0) << " GiB" << std::endl;
    } else if (m_device.find("GPU") != std::string::npos) {
        std::cerr << "GPU ??? GiB" << std::endl;
    } else {
        std::cerr << "CPU " << std::fixed << std::setprecision(2)
                  << (proc_field_kb("VmRSS:") / (1024.0 * 1024.0)) << " GiB"
                  << std::endl;
    }
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
    // Arbitrary requested sizes (e.g. 1280x720, where 720 is not a multiple of
    // 32) are generated at the next multiple of 32 -- the LTX-Video VAE's
    // spatial compression factor -- and resized back to the exact requested
    // dimensions below.
    const bool both_dims = opts.width && opts.height;
    if (opts.height)
        properties[ov::genai::height.name()] = static_cast<std::int64_t>(
            both_dims ? align32(*opts.height) : *opts.height);
    if (opts.width)
        properties[ov::genai::width.name()] = static_cast<std::int64_t>(
            both_dims ? align32(*opts.width) : *opts.width);
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

    // Deliver exactly the requested size even when the model output was padded
    // up to a multiple of 32.
    if (both_dims) {
        const std::size_t out_w = static_cast<std::size_t>(*opts.width);
        const std::size_t out_h = static_cast<std::size_t>(*opts.height);
        for (auto& v : videos) {
            for (auto& f : v.frames) {
                f = resize_frame(f, out_w, out_h);
            }
        }
        const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        std::cerr << "video gen done: req=" << req_id << " ms=" << elapsed
                  << " videos=" << num_videos << " frames=" << num_frames
                  << " decoded=" << height << "x" << width
                  << " output=" << out_h << "x" << out_w << std::endl;
        return videos;
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