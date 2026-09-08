// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/image_generation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include <openvino/genai/image_generation/generation_config.hpp>
#include <openvino/genai/image_generation/text2image_pipeline.hpp>
#include <openvino/runtime/core.hpp>
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

}  // namespace

ImageResult extract_image(const ov::Tensor& result, std::size_t index) {
    auto shape = result.get_shape();
    ImageResult img;
    img.height = static_cast<int>(shape[1]);
    img.width = static_cast<int>(shape[2]);
    img.channels = static_cast<int>(shape[3]);

    const auto* data = result.data<uint8_t>();
    const std::size_t plane =
        static_cast<std::size_t>(img.height * img.width * img.channels);
    const std::size_t offset = index * plane;
    img.data.assign(data + offset, data + offset + plane);
    return img;
}

ImageGenerationModel::ImageGenerationModel(
    const std::string& id,
    const std::filesystem::path& models_path,
    const std::string& device)
    : m_id(id), m_models_path(models_path), m_device(device) {
    // Build the pipeline on startup, exactly like the reference Python path
    // (Text2ImagePipeline(path, device)): no shape or plugin configuration.
    const auto t0 = std::chrono::steady_clock::now();
    std::cerr << "[image model '" << id << "'] loading from " << models_path
              << " on " << device << " ..." << std::endl;
    try {
        m_pipeline = std::make_shared<ov::genai::Text2ImagePipeline>(
            m_models_path, m_device);
    } catch (const std::exception& e) {
        std::cerr << "[image model '" << id << "'] loading FAILED: " << e.what()
                  << std::endl;
        throw;
    }
    const auto load_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    std::cerr << "[image model '" << id << "'] loaded in " << load_s
              << " s" << std::endl;
    std::cerr << "[image model '" << id << "'] memory: ";
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

    m_last_snapshot = std::chrono::steady_clock::now();
    m_snapshot_thread = std::thread([this] { snapshot_run(); });
}

ImageGenerationModel::~ImageGenerationModel() {
    m_stop = true;
    if (m_snapshot_thread.joinable()) {
        m_snapshot_thread.join();
    }
}

void ImageGenerationModel::snapshot_run() {
    while (!m_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - m_last_snapshot).count() < 1.0) {
            continue;
        }
        std::lock_guard<std::mutex> lock(m_metrics_mutex);
        if (m_steps.empty()) {
            continue;
        }
        // Only requests that have completed at least one denoising step are
        // eligible: average their latest step duration.
        double acc = 0.0;
        std::size_t n = 0;
        for (const auto& [id, s] : m_steps) {
            if (s.steps == 0) {
                continue;
            }
            acc += s.last_step_s;
            ++n;
        }
        if (n == 0) {
            continue;
        }
        m_last_snapshot = now;
        std::cerr << "[image model '" << m_id << "'] diffusion "
                  << (acc / static_cast<double>(n)) << " s/step (" << n
                  << " req)" << std::endl;
    }
}

std::vector<ImageResult> ImageGenerationModel::generate(
    const ImageGenerateOptions& opts) {
    ov::AnyMap properties;
    properties[ov::genai::num_images_per_prompt.name()] = opts.num_images;

    if (opts.negative_prompt) {
        properties[ov::genai::negative_prompt.name()] = *opts.negative_prompt;
    }
    if (opts.guidance_scale) {
        properties[ov::genai::guidance_scale.name()] = *opts.guidance_scale;
    }
    if (opts.height) {
        properties[ov::genai::height.name()] = *opts.height;
    }
    if (opts.width) {
        properties[ov::genai::width.name()] = *opts.width;
    }
    if (opts.num_inference_steps) {
        properties[ov::genai::num_inference_steps.name()] =
            *opts.num_inference_steps;
    }
    if (opts.rng_seed) {
        properties[ov::genai::rng_seed.name()] = *opts.rng_seed;
    }

    const std::uint64_t req_id = m_next_req_id++;
    {
        std::lock_guard<std::mutex> lock(m_metrics_mutex);
        m_steps[req_id] = {};
    }

    const auto gen_start = std::chrono::steady_clock::now();
    // Record seconds per completed denoising step (invoked on the pipeline's
    // callback thread; negligible overhead). The snapshot thread averages them.
    properties[ov::genai::callback.name()] =
        std::function<bool(size_t, size_t, ov::Tensor&)>(
            [this, req_id, last = gen_start](size_t, size_t, ov::Tensor&)
                mutable -> bool {
                const auto now = std::chrono::steady_clock::now();
                const auto step_s =
                    std::chrono::duration<double>(now - last).count();
                last = now;
                std::lock_guard<std::mutex> lock(m_metrics_mutex);
                StepStats& s = m_steps[req_id];
                s.last_step_s = step_s;
                ++s.steps;
                return false;
            });

    ov::Tensor result;
    std::size_t steps = 0;
    double gen_s = 0.0;
    try {
        // clone() shares the compiled models while giving every concurrent
        // request a private scheduler.
        std::lock_guard<std::mutex> lock(m_mutex);
        ov::genai::Text2ImagePipeline pipe = m_pipeline->clone();

        result = pipe.generate(opts.prompt, properties);
        gen_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - gen_start)
                    .count();
    } catch (...) {
        std::lock_guard<std::mutex> lock(m_metrics_mutex);
        steps = m_steps[req_id].steps;
        m_steps.erase(req_id);
        throw;
    }
    {
        std::lock_guard<std::mutex> lock(m_metrics_mutex);
        steps = m_steps[req_id].steps;
        m_steps.erase(req_id);
    }
    std::cerr << "[image model '" << m_id << "'] request " << req_id
              << " finished in " << gen_s << " s (" << steps << " steps";
    if (steps > 0) {
        std::cerr << ", avg " << (gen_s / static_cast<double>(steps))
                  << " s/step";
    }
    std::cerr << ")" << std::endl;

    // result shape: [N, H, W, C] u8
    auto shape = result.get_shape();
    const std::size_t num_images = shape[0];

    std::vector<ImageResult> images;
    images.reserve(num_images);
    for (std::size_t i = 0; i < num_images; ++i) {
        images.push_back(extract_image(result, i));
    }
    return images;
}

}  // namespace ovserver