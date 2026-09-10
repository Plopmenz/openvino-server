// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/txt2img.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <utility>

#include <openvino/genai/image_generation/generation_config.hpp>
#include <openvino/genai/image_generation/text2image_pipeline.hpp>

#include "ovserver/common.hpp"

namespace ovserver {

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
    const std::string& device,
    const std::string& cache_dir)
    : m_id(id), m_models_path(models_path), m_device(device) {
    // Build the pipeline on startup, exactly like the reference Python path
    // (Text2ImagePipeline(path, device)): no shape or plugin configuration.
    PipelineLoadLog load_log("image", id, models_path, device);
    try {
        m_pipeline = std::make_shared<ov::genai::Text2ImagePipeline>(
            m_models_path, m_device,
            inference_properties(m_device, cache_dir));
    } catch (const std::exception& e) {
        std::cerr << "[image model '" << id << "'] loading FAILED: " << e.what()
                  << std::endl;
        throw;
    }
    load_log.completion();
}

ImageGenerationModel::~ImageGenerationModel() = default;

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

    std::cerr << "[image model '" << m_id << "'] request " << req_id
              << " start" << std::endl;
    const auto gen_start = std::chrono::steady_clock::now();
    // Log each denoising step's duration individually as it completes.
    properties[ov::genai::callback.name()] =
        std::function<bool(size_t, size_t, ov::Tensor&)>(
            [this, req_id, last = gen_start](size_t step, size_t total,
                                             ov::Tensor&) mutable -> bool {
                const auto now = std::chrono::steady_clock::now();
                const auto step_s =
                    std::chrono::duration<double>(now - last).count();
                last = now;
                std::cerr << "[image model '" << m_id << "'] request " << req_id
                          << " step " << step + 1 << "/" << total << ": "
                          << step_s << " s" << std::endl;
                return false;
            });

    ov::Tensor result;
    double gen_s = 0.0;
    {
        // clone() shares the compiled models while giving every concurrent
        // request a private scheduler.
        std::lock_guard<std::mutex> lock(m_mutex);
        ov::genai::Text2ImagePipeline pipe = m_pipeline->clone();

        result = pipe.generate(opts.prompt, properties);
        gen_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - gen_start)
                    .count();
    }
    // result shape: [N, H, W, C] u8
    auto shape = result.get_shape();
    const std::size_t num_images = shape[0];
    const std::size_t height = static_cast<std::size_t>(shape[1]);
    const std::size_t width = static_cast<std::size_t>(shape[2]);

    // The callback already logged each step's duration above.
    std::cerr << "[image model '" << m_id << "'] request " << req_id
              << " finished in " << gen_s << " s resolution=" << width << "x"
              << height << std::endl;

    std::vector<ImageResult> images;
    images.reserve(num_images);
    for (std::size_t i = 0; i < num_images; ++i) {
        images.push_back(extract_image(result, i));
    }
    return images;
}

}  // namespace ovserver