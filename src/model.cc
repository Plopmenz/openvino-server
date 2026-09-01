// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/model.hpp"

#include <openvino/genai/image_generation/generation_config.hpp>

namespace {

ovserver::ImageResult extract_image(const ov::Tensor& result, std::size_t index) {
    auto shape = result.get_shape();
    ovserver::ImageResult img;
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

}  // namespace

namespace ovserver {

Model::Model(const std::string& id,
             const std::filesystem::path& models_path,
             const std::string& device,
             const ov::AnyMap& properties)
    : m_id(id), m_pipeline(models_path, device, properties) {
}

std::vector<ImageResult> Model::generate(const GenerateOptions& opts) {
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

    // clone() shares the compiled models while giving each concurrent request a
    // private scheduler, so generation is safe in parallel.
    ov::genai::Text2ImagePipeline pipe_clone = m_pipeline.clone();
    ov::Tensor result = pipe_clone.generate(opts.prompt, properties);

    // result shape: [N, H, W, C] u8
    auto shape = result.get_shape();
    const size_t num_images = shape[0];

    std::vector<ImageResult> images;
    images.reserve(num_images);
    for (size_t i = 0; i < num_images; ++i) {
        images.push_back(extract_image(result, i));
    }
    return images;
}

}  // namespace ovserver