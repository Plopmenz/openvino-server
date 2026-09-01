// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/image_generation/text2image_pipeline.hpp>

namespace ovserver {

// All fields are optional so that model-specific defaults from the pipeline
// (e.g. Qwen-Image's guidance=4.0, 50 steps, 1024x1024) are preserved unless a
// client explicitly overrides them.
struct GenerateOptions {
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

class Model {
public:
    Model(const std::string& id,
          const std::filesystem::path& models_path,
          const std::string& device,
          const ov::AnyMap& properties);

    const std::string& id() const { return m_id; }

    // Runs generation, returning one ImageResult per generated image.
    std::vector<ImageResult> generate(const GenerateOptions& opts);

private:
    std::string m_id;
    ov::genai::Text2ImagePipeline m_pipeline;
};

struct ModelSpec {
    std::filesystem::path path;
    std::string device;
    ov::AnyMap properties;
};

}  // namespace ovserver