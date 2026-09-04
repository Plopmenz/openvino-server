// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/image_generation/generation_config.hpp>
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

// Parameters that determine the compile-time static shapes and therefore group
// requests that can share one compiled pipeline. Guidance and num_images are
// bucketed because genai's reshape treats every value > 1 the same.
struct CompileKey {
    int num_images_bucket;
    int guidance_bucket;
    int64_t height;
    int64_t width;

    bool operator<(const CompileKey& o) const {
        if (num_images_bucket != o.num_images_bucket) {
            return num_images_bucket < o.num_images_bucket;
        }
        if (guidance_bucket != o.guidance_bucket) {
            return guidance_bucket < o.guidance_bucket;
        }
        if (height != o.height) {
            return height < o.height;
        }
        return width < o.width;
    }

    bool operator==(const CompileKey& o) const {
        return num_images_bucket == o.num_images_bucket &&
               guidance_bucket == o.guidance_bucket && height == o.height &&
               width == o.width;
    }
};

class Model {
public:
    // Each stage can run on its own device (genai's staged compile). Empty
    // per-stage devices fall back to `device`.
    Model(const std::string& id,
          const std::filesystem::path& models_path,
          const std::string& device,
          const ov::AnyMap& properties,
          std::string text_encoder_device = "",
          std::string transformer_device = "",
          std::string vae_device = "",
          bool static_shapes = true,
          bool naive = false);

    const std::string& id() const { return m_id; }

    // Runs generation, returning one ImageResult per generated image.
    std::vector<ImageResult> generate(const GenerateOptions& opts);

private:
    // Returns a clone of the compiled pipeline matching the config's
    // shape-sensitive parameters, compiling it on first use for each key.
    ov::genai::Text2ImagePipeline compiled_pipeline(
        const ov::genai::ImageGenerationConfig& generation_config);

    // Returns a clone of a single Text2ImagePipeline(model_path, device),
    // built lazily on first use. This mirrors the plain Python/optimum path:
    // no reshape, no staged compile, no per-shape keys.
    ov::genai::Text2ImagePipeline naive_pipeline();

    static constexpr std::size_t kMaxCompiledPipelines = 4;

    std::string m_id;
    std::filesystem::path m_models_path;
    std::string m_device;
    std::string m_text_encoder_device;
    std::string m_transformer_device;
    std::string m_vae_device;
    bool m_static_shapes;
    bool m_bound_dynamic;
    bool m_naive;
    int64_t m_bound_max;
    ov::AnyMap m_properties;
    ov::genai::ImageGenerationConfig m_default_config;

    std::mutex m_mutex;
    std::map<CompileKey, std::shared_ptr<ov::genai::Text2ImagePipeline>>
        m_compiled;
    std::list<CompileKey> m_lru;
    std::shared_ptr<ov::genai::Text2ImagePipeline> m_naive_pipeline;
};

struct ModelSpec {
    std::filesystem::path path;
    std::string device;
    std::string text_encoder_device;
    std::string transformer_device;
    std::string vae_device;
    bool static_shapes = true;
    bool naive = false;
    ov::AnyMap properties;
};

}  // namespace ovserver