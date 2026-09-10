// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <openvino/genai/speech_generation/text2speech_pipeline.hpp>
#include <openvino/runtime/tensor.hpp>

namespace ovserver {

// Opens the reference speech_generation_config.hpp path the same way
// genai's own samples do: Text2SpeechPipeline(path, device) produces
// Text2SpeechDecodedResults with float speeches.
struct TTSResult {
    std::vector<std::int16_t> samples;
    std::uint32_t sample_rate = 0;
};

class TTSModel {
public:
    TTSModel(const std::string& id,
             const std::filesystem::path& models_path,
             const std::string& device,
             const std::string& cache_dir);

    TTSModel(const TTSModel&) = delete;
    TTSModel& operator=(const TTSModel&) = delete;

    ~TTSModel();

    const std::string& id() const { return m_id; }

    const std::filesystem::path& models_path() const { return m_models_path; }

    TTSResult generate(const std::string& text,
                       const ov::Tensor& speaker_embedding = ov::Tensor());

private:
    std::string m_id;
    std::filesystem::path m_models_path;
    std::string m_device;

    std::mutex m_mutex;
    std::shared_ptr<ov::genai::Text2SpeechPipeline> m_pipeline;
    std::uint64_t m_next_req_id = 0;
};

struct TTSSpec {
    std::filesystem::path path;
    std::string device;
    // Directory for OpenVINO compiled-model blobs. Empty disables caching.
    std::string cache_dir;
};

}  // namespace ovserver