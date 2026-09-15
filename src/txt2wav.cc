// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/txt2wav.hpp"

#include <chrono>
#include <iostream>

#include <openvino/genai/speech_generation/text2speech_pipeline.hpp>

#include "ovserver/common.hpp"

namespace ovserver {

TTSModel::TTSModel(const std::string& id,
                   const std::filesystem::path& models_path,
                   const std::string& device,
                   const std::string& cache_dir)
    : m_id(id), m_models_path(models_path), m_device(device) {
    PipelineLoadLog load_log("tts", id, m_models_path, m_device);
    try {
        m_pipeline =
            std::make_shared<ov::genai::Text2SpeechPipeline>(
                m_models_path.string(), m_device,
                inference_properties(m_device, cache_dir));
    } catch (const std::exception& e) {
        std::cerr << "[tts model '" << id << "'] loading FAILED: " << e.what()
                  << std::endl;
        throw;
    }
    load_log.completion();
}

TTSModel::~TTSModel() = default;

TTSResult TTSModel::generate(const std::string& text,
                             const ov::Tensor& speaker_embedding) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const std::uint64_t req_id = m_next_req_id++;
    std::cerr << "[tts model '" << m_id << "'] request " << req_id << " start"
              << std::endl;
    const auto start = std::chrono::steady_clock::now();
    const ov::genai::Text2SpeechDecodedResults result =
        m_pipeline->generate(text, speaker_embedding);
    if (result.speeches.empty())
        throw std::runtime_error("tts: no speech produced");
    const ov::Tensor& speech = result.speeches.front();
    if (speech.get_element_type() != ov::element::f32)
        throw std::runtime_error("tts: unexpected sample element type " +
                                 speech.get_element_type().get_type_name());
    const float* data = speech.data<float>();
    const std::size_t count = static_cast<std::size_t>(speech.get_size());
    TTSResult out;
    out.sample_rate = result.output_sample_rate;
    out.samples.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float v = data[i];
        const float c = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
        out.samples.push_back(
            static_cast<std::int16_t>(c * 32767.0f));
    }
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    const double audio_s =
        out.sample_rate > 0
            ? static_cast<double>(count) / static_cast<double>(out.sample_rate)
            : 0.0;
    std::cerr << "[tts model '" << m_id << "'] request " << req_id
              << " finished in " << elapsed << " s audio_s=" << audio_s
              << " ratio=" << (elapsed > 0 ? audio_s / elapsed : 0.0) << "x"
              << std::endl;
    return out;
}

}  // namespace ovserver