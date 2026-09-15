// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/wav2txt.hpp"

#include <chrono>
#include <iostream>

#include <openvino/genai/automatic_speech_recognition/pipeline.hpp>

#include "ovserver/common.hpp"

namespace ovserver {

ASRModel::ASRModel(const std::string& id,
                   const std::filesystem::path& models_path,
                   const std::string& device,
                   const std::string& cache_dir)
    : m_id(id) {
    PipelineLoadLog load_log("asr", id, models_path, device);
    try {
        m_pipeline = std::make_shared<ov::genai::ASRPipeline>(
            models_path.string(), device,
            inference_properties(device, cache_dir));
    } catch (const std::exception& e) {
        std::cerr << "[asr model '" << id << "'] loading FAILED: " << e.what()
                  << std::endl;
        throw;
    }
    load_log.completion();
}

ASRModel::~ASRModel() = default;

ASRResult ASRModel::generate(const ASRGenerateOptions& opts,
                             ASROnText on_text) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const std::uint64_t req_id = m_next_req_id++;
    std::cerr << "[asr model '" << m_id << "'] request " << req_id << " start"
              << std::endl;

    ov::genai::ASRGenerationConfig cfg;
    cfg.language = opts.language;
    cfg.task = opts.task;
    cfg.initial_prompt = opts.initial_prompt;
    cfg.context = opts.context;
    cfg.return_timestamps = opts.return_timestamps;
    if (opts.temperature)
        cfg.temperature = *opts.temperature;
    if (opts.max_new_tokens)
        cfg.max_new_tokens = *opts.max_new_tokens;

    const auto start = std::chrono::steady_clock::now();
    const ov::genai::AudioInputs input{opts.samples};
    ov::genai::ASRDecodedResults result;
    if (on_text) {
        result = m_pipeline->generate(input, cfg, on_text);
    } else {
        result = m_pipeline->generate(input, cfg);
    }

    ASRResult out;
    if (!result.texts.empty())
        out.text = result.texts[0];
    if (!result.languages.empty())
        out.language = result.languages[0];
    if (result.chunks && !result.chunks->empty()) {
        for (const auto& chunk : (*result.chunks)[0]) {
            ASRSegment seg;
            seg.start = chunk.start_ts;
            seg.end = chunk.end_ts;
            seg.text = chunk.text;
            out.segments.push_back(std::move(seg));
        }
    }

    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    const double audio_s =
        static_cast<double>(opts.samples.size()) / 16000.0;
    std::cerr << "[asr model '" << m_id << "'] request " << req_id
              << " finished in " << elapsed << " s audio_s=" << audio_s
              << " ratio=" << (elapsed > 0 ? audio_s / elapsed : 0.0) << "x"
              << std::endl;
    return out;
}

}  // namespace ovserver