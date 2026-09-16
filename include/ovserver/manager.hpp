// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "ovserver/wav2txt.hpp"
#include "ovserver/txt2img.hpp"
#include "ovserver/txt2txt.hpp"
#include "ovserver/txt2wav.hpp"
#include "ovserver/txt2vid.hpp"
#include "ovserver/qwen3_tts.hpp"

namespace ovserver {

class ModelManager {
public:
    static ModelManager& instance();

    void load_image(const std::string& id, const ImageGenerationSpec& spec);
    std::shared_ptr<ImageGenerationModel> get_image(const std::string& id) const;

    void load_text(const std::string& id, const TextGenerationSpec& spec);
    std::shared_ptr<TextGenerationModel> get_text(const std::string& id) const;

    void load_video(const std::string& id, const VideoGenerationSpec& spec);
    std::shared_ptr<VideoGenerationModel> get_video(const std::string& id) const;

    void load_asr(const std::string& id, const ASRSpec& spec);
    std::shared_ptr<ASRModel> get_asr(const std::string& id) const;

    void load_tts(const std::string& id, const TTSSpec& spec);
    std::shared_ptr<TTSModel> get_tts(const std::string& id) const;
    std::shared_ptr<Qwen3TTSModel> get_qwen3_tts(const std::string& id) const;

    // Releases all loaded models. Called explicitly on shutdown so genai/OV
    // objects are destroyed while the process and OpenVINO plugins are still
    // fully initialized (teardown during C++ static destruction segfaults).
    // Callers holding a shared_ptr from the get_*() accessors keep their model
    // alive across this call.
    void shutdown();

    // Snapshots of the loaded models, taken under the registry lock so the
    // caller can iterate without racing concurrent load()/shutdown() calls.
    std::unordered_map<std::string, std::shared_ptr<ImageGenerationModel>>
    all_images() const;

    std::unordered_map<std::string, std::shared_ptr<TextGenerationModel>>
    all_text() const;

    std::unordered_map<std::string, std::shared_ptr<VideoGenerationModel>>
    all_video() const;

    std::unordered_map<std::string, std::shared_ptr<ASRModel>> all_asr() const;

    std::unordered_map<std::string, std::shared_ptr<TTSModel>> all_tts() const;

    std::unordered_map<std::string, std::shared_ptr<Qwen3TTSModel>>
    all_qwen3_tts() const;

private:
    mutable std::mutex m_mutex;
    std::unordered_map<std::string, std::shared_ptr<ImageGenerationModel>>
        m_image_models;
    std::unordered_map<std::string, std::shared_ptr<TextGenerationModel>>
        m_text_models;
    std::unordered_map<std::string, std::shared_ptr<VideoGenerationModel>>
        m_video_models;
    std::unordered_map<std::string, std::shared_ptr<ASRModel>> m_asr_models;
    std::unordered_map<std::string, std::shared_ptr<TTSModel>> m_tts_models;
    std::unordered_map<std::string, std::shared_ptr<Qwen3TTSModel>>
        m_qwen3_tts_models;
};

}  // namespace ovserver