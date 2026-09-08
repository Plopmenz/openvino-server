// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "ovserver/audio.hpp"
#include "ovserver/image_generation.hpp"
#include "ovserver/text_generation.hpp"
#include "ovserver/video_generation.hpp"

namespace ovserver {

class ModelManager {
public:
    static ModelManager& instance();

    void load_image(const std::string& id, const ImageGenerationSpec& spec);
    ImageGenerationModel* get_image(const std::string& id) const;

    void load_text(const std::string& id, const TextGenerationSpec& spec);
    TextGenerationModel* get_text(const std::string& id) const;

    void load_video(const std::string& id, const VideoGenerationSpec& spec);
    VideoGenerationModel* get_video(const std::string& id) const;

    void load_asr(const std::string& id, const ASRSpec& spec);
    ASRModel* get_asr(const std::string& id) const;

    void load_tts(const std::string& id, const TTSSpec& spec);
    TTSModel* get_tts(const std::string& id) const;

    // Releases all loaded models. Called explicitly on shutdown so genai/OV
    // objects are destroyed while the process and OpenVINO plugins are still
    // fully initialized (teardown during C++ static destruction segfaults).
    void shutdown();

    const std::unordered_map<std::string, std::shared_ptr<ImageGenerationModel>>&
    all_images() const {
        return m_image_models;
    }

    const std::unordered_map<std::string, std::shared_ptr<TextGenerationModel>>&
    all_text() const {
        return m_text_models;
    }

    const std::unordered_map<std::string, std::shared_ptr<VideoGenerationModel>>&
    all_video() const {
        return m_video_models;
    }

    const std::unordered_map<std::string, std::shared_ptr<ASRModel>>& all_asr()
        const {
        return m_asr_models;
    }

    const std::unordered_map<std::string, std::shared_ptr<TTSModel>>& all_tts()
        const {
        return m_tts_models;
    }

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
};

}  // namespace ovserver