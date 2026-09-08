// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/manager.hpp"

#include <stdexcept>

namespace ovserver {

ModelManager& ModelManager::instance() {
    static ModelManager mgr;
    return mgr;
}

void ModelManager::load_image(const std::string& id,
                              const ImageGenerationSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_image_models.find(id) != m_image_models.end()) {
        throw std::runtime_error("image model '" + id + "' already loaded");
    }
    auto model = std::make_shared<ImageGenerationModel>(id, spec.path, spec.device);
    m_image_models.emplace(id, std::move(model));
}

ImageGenerationModel* ModelManager::get_image(const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_image_models.find(id);
    if (it == m_image_models.end()) {
        return nullptr;
    }
    return it->second.get();
}

void ModelManager::load_text(const std::string& id,
                             const TextGenerationSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_text_models.find(id) != m_text_models.end()) {
        throw std::runtime_error("text model '" + id + "' already loaded");
    }
    auto model = std::make_shared<TextGenerationModel>(id, spec);
    m_text_models.emplace(id, std::move(model));
}

TextGenerationModel* ModelManager::get_text(const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_text_models.find(id);
    if (it == m_text_models.end()) {
        return nullptr;
    }
    return it->second.get();
}

void ModelManager::load_video(const std::string& id,
                              const VideoGenerationSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_video_models.find(id) != m_video_models.end()) {
        throw std::runtime_error("video model '" + id + "' already loaded");
    }
    auto model =
        std::make_shared<VideoGenerationModel>(id, spec.path, spec.device);
    m_video_models.emplace(id, std::move(model));
}

VideoGenerationModel* ModelManager::get_video(const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_video_models.find(id);
    if (it == m_video_models.end()) {
        return nullptr;
    }
    return it->second.get();
}

void ModelManager::load_asr(const std::string& id, const ASRSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_asr_models.find(id) != m_asr_models.end()) {
        throw std::runtime_error("asr model '" + id + "' already loaded");
    }
    auto model = std::make_shared<ASRModel>(id, spec.path, spec.device);
    m_asr_models.emplace(id, std::move(model));
}

ASRModel* ModelManager::get_asr(const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_asr_models.find(id);
    if (it == m_asr_models.end()) {
        return nullptr;
    }
    return it->second.get();
}

void ModelManager::load_tts(const std::string& id, const TTSSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_tts_models.find(id) != m_tts_models.end()) {
        throw std::runtime_error("tts model '" + id + "' already loaded");
    }
    auto model = std::make_shared<TTSModel>(id, spec.path, spec.device);
    m_tts_models.emplace(id, std::move(model));
}

TTSModel* ModelManager::get_tts(const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_tts_models.find(id);
    if (it == m_tts_models.end()) {
        return nullptr;
    }
    return it->second.get();
}

void ModelManager::shutdown() {
    std::lock_guard lock(m_mutex);
    m_image_models.clear();
    m_text_models.clear();
    m_video_models.clear();
    m_asr_models.clear();
    m_tts_models.clear();
}

}  // namespace ovserver