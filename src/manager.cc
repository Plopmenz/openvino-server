// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/manager.hpp"

#include <stdexcept>

namespace ovserver {

ModelManager& ModelManager::instance() {
    static ModelManager mgr;
    return mgr;
}

std::shared_ptr<ImageGenerationModel> ModelManager::get_image(
    const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_image_models.find(id);
    if (it == m_image_models.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<TextGenerationModel> ModelManager::get_text(
    const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_text_models.find(id);
    if (it == m_text_models.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<VideoGenerationModel> ModelManager::get_video(
    const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_video_models.find(id);
    if (it == m_video_models.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<ASRModel> ModelManager::get_asr(const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_asr_models.find(id);
    if (it == m_asr_models.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<TTSModel> ModelManager::get_tts(const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_tts_models.find(id);
    if (it == m_tts_models.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<Qwen3TTSModel> ModelManager::get_qwen3_tts(
    const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_qwen3_tts_models.find(id);
    if (it == m_qwen3_tts_models.end()) {
        return nullptr;
    }
    return it->second;
}

void ModelManager::load_image(const std::string& id,
                              const ImageGenerationSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_image_models.find(id) != m_image_models.end()) {
        throw std::runtime_error("image model '" + id + "' already loaded");
    }
    auto model = std::make_shared<ImageGenerationModel>(
        id, spec.path, spec.device, spec.cache_dir);
    m_image_models.emplace(id, std::move(model));
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

void ModelManager::load_video(const std::string& id,
                              const VideoGenerationSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_video_models.find(id) != m_video_models.end()) {
        throw std::runtime_error("video model '" + id + "' already loaded");
    }
    auto model =
        std::make_shared<VideoGenerationModel>(id, spec.path, spec.device,
                                               spec.cache_dir);
    m_video_models.emplace(id, std::move(model));
}

void ModelManager::load_asr(const std::string& id, const ASRSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_asr_models.find(id) != m_asr_models.end()) {
        throw std::runtime_error("asr model '" + id + "' already loaded");
    }
    auto model = std::make_shared<ASRModel>(id, spec.path, spec.device,
                                            spec.cache_dir);
    m_asr_models.emplace(id, std::move(model));
}

void ModelManager::load_tts(const std::string& id, const TTSSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_tts_models.find(id) != m_tts_models.end() ||
        m_qwen3_tts_models.find(id) != m_qwen3_tts_models.end()) {
        throw std::runtime_error("tts model '" + id + "' already loaded");
    }
    if (is_qwen3_tts_layout(spec.path)) {
        auto model = std::make_shared<Qwen3TTSModel>(id, spec.path, spec.device,
                                                     spec.cache_dir);
        m_qwen3_tts_models.emplace(id, std::move(model));
    } else {
        auto model = std::make_shared<TTSModel>(id, spec.path, spec.device,
                                                spec.cache_dir);
        m_tts_models.emplace(id, std::move(model));
    }
}

std::unordered_map<std::string, std::shared_ptr<ImageGenerationModel>>
ModelManager::all_images() const {
    std::lock_guard lock(m_mutex);
    return m_image_models;
}

std::unordered_map<std::string, std::shared_ptr<TextGenerationModel>>
ModelManager::all_text() const {
    std::lock_guard lock(m_mutex);
    return m_text_models;
}

std::unordered_map<std::string, std::shared_ptr<VideoGenerationModel>>
ModelManager::all_video() const {
    std::lock_guard lock(m_mutex);
    return m_video_models;
}

std::unordered_map<std::string, std::shared_ptr<ASRModel>>
ModelManager::all_asr() const {
    std::lock_guard lock(m_mutex);
    return m_asr_models;
}

std::unordered_map<std::string, std::shared_ptr<TTSModel>>
ModelManager::all_tts() const {
    std::lock_guard lock(m_mutex);
    return m_tts_models;
}

std::unordered_map<std::string, std::shared_ptr<Qwen3TTSModel>>
ModelManager::all_qwen3_tts() const {
    std::lock_guard lock(m_mutex);
    return m_qwen3_tts_models;
}

void ModelManager::shutdown() {
    std::lock_guard lock(m_mutex);
    m_image_models.clear();
    m_text_models.clear();
    m_video_models.clear();
    m_asr_models.clear();
    m_tts_models.clear();
    m_qwen3_tts_models.clear();
}

}  // namespace ovserver