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
    auto model = std::make_shared<TextGenerationModel>(id, spec.path, spec.device);
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

void ModelManager::shutdown() {
    std::lock_guard lock(m_mutex);
    m_image_models.clear();
    m_text_models.clear();
}

}  // namespace ovserver