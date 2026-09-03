// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/manager.hpp"

#include <stdexcept>

namespace ovserver {

ModelManager& ModelManager::instance() {
    static ModelManager mgr;
    return mgr;
}

void ModelManager::load(const std::string& id, const ModelSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_models.find(id) != m_models.end()) {
        throw std::runtime_error("model '" + id + "' already loaded");
    }
    auto model = std::make_shared<Model>(id, spec.path, spec.device,
                                         spec.properties,
                                         spec.text_encoder_device,
                                         spec.transformer_device,
                                         spec.vae_device,
                                         spec.static_shapes,
                                         spec.bound_dynamic,
                                         spec.naive,
                                         spec.bound_max);
    m_models.emplace(id, std::move(model));
}

Model* ModelManager::get(const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_models.find(id);
    if (it == m_models.end()) {
        return nullptr;
    }
    return it->second.get();
}

void ModelManager::load_vlm(const std::string& id, const VLMModelSpec& spec) {
    std::lock_guard lock(m_mutex);
    if (m_vlms.find(id) != m_vlms.end()) {
        throw std::runtime_error("vlm model '" + id + "' already loaded");
    }
    auto model = std::make_shared<VLMModel>(id, spec.path, spec.device,
                                            spec.properties);
    m_vlms.emplace(id, std::move(model));
}

VLMModel* ModelManager::get_vlm(const std::string& id) const {
    std::lock_guard lock(m_mutex);
    auto it = m_vlms.find(id);
    if (it == m_vlms.end()) {
        return nullptr;
    }
    return it->second.get();
}

}  // namespace ovserver
