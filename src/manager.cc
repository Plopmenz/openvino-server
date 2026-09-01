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
    auto model =
        std::make_shared<Model>(id, spec.path, spec.device, spec.properties);
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

}  // namespace ovserver