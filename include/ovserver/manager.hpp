// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "ovserver/model.hpp"
#include "ovserver/vlm_model.hpp"

namespace ovserver {

class ModelManager {
public:
    static ModelManager& instance();

    void load(const std::string& id, const ModelSpec& spec);
    Model* get(const std::string& id) const;

    void load_vlm(const std::string& id, const VLMModelSpec& spec);
    VLMModel* get_vlm(const std::string& id) const;

    const std::unordered_map<std::string, std::shared_ptr<Model>>& all() const {
        return m_models;
    }

    const std::unordered_map<std::string, std::shared_ptr<VLMModel>>&
    all_vlms() const {
        return m_vlms;
    }

private:
    mutable std::mutex m_mutex;
    std::unordered_map<std::string, std::shared_ptr<Model>> m_models;
    std::unordered_map<std::string, std::shared_ptr<VLMModel>> m_vlms;
};

}  // namespace ovserver