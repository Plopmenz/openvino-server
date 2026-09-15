// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/model_detect.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>

#include <json/json.h>

namespace ovserver {

namespace {

// Identity <-> capabilities table. Kept deliberately small: only the models
// the server is actually used with. To support another model, add one line.
// The identity is either config.json "architectures"[0] or, when config.json
// is absent, model_index.json "_class_name".
const std::unordered_map<std::string, std::vector<ModelCapability>>&
identity_capabilities() {
    static const std::unordered_map<std::string, std::vector<ModelCapability>>
        table = {
            {"Qwen3_5MoeForConditionalGeneration",
             {ModelCapability::Txt2Txt}},
            {"Qwen3ASRForConditionalGeneration", {ModelCapability::Wav2Txt}},
            {"Qwen3TTSForConditionalGeneration", {ModelCapability::Txt2Wav}},
            {"QwenImagePipeline", {ModelCapability::Txt2Img}},
        };
    return table;
}

std::string read_identity(const std::filesystem::path& model_path) {
    // Prefer config.json architectures[0] (transformers export).
    const std::filesystem::path config_path = model_path / "config.json";
    if (std::ifstream config{config_path}) {
        Json::Value root;
        try {
            config >> root;
        } catch (const Json::Exception&) {
            return {};
        }
        if (root.isObject() && root["architectures"].isArray() &&
            root["architectures"].size() > 0 &&
            root["architectures"][0].isString()) {
            return root["architectures"][0].asString();
        }
    }
    // Fall back to the diffusers model_index.json _class_name.
    const std::filesystem::path index_path = model_path / "model_index.json";
    if (std::ifstream index{index_path}) {
        Json::Value root;
        try {
            index >> root;
        } catch (const Json::Exception&) {
            return {};
        }
        if (root.isObject() && root["_class_name"].isString()) {
            return root["_class_name"].asString();
        }
    }
    return {};
}

}  // namespace

std::vector<ModelCapability> detect_model_capabilities(
    const std::filesystem::path& model_path) {
    const std::string id = read_identity(model_path);
    if (id.empty()) {
        return {};
    }
    const auto& table = identity_capabilities();
    const auto it = table.find(id);
    if (it == table.end()) {
        return {};
    }
    return it->second;
}

const char* to_capability_flag(ModelCapability capability) {
    switch (capability) {
        case ModelCapability::Txt2Img: return "--txt2img";
        case ModelCapability::Txt2Txt: return "--txt2txt";
        case ModelCapability::Txt2Vid: return "--txt2vid";
        case ModelCapability::Wav2Txt: return "--wav2txt";
        case ModelCapability::Txt2Wav: return "--txt2wav";
        case ModelCapability::Unknown: return nullptr;
    }
    return nullptr;
}

const char* capability_name(ModelCapability capability) {
    switch (capability) {
        case ModelCapability::Txt2Img: return "txt2img";
        case ModelCapability::Txt2Txt: return "txt2txt";
        case ModelCapability::Txt2Vid: return "txt2vid";
        case ModelCapability::Wav2Txt: return "wav2txt";
        case ModelCapability::Txt2Wav: return "txt2wav";
        case ModelCapability::Unknown: return nullptr;
    }
    return nullptr;
}

}  // namespace ovserver