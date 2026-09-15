// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <vector>

namespace ovserver {

// Mirrors the server's capabilities flags (--txt2img, --txt2txt, ...) so the
// enum maps 1:1 onto the CLI options.
enum class ModelCapability { Unknown, Txt2Img, Txt2Txt, Txt2Vid, Wav2Txt, Txt2Wav };

// Identifies the capabilities a model directory unlocks by matching its
// identity (config.json "architectures"[0], or model_index.json "_class_name"
// when config.json is absent) against a maintained list. Returns an empty
// vector when the identity is not in the list.
std::vector<ModelCapability> detect_model_capabilities(
    const std::filesystem::path& model_path);

// CLI flag for a capability, e.g. "--txt2txt". Nullptr for Unknown.
const char* to_capability_flag(ModelCapability capability);

// Short display name, e.g. "txt2txt". Nullptr for Unknown.
const char* capability_name(ModelCapability capability);

}  // namespace ovserver