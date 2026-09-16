// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3-TTS (12 Hz base, INT8 OpenVINO) direct inference backend. This is a
// hand-rolled C++ port of the reference notebook's OpenVINO helpers
// (_generate_talker_codes, _chunked_ov_decode, speech_tokenizer vc/icl
// encoding) driven with ov::Core directly. It exists because
// ov::genai::Text2SpeechPipeline rejects Qwen3TTSForConditionalGeneration
// ("Unsupported text to speech generation pipeline 'Qwen3TTS...'").

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openvino/openvino.hpp>

#include "ovserver/txt2wav.hpp"

namespace ovserver {

// Byte-level GPT2 BPE encoder (byte->unicode tables + merges lattice),
// faithful port of the reference byte BPE used for prompt building.
class Gpt2BPEEncoder {
public:
    using TokenT = std::int32_t;

    struct Result {
        std::vector<std::int64_t> ids;
        std::vector<std::string> pieces;
    };

    Gpt2BPEEncoder() = default;

    // vocab.json + merges.txt from the model directory.
    bool load(const std::filesystem::path& vocab_path,
              const std::filesystem::path& merges_path);

    Result encode(const std::string& text) const;

private:
    std::unordered_map<std::string, TokenT> m_vocab;
    std::vector<std::pair<std::string, std::string>> m_merges;
};

// Computes plain single-axis RoPE tables (theta=1e6, head_dim=128), as used
// by the exported talker/code_predictor `cos`/`sin` inputs [1,1,L,128].
void compute_rope_tables(std::int64_t length, std::int64_t head_dim,
                         double theta, double dt, bool mrc_repeat,
                         std::vector<float>& cos, std::vector<float>& sin);

// Auto-detection: does `path` point at a Qwen3-TTS IR export (talker +
// code_predictor + speech codec, NOT a genai speech model)?
bool is_qwen3_tts_layout(const std::filesystem::path& path);

class Qwen3TTSModel {
public:
    Qwen3TTSModel(const std::string& id,
                  const std::filesystem::path& models_path,
                  const std::string& device,
                  const std::string& cache_dir);

    Qwen3TTSModel(const Qwen3TTSModel&) = delete;
    Qwen3TTSModel& operator=(const Qwen3TTSModel&) = delete;
    ~Qwen3TTSModel();

    const std::string& id() const { return m_id; }

    const std::filesystem::path& models_path() const { return m_models_path; }

    TTSResult generate(const std::string& text,
                       const ov::Tensor& speaker_embedding = ov::Tensor());

private:
    std::string m_id;
    std::filesystem::path m_models_path;
    std::string m_device;
    std::string m_cache_dir;

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ovserver
