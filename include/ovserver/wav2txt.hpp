// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/automatic_speech_recognition/pipeline.hpp>

namespace ovserver {

struct ASRGenerateOptions {
    // Raw mono PCM samples. Must be float32. The endpoint layer resamples
    // whatever the client uploaded to 16 kHz mono before calling this (the
    // rate the speech models are trained on).
    std::vector<float> samples;
    std::optional<std::string> language;
    std::optional<std::string> task;  // "transcribe" | "translate"
    std::optional<std::string> initial_prompt;
    std::optional<std::string> context;
    bool return_timestamps = false;
    std::optional<float> temperature;
    std::optional<std::size_t> max_new_tokens;
};

struct ASRSegment {
    float start = 0.0f;
    float end = 0.0f;
    std::string text;
};

struct ASRResult {
    std::string text;
    std::string language;
    std::vector<ASRSegment> segments;
};

// Callback invoked with decoded text fragments as ASR generation proceeds
// (per token for Whisper without timestamps, per segment when timestamps are
// requested). Return STOP to halt, RUNNING to continue.
using ASROnText =
    std::function<ov::genai::StreamingStatus(const std::string&)>;

// Wraps an ov::genai::ASRPipeline (Whisper / Qwen3-ASR / BLING). Built on
// startup like the Python reference. generate() is serialized with a mutex --
// the speech pipelines are not documented as safe for concurrent generate
// calls.
class ASRModel {
public:
    ASRModel(const std::string& id,
             const std::filesystem::path& models_path,
             const std::string& device,
             const std::string& cache_dir);

    ASRModel(const ASRModel&) = delete;
    ASRModel& operator=(const ASRModel&) = delete;

    ~ASRModel();

    const std::string& id() const { return m_id; }

    ASRResult generate(const ASRGenerateOptions& opts,
                       ASROnText on_text = {});

private:
    std::string m_id;

    std::mutex m_mutex;
    std::shared_ptr<ov::genai::ASRPipeline> m_pipeline;
    std::atomic<std::uint64_t> m_next_req_id{0};
};

struct ASRSpec {
    std::filesystem::path path;
    std::string device;
    // Directory for OpenVINO compiled-model blobs. Empty disables caching.
    std::string cache_dir;
};

}  // namespace ovserver