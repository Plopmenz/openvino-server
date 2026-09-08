// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <openvino/genai/automatic_speech_recognition/pipeline.hpp>
#include <openvino/genai/speech_generation/text2speech_pipeline.hpp>

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
             const std::string& device);

    ASRModel(const ASRModel&) = delete;
    ASRModel& operator=(const ASRModel&) = delete;

    ~ASRModel();

    const std::string& id() const { return m_id; }

    ASRResult generate(const ASRGenerateOptions& opts,
                       ASROnText on_text = {});

private:
    std::string m_id;
    std::filesystem::path m_models_path;
    std::string m_device;

    std::mutex m_mutex;
    std::shared_ptr<ov::genai::ASRPipeline> m_pipeline;
};

// Opens the reference speech_generation_config.hpp path the same way
// genai's own samples do: Text2SpeechPipeline(path, device) produces
// Text2SpeechDecodedResults with float speeches.
struct TTSResult {
    std::vector<std::int16_t> samples;
    std::uint32_t sample_rate = 0;
};

class TTSModel {
public:
    TTSModel(const std::string& id,
             const std::filesystem::path& models_path,
             const std::string& device);

    TTSModel(const TTSModel&) = delete;
    TTSModel& operator=(const TTSModel&) = delete;

    ~TTSModel();

    const std::string& id() const { return m_id; }

    TTSResult generate(const std::string& text);

private:
    std::string m_id;
    std::filesystem::path m_models_path;
    std::string m_device;

    std::mutex m_mutex;
    std::shared_ptr<ov::genai::Text2SpeechPipeline> m_pipeline;
};

struct ASRSpec {
    std::filesystem::path path;
    std::string device;
};

struct TTSSpec {
    std::filesystem::path path;
    std::string device;
};

// Encodes int16 PCM samples into a self-contained WAV file (RIFF). Computed
// natively, so it works even without ffmpeg.
std::vector<std::uint8_t> wav_pcm16(const std::vector<std::int16_t>& samples,
                                    std::uint32_t sample_rate);

// Decodes an arbitrary audio byte stream (any ffmpeg-supported container /
// codec) into float32 mono PCM at the given sample rate (default 16 kHz).
// Implemented by piping the bytes to ffmpeg. Throws std::runtime_error if
// ffmpeg is unavailable or decoding fails.
std::vector<float> decode_audio_to_f32(const std::string& data,
                                       int sample_rate = 16000);

// Path to the ffmpeg executable used for audio decoding and MP4 encoding.
// Defaults to "ffmpeg" (resolved via PATH). Calling set_ffmpeg_path("")
// disables ffmpeg-dependent features (they then raise std::runtime_error).
void set_ffmpeg_path(const std::string& path);
const std::string& ffmpeg_path();

}  // namespace ovserver