// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/audio.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <random>

#include <sstream>

#include <iostream>

namespace ovserver {
namespace {

std::string& g_ffmpeg_path() {
    static std::string path = "ffmpeg";
    return path;
}

// Writes `data` to a fresh temp file under the system temp dir and returns
// its path. The caller is responsible for removal.
std::filesystem::path write_temp_file(const std::string& data,
                                      const std::string& suffix) {
    static std::mt19937_64 gen{std::random_device{}()};
    const std::uint64_t salt = gen();
    const std::filesystem::path dir = std::filesystem::temp_directory_path();
    const std::filesystem::path path =
        dir / ("ovserver-" + std::to_string(salt) + suffix);
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("failed to create temp file " +
                                 path.string());
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    out.close();
    if (!out)
        throw std::runtime_error("failed to write temp file " +
                                 path.string());
    return path;
}

std::string shell_quote(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (char c : value) {
        if (c == '"' || c == '\\' || c == '$' || c == '`')
            out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

}  // namespace

std::vector<std::uint8_t> wav_pcm16(const std::vector<std::int16_t>& samples,
                                    std::uint32_t sample_rate) {
    const std::uint32_t num_samples = static_cast<std::uint32_t>(samples.size());
    const std::uint32_t data_bytes = num_samples * 2;
    std::vector<std::uint8_t> out;
    out.reserve(44 + data_bytes);
    auto put_le = [&out](std::uint32_t v, int nbytes) {
        for (int i = 0; i < nbytes; ++i)
            out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xff));
    };
    out.push_back('R');
    out.push_back('I');
    out.push_back('F');
    out.push_back('F');
    put_le(36 + data_bytes, 4);
    out.push_back('W');
    out.push_back('A');
    out.push_back('V');
    out.push_back('E');
    out.push_back('f');
    out.push_back('m');
    out.push_back('t');
    out.push_back(' ');
    put_le(16, 4);            // fmt chunk size
    put_le(1, 2);             // PCM
    put_le(1, 2);             // mono
    put_le(sample_rate, 4);
    put_le(sample_rate * 2, 4);  // byte rate
    put_le(2, 2);             // block align
    put_le(16, 2);            // bits per sample
    out.push_back('d');
    out.push_back('a');
    out.push_back('t');
    out.push_back('a');
    put_le(data_bytes, 4);
    for (std::int16_t s : samples) {
        out.push_back(static_cast<std::uint8_t>(s & 0xff));
        out.push_back(static_cast<std::uint8_t>((s >> 8) & 0xff));
    }
    return out;
}

void set_ffmpeg_path(const std::string& path) {
    g_ffmpeg_path() = path;
}

const std::string& ffmpeg_path() {
    return g_ffmpeg_path();
}

std::vector<float> decode_audio_to_f32(const std::string& data, int sample_rate) {
    if (data.empty())
        return {};
    const std::filesystem::path in = write_temp_file(data, ".in.audio");
    const std::filesystem::path out = write_temp_file("", ".f32");
    std::vector<float> result;
    try {
        const std::string cmd =
            shell_quote(ffmpeg_path()) +
            " -hide_banner -loglevel error -y -i " +
            shell_quote(in.string()) + " -ac 1 -ar " +
            std::to_string(sample_rate) + " -f f32le " + shell_quote(out.string());
        const int status = std::system(cmd.c_str());
        if (status != 0)
            throw std::runtime_error(
                "audio decode: ffmpeg failed (is ffmpeg on PATH?)");
        std::ifstream fin(out, std::ios::binary);
        if (!fin)
            throw std::runtime_error("audio decode: no decoded output");
        std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(fin)),
                                        std::istreambuf_iterator<char>());
        const std::size_t n = bytes.size() / sizeof(float);
        result.resize(n);
        std::memcpy(result.data(), bytes.data(), n * sizeof(float));
    } catch (...) {
        std::filesystem::remove(in);
        std::filesystem::remove(out);
        throw;
    }
    std::filesystem::remove(in);
    std::filesystem::remove(out);
    return result;
}

ASRModel::ASRModel(const std::string& id,
                   const std::filesystem::path& models_path,
                   const std::string& device)
    : m_id(id), m_models_path(models_path), m_device(device) {
    std::cerr << "asr load: warm start" << std::endl;
    m_pipeline = std::make_shared<ov::genai::ASRPipeline>(m_models_path.string(),
                                                          m_device);
    std::cerr << "asr load: done" << std::endl;
}

ASRModel::~ASRModel() = default;

ASRResult ASRModel::generate(const ASRGenerateOptions& opts,
                             ASROnText on_text) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ov::genai::ASRGenerationConfig cfg;
    cfg.language = opts.language;
    cfg.task = opts.task;
    cfg.initial_prompt = opts.initial_prompt;
    cfg.context = opts.context;
    cfg.return_timestamps = opts.return_timestamps;
    if (opts.temperature)
        cfg.temperature = *opts.temperature;
    if (opts.max_new_tokens)
        cfg.max_new_tokens = *opts.max_new_tokens;

    std::cerr << "asr start: samples=" << opts.samples.size() << std::endl;

    const auto start = std::chrono::steady_clock::now();
    const ov::genai::AudioInputs input{opts.samples};
    ov::genai::ASRDecodedResults result;
    if (on_text) {
        result = m_pipeline->generate(input, cfg, on_text);
    } else {
        result = m_pipeline->generate(input, cfg);
    }

    ASRResult out;
    if (!result.texts.empty())
        out.text = result.texts[0];
    if (!result.languages.empty())
        out.language = result.languages[0];
    if (result.chunks && !result.chunks->empty()) {
        for (const auto& chunk : (*result.chunks)[0]) {
            ASRSegment seg;
            seg.start = chunk.start_ts;
            seg.end = chunk.end_ts;
            seg.text = chunk.text;
            out.segments.push_back(std::move(seg));
        }
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();
    std::cerr << "asr done: ms=" << elapsed << std::endl;
    return out;
}

TTSModel::TTSModel(const std::string& id,
                   const std::filesystem::path& models_path,
                   const std::string& device)
    : m_id(id), m_models_path(models_path), m_device(device) {
    std::cerr << "tts load: warm start" << std::endl;
    m_pipeline =
        std::make_shared<ov::genai::Text2SpeechPipeline>(m_models_path.string(),
                                                         m_device);
    std::cerr << "tts load: done" << std::endl;
}

TTSModel::~TTSModel() = default;

TTSResult TTSModel::generate(const std::string& text) {
    std::lock_guard<std::mutex> lock(m_mutex);
    std::cerr << "tts start: text=" << text << std::endl;
    const ov::genai::Text2SpeechDecodedResults result =
        m_pipeline->generate(std::vector<std::string>{text});
    if (result.speeches.empty())
        throw std::runtime_error("tts: no speech produced");
    const ov::Tensor& speech = result.speeches.front();
    if (speech.get_element_type() != ov::element::f32)
        throw std::runtime_error("tts: unexpected sample element type " +
                                 speech.get_element_type().get_type_name());
    const float* data = speech.data<float>();
    const std::size_t count = static_cast<std::size_t>(speech.get_size());
    TTSResult out;
    out.sample_rate = result.output_sample_rate;
    out.samples.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float v = data[i];
        const float c = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
        out.samples.push_back(
            static_cast<std::int16_t>(c * 32767.0f));
    }
    std::cerr << "tts done: samples=" << count << " rate=" << out.sample_rate
          << std::endl;
    return out;
}

}  // namespace ovserver
