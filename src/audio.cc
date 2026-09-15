// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/audio.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>

#include <iostream>

#include "ovserver/common.hpp"

namespace ovserver {
namespace {

std::mutex g_ffmpeg_path_mutex;
std::string g_ffmpeg_path = "ffmpeg";

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

// Removes a temp file on destruction (or immediately on release).
class TempFileCleanup {
public:
    explicit TempFileCleanup(std::filesystem::path path)
        : m_path(std::move(path)) {}
    ~TempFileCleanup() {
        if (!m_path.empty())
            std::filesystem::remove(m_path);
    }
    TempFileCleanup(const TempFileCleanup&) = delete;
    TempFileCleanup& operator=(const TempFileCleanup&) = delete;

private:
    std::filesystem::path m_path;
};

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
    std::lock_guard<std::mutex> lock(g_ffmpeg_path_mutex);
    g_ffmpeg_path = path;
}

std::string ffmpeg_path() {
    std::lock_guard<std::mutex> lock(g_ffmpeg_path_mutex);
    return g_ffmpeg_path;
}

std::vector<float> decode_audio_to_f32(const std::string& data, int sample_rate) {
    if (data.empty())
        return {};
    const std::filesystem::path in = write_temp_file(data, ".in.audio");
    TempFileCleanup in_cleanup(in);
    const std::filesystem::path out = write_temp_file("", ".f32");
    TempFileCleanup out_cleanup(out);
    std::vector<float> result;
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
    return result;
}

}  // namespace ovserver
