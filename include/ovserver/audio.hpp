// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ovserver {

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