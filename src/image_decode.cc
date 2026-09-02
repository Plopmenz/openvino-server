// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/image_decode.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "ovserver/base64.hpp"

namespace ovserver {

namespace {

// Strips a "data:image/<fmt>;base64," prefix if present, returning the raw
// base64 payload.
std::string strip_data_url(const std::string& input) {
    const std::string prefix = "base64,";
    const auto pos = input.find(prefix);
    if (pos == std::string::npos) {
        return input;  // assume the whole string is already base64
    }
    return input.substr(pos + prefix.size());
}

}  // namespace

ov::Tensor decode_image_base64(const std::string& data_url_or_b64) {
    const std::string payload = strip_data_url(data_url_or_b64);
    if (payload.empty()) {
        throw std::runtime_error("empty image payload");
    }
    const std::vector<std::uint8_t> bytes = base64_decode(payload);
    if (bytes.empty()) {
        throw std::runtime_error("empty decoded image bytes");
    }

    int w = 0, h = 0, channels = 0;
    stbi_uc* data = stbi_load_from_memory(
        bytes.data(), static_cast<int>(bytes.size()), &w, &h, &channels,
        STBI_rgb);  // force 3 channels
    if (!data) {
        throw std::runtime_error(std::string("failed to decode image: ") +
                                 (stbi_failure_reason() ? stbi_failure_reason()
                                                        : "unknown error"));
    }
    if (w <= 0 || h <= 0) {
        stbi_image_free(data);
        throw std::runtime_error("invalid image dimensions");
    }

    ov::Tensor tensor(ov::element::u8, ov::Shape{static_cast<size_t>(h),
                                                 static_cast<size_t>(w), 3});
    std::memcpy(tensor.data<uint8_t>(), data,
                static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    stbi_image_free(data);
    return tensor;
}

}  // namespace ovserver
