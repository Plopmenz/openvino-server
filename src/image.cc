// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/image.hpp"
#include "ovserver/base64.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace ovserver {

std::string png_base64(const std::uint8_t* data,
                       std::size_t height,
                       std::size_t width,
                       std::size_t channels) {
    if (!data || height == 0 || width == 0 || (channels != 3 && channels != 4)) {
        throw std::runtime_error("invalid image dimensions for PNG encoding");
    }

    // stbi_write_png writes stride = width * channels bytes per row.
    int stride = static_cast<int>(width * channels);

    int len = 0;
    // stbi_write_png_to_mem returns a malloc'd buffer that must be freed with
    // free(). Returns non-null on success.
    unsigned char* raw = stbi_write_png_to_mem(
        reinterpret_cast<const unsigned char*>(data), stride,
        static_cast<int>(width), static_cast<int>(height),
        static_cast<int>(channels), &len);

    if (!raw || len <= 0) {
        if (raw) std::free(raw);
        throw std::runtime_error("stb_image_write PNG encoding failed");
    }

    std::string b64 = base64_encode(raw, static_cast<std::size_t>(len));
    std::free(raw);
    return b64;
}

bool png_write(const std::string& path,
               const std::uint8_t* data,
               std::size_t height,
               std::size_t width,
               std::size_t channels) {
    if (!data || height == 0 || width == 0 || (channels != 3 && channels != 4)) {
        return false;
    }
    return stbi_write_png(path.c_str(), static_cast<int>(width),
                          static_cast<int>(height),
                          static_cast<int>(channels), data,
                          static_cast<int>(width * channels)) == 1;
}

}  // namespace ovserver