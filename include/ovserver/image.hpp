// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace ovserver {

// Renders an RGB(A) image in tensor form [H, W, C] (C in {3, 4}) into a PNG
// encoded in base64. Throws std::runtime_error if the input is invalid or PNG
// encoding fails.
std::string png_base64(const std::uint8_t* data,
                       std::size_t height,
                       std::size_t width,
                       std::size_t channels);

// Encodes an RGB(A) image tensor [H, W, C] to a PNG file. Returns false on
// failure.
bool png_write(const std::string& path,
               const std::uint8_t* data,
               std::size_t height,
               std::size_t width,
               std::size_t channels);

}  // namespace ovserver