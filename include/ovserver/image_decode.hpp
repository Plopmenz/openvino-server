// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include <openvino/runtime/tensor.hpp>

namespace ovserver {

// Decodes a base64-encoded image (optionally a full data: URL of the form
// "data:image/png;base64,....") into an RGB uint8 tensor shaped [H,W,3].
// The payload is decoded from base64 and loaded with stb_image, which supports
// PNG/JPEG/BMP/GIF/WebP/ etc.; any alpha channel is composited on white.
ov::Tensor decode_image_base64(const std::string& data_url_or_b64);

}  // namespace ovserver
