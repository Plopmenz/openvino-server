// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ovserver {

std::string base64_encode(const std::uint8_t* data, std::size_t len);

// Decodes a standard (RFC 4648) base64 string into raw bytes. Throws
// std::runtime_error on a non-canonical input.
std::vector<std::uint8_t> base64_decode(const std::string& input);

}  // namespace ovserver