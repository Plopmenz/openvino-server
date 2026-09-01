// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ovserver {

std::string base64_encode(const std::uint8_t* data, std::size_t len);

}  // namespace ovserver