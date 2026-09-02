// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/base64.hpp"

#include <stdexcept>

namespace ovserver {

namespace {

int b64_val(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

}  // namespace

std::vector<std::uint8_t> base64_decode(const std::string& input) {
    std::vector<std::uint8_t> out;
    out.reserve((input.size() / 4) * 3);

    int buf = 0;
    int bits = 0;
    bool at_end = false;
    for (std::size_t i = 0; i < input.size(); ++i) {
        const char c = input[i];
        if (c == '=') {
            at_end = true;
            continue;
        }
        if (at_end) {
            throw std::runtime_error("base64: data after padding '='");
        }
        const int v = b64_val(c);
        if (v < 0) {
            throw std::runtime_error("base64: invalid character");
        }
        buf = (buf << 6) | v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>((buf >> bits) & 0xFF));
        }
    }
    return out;
}

std::string base64_encode(const std::uint8_t* data, std::size_t len) {
    static const char kTable[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";

    std::string out;
    out.reserve(((len + 2) / 3) * 4);

    std::size_t i = 0;
    while (i + 2 < len) {
        std::uint32_t n = (static_cast<std::uint32_t>(data[i]) << 16) |
                          (static_cast<std::uint32_t>(data[i + 1]) << 8) |
                          static_cast<std::uint32_t>(data[i + 2]);
        out.push_back(kTable[(n >> 18) & 0x3F]);
        out.push_back(kTable[(n >> 12) & 0x3F]);
        out.push_back(kTable[(n >> 6) & 0x3F]);
        out.push_back(kTable[n & 0x3F]);
        i += 3;
    }

    if (i + 1 < len) {
        std::uint32_t n = (static_cast<std::uint32_t>(data[i]) << 16) |
                          (static_cast<std::uint32_t>(data[i + 1]) << 8);
        out.push_back(kTable[(n >> 18) & 0x3F]);
        out.push_back(kTable[(n >> 12) & 0x3F]);
        out.push_back(kTable[(n >> 6) & 0x3F]);
        out.push_back('=');
    } else if (i < len) {
        std::uint32_t n = static_cast<std::uint32_t>(data[i]) << 16;
        out.push_back(kTable[(n >> 18) & 0x3F]);
        out.push_back(kTable[(n >> 12) & 0x3F]);
        out.push_back('=');
        out.push_back('=');
    }
    return out;
}

}  // namespace ovserver