// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Plain single-axis RoPE tables for the exported talker / code_predictor
// `cos`/`sin` inputs ([1,1,L,128] each).
//
// The reference notebook drives these with a multimodal rotary
// (apply_multimodal_rotary_pos_emb) over three position axes. For audio-only
// prompts get_rope_index() produces the SAME position array on every axis, and
// with the interleave convention this collapses to the standard Qwen2
// rotate_half layout:
//   inv_freq[d] = theta^{-2d/head_dim} for d in [0, head_dim/2)
//   table[t, d] = cos(inv_freq[d] * pos_t) for d < half, mirrored to d >= half
//   (emb = cat(freqs, freqs, dim=-1); attention_scaling = 1.0 at default
//    rope_scaling). The IR routes these tables straight into per-head
//    q*cos + rotate_half(q)*sin multiplies with no in-graph interleaving.

#include "ovserver/qwen3_tts.hpp"

#include <cmath>
#include <stdexcept>

namespace ovserver {

void compute_rope_tables(std::int64_t length, std::int64_t head_dim,
                         double theta, double dt, bool /*mrc_repeat*/,
                         std::vector<float>& cos, std::vector<float>& sin) {
    if (length < 0 || head_dim <= 0 || head_dim % 2 != 0)
        throw std::invalid_argument("compute_rope_tables: bad dims");
    const std::int64_t half = head_dim / 2;
    const std::size_t n = static_cast<std::size_t>(length) *
                          static_cast<std::size_t>(head_dim);
    cos.assign(n, 0.0f);
    sin.assign(n, 0.0f);

    std::vector<double> inv_freq(static_cast<std::size_t>(half));
    for (std::int64_t d = 0; d < half; ++d)
        inv_freq[static_cast<std::size_t>(d)] =
            d == 0 ? 1.0 : std::pow(theta, -2.0 * static_cast<double>(d) /
                                                  static_cast<double>(head_dim));

    const std::int64_t dpos = static_cast<std::int64_t>(dt);
    for (std::int64_t t = 0; t < length; ++t) {
        const double pos = static_cast<double>(t + dpos);
        const std::size_t base = static_cast<std::size_t>(t) *
                                 static_cast<std::size_t>(head_dim);
        for (std::int64_t d = 0; d < half; ++d) {
            const double angle = inv_freq[static_cast<std::size_t>(d)] * pos;
            const float c = static_cast<float>(std::cos(angle));
            const float s = static_cast<float>(std::sin(angle));
            cos[base + static_cast<std::size_t>(d)] = c;
            cos[base + static_cast<std::size_t>(half + d)] = c;
            sin[base + static_cast<std::size_t>(d)] = s;
            sin[base + static_cast<std::size_t>(half + d)] = s;
        }
    }
}

}  // namespace ovserver