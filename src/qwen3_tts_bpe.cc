// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Byte-level GPT-2 BPE port used for Qwen3-TTS prompt tokenization. This is a
// faithful C++ re-implementation of the standard GPT-2 tokenizer that ships
// with the model (vocab.json / merges.txt):
//   * bytes_to_unicode() maps every byte 0..255 through the exact GPT-2 byte
//     table (ASCII 33..126, 161..172, 174..255 kept as-is; the 66 remaining
//     bytes get U+0100..U+0141).
//   * encode() first splits the raw UTF-8 text with the HF GPT-2 regex
//     ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+),
//     byte-encodes each fragment, then runs the standard merge lattice over
//     pair ranks.
//
// The previous ad-hoc version had two correctness bugs: it emitted duplicate
// byte ranges (so the map was 280 entries with 26 duplicates and only 2
// "missing" chars -> out-of-bounds reads), and it tokenized ' '/'\n' directly
// as raw bytes (skipping the byte table), which silently dropped spaces and
// newlines. Both are fixed here.

#include "ovserver/qwen3_tts.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ovserver {

namespace {

// Encode a Unicode code point as UTF-8.
std::string utf8_encode(unsigned cp) {
    if (cp <= 0x7f) {
        return std::string(1, static_cast<char>(cp));
    }
    if (cp <= 0x7ff) {
        return std::string{
            static_cast<char>(0xc0 | (cp >> 6)),
            static_cast<char>(0x80 | (cp & 0x3f)),
        };
    }
    if (cp <= 0xffff) {
        return std::string{
            static_cast<char>(0xe0 | (cp >> 12)),
            static_cast<char>(0x80 | ((cp >> 6) & 0x3f)),
            static_cast<char>(0x80 | (cp & 0x3f)),
        };
    }
    return std::string{
        static_cast<char>(0xf0 | (cp >> 18)),
        static_cast<char>(0x80 | ((cp >> 12) & 0x3f)),
        static_cast<char>(0x80 | ((cp >> 6) & 0x3f)),
        static_cast<char>(0x80 | (cp & 0x3f)),
    };
}

// GPT-2 byte table: bytes in `bs` map to themselves; all other bytes map to
// cs[0..] = the first `count` unused code points starting at U+0100.
std::unordered_map<char, std::string> bytes_to_unicode() {
    std::vector<int> bs;
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);
    // byte 173 (soft hyphen) is the one gap inside 161..255, plus 0..32 and
    // 127..160.
    std::vector<bool> present(256, false);
    for (const int b : bs) present[static_cast<std::size_t>(b)] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b)
        if (!present[static_cast<std::size_t>(b)]) {
            bs.push_back(b);
            ++n;
        }
    std::unordered_map<char, std::string> map;
    map.reserve(bs.size());
    // Last `n` entries of bs are the appended bytes; they line up with code
    // points U+0100..U+0100+n-1.
    const int base = 256 - n;
    for (std::size_t i = 0; i < bs.size(); ++i)
        map[static_cast<char>(bs[i])] =
            i >= static_cast<std::size_t>(base)
                ? utf8_encode(0x100 + static_cast<int>(i - base))
                : utf8_encode(static_cast<unsigned>(bs[i]));
    return map;
}

// HF GPT-2 word-splitting regex, implemented as an ordered alternation over
// the raw (UTF-8) text.
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// `\p{L}` used here: ASCII letters or any byte >= 0x80 (a UTF-8 continuation
// lead, approximating unicode letters); `\p{N}`: ASCII digits; `\s`: ASCII
// whitespace.
struct Splitter {
    explicit Splitter(const std::string& text) : m_text(text) {}

    // Advance to the next alternator match at or after `pos`. Returns the
    // matched range [m_start, m_end) or {0,0} at end.
    void next(std::size_t& start, std::size_t& end) {
        const std::size_t n = m_text.size();
        while (m_pos < n) {
            std::size_t len = 0;
            if (len == 0) len = match_contraction(m_pos);        // 's etc.
            if (len == 0) len = match_word(m_pos, true);         // ?\p{L}+
            if (len == 0) len = match_word(m_pos, false);        // ?\p{N}+
            if (len == 0) len = match_symbol(m_pos);             // ?[^...]+
            if (len == 0) len = match_ws_run(m_pos);             // \s+(?!\S)|\s+
            if (len == 0) {                                        // never
                ++m_pos;
                continue;
            }
            start = m_pos;
            end = m_pos + len;
            m_pos = end;
            return;
        }
        start = end = 0;
    }

private:
    static bool is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
    static bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
    static bool is_space(unsigned char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    }
    static bool is_sym(unsigned char c) { return !is_space(c) && !is_letter(c) && !is_digit(c); }

    std::size_t match_literal(std::size_t pos, const char* lit) {
        std::size_t i = pos;
        for (const char* p = lit; *p; ++p, ++i)
            if (i >= m_text.size() || m_text[i] != *p) return 0;
        return i - pos;
    }

    std::size_t match_contraction(std::size_t pos) {
        static const char* kContr[] = {"'re", "'ve", "'ll", "'d", "'s", "'t", "'m"};
        for (const char* c : kContr) {
            const std::size_t len = match_literal(pos, c);
            if (len) return len;
        }
        return 0;
    }

    std::size_t match_word(std::size_t pos, bool letters) {
        std::size_t i = pos;
        if (i < m_text.size() && m_text[i] == ' ') ++i;         // optional ' '
        const std::size_t body = i;
        while (i < m_text.size() &&
               (letters ? is_letter(static_cast<unsigned char>(m_text[i]))
                        : is_digit(static_cast<unsigned char>(m_text[i]))))
            ++i;
        if (i == body) return 0;
        return i - pos;
    }

    std::size_t match_symbol(std::size_t pos) {
        std::size_t i = pos;
        if (i < m_text.size() && m_text[i] == ' ') ++i;
        const std::size_t body = i;
        while (i < m_text.size() &&
               is_sym(static_cast<unsigned char>(m_text[i])))
            ++i;
        if (i == body) return 0;
        return i - pos;
    }

    std::size_t match_ws_run(std::size_t pos) {
        std::size_t i = pos;
        while (i < m_text.size() && is_space(static_cast<unsigned char>(m_text[i])))
            ++i;
        return i == pos ? 0 : i - pos;
    }

    const std::string& m_text;
    std::size_t m_pos = 0;
};

// Split a UTF-8 string into its constituent characters (each byte-mapped
// unicode char is 1..4 bytes in UTF-8).
std::vector<std::string> split_utf8(const std::string& s) {
    std::vector<std::string> out;
    out.reserve(s.size());
    std::size_t i = 0;
    const std::size_t n = s.size();
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t len = 1;
        if ((c & 0xe0) == 0xc0) {
            len = 2;
        } else if ((c & 0xf0) == 0xe0) {
            len = 3;
        } else if ((c & 0xf8) == 0xf0) {
            len = 4;
        }
        out.emplace_back(s.substr(i, len));
        i += len;
    }
    return out;
}

}  // namespace

bool Gpt2BPEEncoder::load(const std::filesystem::path& vocab_path,
                          const std::filesystem::path& merges_path) {
    m_vocab.clear();
    m_merges.clear();

    {
        std::ifstream fin(vocab_path, std::ios::binary);
        if (!fin.is_open()) return false;
        const std::string js((std::istreambuf_iterator<char>(fin)),
                             std::istreambuf_iterator<char>());
        fin.close();
        std::size_t i = 0;
        if (js.size() >= 3 && static_cast<unsigned char>(js[0]) == 0xef &&
            static_cast<unsigned char>(js[1]) == 0xbb &&
            static_cast<unsigned char>(js[2]) == 0xbf)
            i = 3;  // strip UTF-8 BOM
        while (i < js.size()) {
            while (i < js.size() && (js[i] == ' ' || js[i] == '\t' || js[i] == '\n' ||
                                     js[i] == '\r' || js[i] == ',' || js[i] == '{'))
                ++i;
            if (i >= js.size() || js[i] == '}') break;
            if (js[i] != '"') {
                ++i;
                continue;
            }
            ++i;  // opening quote
            std::string token;
            while (i < js.size() && js[i] != '"') {
                if (js[i] == '\\' && i + 1 < js.size()) {
                    token += js[i + 1];  // unescape \x -> x
                    i += 2;
                } else {
                    token += js[i++];
                }
            }
            if (i >= js.size()) break;
            ++i;  // closing quote
            while (i < js.size() && js[i] != ':') ++i;
            ++i;
            while (i < js.size() && js[i] == ' ') ++i;
            long id = 0;
            bool neg = false;
            if (i < js.size() && js[i] == '-') {
                neg = true;
                ++i;
            }
            while (i < js.size() && js[i] >= '0' && js[i] <= '9') {
                id = id * 10 + (js[i] - '0');
                ++i;
            }
            if (neg) id = -id;
            m_vocab.emplace(token, static_cast<TokenT>(id));
        }
    }

    std::ifstream mfin(merges_path);
    if (!mfin.is_open()) return false;
    std::string line;
    bool first = true;
    while (std::getline(mfin, line)) {
        if (line.empty()) continue;
        if (first) {  // tolerate "#version: 0.2" headers
            first = false;
            if (line.rfind("#version", 0) == 0 || line.rfind("version", 0) == 0)
                continue;
        }
        const std::size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        m_merges.emplace_back(line.substr(0, sp), line.substr(sp + 1));
    }
    mfin.close();
    return !m_vocab.empty();
}

Gpt2BPEEncoder::Result Gpt2BPEEncoder::encode(const std::string& text) const {
    Result result;
    if (text.empty()) return result;
    if (m_vocab.empty())
        throw std::runtime_error("qwen3_tts: tokenizer not loaded (vocab empty)");

    const auto b2u = bytes_to_unicode();

    // Rank map for O(1) pair lookups (m_merges order = rank).
    std::unordered_map<std::string, std::int64_t> ranks;
    ranks.reserve(m_merges.size() * 2);
    {
        std::string key;
        key.reserve(32);
        for (std::size_t r = 0; r < m_merges.size(); ++r) {
            key.clear();
            key += m_merges[r].first;
            key.push_back(' ');
            key += m_merges[r].second;
            ranks.emplace(std::move(key), static_cast<std::int64_t>(r));
        }
    }
    auto rank_of = [&ranks](const std::pair<std::string, std::string>& pair)
        -> std::int64_t {
        std::string key = pair.first;
        key.push_back(' ');
        key += pair.second;
        const auto it = ranks.find(key);
        return it == ranks.end() ? -1 : it->second;
    };

    Splitter split(text);
    std::size_t start = 0, end = 0;
    while (true) {
        split.next(start, end);
        if (end == 0) break;

        // Byte-encode the fragment.
        std::string frag;
        frag.reserve(end - start);
        for (std::size_t i = start; i < end; ++i)
            frag += b2u.at(text[i]);

        // Merge lattice over the byte-encoded fragment.
        if (frag.size() >= 2) {
            std::vector<std::string> word = split_utf8(frag);
            bool changed = true;
            while (changed && word.size() > 1) {
                changed = false;
                if (word.size() >= 2) {
                    // Find best pair rank.
                    std::int64_t best_rank = -1;
                    std::size_t best_i = 0;
                    for (std::size_t i = 0; i + 1 < word.size(); ++i) {
                        const auto r = rank_of({word[i], word[i + 1]});
                        if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                            best_rank = r;
                            best_i = i;
                        }
                    }
                    if (best_rank >= 0) {
                        const std::string merged = word[best_i] + word[best_i + 1];
                        std::vector<std::string> next;
                        next.reserve(word.size() - 1);
                        for (std::size_t i = 0; i < word.size(); ++i) {
                            if (i == best_i) {
                                next.push_back(merged);
                                ++i;  // skip word[best_i+1]
                            } else {
                                next.push_back(word[i]);
                            }
                        }
                        word = std::move(next);
                        changed = true;
                    }
                }
            }
            for (const std::string& piece : word) {
                const auto it = m_vocab.find(piece);
                if (it == m_vocab.end())
                    throw std::runtime_error("qwen3_tts: BPE piece not in vocab");
                result.ids.push_back(static_cast<std::int64_t>(it->second));
                result.pieces.push_back(piece);
            }
        } else {
            const auto it = m_vocab.find(frag);
            if (it == m_vocab.end())
                throw std::runtime_error("qwen3_tts: BPE piece not in vocab");
            result.ids.push_back(static_cast<std::int64_t>(it->second));
            result.pieces.push_back(frag);
        }
    }
    return result;
}

}  // namespace ovserver