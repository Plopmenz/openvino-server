// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3-TTS (12 Hz base, INT8 OpenVINO) direct inference engine. Drives the
// exported IRs (talker, code_predictor, text_model, codec_embedding,
// cp_codec_embedding, speech_decoder) with ov::Core directly, porting the
// reference notebook's _generate_talker_codes / _chunked_ov_decode pipeline:
//   1. Prompt building: BPE text + role/tail ids -> text_model projections,
//      codec prefill embeddings (language "Auto") and codec pad/bos rows are
//      blended per non-streaming mode into the talker prefill inputs_embeds.
//   2. Autoregressive talker loop: each frame samples codec token 0 (talker
//      logits over the 3072 codec vocab); the code predictor is run fresh per
//      frame to sample residual books 1..15; the aggregate codec embedding is
//      fed back (+tts_pad) to continue the talker stateful cache.
//   3. Chunked speech-codec decoding into 24 kHz PCM.
//
// All I/O names/shapes were verified against the on-disk .xml files.

#include "ovserver/qwen3_tts.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

#include <json/json.h>

namespace ovserver {

namespace {

constexpr int kHeadDim = 128;
constexpr double kTheta = 1e6;
constexpr int kTalkerDim = 2048;
constexpr int kTalkerVocab = 3072;
constexpr int kCodecVocab = 2048;
constexpr int kNumCodeGroups = 16;

constexpr std::int64_t kTtsBos = 151672;
constexpr std::int64_t kTtsEos = 151673;
constexpr std::int64_t kTtsPad = 151671;
constexpr std::int64_t kImStart = 151644;
constexpr std::int64_t kAssistant = 77091;
constexpr std::int64_t kNewline = 198;
constexpr std::int64_t kCodecPad = 2148;
constexpr std::int64_t kCodecBos = 2149;
constexpr std::int64_t kCodecEos = 2150;
constexpr std::int64_t kCodecNothink = 2155;
constexpr std::int64_t kCodecThinkBos = 2156;
constexpr std::int64_t kCodecThinkEos = 2157;

constexpr int kDecoderTraceLen = 325;
constexpr int kDecoderChunk = 300;
constexpr int kDecoderLeftCtx = 25;
constexpr int kDecoderUpsample = 1920;
constexpr int kDecoderOffset = 555;
constexpr std::uint32_t kSampleRate = 24000;

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

}  // namespace

struct Qwen3TTSModel::Impl {
    ov::Core core;
    std::string device;

    Gpt2BPEEncoder bpe;

    ov::InferRequest r_text;
    ov::InferRequest r_codec;
    ov::InferRequest r_cp_codec;
    ov::InferRequest r_talker;
    ov::InferRequest r_cp;
    ov::InferRequest r_decoder;

    // Sampling parameters (mirror generation_config.json).
    bool do_sample = true;
    double temperature = 0.9;
    int top_k = 50;
    double top_p = 1.0;
    double repetition_penalty = 1.05;
    bool subtalker_dosample = true;
    double subtalker_temperature = 0.9;
    int subtalker_top_k = 50;
    double subtalker_top_p = 1.0;
    int max_new_tokens = 8192;

    std::mutex mtx;
};

namespace {

std::vector<float> copy_f32(const ov::Tensor& t) {
    const std::size_t n = ov::shape_size(t.get_shape());
    std::vector<float> out(n);
    if (n) std::memcpy(out.data(), t.data(), n * sizeof(float));
    return out;
}

void set_ids(ov::InferRequest& req, const std::string& name,
             const std::vector<std::int64_t>& ids) {
    ov::Tensor t(ov::element::i64, {1, ids.size()});
    std::memcpy(t.data(), ids.data(), ids.size() * sizeof(std::int64_t));
    req.set_tensor(name, t);
}

// text_model: token_ids -> projected (fused text embedding + projection).
std::vector<float> text_embed(ov::InferRequest& r,
                              const std::vector<std::int64_t>& ids) {
    set_ids(r, "token_ids", ids);
    r.infer();
    return copy_f32(r.get_tensor("projected"));
}

// codec_embedding: token_ids -> embeddings.
std::vector<float> codec_embed(ov::InferRequest& r,
                               const std::vector<std::int64_t>& ids) {
    set_ids(r, "token_ids", ids);
    r.infer();
    return copy_f32(r.get_tensor("embeddings"));
}

// cp_codec_embedding: token_ids + step_idx -> embeddings.
std::vector<float> cp_codec_embed(ov::InferRequest& r,
                                  const std::vector<std::int64_t>& ids,
                                  std::int64_t step) {
    set_ids(r, "token_ids", ids);
    ov::Tensor st(ov::element::i64, {});
    *static_cast<std::int64_t*>(st.data()) = step;
    r.set_tensor("step_idx", st);
    r.infer();
    return copy_f32(r.get_tensor("embeddings"));
}

void set_cos_sin(ov::InferRequest& req, int length, std::int64_t dt) {
    std::vector<float> cos, sin;
    compute_rope_tables(length, kHeadDim, kTheta, static_cast<double>(dt),
                        false, cos, sin);
    const ov::Shape shape{1, 1, static_cast<std::size_t>(length),
                          static_cast<std::size_t>(kHeadDim)};
    ov::Tensor tc(ov::element::f32, shape);
    std::memcpy(tc.data(), cos.data(), cos.size() * sizeof(float));
    req.set_tensor("cos", tc);
    ov::Tensor ts(ov::element::f32, shape);
    std::memcpy(ts.data(), sin.data(), sin.size() * sizeof(float));
    req.set_tensor("sin", ts);
}

void set_beam(ov::InferRequest& req) {
    ov::Tensor t(ov::element::i32, {1});
    *static_cast<std::int32_t*>(t.data()) = 0;
    req.set_tensor("beam_idx", t);
}

std::int64_t argmax(const std::vector<float>& v) {
    return static_cast<std::int64_t>(
        std::max_element(v.begin(), v.end()) - v.begin());
}

// HF-style sampling: repetition penalty -> suppress mask -> temperature ->
// top-k -> softmax -> top-p -> multinomial.
std::int64_t sample_token(const std::vector<float>& logits_in, bool do_sample,
                          double temperature, int top_k, double top_p,
                          double repetition_penalty,
                          const std::vector<std::int64_t>* history,
                          int suppress_lo, int suppress_hi, int suppress_except,
                          std::mt19937& rng) {
    std::vector<float> logits = logits_in;
    const std::size_t n = logits.size();
    if (n == 0) return 0;

    if (history && !history->empty() && repetition_penalty > 0.0 &&
        repetition_penalty != 1.0) {
        std::vector<bool> seen(n, false);
        for (const std::int64_t id : *history) {
            const std::size_t i = static_cast<std::size_t>(id);
            if (i < n && !seen[i]) {
                seen[i] = true;
                float& v = logits[i];
                v = v > 0.0f ? v / static_cast<float>(repetition_penalty)
                             : v * static_cast<float>(repetition_penalty);
            }
        }
    }

    if (suppress_hi > suppress_lo)
        for (int v = suppress_lo; v < suppress_hi; ++v)
            if (v != suppress_except) logits[static_cast<std::size_t>(v)] = kNegInf;

    if (temperature > 0.0 && temperature != 1.0)
        for (float& v : logits) v /= static_cast<float>(temperature);

    if (!do_sample) return argmax(logits);

    if (top_k > 0 && static_cast<std::size_t>(top_k) < n) {
        std::vector<float> sorted = logits;
        std::nth_element(sorted.begin(), sorted.begin() + top_k, sorted.end(),
                         std::greater<float>());
        const float threshold = sorted[static_cast<std::size_t>(top_k)];
        for (float& v : logits)
            if (v < threshold) v = kNegInf;
    }

    const float mx = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(n);
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        probs[i] = std::exp(logits[i] - mx);
        sum += probs[i];
    }
    if (!(sum > 0.0f)) return argmax(logits_in);

    for (float& p : probs) p /= sum;

    if (top_p > 0.0 && top_p < 1.0) {
        std::vector<std::size_t> order(n);
        for (std::size_t i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&probs](std::size_t a, std::size_t b) {
            return probs[a] > probs[b];
        });
        float cum = 0.0f;
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t idx = order[k];
            if (cum > static_cast<float>(top_p)) {
                probs[idx] = 0.0f;
            } else {
                cum += probs[idx];
            }
        }
        float total = 0.0f;
        for (const float p : probs) total += p;
        if (total > 0.0f)
            for (float& p : probs) p /= total;
        else
            return argmax(logits_in);
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float r = dist(rng);
    float c = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        c += probs[i];
        if (r <= c) return static_cast<std::int64_t>(i);
    }
    return argmax(probs);
}

// Runs the code predictor for one talker frame: prefill with the aggregate
// talker hidden row + codec token, then 14 residual decode steps -> 15 books.
std::vector<std::int64_t> cp_books(ov::InferRequest& r_cp,
                                   ov::InferRequest& r_codec,
                                   ov::InferRequest& r_cp_codec,
                                   bool do_sample, double temperature,
                                   int top_k, double top_p, std::mt19937& rng,
                                   const std::vector<float>& h_row,
                                   std::int64_t token) {
    r_cp.reset_state();

    std::vector<float> pre(2 * kTalkerDim);
    std::memcpy(pre.data(), h_row.data(), kTalkerDim * sizeof(float));
    const std::vector<float> token_emb = codec_embed(r_codec, {token});
    std::memcpy(pre.data() + kTalkerDim, token_emb.data(),
                kTalkerDim * sizeof(float));

    ov::Tensor te(ov::element::f32, {1, 2, static_cast<std::size_t>(kTalkerDim)});
    std::memcpy(te.data(), pre.data(), pre.size() * sizeof(float));
    r_cp.set_tensor("inputs_embeds", te);

    set_cos_sin(r_cp, 2, 0);
    set_beam(r_cp);
    ov::Tensor gs(ov::element::i64, {});
    *static_cast<std::int64_t*>(gs.data()) = 0;
    r_cp.set_tensor("generation_steps", gs);

    r_cp.infer();
    std::vector<float> logits = copy_f32(r_cp.get_tensor("logits"));

    std::vector<std::int64_t> books;
    books.reserve(kNumCodeGroups - 1);
    // Prefill sampled next token from the last row of the 2-row run.
    books.push_back(sample_token(
        std::vector<float>(logits.end() - kCodecVocab, logits.end()),
        do_sample, temperature, top_k, top_p, 1.0, nullptr, 0, 0, 0, rng));

    for (int s = 1; s < kNumCodeGroups - 1; ++s) {
        const std::vector<float> emb =
            cp_codec_embed(r_cp_codec, {books.back()}, s - 1);
        ov::Tensor et(ov::element::f32,
                      {1, 1, static_cast<std::size_t>(kTalkerDim)});
        std::memcpy(et.data(), emb.data(), kTalkerDim * sizeof(float));
        r_cp.set_tensor("inputs_embeds", et);
        set_cos_sin(r_cp, 1, 1 + s);
        *static_cast<std::int64_t*>(gs.data()) = s;
        r_cp.set_tensor("generation_steps", gs);
        r_cp.infer();
        std::vector<float> step_logits = copy_f32(r_cp.get_tensor("logits"));
        books.push_back(sample_token(
            step_logits, do_sample, temperature, top_k, top_p, 1.0, nullptr,
            0, 0, 0, rng));
    }
    return books;
}

}  // namespace

Qwen3TTSModel::Qwen3TTSModel(const std::string& id,
                             const std::filesystem::path& models_path,
                             const std::string& device,
                             const std::string& cache_dir)
    : m_id(id),
      m_models_path(models_path),
      m_device(device),
      m_cache_dir(cache_dir),
      m_impl(new Impl) {
    Impl& p = *m_impl;
    p.device = device;

    if (!std::filesystem::is_directory(m_models_path))
        throw std::runtime_error("qwen3_tts: models path is not a directory: " +
                                 m_models_path.string());

    try {
        const auto gc_path = m_models_path / "generation_config.json";
        if (std::filesystem::exists(gc_path)) {
            std::ifstream fin(gc_path);
            Json::Value j;
            fin >> j;
            p.do_sample = j.get("do_sample", true).asBool();
            p.temperature = j.get("temperature", 0.9).asDouble();
            p.top_k = j.get("top_k", 50).asInt();
            p.top_p = j.get("top_p", 1.0).asDouble();
            p.repetition_penalty = j.get("repetition_penalty", 1.05).asDouble();
            p.subtalker_dosample = j.get("subtalker_dosample", true).asBool();
            p.subtalker_temperature = j.get("subtalker_temperature", 0.9).asDouble();
            p.subtalker_top_k = j.get("subtalker_top_k", 50).asInt();
            p.subtalker_top_p = j.get("subtalker_top_p", 1.0).asDouble();
            p.max_new_tokens = j.get("max_new_tokens", 8192).asInt();
        }
    } catch (...) {
        // fall back to defaults
    }

    const auto vocab = m_models_path / "vocab.json";
    const auto merges = m_models_path / "merges.txt";
    if (!p.bpe.load(vocab, merges))
        throw std::runtime_error("qwen3_tts: failed to load tokenizer (" +
                                 vocab.string() + ", " + merges.string() + ")");

    p.core = ov::Core();
    if (!m_cache_dir.empty())
        p.core.set_property(ov::cache_dir(m_cache_dir));

    auto cm = [&p](const std::filesystem::path& xml) {
        return p.core.compile_model(xml, p.device);
    };

    ov::CompiledModel text = cm(m_models_path / "text_model.xml");
    ov::CompiledModel codec = cm(m_models_path / "codec_embedding.xml");
    ov::CompiledModel cpc = cm(m_models_path / "cp_codec_embedding.xml");
    ov::CompiledModel talker = cm(m_models_path / "talker.xml");
    ov::CompiledModel cpred = cm(m_models_path / "code_predictor.xml");
    ov::CompiledModel dec =
        cm(m_models_path / "speech_tokenizer" / "speech_decoder.xml");

    p.r_text = text.create_infer_request();
    p.r_codec = codec.create_infer_request();
    p.r_cp_codec = cpc.create_infer_request();
    p.r_talker = talker.create_infer_request();
    p.r_cp = cpred.create_infer_request();
    p.r_decoder = dec.create_infer_request();
}

Qwen3TTSModel::~Qwen3TTSModel() = default;

TTSResult Qwen3TTSModel::generate(const std::string& text,
                                  const ov::Tensor& speaker_embedding) {
    std::lock_guard<std::mutex> lock(m_impl->mtx);
    Impl& p = *m_impl;

    TTSResult result;
    result.sample_rate = kSampleRate;

    if (text.empty())
        throw std::invalid_argument("qwen3_tts: empty input text");

    std::vector<float> spk_row;
    bool has_speaker = false;
    if (speaker_embedding && ov::shape_size(speaker_embedding.get_shape()) ==
                                 static_cast<std::size_t>(kTalkerDim)) {
        if (speaker_embedding.get_element_type() == ov::element::f32) {
            spk_row = copy_f32(speaker_embedding);
            has_speaker = true;
        } else {
            throw std::invalid_argument(
                "qwen3_tts: speaker embedding must be float32 with 2048 elements");
        }
    } else if (speaker_embedding && ov::shape_size(speaker_embedding.get_shape()) != 0) {
        throw std::invalid_argument(
            "qwen3_tts: speaker embedding must hold 2048 float32 elements");
    }

    // ---- Tokenize ----
    const auto tok = p.bpe.encode(text);
    if (tok.ids.empty())
        throw std::invalid_argument("qwen3_tts: text produced no tokens");
    const int T = static_cast<int>(tok.ids.size());

    // ---- Embeddings ----
    const std::vector<float> special =
        text_embed(p.r_text, {kTtsBos, kTtsEos, kTtsPad});        // bos,eos,pad
    const std::vector<float> role = text_embed(p.r_text, {kImStart, kAssistant, kNewline});
    const std::vector<float> content = text_embed(p.r_text, tok.ids);  // T rows

    auto row = [](const std::vector<float>& v, int i) -> const float* {
        return v.data() + static_cast<std::size_t>(i) * kTalkerDim;
    };
    auto wrow = [](std::vector<float>& v, int i) -> float* {
        return v.data() + static_cast<std::size_t>(i) * kTalkerDim;
    };
    auto rowadd = [](std::vector<float>& dst, int i, const float* src) {
        float* d = dst.data() + static_cast<std::size_t>(i) * kTalkerDim;
        for (int j = 0; j < kTalkerDim; ++j) d[j] += src[j];
    };

    // text_embed_with_eos (T+1 rows).
    std::vector<float> text_eos((T + 1) * kTalkerDim);
    std::memcpy(text_eos.data(), content.data(),
                content.size() * sizeof(float));
    std::memcpy(text_eos.data() + static_cast<std::size_t>(T) * kTalkerDim,
                row(special, 1), kTalkerDim * sizeof(float));

    // Codec prefill (language "Auto"): nothink/think_bos/think_eos.
    const std::vector<float> emb0 =
        codec_embed(p.r_codec, {kCodecNothink, kCodecThinkBos, kCodecThinkEos});
    const std::vector<float> emb1 = codec_embed(p.r_codec, {kCodecPad, kCodecBos});
    const std::vector<float> pad_row = codec_embed(p.r_codec, {kCodecPad});
    const std::vector<float> bos_row = codec_embed(p.r_codec, {kCodecBos});

    const int K = has_speaker ? 6 : 5;
    std::vector<float> codec_input(K * kTalkerDim);
    std::memcpy(codec_input.data(), emb0.data(), 3 * kTalkerDim * sizeof(float));
    std::size_t off = 3 * static_cast<std::size_t>(kTalkerDim);
    if (has_speaker) {
        std::memcpy(codec_input.data() + off, spk_row.data(),
                    kTalkerDim * sizeof(float));
        off += kTalkerDim;
    }
    std::memcpy(codec_input.data() + off, emb1.data(), 2 * kTalkerDim * sizeof(float));

    // ---- Prefill inputs_embeds ----
    // role (3) + [tts_pad x (K-2), tts_bos] + codec[:K-1] + [text+eos + codec_pad]
    // + [tts_pad + codec_bos].
    const int pre_rows = 3 + (K - 1);
    const int L = pre_rows + (T + 1) + 1;
    std::vector<float> pref(static_cast<std::size_t>(L) * kTalkerDim);

    // role rows.
    for (int i = 0; i < 3; ++i)
        std::memcpy(wrow(pref, i), row(role, i), kTalkerDim * sizeof(float));
    // tts_pad x (K-2) then tts_bos, each + codec_input row.
    for (int i = 0; i < K - 1; ++i) {
        const float* tts = (i < K - 2) ? row(special, 2) : row(special, 0);
        std::memcpy(wrow(pref, 3 + i), tts, kTalkerDim * sizeof(float));
        rowadd(pref, 3 + i, row(codec_input, i));
    }
    // text_embed_with_eos + codec_pad.
    for (int i = 0; i < T + 1; ++i) {
        std::memcpy(wrow(pref, pre_rows + i), row(text_eos, i),
                    kTalkerDim * sizeof(float));
        rowadd(pref, pre_rows + i, row(pad_row, 0));
    }
    // tts_pad + codec_bos.
    {
        std::memcpy(wrow(pref, L - 1), row(special, 2), kTalkerDim * sizeof(float));
        rowadd(pref, L - 1, row(bos_row, 0));
    }

    // ---- Talker prefill ----
    p.r_talker.reset_state();
    ov::Tensor te(ov::element::f32,
                  {1, static_cast<std::size_t>(L),
                   static_cast<std::size_t>(kTalkerDim)});
    std::memcpy(te.data(), pref.data(), pref.size() * sizeof(float));
    p.r_talker.set_tensor("inputs_embeds", te);
    set_cos_sin(p.r_talker, L, 0);
    set_beam(p.r_talker);

    p.r_talker.infer();
    std::vector<float> pre_logits = copy_f32(p.r_talker.get_tensor("logits"));
    std::vector<float> pre_hidden = copy_f32(p.r_talker.get_tensor("hidden"));

    std::mt19937 rng(std::random_device{}());
    std::vector<std::int64_t> history;
    std::int64_t token = sample_token(
        std::vector<float>(
            pre_logits.end() - kTalkerVocab, pre_logits.end()),
        p.do_sample, p.temperature, p.top_k, p.top_p, p.repetition_penalty,
        &history, kCodecVocab, kTalkerVocab, kCodecEos, rng);
    history.push_back(token);

    std::vector<float> h_prev(pre_hidden.end() - kTalkerDim, pre_hidden.end());
    std::vector<std::vector<std::int64_t>> frames;

    // ---- Autoregressive talker loop ----
    for (int step = 0; step < p.max_new_tokens; ++step) {
        if (token == kCodecEos) break;

        const std::vector<std::int64_t> books =
            cp_books(p.r_cp, p.r_codec, p.r_cp_codec, p.subtalker_dosample,
                     p.subtalker_temperature, p.subtalker_top_k,
                     p.subtalker_top_p, rng, h_prev, token);

        std::vector<std::int64_t> frame;
        frame.reserve(kNumCodeGroups);
        frame.push_back(token);
        frame.insert(frame.end(), books.begin(), books.end());
        frames.push_back(std::move(frame));

        // Aggregate codec embeddings + trailing text hidden (tts_pad).
        std::vector<float> agg(static_cast<std::size_t>(kTalkerDim), 0.0f);
        const std::vector<float> tok_emb = codec_embed(p.r_codec, {token});
        for (int d = 0; d < kTalkerDim; ++d) agg[static_cast<std::size_t>(d)] += tok_emb[static_cast<std::size_t>(d)];
        for (std::size_t i = 0; i < books.size(); ++i) {
            const std::vector<float> be =
                cp_codec_embed(p.r_cp_codec, {books[i]},
                               static_cast<std::int64_t>(i));
            for (int d = 0; d < kTalkerDim; ++d)
                agg[static_cast<std::size_t>(d)] += be[static_cast<std::size_t>(d)];
        }
        for (int d = 0; d < kTalkerDim; ++d)
            agg[static_cast<std::size_t>(d)] += row(special, 2)[d];

        // Decode step: single token at absolute position L + step.
        ov::Tensor dt(ov::element::f32,
                      {1, 1, static_cast<std::size_t>(kTalkerDim)});
        std::memcpy(dt.data(), agg.data(), agg.size() * sizeof(float));
        p.r_talker.set_tensor("inputs_embeds", dt);
        set_cos_sin(p.r_talker, 1, L + step);
        p.r_talker.infer();

        std::vector<float> step_logits = copy_f32(p.r_talker.get_tensor("logits"));
        std::vector<float> step_hidden = copy_f32(p.r_talker.get_tensor("hidden"));
        h_prev.assign(step_hidden.end() - kTalkerDim, step_hidden.end());

        token = sample_token(
            std::vector<float>(step_logits.end() - kTalkerVocab,
                               step_logits.end()),
            p.do_sample, p.temperature, p.top_k, p.top_p, p.repetition_penalty,
            &history, kCodecVocab, kTalkerVocab, kCodecEos, rng);
        history.push_back(token);
        if (token == kCodecEos) break;
    }

    // ---- Chunked speech-codec decode ----
    if (frames.empty()) return result;

    const std::size_t F = frames.size();
    std::vector<float> samples;
    std::size_t start = 0;
    while (start < F) {
        const std::size_t end = std::min(start + kDecoderChunk, F);
        const std::size_t ctx = start > static_cast<std::size_t>(kDecoderLeftCtx)
                                    ? static_cast<std::size_t>(kDecoderLeftCtx)
                                    : start;
        const std::size_t chunk_len = end - (start - ctx);
        const std::size_t rows = chunk_len < kDecoderTraceLen ? kDecoderTraceLen
                                                              : chunk_len;

        std::vector<std::int64_t> codes(static_cast<std::size_t>(kNumCodeGroups) * rows, 0);
        // codes[n] layout [num_groups][rows]; frames[r] is [16] group ids.
        for (std::size_t r = 0; r < chunk_len; ++r) {
            const auto& fm = frames[start - ctx + r];
            for (int g = 0; g < kNumCodeGroups; ++g)
                codes[static_cast<std::size_t>(g) * rows + r] = fm[static_cast<std::size_t>(g)];
        }

        ov::Tensor ct(ov::element::i64,
                      {1, static_cast<std::size_t>(kNumCodeGroups), rows});
        std::memcpy(ct.data(), codes.data(), codes.size() * sizeof(std::int64_t));
        p.r_decoder.set_tensor("codes", ct);
        p.r_decoder.infer();
        std::vector<float> wav = copy_f32(p.r_decoder.get_tensor("waveform"));

        const std::size_t total_valid = chunk_len * kDecoderUpsample - kDecoderOffset;
        const std::size_t ctx_samples = ctx * kDecoderUpsample;
        if (total_valid > ctx_samples && total_valid <= wav.size())
            samples.insert(samples.end(), wav.begin() +
                                              static_cast<std::ptrdiff_t>(ctx_samples),
                           wav.begin() + static_cast<std::ptrdiff_t>(total_valid));

        start = end;
    }

    result.samples.resize(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const long s = std::lround(samples[i] * 32767.0);
        result.samples[i] = static_cast<std::int16_t>(
            std::clamp(s, -32768L, 32767L));
    }
    return result;
}

bool is_qwen3_tts_layout(const std::filesystem::path& path) {
    std::filesystem::path dir = path;
    if (!std::filesystem::exists(path)) return false;
    if (!std::filesystem::is_directory(path)) dir = path.parent_path();
    return std::filesystem::exists(dir / "talker.xml") &&
           std::filesystem::exists(dir / "code_predictor.xml") &&
           std::filesystem::exists(dir / "codec_embedding.xml") &&
           std::filesystem::exists(dir / "cp_codec_embedding.xml") &&
           std::filesystem::exists(dir / "text_model.xml") &&
           std::filesystem::exists(dir / "speech_tokenizer" / "speech_decoder.xml");
}

}  // namespace ovserver