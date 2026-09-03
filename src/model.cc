// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <string>

#include <openvino/core/model.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/genai/image_generation/generation_config.hpp>
#include <openvino/genai/image_generation/image_generation_perf_metrics.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/runtime/core.hpp>

namespace {

ov::Core& shared_core() {
    static ov::Core core;
    return core;
}

// The GPU plugin cannot build a program for a dynamic tensor that has no upper
// bound ("get_tensor() is called for dynamic shape without upper bound"). The
// CPU plugin handles unbounded shapes natively, so only GPU runs need bounds.
// Give every dynamic dim a finite upper bound, never widening an existing one.
std::filesystem::path bound_shapes_copy(const std::filesystem::path& src,
                                        const std::string& id) {
    constexpr int64_t kMaxBound = 4096;
    auto dst = std::filesystem::temp_directory_path() /
               ("ovserver-bounded-" +
                (id.empty() ? std::string("model") : id));
    std::error_code ec;
    std::filesystem::remove_all(dst, ec);
    std::filesystem::copy(src, dst,
                          std::filesystem::copy_options::recursive, ec);
    if (ec) {
        throw std::runtime_error("cannot stage bounded model copy: " +
                                 ec.message());
    }

    auto bound_xml = [](const std::filesystem::path& xml) {
        std::shared_ptr<ov::Model> model = shared_core().read_model(xml);
        std::map<std::string, ov::PartialShape> new_shapes;
        for (const auto& input : model->inputs()) {
            ov::PartialShape ps = input.get_partial_shape();
            std::cerr << "[bounding]   " << xml.filename().parent_path() << "/"
                      << input.get_any_name() << " " << ps << std::endl;
            if (ps.rank().is_dynamic()) {
                continue;  // cannot bound individual dims of a dynamic rank
            }
            bool changed = false;
            for (size_t i = 0; i < ps.size(); ++i) {
                ov::Dimension& dim = ps[i];
                if (!dim.is_dynamic()) {
                    continue;
                }
                // get_max_length() < 0 means no upper bound. Never widen an
                // existing bound; only fill in a finite upper where missing.
                int64_t lo = std::max<int64_t>(1, dim.get_min_length());
                int64_t hi = dim.get_max_length();
                if (hi < 0 || hi > kMaxBound) {
                    hi = kMaxBound;
                }
                if (lo > hi) {
                    lo = hi;
                }
                dim = ov::Dimension(lo, hi);
                changed = true;
            }
            if (changed) {
                std::cerr << "[bounding]     -> " << ps << std::endl;
                new_shapes[input.get_any_name()] = ps;
            }
        }
        if (!new_shapes.empty()) {
            model->reshape(new_shapes);
            // Write to brand-new files then swap the XML in. Never overwrite
            // the stage's own .bin in place: it may still be memory-mapped from
            // the read_model() above, and truncating a mapped file turns the
            // next mapped access into SIGBUS. The bounded XML references the
            // *_bounded.bin, which stays; the stale original .bin is simply
            // left unused.
            const auto dir = xml.parent_path();
            const auto bounded_xml = dir / "openvino_model_bounded.xml";
            const auto bounded_bin = dir / "openvino_model_bounded.bin";
            ov::serialize(model, bounded_xml.string(), bounded_bin.string());
            std::error_code ec;
            std::filesystem::rename(bounded_xml, xml, ec);
            if (ec) {
                throw std::runtime_error("cannot swap bounded IR into place: " +
                                         ec.message());
            }
            if (!std::filesystem::exists(bounded_bin) ||
                std::filesystem::file_size(bounded_bin) == 0) {
                throw std::runtime_error("bounded rewrite produced an empty "
                                         "weights file for " + xml.string());
            }
            std::cerr << "[bounding] rewrote " << xml.string()
                      << " -> " << bounded_bin.filename().string()
                      << " (" << bounded_bin.string() << ")" << std::endl;
            // Sanity: re-read and confirm no input dim is still unbounded.
            auto verify = shared_core().read_model(xml);
            for (const auto& input : verify->inputs()) {
                ov::PartialShape ps = input.get_partial_shape();
                if (ps.rank().is_dynamic()) {
                    continue;
                }
                for (size_t i = 0; i < ps.size(); ++i) {
                    if (ps[i].is_dynamic() && ps[i].get_max_length() < 0) {
                        throw std::runtime_error(
                            "bounded rewrite left an unbounded dim on input '" +
                            input.get_any_name() + "' of " + xml.string());
                    }
                }
            }
        }
    };

    const auto root_xml = dst / "openvino_model.xml";
    if (std::filesystem::exists(root_xml)) {
        bound_xml(root_xml);
    }
    for (const auto& entry : std::filesystem::directory_iterator(dst)) {
        if (!entry.is_directory()) {
            continue;
        }
        const auto xml = entry.path() / "openvino_model.xml";
        if (std::filesystem::exists(xml)) {
            bound_xml(xml);
        }
    }
    std::cerr << "[bounding] staged bounded model copy at " << dst << std::endl;
    return dst;
}

// Wall-clock timestamp (H:M:S) for correlating server events with client-side
// curl errors, which share the same clock when curl runs on the host.
std::string wall_clock() {
    const auto now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%T", &tm);
    return buf;
}

ovserver::ImageResult extract_image(const ov::Tensor& result, std::size_t index) {
    auto shape = result.get_shape();
    ovserver::ImageResult img;
    img.height = static_cast<int>(shape[1]);
    img.width = static_cast<int>(shape[2]);
    img.channels = static_cast<int>(shape[3]);

    const auto* data = result.data<uint8_t>();
    const std::size_t plane =
        static_cast<std::size_t>(img.height * img.width * img.channels);
    const std::size_t offset = index * plane;
    img.data.assign(data + offset, data + offset + plane);
    return img;
}

}  // namespace

namespace ovserver {

Model::Model(const std::string& id,
             const std::filesystem::path& models_path,
             const std::string& device,
             const ov::AnyMap& properties,
             std::string text_encoder_device,
             std::string transformer_device,
             std::string vae_device,
             bool static_shapes,
             bool bound_dynamic,
             bool naive)
    : m_id(id), m_models_path(models_path), m_device(device),
      m_text_encoder_device(text_encoder_device.empty() ? device
                                                        : text_encoder_device),
      m_transformer_device(transformer_device.empty() ? device
                                                      : transformer_device),
      m_vae_device(vae_device.empty() ? device : vae_device),
      m_static_shapes(static_shapes), m_bound_dynamic(bound_dynamic),
      m_naive(naive), m_properties(properties) {
    if (m_naive) {
        // Naive mode mirrors the plain Python/optimum path exactly: the model
        // is handed to Text2ImagePipeline(path, device) unchanged and run
        // as-is on first use. No shape bounding, no reshape, no staged
        // compile, no keying.
        return;
    }
    if (m_bound_dynamic) {
        // Rewrite the exported IRs so the GPU plugin can build programs for
        // them; the copy keeps the original model dir untouched. Once the
        // dims are bounded, reshaping to static shapes would discard the
        // bounds, so static shaping is disabled alongside.
        m_models_path = bound_shapes_copy(models_path, id);
        m_static_shapes = false;
    }
    // Discover the defaults (height, width, guidance, steps, ...) a request
    // inherits when it does not override them, without compiling anything. The
    // un-compiled prototype is discarded here; models are loaded and compiled
    // lazily per requested shape in compiled_pipeline(), so an idle server
    // keeps no model in memory.
    ov::genai::Text2ImagePipeline prototype(models_path);
    m_default_config = prototype.get_generation_config();
}

ov::genai::Text2ImagePipeline Model::naive_pipeline() {
    std::lock_guard lock(m_mutex);
    if (!m_naive_pipeline) {
        std::cerr << "[model '" << m_id << "'] " << wall_clock()
                  << " naive: constructing Text2ImagePipeline(" << m_models_path
                  << ", " << m_device << ") - first request pays this one-time"
                  << " cost" << std::endl;
        m_naive_pipeline = std::make_shared<ov::genai::Text2ImagePipeline>(
            m_models_path, m_device);
    }
    // clone() shares the compiled models while giving each concurrent request
    // a private scheduler, so generation is safe in parallel.
    return m_naive_pipeline->clone();
}

ov::genai::Text2ImagePipeline Model::compiled_pipeline(
    const ov::genai::ImageGenerationConfig& cfg) {
    CompileKey key;
    key.num_images_bucket = cfg.num_images_per_prompt > 1 ? 2 : 1;
    key.guidance_bucket = cfg.guidance_scale > 1.0f ? 2 : 1;
    key.height = cfg.height;
    key.width = cfg.width;
    if (!m_static_shapes) {
        // Dynamic (bounded) shapes serve every size with a single compiled
        // program, so no per-shape key: collapse all requests onto one cached
        // pipeline. Compiling a fresh one per size is both pointless and
        // wasteful (each GPU compile is a multi-GB memory spike).
        key = CompileKey{1, 1, 0, 0};
    }

    // Shape/compile handling:
    //  * static_shapes (default): reshape to the request's static shapes and
    //    compile per shape, reusing recent shapes via the LRU cache.
    //  * !static_shapes (--no-reshape / --bound-dynamic): compile the dynamic
    //    IR once for all sizes; the key is collapsed to a single entry above.
    std::lock_guard lock(m_mutex);
    auto it = m_compiled.find(key);
    if (it != m_compiled.end()) {
        std::cerr << "[model '" << m_id << "'] cache hit key=" << key.height
                  << "x" << key.width << std::endl;
        m_lru.remove(key);
        m_lru.push_front(key);
    } else {
        std::cerr << "[model '" << m_id << "'] compiling key=" << key.height << "x"
                  << key.width << " ..." << std::endl;
        if (m_compiled.size() >= kMaxCompiledPipelines) {
            const CompileKey victim = m_lru.back();
            m_lru.pop_back();
            m_compiled.erase(victim);
        }

        ov::genai::Text2ImagePipeline pipe(m_models_path);
        if (m_static_shapes) {
            pipe.reshape(static_cast<int>(cfg.num_images_per_prompt),
                         static_cast<int>(cfg.height),
                         static_cast<int>(cfg.width),
                         cfg.guidance_scale);
        }
        pipe.compile(m_text_encoder_device, m_transformer_device,
                     m_vae_device, m_properties);

        auto [position, inserted] =
            m_compiled.emplace(key, std::make_shared<ov::genai::Text2ImagePipeline>(std::move(pipe)));
        it = position;
        m_lru.push_front(key);
        std::cerr << "[model '" << m_id << "'] compiled key=" << key.height
                  << "x" << key.width
                  << (m_static_shapes ? " (static per-shape)" 
                                      : " (dynamic, reused for all sizes)")
                  << " - first request pays this one-time cost" << std::endl;
    }

    // clone() shares the compiled models while giving each concurrent request a
    // private scheduler, so generation is safe in parallel. The shared_ptr
    // keeps the pipeline alive past LRU eviction while we clone.
    return it->second->clone();
}

std::vector<ImageResult> Model::generate(const GenerateOptions& opts) {
    ov::AnyMap properties;
    properties[ov::genai::num_images_per_prompt.name()] = opts.num_images;

    if (opts.negative_prompt) {
        properties[ov::genai::negative_prompt.name()] = *opts.negative_prompt;
    }
    if (opts.guidance_scale) {
        properties[ov::genai::guidance_scale.name()] = *opts.guidance_scale;
    }
    if (opts.height) {
        properties[ov::genai::height.name()] = *opts.height;
    }
    if (opts.width) {
        properties[ov::genai::width.name()] = *opts.width;
    }
    if (opts.num_inference_steps) {
        properties[ov::genai::num_inference_steps.name()] =
            *opts.num_inference_steps;
    }
    if (opts.rng_seed) {
        properties[ov::genai::rng_seed.name()] = *opts.rng_seed;
    }

    // Per-step progress logging. Invoked on the pipeline's callback thread
    // after every denoising step, so it shows whether/where generation sticks.
    const auto log_start = [&]() {
        std::cerr << "[model '" << m_id << "'] " << wall_clock()
                  << " generate start"
                  << ((opts.height || opts.width)
                          ? std::string(" size=") +
                                std::to_string(opts.height.value_or(-1)) + "x" +
                                std::to_string(opts.width.value_or(-1))
                          : std::string(" size=default"))
                  << " n=" << opts.num_images;
    };
    properties[ov::genai::callback.name()] =
        std::function<bool(size_t, size_t, ov::Tensor&)>(
            [m_id = m_id, gstart = std::chrono::steady_clock::now(),
             last = std::chrono::steady_clock::now()](
                size_t step, size_t total, ov::Tensor&) mutable -> bool {
                const auto now = std::chrono::steady_clock::now();
                const auto step_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - last)
                        .count();
                last = now;
                const auto cum_s =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                        now - gstart)
                        .count();
                std::cerr << "[model '" << m_id << "'] step " << (step + 1)
                          << "/" << total << " " << step_ms << " ms (cum "
                          << cum_s << " s)" << std::endl;
                return false;
            });

    ov::genai::Text2ImagePipeline pipe = [&]() -> ov::genai::Text2ImagePipeline {
        if (m_naive) {
            // Naive: run the raw pipeline with the request's overrides. Defaults
            // (height, width, guidance, steps) are genai's parsed model defaults,
            // exactly like the Python path. No reshape, no compile keys.
            log_start();
            std::cerr << " devs=" << m_device << " (naive)" << std::endl;
            return naive_pipeline();
        }
        // Apply the same config merge generate() will do, so the compile-time
        // static shapes and the runtime request parameters always agree.
        ov::genai::ImageGenerationConfig cfg = m_default_config;
        cfg.update_generation_config(properties);
        log_start();
        std::cerr << " cfg=" << cfg.height << "x" << cfg.width << " g="
                  << cfg.guidance_scale
                  << (m_static_shapes ? " static" : " dynamic") << " devs="
                  << m_text_encoder_device << "," << m_transformer_device << ","
                  << m_vae_device << std::endl;
        return compiled_pipeline(cfg);
    }();

    const auto t0 = std::chrono::steady_clock::now();
    ov::Tensor result = pipe.generate(opts.prompt, properties);
    const auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    std::cerr << "[model '" << m_id << "'] " << wall_clock() << " generate done in "
              << gen_ms << " ms" << std::endl;

    // Diagnostics: break the step time into transformer GPU infer vs the rest
    // (host scheduler / latent round-trip) so a 10x-slow-GPU bug can be
    // attributed to one or the other.
    try {
        ov::genai::ImageGenerationPerfMetrics pm = pipe.get_performance_metrics();
        const auto& trans = pm.raw_metrics.transformer_inference_durations;
        const auto& iters = pm.raw_metrics.iteration_durations;
        const std::size_t n = std::min(trans.size(), iters.size());
        std::cerr << "[model '" << m_id << "'] perf: load="
                  << static_cast<int>(pm.load_time) << "ms generate="
                  << static_cast<int>(pm.generate_duration) << "ms text_enc=";
        for (const auto& [k, v] : pm.encoder_inference_duration) {
            std::cerr << k << "=" << static_cast<int>(v) << "ms ";
        }
        std::cerr << "\n[model '" << m_id << "'] perf: vae_dec="
                  << static_cast<int>(pm.vae_decoder_inference_duration)
                  << "ms  steps=" << n << std::endl;
        if (n > 0) {
            const auto iths = [](const std::vector<ov::genai::MicroSeconds>& v) {
                double sum = 0;
                for (const auto& x : v) sum += static_cast<double>(x.count());
                return sum / static_cast<double>(v.size());
            };
            std::cerr << "[model '" << m_id << "'] perf: avg step="
                      << static_cast<int64_t>(iths(iters) / 1000.0)
                      << "ms  avg transformer-infer="
                      << static_cast<int64_t>(iths(trans) / 1000.0)
                      << "ms  (rest="
                      << static_cast<int64_t>((iths(iters) - iths(trans)) /
                                              1000.0)
                      << "ms)"
                      << "\n[model '" << m_id << "'] perf first steps (infer/total ms):";
            for (std::size_t i = 0; i < std::min<std::size_t>(n, 5); ++i) {
                std::cerr << " [" << i << "]="
                          << static_cast<int64_t>(trans[i].count() / 1000.f)
                          << "/"
                          << static_cast<int64_t>(iters[i].count() / 1000.f);
            }
            std::cerr << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[model '" << m_id
                  << "'] perf metrics unavailable: " << e.what() << std::endl;
    }

    // result shape: [N, H, W, C] u8
    auto shape = result.get_shape();
    const size_t num_images = shape[0];

    std::vector<ImageResult> images;
    images.reserve(num_images);
    for (size_t i = 0; i < num_images; ++i) {
        images.push_back(extract_image(result, i));
    }
    return images;
}

}  // namespace ovserver