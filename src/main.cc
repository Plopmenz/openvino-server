// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include <drogon/drogon.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/runtime/core.hpp>

#include "ovserver/audio.hpp"
#include "ovserver/controller.hpp"
#include "ovserver/manager.hpp"

namespace {

void usage(const char* argv0) {
    std::cerr
        << "openvino-server: serve image generation (/v1/images/generations)\n"
        << "and text generation (/v1/chat/completions) models over an\n"
        << "OpenAI-compatible HTTP API.\n\n"
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "      --txt2img PATH      Path to an exported OpenVINO GenAI image\n"
        << "                          generation model (e.g. Qwen-Image). May be\n"
        << "                          repeated.\n"
        << "      --txt2img-id ID     Model id reported via /v1/models and the\n"
        << "                          'model' request field. Default: 'qwen-image'.\n"
        << "                          When multiple image models are given each\n"
        << "                          requires an id.\n"
        << "      --txt2txt PATH      Path to an exported OpenVINO GenAI text\n"
        << "                          generation model (e.g. Qwen2.5-VL,\n"
        << "                          Qwen3-VL). May be repeated.\n"
        << "      --txt2txt-id ID     Model id for the corresponding --txt2txt.\n"
        << "                          Default: the directory basename. When multiple\n"
        << "                          text models are given each requires an id.\n"
        << "      --kv-cache-precision TYPE\n"
        << "                          KV cache element type for text models on GPU.\n"
        << "                          Default: u8.\n"
        << "      --dynamic-quant-gsize N\n"
        << "                          Dynamic quantization group size for GPU text\n"
        << "                          inference. Default: 32.\n"
        << "      --enable-sdpa BOOL   Enable SDPA optimization for GPU text\n"
        << "                          inference. Default: true.\n"
        << "      --cache-interval-multiplier N\n"
        << "                          Linear-attention KV checkpoint interval, in\n"
        << "                          KV blocks. Ignored by models without linear\n"
        << "                          attention. Default: 64.\n"
        << "      --no-prefix-caching Disable KV-block prefix caching (on by\n"
        << "                          default). Caching retains previously computed\n"
        << "                          KV blocks for prompt-prefix reuse, improving\n"
        << "                          TTFT at the cost of VRAM.\n"
        << "      --prompt-lookup     Enable prompt-lookup speculative decoding.\n"
        << "                          Uses n-gram matching against the prompt to\n"
        << "                          draft candidate tokens. No extra model needed.\n"
        << "      --num-assistant-tokens N\n"
        << "                          Number of draft tokens proposed per\n"
        << "                          speculative-decoding step. Default: 5.\n"
        << "      --max-ngram-size N  Maximum n-gram size for prompt-lookup\n"
        << "                          matching. Default: 3.\n"
        << "      --no-mtp            Disable auto-detection of bundled MTP\n"
        << "                          (Multi-Token Prediction) heads. When a model\n"
        << "                          directory contains openvino_mtp_model.xml,\n"
        << "                          MTP speculative decoding is enabled\n"
        << "                          automatically.\n"
        << "      --txt2vid PATH      Path to an exported OpenVINO GenAI video\n"
        << "                          generation model (e.g. LTX-Video). May be\n"
        << "                          repeated.\n"
        << "      --txt2vid-id ID     Model id for the corresponding --txt2vid.\n"
        << "                          Default: the directory basename. When multiple\n"
        << "                          video models are given each requires an id.\n"
        << "      --wav2txt PATH      Path to an exported OpenVINO GenAI speech\n"
        << "                          recognition model (e.g. Qwen3-ASR, Whisper).\n"
        << "                          May be repeated.\n"
        << "      --wav2txt-id ID     Model id for the corresponding --wav2txt.\n"
        << "                          Default: the directory basename. When multiple\n"
        << "                          ASR models are given each requires an id.\n"
        << "      --txt2wav PATH      Path to an exported OpenVINO GenAI text-to-\n"
        << "                          speech model (e.g. SpeechT5, Kokoro). May be\n"
        << "                          repeated.\n"
        << "      --txt2wav-id ID     Model id for the corresponding --txt2wav.\n"
        << "                          Default: the directory basename. When multiple\n"
        << "                          TTS models are given each requires an id.\n"
        << "      --ffmpeg PATH       Path to the ffmpeg binary used to decode\n"
        << "                          uploaded audio and encode generated video.\n"
        << "                          Default: 'ffmpeg' on PATH. Set to an empty\n"
        << "                          string to require 16 kHz WAV uploads and skip\n"
        << "                          MP4 output.\n"
        << "  -d, --device DEVICE     OpenVINO device (CPU, GPU, AUTO, ...).\n"
        << "                          Default: CPU.\n"
        << "  -h, --host HOST         Listen address. Default: 0.0.0.0\n"
        << "  -p, --port PORT         Listen port. Default: 8080.\n"
        << "  -t, --threads N         Number of event-loop threads. Default: 4.\n"
        << "  -l, --log-level LEVEL   Verbosity: TRACE, DEBUG, INFO, WARN, ERROR.\n"
        << "                          Default: INFO.\n"
        << "  -c, --config FILE       Load drogon from a JSON config file, which\n"
        << "                          overrides -h/-p/-t/-l above.\n"
        << "      --idle-timeout SECONDS\n"
        << "                          Idle connection timeout. Generations can run\n"
        << "                          for minutes, so the 60s default closes the\n"
        << "                          socket before the response arrives (curl error\n"
        << "                          52). Default: 3600. Set 0 to never close.\n"
        << "  -v, --version           Print version and exit.\n"
        << "      --help              Show this help and exit.\n";
}

std::string get_arg(int argc, char** argv, int& i, const char* flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
    }
    return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> image_models;
    std::vector<std::string> image_model_ids;
    std::vector<std::string> text_models;
    std::vector<std::string> text_model_ids;
    std::vector<std::string> video_models;
    std::vector<std::string> video_model_ids;
    std::vector<std::string> asr_models;
    std::vector<std::string> asr_model_ids;
    std::vector<std::string> tts_models;
    std::vector<std::string> tts_model_ids;
    std::string device = "CPU";
    std::string host = "0.0.0.0";
    int port = 8080;
    int threads = 4;
    std::string log_level = "INFO";
    std::string config_file;
    size_t idle_timeout = 3600;
    std::string kv_cache_precision = "u8";
    size_t dynamic_quant_group_size = 32;
    bool enable_sdpa = true;
    size_t cache_interval_multiplier = 64;
    bool enable_prefix_caching = true;
    bool prompt_lookup = false;
    size_t num_assistant_tokens = 5;
    size_t max_ngram_size = 3;
    bool enable_mtp = true;
    std::string ffmpeg = "ffmpeg";

    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--txt2img") {
                image_models.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--txt2img-id") {
                image_model_ids.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--txt2txt") {
                text_models.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--txt2txt-id") {
                text_model_ids.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--txt2vid") {
                video_models.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--txt2vid-id") {
                video_model_ids.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--wav2txt") {
                asr_models.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--wav2txt-id") {
                asr_model_ids.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--txt2wav") {
                tts_models.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--txt2wav-id") {
                tts_model_ids.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--ffmpeg") {
                ffmpeg = get_arg(argc, argv, i, a.c_str());
            } else if (a == "-d" || a == "--device") {
                device = get_arg(argc, argv, i, a.c_str());
            } else if (a == "-h" || a == "--host") {
                host = get_arg(argc, argv, i, a.c_str());
            } else if (a == "-p" || a == "--port") {
                port = std::stoi(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "-t" || a == "--threads") {
                threads = std::stoi(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "-l" || a == "--log-level") {
                log_level = get_arg(argc, argv, i, a.c_str());
            } else if (a == "-c" || a == "--config") {
                config_file = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--idle-timeout") {
                idle_timeout = std::stoul(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--kv-cache-precision") {
                kv_cache_precision = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--dynamic-quant-gsize") {
                dynamic_quant_group_size =
                    std::stoul(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--enable-sdpa") {
                const std::string v = get_arg(argc, argv, i, a.c_str());
                if (v == "1" || v == "true" || v == "TRUE") {
                    enable_sdpa = true;
                } else if (v == "0" || v == "false" || v == "FALSE") {
                    enable_sdpa = false;
                } else {
                    throw std::runtime_error(
                        "--enable-sdpa expects true or false");
                }
            } else if (a == "--cache-interval-multiplier") {
                cache_interval_multiplier =
                    std::stoul(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--no-prefix-caching") {
                enable_prefix_caching = false;
            } else if (a == "--prompt-lookup") {
                prompt_lookup = true;
            } else if (a == "--num-assistant-tokens") {
                num_assistant_tokens =
                    std::stoul(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--max-ngram-size") {
                max_ngram_size =
                    std::stoul(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--no-mtp") {
                enable_mtp = false;
            } else if (a == "-v" || a == "--version") {
                std::cout << "openvino-server 0.1.0\n";
                return 0;
            } else if (a == "--help") {
                usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown option '" + a + "'");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        usage(argv[0]);
        return 2;
    }

    if (image_models.empty() && text_models.empty() && video_models.empty() &&
        asr_models.empty() && tts_models.empty()) {
        std::cerr << "error: at least one of --txt2img, --txt2txt, --txt2vid, "
                     "--wav2txt or --txt2wav PATH is required\n";
        usage(argv[0]);
        return 2;
    }
    if (!image_model_ids.empty() && image_model_ids.size() != image_models.size()) {
        std::cerr << "error: --txt2img-id must be supplied for every --txt2img\n";
        return 2;
    }
    if (image_models.empty() && !image_model_ids.empty()) {
        std::cerr << "error: --txt2img-id given but no --txt2img\n";
        return 2;
    }
    if (!text_model_ids.empty() && text_model_ids.size() != text_models.size()) {
        std::cerr << "error: --txt2txt-id must be supplied for every --txt2txt\n";
        return 2;
    }
    if (text_models.empty() && !text_model_ids.empty()) {
        std::cerr << "error: --txt2txt-id given but no --txt2txt\n";
        return 2;
    }
    if (!video_model_ids.empty() && video_model_ids.size() != video_models.size()) {
        std::cerr << "error: --txt2vid-id must be supplied for every --txt2vid\n";
        return 2;
    }
    if (video_models.empty() && !video_model_ids.empty()) {
        std::cerr << "error: --txt2vid-id given but no --txt2vid\n";
        return 2;
    }
    if (!asr_model_ids.empty() && asr_model_ids.size() != asr_models.size()) {
        std::cerr << "error: --wav2txt-id must be supplied for every --wav2txt\n";
        return 2;
    }
    if (asr_models.empty() && !asr_model_ids.empty()) {
        std::cerr << "error: --wav2txt-id given but no --wav2txt\n";
        return 2;
    }
    if (!tts_model_ids.empty() && tts_model_ids.size() != tts_models.size()) {
        std::cerr << "error: --txt2wav-id must be supplied for every --txt2wav\n";
        return 2;
    }
    if (tts_models.empty() && !tts_model_ids.empty()) {
        std::cerr << "error: --txt2wav-id given but no --txt2wav\n";
        return 2;
    }

    ovserver::set_ffmpeg_path(ffmpeg);

    drogon::HttpAppFramework& app = drogon::app();

    if (log_level == "TRACE") app.setLogLevel(trantor::Logger::kTrace);
    else if (log_level == "DEBUG") app.setLogLevel(trantor::Logger::kDebug);
    else if (log_level == "INFO") app.setLogLevel(trantor::Logger::kInfo);
    else if (log_level == "WARN") app.setLogLevel(trantor::Logger::kWarn);
    else if (log_level == "ERROR") app.setLogLevel(trantor::Logger::kError);
    else {
        std::cerr << "error: invalid log level '" << log_level << "'\n";
        return 2;
    }

    try {
        if (!config_file.empty()) {
            app.loadConfigFile(config_file);
        } else {
            app.addListener(host, port);
            app.setThreadNum(static_cast<size_t>(threads));
            app.setIdleConnectionTimeout(idle_timeout);
        }
    } catch (const std::exception& e) {
        std::cerr << "error configuring server: " << e.what() << "\n";
        return 2;
    }

    try {
        for (size_t i = 0; i < image_models.size(); ++i) {
            auto id = image_model_ids.empty() ? std::string("qwen-image")
                                              : image_model_ids[i];
            ovserver::ModelManager::instance().load_image(
                id, {std::filesystem::path(image_models[i]), device});
            LOG_INFO << "Image model '" << id
                     << "' ready at /v1/images/generations";
        }

        for (size_t i = 0; i < text_models.size(); ++i) {
            const std::filesystem::path p(text_models[i]);
            auto id = text_model_ids.empty() ? p.filename().string()
                                             : text_model_ids[i];
            ovserver::ModelManager::instance().load_text(
                id, {p,
                     device,
                     kv_cache_precision,
                     dynamic_quant_group_size,
                     enable_sdpa,
                     cache_interval_multiplier,
                     enable_prefix_caching,
                     prompt_lookup,
                     num_assistant_tokens,
                     max_ngram_size,
                     enable_mtp});
            LOG_INFO << "Text model '" << id
                     << "' ready at /v1/chat/completions";
        }

        for (size_t i = 0; i < video_models.size(); ++i) {
            const std::filesystem::path p(video_models[i]);
            auto id = video_model_ids.empty() ? p.filename().string()
                                              : video_model_ids[i];
            ovserver::ModelManager::instance().load_video(
                id, {p, device});
            LOG_INFO << "Video model '" << id
                     << "' ready at /v1/video/generations";
        }

        for (size_t i = 0; i < asr_models.size(); ++i) {
            const std::filesystem::path p(asr_models[i]);
            auto id = asr_model_ids.empty() ? p.filename().string()
                                            : asr_model_ids[i];
            ovserver::ModelManager::instance().load_asr(
                id, {p, device});
            LOG_INFO << "ASR model '" << id
                     << "' ready at /v1/audio/transcriptions";
        }

        for (size_t i = 0; i < tts_models.size(); ++i) {
            const std::filesystem::path p(tts_models[i]);
            auto id = tts_model_ids.empty() ? p.filename().string()
                                            : tts_model_ids[i];
            ovserver::ModelManager::instance().load_tts(
                id, {p, device});
            LOG_INFO << "TTS model '" << id
                     << "' ready at /v1/audio/speech";
        }
    } catch (const std::exception& e) {
        std::cerr << "error loading model: " << e.what() << "\n";
        return 1;
    }

    ovserver::register_api_handlers(app);
    app.run();

    // Graceful teardown: release the models and shut down OpenVINO plugins
    // while the process is still fully initialized. Leaving genai/OV objects
    // to be destroyed during C++ static teardown (after Drogon or OpenVINO's
    // own globals are gone) segfaults on exit.
    ovserver::ModelManager::instance().shutdown();
    ov::shutdown();

    return 0;
}