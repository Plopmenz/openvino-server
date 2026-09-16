// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include <drogon/drogon.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/runtime/core.hpp>

#include "ovserver/wav2txt.hpp"
#include "ovserver/audio.hpp"
#include "ovserver/common.hpp"
#include "ovserver/controller.hpp"
#include "ovserver/manager.hpp"
#include "ovserver/model_detect.hpp"
#include "ovserver/txt2wav.hpp"

namespace {

void usage(const char* argv0) {
    std::cerr
        << "openvino-server: serve image, text, video, speech-recognition and\n"
        << "text-to-speech OpenVINO GenAI models over an OpenAI-compatible\n"
        << "HTTP API.\n\n"
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "      --model PATH       Path to an exported OpenVINO GenAI model\n"
        << "                          directory (e.g. Qwen-Image, Qwen2.5-VL,\n"
        << "                          LTX-Video, Qwen3-ASR, Kokoro). Exactly\n"
        << "                          one; run another instance of the server\n"
        << "                          to serve more models.\n"
        << "      --model-id ID       Model id reported via /v1/models and the\n"
        << "                          'model' request field. Default: the model\n"
        << "                          directory name.\n"
        << "      --txt2img           Serve the model on /v1/images/generations\n"
        << "                          (image generation).\n"
        << "      --txt2txt           Serve the model on /v1/chat/completions\n"
        << "                          (text generation).\n"
        << "      --txt2vid           Serve the model on /v1/videos and\n"
        << "                          /v1/video/generations (video generation).\n"
        << "      --wav2txt           Serve the model on /v1/audio/transcriptions\n"
        << "                          (speech recognition).\n"
        << "      --txt2wav           Serve the model on /v1/audio/speech\n"
        << "                          (text-to-speech).\n"
        << "                          The mode is auto-detected from the model\n"
        << "                          directory when none of the above is given;\n"
        << "                          pass a flag explicitly to override.\n"
        << "      --device-props JSON  Device-scoped inference properties as\n"
        << "                          JSON, e.g. '{\"gpu\":{\"KV_CACHE_PRECISION\":\n"
        << "                          \"u8\"}}'. Top-level keys are device names\n"
        << "                          (case-insensitive) whose values are\n"
        << "                          OpenVINO property maps forwarded verbatim\n"
        << "                          to the GenAI pipeline.\n"
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
        << "      --ffmpeg PATH       Path to the ffmpeg binary used to decode\n"
        << "                          uploaded audio and encode generated video.\n"
        << "                          Default: 'ffmpeg' on PATH. Set to an empty\n"
        << "                          string to require 16 kHz WAV uploads and skip\n"
        << "                          MP4 output.\n"
        << "      --cache-dir PATH    Directory where OpenVINO stores compiled\n"
        << "                          model blobs, reused across restarts to speed\n"
        << "                          up model loading. Disabled when not set.\n"
        << "      --temp-dir PATH     Directory for generated artifacts. Videos\n"
        << "                          are stored under PATH/videos using a random\n"
        << "                          UUID as file name, returned to the client as\n"
        << "                          the video id. Default: system temp dir.\n"
        << "  -d, --device DEVICE     OpenVINO device (CPU, GPU, AUTO, ...).\n"
        << "                          Default: AUTO.\n"
        << "      --second-device DEVICE\n"
        << "                          Device for the text model's bundled MTP\n"
        << "                          draft head when speculative decoding is\n"
        << "                          active. Defaults to --device.\n"
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
    std::string model_path;
    std::string model_id;
    bool enable_txt2img = false;
    bool enable_txt2txt = false;
    bool enable_txt2vid = false;
    bool enable_wav2txt = false;
    bool enable_txt2wav = false;
    std::string device = "AUTO";
    std::string second_device;
    std::string host = "0.0.0.0";
    int port = 8080;
    int threads = 4;
    std::string log_level = "INFO";
    std::string config_file;
    size_t idle_timeout = 3600;
    std::string device_props;
    size_t cache_interval_multiplier = 64;
    bool enable_prefix_caching = true;
    bool prompt_lookup = false;
    size_t num_assistant_tokens = 5;
    size_t max_ngram_size = 3;
    bool enable_mtp = true;
    std::string ffmpeg = "ffmpeg";
    std::string cache_dir;
    std::string temp_dir;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--model") {
                if (!model_path.empty()) {
                    throw std::runtime_error(
                        "only one --model is supported; run another "
                        "instance of openvino-server to serve more models");
                }
                model_path = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--model-id") {
                if (!model_id.empty()) {
                    throw std::runtime_error(
                        "--model-id must be supplied at most once");
                }
                model_id = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--txt2img") {
                enable_txt2img = true;
            } else if (a == "--txt2txt") {
                enable_txt2txt = true;
            } else if (a == "--txt2vid") {
                enable_txt2vid = true;
            } else if (a == "--wav2txt") {
                enable_wav2txt = true;
            } else if (a == "--txt2wav") {
                enable_txt2wav = true;
            } else if (a == "--ffmpeg") {
                ffmpeg = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--cache-dir") {
                cache_dir = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--temp-dir") {
                temp_dir = get_arg(argc, argv, i, a.c_str());
            } else if (a == "-d" || a == "--device") {
                device = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--second-device") {
                second_device = get_arg(argc, argv, i, a.c_str());
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
            } else if (a == "--device-props") {
                device_props = get_arg(argc, argv, i, a.c_str());
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

    if (model_path.empty()) {
        std::cerr << "error: --model PATH is required\n";
        usage(argv[0]);
        return 2;
    }
    if (!enable_txt2img && !enable_txt2txt && !enable_txt2vid &&
        !enable_wav2txt && !enable_txt2wav) {
        const std::vector<ovserver::ModelCapability> detected =
            ovserver::detect_model_capabilities(model_path);
        if (detected.empty()) {
            std::cerr
                << "error: no mode flag given and could not auto-detect the "
                   "model type from '"
                << model_path << "'\n";
            std::cerr << "pass one of --txt2img, --txt2txt, --txt2vid, "
                         "--wav2txt or --txt2wav\n";
            usage(argv[0]);
            return 2;
        }
        for (const ovserver::ModelCapability cap : detected) {
            switch (cap) {
                case ovserver::ModelCapability::Txt2Img: enable_txt2img = true; break;
                case ovserver::ModelCapability::Txt2Txt: enable_txt2txt = true; break;
                case ovserver::ModelCapability::Txt2Vid: enable_txt2vid = true; break;
                case ovserver::ModelCapability::Wav2Txt: enable_wav2txt = true; break;
                case ovserver::ModelCapability::Txt2Wav: enable_txt2wav = true; break;
                case ovserver::ModelCapability::Unknown: break;
            }
            std::cerr << "[auto-detect] '" << model_path << "' identified as "
                      << ovserver::capability_name(cap) << std::endl;
        }
    }

    ovserver::set_ffmpeg_path(ffmpeg);
    try {
        ovserver::set_device_props(device_props);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        usage(argv[0]);
        return 2;
    }

    if (!cache_dir.empty()) {
        try {
            std::filesystem::create_directories(cache_dir);
        } catch (const std::exception& e) {
            std::cerr << "error creating cache dir '" << cache_dir << "': "
                      << e.what() << "\n";
            return 2;
        }
    }

    // Video outputs are stored under <temp-dir>/videos as <random-uuid>.mp4;
    // the uuid is used as the video id returned to clients.
    {
        std::error_code ec;
        std::filesystem::path video_root =
            temp_dir.empty()
                ? std::filesystem::temp_directory_path(ec)
                : std::filesystem::path(temp_dir);
        if (ec) {
            std::cerr << "error resolving temp dir: " << ec.message() << "\n";
            return 2;
        }
        const std::filesystem::path videos_dir = video_root / "videos";
        std::error_code mkec;
        std::filesystem::create_directories(videos_dir, mkec);
        if (mkec) {
            std::cerr << "error creating videos dir '" << videos_dir << "': "
                      << mkec.message() << "\n";
            return 2;
        }
        ovserver::set_video_storage_dir(videos_dir);
    }

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
        const std::filesystem::path p(model_path);
        std::string id = model_id;
        if (id.empty()) {
            // Strip trailing separators so `--model /dir/of/model/` still
            // derives a sensible default id from the directory name.
            std::string dir = model_path;
            while (dir.size() > 1 && dir.back() == '/') {
                dir.pop_back();
            }
            id = std::filesystem::path(dir).filename().string();
        }
        if (enable_txt2img) {
            ovserver::ModelManager::instance().load_image(id, {p, device, cache_dir});
            LOG_INFO << "Image model '" << id
                     << "' ready at /v1/images/generations";
        }
        if (enable_txt2txt) {
            ovserver::ModelManager::instance().load_text(
                id, {p,
                     device,
                     cache_interval_multiplier,
                     enable_prefix_caching,
                     prompt_lookup,
                     num_assistant_tokens,
                     max_ngram_size,
                     enable_mtp,
                     cache_dir,
                     second_device});
            LOG_INFO << "Text model '" << id
                     << "' ready at /v1/chat/completions";
        }
        if (enable_txt2vid) {
            ovserver::ModelManager::instance().load_video(id, {p, device, cache_dir});
            LOG_INFO << "Video model '" << id
                     << "' ready at /v1/videos "
                        "(async) and /v1/video/generations (sync)";
        }
        if (enable_wav2txt) {
            ovserver::ModelManager::instance().load_asr(id, {p, device, cache_dir});
            LOG_INFO << "ASR model '" << id
                     << "' ready at /v1/audio/transcriptions";
        }
        if (enable_txt2wav) {
            ovserver::ModelManager::instance().load_tts(id, {p, device, cache_dir});
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