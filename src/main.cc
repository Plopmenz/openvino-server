// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/core/type/element_type.hpp>

#include <drogon/drogon.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "ovserver/controller.hpp"
#include "ovserver/manager.hpp"
#include "ovserver/vlm_model.hpp"

namespace {

void usage(const char* argv0) {
    std::cerr
        << "openvino-server: serve OpenAI-compatible /v1/images/generations\n"
        << "using OpenVINO GenAI image generation pipelines (e.g. Qwen-Image).\n\n"
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  -m, --model PATH        Path to an exported OpenVINO GenAI image\n"
        << "                          model directory. May be repeated. At least\n"
        << "                          one image model (--model) or VLM model\n"
        << "                          (--vlm-model) is required.\n"
        << "  -i, --model-id ID       Model id reported via /v1/models and the\n"
        << "                          'model' request field. Default: 'qwen-image'.\n"
        << "                          When multiple models are given each requires\n"
        << "                          an id.\n"
        << "  -d, --device DEVICE     OpenVINO device. Default: CPU.\n"
        << "      --text-encoder-device DEVICE\n"
        << "                          Device for the text encoder only (staged\n"
        << "                          compile). Defaults to --device. Useful to run\n"
        << "                          the static text encoding on CPU while the rest\n"
        << "                          stays on GPU.\n"
        << "      --transformer-device DEVICE\n"
        << "                          Device for the denoising transformer only.\n"
        << "                          Defaults to --device.\n"
        << "      --vae-device DEVICE  Device for the VAE decoder only.\n"
        << "                          Defaults to --device.\n"
        << "      --no-reshape         Skip pipe.reshape() and run the pipeline\n"
        << "                          with its exported dynamic shapes (as batch_inc\n"
        << "                          / batch_no_inc prepare). Useful to A/B test\n"
        << "                          dynamic vs static on a device, since the GPU\n"
        << "                          plugin stalls on the static denoiser while the\n"
        << "                          dynamic path completes (producing noise).\n"
        << "      --bound-dynamic    Stage a copy of the model whose dynamic dims\n"
        << "                          all get finite upper bounds (implies\n"
        << "                          --no-reshape). Required on GPU: its plugin\n"
        << "                          throws 'get_tensor() is called for dynamic\n"
        << "                          shape without upper bound' on unbounded IRs.\n"
        << "      --gpu-optimize     Apply the GPU compile knobs optimum-intel\n"
        << "                          relies on, which the default genai static\n"
        << "                          compile omits: FP16 inference precision\n"
        << "                          (INFERENCE_PRECISION_HINT) and explicit SDPA\n"
        << "                          fusion (GPU_ENABLE_SDPA_OPTIMIZATION). These\n"
        << "                          are ignored by non-GPU plugins. Without them\n"
        << "                          the static GPU denoiser is ~100x slower than\n"
        << "                          optimum on the same export (176s/step vs 3s).\n"
        << "      --bound-max N       Upper bound imposed on every dynamic dim by\n"
        << "                          --bound-dynamic. Default: 4096. The default\n"
        << "                          is too large for GPU (compile OOM/freeze);\n"
        << "                          try 1024-2048 for a 512x512 Qwen deployment\n"
        << "                          so the GPU compiles a bounded-dynamic model\n"
        << "                          instead of a frozen static one.\n"
        << "      --naive             Run the plain pipeline exactly like the\n"
        << "                          Python/optimum path: construct\n"
        << "                          Text2ImagePipeline(path, device) on first use\n"
        << "                          and generate() as-is. No reshape, no staged\n"
        << "                          compile, no shape keys. A/B test for the\n"
        << "                          other flags: if naive output differs, our\n"
        << "                          serving machinery is changing the result.\n"
        << "  -V, --vlm-model PATH    Path to an exported OpenVINO GenAI VLM (vision\n"
        << "                          language model) directory, e.g. Qwen2.5-VL /\n"
        << "                          Qwen3-VL / Qwen3.6-35B-A3B. May be repeated.\n"
        << "                          Served via /v1/chat/completions.\n"
        << "      --vlm-model-id ID    Model id for the corresponding --vlm-model.\n"
        << "                          Default: the directory basename. When multiple\n"
        << "                          VLM models are given each requires an id.\n"
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
        << "      --cache-dir DIR     OpenVINO cache directory.\n"
        << "  -v, --version           Print version and exit.\n"
        << "      --help              Show this help and exit.\n";
}

std::string get_arg(int argc, char** argv, int& i, const char* flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
    }
    return argv[++i];
}

bool try_set_device_props(const std::string& device,
                          const std::string& cache_dir,
                          const bool gpu_optimize,
                          ov::AnyMap& props) {
    if (!cache_dir.empty()) {
        props[ov::cache_dir.name()] = cache_dir;
    }
    // No performance mode is forced here: THROUGHPUT on GPU enables multi-stream
    // inference, which has been seen to deadlock the first denoising step when
    // combined with per-request pipeline cloning (see tools/*gpu*.py runs that
    // complete with default hints).
    // Under --gpu-optimize mirror the compile knobs optimum-intel uses on GPU:
    // FP16 inference precision and explicit SDPA fusion. Without these the
    // static GPU denoiser is ~100x slower than optimum on the same export. The
    // properties are inert on CPU/AUTO (they only affect a GPU device), so they
    // can be applied unconditionally.
    if (gpu_optimize &&
        (device.find("GPU") != std::string::npos ||
         device.find("AUTO") != std::string::npos)) {
        props[ov::hint::inference_precision.name()] = ov::element::f16;
        props[ov::intel_gpu::hint::enable_sdpa_optimization.name()] = true;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> models;
    std::vector<std::string> model_ids;
    std::vector<std::string> vlm_models;
    std::vector<std::string> vlm_model_ids;
    std::string device = "CPU";
    std::string text_encoder_device;
    std::string transformer_device;
    std::string vae_device;
    bool static_shapes = true;
    bool bound_dynamic = false;
    bool naive = false;
    bool gpu_optimize = false;
    int64_t bound_max = 4096;
    std::string host = "0.0.0.0";
    int port = 8080;
    int threads = 4;
    std::string log_level = "INFO";
    std::string config_file;
    std::string cache_dir;
    size_t idle_timeout = 3600;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-m" || a == "--model") {
                models.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "-i" || a == "--model-id") {
                model_ids.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "-V" || a == "--vlm-model") {
                vlm_models.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "--vlm-model-id") {
                vlm_model_ids.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "-d" || a == "--device") {
                device = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--text-encoder-device") {
                text_encoder_device = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--transformer-device") {
                transformer_device = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--vae-device") {
                vae_device = get_arg(argc, argv, i, a.c_str());
            } else if (a == "--no-reshape") {
                static_shapes = false;
            } else if (a == "--bound-dynamic") {
                bound_dynamic = true;
            } else if (a == "--naive") {
                naive = true;
            } else if (a == "--gpu-optimize") {
                gpu_optimize = true;
            } else if (a == "--bound-max") {
                bound_max = std::stoll(get_arg(argc, argv, i, a.c_str()));
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
            } else if (a == "--cache-dir") {
                cache_dir = get_arg(argc, argv, i, a.c_str());
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

    if (models.empty() && vlm_models.empty()) {
        std::cerr << "error: at least one --model or --vlm-model PATH is required\n";
        usage(argv[0]);
        return 2;
    }
    if (!model_ids.empty() && model_ids.size() != models.size()) {
        std::cerr << "error: --model-id must be supplied for every --model\n";
        return 2;
    }
    if (models.empty() && !model_ids.empty()) {
        std::cerr << "error: --model-id given but no --model\n";
        return 2;
    }
    if (!vlm_model_ids.empty() && vlm_model_ids.size() != vlm_models.size()) {
        std::cerr << "error: --vlm-model-id must be supplied for every --vlm-model\n";
        return 2;
    }
    if (vlm_models.empty() && !vlm_model_ids.empty()) {
        std::cerr << "error: --vlm-model-id given but no --vlm-model\n";
        return 2;
    }

    drogon::HttpAppFramework& app = drogon::app();

    // Configure logging before any drogon facility is used.
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

    // Load models.
    try {
        for (size_t i = 0; i < models.size(); ++i) {
            auto id = model_ids.empty() ? std::string("qwen-image") : model_ids[i];
            ov::AnyMap props;
            try_set_device_props(device, cache_dir, gpu_optimize, props);
            ovserver::ModelManager::instance().load(
                id,
                {std::filesystem::path(models[i]),
                 device,
                 text_encoder_device,
                 transformer_device,
                 vae_device,
                 static_shapes,
                 bound_dynamic,
                 naive,
                 bound_max,
                 props});
            LOG_INFO << "Model '" << id << "' loaded from " << models[i]
                     << " (text-encoder: " << text_encoder_device
                     << ", transformer: " << transformer_device
                     << ", vae: " << vae_device
                     << ", static_shapes: " << static_shapes
                     << ", bound_dynamic: " << bound_dynamic
                     << ", naive: " << naive << ")";
        }

        for (size_t i = 0; i < vlm_models.size(); ++i) {
            const std::filesystem::path p(vlm_models[i]);
            auto id = vlm_model_ids.empty()
                          ? p.filename().string()
                          : vlm_model_ids[i];
            ov::AnyMap props;
            try_set_device_props(device, cache_dir, gpu_optimize, props);
            ovserver::ModelManager::instance().load_vlm(
                id, {p, device, props});
            LOG_INFO << "VLM model '" << id << "' loaded from " << vlm_models[i]
                     << " (device: " << device
                     << ") served at /v1/chat/completions";
        }
    } catch (const std::exception& e) {
        std::cerr << "error loading model: " << e.what() << "\n";
        return 1;
    }

    ovserver::register_api_handlers(app);
    app.run();

    return 0;
}