// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include <drogon/drogon.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/runtime/core.hpp>

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
    std::string device = "CPU";
    std::string host = "0.0.0.0";
    int port = 8080;
    int threads = 4;
    std::string log_level = "INFO";
    std::string config_file;
    size_t idle_timeout = 3600;

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

    if (image_models.empty() && text_models.empty()) {
        std::cerr << "error: at least one --txt2img or --txt2txt PATH is required\n";
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
                id, {p, device});
            LOG_INFO << "Text model '" << id
                     << "' ready at /v1/chat/completions";
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