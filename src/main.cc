// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include <openvino/runtime/properties.hpp>

#include <drogon/drogon.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "ovserver/controller.hpp"
#include "ovserver/manager.hpp"

namespace {

void usage(const char* argv0) {
    std::cerr
        << "openvino-server: serve OpenAI-compatible /v1/images/generations\n"
        << "using OpenVINO GenAI image generation pipelines (e.g. Qwen-Image).\n\n"
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  -m, --model PATH        Path to an exported OpenVINO GenAI image\n"
        << "                          model directory. May be repeated. REQUIRED.\n"
        << "  -i, --model-id ID       Model id reported via /v1/models and the\n"
        << "                          'model' request field. Default: 'qwen-image'.\n"
        << "                          When multiple models are given each requires\n"
        << "                          an id.\n"
        << "  -d, --device DEVICE     OpenVINO device. Default: CPU.\n"
        << "  -h, --host HOST         Listen address. Default: 0.0.0.0\n"
        << "  -p, --port PORT         Listen port. Default: 8080.\n"
        << "  -t, --threads N         Number of event-loop threads. Default: 4.\n"
        << "  -l, --log-level LEVEL   Verbosity: TRACE, DEBUG, INFO, WARN, ERROR.\n"
        << "                          Default: INFO.\n"
        << "  -c, --config FILE       Load drogon from a JSON config file, which\n"
        << "                          overrides -h/-p/-t/-l above.\n"
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
                          ov::AnyMap& props) {
    if (!cache_dir.empty()) {
        props[ov::cache_dir.name()] = cache_dir;
    }
    if (device == "GPU") {
        props[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::THROUGHPUT;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> models;
    std::vector<std::string> model_ids;
    std::string device = "CPU";
    std::string host = "0.0.0.0";
    int port = 8080;
    int threads = 4;
    std::string log_level = "INFO";
    std::string config_file;
    std::string cache_dir;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-m" || a == "--model") {
                models.push_back(get_arg(argc, argv, i, a.c_str()));
            } else if (a == "-i" || a == "--model-id") {
                model_ids.push_back(get_arg(argc, argv, i, a.c_str()));
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

    if (models.empty()) {
        std::cerr << "error: at least one --model PATH is required\n";
        usage(argv[0]);
        return 2;
    }
    if (!model_ids.empty() && model_ids.size() != models.size()) {
        std::cerr << "error: --model-id must be supplied for every --model\n";
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
            try_set_device_props(device, cache_dir, props);
            ovserver::ModelManager::instance().load(
                id, {std::filesystem::path(models[i]), device, props});
            LOG_INFO << "Model '" << id << "' loaded from " << models[i];
        }
    } catch (const std::exception& e) {
        std::cerr << "error loading model: " << e.what() << "\n";
        return 1;
    }

    ovserver::register_api_handlers(app);
    app.run();

    return 0;
}