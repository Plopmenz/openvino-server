#if 0
// Build inside the flake devShell with the same toolchain as the server:
//   c++ -std=c++17 -O2 tools/ovstage_compile.cc \
//       $(pkg-config --cflags --libs openvino) -o /tmp/ovstage_compile
#endif

// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

// Debug tool: reproduces exactly which stages the genai C++ text2image server
// compiles for the target device, printing per-stage OK/FAILED without running
// any generation. Usage:
//   ovstage_compile <model_dir> [device]
// It probes text_encoder, transformer and vae_decoder (the three subgraphs the
// text-to-image pipeline actually uses; vae_encoder is intentionally excluded).

#include <openvino/runtime/core.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

void compile_stage(ov::Core& core, const std::string& device, const fs::path& xml) {
    const std::string stage = xml.parent_path().filename().string();
    const auto start = std::chrono::steady_clock::now();

    auto model = core.read_model(xml.string());
    std::cout << "[" << stage << "] reading + compiling on '" << device << "' ...\n";
    for (auto&& input : model->inputs()) {
        std::cout << "    in  " << input.get_any_name() << " " << input.get_partial_shape() << "\n";
    }

    try {
        core.compile_model(model, device);  // NOLINT: intentionally the server path
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count();
        std::cout << "[" << stage << "] OK (" << static_cast<int>(ms) << " ms)\n";
    } catch (const std::exception& e) {
        std::cerr << "[" << stage << "] FAILED: " << e.what() << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ovstage_compile <model_dir> [device]\n";
        return 2;
    }
    const fs::path root = argv[1];
    const std::string device = argc > 2 ? argv[2] : "CPU";

    ov::Core core;
    std::cout << "available devices:";
    for (const auto& d : core.get_available_devices()) {
        std::cout << " " << d;
    }
    std::cout << "\n";

    const char* kStages[] = {"text_encoder", "transformer", "vae_decoder"};
    for (const char* stage : kStages) {
        const fs::path xml = root / stage / "openvino_model.xml";
        if (!fs::exists(xml)) {
            std::cout << "[" << stage << "] skipped: no " << xml << "\n";
            continue;
        }
        compile_stage(core, device, xml);
    }
    return 0;
}