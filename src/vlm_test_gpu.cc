// Minimal test: compile/link VLM model on GPU with pkgs openvino-genai 2026.3.0.0
// The pkgs version doesn't have visual_language/pipeline.hpp
#include <openvino/genai/llm_pipeline.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>

using namespace ov::genai;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model-path> <prompt> [device:GPU|CPU] [max_new_tokens:N]"
                  << std::endl;
        return 1;
    }

    const std::filesystem::path model_path{argv[1]};
    const std::string prompt{argv[2]};
    const std::string device = (argc > 3) ? argv[3] : "GPU";
    size_t max_new_tokens = 128;
    if (argc > 4) max_new_tokens = std::stoull(argv[4]);

    std::cout << "Loading VLM model: " << model_path
              << " device=" << device << std::endl;

    try {
        // Use basic LLMPipeline without scheduler_config (no continuous batching).
        // This is the simplest API available in the pkgs version.
        LLMPipeline pipeline(model_path, device);

        GenerationConfig gen_config;
        gen_config.max_new_tokens = max_new_tokens;

        std::cout << "Running: " << prompt << std::endl;
        DecodedResults result = pipeline.generate(
            prompt, gen_config);

        std::cout << "Result: " << result.texts[0] << std::endl;
    } catch (const ov::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
