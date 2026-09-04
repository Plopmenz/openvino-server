#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <filesystem>
#include <cstdlib> // Nodig voor setenv / putenv
#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "openvino/genai/generation_config.hpp"

// Helperfunctie om omgevingsvariabelen cross-platform in te stellen
void set_env_var(const std::string& name, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

int main() {
    // 1. Dwing de runtime backend en GPU plugin parameters af via de omgeving.
    // Dit weerspiegelt exact de parameters uit jouw succesvolle OVMS log.
    set_env_var("ATTENTION_BACKEND", "SDPA");                  // Stopt de sampler_num_threads crash
    set_env_var("OV_GPU_KV_CACHE_PRECISION", "u4");            // Dwingt u4 bounds af op core-niveau
    set_env_var("OV_GPU_INFERENCE_PRECISION_HINT", "f16");      // Dwingt f16 precisie af
    set_env_var("OV_GPU_PERFORMANCE_HINT", "LATENCY");

    std::filesystem::path model_path = "/home/xnode/openvino/Qwen3.6-35B-A3B-int4-ov";
    std::string device = "GPU"; 

    // 2. Initialiseer de Scheduler conform de OVMS 2026.4.0 logdump
    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = 256;      
    scheduler_config.max_num_seqs = 256;                
    scheduler_config.num_kv_blocks = 0;                 
    scheduler_config.cache_size = 0;                    
    scheduler_config.num_linear_attention_blocks = 0;   
    scheduler_config.cache_interval_multiplier = 64;
    scheduler_config.enable_prefix_caching = true;     
    scheduler_config.use_sparse_attention = false;
    scheduler_config.use_cache_eviction = false;
    scheduler_config.dynamic_split_fuse = true;         // Essentieel voor prompt/generation splits

    // 3. Aanroep van de openbare 3-argumenten constructor
    std::cout << "Initialiseren van ContinuousBatchingPipeline op " << device << " via Unified Environment..." << std::endl;
    auto pipeline = std::make_shared<ov::genai::ContinuousBatchingPipeline>(
        model_path, 
        scheduler_config, 
        device
    );

    // 4. Haal de tokenizer op uit de succesvol gecompileerde engine
    ov::genai::Tokenizer tokenizer = pipeline->get_tokenizer();

    // 5. Configureer sampling parameters per request
    ov::genai::GenerationConfig generation_config;
    generation_config.max_new_tokens = 128;

    // 6. Voeg het verzoek toe
    uint64_t request_id = 1001;
    std::string prompt = "Explain continuous batching.";
    ov::genai::GenerationHandle handle = pipeline->add_request(request_id, prompt, generation_config);

    std::cout << "Model succesvol geladen op de GPU! Starten van de parallelle server-loop...\n" << std::endl;
    
    // 7. De asynchrone server loop
    while (pipeline->has_non_finished_requests()) {
        pipeline->step(); 

        auto outputs_map = handle->read();
        for (const auto& [stream_id, output_data] : outputs_map) {
            if (!output_data.generated_ids.empty()) {
                std::string text = tokenizer.decode(output_data.generated_ids);
                std::cout << text << std::flush;
            }
        }
    }
    
    std::cout << "\n\nInference succesvol afgerond." << std::endl;
    return 0;
}
