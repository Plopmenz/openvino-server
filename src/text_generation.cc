// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/text_generation.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <openvino/genai/llm_pipeline.hpp>
#include <openvino/genai/text_streamer.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/properties.hpp>

namespace ovserver {

namespace {

std::size_t proc_field_kb(const std::string& field) {
    std::ifstream status("/proc/self/status");
    if (!status) {
        return 0;
    }
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind(field, 0) == 0) {
            const std::size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos) {
                return static_cast<std::size_t>(std::stoull(line.substr(pos)));
            }
        }
    }
    return 0;
}

// Total GPU device memory currently allocated by the plugin, in MiB.
// Returns -1 when not applicable (non-GPU device, plugin unavailable).
double device_gpu_mib(const std::string& device) {
    if (device.find("GPU") == std::string::npos) {
        return -1.0;
    }
    try {
        ov::Core core;
        auto stats = core.get_property(device, ov::intel_gpu::memory_statistics);
        std::uint64_t bytes = 0;
        for (const auto& [name, value] : stats) {
            bytes += value;
        }
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    } catch (const std::exception&) {
        return -1.0;
    }
}

std::string finish_reason_str(ov::genai::GenerationFinishReason r) {
    switch (r) {
        case ov::genai::GenerationFinishReason::LENGTH:
            return "length";
        case ov::genai::GenerationFinishReason::TOOL_CALL:
            return "tool_calls";
        case ov::genai::GenerationFinishReason::STOP:
        default:
            return "stop";
    }
}

std::string read_utf8_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Splits a decoded text stream into visible "content" and "reasoning_content"
// by scanning for the model's thinking markers. The markers themselves are
// dropped from both streams. Handles markers that are split across chunks by
// holding back the longest suffix of the current text that could be the start
// of a marker.
class ReasoningStreamSplitter {
public:
    ReasoningStreamSplitter(bool expect_open_tag,
                            std::string open_tag,
                            std::string close_tag)
        : m_open_tag(std::move(open_tag)),
          m_close_tag(std::move(close_tag)),
          m_in_reasoning(!expect_open_tag) {}

    // Longest suffix of `text` that is a prefix of `tag` (handles markers split
    // across fragment boundaries).
    static std::size_t prefix_overlap(const std::string& text,
                                      const std::string& tag) {
        const std::size_t max_check = std::min(text.size(), tag.size());
        std::size_t longest = 0;
        for (std::size_t i = 1; i <= max_check; ++i) {
            if (text.compare(text.size() - i, i, tag, 0, i) == 0) {
                longest = i;
            }
        }
        return longest;
    }

    // Feeds one decoded fragment, routing it to the content or reasoning
    // callback. Returns false if a callback requested cancellation.
    template <typename TextCb, typename ReasCb>
    bool feed(const std::string& fragment,
              TextCb&& on_text,
              ReasCb&& on_reasoning) {
        std::string buf = m_pending + fragment;
        m_pending.clear();
        while (!buf.empty()) {
            if (!m_in_reasoning && m_open_tag.empty()) {
                // No open marker expected (DeepSeek-R1 style): reasoning (if
                // any) already ended, the rest is plain content.
                if (!emit(buf, on_text, on_reasoning)) {
                    return false;
                }
                buf.clear();
                break;
            }
            const std::string& tag = m_in_reasoning ? m_close_tag : m_open_tag;
            const std::size_t idx = buf.find(tag);
            if (idx != std::string::npos) {
                if (!emit(buf.substr(0, idx), on_text, on_reasoning)) {
                    return false;
                }
                m_in_reasoning = !m_in_reasoning;
                buf.erase(0, idx + tag.size());
                continue;
            }
            const std::size_t hold = prefix_overlap(buf, tag);
            const std::size_t emit_until = buf.size() - hold;
            if (emit_until > 0) {
                if (!emit(buf.substr(0, emit_until), on_text, on_reasoning)) {
                    return false;
                }
                buf.erase(0, emit_until);
            }
            m_pending = std::move(buf);
            break;
        }
        return true;
    }

    template <typename TextCb, typename ReasCb>
    bool flush(TextCb&& on_text, ReasCb&& on_reasoning) {
        if (m_pending.empty()) {
            return true;
        }
        std::string leftover = std::move(m_pending);
        m_pending.clear();
        return emit(leftover, on_text, on_reasoning);
    }

private:
    template <typename TextCb, typename ReasCb>
    bool emit(const std::string& text, TextCb&& on_text, ReasCb&& on_reasoning) {
        if (text.empty()) {
            return true;
        }
        if (m_in_reasoning) {
            return on_reasoning(text);
        }
        return on_text(text);
    }

    std::string m_open_tag;
    std::string m_close_tag;
    bool m_in_reasoning;
    std::string m_pending;
};

// Finds the JSON object starting at `start` (a balanced brace scan that ignores
// braces inside string literals). Returns the object text on success.
std::optional<std::string> scan_json_object(const std::string& text,
                                            std::size_t start) {
    std::size_t depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (std::size_t i = start; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == '{') {
            ++depth;
        } else if (c == '}') {
            if (depth == 0) {
                return std::nullopt;
            }
            --depth;
            if (depth == 0) {
                return text.substr(start, i - start + 1);
            }
        }
    }
    return std::nullopt;
}

// Escapes `s` so it can be embedded as a JSON string literal.
std::string json_quote(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(c >> 4) & 0xF];
                    out += hex[c & 0xF];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Trims ASCII whitespace from both ends.
std::string trim_ws(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\n' ||
                     s[b] == '\r')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' ||
                     s[e - 1] == '\n' || s[e - 1] == '\r')) {
        --e;
    }
    return s.substr(b, e - b);
}

// Parses the Qwen3.6 / Hermes-style XML tool-call format
//   <tool_call><function=NAME><parameter=KEY1>VAL1</parameter>...</function></tool_call>
// into an OpenAI ToolCall whose `arguments` is a JSON object string. Returns
// nullopt when the block does not look like this format.
std::optional<ToolCall> parse_xml_tool_call(const std::string& text,
                                            std::size_t start) {
    const std::string function_tag = "<function=";
    const std::size_t fpos = text.find(function_tag, start);
    if (fpos == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t name_start = fpos + function_tag.size();
    const std::size_t name_end = text.find('>', name_start);
    if (name_end == std::string::npos) {
        return std::nullopt;
    }
    const std::string name = trim_ws(text.substr(name_start, name_end - name_start));
    if (name.empty()) {
        return std::nullopt;
    }
    const std::size_t fn_close = text.find("</function>", name_end);
    const std::size_t params_end =
        (fn_close == std::string::npos) ? text.size() : fn_close;

    const std::string param_tag = "<parameter=";
    const std::string param_close = "</parameter>";
    std::vector<std::pair<std::string, std::string>> params;
    std::size_t p = name_end + 1;
    while (p < params_end) {
        const std::size_t ps = text.find(param_tag, p);
        if (ps == std::string::npos || ps >= params_end) {
            break;
        }
        const std::size_t kstart = ps + param_tag.size();
        const std::size_t kend = text.find('>', kstart);
        if (kend == std::string::npos || kend >= params_end) {
            break;
        }
        const std::string key =
            trim_ws(text.substr(kstart, kend - kstart));
        const std::size_t pce = text.find(param_close, kend);
        if (pce == std::string::npos || pce >= params_end) {
            break;
        }
        const std::string value =
            trim_ws(text.substr(kend + 1, pce - (kend + 1)));
        params.emplace_back(key, value);
        p = pce + param_close.size();
    }

    // Serialize the parameters as a JSON object string (OpenAI arguments).
    std::string args = "{";
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (i > 0) {
            args += ",";
        }
        args += "\"" + json_quote(params[i].first) + "\":\""
                + json_quote(params[i].second) + "\"";
    }
    args += "}";

    ToolCall call;
    call.name = name;
    call.arguments = std::move(args);
    return call;
}

// Extracts OpenAI-format tool calls from assistant output.
//
// Handles two formats, both optional wrapping in <tool_call>...</tool_call>:
//   - Hermes/Qwen3 JSON: {"name": "...", "arguments": {...}}
//   - Qwen3.6/Hermes XML: <function=NAME><parameter=KEY>VAL</parameter>...</function>
//   - Llama-3.1 JSON: a bare JSON object {"name": ..., "arguments": ...}
// The extraction is conservative: only calls with a non-empty function name are
// reported, so arbitrary JSON in the model output is not misparsed.
std::vector<ToolCall> extract_tool_calls(const std::string& text) {
    std::vector<ToolCall> calls;
    const std::string open_marker = "<tool_call>";
    const std::string close_marker = "</tool_call>";
    // Once the model uses explicit <tool_call> framing, only framed blocks are
    // interpreted as calls (narrative JSON in the surrounding text is ignored).
    const bool wrapped_framing =
        text.find(open_marker) != std::string::npos;
    std::size_t pos = 0;
    std::size_t idx = 0;
    if (wrapped_framing) {
        while (pos < text.size()) {
            const std::size_t open_at = text.find(open_marker, pos);
            if (open_at == std::string::npos) {
                break;
            }
            const std::size_t close_at =
                text.find(close_marker, open_at);
            const std::size_t stop =
                (close_at != std::string::npos) ? close_at : text.size();
            // Prefer the XML <function=...> form (Qwen3.6 / Hermes).
            if (const auto xml = parse_xml_tool_call(text, open_at);
                xml && xml->name.size() + xml->arguments.size() <=
                           stop - open_at) {
                ToolCall call = *xml;
                call.id = "call_" + std::to_string(idx++);
                calls.push_back(std::move(call));
                pos = open_at + open_marker.size();
                if (close_at != std::string::npos) {
                    pos = close_at + close_marker.size();
                }
                continue;
            }
            const std::size_t start = text.find('{', open_at);
            if (start == std::string::npos || start > stop) {
                break;
            }
            const auto obj = scan_json_object(text, start);
            if (obj && start + obj->size() <= stop) {
                try {
                    ov::genai::JsonContainer parsed =
                        ov::genai::JsonContainer::from_json_string(*obj);
                    if (parsed.is_object() && parsed.contains("name") &&
                        parsed["name"].is_string() &&
                        parsed["name"].as_string() &&
                        !parsed["name"].as_string()->empty()) {
                        std::string args_str = "{}";
                        if (parsed.contains("arguments")) {
                            const ov::genai::JsonContainer args =
                                parsed["arguments"];
                            if (args.is_string() && args.as_string()) {
                                args_str = *args.as_string();
                            } else {
                                args_str = args.to_json_string();
                            }
                        }
                        ToolCall call;
                        call.id = "call_" + std::to_string(idx++);
                        call.name = *parsed["name"].as_string();
                        call.arguments = std::move(args_str);
                        calls.push_back(std::move(call));
                    }
                } catch (const std::exception&) {
                    // Not a valid JSON object; keep scanning for the next one.
                }
                pos = start + obj->size();
            } else {
                break;
            }
            if (close_at != std::string::npos &&
                (pos == std::string::npos || pos <= close_at)) {
                pos = close_at + close_marker.size();
            }
        }
        return calls;
    }
    // Bare JSON-object style (e.g. Llama-3.1): scan for objects with "name".
    while (pos < text.size()) {
        const std::size_t start = text.find('{', pos);
        if (start == std::string::npos) {
            break;
        }
        const auto obj = scan_json_object(text, start);
        if (!obj) {
            break;
        }
        try {
            ov::genai::JsonContainer parsed =
                ov::genai::JsonContainer::from_json_string(*obj);
            if (parsed.is_object() && parsed.contains("name") &&
                parsed["name"].is_string() &&
                parsed["name"].as_string() &&
                !parsed["name"].as_string()->empty()) {
                std::string args_str = "{}";
                if (parsed.contains("arguments")) {
                    const ov::genai::JsonContainer args = parsed["arguments"];
                    if (args.is_string() && args.as_string()) {
                        args_str = *args.as_string();
                    } else {
                        args_str = args.to_json_string();
                    }
                }
                ToolCall call;
                call.id = "call_" + std::to_string(idx++);
                call.name = *parsed["name"].as_string();
                call.arguments = std::move(args_str);
                calls.push_back(std::move(call));
            }
        } catch (const std::exception&) {
            // Not a valid JSON object; keep scanning for the next one.
        }
        pos = start + obj->size();
    }
    return calls;
}

const char* ascii_lower(std::string& s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return s.c_str();
}

}  // namespace

TextGenerationModel::TextGenerationModel(const std::string& id,
    const TextGenerationSpec& spec)
    : m_id(id), m_models_path(spec.path), m_device(spec.device) {
    const auto t0 = std::chrono::steady_clock::now();
    std::cerr << "[text model '" << id << "'] loading from " << m_models_path
              << " on " << m_device << " ..." << std::endl;

    ov::genai::SchedulerConfig sched_cfg;
    sched_cfg.cache_interval_multiplier = spec.cache_interval_multiplier;
    sched_cfg.enable_prefix_caching = spec.enable_prefix_caching;
    try {
        // GPU device properties are passed through the head-compile property
        // map (forwarded to ov::Core::compile_model by the pipeline). For
        // non-GPU devices they would be rejected, so they are only set on GPU.
        ov::AnyMap props;
        if (m_device.find("GPU") != std::string::npos) {
            props.emplace(
                ov::hint::kv_cache_precision(ov::element::Type(spec.kv_cache_precision)));
            props.emplace(
                ov::hint::dynamic_quantization_group_size(
                    spec.dynamic_quant_group_size));
            props.emplace(
                ov::intel_gpu::hint::enable_sdpa_optimization(
                    spec.enable_sdpa_optimization));
        }
        // Speculative decoding: prompt-lookup drafts candidates by n-gram
        // matching against the prompt (no extra model); otherwise if the model
        // ships a bundled MTP head (openvino_mtp_model.xml) it is used for
        // multi-token prediction. genai picks the matching strategy from these
        // properties. Both are exclusive: prompt_lookup wins when enabled.
        if (spec.prompt_lookup) {
            props.insert(ov::genai::prompt_lookup(true));
        } else if (spec.enable_mtp &&
                   std::filesystem::exists(spec.path / "openvino_mtp_model.xml")) {
            props.insert(ov::genai::draft_model(spec.path, spec.device));
            std::cerr << "[text model '" << id
                      << "'] MTP speculative decoding enabled (bundled head)"
                      << std::endl;
        }
        m_prompt_lookup_active = spec.prompt_lookup;
        m_mtp_active = !spec.prompt_lookup && spec.enable_mtp &&
                       std::filesystem::exists(spec.path / "openvino_mtp_model.xml");
        m_num_assistant_tokens = spec.num_assistant_tokens;
        m_max_ngram_size = spec.max_ngram_size;
        if (spec.prompt_lookup) {
            std::cerr << "[text model '" << id
                      << "'] prompt-lookup speculative decoding enabled"
                      << std::endl;
        }
        m_pipeline = std::make_shared<ov::genai::ContinuousBatchingPipeline>(
            m_models_path, sched_cfg, m_device, props);
        m_tokenizer = m_pipeline->get_tokenizer();
        detect_parsers();
    } catch (const std::exception& e) {
        std::cerr << "[text model '" << id << "'] loading FAILED: " << e.what()
                  << std::endl;
        throw;
    }
    const auto load_s = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    std::cerr << "[text model '" << id << "'] loaded in " << load_s
              << " s" << std::endl;
    std::cerr << "[text model '" << id << "'] memory: ";
    const double gpu_mib = device_gpu_mib(m_device);
    if (gpu_mib >= 0.0) {
        std::cerr << "GPU " << std::fixed << std::setprecision(2)
                  << (gpu_mib / 1024.0) << " GiB" << std::endl;
    } else if (m_device.find("GPU") != std::string::npos) {
        std::cerr << "GPU ??? GiB" << std::endl;
    } else {
        std::cerr << "CPU " << std::fixed << std::setprecision(2)
                  << (proc_field_kb("VmRSS:") / (1024.0 * 1024.0)) << " GiB"
                  << std::endl;
    }
    m_last_snapshot = std::chrono::steady_clock::now();
    m_executor = std::thread([this] { executor_run(); });
}

TextGenerationModel::~TextGenerationModel() {
    m_stop = true;
    {
        std::lock_guard<std::mutex> lock(m_cv_mutex);
        m_cv.notify_all();
    }
    if (m_executor.joinable()) {
        m_executor.join();
    }
}

std::uint64_t TextGenerationModel::next_request_id() {
    std::lock_guard<std::mutex> lock(m_id_mutex);
    return m_next_id++;
}

// Auto-detects the reasoning markers and the tool-call format for this model so
// clients get reasoning_content / tool_calls without any server-side flags.
// The detection is heuristic and deliberately conservative:
//   - Reasoning is keyed off the chat template's thinking markers; when nothing
//     known is found, no reasoning split is performed and the raw text is
//     returned unchanged (graceful degradation, never a forced format).
//   - Tool calling is only recognized for templates that reference tools
//     (e.g. <tool_call> / <tool_response> in Hermes/Qwen3 style).
void TextGenerationModel::detect_parsers() {
    std::string tmpl;
    try {
        tmpl = m_tokenizer.get_chat_template();
    } catch (const std::exception&) {
        tmpl = {};
    }

    std::string config = read_utf8_file(m_models_path / "config.json");
    std::string config_lower = config;
    const std::string family = ascii_lower(config_lower);
    const std::string tmpl_lower = [&] {
        std::string t = tmpl;
        return std::string(ascii_lower(t));
    }();

    // --- Reasoning markers -----------------------------------------------
    if (tmpl_lower.find("</think>") != std::string::npos) {
        // Qwen3.5 / Qwen3.6: reasoning block ends with </think>.
        m_reasoning = ReasoningMarkers{/*enabled=*/true,
                                       /*expect_open_tag=*/false,
                                       /*open_tag=*/"",
                                       /*close_tag=*/"</think>"};
    } else if (family.find("deepseek") != std::string::npos ||
               tmpl_lower.find("reasoning_content") != std::string::npos ||
               tmpl_lower.find(" reasoning") != std::string::npos) {
        // DeepSeek-R1 (and distills): the open tag is injected by the template
        // (or history), generation starts inside the reasoning section.
        m_reasoning = ReasoningMarkers{/*enabled=*/true,
                                       /*expect_open_tag=*/false,
                                       /*open_tag=*/"",
                                       /*close_tag=*/" response"};
    } else if (tmpl_lower.find(" thinking") != std::string::npos ||
               tmpl_lower.find("thinking") != std::string::npos) {
        // Qwen3 (and Qwen3-style) models emit " thinking ...  response".
        m_reasoning = ReasoningMarkers{/*enabled=*/true,
                                       /*expect_open_tag=*/true,
                                       /*open_tag=*/" thinking",
                                       /*close_tag=*/" response"};
    }

    // --- Tool calling -----------------------------------------------------
    m_tools_supported = tmpl.find("<tool_call>") != std::string::npos ||
                        tmpl.find("<tool_response>") != std::string::npos ||
                        tmpl.find("tools") != std::string::npos;

    std::cerr << "[text model '" << m_id << "'] parsers: reasoning=";
    if (m_reasoning.enabled) {
        if (m_reasoning.expect_open_tag) {
            std::cerr << "'" << m_reasoning.open_tag << "' / '"
                      << m_reasoning.close_tag << "' (expect open tag)";
        } else {
            std::cerr << "'<start>' / '" << m_reasoning.close_tag
                      << "' (no open tag)";
        }
    } else {
        std::cerr << "none (raw output)";
    }
    std::cerr << " | tools=" << (m_tools_supported ? "yes" : "no")
              << std::endl;
}

void TextGenerationModel::executor_run() {
    while (!m_stop) {
        std::unique_lock<std::mutex> lock(m_cv_mutex);
        m_cv.wait(lock, [this] {
            return m_stop || m_pipeline->has_non_finished_requests();
        });
        if (m_stop) {
            return;
        }
        lock.unlock();
        try {
            while (!m_stop && m_pipeline->has_non_finished_requests()) {
                m_pipeline->step();
                const auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double>(now - m_last_snapshot).count()
                    >= 1.0) {
                    const auto window_start = m_last_snapshot;
                    m_last_snapshot = now;
                    snapshot_log(now, window_start);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[text model '" << m_id
                      << "'] scheduler step error: " << e.what() << std::endl;
        }
    }
}

void TextGenerationModel::snapshot_log(
    const std::chrono::steady_clock::time_point now,
    const std::chrono::steady_clock::time_point window_start) {
    std::lock_guard<std::mutex> lock(m_metrics_mutex);
    if (!m_requests.empty()) {
        std::size_t prefill = 0;
        std::size_t decode = 0;
        std::size_t decode_delta = 0;
        for (auto& [rid, m] : m_requests) {
            if (m.tokens == 0) {
                ++prefill;
            } else {
                ++decode;
            }
            if (!m.has_baseline) {
                // Request not present in the previous snapshot: record a
                // baseline now so its rates show up from the next snapshot.
                m.has_baseline = true;
                m.baseline_tokens = m.tokens;
                m.baseline_time = now;
                continue;
            }
            if (m.tokens > m.baseline_tokens) {
                decode_delta += m.tokens - m.baseline_tokens;
            }
            m.baseline_tokens = m.tokens;
            m.baseline_time = now;
        }

        const double dt =
            std::chrono::duration<double>(now - window_start).count();
        if (dt > 0.0) {
            // Aggregate throughput divided by the request count: average speed
            // of one request in each phase.
            const double prefill_rate =
                (prefill > 0)
                    ? (m_prefill_tokens_in_window / dt /
                       static_cast<double>(prefill))
                    : 0.0;
            const double decode_rate =
                (decode > 0)
                    ? (static_cast<double>(decode_delta) / dt /
                       static_cast<double>(decode))
                    : 0.0;
            std::cerr << "[text model '" << m_id << "'] snapshot: prefill "
                      << prefill_rate << " tok/s (" << prefill
                      << " req) | decode " << decode_rate << " tok/s (" << decode
                      << " req)" << std::endl;
        }
    }
    m_prefill_tokens_in_window = 0;
}

TextResult TextGenerationModel::generate(const TextGenerateOptions& opts) {
    ov::genai::GenerationConfig cfg;
    if (opts.max_new_tokens) cfg.max_new_tokens = *opts.max_new_tokens;
    if (opts.temperature) {
        cfg.temperature = *opts.temperature;
        cfg.do_sample = true;
    }
    if (opts.top_p) {
        cfg.top_p = *opts.top_p;
        cfg.do_sample = true;
    }
    if (opts.top_k) {
        cfg.top_k = *opts.top_k;
        cfg.do_sample = true;
    }
    if (opts.rng_seed) cfg.rng_seed = *opts.rng_seed;
    if (!opts.stop_strings.empty()) cfg.stop_strings = opts.stop_strings;
    if (opts.frequency_penalty) cfg.frequency_penalty = *opts.frequency_penalty;
    if (opts.presence_penalty) cfg.presence_penalty = *opts.presence_penalty;
    if (opts.structured_output) {
        cfg.structured_output_config = opts.structured_output;
    }
    if (m_prompt_lookup_active) {
        cfg.num_assistant_tokens = m_num_assistant_tokens;
        cfg.max_ngram_size = m_max_ngram_size;
    } else if (m_mtp_active) {
        cfg.num_assistant_tokens = m_num_assistant_tokens;
    }

    // Apply the model's own chat template so role framing matches training.
    // A full OpenAI-style conversation history (including assistant tool_calls
    // and tool-role results) is preserved when the client supplied one.
    ov::genai::ChatHistory history;
    if (!opts.chat_messages.empty()) {
        for (const auto& message : opts.chat_messages) {
            history.push_back(message);
        }
    } else {
        if (!opts.system_message.empty()) {
            history.push_back({{"role", "system"}, {"content", opts.system_message}});
        }
        history.push_back({{"role", "user"}, {"content", opts.prompt}});
    }
    if (opts.tools) {
        history.set_tools(*opts.tools);
    }
    const std::string templated =
        m_tokenizer.apply_chat_template(history, /*add_generation_prompt=*/true);
    // Prompt token count: needed to measure aggregate prefill throughput (the
    // CB pipeline does not expose per-request prefill timings).
    const std::size_t prompt_tokens =
        m_tokenizer.encode(templated).input_ids.get_size();

    const std::uint64_t req_id = next_request_id();
    ov::genai::GenerationHandle handle =
        m_pipeline->add_request(req_id, templated, opts.images, cfg);
    {
        std::lock_guard<std::mutex> lock(m_cv_mutex);
        m_cv.notify_one();
    }
    {
        std::lock_guard<std::mutex> lock(m_metrics_mutex);
        m_requests[req_id] = {std::chrono::steady_clock::now(),
                              std::chrono::steady_clock::now(), 0, 0,
                              prompt_tokens, false};
    }

    std::string accumulated;
    std::string reasoning_accumulated;
    ov::genai::GenerationFinishReason finish =
        ov::genai::GenerationFinishReason::NONE;
    // Separate reasoning from visible content only when a client consumes it;
    // otherwise the raw stream (including markers) is forwarded unchanged.
    const bool split_stream =
        m_reasoning.enabled && opts.on_reasoning != nullptr;
    ReasoningStreamSplitter splitter(m_reasoning.expect_open_tag,
                                     m_reasoning.open_tag,
                                     m_reasoning.close_tag);
    auto on_word = [&](std::string word) -> ov::genai::CallbackTypeVariant {
        if (split_stream) {
            const bool ok = splitter.feed(
                word,
                [&accumulated, &opts](std::string w) {
                    accumulated += w;
                    if (opts.on_text) {
                        return opts.on_text(std::move(w));
                    }
                    return true;
                },
                [&reasoning_accumulated, &opts](std::string w) {
                    reasoning_accumulated += w;
                    if (opts.on_reasoning) {
                        return opts.on_reasoning(std::move(w));
                    }
                    return true;
                });
            if (!ok) {
                return ov::genai::StreamingStatus::CANCEL;
            }
            return ov::genai::StreamingStatus::RUNNING;
        }
        accumulated += word;
        if (opts.on_text && !opts.on_text(std::move(word))) {
            return ov::genai::StreamingStatus::CANCEL;
        }
        return ov::genai::StreamingStatus::RUNNING;
    };
    auto streamer =
        std::make_shared<ov::genai::TextStreamer>(m_tokenizer, on_word);

    bool aborted = false;
    bool prefill_done = false;
    auto prefill_end = std::chrono::steady_clock::now();
    std::size_t tokens = 0;
    const auto gen_start = std::chrono::steady_clock::now();
    while (handle->get_status() == ov::genai::GenerationStatus::RUNNING ||
           handle->can_read()) {
        ov::genai::GenerationOutputs outputs = handle->read();
        for (auto& [rid, out] : outputs) {
            if (out.finish_reason != ov::genai::GenerationFinishReason::NONE) {
                finish = out.finish_reason;
            }
            for (const int64_t tid : out.generated_ids) {
                if (!prefill_done) {
                    // First generated token: prefill of this request just
                    // completed. Credit its prompt tokens to the prefill
                    // throughput of the current snapshot window, but only if
                    // the request already appeared in a previous snapshot.
                    prefill_done = true;
                    prefill_end = std::chrono::steady_clock::now();
                    std::lock_guard<std::mutex> lock(m_metrics_mutex);
                    RequestMetrics& m = m_requests.at(req_id);
                    if (m.has_baseline) {
                        m_prefill_tokens_in_window += m.prompt_tokens;
                    }
                }
                ++tokens;
                const ov::genai::StreamingStatus st = streamer->write(tid);
                if (st == ov::genai::StreamingStatus::CANCEL) {
                    handle->cancel();
                    aborted = true;
                    break;
                }
                if (st != ov::genai::StreamingStatus::RUNNING) {
                    handle->stop();
                    break;
                }
            }
            if (aborted) {
                break;
            }
        }
        {
            std::lock_guard<std::mutex> lock(m_metrics_mutex);
            m_requests[req_id].tokens = tokens;
        }
        if (aborted) {
            break;
        }
    }
    streamer->end();

    if (split_stream && !aborted) {
        // Deliver any tail text held back for a possibly-marker-prefix.
        splitter.flush(
            [&accumulated, &opts](std::string w) {
                accumulated += w;
                if (opts.on_text) {
                    return opts.on_text(std::move(w));
                }
                return true;
            },
            [&reasoning_accumulated, &opts](std::string w) {
                reasoning_accumulated += w;
                if (opts.on_reasoning) {
                    return opts.on_reasoning(std::move(w));
                }
                return true;
            });
    } else if (!split_stream && !aborted) {
        // Non-streaming: split the complete output once so reasoning_content is
        // still populated even though nothing was streamed.
        if (m_reasoning.enabled) {
            ReasoningStreamSplitter full(m_reasoning.expect_open_tag,
                                         m_reasoning.open_tag,
                                         m_reasoning.close_tag);
            std::string content;
            std::string reasoning;
            full.feed(accumulated,
                      [&content](std::string w) {
                          content += w;
                          return true;
                      },
                      [&reasoning](std::string w) {
                          reasoning += w;
                          return true;
                      });
            full.flush(
                [&content](std::string w) {
                    content += w;
                    return true;
                },
                [&reasoning](std::string w) {
                    reasoning += w;
                    return true;
                });
            accumulated = std::move(content);
            reasoning_accumulated = std::move(reasoning);
        }
    }

    const auto gen_end = std::chrono::steady_clock::now();
    const double gen_s =
        std::chrono::duration<double>(gen_end - gen_start).count();
    {
        std::lock_guard<std::mutex> lock(m_metrics_mutex);
        m_requests.erase(req_id);
    }
    std::cerr << "[text model '" << m_id << "'] request " << req_id
              << " finished in " << gen_s << " s (" << tokens << " tokens";
    if (prefill_done && tokens > 0) {
        const double prefill_s =
            std::chrono::duration<double>(prefill_end - gen_start).count();
        const double decode_s =
            std::chrono::duration<double>(gen_end - prefill_end).count();
        if (prefill_s > 0.0 && decode_s > 0.0) {
            std::cerr << ", avg " << (static_cast<double>(prompt_tokens) / prefill_s)
                      << " prefill tok/s, avg "
                      << (static_cast<double>(tokens) / decode_s)
                      << " decode tok/s";
        }
    }
    std::cerr << ")" << std::endl;

    TextResult result;
    result.text = std::move(accumulated);
    result.reasoning_content = std::move(reasoning_accumulated);
    result.finish_reason = aborted ? "abort" : finish_reason_str(finish);
    result.prompt_tokens = prompt_tokens;
    result.completion_tokens = tokens;
    // Tool calls are only extracted when the request provided tools; otherwise
    // the model's ordinary answer text is kept untouched.
    if (opts.tools && !aborted && !result.text.empty()) {
        result.tool_calls = extract_tool_calls(result.text);
        if (!result.tool_calls.empty()) {
            result.finish_reason = "tool_calls";
            // Strip raw <tool_call> blocks from displayed content.
            const std::string open_m = "<tool_call>";
            const std::string close_m = "</tool_call>";
            std::size_t p = 0;
            while (p < result.text.size()) {
                const auto o = result.text.find(open_m, p);
                if (o == std::string::npos) break;
                const auto c = result.text.find(close_m, o);
                const auto end = (c != std::string::npos)
                                     ? c + close_m.size()
                                     : result.text.size();
                result.text.erase(o, end - o);
            }
        }
    }
    if (opts.on_done) {
        opts.on_done(result.text);
    }
    return result;
}

}  // namespace ovserver