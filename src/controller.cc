// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/controller.hpp"

#include <json/json.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "ovserver/wav2txt.hpp"
#include "ovserver/audio.hpp"
#include "ovserver/base64.hpp"
#include "ovserver/image.hpp"
#include "ovserver/image_decode.hpp"
#include "ovserver/manager.hpp"
#include "ovserver/txt2wav.hpp"
#include "ovserver/txt2img.hpp"
#include "ovserver/txt2txt.hpp"
#include "ovserver/txt2vid.hpp"

namespace ovserver {
namespace {

// Reads string member, throwing if missing or not a JSON string.
std::string getString(const Json::Value& obj, const char* key) {
    if (!obj.isMember(key)) {
        throw std::runtime_error(std::string("missing required field '") + key + "'");
    }
    if (!obj[key].isString()) {
        throw std::runtime_error(std::string("'") + key + "' must be a string");
    }
    return obj[key].asString();
}

bool getInt64(const Json::Value& obj, const char* key, int64_t& out) {
    if (!obj.isMember(key)) {
        return false;
    }
    if (!obj[key].isIntegral()) {
        throw std::runtime_error(std::string("'") + key + "' must be an integer");
    }
    out = obj[key].asInt64();
    return true;
}

// Converts a (sub)tree of the request JSON into the genai JsonContainer used by
// ChatHistory / chat templates.
ov::genai::JsonContainer json_to_genai(const Json::Value& v) {
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "";
    builder["commentStyle"] = "None";
    return ov::genai::JsonContainer::from_json_string(
        Json::writeString(builder, v));
}

// OpenAI "response_format" -> structured output constraint.
//   {"type": "json_schema", "json_schema": {"schema": {...}, ...}}
//   {"type": "json_object"}                          (any JSON object)
//   {"type": "text"} or absent                       (unconstrained)
std::optional<ov::genai::StructuredOutputConfig> parse_response_format(
    const Json::Value& body) {
    if (!body.isMember("response_format") || body["response_format"].isNull()) {
        return std::nullopt;
    }
    const Json::Value& rf = body["response_format"];
    if (!rf.isObject() || !rf.isMember("type") || !rf["type"].isString()) {
        throw std::runtime_error(
            "'response_format' must be an object with a string 'type'");
    }
    const std::string type = rf["type"].asString();
    if (type == "text") {
        return std::nullopt;
    }
    std::string schema;
    if (type == "json_object") {
        // OpenAI "json_object": valid JSON object output. Constrain with a
        // JSON Schema that accepts any object.
        schema = R"({"type": "object", "properties": {}, "required": []})";
    } else if (type == "json_schema") {
        if (!rf.isMember("json_schema") || !rf["json_schema"].isObject() ||
            !rf["json_schema"].isMember("schema") ||
            !rf["json_schema"]["schema"].isObject()) {
            throw std::runtime_error(
                "'response_format' of type \"json_schema\" requires a "
                "\"json_schema.schema\" JSON object");
        }
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "";
        builder["commentStyle"] = "None";
        schema = Json::writeString(builder, rf["json_schema"]["schema"]);
    } else {
        throw std::runtime_error(
            "'response_format.type' must be \"text\", \"json_object\" or "
            "\"json_schema\"");
    }
    return ov::genai::StructuredOutputConfig(
        ov::AnyMap{{ov::genai::json_schema(schema)}});
}

drogon::HttpResponsePtr json_response(const Json::Value& body,
                                      drogon::HttpStatusCode code =
                                          drogon::k200OK) {
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "";
    std::string payload = Json::writeString(builder, body);
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(code);
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(payload);
    return resp;
}

drogon::HttpResponsePtr error_response(std::string message,
                                       drogon::HttpStatusCode code) {
    Json::Value body;
    body["error"]["message"] = std::move(message);
    body["error"]["type"] = "invalid_request_error";
    body["error"]["param"] = Json::nullValue;
    body["error"]["code"] = Json::nullValue;
    return json_response(body, code);
}

drogon::HttpResponsePtr internal_error_response(const std::exception& e) {
    return error_response("Failed to generate image: " + std::string(e.what()),
                          drogon::k500InternalServerError);
}

drogon::HttpResponsePtr bytes_response(std::string body,
                                       drogon::ContentType type) {
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setContentTypeCode(type);
    resp->setBody(std::move(body));
    return resp;
}

// A bounded worker pool that serialises generation through one pipeline at a
// time. Generators are per-worker so a slow pipeline never stalls every client.
class WorkerPool {
public:
    explicit WorkerPool(std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            m_threads.emplace_back([this] { run(); });
        }
    }

    ~WorkerPool() {
        {
            std::lock_guard lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        for (auto& t : m_threads) {
            if (t.joinable()) t.join();
        }
    }

    template <typename F>
    auto enqueue(F&& f) -> std::future<decltype(f())> {
        using Ret = decltype(f());
        auto task = std::make_shared<std::packaged_task<Ret()>>(
            std::forward<F>(f));
        auto future = task->get_future();
        {
            std::lock_guard lock(m_mutex);
            if (m_stop) {
                throw std::runtime_error("worker pool shutting down");
            }
            m_queue.emplace([task] { (*task)(); });
        }
        m_cv.notify_one();
        return future;
    }

private:
    void run() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock lock(m_mutex);
                m_cv.wait(lock, [this] { return m_stop || !m_queue.empty(); });
                if (m_stop && m_queue.empty()) return;
                job = std::move(m_queue.front());
                m_queue.pop();
            }
            job();
        }
    }

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<std::function<void()>> m_queue;
    std::vector<std::thread> m_threads;
    bool m_stop = false;
};

WorkerPool& pool() {
    static auto num_threads = std::max<std::size_t>(
        1u, std::thread::hardware_concurrency());
    static WorkerPool p(num_threads);
    return p;
}

// Wraps a payload in a Server-Sent Events frame: "data: <json>\n\n".
std::string sse_message(const std::string& json) {
    std::string out = "data: ";
    out.reserve(json.size() + 8);
    out += json;
    out += "\n\n";
    return out;
}

// One streaming chunk of an OpenAI chat.completion.chunk object.
Json::Value chat_chunk(const std::string& id, const Json::Value& delta,
                       const std::string& finish_reason) {
    Json::Value chunk;
    chunk["id"] = id;
    chunk["object"] = "chat.completion.chunk";
    chunk["created"] = static_cast<int>(std::time(nullptr));
    chunk["model"] = Json::nullValue;  // filled in by caller
    Json::Value choice;
    choice["index"] = 0;
    // delta must always serialize as an object ({...}), never null, or clients
    // like vllm crash on choices[0]["delta"].get("content").
    choice["delta"] = delta.isObject() ? delta : Json::Value(Json::objectValue);
    if (!finish_reason.empty()) {
        choice["finish_reason"] = finish_reason;
    } else {
        choice["finish_reason"] = Json::nullValue;
    }
    Json::Value choices = Json::arrayValue;
    choices.append(choice);
    chunk["choices"] = choices;
    return chunk;
}

// Extracts the full conversation and image tensors from an OpenAI-style
// "messages" array. Every message is preserved (system/user/assistant/tool
// roles, assistant tool_calls, tool tool_call_id). Content may be a plain
// string or an array of typed parts ({type:"text",text:...} and
// {type:"image_url",image_url:{url:"..."}}); image_url parts are decoded into
// out.images and contribute no text (matching the pipeline's separate-image
// input path).
struct ParsedChat {
    std::vector<ov::genai::JsonContainer> messages;
    std::vector<ov::Tensor> images;
};

void registerHealth(drogon::HttpAppFramework& app) {
    app.registerHandler("/health",
                        [](const drogon::HttpRequestPtr& req,
                           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
                            (void)req;
                            Json::Value body;
                            body["status"] = "ok";
                            callback(json_response(body));
                        },
                        {drogon::Get});
}

void registerModels(drogon::HttpAppFramework& app) {
    app.registerHandler("/v1/models",
                        [](const drogon::HttpRequestPtr& req,
                           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
                            (void)req;
                            Json::Value arr = Json::arrayValue;
                            const auto& models = ModelManager::instance().all_images();
                            for (const auto& entry : models) {
                                Json::Value m;
                                m["id"] = entry.first;
                                m["object"] = "model";
                                m["created"] = 0;
                                m["owned_by"] = "openvino-genai";
                                arr.append(m);
                            }
                            const auto& text_models =
                                ModelManager::instance().all_text();
                            for (const auto& entry : text_models) {
                                Json::Value m;
                                m["id"] = entry.first;
                                m["object"] = "model";
                                m["created"] = 0;
                                m["owned_by"] = "openvino-genai";
                                arr.append(m);
                            }
                            const auto& video_models =
                                ModelManager::instance().all_video();
                            for (const auto& entry : video_models) {
                                Json::Value m;
                                m["id"] = entry.first;
                                m["object"] = "model";
                                m["created"] = 0;
                                m["owned_by"] = "openvino-genai";
                                arr.append(m);
                            }
                            const auto& asr_models =
                                ModelManager::instance().all_asr();
                            for (const auto& entry : asr_models) {
                                Json::Value m;
                                m["id"] = entry.first;
                                m["object"] = "model";
                                m["created"] = 0;
                                m["owned_by"] = "openvino-genai";
                                arr.append(m);
                            }
                            const auto& tts_models =
                                ModelManager::instance().all_tts();
                            for (const auto& entry : tts_models) {
                                Json::Value m;
                                m["id"] = entry.first;
                                m["object"] = "model";
                                m["created"] = 0;
                                m["owned_by"] = "openvino-genai";
                                arr.append(m);
                            }
                            Json::Value body;
                            body["object"] = "list";
                            body["data"] = arr;
                            callback(json_response(body));
                        },
                        {drogon::Get});
}

void registerImageGenerations(drogon::HttpAppFramework& app) {
    app.registerHandler(
        "/v1/images/generations",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            Json::CharReaderBuilder reader;
            reader["failIfExtra"] = false;
            auto body_view = req->getBody();
            std::istringstream stream(std::string(body_view.begin(), body_view.end()));
            Json::Value body;
            std::string errs;
            if (!Json::parseFromStream(reader, stream, &body, &errs) ||
                !body.isObject()) {
                callback(error_response("Invalid JSON request body: " + errs,
                                        drogon::k400BadRequest));
                return;
            }

            try {
                const std::string prompt = getString(body, "prompt");
                ImageGenerationModel* model = nullptr;

                if (!body.isMember("model")) {
                    auto& models = ModelManager::instance().all_images();
                    if (models.size() == 1) {
                        model = models.begin()->second.get();
                    }
                } else if (body["model"].isString()) {
                    model = ModelManager::instance().get_image(body["model"].asString());
                }
                if (!model) {
                    callback(error_response(
                        "The requested model is not available. Start the server "
                        "with --txt2img or provide a valid 'model' field.",
                        drogon::k404NotFound));
                    return;
                }

                ImageGenerateOptions opts;
                opts.prompt = prompt;
                if (body.isMember("negative_prompt") && body["negative_prompt"].isString()) {
                    opts.negative_prompt = body["negative_prompt"].asString();
                }
                if (body.isMember("guidance_scale") && body["guidance_scale"].isNumeric()) {
                    opts.guidance_scale =
                        static_cast<float>(body["guidance_scale"].asDouble());
                }
                if (body.isMember("steps") && body["steps"].isIntegral()) {
                    opts.num_inference_steps = static_cast<std::size_t>(
                        std::max<int64_t>(1, body["steps"].asInt64()));
                }
                if (body.isMember("seed") && body["seed"].isIntegral()) {
                    opts.rng_seed = static_cast<std::size_t>(
                        std::max<int64_t>(0, body["seed"].asInt64()));
                }
                if (body.isMember("n") && body["n"].isIntegral()) {
                    opts.num_images = static_cast<std::size_t>(
                        std::clamp<int64_t>(body["n"].asInt64(), 1, 10));
                }

                // Accept either a "size" string ("1024x1024", OpenAI-style) or
                // explicit width/height. Dimensions are snapped to a multiple of
                // 16, which Qwen-Image requires.
                auto snap16 = [](int64_t v) {
                    const int64_t step = 16;
                    const int64_t lo = 64, hi = 4096;
                    v = std::max(lo, std::min(hi, (v + step / 2) / step * step));
                    return v;
                };
                if (body.isMember("size") && body["size"].isString()) {
                    std::string size = body["size"].asString();
                    auto x = size.find('x');
                    if (x == std::string::npos) {
                        throw std::runtime_error(
                            "'size' must be formatted as \"WxH\", e.g. \"1024x1024\"");
                    }
                    opts.width = snap16(std::stoll(size.substr(0, x)));
                    opts.height = snap16(std::stoll(size.substr(x + 1)));
                } else {
                    int64_t w = 0, h = 0;
                    getInt64(body, "width", w);
                    getInt64(body, "height", h);
                    if (w > 0) opts.width = snap16(w);
                    if (h > 0) opts.height = snap16(h);
                }

                try {
                    // OpenAI "response_format": "b64_json" (default) returns
                    // base64 in the response; "url" writes a PNG and returns its
                    // file path, which requires an "output_dir" to write into.
                    // "output_format" only supports "png".
                    const bool has_output_dir = body.isMember("output_dir") &&
                                               body["output_dir"].isString();
                    const bool want_url =
                        body.isMember("response_format") &&
                        body["response_format"].isString() &&
                        body["response_format"].asString() == "url";
                    if (body.isMember("response_format") &&
                        body["response_format"].isString() &&
                        !body["response_format"].isNull() &&
                        body["response_format"].asString() != "b64_json" &&
                        body["response_format"].asString() != "url") {
                        throw std::runtime_error(
                            "'response_format' must be \"b64_json\" or \"url\"");
                    }
                    if (body.isMember("output_format") &&
                        body["output_format"].isString() &&
                        body["output_format"].asString() != "png") {
                        throw std::runtime_error(
                            "'output_format' only supports \"png\"");
                    }
                    const bool save = want_url || has_output_dir;
                    std::string output_dir;
                    if (save) {
                        if (!has_output_dir) {
                            throw std::runtime_error(
                                "'response_format' \"url\" requires an "
                                "'output_dir' to write files into");
                        }
                        output_dir = body["output_dir"].asString();
                    }

                    // Defer the actual generation to a worker thread so the
                    // event loop is never blocked; the response callback is
                    // invoked from the worker once inference completes.
                    pool().enqueue([model, opts, save, output_dir, callback] {
                        drogon::HttpResponsePtr resp;
                        try {
                            auto result = model->generate(opts);

                            Json::Value data = Json::arrayValue;
                            for (std::size_t i = 0; i < result.size(); ++i) {
                                const auto& img = result[i];
                                std::string b64 = png_base64(
                                    img.data.data(), img.height, img.width,
                                    img.channels);

                                Json::Value item;
                                if (!save) {
                                    item["b64_json"] = b64;
                                    item["url"] = Json::nullValue;
                                } else {
                                    std::string path =
                                        output_dir + "/" + "ovserver_" +
                                        std::to_string(std::time(nullptr)) +
                                        "_" + std::to_string(i) + ".png";
                                    if (!png_write(path, img.data.data(),
                                                   img.height, img.width,
                                                   img.channels)) {
                                        throw std::runtime_error(
                                            "cannot write PNG to " + path);
                                    }
                                    item["b64_json"] = Json::nullValue;
                                    item["url"] = path;
                                }
                                data.append(item);
                            }
                            Json::Value out;
                            out["created"] = static_cast<int>(std::time(nullptr));
                            out["data"] = data;
                            resp = json_response(out);
                        } catch (const std::exception& e) {
                            resp = internal_error_response(e);
                        }
                        callback(resp);
                    });
                } catch (const std::exception& e) {
                    callback(internal_error_response(e));
                }
            } catch (const std::exception& e) {
                callback(error_response(e.what(), drogon::k400BadRequest));
            }
        },
        {drogon::Post});
}

// Parses the message array into a full OpenAI-shaped conversation history.
ParsedChat parse_messages(const Json::Value& messages) {
    ParsedChat out;
    if (!messages.isArray()) {
        throw std::runtime_error("'messages' must be an array");
    }
    for (const auto& msg : messages) {
        if (!msg.isMember("role") || !msg["role"].isString()) {
            throw std::runtime_error("each message requires a 'role'");
        }
        const std::string role = msg["role"].asString();
        if (role != "system" && role != "user" && role != "assistant" &&
            role != "tool") {
            throw std::runtime_error(
                "unsupported message role '" + role + "'");
        }

        Json::Value gen_msg;
        gen_msg["role"] = role;
        // "content": string | array of typed parts | null (assistant tool_calls).
        if (msg.isMember("content") && !msg["content"].isNull()) {
            const Json::Value& content = msg["content"];
            std::string text;
            if (content.isString()) {
                text = content.asString();
            } else if (content.isArray()) {
                for (const auto& part : content) {
                    if (!part.isMember("type") || !part["type"].isString()) {
                        continue;
                    }
                    const std::string type = part["type"].asString();
                    if (type == "text") {
                        text += part.get("text", Json::Value()).asString();
                    } else if (type == "image_url") {
                        if (!part.isMember("image_url")) {
                            continue;
                        }
                        const Json::Value& iu = part["image_url"];
                        if (iu.isMember("url") && iu["url"].isString()) {
                            out.images.push_back(
                                decode_image_base64(iu["url"].asString()));
                        }
                    }
                }
            } else {
                throw std::runtime_error(
                    "message 'content' must be a string or an array of parts");
            }
            gen_msg["content"] = text;
        } else {
            gen_msg["content"] = "";
        }

        if (role == "assistant") {
            // Preserve any tool_calls this assistant turn contains so multi-turn
            // function-calling conversations survive through the chat template.
            if (msg.isMember("tool_calls")) {
                if (!msg["tool_calls"].isArray()) {
                    throw std::runtime_error(
                        "assistant 'tool_calls' must be an array");
                }
                gen_msg["tool_calls"] = msg["tool_calls"];
            }
        } else if (role == "tool") {
            if (!msg.isMember("tool_call_id") ||
                !msg["tool_call_id"].isString()) {
                throw std::runtime_error(
                    "tool message requires a string 'tool_call_id'");
            }
            gen_msg["tool_call_id"] = msg["tool_call_id"].asString();
        }

        out.messages.push_back(json_to_genai(gen_msg));
    }
    if (out.messages.empty()) {
        throw std::runtime_error("'messages' must not be empty");
    }
    return out;
}

void registerChatCompletions(drogon::HttpAppFramework& app) {
    app.registerHandler(
        "/v1/chat/completions",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            Json::CharReaderBuilder reader;
            reader["failIfExtra"] = false;
            auto body_view = req->getBody();
            std::istringstream stream(std::string(body_view.begin(), body_view.end()));
            Json::Value body;
            std::string errs;
            if (!Json::parseFromStream(reader, stream, &body, &errs) ||
                !body.isObject()) {
                callback(error_response("Invalid JSON request body: " + errs,
                                        drogon::k400BadRequest));
                return;
            }

            try {
                TextGenerationModel* model = nullptr;
                if (body.isMember("model") && body["model"].isString()) {
                    model = ModelManager::instance().get_text(body["model"].asString());
                }
                if (!model) {
                    auto& text_models = ModelManager::instance().all_text();
                    if (text_models.size() == 1) {
                        model = text_models.begin()->second.get();
                    }
                }
                if (!model) {
                    callback(error_response(
                        "The requested text model is not available. Start the "
                        "server with --txt2txt or provide a valid 'model' field.",
                        drogon::k404NotFound));
                    return;
                }

                ParsedChat chat = parse_messages(body["messages"]);
                TextGenerateOptions opts;
                opts.chat_messages = std::move(chat.messages);
                opts.images = std::move(chat.images);
                if (body.isMember("max_tokens") && body["max_tokens"].isIntegral()) {
                    opts.max_new_tokens = static_cast<std::size_t>(
                        std::max<int64_t>(1, body["max_tokens"].asInt64()));
                }
                // OpenAI renamed max_tokens to max_completion_tokens for recent
                // models; at most one may be supplied.
                if (body.isMember("max_completion_tokens") &&
                    body["max_completion_tokens"].isIntegral()) {
                    if (body.isMember("max_tokens") &&
                        body["max_tokens"].isIntegral()) {
                        throw std::runtime_error(
                            "provide either 'max_tokens' or "
                            "'max_completion_tokens', not both");
                    }
                    opts.max_new_tokens = static_cast<std::size_t>(std::max<int64_t>(
                        1, body["max_completion_tokens"].asInt64()));
                }
                if (body.isMember("temperature") && body["temperature"].isNumeric()) {
                    opts.temperature =
                        static_cast<float>(body["temperature"].asDouble());
                }
                if (body.isMember("top_p") && body["top_p"].isNumeric()) {
                    opts.top_p = static_cast<float>(body["top_p"].asDouble());
                }
                if (body.isMember("top_k") && body["top_k"].isIntegral()) {
                    opts.top_k = static_cast<std::size_t>(
                        std::max<int64_t>(1, body["top_k"].asInt64()));
                }
                // "stop": single string or array of strings.
                if (body.isMember("stop") && !body["stop"].isNull()) {
                    Json::Value stop_set = Json::arrayValue;
                    if (body["stop"].isString()) {
                        stop_set.append(body["stop"]);
                    } else if (body["stop"].isArray()) {
                        for (const auto& s : body["stop"]) {
                            if (!s.isString()) {
                                throw std::runtime_error(
                                    "'stop' must be a string or an array of "
                                    "strings");
                            }
                            stop_set.append(s);
                        }
                    } else {
                        throw std::runtime_error(
                            "'stop' must be a string or an array of strings");
                    }
                    for (const auto& s : stop_set) {
                        if (!s.asString().empty()) {
                            opts.stop_strings.insert(s.asString());
                        }
                    }
                }
                // OpenAI penalties are defined in [-2, 2].
                if (body.isMember("frequency_penalty") &&
                    body["frequency_penalty"].isNumeric()) {
                    opts.frequency_penalty = std::clamp(
                        static_cast<float>(body["frequency_penalty"].asDouble()),
                        -2.0f, 2.0f);
                }
                if (body.isMember("presence_penalty") &&
                    body["presence_penalty"].isNumeric()) {
                    opts.presence_penalty = std::clamp(
                        static_cast<float>(body["presence_penalty"].asDouble()),
                        -2.0f, 2.0f);
                }
                if (body.isMember("seed") && body["seed"].isIntegral()) {
                    opts.rng_seed = static_cast<std::size_t>(
                        std::max<int64_t>(0, body["seed"].asInt64()));
                }
                // OpenAI "tools" (function calling). Definitions are injected
                // into the model's own chat template; "tool_choice" only
                // controls whether they are injected at all (and optionally
                // narrows to one function):
                //   "none"    -> tools are not injected; no tool parsing.
                //   "auto"    -> model decides (default).
                //   "required"-> model is encouraged to call a tool.
                //   {function:{name}} -> only that function is offered.
                if (body.isMember("tools")) {
                    if (!body["tools"].isArray()) {
                        throw std::runtime_error("'tools' must be an array");
                    }
                    bool none = false;
                    std::optional<std::string> forced_name;
                    if (body.isMember("tool_choice") &&
                        !body["tool_choice"].isNull()) {
                        const Json::Value& tc = body["tool_choice"];
                        if (tc.isString()) {
                            const std::string s = tc.asString();
                            if (s == "none") {
                                none = true;
                            } else if (s != "auto" && s != "required") {
                                throw std::runtime_error(
                                    "'tool_choice' must be \"none\", \"auto\", "
                                    "\"required\" or a function object");
                            }
                        } else if (tc.isObject()) {
                            if (!tc.isMember("function") ||
                                !tc["function"].isObject() ||
                                !tc["function"].isMember("name") ||
                                !tc["function"]["name"].isString()) {
                                throw std::runtime_error(
                                    "'tool_choice' function object must have "
                                    "function.name");
                            }
                            forced_name = tc["function"]["name"].asString();
                        } else {
                            throw std::runtime_error(
                                "'tool_choice' must be a string or a function "
                                "object");
                        }
                    }
                    if (!none && !body["tools"].empty()) {
                        Json::Value tools = body["tools"];
                        if (forced_name) {
                            Json::Value filtered = Json::arrayValue;
                            for (auto& t : tools) {
                                if (t.isMember("function") &&
                                    t["function"].isMember("name") &&
                                    t["function"]["name"].isString() &&
                                    t["function"]["name"].asString() ==
                                        *forced_name) {
                                    filtered.append(t);
                                }
                            }
                            if (filtered.empty()) {
                                throw std::runtime_error(
                                    "tool_choice names a function that is not "
                                    "in 'tools'");
                            }
                            tools = filtered;
                        }
                        opts.tools = json_to_genai(tools);
                    }
                }
                // Number of independent completions to sample. Streaming several
                // concurrent generations in one SSE stream is not supported.
                std::size_t n_choices = 1;
                if (body.isMember("n") && body["n"].isIntegral()) {
                    n_choices = static_cast<std::size_t>(
                        std::clamp<int64_t>(body["n"].asInt64(), 1, 10));
                }
                // OpenAI "response_format" -> structured output (xgrammar
                // backend): constrain output to a JSON schema.
                opts.structured_output = parse_response_format(body);
                // stream_options.include_usage: emit a final usage-only chunk.
                const bool stream_usage = body.isMember("stream_options") &&
                                          body["stream_options"].isObject() &&
                                          body["stream_options"]
                                              .isMember("include_usage") &&
                                          body["stream_options"]["include_usage"]
                                              .isBool() &&
                                          body["stream_options"]["include_usage"]
                                              .asBool();

                const std::string model_name = model->id();
                const std::string req_id = "chatcmpl-" + std::to_string(std::time(nullptr));

                // Streaming several concurrent generations in one SSE stream is
                // not supported.
                const bool do_stream = body.isMember("stream") &&
                                       body["stream"].isBool() &&
                                       body["stream"].asBool();
                if (do_stream && n_choices > 1) {
                    throw std::runtime_error(
                        "'stream' with 'n' > 1 is not supported");
                }

                if (do_stream) {
                    // SSE: chunked transfer. Generation runs on a worker thread;
                    // each decoded fragment is pushed out as an SSE frame as the
                    // streamer produces it. Headers are set once here. The async
                    // stream is owned by a shared struct so both the genai
                    // streamer callback (genai thread) and the worker thread that
                    // drives generation can resolve and send on it safely.
                    struct StreamState {
                        std::shared_ptr<drogon::ResponseStream> stream;
                        std::mutex mutex;
                    };
                    auto state = std::make_shared<StreamState>();
                    std::shared_ptr<const std::string> shared_id =
                        std::make_shared<const std::string>(req_id);
                    std::shared_ptr<const std::string> shared_model =
                        std::make_shared<const std::string>(model_name);

                    drogon::HttpResponsePtr resp =
                        drogon::HttpResponse::newAsyncStreamResponse(
                            [state, model, opts = std::move(opts), shared_id,
                             shared_model, req_id,
                             stream_usage](drogon::ResponseStreamPtr stream) {
                                state->stream.reset(stream.release());

                                // Sends one chunk carrying `delta`. Returns
                                // false if the client is gone.
                                auto send_delta =
                                    [state, shared_id, shared_model](
                                        Json::Value delta) {
                                        std::lock_guard lock(state->mutex);
                                        auto s = state->stream;
                                        if (!s) {
                                            return false;
                                        }
                                        Json::Value chunk = chat_chunk(
                                            *shared_id, delta, "");
                                        chunk["model"] = *shared_model;
                                        Json::StreamWriterBuilder b;
                                        b["indentation"] = "";
                                        std::string payload =
                                            Json::writeString(b, chunk);
                                        const bool ok =
                                            s->send(sse_message(payload));
                                        if (!ok) {
                                            state->stream.reset();
                                        }
                                        return ok;
                                    };

                                TextGenerateOptions gen = opts;
                                auto toolbuf = std::make_shared<std::string>();
                                auto in_tool = std::make_shared<bool>(false);
                                const std::string tool_open = "<tool_call>";
                                const std::string tool_close = "</tool_call>";
                                gen.on_text =
                                    [send_delta, toolbuf, in_tool, tool_open,
                                     tool_close](std::string word) {
                                        if (*in_tool) {
                                            *toolbuf += word;
                                            const auto c = toolbuf->find(
                                                tool_close);
                                            if (c == std::string::npos) {
                                                return true;
                                            }
                                            toolbuf->erase(0,
                                                           c + tool_close.size());
                                            *in_tool = false;
                                            if (toolbuf->empty()) {
                                                return true;
                                            }
                                            word = std::move(*toolbuf);
                                            toolbuf->clear();
                                        } else {
                                            const auto o = word.find(tool_open);
                                            if (o != std::string::npos) {
                                                if (o > 0) {
                                                    Json::Value delta;
                                                    delta["content"] =
                                                        word.substr(0, o);
                                                    if (!send_delta(
                                                            std::move(delta))) {
                                                        return false;
                                                    }
                                                }
                                                *in_tool = true;
                                                toolbuf->clear();
                                                toolbuf->append(word.substr(o));
                                                return true;
                                            }
                                        }
                                        if (word.empty()) {
                                            return true;
                                        }
                                        Json::Value delta;
                                        delta["content"] = std::move(word);
                                        return send_delta(std::move(delta));
                                    };
                                gen.on_reasoning =
                                    [send_delta](std::string word) {
                                        Json::Value delta;
                                        delta["reasoning_content"] =
                                            std::move(word);
                                        return send_delta(std::move(delta));
                                    };

                                // Sends the trailing finish_reason (+ usage when
                                // requested) and [DONE], then closes the stream.
                                auto finish = [state, shared_id, shared_model,
                                               stream_usage](
                                                  const TextResult& r) {
                                    std::lock_guard lock(state->mutex);
                                    if (!state->stream) {
                                        return;
                                    }
                                    Json::StreamWriterBuilder b;
                                    b["indentation"] = "";
                                    Json::Value delta;
                                    if (!r.tool_calls.empty()) {
                                        // Function calls are only fully known
                                        // once generation ends, so they arrive
                                        // with arguments pre-assembled in the
                                        // final chunk alongside
                                        // finish_reason="tool_calls".
                                        Json::Value tcs = Json::arrayValue;
                                        for (std::size_t i = 0;
                                             i < r.tool_calls.size(); ++i) {
                                            const auto& tc = r.tool_calls[i];
                                            Json::Value item;
                                            item["index"] = static_cast<int>(i);
                                            item["id"] = tc.id;
                                            item["type"] = "function";
                                            item["function"]["name"] = tc.name;
                                            item["function"]["arguments"] =
                                                tc.arguments;
                                            tcs.append(item);
                                        }
                                        delta["content"] = Json::nullValue;
                                        delta["tool_calls"] = tcs;
                                    }
                                    Json::Value chunk = chat_chunk(
                                        *shared_id, delta,
                                        r.finish_reason.empty()
                                            ? "stop"
                                            : r.finish_reason);
                                    chunk["model"] = *shared_model;
                                    state->stream->send(sse_message(
                                        Json::writeString(b, chunk)));
                                    if (stream_usage) {
                                        Json::Value usage;
                                        usage["prompt_tokens"] =
                                            static_cast<int64_t>(r.prompt_tokens);
                                        usage["completion_tokens"] =
                                            static_cast<int64_t>(
                                                r.completion_tokens);
                                        usage["total_tokens"] =
                                            static_cast<int64_t>(
                                                r.prompt_tokens +
                                                r.completion_tokens);
                                        Json::Value frame;
                                        frame["id"] = *shared_id;
                                        frame["object"] =
                                            "chat.completion.chunk";
                                        frame["created"] = static_cast<int>(
                                            std::time(nullptr));
                                        frame["model"] = *shared_model;
                                        frame["choices"] = Json::arrayValue;
                                        frame["usage"] = usage;
                                        state->stream->send(sse_message(
                                            Json::writeString(b, frame)));
                                    }
                                    state->stream->send(
                                        sse_message("[DONE]"));
                                    state->stream->close();
                                    state->stream.reset();
                                };

                                auto worker = [state, gen, model, finish] {
                                    TextResult result;
                                    try {
                                        result = model->generate(gen);
                                    } catch (const std::exception& e) {
                                        std::cerr << "text streaming error: "
                                                  << e.what() << std::endl;
                                    }
                                    finish(result);
                                };
                                try {
                                    pool().enqueue(worker);
                                } catch (const std::exception& e) {
                                    finish(TextResult{});
                                }
                            },
                            true /*disableKickoffTimeout*/);
                    resp->setContentTypeString("text/event-stream");
                    resp->addHeader("Cache-Control", "no-cache");
                    resp->addHeader("Connection", "keep-alive");
                    callback(resp);
                    return;
                }

                // Non-streaming: full response in one shot. With "n" > 1 the prompt is
                // sampled multiple times and the seeds are varied per completed
                // so distinct generations come back (a user-supplied seed is
                // still reproduced per choice via a fixed offset).
                pool().enqueue([model, opts, model_name, req_id, callback,
                                n_choices] {
                    drogon::HttpResponsePtr resp;
                    try {
                        Json::Value choices = Json::arrayValue;
                        Json::Value usage;
                        int64_t p_tokens = 0, c_tokens = 0;
                        const std::size_t seed_base =
                            opts.rng_seed
                                ? *opts.rng_seed
                                : static_cast<std::size_t>(std::time(nullptr));
                        for (std::size_t i = 0; i < n_choices; ++i) {
                            TextGenerateOptions gen = opts;
                            if (n_choices > 1) {
                                gen.rng_seed = seed_base + i;
                            }
                            TextResult result = model->generate(gen);
                            Json::Value msg;
                            msg["role"] = "assistant";
                            if (!result.tool_calls.empty()) {
                                // OpenAI: content is null when tool_calls are
                                // returned; arguments are a JSON string.
                                msg["content"] = Json::nullValue;
                                Json::Value tcs = Json::arrayValue;
                                for (const auto& tc : result.tool_calls) {
                                    Json::Value item;
                                    item["id"] = tc.id;
                                    item["type"] = "function";
                                    item["function"]["name"] = tc.name;
                                    item["function"]["arguments"] = tc.arguments;
                                    tcs.append(item);
                                }
                                msg["tool_calls"] = tcs;
                            } else {
                                msg["content"] = result.text;
                            }
                            if (!result.reasoning_content.empty()) {
                                msg["reasoning_content"] =
                                    result.reasoning_content;
                            }
                            Json::Value choice;
                            choice["index"] = static_cast<int>(i);
                            choice["message"] = msg;
                            choice["finish_reason"] =
                                result.finish_reason.empty()
                                    ? Json::Value("stop")
                                    : Json::Value(result.finish_reason);
                            choices.append(choice);
                            p_tokens += static_cast<int64_t>(result.prompt_tokens);
                            c_tokens +=
                                static_cast<int64_t>(result.completion_tokens);
                        }
                        Json::Value out;
                        out["id"] = req_id;
                        out["object"] = "chat.completion";
                        out["created"] = static_cast<int>(std::time(nullptr));
                        out["model"] = model_name;
                        out["choices"] = choices;
                        usage["prompt_tokens"] = p_tokens;
                        usage["completion_tokens"] = c_tokens;
                        usage["total_tokens"] = p_tokens + c_tokens;
                        out["usage"] = usage;
                        resp = json_response(out);
                    } catch (const std::exception& e) {
                        resp = error_response(
                            "Failed to generate chat completion: " +
                                std::string(e.what()),
                            drogon::k500InternalServerError);
                    }
                    callback(resp);
                });
            } catch (const std::exception& e) {
                callback(error_response(e.what(), drogon::k400BadRequest));
            }
        },
        {drogon::Post});
}

// ---------------------------------------------------------------------------
// Video generation job API. Implements the OpenAI Videos library the way vLLM
// and SGLang do: POST /v1/videos creates an asynchronous job whose id is a
// random 128-bit UUID; the same UUID names the stored MP4 file (<uuid>.mp4)
// under <temp-dir>/videos. GET /v1/videos/{id} polls status, GET
// /v1/videos/{id}/content streams the finished file. No list endpoint: ids are
// returned to the client and never enumerate server-side. The synchronous
// /v1/video/generations endpoint shares the same storage and returns the OpenAI
// video-generations response shape.
// ---------------------------------------------------------------------------

// Standard RFC 4122 UUID v4 (122 random bits, 128 in the string). Large enough
// that ids cannot be brute-forced.
std::string random_uuid() {
    thread_local std::mt19937_64 rng([] {
        std::random_device rd;
        const auto now = std::chrono::high_resolution_clock::now()
                             .time_since_epoch()
                             .count();
        std::seed_seq seed{rd(), rd(), rd(), rd(),
                           static_cast<unsigned>(static_cast<std::uint64_t>(now)),
                           static_cast<unsigned>(static_cast<std::uint64_t>(now) >> 32)};
        return std::mt19937_64(seed);
    }());
    const std::uint64_t hi = rng();
    const std::uint64_t lo = rng();
    std::array<std::uint8_t, 16> bytes;
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<std::uint8_t>(hi >> (8 * (7 - i)));
        bytes[8 + i] = static_cast<std::uint8_t>(lo >> (8 * (7 - i)));
    }
    bytes[6] = static_cast<std::uint8_t>((bytes[6] & 0x0F) | 0x40);  // version 4
    bytes[8] = static_cast<std::uint8_t>((bytes[8] & 0x3F) | 0x80);  // variant
    static const char* const hex = "0123456789abcdef";
    std::string out;
    out.reserve(36);
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) out += '-';
        out += hex[bytes[i] >> 4];
        out += hex[bytes[i] & 0x0F];
    }
    return out;
}

enum class VideoJobStatus { queued, running, completed, failed };

// One asynchronous video generation job. Jobs are never enumerated (there is no
// list endpoint), so records simply live until the server exits.
struct VideoJob {
    std::string id;
    std::string model_id;
    std::string prompt;
    VideoJobStatus status = VideoJobStatus::queued;
    std::time_t created_at = std::time(nullptr);
    std::time_t completed_at = 0;
    double inference_time_s = 0.0;
    std::string error;
    std::vector<std::string> file_paths;  // absolute paths to stored MP4 files
    std::string size;                     // requested "WxH", may be empty
    std::string seconds;                  // requested duration, may be empty
    std::size_t num_outputs = 1;
};

std::mutex g_video_jobs_mutex;
std::unordered_map<std::string, VideoJob> g_video_jobs;
// Absolute path to the directory holding generated videos (<dir>/videos).
std::filesystem::path g_video_storage_dir;

// Request fields shared by the async job API and the synchronous generations
// endpoint.
struct ParsedVideoRequest {
    VideoGenerationModel* model = nullptr;
    VideoGenerateOptions opts;
    std::size_t num_videos = 1;
    std::string size;
    std::string seconds;
};

// Parses a vLLM/SGLang-style video request body into VideoGenerateOptions.
// Throws std::runtime_error (mapped to HTTP 400) on malformed input. A null
// model means the caller should return HTTP 404.
ParsedVideoRequest parse_video_request(const Json::Value& body) {
    ParsedVideoRequest p;
    p.opts.prompt = getString(body, "prompt");

    if (!body.isMember("model")) {
        auto& models = ModelManager::instance().all_video();
        if (models.size() == 1) {
            p.model = models.begin()->second.get();
        }
    } else if (body["model"].isString()) {
        p.model = ModelManager::instance().get_video(body["model"].asString());
    }

    if (body.isMember("negative_prompt") && body["negative_prompt"].isString()) {
        p.opts.negative_prompt = body["negative_prompt"].asString();
    }
    if (body.isMember("guidance_scale") && body["guidance_scale"].isNumeric()) {
        p.opts.guidance_scale =
            static_cast<float>(body["guidance_scale"].asDouble());
    }
    int64_t steps = 0;
    if (body.isMember("num_inference_steps") &&
        body["num_inference_steps"].isIntegral()) {
        steps = body["num_inference_steps"].asInt64();
    } else if (body.isMember("steps") && body["steps"].isIntegral()) {
        steps = body["steps"].asInt64();
    }
    if (steps > 0) {
        p.opts.num_inference_steps = static_cast<std::size_t>(steps);
    }
    if (body.isMember("seed") && body["seed"].isIntegral()) {
        p.opts.rng_seed = static_cast<std::size_t>(
            std::max<int64_t>(0, body["seed"].asInt64()));
    }
    if (body.isMember("n") && body["n"].isIntegral()) {
        p.num_videos = static_cast<std::size_t>(
            std::clamp<int64_t>(body["n"].asInt64(), 1, 4));
        p.opts.num_videos = p.num_videos;
    }
    if (body.isMember("num_outputs_per_prompt") &&
        body["num_outputs_per_prompt"].isIntegral()) {
        p.num_videos = static_cast<std::size_t>(
            std::clamp<int64_t>(body["num_outputs_per_prompt"].asInt64(), 1, 4));
        p.opts.num_videos = p.num_videos;
    }

    double fps = 25.0;
    if (body.isMember("fps") && body["fps"].isNumeric()) {
        fps = body["fps"].asDouble();
        if (fps > 0) p.opts.frame_rate = static_cast<float>(fps);
    }
    if (body.isMember("num_frames") && body["num_frames"].isIntegral()) {
        p.opts.num_frames = static_cast<std::size_t>(
            std::max<int64_t>(1, body["num_frames"].asInt64()));
    }
    if (!p.opts.num_frames && body.isMember("seconds")) {
        double seconds = 0.0;
        if (body["seconds"].isIntegral()) {
            seconds = static_cast<double>(body["seconds"].asInt64());
        } else if (body["seconds"].isString()) {
            std::string s = body["seconds"].asString();
            if (!s.empty() && (s.back() == 's' || s.back() == 'S')) {
                s.pop_back();
            }
            try {
                seconds = std::stod(s);
            } catch (const std::exception&) {
                seconds = 0.0;
            }
        }
        if (seconds > 0) {
            p.opts.num_frames = static_cast<std::size_t>(
                std::max(1.0, seconds * fps));
        }
    }
    if (body.isMember("seconds")) {
        if (body["seconds"].isIntegral()) {
            p.seconds = std::to_string(body["seconds"].asInt64());
        } else if (body["seconds"].isString()) {
            p.seconds = body["seconds"].asString();
        }
    }

    // Dimensions: OpenAI "size" ("1280x720") or explicit width/height. Passed
    // through exactly -- no 16/32 multiple rounding here. The model generates
    // at a padded multiple of 32 and the server resizes the frames back to the
    // requested size (see VideoGenerationModel::generate), so arbitrary sizes
    // like 1280x720 work.
    const bool have_size = body.isMember("size") && body["size"].isString();
    if (have_size) {
        const std::string size = body["size"].asString();
        const auto x = size.find_first_of("xX");
        if (x == std::string::npos) {
            throw std::runtime_error(
                "'size' must be formatted as \"WxH\", e.g. \"1280x720\"");
        }
        const int64_t w = std::stoll(size.substr(0, x));
        const int64_t h = std::stoll(size.substr(x + 1));
        if (w <= 0 || h <= 0) {
            throw std::runtime_error("'size' dimensions must be positive");
        }
        p.opts.width = w;
        p.opts.height = h;
        p.size = size;
    } else {
        int64_t w = 0, h = 0;
        getInt64(body, "width", w);
        getInt64(body, "height", h);
        if (w > 0 || h > 0) {
            if (w <= 0 || h <= 0) {
                throw std::runtime_error(
                    "'width' and 'height' must be provided together");
            }
            p.opts.width = w;
            p.opts.height = h;
            p.size = std::to_string(w) + "x" + std::to_string(h);
        }
    }
    return p;
}

// Parses a JSON request body into a Json::Value, throwing std::runtime_error
// (mapped to HTTP 400) on malformed input.
Json::Value parse_json_body(const drogon::HttpRequestPtr& req) {
    Json::CharReaderBuilder reader;
    reader["failIfExtra"] = false;
    auto body_view = req->getBody();
    std::istringstream stream(std::string(body_view.begin(), body_view.end()));
    Json::Value body;
    std::string errs;
    if (!Json::parseFromStream(reader, stream, &body, &errs) ||
        !body.isObject()) {
        throw std::runtime_error("Invalid JSON request body: " + errs);
    }
    return body;
}

const char* video_job_status_name(VideoJobStatus s) {
    switch (s) {
        case VideoJobStatus::queued: return "queued";
        case VideoJobStatus::running: return "running";
        case VideoJobStatus::completed: return "completed";
        case VideoJobStatus::failed: return "failed";
    }
    return "queued";
}

Json::Value video_job_json(const VideoJob& job) {
    Json::Value out;
    out["id"] = job.id;
    out["object"] = "video";
    out["model"] = job.model_id;
    out["prompt"] = job.prompt;
    out["status"] = video_job_status_name(job.status);
    out["created_at"] = static_cast<int>(job.created_at);
    if (job.size.empty()) {
        out["size"] = Json::Value();
    } else {
        out["size"] = job.size;
    }
    if (job.seconds.empty()) {
        out["seconds"] = Json::Value();
    } else {
        out["seconds"] = job.seconds;
    }
    out["quality"] = "standard";
    out["progress"] =
        (job.status == VideoJobStatus::completed ||
         job.status == VideoJobStatus::failed)
            ? 100
            : 0;
    if (job.completed_at) {
        out["completed_at"] = static_cast<int>(job.completed_at);
    }
    if (job.inference_time_s > 0.0) {
        out["inference_time_s"] = job.inference_time_s;
    }
    if (job.status == VideoJobStatus::failed) {
        out["error"] = job.error;
    } else {
        out["error"] = Json::nullValue;
    }
    if (job.status == VideoJobStatus::completed && !job.file_paths.empty()) {
        out["url"] = "/v1/videos/" + job.id + "/content";
        out["file_name"] =
            std::filesystem::path(job.file_paths.front()).filename().string();
        out["num_outputs"] = static_cast<int>(job.file_paths.size());
        Json::Value names = Json::arrayValue;
        for (const auto& p : job.file_paths) {
            names.append(std::filesystem::path(p).filename().string());
        }
        out["file_names"] = names;
    }
    return out;
}

// Generates with `model`, stores every output video in the storage directory as
// <id>.mp4 (additional outputs as <id>_N.mp4) and updates the job status.
// Returns an empty string on success, otherwise the error message; the job is
// marked failed in that case.
std::string run_video_job(VideoGenerationModel* model,
                          const VideoGenerateOptions& opts,
                          const std::string& id) {
    std::string error;
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::lock_guard lock(g_video_jobs_mutex);
        auto it = g_video_jobs.find(id);
        if (it != g_video_jobs.end()) {
            it->second.status = VideoJobStatus::running;
        }
    }
    try {
        const auto results = model->generate(opts);
        std::filesystem::path storage_dir;
        {
            std::lock_guard lock(g_video_jobs_mutex);
            storage_dir = g_video_storage_dir;
        }
        if (storage_dir.empty()) {
            std::error_code ec;
            storage_dir = std::filesystem::temp_directory_path(ec) / "videos";
        }
        std::error_code ec;
        std::filesystem::create_directories(storage_dir, ec);
        if (ec) {
            throw std::runtime_error("cannot create videos directory '" +
                                     storage_dir.string() + "': " + ec.message());
        }

        std::vector<std::string> paths;
        paths.reserve(results.size());
        for (std::size_t i = 0; i < results.size(); ++i) {
            std::string name = id + ".mp4";
            if (i > 0) {
                name = id + "_" + std::to_string(i + 1) + ".mp4";
            }
            const std::string path = (storage_dir / name).string();
            if (!write_video_mp4(path, results[i])) {
                throw std::runtime_error("cannot write MP4 to " + path);
            }
            paths.push_back(path);
        }

        const double elapsed = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t0)
                                   .count();
        std::lock_guard lock(g_video_jobs_mutex);
        auto it = g_video_jobs.find(id);
        if (it != g_video_jobs.end()) {
            it->second.file_paths = std::move(paths);
            it->second.status = VideoJobStatus::completed;
            it->second.completed_at = std::time(nullptr);
            it->second.inference_time_s = elapsed;
        }
    } catch (const std::exception& e) {
        error = e.what();
        std::lock_guard lock(g_video_jobs_mutex);
        auto it = g_video_jobs.find(id);
        if (it != g_video_jobs.end()) {
            it->second.status = VideoJobStatus::failed;
            it->second.error = error;
            it->second.completed_at = std::time(nullptr);
        }
    }
    return error;
}

}  // namespace

void set_video_storage_dir(const std::filesystem::path& dir) {
    std::lock_guard lock(g_video_jobs_mutex);
    g_video_storage_dir = dir;
}

void registerVideoGenerations(drogon::HttpAppFramework& app) {
    // POST /v1/videos -- create an asynchronous video generation job.
    app.registerHandler(
        "/v1/videos",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            Json::Value body;
            try {
                body = parse_json_body(req);
            } catch (const std::exception& e) {
                callback(error_response(e.what(), drogon::k400BadRequest));
                return;
            }
            try {
                ParsedVideoRequest p = parse_video_request(body);
                if (!p.model) {
                    callback(error_response(
                        "The requested video model is not available. Start the "
                        "server with --txt2vid or provide a valid 'model' field.",
                        drogon::k404NotFound));
                    return;
                }
                const std::string id = random_uuid();
                {
                    std::lock_guard lock(g_video_jobs_mutex);
                    g_video_jobs.emplace(
                        id, VideoJob{id, p.model->id(), p.opts.prompt,
                                     VideoJobStatus::queued, std::time(nullptr),
                                     0, 0.0, "", {}, p.size, p.seconds,
                                     p.num_videos});
                }
                pool().enqueue([model = p.model, opts = p.opts, id] {
                    const std::string err = run_video_job(model, opts, id);
                    std::cerr << "video job " << id
                              << (err.empty() ? " completed" : " failed: " + err)
                              << std::endl;
                });
                Json::Value out;
                {
                    std::lock_guard lock(g_video_jobs_mutex);
                    out = video_job_json(g_video_jobs.at(id));
                }
                callback(json_response(out));
            } catch (const std::exception& e) {
                callback(error_response(e.what(), drogon::k400BadRequest));
            }
        },
        {drogon::Post});

    // GET /v1/videos/{id} -- poll the status of a video generation job.
    app.registerHandler(
        "/v1/videos/{1}",
        [](const drogon::HttpRequestPtr& /*req*/,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback,
           const std::string& id) {
            std::lock_guard lock(g_video_jobs_mutex);
            const auto it = g_video_jobs.find(id);
            if (it == g_video_jobs.end()) {
                callback(error_response("video job not found",
                                        drogon::k404NotFound));
                return;
            }
            callback(json_response(video_job_json(it->second)));
        },
        {drogon::Get});

    // GET /v1/videos/{id}/content -- download the finished video. ?index=N
    // selects the Nth output when a job has multiple videos (default 1).
    app.registerHandler(
        "/v1/videos/{1}/content",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback,
           const std::string& id) {
            std::size_t index = 0;
            const std::string idx = req->getParameter("index");
            if (!idx.empty()) {
                try {
                    index = static_cast<std::size_t>(std::max(1, std::stoi(idx))) - 1;
                } catch (const std::exception&) {
                    index = 0;
                }
            }
            std::string path;
            {
                std::lock_guard lock(g_video_jobs_mutex);
                const auto it = g_video_jobs.find(id);
                if (it == g_video_jobs.end()) {
                    callback(error_response("video job not found",
                                            drogon::k404NotFound));
                    return;
                }
                if (it->second.status == VideoJobStatus::completed &&
                    index < it->second.file_paths.size()) {
                    path = it->second.file_paths[index];
                }
            }
            if (path.empty()) {
                callback(error_response("video output is not ready",
                                        drogon::k404NotFound));
                return;
            }
            std::ifstream fin(path, std::ios::binary);
            if (!fin) {
                callback(error_response("video file missing",
                                        drogon::k404NotFound));
                return;
            }
            std::string bytes((std::istreambuf_iterator<char>(fin)),
                              std::istreambuf_iterator<char>());
            callback(bytes_response(std::move(bytes), drogon::CT_VIDEO_MP4));
        },
        {drogon::Get});

    // POST /v1/video/generations -- synchronous, OpenAI-standard response.
    // Defaults to stored files with fetchable URLs; "response_format":
    // "b64_json" returns inline base64 instead.
    app.registerHandler(
        "/v1/video/generations",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            Json::Value body;
            try {
                body = parse_json_body(req);
            } catch (const std::exception& e) {
                callback(error_response(e.what(), drogon::k400BadRequest));
                return;
            }
            try {
                ParsedVideoRequest p = parse_video_request(body);
                if (!p.model) {
                    callback(error_response(
                        "The requested video model is not available. Start the "
                        "server with --txt2vid or provide a valid 'model' field.",
                        drogon::k404NotFound));
                    return;
                }
                const bool want_b64 =
                    body.isMember("response_format") &&
                    body["response_format"].isString() &&
                    body["response_format"].asString() == "b64_json";
                if (body.isMember("response_format") &&
                    body["response_format"].isString() &&
                    body["response_format"].asString() != "b64_json" &&
                    body["response_format"].asString() != "url") {
                    throw std::runtime_error(
                        "'response_format' must be \"b64_json\" or \"url\"");
                }
                if (body.isMember("output_format") &&
                    body["output_format"].isString() &&
                    body["output_format"].asString() != "mp4") {
                    throw std::runtime_error(
                        "'output_format' only supports \"mp4\"");
                }

                const std::string id = random_uuid();
                pool().enqueue([model = p.model, opts = p.opts, id,
                                num_videos = p.num_videos, want_b64,
                                size = p.size, seconds = p.seconds,
                                callback] {
                    drogon::HttpResponsePtr resp;
                    try {
                        Json::Value data = Json::arrayValue;
                        if (want_b64) {
                            // Inline base64 without storing files.
                            const auto results = model->generate(opts);
                            for (std::size_t i = 0; i < results.size(); ++i) {
                                const std::filesystem::path tmp =
                                    std::filesystem::temp_directory_path() /
                                    ("ovserver_vid_" + id + "_" +
                                     std::to_string(i) + ".mp4");
                                if (!write_video_mp4(tmp.string(), results[i])) {
                                    throw std::runtime_error(
                                        "encoding MP4 failed (is ffmpeg on "
                                        "PATH?)");
                                }
                                std::ifstream fin(tmp, std::ios::binary);
                                std::string bytes(
                                    (std::istreambuf_iterator<char>(fin)),
                                    std::istreambuf_iterator<char>());
                                fin.close();
                                std::filesystem::remove(tmp);
                                Json::Value item;
                                item["b64_json"] = base64_encode(
                                    reinterpret_cast<const std::uint8_t*>(
                                        bytes.data()),
                                    bytes.size());
                                item["url"] = Json::nullValue;
                                item["filename"] =
                                    id + (i > 0 ? "_" + std::to_string(i + 1)
                                                : "") +
                                    ".mp4";
                                data.append(item);
                            }
                        } else {
                            // Store under <id>.mp4 and return fetchable URLs.
                            {
                                std::lock_guard lock(g_video_jobs_mutex);
                                g_video_jobs.emplace(
                                    id, VideoJob{id, model->id(), opts.prompt,
                                                 VideoJobStatus::queued,
                                                 std::time(nullptr), 0, 0.0,
                                                 "", {}, size, seconds,
                                                 num_videos});
                            }
                            const std::string err = run_video_job(model, opts, id);
                            if (!err.empty()) {
                                throw std::runtime_error(err);
                            }
                            std::vector<std::string> paths;
                            {
                                std::lock_guard lock(g_video_jobs_mutex);
                                const auto it = g_video_jobs.find(id);
                                if (it != g_video_jobs.end()) {
                                    paths = it->second.file_paths;
                                }
                            }
                            for (std::size_t i = 0; i < paths.size(); ++i) {
                                Json::Value item;
                                std::string url =
                                    "/v1/videos/" + id + "/content";
                                if (i > 0) {
                                    url += "?index=" + std::to_string(i + 1);
                                }
                                item["url"] = url;
                                item["filename"] =
                                    std::filesystem::path(paths[i])
                                        .filename()
                                        .string();
                                item["expiry"] =
                                    static_cast<int>(std::time(nullptr)) + 86400;
                                item["b64_json"] = Json::nullValue;
                                data.append(item);
                            }
                        }
                        Json::Value out;
                        out["id"] = id;
                        out["object"] = "video";
                        out["created"] = static_cast<int>(std::time(nullptr));
                        out["model"] = model->id();
                        out["status"] = "completed";
                        out["error"] = Json::nullValue;
                        out["data"] = data;
                        resp = json_response(out);
                    } catch (const std::exception& e) {
                        resp = internal_error_response(e);
                    }
                    callback(resp);
                });
            } catch (const std::exception& e) {
                callback(error_response(e.what(), drogon::k400BadRequest));
            }
        },
        {drogon::Post});
}

void registerAudioTranscriptions(drogon::HttpAppFramework& app) {
    app.registerHandler(
        "/v1/audio/transcriptions",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            drogon::MultiPartParser parser;
            const int parse_status = parser.parse(req);
            if (parse_status != 0) {
                callback(error_response(
                    "multipart/form-data upload is required",
                    drogon::k400BadRequest));
                return;
            }
            const auto& files = parser.getFiles();
            if (files.empty()) {
                callback(error_response(
                    "a 'file' field with the audio to transcribe is required",
                    drogon::k400BadRequest));
                return;
            }

            try {
                ASRModel* model = nullptr;
                const auto& params = parser.getParameters();
                auto model_it = params.find("model");
                const std::string requested =
                    model_it == params.end() ? std::string{} : model_it->second;
                if (requested.empty()) {
                    auto& models = ModelManager::instance().all_asr();
                    if (models.size() == 1) {
                        model = models.begin()->second.get();
                    }
                } else {
                    model = ModelManager::instance().get_asr(requested);
                }
                if (!model) {
                    callback(error_response(
                        "The requested ASR model is not available. Start the "
                        "server with --wav2txt or provide a valid 'model' "
                        "field.",
                        drogon::k404NotFound));
                    return;
                }

                ASRGenerateOptions opts;
                auto lang_it = params.find("language");
                if (lang_it != params.end()) {
                    opts.language = lang_it->second;
                }
                auto prompt_it = params.find("prompt");
                if (prompt_it != params.end()) {
                    opts.initial_prompt = prompt_it->second;
                }
                auto temp_it = params.find("temperature");
                if (temp_it != params.end()) {
                    try {
                        opts.temperature =
                            static_cast<float>(std::stod(temp_it->second));
                    } catch (const std::exception&) {
                        // ignore malformed temperature
                    }
                }
                auto rfmt_it = params.find("response_format");
                const std::string response_format =
                    rfmt_it == params.end() ? std::string{} : rfmt_it->second;
                if (!response_format.empty() &&
                    response_format != "json" && response_format != "text" &&
                    response_format != "verbose_json") {
                    callback(error_response(
                        "'response_format' must be \"json\", \"text\" or "
                        "\"verbose_json\"",
                        drogon::k400BadRequest));
                    return;
                }
                auto as_bool = [](const std::string& v) {
                    std::string low;
                    low.reserve(v.size());
                    for (char c : v)
                        low.push_back(
                            static_cast<char>(std::tolower(
                                static_cast<unsigned char>(c))));
                    return low == "1" || low == "true" || low == "on" ||
                           low == "yes";
                };
                auto stream_it = params.find("stream");
                const bool stream =
                    stream_it != params.end() && as_bool(stream_it->second);
                if (stream && !response_format.empty() &&
                    response_format != "json" && response_format != "text") {
                    callback(error_response(
                        "stream=true supports only 'response_format' \"json\" "
                        "or \"text\"",
                        drogon::k400BadRequest));
                    return;
                }
                auto tg_it = params.find("timestamp_granularities");
                if (tg_it != params.end()) {
                    const std::string& tg = tg_it->second;
                    if (tg.find("word") != std::string::npos) {
                        callback(error_response(
                            "word-level timestamps are not supported; use \"segment\"",
                            drogon::k400BadRequest));
                        return;
                    }
                    if (tg.find("segment") != std::string::npos) {
                        opts.return_timestamps = true;
                    }
                }

                auto file_content = files.front().fileContent();
                const std::string audio_bytes =
                    file_content.empty()
                        ? std::string{}
                        : std::string(file_content.data(),
                                      file_content.size());

                const std::string model_name =
                    requested.empty() ? model->id() : requested;

                if (stream) {
                    // Degrades to the SGLang-style SSE transcript stream that
                    // vLLM/llama.cpp-family clients also consume:
                    //   data: {"type":"transcript.text.delta","delta":...}
                    //   data: {"type":"transcript.text.done","text":...,"usage":...}
                    //   data: [DONE]
                    struct StreamState {
                        std::shared_ptr<drogon::ResponseStream> stream;
                        std::mutex mutex;
                    };
                    auto state = std::make_shared<StreamState>();
                    drogon::HttpResponsePtr resp =
                        drogon::HttpResponse::newAsyncStreamResponse(
                            [state, model, opts, audio_bytes](
                                drogon::ResponseStreamPtr stream) {
                                state->stream.reset(stream.release());

                                // Sends one JSON payload as an SSE frame.
                                // Returns false if the client is gone.
                                auto send_event =
                                    [state](const std::string& payload) {
                                        std::lock_guard lock(state->mutex);
                                        auto s = state->stream;
                                        if (!s) {
                                            return false;
                                        }
                                        const bool ok =
                                            s->send(sse_message(payload));
                                        if (!ok) {
                                            state->stream.reset();
                                        }
                                        return ok;
                                    };

                                auto worker =
                                    [state, model, opts, audio_bytes,
                                     send_event]() mutable {
                                        ASRResult result;
                                        try {
                                            opts.samples =
                                                decode_audio_to_f32(audio_bytes);
                                            const std::size_t duration_s =
                                                (opts.samples.size() + 15999) /
                                                16000;
                                            ASROnText on_text =
                                                [send_event](
                                                    const std::string& delta) {
                                                    Json::StreamWriterBuilder b;
                                                    b["indentation"] = "";
                                                    Json::Value ev;
                                                    ev["type"] =
                                                        "transcript.text.delta";
                                                    ev["delta"] = delta;
                                                    const bool ok = send_event(
                                                        Json::writeString(
                                                            b, ev));
                                                    return ok
                                                               ? ov::genai::
                                                                     StreamingStatus::
                                                                         RUNNING
                                                               : ov::genai::
                                                                     StreamingStatus::
                                                                         STOP;
                                                };
                                            result =
                                                model->generate(opts, on_text);

                                            Json::StreamWriterBuilder b;
                                            b["indentation"] = "";
                                            Json::Value done;
                                            done["type"] =
                                                "transcript.text.done";
                                            done["text"] = result.text;
                                            if (duration_s > 0) {
                                                Json::Value usage;
                                                usage["type"] = "duration";
                                                usage["seconds"] = static_cast<
                                                    int64_t>(duration_s);
                                                done["usage"] = usage;
                                            }
                                            send_event(
                                                Json::writeString(b, done));
                                            send_event("[DONE]");
                                        } catch (const std::exception& e) {
                                            Json::StreamWriterBuilder b;
                                            b["indentation"] = "";
                                            Json::Value err;
                                            err["type"] = "error";
                                            Json::Value msg;
                                            msg["message"] =
                                                "Failed to transcribe audio: " +
                                                std::string(e.what());
                                            err["error"] = msg;
                                            send_event(
                                                Json::writeString(b, err));
                                        }
                                        std::lock_guard lock(state->mutex);
                                        if (state->stream) {
                                            state->stream->close();
                                            state->stream.reset();
                                        }
                                    };
                                try {
                                    pool().enqueue(worker);
                                } catch (const std::exception& e) {
                                    Json::StreamWriterBuilder b;
                                    b["indentation"] = "";
                                    Json::Value err;
                                    err["type"] = "error";
                                    Json::Value msg;
                                    msg["message"] =
                                        "Failed to transcribe audio: " +
                                        std::string(e.what());
                                    err["error"] = msg;
                                    send_event(Json::writeString(b, err));
                                    std::lock_guard lock(state->mutex);
                                    if (state->stream) {
                                        state->stream->close();
                                        state->stream.reset();
                                    }
                                }
                            },
                            true /*disableKickoffTimeout*/);
                    resp->setContentTypeString("text/event-stream");
                    resp->addHeader("Cache-Control", "no-cache");
                    resp->addHeader("Connection", "keep-alive");
                    callback(resp);
                    return;
                }

                pool().enqueue([model, opts, audio_bytes, response_format,
                                model_name, callback]() mutable {
                    drogon::HttpResponsePtr resp;
                    try {
                        opts.samples = decode_audio_to_f32(audio_bytes);
                        ASRResult result = model->generate(opts);

                        if (response_format == "text") {
                            resp = bytes_response(result.text,
                                                  drogon::CT_TEXT_PLAIN);
                        } else if (response_format == "verbose_json") {
                            Json::Value segs = Json::arrayValue;
                            const double duration =
                                static_cast<double>(opts.samples.size()) /
                                16000.0;
                            for (std::size_t i = 0; i < result.segments.size();
                                 ++i) {
                                Json::Value s;
                                s["id"] = static_cast<int>(i);
                                s["start"] = result.segments[i].start;
                                s["end"] = result.segments[i].end;
                                s["text"] = result.segments[i].text;
                                segs.append(s);
                            }
                            Json::Value out;
                            out["task"] = "transcribe";
                            out["language"] = result.language;
                            out["duration"] = duration;
                            out["text"] = result.text;
                            out["segments"] = segs;
                            resp = json_response(out);
                        } else {
                            Json::Value out;
                            out["text"] = result.text;
                            resp = json_response(out);
                        }
                    } catch (const std::exception& e) {
                        resp = error_response(
                            "Failed to transcribe audio: " +
                                std::string(e.what()),
                            drogon::k500InternalServerError);
                    }
                    callback(resp);
                });
            } catch (const std::exception& e) {
                callback(error_response(e.what(), drogon::k400BadRequest));
            }
        },
        {drogon::Post});
}

void registerAudioSpeech(drogon::HttpAppFramework& app) {
    app.registerHandler(
        "/v1/audio/speech",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            Json::CharReaderBuilder reader;
            reader["failIfExtra"] = false;
            auto body_view = req->getBody();
            std::istringstream stream(std::string(body_view.begin(), body_view.end()));
            Json::Value body;
            std::string errs;
            if (!Json::parseFromStream(reader, stream, &body, &errs) ||
                !body.isObject()) {
                callback(error_response("Invalid JSON request body: " + errs,
                                        drogon::k400BadRequest));
                return;
            }

            try {
                const std::string text = getString(body, "input");
                TTSModel* model = nullptr;
                if (!body.isMember("model")) {
                    auto& models = ModelManager::instance().all_tts();
                    if (models.size() == 1) {
                        model = models.begin()->second.get();
                    }
                } else if (body["model"].isString()) {
                    model = ModelManager::instance().get_tts(body["model"].asString());
                }
                if (!model) {
                    callback(error_response(
                        "The requested TTS model is not available. Start the "
                        "server with --txt2wav or provide a valid 'model' field.",
                        drogon::k404NotFound));
                    return;
                }

                std::string response_format = "wav";
                if (body.isMember("response_format") &&
                    body["response_format"].isString()) {
                    response_format = body["response_format"].asString();
                }
                if (response_format != "wav" && response_format != "pcm") {
                    callback(error_response(
                        "'response_format' must be \"wav\" or \"pcm\"",
                        drogon::k400BadRequest));
                    return;
                }

                std::string voice;
                if (body.isMember("voice") && body["voice"].isString()) {
                    voice = body["voice"].asString();
                }

                std::shared_ptr<ov::Tensor> speaker_embedding;
                if (!voice.empty()) {
                    const std::filesystem::path emb_path =
                        model->models_path() / "voices" / (voice + ".bin");
                    std::ifstream fin(emb_path, std::ios::binary);
                    if (!fin) {
                        throw std::runtime_error(
                            "voice '" + voice + "' not found (expected " +
                            emb_path.string() + ")");
                    }
                    fin.seekg(0, std::ios::end);
                    const std::streamoff size = fin.tellg();
                    fin.seekg(0, std::ios::beg);
                    if (size <= 0 || size % 1024 != 0) {
                        throw std::runtime_error(
                            "invalid speaker embedding file '" + voice +
                            ".bin': expected a float32 [N,1,256] tensor");
                    }
                    const std::size_t rows =
                        static_cast<std::size_t>(size) / 1024;
                    speaker_embedding = std::make_shared<ov::Tensor>(
                        ov::element::f32, ov::Shape{rows, 1, 256});
                    fin.read(
                        reinterpret_cast<char*>(speaker_embedding->data()),
                        size);
                    if (!fin) {
                        throw std::runtime_error(
                            "failed to read speaker embedding file '" + voice +
                            ".bin'");
                    }
                }

                pool().enqueue([model, text, response_format,
                                speaker_embedding, callback] {
                    drogon::HttpResponsePtr resp;
                    try {
                        TTSResult result = model->generate(
                            text, speaker_embedding ? *speaker_embedding
                                                    : ov::Tensor{});
                        std::vector<std::uint8_t> raw =
                            wav_pcm16(result.samples, result.sample_rate);
                        resp = bytes_response(
                            std::string(
                                reinterpret_cast<const char*>(raw.data()),
                                raw.size()),
                            response_format == "wav"
                                ? drogon::CT_AUDIO_WAVE
                                : drogon::CT_APPLICATION_OCTET_STREAM);
                    } catch (const std::exception& e) {
                        resp = error_response(
                            "Failed to generate speech: " +
                                std::string(e.what()),
                            drogon::k500InternalServerError);
                    }
                    callback(resp);
                });
            } catch (const std::exception& e) {
                callback(error_response(e.what(), drogon::k400BadRequest));
            }
        },
        {drogon::Post});
}

void register_api_handlers(drogon::HttpAppFramework& app) {
    registerHealth(app);
    registerModels(app);
    registerImageGenerations(app);
    registerVideoGenerations(app);
    registerAudioTranscriptions(app);
    registerAudioSpeech(app);
    registerChatCompletions(app);
}

}  // namespace ovserver