// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/controller.hpp"

#include <json/json.h>

#include <algorithm>
#include <condition_variable>
#include <ctime>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ovserver/image.hpp"
#include "ovserver/image_decode.hpp"
#include "ovserver/manager.hpp"
#include "ovserver/model.hpp"
#include "ovserver/vlm_model.hpp"

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
    choice["delta"] = delta;
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

// Extracts the full decoded text and image tensors from an OpenAI-style
// "messages" array. Content may be a plain string or an array of typed parts
// ({type:"text",text:...} and {type:"image_url",image_url:{url:"..."}}).
struct ParsedChat {
    std::string prompt;
    std::string system_message;
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
                            const auto& models = ModelManager::instance().all();
                            for (const auto& entry : models) {
                                Json::Value m;
                                m["id"] = entry.first;
                                m["object"] = "model";
                                m["created"] = 0;
                                m["owned_by"] = "openvino-genai";
                                arr.append(m);
                            }
                            const auto& vlms = ModelManager::instance().all_vlms();
                            for (const auto& entry : vlms) {
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
                Model* model = nullptr;

                if (!body.isMember("model")) {
                    auto& models = ModelManager::instance().all();
                    if (models.size() == 1) {
                        model = models.begin()->second.get();
                    }
                } else if (body["model"].isString()) {
                    model = ModelManager::instance().get(body["model"].asString());
                }
                if (!model) {
                    callback(error_response(
                        "The requested model is not available. Start the server "
                        "with --model or provide a valid 'model' field.",
                        drogon::k404NotFound));
                    return;
                }

                GenerateOptions opts;
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
                    const bool save = body.isMember("output_dir") &&
                                      body["output_dir"].isString();
                    std::string output_dir;
                    if (save) {
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

// Parses the message array into text + images. OpenAI-compatible content blocks
// ("text" and "image_url") are supported.
ParsedChat parse_messages(const Json::Value& messages) {
    ParsedChat out;
    if (!messages.isArray()) {
        throw std::runtime_error("'messages' must be an array");
    }
    std::string user_text;
    for (const auto& msg : messages) {
        if (!msg.isMember("role") || !msg["role"].isString()) {
            throw std::runtime_error("each message requires a 'role'");
        }
        const std::string role = msg["role"].asString();
        if (!msg.isMember("content")) {
            continue;
        }
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
                    text += part["text"].asString();
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
        }
        if (role == "system") {
            out.system_message += text;
        } else if (role == "user") {
            user_text += text;
        }
    }
    out.prompt = user_text;
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
                VLMModel* model = nullptr;
                if (body.isMember("model") && body["model"].isString()) {
                    model = ModelManager::instance().get_vlm(body["model"].asString());
                }
                if (!model) {
                    auto& vlms = ModelManager::instance().all_vlms();
                    if (vlms.size() == 1) {
                        model = vlms.begin()->second.get();
                    }
                }
                if (!model) {
                    callback(error_response(
                        "The requested VLM model is not available. Start the "
                        "server with --vlm-model or provide a valid 'model' field.",
                        drogon::k404NotFound));
                    return;
                }

                ParsedChat chat = parse_messages(body["messages"]);
                VLMGenerateOptions opts;
                opts.prompt = chat.prompt;
                opts.system_message = chat.system_message;
                opts.images = std::move(chat.images);
                if (body.isMember("max_tokens") && body["max_tokens"].isIntegral()) {
                    opts.max_new_tokens = static_cast<std::size_t>(
                        std::max<int64_t>(1, body["max_tokens"].asInt64()));
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
                if (body.isMember("seed") && body["seed"].isIntegral()) {
                    opts.rng_seed = static_cast<std::size_t>(
                        std::max<int64_t>(0, body["seed"].asInt64()));
                }

                const std::string model_name = model->id();
                const std::string req_id = "chatcmpl-" + std::to_string(std::time(nullptr));

                const bool do_stream = body.isMember("stream") &&
                                       body["stream"].isBool() &&
                                       body["stream"].asBool();

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
                             shared_model,
                             req_id](drogon::ResponseStreamPtr stream) {
                                state->stream.reset(stream.release());

                                // Sends one content fragment as an SSE chunk.
                                // Returns false if the client is gone.
                                auto send_chunk =
                                    [state, shared_id, shared_model](
                                        std::string word) {
                                        std::lock_guard lock(state->mutex);
                                        auto s = state->stream;
                                        if (!s) {
                                            return false;
                                        }
                                        Json::Value delta;
                                        delta["content"] = std::move(word);
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

                                VLMGenerateOptions gen = opts;
                                gen.on_text =
                                    [send_chunk](std::string word) {
                                        return send_chunk(std::move(word));
                                    };

                                // Sends the trailing finish_reason + [DONE] and
                                // closes the stream.
                                auto finish =
                                    [state, shared_id, shared_model]() {
                                        std::lock_guard lock(state->mutex);
                                        if (!state->stream) {
                                            return;
                                        }
                                        Json::Value delta;
                                        Json::Value chunk = chat_chunk(
                                            *shared_id, delta, "stop");
                                        chunk["model"] = *shared_model;
                                        Json::StreamWriterBuilder b;
                                        b["indentation"] = "";
                                        state->stream->send(sse_message(
                                            Json::writeString(b, chunk)));
                                        state->stream->send(
                                            sse_message("[DONE]"));
                                        state->stream->close();
                                        state->stream.reset();
                                    };

                                auto worker = [state, gen, model, finish] {
                                    try {
                                        model->generate(gen);
                                    } catch (const std::exception& e) {
                                        std::cerr << "vlm streaming error: "
                                                  << e.what() << std::endl;
                                    }
                                    finish();
                                };
                                try {
                                    pool().enqueue(worker);
                                } catch (const std::exception& e) {
                                    finish();
                                }
                            },
                            true /*disableKickoffTimeout*/);
                    resp->setContentTypeString("text/event-stream");
                    resp->addHeader("Cache-Control", "no-cache");
                    resp->addHeader("Connection", "keep-alive");
                    callback(resp);
                    return;
                }

                // Non-streaming: full response in one shot.
                pool().enqueue([model, opts, model_name, req_id, callback] {
                    drogon::HttpResponsePtr resp;
                    try {
                        VLMResult result = model->generate(opts);
                        Json::Value out;
                        out["id"] = req_id;
                        out["object"] = "chat.completion";
                        out["created"] = static_cast<int>(std::time(nullptr));
                        out["model"] = model_name;
                        Json::Value msg;
                        msg["role"] = "assistant";
                        msg["content"] = result.text;
                        Json::Value choice;
                        choice["index"] = 0;
                        choice["message"] = msg;
                        choice["finish_reason"] =
                            result.finish_reason.empty()
                                ? Json::Value("stop")
                                : Json::Value(result.finish_reason);
                        Json::Value choices = Json::arrayValue;
                        choices.append(choice);
                        out["choices"] = choices;
                        Json::Value usage;
                        usage["prompt_tokens"] = 0;
                        usage["completion_tokens"] = 0;
                        usage["total_tokens"] = 0;
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

}  // namespace

void register_api_handlers(drogon::HttpAppFramework& app) {
    registerHealth(app);
    registerModels(app);
    registerImageGenerations(app);
    registerChatCompletions(app);
}

}  // namespace ovserver