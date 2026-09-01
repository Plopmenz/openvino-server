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
#include "ovserver/manager.hpp"
#include "ovserver/model.hpp"

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
                    const int64_t lo = 512, hi = 4096;
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

}  // namespace

void register_api_handlers(drogon::HttpAppFramework& app) {
    registerHealth(app);
    registerModels(app);
    registerImageGenerations(app);
}

}  // namespace ovserver