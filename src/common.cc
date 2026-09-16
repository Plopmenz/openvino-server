// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#include "ovserver/common.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <json/json.h>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>

namespace ovserver {

namespace {

// Device-scoped property maps from set_device_props(). Keys are uppercased
// device names as reported by OpenVINO (e.g. "GPU").
using DevicePropsMap = std::map<std::string, ov::AnyMap>;

DevicePropsMap& g_device_props() {
    static DevicePropsMap props;
    return props;
}

std::string upper_copy(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(static_cast<char>(
            std::toupper(static_cast<unsigned char>(c))));
    }
    return out;
}

// Converts a JSON scalar to the closest native OpenVINO property value.
ov::Any json_to_any(const Json::Value& v) {
    if (v.isBool()) {
        return v.asBool();
    }
    if (v.isUInt64() || v.isInt64()) {
        return v.asInt64();
    }
    if (v.isDouble()) {
        return v.asDouble();
    }
    if (v.isString()) {
        return v.asString();
    }
    throw std::runtime_error("device property values must be a string, "
                             "number or boolean");
}

ov::AnyMap json_to_props(const Json::Value& obj) {
    if (!obj.isObject()) {
        throw std::runtime_error(
            "each device entry must be a JSON object of properties");
    }
    ov::AnyMap props;
    for (const std::string& key : obj.getMemberNames()) {
        props.emplace(key, json_to_any(obj[key]));
    }
    return props;
}

// Reads a sized field in kB from /proc/self/status (e.g. VmRSS).
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
// Returns -1.0 when not applicable (non-GPU device, plugin unavailable).
double device_gpu_mib(const std::string& device) {
    if (device.find("GPU") == std::string::npos) {
        return -1.0;
    }
    try {
        ov::Core core;
        auto stats =
            core.get_property(device, ov::intel_gpu::memory_statistics);
        std::uint64_t bytes = 0;
        for (const auto& [name, value] : stats) {
            bytes += value;
        }
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    } catch (const std::exception&) {
        return -1.0;
    }
}

// Total NPU DDR memory currently allocated, in MiB.
// Returns -1.0 when not applicable (non-NPU device, plugin unavailable).
double device_npu_mib(const std::string& device) {
    if (device.find("NPU") == std::string::npos) {
        return -1.0;
    }
    try {
        ov::Core core;
        const std::uint64_t bytes =
            core.get_property(device, ov::intel_npu::device_alloc_mem_size);
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    } catch (const std::exception&) {
        return -1.0;
    }
}

}  // namespace

void set_device_props(const std::string& json) {
    if (json.empty()) {
        g_device_props().clear();
        return;
    }
    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(json.data(), json.data() + json.size(), &root,
                       &errors)) {
        throw std::runtime_error("failed to parse --device-props JSON: " +
                                 errors);
    }
    if (!root.isObject()) {
        throw std::runtime_error("--device-props must be a JSON object "
                                 "mapping device names to property maps");
    }
    DevicePropsMap& props = g_device_props();
    props.clear();
    for (const std::string& device : root.getMemberNames()) {
        props.emplace(upper_copy(device), json_to_props(root[device]));
    }
}

ov::AnyMap inference_properties(const std::string& device,
                                const std::string& cache_dir) {
    ov::AnyMap properties;
    if (!cache_dir.empty()) {
        properties.emplace(ov::cache_dir(cache_dir));
    }
    const DevicePropsMap& props = g_device_props();
    if (props.empty()) {
        return properties;
    }
    auto it = props.find(upper_copy(device));
    if (it == props.end()) {
        return properties;
    }
    // Wrap device-scoped hints in ov::device::properties so they are forwarded
    // to the OpenVINO plugin for this device only. GenAI forwards the full
    // property map to the compile call; without device-scoping the CPU plugin
    // would reject device-specific keys.
    properties.emplace(ov::device::properties(device, it->second));
    return properties;
}

std::string shell_quote(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (char c : value) {
        if (c == '"' || c == '\\' || c == '$' || c == '`') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::function<bool(std::size_t, std::size_t, ov::Tensor&)> step_logger(
    const std::string& tag,
    const std::string& id,
    std::uint64_t req_id,
    std::chrono::steady_clock::time_point start) {
    return [tag, id, req_id, last = start](std::size_t step, std::size_t total,
                                           ov::Tensor&) mutable -> bool {
        const auto now = std::chrono::steady_clock::now();
        const auto step_s = std::chrono::duration<double>(now - last).count();
        last = now;
        std::cerr << "[" << tag << " model '" << id << "'] request " << req_id
                  << " step " << step + 1 << "/" << total << ": " << step_s
                  << " s" << std::endl;
        return false;
    };
}

void log_model_memory(const std::string& tag,
                      const std::string& id,
                      const std::vector<std::string>& devices) {
    std::cerr << "[" << tag << " model '" << id << "'] memory: ";
    std::vector<std::string> seen;
    bool first = true;
    for (const auto& device : devices) {
        if (std::find(seen.begin(), seen.end(), device) != seen.end()) {
            continue;
        }
        seen.push_back(device);

        bool unknown = false;
        double gib = -1.0;
        std::string label;
        const double gpu_mib = device_gpu_mib(device);
        if (gpu_mib >= 0.0) {
            label = "GPU";
            gib = gpu_mib / 1024.0;
        } else if (device.find("GPU") != std::string::npos) {
            label = "GPU";
            unknown = true;
        } else {
            const double npu_mib = device_npu_mib(device);
            if (npu_mib >= 0.0) {
                label = "NPU";
                gib = npu_mib / 1024.0;
            } else if (device.find("NPU") != std::string::npos) {
                label = "NPU";
                unknown = true;
            } else {
                label = "CPU";
                gib = proc_field_kb("VmRSS:") / (1024.0 * 1024.0);
            }
        }

        if (!first) {
            std::cerr << ", ";
        }
        first = false;
        std::cerr << label << " ";
        if (unknown) {
            std::cerr << "??? GiB";
        } else {
            std::cerr << std::fixed << std::setprecision(2) << gib << " GiB";
        }
    }
    std::cerr << std::endl;
}

PipelineLoadLog::PipelineLoadLog(std::string tag,
                                 std::string id,
                                 const std::filesystem::path& path,
                                 std::string device,
                                 std::string second_device)
    : m_tag(std::move(tag)),
      m_id(std::move(id)),
      m_path(path),
      m_device(std::move(device)),
      m_second_device(std::move(second_device)),
      m_t0(std::chrono::steady_clock::now()) {
    std::cerr << "[" << m_tag << " model '" << m_id << "'] loading from "
              << m_path << " on " << m_device << " ..." << std::endl;
}

void PipelineLoadLog::completion() {
    const double load_s = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - m_t0)
                              .count();
    std::cerr << "[" << m_tag << " model '" << m_id << "'] loaded in "
              << load_s << " s" << std::endl;
    std::vector<std::string> devices{m_device};
    if (!m_second_device.empty() && m_second_device != m_device) {
        devices.push_back(m_second_device);
    }
    log_model_memory(m_tag, m_id, devices);
}

void notify_systemd_ready() {
    const char* sock_name = std::getenv("NOTIFY_SOCKET");
    if (sock_name == nullptr || sock_name[0] == '\0') {
        // Not started by systemd; run() semantics are unchanged.
        return;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::size_t prefix = 0;
    if (sock_name[0] == '@') {
        // Abstract socket namespace: leading NUL inside sun_path.
        prefix = 1;
        addr.sun_path[0] = '\0';
    }
    const std::size_t len = std::strlen(sock_name) - prefix;
    if (len >= sizeof(addr.sun_path)) {
        std::cerr << "warning: NOTIFY_SOCKET too long, skipping READY=1"
                  << std::endl;
        return;
    }
    std::memcpy(addr.sun_path + (prefix ? 1 : 0), sock_name + prefix, len);

    const socklen_t sun_len =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) +
                               (prefix ? 1 : 0) + len);
    const int fd = ::socket(AF_UNIX, SOCK_DGRAM, 0);
    if (fd < 0) {
        return;
    }
    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sun_len) == 0) {
        static constexpr char kReady[] = "READY=1\n";
        (void)::send(fd, kReady, sizeof(kReady) - 1, MSG_NOSIGNAL);
    }
    ::close(fd);
}

}  // namespace ovserver
