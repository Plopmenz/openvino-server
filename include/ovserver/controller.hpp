// Copyright (C) 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include <drogon/drogon.h>

namespace ovserver {

void register_api_handlers(drogon::HttpAppFramework& app);

// Sets the directory under which generated videos are stored. The actual
// storage is <dir>/videos; each video is written as <random-uuid>.mp4 and the
// UUID is used as the job/video id returned by the API. Must be called before
// register_api_handlers().
void set_video_storage_dir(const std::filesystem::path& dir);

}  // namespace ovserver