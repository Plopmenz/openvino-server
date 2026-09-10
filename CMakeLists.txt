cmake_minimum_required(VERSION 3.16)
project(openvino-server VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

find_package(OpenVINOGenAI REQUIRED)
find_package(Drogon REQUIRED)
find_package(Threads REQUIRED)

add_executable(openvino-server
    src/main.cc
    src/image_generation.cc
    src/manager.cc
    src/controller.cc
    src/image.cc
    src/image_decode.cc
    src/text_generation.cc
    src/video_generation.cc
    src/audio.cc
    src/asr.cc
    src/tts.cc
    src/base64.cc
)

target_include_directories(openvino-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(openvino-server PRIVATE
    openvino::genai
    Drogon::Drogon
    Threads::Threads
)

install(TARGETS openvino-server
    RUNTIME DESTINATION bin
    COMPONENT runtime
)
