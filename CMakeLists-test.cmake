cmake_minimum_required(VERSION 3.16)
project(vlm_test_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINOGenAI REQUIRED)

add_executable(vlm_test_gpu src/vlm_test_gpu.cc)
target_link_libraries(vlm_test_gpu PRIVATE openvino::genai)
