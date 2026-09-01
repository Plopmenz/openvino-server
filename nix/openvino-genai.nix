{
  lib,
  stdenv,
  autoPatchelfHook,
  cmake,
  fetchFromGitHub,
  nlohmann_json,
  ocl-icd,
  opencl-clhpp,
  opencl-headers,
  onetbb,
  openvino,
  pkg-config,
  src,
}:

let
  inherit (lib) cmakeBool cmakeFeature;

  # Third-party sources pulled in by openvino.genai's CMake FetchContent. They
  # are pre-seeded so the build never touches the network.
  minja-src = fetchFromGitHub {
    owner = "google";
    repo = "minja";
    rev = "3e4c61c616eda133cfb1e440fc7a14bf1729bbee";
    hash = "sha256-aqOpLNB7XiY/W2gDRtnAqx37gMhHMRCJQmcX24vGIxA=";
  };

  safetensors-h-src = fetchFromGitHub {
    owner = "hsnyder";
    repo = "safetensors.h";
    rev = "974a85d7dfd6e010558353226638bb26d6b9d756";
    hash = "sha256-sBgm4panHB9X2RghC3Qi0wEL0k9uUz+h60pfxTGZ0BA=";
  };

  gguflib-src = fetchFromGitHub {
    owner = "Lourdle";
    repo = "gguf-tools";
    rev = "bac796ada809ac293e685db59b075971181cb008";
    hash = "sha256-yoIjTATYs2IzT/LnCz+G6PtxVXZ27B0mZOIipZbaOI8=";
  };
in

stdenv.mkDerivation (finalAttrs: {
  pname = "openvino-genai";
  version = "master-2026-09-01";

  inherit src;

  outputs = [ "out" ];

  nativeBuildInputs = [
    autoPatchelfHook
    cmake
    pkg-config
  ];

  buildInputs = [
    nlohmann_json
    ocl-icd
    opencl-clhpp
    opencl-headers
    onetbb
    openvino
  ];

  patches = [ ];

  cmakeFlags = [
    (cmakeFeature "OpenVINO_DIR" "${openvino}/runtime/cmake")
    # Pre-seed FetchContent targets.
    (cmakeFeature "FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON" "${nlohmann_json.src}")
    (cmakeFeature "FETCHCONTENT_SOURCE_DIR_MINJA" "${minja-src}")
    (cmakeFeature "FETCHCONTENT_SOURCE_DIR_SAFETENSORS.H" "${safetensors-h-src}")
    (cmakeFeature "FETCHCONTENT_SOURCE_DIR_GGUFLIB" "${gguflib-src}")

    # Normalise install destinations to the standard Nix layout.
    (cmakeFeature "ARCHIVE_DESTINATION" "lib")
    (cmakeFeature "LIBRARY_DESTINATION" "lib")
    (cmakeFeature "RUNTIME_DESTINATION" "bin")

    (cmakeBool "BUILD_TOKENIZERS" false)
    (cmakeBool "ENABLE_PYTHON" false)
    (cmakeBool "ENABLE_JS" false)
    (cmakeBool "ENABLE_SAMPLES" false)
    (cmakeBool "ENABLE_TESTS" false)
    (cmakeBool "ENABLE_TOOLS" false)
    # GGUF is required even for non-GGUF pipelines: continuous-batching sources
    # include gguf_utils/gguf.hpp, which unconditionally needs <gguflib.h>.
    (cmakeBool "ENABLE_GGUF" true)
    (cmakeBool "ENABLE_XGRAMMAR" false)
    (cmakeBool "ENABLE_SYSTEM_OPENCL" true)
  ];

  # Keep CMake's native install layout ($out/runtime/{include,cmake} +
  # $out/lib) so the installed OpenVINOGenAITargets.cmake's _IMPORT_PREFIX
  # resolution keeps working without path rewriting.
  postFixup = ''
    # Help autoPatchelfHook resolve libopenvino.so.*, which openvino installs
    # to runtime/lib/intel64/ instead of a standard lib dir.
    autoPatchelfLibs+=("$out/lib" "${openvino}/runtime/lib/intel64")
  '';

  meta = {
    homepage = "https://github.com/openvinotoolkit/openvino.genai";
    description = "OpenVINO GenAI toolkit (C++ API), built from master with Qwen-Image support";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ ];
    platforms = lib.platforms.linux;
  };
})