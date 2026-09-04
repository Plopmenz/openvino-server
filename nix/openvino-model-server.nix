{ lib
, stdenv
, fetchFromGitHub
, buildBazelPackage

# OVMS dependencies from nixpkgs (pre-built)
, bazel
, protobuf_3
, grpc
, glog
, gflags
, spdlog
, fmt
, rapidjson-fastjsonrpc
, prometheus-cpp
, libevent
, abseil-cpp
, nlohmann_json
, cxxopts
, cpp-httplib
, stb
, dr_libs
, curl
, openssl
, python3
, espeak-ng
, git2
, aws-sdk-cpp
, google-cloud-cpp
, drogon
, pugixml
, libjpeg-turbo
, giflib
, snappy
, re2
, flatbuffers
, farmhash
, cpuinfo
, double-conversion
, zlib
, jsoncpp
, libxml2
, libnuma
, libgit2
, nodejs

# OVMS-specific dependencies (custom-built)
, openvino
, opencv

## OVMS v2026.3.1 built against our custom OpenVINO stack
##
## Uses `buildBazelPackage` with a WORKSPACE that replaces all nixpkgs deps
## with `nix_store` references.  TensorFlow and MediaPipe (OVMS fork) build
## from source as the original does.

stdenv.mkDerivation {
  pname = "openvino-model-server";
  version = "2026.3.1";

  src = fetchFromGitHub {
    owner = "openvinotoolkit";
    repo = "model_server";
    rev = "3a28d490b704fc7021ced337d2240abe818a1e09"; # v2026.3.1
    hash = "sha256-PLACEHOLDER"; # nix will compute
  };

  # Bazel build targets
  bazelTargets = [ "//src:ovms" ];

  # Generate WORKSPACE with nix_store overrides
  preBuild = ''
    # Copy the original WORKSPACE and patch it to use nix_store
    cp $src/WORKSPACE WORKSPACE

    # Generate the patched WORKSPACE
    cat > WORKSPACE.patched << 'WORKSPACE_EOF'
    workspace(name = "ovms")
    load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
    load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

    # Load Bazel tooling dependencies
    load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
    bazel_skylib_workspace()
    load("@bazel_skylib//lib:versions.bzl", "versions")
    versions.check(minimum_bazel_version = "6.0.0")

    # Nix store references for all known nixpkgs dependencies
    nix_store "com_github_glog_glog" { src = "${glog}"; }
    nix_store "com_github_gflags_gflags" { src = "${gflags}"; }
    nix_store "com_github_gabime_spdlog" { src = "${spdlog}"; }
    nix_store "fmtlib" { src = "${fmt}"; }
    nix_store "com_google_protobuf" { src = "${protobuf_3}"; }
    nix_store "com_github_grpc_grpc" { src = "${grpc}"; }
    nix_store "linux_curl" { src = "${curl}"; }
    nix_store "boringssl" { src = "${openssl}"; }
    nix_store "com_google_absl" { src = "${abseil-cpp}"; }
    nix_store "com_github_tencent_rapidjson" { src = "${rapidjson-fastjsonrpc}"; }
    nix_store "com_github_jupp0r_prometheus_cpp" { src = "${prometheus-cpp}"; }
    nix_store "com_github_libevent_libevent" { src = "${libevent}"; }
    nix_store "nlohmann_json" { src = "${nlohmann_json}"; }
    nix_store "com_github_jarro2783_cxxopts" { src = "${cxxopts}"; }
    nix_store "cpp_httplib" { src = "${cpp-httplib}"; }
    nix_store "stb" { src = "${stb}"; }
    nix_store "dr_libs" { src = "${dr_libs}"; }
    nix_store "espeak_ng" { src = "${espeak-ng}"; }
    nix_store "libgit2_engine" { src = "${git2}"; }
    nix_store "azure" { src = "${aws-sdk-cpp}"; }
    nix_store "google_cloud_cpp" { src = "${google-cloud-cpp}"; }
    nix_store "drogon" { src = "${drogon}"; }
    nix_store "pugixml" { src = "${pugixml}"; }
    nix_store "libjpeg_turbo" { src = "${libjpeg-turbo}"; }
    nix_store "gif" { src = "${giflib}"; }
    nix_store "snappy" { src = "${snappy}"; }
    nix_store "com_googlesource_code_re2" { src = "${re2}"; }
    nix_store "flatbuffers" { src = "${flatbuffers}"; }
    nix_store "farmhash_archive" { src = "${farmhash}"; }
    nix_store "cpuinfo" { src = "${cpuinfo}"; }
    nix_store "double_conversion" { src = "${double-conversion}"; }
    nix_store "zlib" { src = "${zlib}"; }
    nix_store "jsoncpp" { src = "${jsoncpp}"; }
    nix_store "libxml2" { src = "${libxml2}"; }
    nix_store "numa" { src = "${libnuma}"; }

    # OpenVINO and OpenCV (system/local repos)
    nix_store "linux_openvino" { src = "${openvino}"; }
    nix_store "linux_opencv" { src = "${opencv}"; }

    # TensorFlow builds from source (same as original)
    git_repository(
      name = "org_tensorflow",
      remote = "https://github.com/tensorflow/tensorflow.git",
      commit = "5329ec8dd396487982ef3e743f98c0195af39a6b",
      shallow = true,
    )

    # MediaPipe (OVMS fork) builds from source
    git_repository(
      name = "mediapipe",
      remote = "https://github.com/openvinotoolkit/mediapipe.git",
      commit = "12e8d511cfbc5f471c498278a65a02dd250963e8",
      shallow = true,
    )

    # Python setup
    load("@ovms//third_party/python:python_repo.bzl", "python_repository")
    python_repository(name = "_python3-linux")

    # Third-party build rules
    load("@ovms//third_party/aws-sdk-cpp:aws-sdk-cpp.bzl", "aws_sdk_cpp")
    aws_sdk_cpp()
    load("@ovms//third_party/libgit2:libgit2_engine.bzl", "libgit2_engine")
    libgit2_engine()
    load("@ovms//third_party/espeak_ng:espeak_ng.bzl", "espeak_ng")
    espeak_ng()
    load("@ovms//third_party/drogon:drogon.bzl", "drogon_cpp")
    drogon_cpp()
WORKSPACE_EOF
    cp WORKSPACE.patched WORKSPACE
  '';

  buildInputs = [
    protobuf_3
    grpc
    glog
    gflags
    spdlog
    fmt
    rapidjson-fastjsonrpc
    prometheus-cpp
    libevent
    abseil-cpp
    nlohmann_json
    cxxopts
    cpp-httplib
    stb
    dr_libs
    curl
    openssl
    espeak-ng
    git2
    aws-sdk-cpp
    google-cloud-cpp
    drogon
    pugixml
    libjpeg-turbo
    giflib
    snappy
    re2
    flatbuffers
    farmhash
    cpuinfo
    double-conversion
    zlib
    jsoncpp
    libxml2
    libnuma
    libgit2
  ];

  meta = with lib; {
    description = "OpenVINO Model Server - high-performance model serving for Generative AI and classic deep learning";
    homepage = "https://github.com/openvinotoolkit/model_server";
    license = with licenses; [ apache20 ];
    maintainers = with maintainers; [ ];
    platforms = platforms.linux;
  };
}
