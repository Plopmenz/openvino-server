{
  lib,
  stdenv,
  fetchFromGitHub,
  fetchzip,
  cmake,
  pkg-config,
  onetbb,
  openvino,
}:

let
  inherit (lib) cmakeBool cmakeFeature;

  # The upstream build pulls sentencepiece and pcre2 via FetchContent (URLs).
  # FetchContent needs an unpacked source *directory* (not an archive), so seed
  # them with unpacked sources instead. pcre2 must come from the release
  # archive: the git tree's deps/sljit is a submodule that is not populated, and
  # pcre2's JIT build #includes ../deps/sljit/sljit_src/sljitLir.c.
  sentencepiece-src = fetchFromGitHub {
    owner = "google";
    repo = "sentencepiece";
    rev = "v0.2.1";
    hash = "sha256-q0JgMxoD9PLqr6zKmOdrK2A+9RXVDub6xy7NOapS+vs=";
  };

  pcre2-src = fetchzip {
    url = "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.46/pcre2-10.46.zip";
    hash = "sha256-doOcdsoJKt7l+pt7yGESdf5q0r3phRDxHNGnneUKGWs=";
  };
in

# openvino-tokenizers master, built against our master OpenVINO (2026.5.0).
# The nixpkgs package (2026.2.1.0) is built against OpenVINO 2026.2.x, whose
# ABI soname (libopenvino.so.2620) does not exist in the 2026.5.0 runtime that
# openvino-genai masters requires, so genai's dlopen of the tokenizers plugin
# would fail.
stdenv.mkDerivation (finalAttrs: {
  pname = "openvino-tokenizers";
  version = "2026.5.0.0";

  src = fetchFromGitHub {
    owner = "openvinotoolkit";
    repo = "openvino_tokenizers";
    rev = "4813f2b8dac54c4946ed76d91709f949dfad5455";
    hash = "sha256-Mir9TVsGynI9poPOmMuMVTMCDAKfkyMt+DG7V1ZhZ+c=";
  };

  nativeBuildInputs = [
    cmake
    pkg-config
  ];

  buildInputs = [
    onetbb
    openvino
  ];

  enableParallelBuilding = true;

  cmakeFlags = [
    (cmakeFeature "OpenVINO_DIR" "${openvino}/runtime/cmake")
    # Seed FetchContent sources so nothing is downloaded at build time.
    (cmakeFeature "FETCHCONTENT_SOURCE_DIR_SENTENCEPIECE" "${sentencepiece-src}")
    (cmakeFeature "FETCHCONTENT_SOURCE_DIR_PCRE2" "${pcre2-src}")
    # Install the plugin to standard lib/ rather than runtime/lib/<arch>/.
    (cmakeFeature "OPENVINO_TOKENIZERS_INSTALL_LIBDIR" "lib")
    (cmakeFeature "OPENVINO_TOKENIZERS_INSTALL_BINDIR" "bin")
    (cmakeBool "BUILD_CPP_EXTENSION" true)
    (cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
  ];

  meta = {
    description = "OpenVINO Tokenizers - text tokenisation extensions for OpenVINO (master, 2026.5.0)";
    homepage = "https://github.com/openvinotoolkit/openvino_tokenizers";
    changelog = "https://github.com/openvinotoolkit/openvino_tokenizers";
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
  };
})