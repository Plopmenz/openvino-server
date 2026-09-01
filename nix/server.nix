{
  lib,
  stdenv,
  autoPatchelfHook,
  makeWrapper,
  cmake,
  ninja,
  nlohmann_json,
  openvino,
  openvino-tokenizers,
  openvino-genai,
  drogon,
}:

# openvino / openvino-genai install their shared libraries under
# runtime/lib/intel64/ (native OpenVINO layout), but autoPatchelfHook only
# searches <dep>/lib. Expose a plain lib/ symlink farm so the loader can
# resolve the versioned libopenvino*.so.2650 / libopenvino_genai*.so.2650
# sonames the server binary asks for.
let
  openvino-runtime-libs = stdenv.mkDerivation {
    pname = "openvino-runtime-libs";
    version = openvino.version;
    dontUnpack = true;
    installPhase = ''
      runHook preInstall
      mkdir -p "$out/lib"
      for d in \
        "${openvino}/runtime/lib/intel64" \
        "${openvino-genai}/runtime/lib/intel64"
      do
        for f in "$d"/*.so*; do
          [ -e "$f" ] || continue
          ln -s "$f" "$out/lib/$(basename "$f")"
        done
      done
      runHook postInstall
    '';
  };
in

stdenv.mkDerivation (finalAttrs: {
  pname = "openvino-server";
  version = "0.1.0";

  src = lib.cleanSource ../.;

  nativeBuildInputs = [
    autoPatchelfHook
    cmake
    makeWrapper
    ninja
  ];

  buildInputs = [
    openvino-runtime-libs
    nlohmann_json
    openvino
    openvino-genai
    openvino-tokenizers
    drogon
  ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "CMAKE_INSTALL_PREFIX" "${placeholder "out"}")
    (lib.cmakeFeature "OpenVINOGenAI_DIR" "${openvino-genai}/runtime/cmake")
    (lib.cmakeFeature "nlohmann_json_DIR" "${nlohmann_json}/lib/cmake/nlohmann_json")
    (lib.cmakeFeature "Drogon_DIR" "${drogon}/lib/cmake/Drogon")
    (lib.cmakeFeature "OpenVINO_DIR" "${openvino}/runtime/cmake")
  ];

  # Shared libraries (openvino-genai, openvino) and their transitive deps live
  # outside the nix store prefix of this output; autoPatchelfHook fixes the
  # RPATH so the binary runs without plumbing LD_LIBRARY_PATH.
  runtimeDependencies = [
    openvino-runtime-libs
    openvino-genai
    openvino
    drogon
  ];

  # genai locates libopenvino_tokenizers.so through OPENVINO_TOKENIZERS_PATH_GENAI
  # (dlopen, not DT_NEEDED), so it must be set in the runtime environment.
  postInstall = ''
    wrapProgram $out/bin/openvino-server \
      --set OPENVINO_TOKENIZERS_PATH_GENAI \
      "${openvino-tokenizers}/lib/libopenvino_tokenizers.so"
  '';

  meta = {
    description = "OpenAI-compatible image generation server backed by OpenVINO GenAI";
    homepage = "https://github.com/plopmenz/openvino-server";
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
  };
})