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
}: # nixpkgs installs .so files directly under lib/ and CMake config under
   # lib/cmake/OpenVINOGenAI/. This mirrors that layout.
stdenv.mkDerivation (finalAttrs: {
  pname = "openvino-server";
  version = "0.1.0";

  src = lib.cleanSource ../.;

  nativeBuildInputs =
    [
      autoPatchelfHook
      cmake
      makeWrapper
      ninja
    ];

  buildInputs =
    [ nlohmann_json openvino openvino-genai openvino-tokenizers drogon ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "CMAKE_INSTALL_PREFIX" "${placeholder "out"}")
    (lib.cmakeFeature "OpenVINOGenAI_DIR" "${openvino-genai.dev}/lib/cmake")
    (lib.cmakeFeature "nlohmann_json_DIR" "${nlohmann_json}/lib/cmake/nlohmann_json")
    (lib.cmakeFeature "Drogon_DIR" "${drogon}/lib/cmake/Drogon")
    (lib.cmakeFeature "OpenVINO_DIR" "${openvino.dev}/lib/cmake")
  ];

  runtimeDependencies = [
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
