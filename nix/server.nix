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
  openvino-genai-dev,
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
    ] ++ lib.optionals (builtins.pathExists (openvino-genai + "/lib/cmake/OpenVINOGenAI"))
      [ openvino-genai ];

  buildInputs =
    [ nlohmann_json openvino openvino-genai openvino-tokenizers drogon ];

  # Explicitly specify CMake config directories, bypassing the nixpkgs cmake
  # wrapper that disables CMAKE_PREFIX_PATH search via hardcoded registry flags.
  buildCommand =
    let
      genai_dev = openvino-genai.dev;
      openvino_lib = "${openvino}/runtime/lib/intel64";
      genai_lib = "${openvino-genai}/lib";
      tokenizers_lib = "${openvino-tokenizers}/lib";
    in
    ''
      mkdir -p $out/bin
      cd ${finalAttrs.src}
      g++ -std=c++17 -O2 -o $out/bin/vlm_test_gpu \
        -I${genai_dev}/include \
        -I${openvino}/runtime/include \
        -L${tokenizers_lib} -lopenvino_tokenizers \
        -L${genai_lib} -lopenvino_genai \
        -L${openvino_lib} -lopenvino \
        src/vlm_test_gpu.cc \
        -Wl,-rpath,${tokenizers_lib}:${genai_lib}:${openvino_lib}
      wrapProgram $out/bin/vlm_test_gpu \
        --set OPENVINO_TOKENIZERS_PATH_GENAI "${openvino-tokenizers}/lib/libopenvino_tokenizers.so"
    '';

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