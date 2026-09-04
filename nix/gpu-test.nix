{
  lib,
  python3,
  openvino-genai,
}: # nixpkgs openvino-genai package provides the Python bindings for OpenVINO
   # GenAI. This script tests GPU model loading through the Python API.
stdenv.mkDerivation {
  pname = "gpu-test";
  version = "0.1.0";

  src = lib.cleanSource ../.;

  nativeBuildInputs = [
    python3.pkgs.makeWrapper
  ];

  buildCommand =
    let
      genai = openvino-genai;
    in
    ''
      mkdir -p $out/bin
      cp ${src}/tools/gpu_test.py $out/bin/gpu_test.py
      makeWrapper \
        "${python3}/bin/python3" \
        $out/bin/gpu_test \
        --set PYTHONPATH "${genai}/lib/python" \
        --add-flags "${src}/tools/gpu_test.py"
    '';

  buildInputs = [ openvino-genai ];

  meta = {
    description = "Test GPU model loading through OpenVINO GenAI Python bindings";
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
  };
}
