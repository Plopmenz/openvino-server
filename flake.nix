{
  description = "openvino-server: OpenAI-compatible image generation server for OpenVINO GenAI (Qwen-Image) on Drogon";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixos-unstable/nixexprs.tar.xz";
  };

  outputs =
    {
      self,
      nixpkgs,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
        };
      };

      # Qwen-Image support (openvinotoolkit/openvino.genai#4220) exists only on
      # openvino.genai's master branch, which in turn tracks OpenVINO master
      # (it requires OpenVINO >= 2026.5.0, newer than any release or nixpkgs
      # package). So both OpenVINO and openvino-genai are built from master.
      openvino = pkgs.openvino.overrideAttrs (old: {
        version = "2026.5.0-master";
        src = pkgs.fetchFromGitHub {
          owner = "openvinotoolkit";
          repo = "openvino";
          rev = "227c33757d1ef95d4da506d00686f923fdd2a535";
          fetchSubmodules = true;
          hash = "sha256-Izno1afpXxiqMVrlyDvTHXsnJ9wo0Q63IlEjdKrTzqc=";
        };
      });

      # openvino-genai built from master so that Qwen-Image support is available.
      # The nixpkgs package pins 2026.2.0.0 which predates it.
      # Uses overrideAttrs so the C++ package inherits nixpkgs build flags,
      # with our custom source/rev.
      openvino-genai = (pkgs.openvino-genai.override { inherit openvino openvino-tokenizers; }).overrideAttrs (old: {
        version = "master-2026-09-01";
        src = pkgs.fetchFromGitHub {
          owner = "openvinotoolkit";
          repo = "openvino.genai";
          rev = "7ea2546852a382cd16bd22dea0cfad2db70ed744";
          hash = "sha256-sRJbnXF7/CaHx86+dbIDv9FC1GthMW58vstQ4elf16Q=";
        };
        # Remove stale patch
        patches = [ ];
        postPatch = "";
      });

      # openvino-tokenizers built against our custom OpenVINO so the ABI matches
      # (libopenvino.so.2650). The nixpkgs package ships 2026.2.x which targets
      # libopenvino.so.2620 and would fail to dlopen at runtime on the 2026.5.0
      # stack. Uses overrideAttrs so the C++ package inherits nixpkgs .python output.
      openvino-tokenizers = (pkgs.openvino-tokenizers.override { inherit openvino; }).overrideAttrs (old: {
        version = "master-2026-09-01";
        src = pkgs.fetchFromGitHub {
          owner = "openvinotoolkit";
          repo = "openvino_tokenizers";
          rev = "a04accf6282d9b304214b492694b18c3979f667a";
          hash = "sha256-M+HPqxwCZCBxSmpcnaIPBtEniRD9H1lCFgOrqH/eAFQ=";
        };
        patches = [ ./nix/openvino-tokenizers-use-system-pcre2-and-sentencepiece-binary-dir.patch ];
      });

      openvino-python = pkgs.python3Packages.openvino.override { openvino-native = openvino; };
      openvino-genai-python = pkgs.python3Packages.openvino-genai.override { openvino-tokenizers = openvino-tokenizers-python; openvino-genai-native = openvino-genai; };
       openvino-tokenizers-python = pkgs.python3Packages.openvino-tokenizers.override { openvino = openvino-python; openvino-tokenizers-native = openvino-tokenizers; };

       # OVMS v2026.3.1 built against our custom OpenVINO stack
       openvino-model-server = pkgs.callPackage ./nix/openvino-model-server.nix { };
     in
    {
      packages.${system} = {
        # Quick test with nixpkgs cached packages.
        # NOTE: Qwen-Image support is not available in nixpkgs openvino-genai (2026.2.0.0).
        default-test-pkgs = pkgs.callPackage ./nix/server.nix {
          openvino = pkgs.openvino;
          openvino-genai = pkgs.openvino-genai;
          openvino-tokenizers = pkgs.openvino-tokenizers;
          openvino-genai-dev = pkgs.openvino-genai.dev;
        };
        # GPU model loading test using nixpkgs openvino-genai Python bindings.
        gpu-test = let python = pkgs.python3.withPackages (ps: [ openvino-python openvino-genai-python ]); in pkgs.writeShellApplication {
          name = "gpu-test";
          runtimeInputs = [
            python
          ];
          text = ''
            exec ${python}/bin/python3 ${./tools/gpu_test.py} "$@"
          '';
        };
        # Local build (from GitHub master):
        default = pkgs.callPackage ./nix/server.nix {
          inherit openvino openvino-genai openvino-tokenizers;
          openvino-genai-dev = openvino-genai.dev;
        };
        inherit openvino;
        openvino-genai = openvino-genai;
        openvino-tokenizers = openvino-tokenizers;
        openvino-model-server = openvino-model-server;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          pkgs.openssl
          pkgs.nlohmann_json
          openvino-tokenizers
          openvino
          openvino-genai
          pkgs.drogon
        ];

        shellHook = ''
          export OpenVINOGenAI_DIR="${pkgs.openvino-genai.dev}/lib/cmake/OpenVINOGenAI"
          export Drogon_DIR="${pkgs.drogon}/lib/cmake/Drogon"
          export OpenVINO_DIR="${pkgs.openvino}/runtime/cmake"
          export CMAKE_PREFIX_PATH="${pkgs.openvino-genai.dev}:${pkgs.nlohmann_json}:${pkgs.drogon}:${pkgs.openvino}/runtime"
          # genai dlopens libopenvino_tokenizers.so (see tokenizer/tokenizers_path.cpp).
          export OPENVINO_TOKENIZERS_PATH_GENAI="${pkgs.openvino-tokenizers}/lib/libopenvino_tokenizers.so"
          export LD_LIBRARY_PATH="${pkgs.openvino-genai}/lib:${pkgs.openvino}/runtime/lib:${pkgs.drogon}/lib:''${LD_LIBRARY_PATH:-}"
          echo "openvino-server dev shell ready."
          echo "  cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release"
          echo "  cmake --build build"
          echo "  ./build/openvino-server --model /path/to/qwen-image"
        '';
      };
    };
}
