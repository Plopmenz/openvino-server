{
  description = "openvino-server: OpenAI-compatible image generation server for OpenVINO GenAI (Qwen-Image) on Drogon";

  inputs = {
    # Pinned to the same nixpkgs revision the host system resolves to.
    nixpkgs.url = "github:NixOS/nixpkgs/567a49d1913ce81ac6e9582e3553dd90a955875f";
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
          rev = "ec4aa1d1588029f91e85bd1bf35b86a4f7e6a072";
          fetchSubmodules = true;
          hash = "sha256-Hr7vlXGP1/Jv9yTpEOqM4atqkpbRU0hc2wGBxWTb3EY=";
        };
        meta = old.meta // {
          changelog = "https://github.com/openvinotoolkit/openvino";
        };
      });

      # openvino-genai is built from master so that Qwen-Image support is
      # available. The nixpkgs `openvino-genai` package pins 2026.2.0.0 which
      # predates it.
      openvino-genai = pkgs.callPackage ./nix/openvino-genai.nix {
        inherit openvino;
        src = pkgs.fetchFromGitHub {
          owner = "openvinotoolkit";
          repo = "openvino.genai";
          rev = "6d77d7c7ffc05de9eebe2bda18ac43d6966a7036";
          hash = "sha256-PZlhoj8EZ0S1+rhg5RD3llP+uRfoy0+c1RetIOJINEE=";
        };
      };

      # openvino-tokenizers master, built against the same master OpenVINO so
      # its ABI matches (libopenvino.so.2650). The nixpkgs package ships
      # 2026.2.x which targets libopenvino.so.2620 and would fail to dlopen at
      # runtime on the 2026.5.0 stack.
      openvino-tokenizers = pkgs.callPackage ./nix/openvino-tokenizers.nix {
        inherit openvino;
      };
    in
    {
      packages.${system} = {
        default = pkgs.callPackage ./nix/server.nix {
          inherit openvino openvino-genai openvino-tokenizers;
        };
        inherit openvino;
        openvino-genai = openvino-genai;
        openvino-tokenizers = openvino-tokenizers;
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
          export OpenVINOGenAI_DIR="${openvino-genai}/runtime/cmake"
          export Drogon_DIR="${pkgs.drogon}/lib/cmake/Drogon"
          export OpenVINO_DIR="${openvino}/runtime/cmake"
          export CMAKE_PREFIX_PATH="${openvino-genai}:${pkgs.nlohmann_json}:${pkgs.drogon}:${openvino}/runtime"
          # genai dlopens libopenvino_tokenizers.so (see tokenizer/tokenizers_path.cpp).
          export OPENVINO_TOKENIZERS_PATH_GENAI="${openvino-tokenizers}/lib/libopenvino_tokenizers.so"
          export LD_LIBRARY_PATH="${openvino-genai}/lib:${openvino}/runtime/lib:${pkgs.drogon}/lib:''${LD_LIBRARY_PATH:-}"
          echo "openvino-server dev shell ready."
          echo "  cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release"
          echo "  cmake --build build"
          echo "  ./build/openvino-server --model /path/to/qwen-image"
        '';
      };
    };
}
