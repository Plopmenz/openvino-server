{
  description = "openvino-server: OpenAI-compatible image/text generation server for OpenVINO GenAI on Drogon";

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

    intel-npu-compiler = pkgs.stdenv.mkDerivation rec {
  pname = "intel-npu-compiler";
  version = "8.1.0";

  src = pkgs.fetchurl {
    url = "https://github.com/openvinotoolkit/npu_compiler/releases/download/npu_ud_2026_38_rc1/l_vpux_compiler_l0_linux_ubuntu_24_04-7_10_0-Release_dyntbb_postcommit_cid_0b38f7d42113ff329ac2bdd33583d123de4ccf2f_260902_1709.tar.gz";
    hash = "sha256-hdAS06yitjDMNJcs5ABoBOs3e0PF6MNn6UpAjK7cvsM=";
  };

  nativeBuildInputs = [
    pkgs.autoPatchelfHook
  ];

  buildInputs = [
    pkgs.onetbb
    pkgs.level-zero
    pkgs.stdenv.cc.cc.lib
    pkgs.libz
    pkgs.zstd
  ];

  unpackPhase = ''
    runHook preUnpack
    tar -xf $src
    runHook postUnpack
  '';

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall
    mkdir -p $out/lib
    cp -v cid/lib/libopenvino_intel_npu_compiler.so $out/lib/
    cp -v cid/lib/libopenvino_intel_npu_compiler_loader.so $out/lib/
    cp -v cid/lib/libopenvino_intel_npu_vm_runtime.so $out/lib/
    runHook postInstall
  '';

  meta = {
    description = "Intel NPU Compiler-in-Driver: libopenvino_intel_npu_compiler_loader.so and libopenvino_intel_npu_compiler.so";
    homepage = "https://github.com/openvinotoolkit/npu_compiler";
    license = pkgs.lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = [ pkgs.lib.sourceTypes.binaryNativeCode ];
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
          rev = "cae33e7271d2a066929e6d1e53d7d2a8747c6a8e";
          fetchSubmodules = true;
          hash = "sha256-mq/5uIp5CKA5pyHRtI5EJPnKl1ioxG5PzG6woFNjJ3Y=";
        };
        cmakeFlags = (pkgs.lib.remove (pkgs.lib.cmakeBool "ENABLE_ONEDNN_FOR_GPU" false) old.cmakeFlags) ++ [ (pkgs.lib.cmakeBool "ENABLE_ONEDNN_FOR_GPU" true) ];
        postInstall = (old.postInstall or "") + ''
           ln -s ${intel-npu-compiler}/lib/libopenvino_intel_npu_compiler_loader.so \
             ${placeholder "lib"}/lib/openvino/libopenvino_intel_npu_compiler_loader.so
           ln -s ${intel-npu-compiler}/lib/libopenvino_intel_npu_compiler.so \
             ${placeholder "lib"}/lib/openvino/libopenvino_intel_npu_compiler.so
           ln -s ${intel-npu-compiler}/lib/libopenvino_intel_npu_vm_runtime.so \
             ${placeholder "lib"}/lib/openvino/libopenvino_intel_npu_vm_runtime.so
        '';
      });

      # Graphics-constrained generation backend for structured output
      # (json_schema / regex / EBNF). xgrammar is built inside the openvino-genai
      # derivation via FetchContent; nix feeds it the source below (its own build
      # uses vendored 3rdparty deps only, so nothing else needs fetching).
      xgrammar-src-raw = pkgs.fetchFromGitHub {
        owner = "mlc-ai";
        repo = "xgrammar";
        rev = "v0.1.31";
        hash = "sha256-YFuH1HytDSiBRLs2cEkO2brGm8DgwcTeTRsBKf6lmrY=";
      };

      # xgrammar bundles dlpack as a git submodule, so it is absent from the
      # GitHub tarball; fetch it separately.
      dlpack-src = pkgs.fetchFromGitHub {
        owner = "dmlc";
        repo = "dlpack";
        rev = "v0.7";
        hash = "sha256-wDPWTEeEfcqtGHv5Q6oANYGu1ZhYDqNHPt9K4zQ0Jgo=";
      };

      # xgrammar shipped with Python bindings enabled by default in its
      # cmake/config.cmake. The nix store is read-only, so openvino-genai's
      # FetchContent step cannot rewrite it there; pre-patch a writable copy
      # so the in-build copy builds without nanobind, and restore the dlpack
      # submodule that the archive omits.
      xgrammar-src = pkgs.runCommand "xgrammar-src-0.1.31" { } ''
        mkdir -p $out
        cp -r ${xgrammar-src-raw}/. $out/
        chmod -R u+w $out
        mkdir -p $out/3rdparty/dlpack
        cp -r ${dlpack-src}/. $out/3rdparty/dlpack/
        substituteInPlace $out/cmake/config.cmake \
          --replace-fail \
            'set(XGRAMMAR_BUILD_PYTHON_BINDINGS ON)' \
            'set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)'
        # xgrammar forces -flto=auto on GCC, producing an LTO archive that the
        # non-LTO openvino_genai .so link cannot consume (its symbols stay
        # undefined in libopenvino_genai.so). Build plain objects instead.
        substituteInPlace $out/CMakeLists.txt \
          --replace-fail \
            '-Wno-error=free-nonheap-object -flto=auto' \
            '-Wno-error=free-nonheap-object'
      '';

      # openvino-genai built from master so that Qwen-Image support is available.
      # The nixpkgs package pins 2026.2.0.0 which predates it.
      # Uses overrideAttrs so the C++ package inherits nixpkgs build flags,
      # with our custom source/rev.
      openvino-genai = (pkgs.openvino-genai.override { inherit openvino openvino-tokenizers; }).overrideAttrs (old: {
        version = "master-2026-09-01";
        src = pkgs.fetchFromGitHub {
          owner = "openvinotoolkit";
          repo = "openvino.genai";
          rev = "3abf349be2c53f911d5de6edc2744e20c1780ba5";
          hash = "sha256-O+Ub30ZSNbml50gg2ee/J8Lbmj9n1oqjPBvJHef21aM=";
        };
        # Remove stale patch
        patches = [ ];
        # xgrammar is supplied pre-patched (Python bindings already OFF), so
        # drop openvino-genai's read-only rewrite of the config (comment the
        # file(WRITE) out so nothing is written; the read is harmless).
        postPatch = ''
          substituteInPlace src/cpp/CMakeLists.txt \
            --replace-fail \
              '            file(WRITE "''${CONFIG_FILE}" "''${MODIFIED_CONFIG}")' \
              '            # file(WRITE "''${CONFIG_FILE}" "''${MODIFIED_CONFIG}")'
        '';
        # nixpkgs disables xgrammar; flip it on and feed the source through
        # FetchContent so structured output works. genai's own test binaries
        # don't link the xgrammar static lib (link errors), and we don't run
        # its check suite here, so tests are disabled.
        cmakeFlags = (pkgs.lib.remove (pkgs.lib.cmakeBool "ENABLE_XGRAMMAR" false) old.cmakeFlags)
          ++ [
            (pkgs.lib.cmakeBool "ENABLE_XGRAMMAR" true)
            (pkgs.lib.cmakeBool "ENABLE_TESTS" false)
            (pkgs.lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_XGRAMMAR" "${xgrammar-src}")
          ];
        doCheck = false;
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
          rev = "971e835eea677e7cd3eda8ad4e582525846c1df5";
          hash = "sha256-ZfHzwFSMpR/EJlazoucZnBAA5y3+dqvPI6Oz9PZuGBc=";
        };
        patches = [ ./nix/openvino-tokenizers-use-system-pcre2-and-sentencepiece-binary-dir.patch ];
      });
    in
    {
      packages.${system} = {
        default = pkgs.callPackage ./nix/server.nix {
          inherit openvino openvino-genai openvino-tokenizers;
        };
        inherit openvino openvino-genai openvino-tokenizers;
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.cmake
          pkgs.ninja
          pkgs.pkg-config
          pkgs.openssl
          pkgs.nlohmann_json
          pkgs.ffmpeg
          openvino-tokenizers
          openvino
          openvino-genai
          pkgs.drogon
        ];

        shellHook = ''
          export OpenVINOGenAI_DIR="${pkgs.openvino-genai.dev}/lib/cmake"
          export nlohmann_json_DIR="${pkgs.nlohmann_json}/lib/cmake/nlohmann_json"
          export Drogon_DIR="${pkgs.drogon}/lib/cmake/Drogon"
          export OpenVINO_DIR="${pkgs.openvino.dev}/lib/cmake"
          export CMAKE_PREFIX_PATH="${pkgs.openvino-genai.dev}:${pkgs.nlohmann_json}:${pkgs.drogon}:${pkgs.openvino}/runtime"
          # genai dlopens libopenvino_tokenizers.so (see tokenizer/tokenizers_path.cpp).
          export OPENVINO_TOKENIZERS_PATH_GENAI="${pkgs.openvino-tokenizers}/lib/libopenvino_tokenizers.so"
          export LD_LIBRARY_PATH="${pkgs.openvino-genai}/lib:${pkgs.openvino}/runtime/lib:${pkgs.drogon}/lib:''${LD_LIBRARY_PATH:-}"
          echo "openvino-server dev shell ready."
          echo "  cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release"
          echo "  cmake --build build"
          echo "  ./build/openvino-server --txt2img /path/to/qwen-image"
        '';
      };
    };
}
