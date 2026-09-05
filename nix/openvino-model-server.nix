{
  lib,
  stdenv,
  fetchFromGitHub,
  cacert,
  pkgs,
  makeWrapper,
  autoPatchelfHook,
  bazel_7,
  git,
  gzip,
  unzip,
  which,
  python3,

  # OVMS consumes these prebuilt packages directly via `new_local_repository`
  # pointing at per-workspace "nix-runtime" symlink trees. We do NOT use
  # buildBazelPackage: upstream WORKSPACE repos for these are swapped for
  # local repositories so nothing is compiled through bazel for them.
  openvino,
  openvino-genai,
  opencv,
  curl,
  openssl,
  drogon,
  libgit2,
  tbb,
  ocl-icd,
}:

let
  pname = "openvino-model-server";
  version = "2026.3.1";

  # Use plain bazel_7 (no enableNixHacks): the nixpkgs nix-hacks patch breaks
  # git_repository fetches that carry a ``patches`` attr (mediapipe), leaving an
  # empty repo dir and producing "Every .bzl file must have a corresponding
  # package" errors in the main repo mapping pass. The unpatched bazel handles
  # patches correctly and needs only the writable output/install roots we pass
  # explicitly (bazelOut/bazelUserRoot) plus the tools on PATH.
  bazel = bazel_7;

  src = fetchFromGitHub {
    owner = "openvinotoolkit";
    repo = "model_server";
    rev = "3a28d490b704fc7021ced337d2240abe818a1e09";
    hash = "sha256-7XcETkFiMLWI7LHrThlcedrpYHhxBXPfwe/Qw9toSVs=";
  };

  # Match the upstream release build: default -c opt comes from .bazelrc.
  # MEDIAPIPE/PYTHON/CLOUD must all be disabled together; genai/llm is then
  # also excluded by OVMS itself (`//:not_disable_mediapipe` selects).
  bazelFlags = [
    # Force classic WORKSPACE mode: OVMS has no MODULE.bazel, and bazel 7's
    # default bzlmod breaks WORKSPACE load ordering for fetch-on-load.
    "--noenable_bzlmod"
    # mdl dpf python etc
    "--repo_env=PYTHON_BIN_PATH=${python3}/bin/python3"
    "--define=MEDIAPIPE_DISABLE=1"
    "--define=PYTHON_DISABLE=1"
    "--define=CLOUD_DISABLE=1"
  ];

  # Shared for both the fixed-output fetch phase and the offline build phase.
  commonPreHook = ''
    export bazelOut="$NIX_BUILD_TOP/output"
    export bazelUserRoot="$NIX_BUILD_TOP/tmp"
    export HOME="$NIX_BUILD_TOP"
    export USER="nix"
    # Needed by git_repository / other bazel fetchers over https.
    export GIT_SSL_CAINFO="${cacert}/etc/ssl/certs/ca-bundle.crt"
    export SSL_CERT_FILE="${cacert}/etc/ssl/certs/ca-bundle.crt"
  '';

  # Bridge dirs + BUILD files that redirect bazel ``new_local_repository``
  # slots to nix store packages. Relative paths only, so it produces the same
  # WORKSPACE in the fetch phase and in the build phase.
  prepWorkspace = ''
    cp ${./WORKSPACE} WORKSPACE
    cp ${./mediapipe-nix.patch} mediapipe-nix.patch
    # nixpkgs drogon (1.9.12) has no MultiPartParser::getParametersVector()
    # (drogon never shipped such an API); adapt OVMS to use getParameters().
    cp ${./drogon-nix.patch} drogon-nix.patch
    patch -p1 -i drogon-nix.patch

    # OVMS expects trantor's (non-existent) TcpConnection::getCloseCallback().
    cp ${./drogon-disconnect.patch} drogon-disconnect.patch
    patch -p1 -i drogon-disconnect.patch

    # Provide OVMS's libgit2 LFS-cancellation accessors (nix libgit2 lacks the patch).
    cp ${./libgit2-lfs-cancel.patch} libgit2-lfs-cancel.patch
    patch -p1 -i libgit2-lfs-cancel.patch

    # Expose nix-runtime/openvino/include/CL/* (opencl cl2.hpp) through the
    # openvino_new_headers glob so <CL/cl2.hpp> resolves from OVMS sources.
    cp ${./openvino-new-headers.patch} openvino-new-headers.patch
    patch -p1 -i openvino-new-headers.patch

    # OVMS pins an old bazel (6.1.1) in .bazelversion; the nixpkgs bazel
    # wrapper dispatches on it, so align it with the bazel we provide.
    echo '${bazel.version}' > .bazelversion

    rm -rf nix-runtime
    # --- linux_openvino (@//third_party/openvino:BUILD) ---
    mkdir -p nix-runtime/openvino/include nix-runtime/openvino/lib
    ln -s ${openvino.dev}/include/openvino nix-runtime/openvino/include/openvino
    # openvino/runtime/intel_gpu/ocl needs the OpenCL C++ wrapper (<CL/cl2.hpp>).
    mkdir -p nix-runtime/openvino/include/CL
    ln -s ${pkgs.opencl-headers}/include/CL/* nix-runtime/openvino/include/CL/
    ln -s ${pkgs.opencl-clhpp}/include/CL/* nix-runtime/openvino/include/CL/
    # openvino/runtime/intel_gpu/ocl/va.hpp needs VA-API headers.
    ln -s ${pkgs.libva.dev}/include/va nix-runtime/openvino/include/va
    ln -s ${openvino}/lib nix-runtime/openvino/lib/intel64
    # --- linux_genai (@//third_party/genai:BUILD) ---
    mkdir -p nix-runtime/genai/include nix-runtime/genai/lib
    ln -s ${openvino-genai}/runtime/include/openvino nix-runtime/genai/include/openvino
    ln -s ${openvino-genai}/runtime/lib/intel64 nix-runtime/genai/lib/intel64
    # --- linux_opencv (@//third_party/opencv:BUILD) ---
    mkdir -p nix-runtime/opencv
    ln -s ${opencv}/include nix-runtime/opencv/include
    ln -s ${opencv}/lib nix-runtime/opencv/lib
    # --- linux_curl (curl.BUILD) ---
    mkdir -p nix-runtime/curl/include nix-runtime/curl/lib
    ln -s ${curl.dev}/include/curl nix-runtime/curl/include/curl
    ln -s ${curl}/lib/libcurl.so nix-runtime/curl/lib/libcurl.so
    # --- boringssl replacement (@//third_party/boringssl:BUILD) ---
    mkdir -p nix-runtime/openssl/include nix-runtime/openssl/lib/x86_64-linux-gnu
    ln -s ${openssl.dev}/include/openssl nix-runtime/openssl/include/openssl
    ln -s ${openssl.out}/lib/libssl.so nix-runtime/openssl/lib/x86_64-linux-gnu/libssl.so
    ln -s ${openssl.out}/lib/libcrypto.so nix-runtime/openssl/lib/x86_64-linux-gnu/libcrypto.so
    # --- libgit2_engine (libgit2.BUILD) ---
    mkdir -p nix-runtime/libgit2/include nix-runtime/libgit2/lib
    ln -s ${libgit2.dev}/include/git2 nix-runtime/libgit2/include/git2
    ln -s ${libgit2.lib}/lib/libgit2.so nix-runtime/libgit2/lib/libgit2.so
    # --- drogon (drogon.BUILD) ---
    mkdir -p nix-runtime/drogon/include nix-runtime/drogon/lib
    ln -s ${drogon}/include/drogon nix-runtime/drogon/include/drogon
    ln -s ${drogon}/include/trantor nix-runtime/drogon/include/trantor
    ln -s ${drogon}/lib/libdrogon.a nix-runtime/drogon/lib/libdrogon.a
    ln -s ${drogon}/lib/libtrantor.a nix-runtime/drogon/lib/libtrantor.a

    # --- Stub repositories (empty dirs; real build never uses them) ---
    mkdir -p nix-runtime/stubs/aws-sdk-cpp \
             nix-runtime/stubs/azure \
             nix-runtime/stubs/cpprest \
             nix-runtime/stubs/boost \
             nix-runtime/stubs/google-cloud-cpp-common \
             nix-runtime/stubs/google-cloud-cpp \
             nix-runtime/stubs/pip_deps
    echo 'all_requirements = []' > nix-runtime/stubs/pip_deps/requirements.bzl

    # Stub build_bazel_rules_apple: TF's real rules use apple_common.multi_arch_split,
    # removed in bazel 7. Nothing in the graph instantiates iOS rules; only the .bzl
    # loads need to resolve.
    mkdir -p nix-runtime/stubs/rules-apple/apple/internal
    touch nix-runtime/stubs/rules-apple/apple/BUILD \
          nix-runtime/stubs/rules-apple/apple/internal/BUILD
    cat > nix-runtime/stubs/rules-apple/apple/ios.bzl <<'EOF'
def ios_unit_test(**kwargs):
    pass
EOF
    cat > nix-runtime/stubs/rules-apple/apple/apple.bzl <<'EOF'
def apple_static_xcframework(**kwargs):
    pass
EOF
    cat > nix-runtime/stubs/rules-apple/apple/apple_static_library.bzl <<'EOF'
def apple_static_library(**kwargs):
    pass
EOF
    cat > nix-runtime/stubs/rules-apple/apple/internal/resources.bzl <<'EOF'
def resources(**kwargs):
    pass
EOF
    cat > nix-runtime/stubs/rules-apple/apple/repositories.bzl <<'EOF'
def apple_rules_dependencies(ignore_version_differences = False):
    pass
EOF

    cat > curl.BUILD <<'EOF'
cc_library(
    name = "curl",
    hdrs = glob(["include/curl/*"]),
    srcs = glob(["lib/libcurl.so"]),
    linkopts = ["-lcrypto", "-lssl", "-lz"],
    visibility = ["//visibility:public"],
)
EOF

    cat > libgit2.BUILD <<'EOF'
cc_library(
    name = "libgit2_engine",
    hdrs = glob(["include/git2/**/*.h"]),
    srcs = glob(["lib/libgit2.so"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)
EOF

    # T::Drogon/trantor are static-only in nixpkgs; their transitive deps are
    # provided through the global --linkopt path from buildInputs.
    cat > drogon.BUILD <<'EOF'
cc_library(
    name = "drogon_cmake",
    hdrs = glob(["include/**/*.h", "include/**/*.hpp"]),
    srcs = glob(["lib/libdrogon.a", "lib/libtrantor.a"]),
    includes = ["include"],
    linkopts = [
        "-ljsoncpp",
        "-lbrotlidec",
        "-lbrotlienc",
        "-luuid",
        "-lrt",
        "-lpthread",
        "-ldl",
    ],
    visibility = ["//visibility:public"],
)
EOF

    cat > stubs.BUILD <<'EOF'
cc_library(
    name = "storage",
    visibility = ["//visibility:public"],
)
cc_library(
    name = "sdk",
    visibility = ["//visibility:public"],
)
cc_library(
    name = "boost",
    visibility = ["//visibility:public"],
)
cc_library(
    name = "aws-sdk-cpp",
    visibility = ["//visibility:public"],
)
cc_library(
    name = "stub",
    visibility = ["//visibility:public"],
)
EOF
  '';

  runtimeLibs = [
    openvino
    openvino-genai
    opencv
    curl
    openssl
    drogon
    libgit2
    tbb
    ocl-icd
    pkgs.zlib
    pkgs.jsoncpp
    pkgs.brotli
    pkgs.libxml2
    pkgs.libva
    pkgs.util-linux
  ];

  # Phase 1: fetch & configure all bazel external repositories (network).
  # Reimplemented fixed-output derivation, mirroring nixpkgs'
  # buildBazelPackage fetch phase but a plain derivation.
  deps =
    stdenv.mkDerivation {
      name = "${pname}-${version}-deps.tar";

      inherit src;

      nativeBuildInputs = [
        bazel
        git
        unzip
        which
        python3
        gzip
      ] ++ runtimeLibs;

      impureEnvVars = lib.fetchers.proxyImpureEnvVars;

      outputHashAlgo = "sha256";
      outputHashMode = "flat";
      outputHash = "sha256-2y1VJLnsSYyFG9Gsn48Uhe55H6B66XorCcvQ+T75wsI=";
      allowedRequisites = [ ];
      # The tar contains /nix/store strings (env configs, symlink rewrites) that
      # are not real runtime dependencies; phase 2 regenerates them.
      __structuredAttrs = true;
      unsafeDiscardReferences = { out = true; };

      preHook = commonPreHook;

      buildPhase = ''
        runHook preBuild
        ${prepWorkspace}

        ${bazel}/bin/bazel \
          --batch \
          --output_base="$bazelOut" \
          --output_user_root="$bazelUserRoot" \
          build --nobuild \
          --curses=no \
          --loading_phase_threads=1 \
          --jobs="$NIX_BUILD_CORES" \
          ${lib.escapeShellArgs bazelFlags} \
          //src:ovms
        runHook postBuild
      '';

      installPhase = ''
        runHook preInstall

        # Only bazel_tools is dropped here: it is regenerated by the bazel binary
        # itself at startup. rules_cc is a real http_archive (pulled in via
        # rules_python rulesets) that would otherwise be refetched; it must
        # stay packed in the tar so the build phase stays offline.
        rm -rf $bazelOut/external/{bazel_tools,@bazel_tools.marker}
        # Embedded JDK is bundled inside the bazel nix package.
        rm -rf $bazelOut/external/{embedded_jdk,@embedded_jdk.marker}
        # Auto-detected configs are regenerated at build time.
        rm -rf $bazelOut/external/{local_*,@local_*.marker}

        # Keep all remaining @*.marker files intact: the unpatched bazel
        # validates them and reuses the matching content, so no repo is
        # refetched during the offline build phase.

        # Drop vcs metadata that is not needed at build time.
        find $bazelOut/external -type d \( -name .git -o -name .svn -o -name .hg \) -exec rm -rf {} +

        # Remove top-level symlinks together with their markers: they can
        # point at temporary/absolute paths that differ per phase.
        find $bazelOut/external -maxdepth 1 -type l | while read symlink; do
          name="$(basename "$symlink")"
          rm "$symlink"
          test -f "$bazelOut/external/@$name.marker" && rm "$bazelOut/external/@$name.marker" || true
        done

        # Rewrite the leftover symlinks so they no longer reference the
        # build directory; the build phase re-maps NIX_BUILD_TOP.
        find $bazelOut/external -type l | while read symlink; do
          new_target="$(readlink "$symlink" | sed "s,$NIX_BUILD_TOP,NIX_BUILD_TOP,")"
          rm "$symlink"
          ln -sf "$new_target" "$symlink"
        done

        echo '${bazel.name}' > $bazelOut/external/.nix-bazel-version

        (cd $bazelOut && tar cf $out --sort=name --mtime='@1' --owner=0 --group=0 --numeric-owner external/)

        runHook postInstall
      '';

      dontFixup = true;
    };
in
stdenv.mkDerivation {
  inherit
    pname
    version
    src
    ;

  deps = deps;

  # Forward proxy env vars used during the fixed-output fetch phase, so any
  # lazy/regenerated repo fetch during the build phase can also reach the
  # network (host requires a proxy; see the deps derivation).
  impureEnvVars = lib.fetchers.proxyImpureEnvVars;

  nativeBuildInputs = [
    bazel
    autoPatchelfHook
    makeWrapper
    git
    unzip
    which
    python3
    gzip
  ] ++ runtimeLibs;

  preHook = commonPreHook;

  preConfigure = ''
    mkdir -p "$bazelOut"

    (cd $bazelOut && tar xf $deps)

    test '${bazel.name}' = "$(<$bazelOut/external/.nix-bazel-version)" || {
      echo "fixed output derivation was built for a different bazel version" >&2
      echo "     got: $(<$bazelOut/external/.nix-bazel-version)" >&2
      echo "expected: ${bazel.name}" >&2
      exit 1
    }

    chmod -R +w $bazelOut
    find $bazelOut -type l | while read symlink; do
      if [[ $(readlink "$symlink") == *NIX_BUILD_TOP* ]]; then
        ln -sf $(readlink "$symlink" | sed "s,NIX_BUILD_TOP,$NIX_BUILD_TOP,") "$symlink"
      fi
    done

    # cxxopts (pinned by OVMS) omits <cstdint>; gcc 15 requires the typedefs.
    # Patch the header in place after unpacking the fixed-output tar.
    sed -i '1i #include <cstdint>' \
      $bazelOut/external/com_github_jarro2783_cxxopts/include/cxxopts.hpp
  '';

  # Bazel sandboxes the execution of the tools it invokes, so pass the nix
  # compiler flags explicitly instead of relying on env the wrappers expect.
  preBuild = ''
    copts=()
    host_copts=()
    linkopts=()
    host_linkopts=()
    for flag in $NIX_CFLAGS_COMPILE; do
      copts+=( "--copt=$flag" )
      host_copts+=( "--host_copt=$flag" )
    done
    for flag in $NIX_CXXSTDLIB_COMPILE; do
      copts+=( "--copt=$flag" )
      host_copts+=( "--host_copt=$flag" )
    done
    for flag in $NIX_LDFLAGS; do
      linkopts+=( "--linkopt=-Wl,$flag" )
      host_linkopts+=( "--host_linkopt=-Wl,$flag" )
    done
    # Bazel omits the prebuilt shared libs (linux_openvino, curl.BUILD, ...)
    # from this fully-static binary link unless a -l tells the linker about
    # them; the -L dirs come from NIX_LDFLAGS above.
    for lib in openvino curl git2 cares sqlite3; do
      linkopts+=( "--linkopt=-l$lib" )
    done
    ${prepWorkspace}
  '';

  buildPhase = ''
    runHook preBuild
    ${bazel}/bin/bazel \
      --batch \
      --output_base="$bazelOut" \
      --output_user_root="$bazelUserRoot" \
      build \
      --curses=no \
      --jobs="$NIX_BUILD_CORES" \
      --loading_phase_threads="$NIX_BUILD_CORES" \
      "''${copts[@]}" "''${host_copts[@]}" "''${linkopts[@]}" "''${host_linkopts[@]}" \
      ${lib.escapeShellArgs bazelFlags} \
      //src:ovms \
      || {
        # Dump the failing linker params so we can inspect -l/-L ordering.
        echo "==== CWD=$(pwd)"
        ls -la bazel-out/k8-opt/bin/src/*.params 2>&1 || true
        for p in $(find -L bazel-out -name '*.params' 2>/dev/null); do
          echo "==== link params: $p"
          tail -n 500 "$p"
        done
        exit 1
      }
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin
    cp bazel-bin/src/ovms $out/bin/ovms
    chmod +x $out/bin/ovms
    runHook postInstall
  '';

  postFixup = ''
    # dlopened OpenVINO plugins & frontends are looked up relative to where
    # libopenvino.so is found at runtime; this is a safety net.
    wrapProgram $out/bin/ovms \
      --prefix LD_LIBRARY_PATH : "${openvino}/lib:${opencv}/lib:${tbb}/lib:${ocl-icd}/lib"
  '';

  meta = {
    description = "Intel's serving engine for AI models, built against a custom OpenVINO master stack";
    homepage = "https://github.com/openvinotoolkit/model_server";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ ];
    platforms = lib.platforms.linux;
  };
}