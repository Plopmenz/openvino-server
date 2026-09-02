# Copyright (c) 2026
#
# Python OpenVINO stack image: openvino, openvino-tokenizers, openvino-genai
# (built from upstream master git repos so Qwen-Image support and the latest
# runtime fixes are present), plus up-to-date optimum-intel / transformers /
# diffusers for model export and reference inference.
#
# Build (context = the docker/ dir so .dockerignore trims the payload):
#   docker build -t openvino-py -f docker/Dockerfile docker
#
# The OpenVINO stack (openvino / openvino-tokenizers / openvino-genai) is built
# from pinned upstream master revisions because Qwen-Image support
# (openvinotoolkit/openvino.genai#4220) and the runtime GPU fixes only exist on
# master, newer than any tagged release. Everything else tracks the latest
# release on PyPI.
#
# ARG GH_TOKEN: if your build exhausts GitHub's anonymous API rate limit while
# fetching submodules, provide a personal access token and it is threaded into
# git so the build does not prompt for credentials:
#   docker build --build-arg GH_TOKEN=$(cat ~/.gh_token) -t openvino-py -f docker/Dockerfile docker
#
# The image runs as root for maximum hardware-access flexibility (GPU via
# /dev/dri, /dev/level_zero, etc.). Pass through devices with, e.g.:
#   docker run --rm --device /dev/dri -v $PWD:/work openvino-py
ARG OPENVINO_REV=f1a195380ab7a0ee95a94b2d2de655d23d0b784b
ARG GENAI_REV=6d77d7c7ffc05de9eebe2bda18ac43d6966a7036
ARG TOKENIZERS_REV=master

FROM ubuntu:24.04 AS build

ARG OPENVINO_REV
ARG GENAI_REV
ARG TOKENIZERS_REV
ARG GH_TOKEN

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    cmake \
    git \
    ca-certificates \
    curl \
    wget \
    pkg-config \
    perl \
    ccache \
    libtbb-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libva-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    libgl1 \
    libglib2.0-0 \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Thread a GitHub token into git so submodule fetches do not hit the anonymous
# 60 req/hr API rate limit.
RUN if [ -n "$GH_TOKEN" ] && [ "$GH_TOKEN" != "false" ]; then \
        git config --global url."https://${GH_TOKEN}@github.com/".insteadOf "https://github.com/"; \
    fi

WORKDIR /src

# --- OpenVINO (master, with Python + GPU plugin) -------------------------
RUN git clone https://github.com/openvinotoolkit/openvino.git openvino \
    && cd openvino && git checkout ${OPENVINO_REV} \
    && git submodule update --init --recursive

RUN cmake -S openvino -B openvino/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/openvino \
        -DENABLE_PYTHON=ON \
        -DENABLE_SAMPLES=OFF \
        -DENABLE_CPPLINT=OFF \
        -DENABLE_INTEL_NPU_PROTOPIPE=OFF \
        -DENABLE_SYSTEM_LEVEL_ZERO=OFF \
    && cmake --build openvino/build --parallel 8 \
    && cmake --install openvino/build

# --- openvino-tokenizers (master, against the master runtime) ------------
RUN git clone https://github.com/openvinotoolkit/openvino_tokenizers.git tokenizers \
    && cd tokenizers && git checkout ${TOKENIZERS_REV} \
    && git submodule update --init --recursive

RUN cmake -S tokenizers -B tokenizers/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/openvino \
        -DOpenVINO_DIR=/opt/openvino/runtime/cmake \
    && cmake --build tokenizers/build --parallel 8 \
    && cmake --install tokenizers/build \
    && mkdir -p /opt/openvino/python \
    && cp -r tokenizers/python/* /opt/openvino/python/ \
    && cp -r tokenizers/build/python/* /opt/openvino/python/openvino_tokenizers/

# --- openvino-genai (master, against the master runtime) -----------------
RUN git clone https://github.com/openvinotoolkit/openvino.genai.git genai \
    && cd genai && git checkout ${GENAI_REV} \
    && git submodule update --init --recursive

RUN cmake -S genai -B genai/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/openvino \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DENABLE_SYSTEM_ICU=ON \
        -DBUILD_TOKENIZERS=OFF \
        -DENABLE_SAMPLES=OFF \
        -DENABLE_TOOLS=OFF \
        -DENABLE_TESTS=OFF \
        -DOpenVINO_DIR=/opt/openvino/runtime/cmake \
    && cmake --build genai/build --parallel 8 \
    && cmake --install genai/build \
    && cp -r genai/build/openvino_genai/*py* /opt/openvino/python/ 2>/dev/null || true

# --- Runtime image --------------------------------------------------------
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV OpenVINO_DIR=/opt/openvino/runtime/cmake
ENV OpenVINOGenAI_DIR=/opt/openvino/runtime/cmake
ENV OPENVINO_TOKENIZERS_PATH_GENAI=/opt/openvino/runtime/lib/intel64/libopenvino_tokenizers.so
ENV LD_LIBRARY_PATH=/opt/openvino/runtime/lib/intel64:/opt/openvino/runtime/lib:/opt/openvino/runtime/3rdparty/tbb/lib:${LD_LIBRARY_PATH:-}
ENV PATH=/opt/openvino/tools/openvino:${PATH}
# The source-built Python binding (ENABLE_PYTHON=ON) installs under one of
# these two layouts depending on the OpenVINO revision; both are on the path.
ENV PYTHONPATH=/opt/openvino/python:/opt/openvino/runtime/python:${PYTHONPATH:-}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ocl-icd-libopencl1 \
    libva2 \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/openvino /opt/openvino

# --- Python package install -----------------------------------------------
# openvino / openvino_genai / openvino_tokenizers modules come from the source
# build above (on PYTHONPATH). The remaining packages are pinned to latest
# (optimum-intel, transformers, diffusers).
RUN pip install --no-cache-dir --break-system-packages -U \
        pip \
        setuptools \
        wheel \
        numpy \
        pillow \
        optimum-intel \
        diffusers \
        transformers \
        accelerate \
        safetensors \
    && python3 -c "import openvino, openvino_genai, openvino_tokenizers, optimum, transformers, diffusers; print('openvino stack import OK')"

WORKDIR /work
CMD ["python3"]
