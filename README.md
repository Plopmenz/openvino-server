# openvino-server

Serve OpenVINO GenAI image-generation models over an OpenAI-compatible HTTP API
using the [Drogon](https://github.com/drogonframework/drogon) C++ web framework.

Currently supported models: any model the OpenVINO GenAI `Text2ImagePipeline`
understands, starting with **Qwen-Image** (support merged in
openvinotoolkit/openvino.genai#4220).

## Endpoints

| Method | Path                       | Purpose                              |
| ------ | -------------------------- | ------------------------------------ |
| GET    | `/v1/models`               | List loaded models                   |
| POST   | `/v1/images/generations`   | Generate image(s) from a text prompt |
| GET    | `/health`                  | Liveness check                       |

Example using curl:

```sh
curl http://localhost:8080/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-image",
    "prompt": "a red fox in a snowy forest, photorealistic",
    "n": 1,
    "size": "512x512",
    "negative_prompt": "blurry, low quality",
    "guidance_scale": 4.0,
    "steps": 20
  }'
```

Response body:

```json
{
  "created": 1778533200,
  "data": [
    { "b64_json": "<base64-encoded PNG>", "url": null }
  ]
}
```

### Request fields

| Field              | Type   | Notes                                                  |
| ------------------ | ------ | ------------------------------------------------------ |
| `prompt`           | string | **Required.** The positive prompt.                     |
| `model`            | string | Required when more than one model is loaded.           |
| `n`                | int    | Number of images (1-10, default 1). Batched internally.|
| `size`             | string | `"WxH"`, e.g. `"1024x1024"`. Snapped to a multiple of 16. |
| `width` / `height` | int    | Alternative to `size` (also snapped to multiple of 16).|
| `response_format`  | string | Only `b64_json` is returned; `url` reserved.           |
| `output_dir`       | string | If set, PNGs are written here and `url` is a filesystem path instead of base64. |
| `negative_prompt`  | string | OpenAI extension.                                     |
| `guidance_scale`   | number | CFG scale; default from the model (Qwen-Image: 4.0).   |
| `steps`            | int    | Inference steps; default from the model (50).          |
| `seed`             | int    | RNG seed.                                             |

`n`, `negative_prompt`, `guidance_scale`, `size`, `steps`, and `seed` are
OpenAI-compatible extensions; each is optional so the model's own defaults are
preserved unless overridden.

## Building

Dependencies:

- [Drogon](https://github.com/drogonframework/drogon) (v1.9+, installed via its
  CMake package as `Drogon::Drogon`)
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) (built
  with the C++ API, exposes the `openvino_genai` CMake target via
  `find_package(OpenVINOGenAI)`)
- CMake 3.16+

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/openvino;/path/to/genai/dist;/path/to/drogon"
cmake --build build -j
```

`openvino.genai` installs its CMake config under `<prefix>/runtime/cmake`, so add
`<prefix>/runtime` to `CMAKE_PREFIX_PATH` if the config is not found.

### Nix

A `flake.nix` is provided. Because Qwen-Image support only exists on
`openvino.genai` master, which tracks OpenVINO master, the flake builds both
OpenVINO and openvino-genai from their master branches:

```sh
nix build .#default              # builds the openvino-server binary
nix develop                      # shell with all CMake/deps configured
```

`nix develop` sets `OpenVINOGenAI_DIR`, `Drogon_DIR` and `OpenVINO_DIR` so an
in-tree build works out of the box:

```sh
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Running

```sh
./build/openvino-server --model /models/qwen-image \
  --model-id qwen-image --device CPU --port 8080
```

Options:

```
-m, --model PATH        Path to an exported OpenVINO model dir (repeatable; REQUIRED)
-i, --model-id ID       Model id served as 'model' (default: qwen-image)
-d, --device DEVICE     OpenVINO device (default: CPU)
-h, --host HOST         Listen address (default: 0.0.0.0)
-p, --port PORT         Listen port (default: 8080)
-t, --threads N         Drogon event-loop threads (default: 4)
-l, --log-level LEVEL   TRACE|DEBUG|INFO|WARN|ERROR (default: INFO)
-c, --config FILE       Drogon JSON config file
    --cache-dir DIR     OpenVINO model cache directory
    --help / -v
```

For GPU offload use `--device GPU`; the server sets
`hint::performance_mode=THROUGHPUT` automatically for that device.

OpenVINO GenAI locates `libopenvino_tokenizers.so` through the
`OPENVINO_TOKENIZERS_PATH_GENAI` environment variable (it may also look for it
next to `libopenvino_genai.so`). The Nix derivation wraps the binary to set it;
for manual runs point it at your tokenizers library:

```sh
export OPENVINO_TOKENIZERS_PATH_GENAI=/path/to/libopenvino_tokenizers.so
```

### Concurrency

`/v1/images/generations` is non-blocking. Request handling and response
packaging run on Drogon's event loops; the blocking OpenVINO inference runs on a
dedicated worker thread pool sized to the CPU. Each request `clone()`s the
loaded pipeline, which shares the compiled models and gives every request its
own scheduler, so multiple requests can infer in parallel.

## Preparing a Qwen-Image model

Export the Hugging Face checkpoint to OpenVINO IR with optimum-intel (requires
the Qwen-Image support present in `optimum-intel` `main`):

```sh
pip install optimum-intel diffusers transformers --upgrade
python tools/export_model.py --model Qwen/Qwen-Image --output /models/qwen-image
```

The exported `config.json` sets `_class_name: "QwenImagePipeline"`, which the
OpenVINO GenAI `Text2ImagePipeline` auto-detects and routes to the Qwen-Image
pipeline.

## Testing

```sh
tools/smoke_test.sh "a red fox in a snowy forest"
```