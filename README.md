# openvino-server

Serve OpenVINO GenAI image-generation and text-generation models over an
OpenAI-compatible HTTP API using the
[Drogon](https://github.com/drogonframework/drogon) C++ web framework.

Supported models: anything the OpenVINO GenAI `Text2ImagePipeline` (e.g.
**Qwen-Image**) and `ContinuousBatchingPipeline` (e.g. **Qwen2.5-VL**,
**Qwen3-VL**) understand, exported to OpenVINO IR.

## Endpoints

| Method | Path                       | Purpose                               |
| ------ | -------------------------- | ------------------------------------- |
| GET    | `/v1/models`               | List loaded models                    |
| POST   | `/v1/images/generations`   | Generate image(s) from a text prompt  |
| POST   | `/v1/chat/completions`     | Text completion (streaming supported) |
| GET    | `/health`                  | Liveness check                        |

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

A `flake.nix` is provided. Qwen-Image support only exists on `openvino.genai`
master, which tracks OpenVINO master, so the flake builds both OpenVINO and
openvino-genai from their master branches. `openvino-tokenizers` (master) is
built alongside so the tokenizer plugin's ABI matches the master runtime
(`libopenvino.so.2650`) — the nixpkgs package targets 2026.2.x
(`libopenvino.so.2620`) and fails to dlopen against it:

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
./build/openvino-server --txt2img /models/qwen-image \
  --txt2txt /models/qwen2.5-vl --device CPU --port 8080
```

Options:

```
    --txt2img PATH       Path to an exported image-generation model dir
                         (repeatable; at least one image or text model required)
    --txt2img-id ID      Model id served as 'model' (default: qwen-image)
    --txt2txt PATH       Path to an exported text-generation model dir
                         (repeatable)
    --txt2txt-id ID      Model id served as 'model' (default: dir basename)
-d, --device DEVICE      OpenVINO device (default: CPU)
-h, --host HOST          Listen address (default: 0.0.0.0)
-p, --port PORT          Listen port (default: 8080)
-t, --threads N          Drogon event-loop threads (default: 4)
-l, --log-level LEVEL    TRACE|DEBUG|INFO|WARN|ERROR (default: INFO)
-c, --config FILE        Drogon JSON config file
    --idle-timeout SECS  Idle connection timeout (default: 3600)
    --help / -v
```

OpenVINO GenAI locates `libopenvino_tokenizers.so` through the
`OPENVINO_TOKENIZERS_PATH_GENAI` environment variable (it may also look for it
next to `libopenvino_genai.so`). The Nix derivation wraps the binary to set it;
for manual runs point it at your tokenizers library:

```sh
export OPENVINO_TOKENIZERS_PATH_GENAI=/path/to/libopenvino_tokenizers.so
```

## Image generation

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

| Field              | Type   | Notes                                              |
| ------------------ | ------ | -------------------------------------------------- |
| `prompt`           | string | **Required.** The positive prompt.                 |
| `model`            | string | Required when more than one image model is loaded. |
| `n`                | int    | Number of images (1-10, default 1).                |
| `size`             | string | `"WxH"`, e.g. `"1024x1024"`. Snapped to a multiple of 16. |
| `width` / `height` | int    | Alternative to `size` (also snapped to multiple of 16). |
| `output_dir`       | string | If set, PNGs are written here and `url` is a filesystem path instead of base64. |
| `negative_prompt`  | string | OpenAI extension.                                  |
| `guidance_scale`   | number | CFG scale; default from the model.                 |
| `steps`            | int    | Inference steps; default from the model.           |
| `seed`             | int    | RNG seed.                                          |

All fields except `prompt` are optional, so the model's own defaults are
preserved unless overridden.

## Text generation

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-vl",
    "messages": [
      { "role": "user", "content": [
          { "type": "text", "text": "Describe this image." },
          { "type": "image_url",
            "image_url": { "url": "data:image/png;base64,..." } }
      ] }
    ],
    "max_tokens": 256,
    "stream": false
  }'
```

Content may be a plain string or an array of `{type:"text"}` /
`{type:"image_url"}` parts (OpenAI-compatible message content). With
`"stream": true` the response is a Server-Sent Events stream of
`chat.completion.chunk` objects.

## Concurrency

Request handling and response packaging run on Drogon's event loops; blocking
inference runs on a dedicated worker thread pool sized to the CPU. Every image
request `clone()`s the loaded pipeline, which shares the compiled models and
gives each request its own scheduler, so requests can generate in parallel.
Text generations share one continuous-batching pipeline whose scheduler batches
concurrent requests into a shared KV cache.

## Preparing a Qwen-Image model

Export the Hugging Face checkpoint to OpenVINO IR with optimum-intel (requires
the Qwen-Image support present in `optimum-intel` `main`):

```sh
pip install optimum-intel diffusers transformers --upgrade
python tools/export_model.py --model Qwen/Qwen-Image --output /models/qwen-image
```

The exported `config.json` sets `_class_name: "QwenImagePipeline"`, which the
OpenVINO GenAI `Text2ImagePipeline` auto-detects.

## Testing

```sh
tools/smoke_test.sh "a red fox in a snowy forest"
```