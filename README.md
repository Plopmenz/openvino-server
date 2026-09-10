# openvino-server

Serve OpenVINO GenAI image-generation, text-generation, video-generation,
speech-recognition and text-to-speech models over an OpenAI-compatible HTTP API
using the [Drogon](https://github.com/drogonframework/drogon) C++ web framework.

Supported models: anything the OpenVINO GenAI `Text2ImagePipeline` (e.g.
**Qwen-Image**), `ContinuousBatchingPipeline` (e.g. **Qwen2.5-VL**,
**Qwen3-VL**), `Text2VideoPipeline` (e.g. **LTX-Video**), `ASRPipeline` (e.g.
**Qwen3-ASR**, **Whisper**) and `Text2SpeechPipeline` (e.g. **SpeechT5**,
**Kokoro**) understand, exported to OpenVINO IR.

## Endpoints

| Method | Path                       | Purpose                                        |
| ------ | -------------------------- | ---------------------------------------------- |
| GET    | `/v1/models`               | List loaded models                             |
| POST   | `/v1/images/generations`   | Generate image(s) from a text prompt           |
| POST   | `/v1/video/generations`    | Generate video clip(s) from a text prompt      |
| POST   | `/v1/audio/transcriptions` | Transcribe an uploaded audio file (multipart)  |
| POST   | `/v1/audio/speech`         | Synthesize speech (WAV/PCM) from text          |
| POST   | `/v1/chat/completions`     | Text completion (streaming supported)          |
| GET    | `/health`                  | Liveness check                                 |

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
./build/openvino-server --model /models/qwen-image --txt2img --device CPU \
  --port 8080
```

`--model PATH` names an exported OpenVINO GenAI model directory. The endpoint
flags (`--txt2img`, `--txt2txt`, `--txt2vid`, `--wav2txt`, `--txt2wav`) enable
the model at the corresponding endpoint, so a multi-capable model directory can
be served at several endpoints at once:

```sh
./build/openvino-server --model /models/qwen3 --txt2txt
```

Only one `--model` is served per instance; to serve more models, run another
instance (e.g. on another `--port`). `--model-id` names the id served as
`model` (default: the model directory name). Requests whose `model` field does
not match that id exactly are rejected with 404.

Options:

```
      --model PATH       Path to an exported OpenVINO GenAI model directory
                         (e.g. Qwen-Image, Qwen2.5-VL, LTX-Video, Qwen3-ASR,
                         Kokoro). Exactly one; run another instance of the
                         server to serve more models.
      --model-id ID      Model id served as 'model' (default: the directory
                         name).
      --txt2img          Serve the model on /v1/images/generations (image
                         generation).
      --txt2txt          Serve the model on /v1/chat/completions (text
                         generation).
      --txt2vid          Serve the model on /v1/videos and
                         /v1/video/generations (video generation).
      --wav2txt          Serve the model on /v1/audio/transcriptions (speech
                         recognition).
      --txt2wav          Serve the model on /v1/audio/speech (text-to-speech).
      --kv-cache-precision TYPE
                         KV cache element type for text models on GPU
                         (default: u8)
    --dynamic-quant-gsize N
                         Dynamic quantization group size for GPU text
                         inference (default: 32)
    --enable-sdpa BOOL   Enable SDPA optimization for GPU text inference
                         (default: true)
    --cache-interval-multiplier N
                         Linear-attention KV checkpoint interval in KV blocks
                         (default: 64)
    --no-prefix-caching  Disable KV-block prefix caching (enabled by default).
                         Caching retains previously computed KV blocks to reuse
                         shared prompt prefixes across requests (better TTFT).
    --prompt-lookup      Enable prompt-lookup speculative decoding: draft
                         candidates by n-gram matching against the prompt. No
                         extra model needed.
    --num-assistant-tokens N
                         Draft tokens proposed per speculative-step (default: 5)
    --max-ngram-size N   Max n-gram size for prompt-lookup matching (default: 3)
    --no-mtp             Disable auto-detection of a bundled MTP (Multi-Token
                         Prediction) head. Models exported with
                         openvino_mtp_model.xml (e.g. Qwen3.6/3.8) enable MTP
                         speculative decoding automatically.
    --ffmpeg PATH        ffmpeg binary for audio decode / MP4 encode
                         (default: "ffmpeg" on PATH; empty disables those)
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

### Reasoning

For models whose chat template emits thinking markers (e.g. Qwen3's
` thinking ...  response` or DeepSeek-R1 family), the server automatically
splits the output: reasoning is returned under `message.reasoning_content`
(non-streaming) or streamed chunk-by-chunk as `delta.reasoning_content`
despite `delta.content`, and the markers themselves are never sent. Detection
is heuristic and keyed off the **chat template**, not the model id; when no
known markers are found, the raw text is returned unchanged. Nothing is forced:
whether a model "thinks" follows its own template default (the server does not
inject an `enable_thinking` option).

### Function calling

OpenAI `tools` and `tool_choice` are supported. Definitions are injected into
the model's own chat template, and tool calls are parsed out of the response:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3",
    "messages": [
      { "role": "user", "content": "What is the weather in New York?" }
    ],
    "tools": [
      { "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Current weather for a city",
          "parameters": {
            "type": "object",
            "properties": { "location": { "type": "string" } },
            "required": ["location"]
          }
        } }
    ],
    "tool_choice": "auto"
  }'
```

### Speculative decoding

Text models transparently benefit from two speed-ups, configured per model at
startup (not per request):

- **Prefix caching** (on by default): previously computed KV blocks are kept in
  memory and reused whenever a request shares a prompt prefix with an earlier
  one. In multi-turn chat (repeated system prompt, growing history) this
  reduces time-to-first-token significantly. The retained blocks are evicted
  only when the cache budget is exhausted, so the only cost is VRAM. Disable
  with `--no-prefix-caching`.
- **Prompt lookup** (`--prompt-lookup`): drafts candidate tokens by n-gram
  matching against the prompt, then verifies them in a single forward pass.
  No second model or extra memory is needed; most effective for input-grounded
  workloads (RAG, summarization, code editing). Controls:
  `--num-assistant-tokens N` (default 5) and `--max-ngram-size N` (default 3).
- **Bundled MTP head** (auto-detected): when a model directory contains
  `openvino_mtp_model.xml` (how optimum-intel exports Qwen3.6/3.8-family
  models), MTP speculative decoding is enabled automatically — the model's own
  Multi-Token-Prediction head drafts several tokens per step. Disable with
  `--no-mtp`. MTP requires greedy decoding (`temperature` unset) and is
  mutually exclusive with prompt lookup (prompt lookup wins if both are
  requested).

The assistant message then contains `tool_calls`
(`[{id, type, function:{name, arguments}}]` with `arguments` as a JSON string)
and `finish_reason: "tool_calls"`. The full conversation history holds:
`tool_choice: "none"` disables injection; `"required"` or a
`{function:{name}}` object encourages a call (offering only that function);
`"auto"` (default) lets the model decide. Multi-turn exchanges are supported —
feed the previous `assistant` message with its `tool_calls`, then a `tool`
message (`{"role":"tool","content":result,"tool_call_id":id}`), then another
`user` message.

Tool parsing recognizes the `<tool_call>{...}</tool_call>` (Hermes/Qwen3) and
bare JSON-object (Llama-3.1) formats. The server does not force a schema for
tool calls (that is opt-in via `response_format`); if a model family's format
is not recognized, `tools` are still rendered by the template but `tool_calls`
are not extracted and the raw text is returned.

### Structured output

OpenAI `response_format` with `json_schema` or `json_object` constrains
generation so the output matches the requested structure (guided decoding via
the xgrammar backend):

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3",
    "messages": [
      { "role": "user",
        "content": "Return the weather for New York as a JSON object." }
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "weather",
        "schema": {
          "type": "object",
          "properties": {
            "city": { "type": "string" },
            "temp_c": { "type": "number" },
            "conditions": { "type": "string" }
          },
          "required": ["city", "temp_c", "conditions"]
        }
      }
    }
  }'
```

`"type": "json_object"` constrains the output to any valid JSON object;
`"type": "text"` (the default) leaves generation unconstrained. The schema
follows the JSON Schema subset understood by xgrammar; unsupported constructs
cause a 400 error at request time.

## Video generation

```sh
curl http://localhost:8080/v1/video/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ltx-video",
    "prompt": "a robot walking through a rainy city street at night",
    "negative_prompt": "blurry, low quality",
    "guidance_scale": 3.0,
    "steps": 40,
    "size": "512x512",
    "num_frames": 97,
    "fps": 30,
    "seed": 42
  }'
```

`response_format` defaults to `"b64_json"` (base64-encoded MP4, H.264); set it
to `"url"` to write MP4s into an `output_dir` and return filesystem paths.
`n` (1-4) generates multiple clips, `duration` (seconds) is an alternative to
`num_frames`, and `output_format` accepts only `"mp4"`. Encoding is done by the
`ffmpeg` binary (see `--ffmpeg`).

## Audio transcription

`/v1/audio/transcriptions` is a multipart/form-data upload, matching the OpenAI
client:

```sh
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.mp3 -F model=qwen3-asr \
  -F language=en -F response_format=verbose_json
```

Any ffmpeg-decodable audio is accepted; it is resampled to 16 kHz mono before
recognition. `response_format` may be `json` (default, `{"text": ...}`), `text`
(plain text) or `verbose_json` (adds `language`, `duration` and `segments`).
`timestamp_granularities` may be `["segment"]` (word-level timestamps are not
supported).

Setting the multipart field `stream` to `true` (with `response_format` `json` or
`text`) transcribes incrementally over Server-Sent Events, matching the
OpenAI-compatible streaming used by SGLang/vLLM-family servers: zero or more
`transcript.text.delta` events followed by one `transcript.text.done` event with
the full transcript, then `data: [DONE]`.

```sh
curl -N http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.mp3 -F model=qwen3-asr \
  -F language=en -F response_format=json -F stream=true
```

## Speech synthesis

```sh
curl http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{ "model": "kokoro", "input": "Hello from OpenVINO.", "response_format": "wav" }' \
  -o speech.wav
```

`response_format` is `wav` (default) or `pcm` (raw signed 16-bit little-endian
mono). `voice` and `speed` are accepted for OpenAI-client compatibility; the
loaded model's own voice/speaker is used and non-1.0 speeds are not applied.

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