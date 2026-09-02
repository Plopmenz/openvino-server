#!/usr/bin/env python3
"""Load an exported Qwen-Image IR on a device via optimum and generate.

Stand-in for the genai C++ server path: both run core.compile_model on the
same submodels, so if this generates on GPU the server load is out of the
compile-failure hypothesis.

Text-to-image never invokes vae_encoder (only vae_decoder decodes), so its
GPU-unfriendly compile is monkeypatched into a no-op; the three stages the
pipeline actually uses (text_encoder / transformer / vae_decoder) are compiled
on the requested device.
"""
import os
import sys

from optimum.intel import OVQwenImagePipeline

model_dir = sys.argv[1] if len(sys.argv) > 1 else "Qwen-Image-2512-ov2"
device = sys.argv[2] if len(sys.argv) > 2 else "GPU"
prompt = "a red fox in a snowy forest, photorealistic"
ov_config = {}
if os.environ.get("OV_FP32"):
    ov_config["INFERENCE_PRECISION_HINT"] = "f32"
    print("forcing FP32 compute (INFERENCE_PRECISION_HINT=f32)", flush=True)

if device != "CPU":
    import optimum.intel.openvino.modeling_diffusion as md

    try:
        md.OVModelVaeEncoder.compile = lambda self: None
        print("vae_encoder compile skipped (unused in text2image)", flush=True)
    except AttributeError:
        print("warning: could not patch OVModelVaeEncoder.compile", file=sys.stderr)

print(f"loading {model_dir} on '{device}' ...", flush=True)
pipe = OVQwenImagePipeline.from_pretrained(model_dir, device=device, ov_config=ov_config)
print("loaded OK on GPU", flush=True)

img = pipe(
    prompt=prompt,
    negative_prompt="",
    width=512,
    height=512,
    num_inference_steps=4,
    true_cfg_scale=4.0,
).images[0]
img.save("out_gpu.png")
print("generated OK -> out_gpu.png", flush=True)