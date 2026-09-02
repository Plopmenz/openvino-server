#!/usr/bin/env python3
"""Export a HF diffusers pipeline to OpenVINO IR (optimum-intel).

Builds an OpenVINO-format model directory that openvino-server can load.

Requires:
    pip install optimum-intel diffusers transformers --upgrade

Usage:
    python tools/export_model.py --model Qwen/Qwen-Image --output /models/qwen-image

The OPTIMAL target produces a QwenImagePipeline config.json entry that the
OpenVINO GenAI C++ API Text2ImagePipeline auto-detects.
"""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a diffusers pipeline to OpenVINO IR"
    )
    parser.add_argument(
        "--model", required=True, help="HF model id or local path, e.g. Qwen/Qwen-Image"
    )
    parser.add_argument("--output", required=True, help="Output directory (OpenVINO model dir)")
    parser.add_argument("--device", default="CPU", help="Inference device to target (CPU/GPU)")
    args = parser.parse_args()

    from optimum.intel import OVQwenImagePipeline

    os.makedirs(args.output, exist_ok=True)
    print(f"Exporting {args.model} -> {args.output} (device={args.device})")
    ov_pipe = OVQwenImagePipeline.from_pretrained(
        args.model, export=True, device=args.device
    )
    ov_pipe.save_pretrained(args.output)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())