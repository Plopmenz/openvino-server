#!/usr/bin/env python3
"""Test GPU model loading through OpenVINO GenAI Python bindings."""

import sys

import openvino as ov
import openvino_genai as ovgenai


def main():
    core = ov.Core()
    print("Beschikbare apparaten:", core.available_devices)

    if "GPU" in core.available_devices:
        print("\n=== GPU CAPABILITIES ===")
        capabilities = core.get_property("GPU", "OPTIMIZATION_CAPABILITIES")
        print("Optimization Capabilities:", capabilities)

        full_name = core.get_property("GPU", "FULL_DEVICE_NAME")
        print("Full Device Name:", full_name)

        supported_props = core.get_property("GPU", "SUPPORTED_PROPERTIES")
        print("\nOndersteunde properties van de actieve driver:")
        for prop in supported_props:
            print(f" - {prop}")

    if len(sys.argv) < 3:
        print(
            "Usage: gpu_test <model-path> <prompt> [device:GPU|CPU] [max_new_tokens:N]"
        )
        sys.exit(1)

    model_path = sys.argv[1]
    prompt = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "GPU"
    max_new_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 64

    print(f"OpenVINO version: {ov.__version__}")
    print(f"\nModel: {model_path}")
    print(f"Device: {device}")
    print(f"\nAttempting LLMPipeline on GPU...")

    try:
        pipe = ovgenai.LLMPipeline(model_path, device)
        print("SUCCESS: LLMPipeline created on GPU")
        result = pipe.generate(prompt, max_new_tokens=max_new_tokens)
        print(f"\nResult: {result}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
