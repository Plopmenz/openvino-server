#!/usr/bin/env python3
"""Probe an exported OpenVINO text-to-image model directory.

For each submodel (text_encoder / transformer / vae_decoder / vae_encoder)
prints every input's partial shape (marking dynamic dims and upper bounds),
then attempts an on-device compile of that submodel alone, reporting which
stages the GPU (or selected device) accepts or rejects.

Usage:
    python tools/ov_probe.py /path/to/model_dir --device GPU
"""
import argparse
import sys
from pathlib import Path

import openvino as ov


def _attr(dim, name, default=None):
    """Read a binding attribute that may be a method or a property."""
    try:
        value = getattr(dim, name)
    except AttributeError:
        return default
    return value() if callable(value) else value


def _dynamic(dim) -> bool:
    return bool(_attr(dim, "is_dynamic", False))


def _max_len(dim) -> int:
    value = _attr(dim, "get_max_length", None)
    if value is None:
        value = _attr(dim, "max_length")
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _has_upper(dim) -> bool:
    # Unbounded dynamic dimensions report dim::inf == -1 as max length.
    return _max_len(dim) >= 0


def probe_submodel(core: ov.Core, xlm_path: Path, device: str) -> None:
    name = xlm_path.parent.name
    print(f"\n=== {name} ({xlm_path.name}) ===")
    try:
        model = core.read_model(str(xlm_path))
    except Exception as exc:  # noqa: BLE001
        print(f"  read_model FAILED: {exc}")
        return

    for inp in model.inputs:
        name_s = inp.get_any_name()
        elem_s = inp.get_element_type().get_type_name()
        ps = inp.get_partial_shape()
        dims = []
        for dim in ps:
            if _dynamic(dim):
                if not _has_upper(dim):
                    dims.append("[-inf,+inf]")
                else:
                    dims.append(f"[..{_max_len(dim)}]")
            else:
                dims.append(dim.get_length())
        print(f"  in {name_s:32s} {elem_s:6s} {'x'.join(map(str, dims))}")
    for out in model.outputs:
        print(f"  out {out.get_any_name():32s} {out.get_partial_shape()}")

    print(f"  -- compile on '{device}' ...", end="", flush=True)
    try:
        core.compile_model(model, device)
        print(" OK")
    except Exception as exc:  # noqa: BLE001
        print(f" FAILED: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        detail = getattr(exc, "__notes__", None)
        if detail:
            print("    notes:", "; ".join(str(d) for d in detail))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()

    core = ov.Core()
    print("Available devices:", core.available_devices)
    if args.device not in core.available_devices and args.device != "AUTO":
        print(f"WARNING: device '{args.device}' not in available devices", file=sys.stderr)

    xmls = sorted(args.model_dir.rglob("openvino_model.xml"))
    if not xmls:
        print(f"no openvino_model.xml found under {args.model_dir}", file=sys.stderr)
        return 1
    for xlm in xmls:
        probe_submodel(core, xlm, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())