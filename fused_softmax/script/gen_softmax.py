#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


BLOCKS = 8
SOFTMAX_GROUPS_PER_BLOCK = 2
SOFTMAX_LENGTH = 1024
SHAPE = (BLOCKS, SOFTMAX_GROUPS_PER_BLOCK, SOFTMAX_LENGTH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FP16 input and an FP16 NumPy golden output for "
            "softmax__kernel0. The kernel shape is fixed at (8, 2, 1024)."
        )
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--input", default="softmax_input_fp16.bin")
    parser.add_argument("--golden", default="softmax_golden_fp16.bin")
    args = parser.parse_args()

    if not np.isfinite(args.scale) or args.scale <= 0.0:
        parser.error("--scale must be a finite positive number")

    rng = np.random.default_rng(seed=args.seed)

    # Quantize the input to FP16 before computing the reference, because those
    # exact FP16 values are what the device kernel receives.
    x = rng.normal(loc=0.0, scale=args.scale, size=SHAPE).astype(np.float16)

    # Compute the mathematical reference in FP32, then round to FP16 because
    # the kernel output is FP16. The device performs its subtraction in FP16,
    # so the host verifier uses tolerances rather than requiring bit equality.
    x32 = x.astype(np.float32)
    shifted = x32 - np.max(x32, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    golden = softmax.astype(np.float16)

    input_path = Path(args.input)
    golden_path = Path(args.golden)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.parent.mkdir(parents=True, exist_ok=True)

    x.tofile(input_path)
    golden.tofile(golden_path)

    sums = golden.astype(np.float32).sum(axis=-1)
    print("Generated softmax test data")
    print(f"  shape: {SHAPE}")
    print(f"  input:  {input_path} ({x.nbytes} bytes)")
    print(f"  golden: {golden_path} ({golden.nbytes} bytes)")
    print("  softmax axis: -1")
    print(
        "  FP16 golden row-sum range: "
        f"[{float(sums.min()):.8f}, {float(sums.max()):.8f}]"
    )


if __name__ == "__main__":
    main()
