#!/usr/bin/env python3
"""Convert LLM checkpoints to Core ML .mlpackage with INT4 weights.

This script loads a Hugging Face Transformers model (or a gguf checkpoint when
the `--gguf` flag is supplied) and converts it into a Core ML package. The
resulting `.mlpackage` is quantized to 4-bit weights using Core ML Tools.

The implementation is intentionally minimal; it assumes the environment has
`coremltools` and `transformers` available.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def convert_pytorch(model_id: str, output: Path, *, seq_length: int) -> None:
    """Convert a Hugging Face model to Core ML."""
    import coremltools as ct
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    mlmodel = ct.converters.transformers.convert(
        model=model,
        tokenizer=tokenizer,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        precision="int4",
        sequence_length=seq_length,
    )
    mlmodel.save(str(output))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert model to .mlpackage")
    parser.add_argument("model", help="Model id or path to gguf file")
    parser.add_argument("output", type=Path, help="Output .mlpackage path")
    parser.add_argument("--seq-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--gguf", action="store_true", help="Treat input as gguf")
    args = parser.parse_args(argv)

    if args.gguf:
        raise NotImplementedError("gguf conversion not implemented yet")
    convert_pytorch(args.model, args.output, seq_length=args.seq_length)


if __name__ == "__main__":
    main()
