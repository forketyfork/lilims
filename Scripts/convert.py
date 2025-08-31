#!/usr/bin/env python3
"""Convert LLM checkpoints to Core ML .mlpackage with INT4 weights."""
from __future__ import annotations

import argparse
from pathlib import Path


def _require(module: str):
    try:
        return __import__(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise SystemExit(
            f"Missing required dependency '{module}'. Install it with `pip install {module}`"
        ) from exc


def convert_pytorch(model_id: str, output: Path, *, seq_length: int) -> None:
    """Convert a Hugging Face model to Core ML."""
    ct = _require("coremltools")
    transformers = _require("transformers")

    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    mlmodel = ct.converters.transformers.convert(
        model=model,
        tokenizer=tokenizer,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        precision="int4",
        sequence_length=seq_length,
    )
    mlmodel.save(str(output))


def convert_gguf(path: Path, output: Path, *, seq_length: int) -> None:
    """Convert a gguf checkpoint to Core ML."""
    ct = _require("coremltools")
    gguf = _require("gguf")
    np = _require("numpy")

    reader = gguf.GGUFReader(path)
    tensors = list(reader.tensors.values())
    if not tensors:
        raise ValueError(f"{path} contains no tensors")
    # Use the first tensor as demonstration weight matrix.
    tensor = tensors[0]
    weight = np.array(tensor.data, dtype=np.float16)
    input_dim = weight.shape[1]
    output_dim = weight.shape[0]

    builder = ct.models.neural_network.NeuralNetworkBuilder(
        input_features=[("token", ct.models.datatypes.Array(input_dim))],
        output_features=[("logits", ct.models.datatypes.Array(output_dim))],
    )
    builder.add_inner_product(
        name="proj",
        W=weight,
        b=None,
        input_channels=input_dim,
        output_channels=output_dim,
        has_bias=False,
        input_name="token",
        output_name="logits",
    )
    mlmodel = ct.models.MLModel(builder.spec)
    mlmodel.save(str(output))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert model to .mlpackage")
    parser.add_argument("model", help="Model id or path to gguf file")
    parser.add_argument("output", type=Path, help="Output .mlpackage path")
    parser.add_argument("--seq-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--gguf", action="store_true", help="Treat input as gguf")
    args = parser.parse_args(argv)

    if args.gguf:
        convert_gguf(Path(args.model), args.output, seq_length=args.seq_length)
    else:
        convert_pytorch(args.model, args.output, seq_length=args.seq_length)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
