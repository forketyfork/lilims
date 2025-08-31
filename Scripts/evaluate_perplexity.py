#!/usr/bin/env python3
"""Evaluate model perplexity on TinyStories."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

try:  # Support both package and script execution
    from . import tinystories
except Exception:  # pragma: no cover - execution as script
    import tinystories  # type: ignore


def _load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _require(module: str):
    """Import *module* or exit with a helpful message."""
    try:
        return __import__(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - simple import guard
        raise SystemExit(
            f"Missing required dependency '{module}'. Install it with `pip install {module}`"
        ) from exc


def compute_perplexity(model_id: str, dataset: Path) -> float:
    """Return perplexity of a Hugging Face model over *dataset*."""
    transformers = _require("transformers")
    torch = _require("torch")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    losses: list[float] = []
    for line in _load_lines(dataset):
        inputs = tokenizer(line, return_tensors="pt")
        labels = inputs["input_ids"]
        with torch.no_grad():
            out = model(**inputs, labels=labels)
        losses.append(out.loss.item())
    return float(math.exp(float(np.mean(losses))))


def compute_perplexity_coreml(model_path: Path, dataset: Path, *, tokenizer_id: str) -> float:
    """Return perplexity for a Core ML model created by ``convert.py``."""
    ct = _require("coremltools")
    transformers = _require("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
    mlmodel = ct.models.MLModel(str(model_path))

    losses: list[float] = []
    for line in _load_lines(dataset):
        tokens = tokenizer.encode(line, add_special_tokens=False)
        tokens.append(tokenizer.eos_token_id)
        state: dict[str, object] | None = None
        for i in range(len(tokens) - 1):
            token = tokens[i]
            expected = tokens[i + 1]
            features: dict[str, object] = {"token": token}
            if state is not None:
                features.update(state)
            out = mlmodel.predict(features)
            state = {k: v for k, v in out.items() if k != "logits"}
            logits = np.array(out["logits"])  # type: ignore[index]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            losses.append(-math.log(float(probs[expected]) + 1e-12))
    return float(math.exp(float(np.mean(losses))))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate quantized perplexity")
    parser.add_argument("baseline", help="Reference model id")
    parser.add_argument("quantized", type=Path, help="Path to .mlpackage")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="TinyStories text file; downloaded if omitted",
    )
    parser.add_argument(
        "--tokenizer-id", default=None, help="Tokenizer id for quantized model"
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=0.03,
        help="Allowed relative perplexity increase",
    )
    args = parser.parse_args(argv)

    dataset = args.dataset or tinystories.ensure_dataset()
    if not dataset.exists():
        raise SystemExit(f"Dataset not found at {dataset}")

    base_ppl = compute_perplexity(args.baseline, dataset)
    tokenizer_id = args.tokenizer_id or args.baseline
    quant_ppl = compute_perplexity_coreml(
        args.quantized, dataset, tokenizer_id=tokenizer_id
    )

    delta = (quant_ppl - base_ppl) / base_ppl
    print(f"Baseline {base_ppl:.2f}, quantized {quant_ppl:.2f}, delta {delta:.2%}")
    if delta > args.max_delta:
        raise SystemExit(
            f"Perplexity increased by {delta:.2%} > {args.max_delta:.2%}"
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
