#!/usr/bin/env python3
"""Generate a manifest JSON file for a converted Core ML model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def generate_manifest(model_path: Path, runtime_version: str, *, name: str | None = None) -> dict[str, object]:
    """Return manifest data for *model_path*."""
    if name is None:
        name = model_path.stem
    size = model_path.stat().st_size
    with model_path.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    return {
        "name": name,
        "size": size,
        "sha256": digest,
        "runtime_version": runtime_version,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate manifest.json")
    parser.add_argument("model", type=Path, help="Path to .mlpackage")
    parser.add_argument("output", type=Path, help="Path to manifest.json")
    parser.add_argument("--runtime-version", required=True, help="Runtime version")
    parser.add_argument("--name", help="Model display name")
    args = parser.parse_args(argv)

    manifest = generate_manifest(
        args.model, args.runtime_version, name=args.name
    )
    with args.output.open("w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()

