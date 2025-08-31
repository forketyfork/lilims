#!/usr/bin/env python3
"""Generate a manifest JSON file for a converted Core ML model."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


def validate_semantic_version(version: str) -> bool:
    """Validate that a version string follows semantic versioning (X.Y.Z or X.Y.Z-prerelease)."""
    pattern = r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'
    return bool(re.match(pattern, version))


def generate_manifest(
    model_path: Path, 
    runtime_version: str, 
    *, 
    name: str | None = None,
    model_version: str | None = None,
    description: str | None = None
) -> dict[str, object]:
    """Return manifest data for *model_path*."""
    if name is None:
        name = model_path.stem
    
    # Validate semantic versions if provided
    if model_version and not validate_semantic_version(model_version):
        raise ValueError(f"Invalid model version '{model_version}'. Must follow semantic versioning (X.Y.Z)")
    
    if not validate_semantic_version(runtime_version):
        raise ValueError(f"Invalid runtime version '{runtime_version}'. Must follow semantic versioning (X.Y.Z)")
    
    size = model_path.stat().st_size
    with model_path.open("rb") as fh:
        digest = hashlib.file_digest(fh, "sha256").hexdigest()
    
    manifest = {
        "name": name,
        "size": size,
        "sha256": digest,
        "runtime_version": runtime_version,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    
    # Add optional fields if provided
    if model_version:
        manifest["model_version"] = model_version
    if description:
        manifest["description"] = description
    
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate manifest.json for a CoreML model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s model.mlpackage manifest.json --runtime-version 1.0.0
  %(prog)s model.mlpackage manifest.json --runtime-version 1.0.0 --model-version 2.1.0 --name "GPT-2 125M"
        """.strip()
    )
    parser.add_argument("model", type=Path, help="Path to .mlpackage")
    parser.add_argument("output", type=Path, help="Path to manifest.json")
    parser.add_argument("--runtime-version", required=True, 
                       help="Runtime version (must follow semantic versioning)")
    parser.add_argument("--name", help="Model display name")
    parser.add_argument("--model-version", 
                       help="Model version (must follow semantic versioning)")
    parser.add_argument("--description", help="Model description")
    args = parser.parse_args(argv)

    try:
        manifest = generate_manifest(
            args.model, 
            args.runtime_version, 
            name=args.name,
            model_version=args.model_version,
            description=args.description
        )
        with args.output.open("w") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        print(f"Manifest generated: {args.output}")
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()

