"""Helper to download the TinyStories evaluation corpus."""
from __future__ import annotations

import urllib.request
from pathlib import Path

_DATA_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStories_all_data.txt?download=1"
)


def ensure_dataset(cache_dir: Path | None = None) -> Path:
    """Return the path to the TinyStories dataset, downloading if needed.

    The file is cached under ``cache_dir`` (defaults to ``~/.cache/lilims``).
    Later calls reuse the cached file.
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "lilims"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "TinyStories_all_data.txt"
    if dest.exists():
        return dest
    try:
        with urllib.request.urlopen(_DATA_URL) as response, dest.open("wb") as fh:
            fh.write(response.read())
    except OSError as exc:  # pragma: no cover - network errors are rare
        raise RuntimeError(
            "Failed to download TinyStories dataset"
        ) from exc
    return dest


__all__ = ["ensure_dataset"]
