from __future__ import annotations

from pathlib import Path
import sys
import builtins

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

import Scripts.evaluate_perplexity as ep


def test_compute_perplexity_missing_dep(monkeypatch, tmp_path):
    data = tmp_path / "tiny.txt"
    data.write_text("hi")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"transformers", "torch"}:
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit):
        ep.compute_perplexity("model", data)
