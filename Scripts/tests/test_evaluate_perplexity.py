from __future__ import annotations

from __future__ import annotations

from pathlib import Path
import sys
import numpy  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

import Scripts.evaluate_perplexity as ep


def test_main_pass(monkeypatch, tmp_path):
    data = tmp_path / "tiny.txt"
    data.write_text("hello")

    def fake_compute(model_id: str, dataset: Path) -> float:
        assert dataset == data
        return 10.0

    def fake_coreml(model_path: Path, dataset: Path, *, tokenizer_id: str) -> float:
        assert dataset == data
        return 10.2

    monkeypatch.setattr(ep, "compute_perplexity", fake_compute)
    monkeypatch.setattr(ep, "compute_perplexity_coreml", fake_coreml)

    ep.main([
        "ref",
        str(tmp_path / "model.mlpackage"),
        "--dataset",
        str(data),
        "--max-delta",
        "0.05",
    ])


def test_main_fail(monkeypatch, tmp_path):
    data = tmp_path / "tiny.txt"
    data.write_text("hi")

    def fake_compute(model_id: str, dataset: Path) -> float:
        return 10.0

    def fake_coreml(model_path: Path, dataset: Path, *, tokenizer_id: str) -> float:
        return 11.0

    monkeypatch.setattr(ep, "compute_perplexity", fake_compute)
    monkeypatch.setattr(ep, "compute_perplexity_coreml", fake_coreml)

    with pytest.raises(SystemExit):
        ep.main([
            "ref",
            str(tmp_path / "model.mlpackage"),
            "--dataset",
            str(data),
            "--max-delta",
            "0.05",
        ])


def test_main_download(monkeypatch, tmp_path):
    dataset = tmp_path / "download.txt"
    dataset.write_text("story")

    def fake_compute(model_id: str, ds: Path) -> float:
        assert ds == dataset
        return 10.0

    def fake_coreml(model_path: Path, ds: Path, *, tokenizer_id: str) -> float:
        assert ds == dataset
        return 10.1

    monkeypatch.setattr(ep, "compute_perplexity", fake_compute)
    monkeypatch.setattr(ep, "compute_perplexity_coreml", fake_coreml)
    monkeypatch.setattr(ep.tinystories, "ensure_dataset", lambda: dataset)

    ep.main(["ref", str(tmp_path / "model.mlpackage")])
