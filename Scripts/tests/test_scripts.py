from __future__ import annotations

import subprocess

from pathlib import Path
import sys
import builtins

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_convert_main_gguf():
    import Scripts.convert as convert

    called = {}

    def fake_convert(path: Path, output: Path, *, seq_length: int) -> None:
        called["args"] = (path, output, seq_length)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(convert, "convert_gguf", fake_convert)
    monkeypatch.setattr(convert, "convert_pytorch", lambda *a, **k: None)
    convert.main(["model.gguf", "out.mlpackage", "--gguf", "--seq-length", "32"])
    monkeypatch.undo()

    assert called["args"] == (Path("model.gguf"), Path("out.mlpackage"), 32)


def test_convert_main_pytorch(monkeypatch, tmp_path):
    import Scripts.convert as convert

    called = {}

    def fake_convert(model_id: str, output: Path, *, seq_length: int) -> None:
        called["args"] = (model_id, output, seq_length)

    monkeypatch.setattr(convert, "convert_pytorch", fake_convert)
    convert.main(["mymodel", str(tmp_path / "out.mlpackage"), "--seq-length", "16"])

    assert called["args"] == (
        "mymodel",
        tmp_path / "out.mlpackage",
        16,
    )


def test_convert_pytorch_missing_dep(monkeypatch, tmp_path):
    import Scripts.convert as convert

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "coremltools":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(SystemExit):
        convert.convert_pytorch("model", tmp_path / "out.mlpackage", seq_length=8)
