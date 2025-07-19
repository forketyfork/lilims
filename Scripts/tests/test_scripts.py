import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_hello_script():
    output = subprocess.check_output([
        "python",
        Path(__file__).resolve().parents[1] / "hello.py",
    ])
    assert output.decode().strip() == "Hello from Python scripts"


def test_convert_main_gguf():
    import Scripts.convert as convert

    with pytest.raises(NotImplementedError):
        convert.main(["dummy", "out.mlpackage", "--gguf"])


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
