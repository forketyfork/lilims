from pathlib import Path
import json
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import Scripts.manifest as manifest


def test_generate_manifest(tmp_path: Path) -> None:
    file_path = tmp_path / "model.mlpackage"
    file_path.write_bytes(b"test")
    data = manifest.generate_manifest(file_path, "1.0")
    assert data["name"] == "model"
    assert data["size"] == 4
    assert data["sha256"] == hashlib.sha256(b"test").hexdigest()
    assert data["runtime_version"] == "1.0"


def test_manifest_main(tmp_path: Path) -> None:
    model = tmp_path / "a.mlpackage"
    model.write_text("hi")
    out = tmp_path / "manifest.json"
    manifest.main([
        str(model),
        str(out),
        "--runtime-version",
        "2.0",
        "--name",
        "A",
    ])
    result = json.loads(out.read_text())
    assert result["name"] == "A"
    assert result["runtime_version"] == "2.0"
