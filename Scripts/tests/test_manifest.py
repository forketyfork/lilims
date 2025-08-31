from pathlib import Path
import json
import hashlib
import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import Scripts.manifest as manifest


def test_generate_manifest(tmp_path: Path) -> None:
    file_path = tmp_path / "model.mlpackage"
    file_path.write_bytes(b"test")
    data = manifest.generate_manifest(file_path, "1.0.0")
    assert data["name"] == "model"
    assert data["size"] == 4
    assert data["sha256"] == hashlib.sha256(b"test").hexdigest()
    assert data["runtime_version"] == "1.0.0"
    assert "created_at" in data


def test_generate_manifest_with_optional_fields(tmp_path: Path) -> None:
    file_path = tmp_path / "model.mlpackage"
    file_path.write_bytes(b"test")
    data = manifest.generate_manifest(
        file_path, 
        "1.0.0",
        name="Custom Model",
        model_version="2.1.0",
        description="A test model"
    )
    assert data["name"] == "Custom Model"
    assert data["model_version"] == "2.1.0" 
    assert data["description"] == "A test model"


def test_semantic_version_validation() -> None:
    # Valid versions
    assert manifest.validate_semantic_version("1.0.0")
    assert manifest.validate_semantic_version("1.2.3")
    assert manifest.validate_semantic_version("1.0.0-alpha")
    assert manifest.validate_semantic_version("1.0.0-alpha.1")
    assert manifest.validate_semantic_version("1.0.0+build.1")
    
    # Invalid versions
    assert not manifest.validate_semantic_version("1.0")
    assert not manifest.validate_semantic_version("1")
    assert not manifest.validate_semantic_version("1.0.0.0")
    assert not manifest.validate_semantic_version("v1.0.0")


def test_generate_manifest_invalid_versions(tmp_path: Path) -> None:
    file_path = tmp_path / "model.mlpackage"
    file_path.write_bytes(b"test")
    
    # Invalid runtime version
    with pytest.raises(ValueError, match="Invalid runtime version"):
        manifest.generate_manifest(file_path, "1.0")
    
    # Invalid model version
    with pytest.raises(ValueError, match="Invalid model version"):
        manifest.generate_manifest(file_path, "1.0.0", model_version="2.0")


def test_manifest_main(tmp_path: Path) -> None:
    model = tmp_path / "a.mlpackage"
    model.write_text("hi")
    out = tmp_path / "manifest.json"
    manifest.main([
        str(model),
        str(out),
        "--runtime-version",
        "2.0.0",
        "--name",
        "A",
    ])
    result = json.loads(out.read_text())
    assert result["name"] == "A"
    assert result["runtime_version"] == "2.0.0"


def test_manifest_main_with_all_options(tmp_path: Path) -> None:
    model = tmp_path / "model.mlpackage"
    model.write_text("test content")
    out = tmp_path / "manifest.json"
    manifest.main([
        str(model),
        str(out),
        "--runtime-version", "1.2.0",
        "--name", "Test Model",
        "--model-version", "3.1.4", 
        "--description", "A comprehensive test model"
    ])
    result = json.loads(out.read_text())
    assert result["name"] == "Test Model"
    assert result["runtime_version"] == "1.2.0"
    assert result["model_version"] == "3.1.4"
    assert result["description"] == "A comprehensive test model"
    assert result["size"] == len("test content")
    assert "created_at" in result
