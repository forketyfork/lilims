from __future__ import annotations
import urllib.request

import Scripts.tinystories as ts


def test_ensure_dataset_download(monkeypatch, tmp_path):
    data = b"tiny"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def read(self):
            return data

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: FakeResponse())
    path = ts.ensure_dataset(cache_dir=tmp_path)
    assert path.exists()
    assert path.read_bytes() == data

    called: dict[str, object] = {}

    def fake_urlopen(url):
        called["called"] = True
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert ts.ensure_dataset(cache_dir=tmp_path) == path
    assert "called" not in called
