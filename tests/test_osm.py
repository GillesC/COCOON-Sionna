import json
from pathlib import Path

import requests
from shapely.geometry import Polygon

from cocoon_sionna.osm import OverpassClient


def _boundary() -> Polygon:
    return Polygon(
        [
            (3.706920, 51.058836),
            (3.710187, 51.058836),
            (3.710187, 51.060565),
            (3.706920, 51.060565),
            (3.706920, 51.058836),
        ]
    )


def _payload() -> dict:
    return {
        "elements": [
            {"type": "node", "id": 1, "lon": 3.707000, "lat": 51.059000},
            {"type": "node", "id": 2, "lon": 3.708000, "lat": 51.059000},
            {"type": "node", "id": 3, "lon": 3.708000, "lat": 51.060000},
            {"type": "node", "id": 4, "lon": 3.707000, "lat": 51.060000},
            {"type": "way", "id": 10, "nodes": [1, 2, 3, 4, 1], "tags": {"building": "yes"}},
        ]
    }


def test_overpass_client_retries_and_saves_cache(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "osm_overpass_cache.json"
    payload = _payload()
    responses = [requests.RequestException("temporary timeout"), payload]
    calls = {"count": 0}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    def _fake_post(*_args, **_kwargs):
        calls["count"] += 1
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _Response()

    monkeypatch.setattr("cocoon_sionna.osm.requests.post", _fake_post)
    monkeypatch.setattr("cocoon_sionna.osm.time.sleep", lambda _seconds: None)

    parsed = OverpassClient(
        "https://overpass-api.de/api/interpreter",
        cache_path=cache_path,
        max_attempts_per_endpoint=2,
        retry_backoff_s=0.0,
    ).fetch(_boundary())

    assert calls["count"] == 2
    assert len(parsed.nodes) == 4
    assert len(parsed.ways) == 1
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "payload" in cached
    assert cached["payload"]["elements"][0]["type"] == "node"


def test_overpass_client_falls_back_to_matching_cache(monkeypatch, tmp_path: Path):
    boundary = _boundary()
    cache_path = tmp_path / "osm_overpass_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "bbox_lonlat": list(boundary.bounds),
                "source_url": "cached",
                "payload": _payload(),
            }
        ),
        encoding="utf-8",
    )

    def _fail_post(*_args, **_kwargs):
        raise requests.RequestException("all endpoints down")

    monkeypatch.setattr("cocoon_sionna.osm.requests.post", _fail_post)
    monkeypatch.setattr("cocoon_sionna.osm.time.sleep", lambda _seconds: None)

    parsed = OverpassClient(
        "https://overpass-api.de/api/interpreter",
        cache_path=cache_path,
        max_attempts_per_endpoint=1,
        retry_backoff_s=0.0,
    ).fetch(boundary)

    assert len(parsed.nodes) == 4
    assert len(parsed.ways) == 1
