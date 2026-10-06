import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient

    from neural_forge.app import create_app
    return TestClient(create_app(str(tmp_path / "test.sqlite3")))


@pytest.fixture()
def player(client):
    """A fresh Beginner-mode player; returns (client, pid)."""
    r = client.post("/api/players", json={"name": "Test", "mode": 1})
    assert r.status_code == 200
    return client, r.json()["player"]["id"]
