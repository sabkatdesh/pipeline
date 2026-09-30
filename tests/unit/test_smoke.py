import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_endpoint():
    """Smoke test: ensure the FastAPI app responds on /health."""
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

