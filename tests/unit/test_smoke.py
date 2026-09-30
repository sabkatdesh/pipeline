from fastapi.testclient import TestClient

from app.main import create_app


def test_health_endpoint():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_reset_rejects_bad_token():
    client = TestClient(create_app())
    r = client.post("/api/v1/ingest/reset", json={"confirmation_token": "nope"})
    assert r.status_code == 400
