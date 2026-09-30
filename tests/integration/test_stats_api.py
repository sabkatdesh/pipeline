"""Stats endpoint wiring with the DB dependency overridden (no Postgres needed)."""

from fastapi.testclient import TestClient

from app.core.database import get_db
from app.main import create_app


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    async def execute(self, *_a, **_k):
        return _Result([type("R", (), {"author": "Ada Lovelace", "paper_count": 3})()])


def test_top_authors_returns_aggregated_rows():
    app = create_app()

    async def fake_db():
        yield _FakeSession()

    app.dependency_overrides[get_db] = fake_db
    r = TestClient(app).get("/api/v1/stats/top-authors?limit=1")
    assert r.status_code == 200
    assert r.json()["data"] == [{"author": "Ada Lovelace", "paper_count": 3}]
