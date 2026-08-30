import asyncio

import pytest

from app.core import db


class _FakeQuery:
    def __init__(self, name, op="select"):
        self.name = name
        self.op = op
        self._records = []
        self._filters = []

    def execute(self):
        data = self._records
        for key, value in self._filters:
            data = [r for r in data if r.get(key) == value]
        return type("R", (), {"data": data})()

    def select(self, *a, **k):
        self.op = "select"
        return self

    def insert(self, records):
        self.op = "insert"
        self._records = records if isinstance(records, list) else [records]
        return self

    def upsert(self, records):
        self.op = "upsert"
        return self

    def update(self, fields):
        self.op = "update"
        return self

    def eq(self, key, value):
        self._filters.append((key, value))
        return self

    def order(self, *a, **k):
        return self

    def limit(self, n):
        return self


class _FakeClient:
    def __init__(self, records):
        self._records = records

    def table(self, name):
        q = _FakeQuery(name)
        q._records = self._records.get(name, [])
        return q


@pytest.fixture
def fake_client(monkeypatch):
    records = {
        "documents": [{"id": "d1", "title": "t", "status": "done", "content_hash": "abc"}]
    }
    monkeypatch.setattr(db, "_client", lambda: _FakeClient(records))
    return records


def test_document_get(fake_client):
    doc = asyncio.run(db.document_get("d1"))
    assert doc["title"] == "t"


def test_document_get_missing(fake_client):
    assert asyncio.run(db.document_get("nope")) is None


def test_document_by_hash(fake_client):
    assert asyncio.run(db.document_by_hash("abc"))["id"] == "d1"
    assert asyncio.run(db.document_by_hash("zzz")) is None