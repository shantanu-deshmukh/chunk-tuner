"""FastAPI route smoke tests (no external API calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from chunktuner.api.routes import router


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, Path, Path]:
    monkeypatch.setenv("CHUNK_TUNER_BASE_DIR", str(tmp_path))
    corpus = tmp_path / "docs"
    corpus.mkdir()
    (corpus / "a.md").write_text("# Hello\n\nThis is a test document." * 20)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), tmp_path, corpus


def test_health(client: tuple[TestClient, Path, Path]) -> None:
    c, *_ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_strategies(client: tuple[TestClient, Path, Path]) -> None:
    c, *_ = client
    r = c.get("/list_strategies")
    assert r.status_code == 200
    names = {s["name"] for s in r.json()}
    assert "fixed_tokens" in names


def test_preview_chunks_valid(client: tuple[TestClient, Path, Path]) -> None:
    c, *_ = client
    r = c.post(
        "/preview_chunks",
        json={
            "text": "alpha beta " * 40,
            "strategy_name": "fixed_tokens",
            "config": {"max_tokens": 20},
        },
    )
    assert r.status_code == 200
    assert len(r.json()) > 0


def test_preview_chunks_invalid_strategy(client: tuple[TestClient, Path, Path]) -> None:
    c, *_ = client
    r = c.post(
        "/preview_chunks",
        json={"text": "x", "strategy_name": "nonexistent", "config": {}},
    )
    assert r.status_code == 400


def test_evaluate_chunking_dry_run(client: tuple[TestClient, Path, Path]) -> None:
    c, _, corpus = client
    r = c.post(
        "/evaluate_chunking",
        json={
            "path": str(corpus),
            "use_case": "rag_qa",
            "dry_run": True,
            "strategies": ["fixed_tokens"],
        },
    )
    assert r.status_code == 200
    assert "total_tokens" in r.json()


def test_evaluate_chunking_path_outside_base(client: tuple[TestClient, Path, Path]) -> None:
    c, *_ = client
    r = c.post("/evaluate_chunking", json={"path": "/etc/passwd", "use_case": "rag_qa"})
    assert r.status_code == 400


def test_recommend_config(client: tuple[TestClient, Path, Path]) -> None:
    c, _, corpus = client
    r = c.post(
        "/recommend_config",
        json={
            "path": str(corpus),
            "use_case": "rag_qa",
            "strategies": ["fixed_tokens"],
            "max_docs": 2,
        },
    )
    assert r.status_code == 200
    assert "best" in r.json()
