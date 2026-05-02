"""Tests for ``chunktuner.mcp.service`` helpers (no MCP transport)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chunktuner.mcp.service import list_strategies_impl, preview_chunks_impl, recommend_config_impl


def test_list_strategies_nonempty() -> None:
    rows = list_strategies_impl(None)
    names = {r["name"] for r in rows}
    assert "fixed_tokens" in names
    assert "recursive_character" in names


def test_preview_chunks_fixed_tokens() -> None:
    out = preview_chunks_impl(
        "alpha beta " * 30,
        "fixed_tokens",
        {"max_tokens": 24, "overlap_tokens": 0},
    )
    assert out and "start_offset" in out[0]


def test_path_must_be_under_base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    f = corpus / "a.md"
    f.write_text("# Hi\n")
    monkeypatch.setenv("CHUNK_TUNER_BASE_DIR", str(corpus))
    recommend_config_impl(str(f), "rag_qa", strategies=["fixed_tokens"], max_docs=1)

    outside = tmp_path / "outside.md"
    outside.write_text("x")
    with pytest.raises(ValueError):
        recommend_config_impl(str(outside), "rag_qa", strategies=["fixed_tokens"], max_docs=1)
