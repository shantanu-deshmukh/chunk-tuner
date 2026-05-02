"""In-process MCP session tests via ``mcp.shared.memory`` (mcp >= 1.7)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent

pytest.importorskip("mcp.shared.memory", reason="mcp with in-memory transport required")

from mcp.shared.memory import create_connected_server_and_client_session

from chunktuner.mcp.server import create_mcp_app


def _tool_payload(result: CallToolResult) -> Any:
    """Decode tool output.

    FastMCP may split structured results across several ``TextContent`` blocks.
    """
    assert result.content, "tool returned empty content"
    parts: list[str] = []
    for block in result.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            raise AssertionError(f"unexpected content block type: {type(block)}")
    joined = "".join(parts).strip()
    if not joined:
        return None
    if not (joined.startswith("{") or joined.startswith("[")):
        return joined
    decoder = json.JSONDecoder()
    values: list[Any] = []
    idx = 0
    while idx < len(joined):
        while idx < len(joined) and joined[idx].isspace():
            idx += 1
        if idx >= len(joined):
            break
        obj, end = decoder.raw_decode(joined, idx)
        values.append(obj)
        idx = end
    if len(values) == 1:
        return values[0]
    return values


def test_mcp_session_lists_chunktuner_tools() -> None:
    async def body() -> None:
        async with create_connected_server_and_client_session(create_mcp_app()) as session:
            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            assert names >= {
                "list_strategies",
                "preview_chunks",
                "evaluate_chunking",
                "recommend_config",
            }

    asyncio.run(body())


def test_mcp_session_list_strategies() -> None:
    async def body() -> None:
        async with create_connected_server_and_client_session(create_mcp_app()) as session:
            raw = await session.call_tool("list_strategies", {"content_type": None})
            assert raw.isError is False
            rows = _tool_payload(raw)
            assert isinstance(rows, list)
            names = {r["name"] for r in rows}
            assert "fixed_tokens" in names
            assert "recursive_character" in names

    asyncio.run(body())


def test_mcp_session_preview_chunks() -> None:
    async def body() -> None:
        async with create_connected_server_and_client_session(create_mcp_app()) as session:
            raw = await session.call_tool(
                "preview_chunks",
                {
                    "text": "alpha beta " * 20,
                    "strategy_name": "fixed_tokens",
                    "config": {"max_tokens": 16, "overlap_tokens": 0},
                },
            )
            assert raw.isError is False
            chunks = _tool_payload(raw)
            assert isinstance(chunks, list) and chunks
            assert "start_offset" in chunks[0]

    asyncio.run(body())


def test_mcp_session_recommend_rejects_path_outside_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    f = corpus / "a.md"
    f.write_text("# Hi\n")
    monkeypatch.setenv("CHUNK_TUNER_BASE_DIR", str(corpus))

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside = outside_dir / "x.md"
    outside.write_text("nope")

    async def body() -> None:
        async with create_connected_server_and_client_session(create_mcp_app()) as session:
            ok = await session.call_tool(
                "recommend_config",
                {
                    "path": str(f),
                    "use_case": "rag_qa",
                    "strategies": ["fixed_tokens"],
                    "max_docs": 1,
                },
            )
            assert ok.isError is False

            bad = await session.call_tool(
                "recommend_config",
                {
                    "path": str(outside),
                    "use_case": "rag_qa",
                    "strategies": ["fixed_tokens"],
                    "max_docs": 1,
                },
            )
            assert bad.isError is True

    asyncio.run(body())
