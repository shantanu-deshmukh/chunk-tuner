"""FastMCP tool registrations (stdio-safe logging to stderr only)."""

from __future__ import annotations

import logging
import time
from threading import BoundedSemaphore

from chunktuner.mcp.service import (
    evaluate_chunking_impl,
    list_strategies_impl,
    preview_chunks_impl,
    recommend_config_impl,
)

log = logging.getLogger("chunktuner.mcp.tools")
_eval_sem = BoundedSemaphore(5)


def _log_tool(name: str, start: float, ok: bool) -> None:
    ms = int((time.perf_counter() - start) * 1000)
    log.info("mcp_tool name=%s ok=%s latency_ms=%s", name, ok, ms)


def register_tools(mcp: object) -> None:
    from mcp.server.fastmcp import FastMCP

    if not isinstance(mcp, FastMCP):
        raise TypeError("Expected FastMCP instance")

    @mcp.tool()
    def list_strategies(content_type: str | None = None) -> list[dict]:
        """List registered chunking strategies, optionally filtered by content type."""
        t0 = time.perf_counter()
        try:
            out = list_strategies_impl(content_type)
            _log_tool("list_strategies", t0, True)
            return out
        except Exception:
            _log_tool("list_strategies", t0, False)
            raise

    @mcp.tool()
    def preview_chunks(
        text: str,
        strategy_name: str,
        config: dict | None = None,
    ) -> list[dict]:
        """Chunk inline text with one strategy + params (no embeddings)."""
        t0 = time.perf_counter()
        try:
            out = preview_chunks_impl(text, strategy_name, config or {})
            _log_tool("preview_chunks", t0, True)
            return out
        except Exception:
            _log_tool("preview_chunks", t0, False)
            raise

    @mcp.tool()
    def evaluate_chunking(
        path: str,
        use_case: str = "rag_qa",
        content_type: str | None = None,
        strategies: list[str] | None = None,
        max_docs: int = 20,
        top_k: int = 5,
        dry_run: bool = False,
        embedding_model: str | None = None,
    ) -> dict:
        """Dry-run cost estimate or full evaluation (``DummyEmbeddingFunction`` if no model)."""
        t0 = time.perf_counter()
        with _eval_sem:
            try:
                out = evaluate_chunking_impl(
                    path,
                    use_case,
                    content_type=content_type,
                    strategies=strategies,
                    max_docs=max_docs,
                    top_k=top_k,
                    dry_run=dry_run,
                    embedding_model=embedding_model,
                )
                _log_tool("evaluate_chunking", t0, True)
                return out
            except Exception:
                _log_tool("evaluate_chunking", t0, False)
                raise

    @mcp.tool()
    def recommend_config(
        path: str,
        use_case: str = "rag_qa",
        content_type: str | None = None,
        strategies: list[str] | None = None,
        max_docs: int = 20,
        top_k: int = 5,
        embedding_model: str | None = None,
    ) -> dict:
        """Run tuner and return ranked ``Recommendation`` (uses dummy embeddings by default)."""
        t0 = time.perf_counter()
        with _eval_sem:
            try:
                out = recommend_config_impl(
                    path,
                    use_case,
                    content_type=content_type,
                    strategies=strategies,
                    max_docs=max_docs,
                    top_k=top_k,
                    embedding_model=embedding_model,
                )
                _log_tool("recommend_config", t0, True)
                return out
            except Exception:
                _log_tool("recommend_config", t0, False)
                raise
