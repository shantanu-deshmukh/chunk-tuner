"""Semantic-ish chunking via ``semchunk`` (optional extra ``chunktuner[semantic]``)."""

from __future__ import annotations

from typing import Any

import tiktoken

from chunktuner.models import Chunk, ChunkConfig, Document


def _require_semchunk():
    try:
        import semchunk  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "Install the semantic extra: uv add 'chunktuner[semantic]' or pip install semchunk"
        ) from e
    return __import__("semchunk")


class SemanticStrategy:
    name = "semantic"
    supported_content_types = ["text", "markdown", "html"]
    description = "Hierarchical semantic-friendly splits via semchunk + tiktoken sizing."

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoding_name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        semchunk = _require_semchunk()
        max_tokens = max(16, int(config.params.get("max_tokens", 512)))
        ot = config.params.get("overlap_tokens")
        if ot is not None:
            overlap: int | None = int(ot)
        else:
            thr = float(config.params.get("similarity_threshold", 0.0) or 0.0)
            overlap = int(max_tokens * thr) if thr > 0 else None
        text = doc.content
        if not text:
            return []
        out = semchunk.chunk(
            text,
            max_tokens,
            self._count,
            offsets=True,
            overlap=overlap,
        )
        pieces, offsets = out
        chunks: list[Chunk] = []
        for idx, (piece, (a, b)) in enumerate(zip(pieces, offsets, strict=True)):
            slice_text = text[a:b]
            if slice_text != piece:
                slice_text = piece
            chunks.append(
                Chunk(
                    id=f"{doc.id}_sem_{idx}",
                    document_id=doc.id,
                    text=slice_text,
                    start_offset=a,
                    end_offset=b,
                    tokens=self._count(slice_text),
                )
            )
        return chunks

    def param_schema(self) -> dict[str, Any]:
        return {
            "max_tokens": {"type": "integer", "minimum": 16},
            "min_tokens": {"type": "integer", "minimum": 0},
            "similarity_threshold": {
                "type": "number",
                "description": "Repurposed as overlap fraction when semchunk overlap used",
            },
            "overlap_tokens": {"type": "integer", "minimum": 0},
        }

    def default_param_grid(self) -> list[dict]:
        grid: list[dict] = []
        for mt in (512, 1024):
            for thr in (0.0, 0.1, 0.2):
                grid.append({"max_tokens": mt, "similarity_threshold": thr})
        return grid
