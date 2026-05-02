"""Chunk offset invariants against document content."""

from __future__ import annotations

import os

from chunktuner.models import Chunk, Document

_SKIP = os.environ.get("CHUNKTUNER_SKIP_OFFSET_VALIDATION") == "1"


def validate_chunk_offsets(doc: Document, chunks: list[Chunk]) -> None:
    """Raise ValueError if any chunk's text doesn't match its offsets in doc.content."""
    if _SKIP or not chunks:
        return
    for c in chunks:
        got = doc.content[c.start_offset : c.end_offset]
        if got != c.text:
            raise ValueError(
                f"Offset invariant violated for chunk {c.id!r} in doc {doc.id!r}: "
                f"content[{c.start_offset}:{c.end_offset}]={got[:60]!r} "
                f"!= chunk.text={c.text[:60]!r}"
            )
