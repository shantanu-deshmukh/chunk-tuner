"""Offset invariant: doc.content[start:end] == chunk.text."""

from __future__ import annotations

import uuid

import pytest

from chunktuner.chunking.fixed_tokens import FixedTokenStrategy
from chunktuner.chunking.recursive_character import RecursiveCharacterStrategy
from chunktuner.models import ChunkConfig, Document


def _assert_offsets(doc: Document, chunks) -> None:
    for c in chunks:
        assert doc.content[c.start_offset : c.end_offset] == c.text


@pytest.mark.parametrize(
    "content",
    [
        "Short",
        "Paragraph one.\n\nParagraph two.\n\n" + "x" * 5000,
        "Unicode: café résumé naïve",
    ],
)
def test_recursive_character_offsets(content: str) -> None:
    doc = Document(id=str(uuid.uuid4()), content=content, content_type="markdown")
    strat = RecursiveCharacterStrategy()
    cfg = ChunkConfig(
        name="recursive_character",
        params={"chunk_size_chars": 80, "chunk_overlap_chars": 10},
    )
    chunks = strat.chunk(doc, cfg)
    _assert_offsets(doc, chunks)


def test_fixed_tokens_offsets() -> None:
    doc = Document(
        id=str(uuid.uuid4()),
        content="alpha beta " * 200 + "gamma",
        content_type="text",
    )
    strat = FixedTokenStrategy()
    cfg = ChunkConfig(name="fixed_tokens", params={"max_tokens": 32, "overlap_tokens": 8})
    chunks = strat.chunk(doc, cfg)
    _assert_offsets(doc, chunks)
    assert len(chunks) >= 1
