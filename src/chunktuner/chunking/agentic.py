"""LLM-proposed chunk boundaries (expensive; opt-in via strategy selection)."""

from __future__ import annotations

import json
import uuid
from typing import Any

import tiktoken

from chunktuner.models import Chunk, ChunkConfig, Document


class AgenticStrategy:
    name = "agentic"
    supported_content_types = ["text", "markdown"]
    description = "Uses an LLM to propose chunk spans; requires API access and ``litellm``."

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._enc = tiktoken.get_encoding(encoding_name)

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]:
        import litellm

        model = str(config.params.get("model", "gpt-4o-mini"))
        max_props = int(config.params.get("max_propositions", 40))
        text = doc.content[:50_000]
        prompt = (
            "Split the following document into coherent RAG chunks. "
            "Return JSON object with key chunks: array of "
            '{\"start_offset\": int, \"end_offset\": int} using UTF-16? NO — use character offsets '
            "into the exact input string (Python slicing). Max chunks: "
            f"{max_props}.\n\nDOCUMENT:\n{text}"
        )
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        raw_items = data.get("chunks", data) if isinstance(data, dict) else data
        if isinstance(raw_items, dict):
            raw_items = raw_items.get("chunks", [])
        chunks: list[Chunk] = []
        for i, item in enumerate(raw_items[:max_props]):
            if not isinstance(item, dict):
                continue
            a = int(item.get("start_offset", item.get("start", 0)))
            b = int(item.get("end_offset", item.get("end", 0)))
            a = max(0, min(a, len(doc.content)))
            b = max(a, min(b, len(doc.content)))
            piece = doc.content[a:b]
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=doc.id,
                    text=piece,
                    start_offset=a,
                    end_offset=b,
                    tokens=len(self._enc.encode(piece)),
                )
            )
        if not chunks:
            from chunktuner.chunking.recursive_character import RecursiveCharacterStrategy

            return RecursiveCharacterStrategy(encoding_name=self._enc.name).chunk(
                doc,
                ChunkConfig(
                    name="recursive_character",
                    params={"chunk_size_chars": 1200, "chunk_overlap_chars": 100},
                ),
            )
        return chunks

    def param_schema(self) -> dict[str, Any]:
        return {
            "model": {"type": "string"},
            "max_propositions": {"type": "integer"},
        }

    def default_param_grid(self) -> list[dict]:
        return [{"model": "gpt-4o-mini", "max_propositions": 30}]
