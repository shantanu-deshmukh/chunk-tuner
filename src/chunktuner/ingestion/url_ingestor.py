"""Fetch remote URLs into ``Document`` objects."""

from __future__ import annotations

import uuid
from urllib.parse import urlparse

import httpx

from chunktuner.ingestion.preprocessor import preprocess
from chunktuner.models import Document


class URLIngestor:
    def ingest_url(self, url: str, *, timeout: float = 30.0) -> Document:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.get(url)
            resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
        raw = resp.text
        if "html" in ctype:
            body = preprocess(raw, "html")
            content_type = "html"
        else:
            body = raw
            content_type = "markdown"
        return Document(
            id=str(uuid.uuid4()),
            content=body,
            content_type=content_type,
            source_url=url,
            metadata={"content_type_header": ctype},
        )
