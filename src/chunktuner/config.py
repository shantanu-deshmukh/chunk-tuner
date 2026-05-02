"""Application configuration, tokenizer profile, and score weight defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Default tiktoken encoding (GPT-4 / cl100k family)
DEFAULT_TOKENIZER_ENCODING = "cl100k_base"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_TOP_K = 5
DEFAULT_CONTEXT_BUDGET_TOKENS = 2000

# Rough throughput for wall-time estimates (tokens per minute)
THROUGHPUT_LOCAL_TOKENS_PER_MIN = 500_000
THROUGHPUT_API_TOKENS_PER_MIN = 200_000


class WorkspaceConfig(BaseModel):
    """Shape of `.autochunk.yaml` (user workspace)."""

    version: int = 1
    provider: str = "openai"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_model: str = "gpt-4o-mini"
    use_case: str = "rag_qa"
    max_docs: int = 100
    max_tokens_per_run: int = 250_000
    top_k: int = DEFAULT_TOP_K
    cache_dir: str = "~/.cache/chunktuner"
    log_level: str = "INFO"
    strip_patterns: list[str] = Field(default_factory=list)
    tokenizer_encoding: str = DEFAULT_TOKENIZER_ENCODING


def load_workspace_config(path: Path | None) -> WorkspaceConfig:
    if path is None or not path.is_file():
        return WorkspaceConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    return WorkspaceConfig.model_validate(raw)


def default_cache_dir() -> Path:
    override = os.environ.get("CHUNKTUNER_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "chunktuner"


def score_profile_weights(use_case: str) -> dict[str, float]:
    """Default metric weights (subset used per phase)."""
    profiles: dict[str, dict[str, float]] = {
        "rag_qa": {
            "token_recall": 0.35,
            "token_iou": 0.20,
            "mrr": 0.20,
            "faithfulness": 0.15,
            "avg_tokens_per_query": -0.05,
            "duplication_ratio": -0.05,
        },
        "search": {
            "recall_at_1": 0.40,
            "mrr": 0.30,
            "avg_tokens_per_query": -0.15,
            "duplication_ratio": -0.15,
        },
        "summarization": {
            "token_recall": 0.40,
            "avg_chunk_length": 0.30,
            "duplication_ratio": -0.30,
        },
        "code_assist": {
            "token_recall": 0.35,
            "mrr": 0.30,
            "chunk_length_std": -0.20,
            "duplication_ratio": -0.15,
        },
    }
    return dict(profiles.get(use_case, profiles["rag_qa"]))


def default_init_yaml() -> dict[str, Any]:
    return {
        "version": 1,
        "provider": "openai",
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "llm_model": "gpt-4o-mini",
        "use_case": "rag_qa",
        "max_docs": 100,
        "max_tokens_per_run": 250_000,
        "top_k": DEFAULT_TOP_K,
        "cache_dir": str(Path.home() / ".cache" / "chunktuner"),
        "log_level": "INFO",
        "strip_patterns": [],
    }
