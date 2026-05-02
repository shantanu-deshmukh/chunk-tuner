"""Domain models — single source of truth for library, CLI, and API."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator

ContentType = Literal["text", "markdown", "html", "pdf", "docx", "pptx", "code"]
UseCase = Literal["rag_qa", "search", "summarization", "code_assist"]
DatasetSource = Literal["llm_generated", "user_provided"]


class Document(BaseModel):
    """Ingested unit of text (file, URL, or synthetic) passed to chunking and evaluation."""

    id: str
    content: str
    content_type: ContentType
    path: str | None = None
    source_url: str | None = None
    metadata: dict = Field(default_factory=dict)
    group_id: str | None = None
    page_number: int | None = None
    language: str | None = None


class Chunk(BaseModel):
    """Text span within a `Document`; offsets must satisfy ``doc.content[start:end] == text``."""

    id: str
    document_id: str
    text: str
    start_offset: int
    end_offset: int
    tokens: int | None = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_offsets(self) -> Chunk:
        if self.start_offset < 0:
            raise ValueError(f"start_offset must be >= 0, got {self.start_offset}")
        if self.end_offset <= self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be > start_offset ({self.start_offset})"
            )
        return self


class ChunkConfig(BaseModel):
    """Named strategy plus strategy-specific hyperparameters (``params``)."""

    name: str
    params: dict = Field(default_factory=dict)


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Pluggable chunker: exposes metadata, ``chunk()``, parameter schema, and search grid."""

    name: str
    supported_content_types: list[str]
    description: str

    def chunk(self, doc: Document, config: ChunkConfig) -> list[Chunk]: ...

    def param_schema(self) -> dict: ...

    def default_param_grid(self) -> list[dict]: ...


class EvalQuery(BaseModel):
    id: str
    question: str
    document_id: str
    answer_spans: list[tuple[int, int]]
    reference_answer: str | None = None


class EvalDataset(BaseModel):
    name: str
    queries: list[EvalQuery]
    source: DatasetSource = "llm_generated"


class EvalMetrics(BaseModel):
    """Retrieval and optional generation metrics aggregated for one strategy run."""

    token_iou: float = 0.0
    token_precision: float = 0.0
    token_recall: float = 0.0
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    avg_tokens_per_query: float = 0.0
    duplication_ratio: float = 0.0
    avg_chunk_length: float = 0.0
    chunk_length_std: float = 0.0
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    embedding_latency_ms: float = 0.0
    total_embedding_tokens: int = 0


class EvalResult(BaseModel):
    """Outcome of evaluating one ``(strategy, ChunkConfig)`` on a corpus and dataset."""

    strategy_name: str
    config: ChunkConfig
    embedding_profile: str
    metrics: EvalMetrics
    score: float


class Recommendation(BaseModel):
    """Ranked evaluation results from tuning, including the best config and optional baseline."""

    content_type: str
    use_case: UseCase
    embedding_profile: str
    best: EvalResult
    ranked: list[EvalResult]
    baseline: EvalResult | None = None


class CostEstimate(BaseModel):
    total_tokens: int
    embedding_cost_usd: float
    llm_cost_usd: float
    estimated_wall_time_min: float
    strategy_configs: int


class EmbeddingFunction(Protocol):
    """Embeds chunk texts and queries; ``profile_name`` labels the model or dummy profile."""

    profile_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
