"""Domain models — single source of truth for library, CLI, and API."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

ContentType = Literal["text", "markdown", "html", "pdf", "docx", "pptx", "code"]
UseCase = Literal["rag_qa", "search", "summarization", "code_assist"]
DatasetSource = Literal["llm_generated", "user_provided"]


class Document(BaseModel):
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
    id: str
    document_id: str
    text: str
    start_offset: int
    end_offset: int
    tokens: int | None = None
    metadata: dict = Field(default_factory=dict)


class ChunkConfig(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


@runtime_checkable
class ChunkingStrategy(Protocol):
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
    strategy_name: str
    config: ChunkConfig
    embedding_profile: str
    metrics: EvalMetrics
    score: float


class Recommendation(BaseModel):
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
    profile_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
