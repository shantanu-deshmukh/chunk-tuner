"""Weighted scalar score from ``EvalMetrics`` for a use case."""

from __future__ import annotations

from chunktuner.config import score_profile_weights
from chunktuner.models import EvalMetrics


class ScoreCalculator:
    """Weighted sum of `EvalMetrics` fields for a use-case profile."""

    def __init__(self, use_case: str, custom_weights: dict[str, float] | None = None):
        self.use_case = use_case
        self.weights = dict(custom_weights) if custom_weights else score_profile_weights(use_case)
        if custom_weights is not None:
            pos = sum(w for w in self.weights.values() if w > 0)
            if pos <= 0:
                raise ValueError(
                    f"custom_weights must contain at least one positive weight; got {self.weights}"
                )

    def score(self, metrics: EvalMetrics) -> float:
        """Return a scalar score from weighted metric contributions (missing keys skipped)."""
        total = 0.0
        for key, w in self.weights.items():
            val = _metric_value(metrics, key)
            if val is None:
                continue
            total += w * float(val)
        return total


def _metric_value(m: EvalMetrics, key: str) -> float | None:
    if key == "faithfulness":
        return m.faithfulness if m.faithfulness is not None else 0.0
    if key == "answer_relevancy":
        return m.answer_relevancy if m.answer_relevancy is not None else 0.0
    if key == "recall_at_1":
        return m.recall_at_k.get(1, 0.0)
    if key == "token_recall":
        return m.token_recall
    if key == "token_iou":
        return m.token_iou
    if key == "mrr":
        return m.mrr
    if key == "avg_tokens_per_query":
        return m.avg_tokens_per_query
    if key == "duplication_ratio":
        return m.duplication_ratio
    if key == "avg_chunk_length":
        return m.avg_chunk_length
    if key == "chunk_length_std":
        return m.chunk_length_std
    return getattr(m, key, None)
