"""Parallel vs sequential AutoTuner parity."""

from __future__ import annotations

from pathlib import Path

import pytest

from chunktuner.chunking.bootstrap import build_full_registry
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.tuner.auto_tuner import AutoTuner


@pytest.fixture
def fixture_docs(tmp_path: Path):
    for i in range(3):
        (tmp_path / f"doc{i}.txt").write_text("The quick brown fox. " * 100)
    return FileIngestor(root=tmp_path).ingest_dir(tmp_path)


def test_parallel_equals_sequential(fixture_docs) -> None:
    registry = build_full_registry()
    embedding_fn = DummyEmbeddingFunction()

    def make_tuner() -> AutoTuner:
        return AutoTuner(
            strategies=registry,
            evaluator=Evaluator(embedding_fn, top_k=3),
            scorer=ScoreCalculator(use_case="rag_qa"),
        )

    small_grid = {
        "fixed_tokens": [{"max_tokens": 64, "overlap_tokens": 0}],
        "recursive_character": [{"chunk_size_chars": 200, "chunk_overlap_chars": 0}],
    }

    result_seq = make_tuner().recommend(
        fixture_docs,
        use_case="rag_qa",
        strategies=["fixed_tokens", "recursive_character"],
        param_grid=small_grid,
        max_docs=3,
        parallel=False,
        baseline=False,
    )
    result_par = make_tuner().recommend(
        fixture_docs,
        use_case="rag_qa",
        strategies=["fixed_tokens", "recursive_character"],
        param_grid=small_grid,
        max_docs=3,
        parallel=True,
        max_workers=2,
        baseline=False,
    )

    assert len(result_seq.ranked) == len(result_par.ranked)
    assert result_seq.best.strategy_name == result_par.best.strategy_name
    for r_seq, r_par in zip(
        sorted(result_seq.ranked, key=lambda r: r.strategy_name),
        sorted(result_par.ranked, key=lambda r: r.strategy_name),
        strict=True,
    ):
        assert abs(r_seq.score - r_par.score) < 1e-6, (
            f"Score mismatch for {r_seq.strategy_name}: seq={r_seq.score}, par={r_par.score}"
        )
