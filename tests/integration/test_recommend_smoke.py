"""End-to-end recommend with dummy embeddings."""

from __future__ import annotations

from chunktuner.chunking import default_registry
from chunktuner.eval.embeddings import DummyEmbeddingFunction
from chunktuner.eval.evaluator import Evaluator
from chunktuner.eval.score_calculator import ScoreCalculator
from chunktuner.ingestion.file_ingestor import FileIngestor
from chunktuner.tuner.auto_tuner import AutoTuner


def test_auto_tuner_recommend_minimal(tmp_path) -> None:
    f = tmp_path / "note.md"
    f.write_text("# Doc\n\n" + ("Some content. " * 80))
    fi = FileIngestor(root=tmp_path)
    docs = fi.ingest_dir(tmp_path)
    assert len(docs) == 1

    ev = Evaluator(DummyEmbeddingFunction(), top_k=3)
    tuner = AutoTuner(default_registry, ev, ScoreCalculator("rag_qa"))
    rec = tuner.recommend(
        docs,
        "rag_qa",
        strategies=["fixed_tokens"],
        param_grid={"fixed_tokens": [{"max_tokens": 128, "overlap_tokens": 0}]},
        baseline=False,
        max_docs=5,
    )
    assert rec.best.strategy_name
    assert rec.ranked
