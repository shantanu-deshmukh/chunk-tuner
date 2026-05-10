"""Integration check for examples/financial_analysis/run_benchmark.py (fixture mode, no HF)."""

import importlib.util
from pathlib import Path


def _load_financial_benchmark_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "examples" / "financial_analysis" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("financial_analysis_run_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_financial_example_uses_domain_dataset() -> None:
    """Verify the financial eval dataset builds non-trivial queries from enriched fixtures."""
    mod = _load_financial_benchmark_module()
    rows = mod.synthetic_fixture_records()
    docs = [
        mod.transcript_record_to_document(
            r,
            text_mode="structured_prefixed",
            max_chars=None,
            index=i,
        )
        for i, r in enumerate(rows)
    ]
    dataset = mod.build_financial_eval_dataset(docs)
    assert len(dataset.queries) > 0, "Expected at least one domain-specific query"
    for q in dataset.queries:
        start, end = q.answer_spans[0]
        assert start > 0 or end > 200, (
            f"Query {q.id!r} looks like a trivial fallback query (span=({start}, {end}))"
        )


def test_financial_example_fixture_rag_qa() -> None:
    """End-to-end: fixture + rag_qa use case produces a valid ranked recommendation."""
    mod = _load_financial_benchmark_module()
    rec = mod.run_benchmark_cli(
        [
            "--fixture",
            "--num-transcripts",
            "2",
            "--no-truncate",
            "--use-case",
            "rag_qa",
            "-q",
            "--no-baseline",
        ]
    )
    names_in_ranked = {r.strategy_name for r in rec.ranked}
    assert "fixed_tokens" in names_in_ranked
    assert "recursive_character" in names_in_ranked
    assert rec.best.config.name == rec.best.strategy_name
    assert rec.ranked
    assert len(rec.ranked) >= 6
    if len(rec.ranked) >= 2:
        assert rec.ranked[0].score >= rec.ranked[1].score


def test_transcript_record_to_document_offsets() -> None:
    """Invariant: slice doc.content by chunk offsets equals chunk.text (via chunker)."""
    from chunktuner.chunking import RecursiveCharacterStrategy
    from chunktuner.chunking.registry import StrategyRegistry
    from chunktuner.models import ChunkConfig

    mod = _load_financial_benchmark_module()
    row = mod.synthetic_fixture_records()[0]
    doc = mod.transcript_record_to_document(
        row, text_mode="structured_prefixed", max_chars=None, index=0
    )
    reg = StrategyRegistry()
    reg.register(RecursiveCharacterStrategy(encoding_name="cl100k_base"))
    strat = reg.get("recursive_character")
    cfg = ChunkConfig(
        name="recursive_character",
        params={"chunk_size_chars": 200, "chunk_overlap_chars": 0},
    )
    chunks = strat.chunk(doc, cfg)
    for c in chunks:
        assert doc.content[c.start_offset : c.end_offset] == c.text
