"""Picklable entry point for ``ProcessPoolExecutor`` evaluation."""

from __future__ import annotations


def mp_evaluate_task(task: dict) -> dict:
    from chunktuner.chunking.bootstrap import build_full_registry
    from chunktuner.eval.embeddings import DummyEmbeddingFunction, LiteLLMEmbeddingFunction
    from chunktuner.eval.evaluator import Evaluator
    from chunktuner.eval.score_calculator import ScoreCalculator
    from chunktuner.models import ChunkConfig, Document, EvalDataset

    reg = build_full_registry(task.get("encoding", "cl100k_base"))
    strat = reg.get(task["strategy_name"])
    cfg = ChunkConfig.model_validate(task["config"])
    docs = [Document.model_validate(x) for x in task["docs"]]
    ds = EvalDataset.model_validate(task["dataset"])
    if task.get("embed_type") == "litellm":
        fn = LiteLLMEmbeddingFunction(task["embed_model"])
    else:
        fn = DummyEmbeddingFunction(task.get("profile", "dummy/test"))
    ev = Evaluator(
        fn,
        top_k=int(task.get("top_k", 5)),
        enable_generation_metrics=bool(task.get("enable_generation_metrics", False)),
    )
    res = ev.evaluate(strat, cfg, docs, ds, scorer=ScoreCalculator(task["use_case"]))
    return res.model_dump()
