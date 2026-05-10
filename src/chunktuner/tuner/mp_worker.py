"""Picklable entry point for ``ProcessPoolExecutor`` evaluation."""

from __future__ import annotations


def mp_evaluate_task(task: dict) -> dict:
    from pathlib import Path

    from chunktuner.cache.embedding_cache import EmbeddingCache
    from chunktuner.cache.wrapped_embeddings import CachedEmbeddingFunction
    from chunktuner.chunking.bootstrap import build_full_registry
    from chunktuner.eval.embeddings import DummyEmbeddingFunction, LiteLLMEmbeddingFunction
    from chunktuner.eval.evaluator import Evaluator
    from chunktuner.eval.score_calculator import ScoreCalculator
    from chunktuner.models import ChunkConfig, Document, EvalDataset

    reg = build_full_registry(task.get("encoding", "cl100k_base"))
    strategy_name = task["strategy_name"]
    if strategy_name not in reg.names():
        available = sorted(reg.names())
        raise ValueError(
            f"Strategy {strategy_name!r} is not available in this worker process "
            f"(optional dependency may be missing). Available: {available}"
        )
    strat = reg.get(strategy_name)
    cfg = ChunkConfig.model_validate(task["config"])
    docs = [Document.model_validate(x) for x in task["docs"]]
    ds = EvalDataset.model_validate(task["dataset"])
    embed_type = task.get("embed_type", "dummy")
    if embed_type == "cached":
        cache = EmbeddingCache(Path(task["cache_db_path"]), task["cache_model"])
        if task.get("inner_embed_type") == "litellm":
            inner = LiteLLMEmbeddingFunction(task["embed_model"])
        else:
            inner = DummyEmbeddingFunction(task.get("profile", "dummy/test"))
        fn = CachedEmbeddingFunction(inner, cache)
    elif embed_type == "litellm":
        fn = LiteLLMEmbeddingFunction(task["embed_model"])
    else:
        fn = DummyEmbeddingFunction(task.get("profile", "dummy/test"))
    ev = Evaluator(
        fn,
        top_k=int(task.get("top_k", 5)),
        enable_generation_metrics=bool(task.get("enable_generation_metrics", False)),
    )
    ucw = task.get("user_custom_weights")
    scorer = ScoreCalculator(task["use_case"], custom_weights=ucw)
    res = ev.evaluate(strat, cfg, docs, ds, scorer=scorer)
    return res.model_dump()
