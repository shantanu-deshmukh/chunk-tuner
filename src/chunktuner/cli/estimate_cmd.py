"""``chunk-tune estimate`` — dry-run cost estimate."""

from __future__ import annotations

from pathlib import Path

import typer

from chunktuner.chunking import default_registry
from chunktuner.cli.common import load_workspace_path, validate_output_format
from chunktuner.config import DEFAULT_EMBEDDING_MODEL, load_workspace_config
from chunktuner.eval.cost_estimator import CostEstimator
from chunktuner.ingestion.file_ingestor import FileIngestor


def register(app: typer.Typer) -> None:
    @app.command("estimate")
    def estimate_cmd(
        path: Path = typer.Argument(..., exists=True),
        strategies: str = typer.Option(
            "fixed_tokens,recursive_character",
            "--strategies",
            help="Comma-separated strategy names",
        ),
        use_case: str = typer.Option("rag_qa", "--use-case"),
        max_docs: int = typer.Option(100, "--max-docs"),
        config: Path | None = typer.Option(None, "--config", help="Path to .autochunk.yaml"),
        output_format: str = typer.Option("table", "--output-format"),
    ) -> None:
        """Estimate tokens, cost, and wall time (no API calls)."""
        validate_output_format(output_format)
        ws = load_workspace_config(load_workspace_path(config))
        root = path.resolve().parent if path.is_file() else path.resolve()
        fi = FileIngestor(root=root)
        docs = fi.ingest_path(path) if path.is_file() else fi.ingest_dir(path)
        docs = docs[:max_docs]
        names = [s.strip() for s in strategies.split(",") if s.strip()]
        param_grid: dict[str, list[dict]] = {}
        for n in names:
            param_grid[n] = default_registry.get(n).default_param_grid()
        est = CostEstimator().estimate(
            docs,
            names,
            param_grid,
            ws.embedding_model or DEFAULT_EMBEDDING_MODEL,
            generate_dataset=True,
        )
        if output_format == "json":
            typer.echo(est.model_dump_json(indent=2))
        elif output_format == "yaml":
            typer.echo(__import__("yaml").safe_dump(est.model_dump(), sort_keys=False))
        else:
            typer.echo(f"Total embedding tokens (est.): {est.total_tokens}")
            typer.echo(f"Embedding cost (USD est.): {est.embedding_cost_usd}")
            typer.echo(f"LLM dataset cost (USD est.): {est.llm_cost_usd}")
            typer.echo(f"Wall time (min est.): {est.estimated_wall_time_min}")
            typer.echo(f"Strategy-config combinations: {est.strategy_configs}")
