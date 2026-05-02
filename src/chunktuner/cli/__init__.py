"""Typer CLI entrypoint: ``chunk-tune``."""

from __future__ import annotations

import typer

from chunktuner.cli import (
    analyze_cmd,
    cache_cmd,
    compare_cmd,
    estimate_cmd,
    evaluate_cmd,
    init_cmd,
    preview_cmd,
    recommend_cmd,
)

app = typer.Typer(
    name="chunk-tune",
    no_args_is_help=True,
    help="Auto chunking tuner for RAG pipelines",
)

init_cmd.register(app)
analyze_cmd.register(app)
estimate_cmd.register(app)
evaluate_cmd.register(app)
recommend_cmd.register(app)
preview_cmd.register(app)
compare_cmd.register(app)
cache_cmd.register(app)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
