"""``chunk-tune init`` — bootstrap workspace config."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from chunktuner.config import default_init_yaml


def register(app: typer.Typer) -> None:
    @app.command("init")
    def init_cmd(
        provider: str = typer.Option("openai", help="Embedding provider label"),
        model: str = typer.Option(
            "text-embedding-3-small",
            "--model",
            "-m",
            help="Default embedding model id",
        ),
    ) -> None:
        """Create ``.autochunk.yaml`` in the current directory."""
        path = Path.cwd() / ".autochunk.yaml"
        if path.exists():
            typer.echo(
                f"Refusing to overwrite existing {path}. "
                "Delete the file manually and re-run, or edit it directly.",
                err=True,
            )
            raise typer.Exit(code=1)
        data = default_init_yaml()
        data["provider"] = provider
        data["embedding_model"] = model
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        typer.echo(f"Wrote {path}")
