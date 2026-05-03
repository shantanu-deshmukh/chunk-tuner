# Contributing to chunktuner

## Dev setup

```bash
git clone https://github.com/shantanu-deshmukh/chunktuner.git
cd chunktuner
uv sync --all-extras --dev
uv run pytest
```

## Good first issues

Issues labeled **`good first issue`** are intended to be self-contained starter tasks. If you pick one up, comment on the issue so work is not duplicated.

## Adding a new chunking strategy

1. Implement the `ChunkingStrategy` protocol in `src/chunktuner/chunking/<name>.py`.
2. Register the strategy in `src/chunktuner/chunking/bootstrap.py` (and `__init__.py` if exported).
3. Add an offset invariant test (see `tests/unit/test_chunking_offsets.py`).
4. Benchmark the strategy against the built-in `fixed_tokens` baseline (`max_tokens=512`, `overlap_tokens=0`) in your PR description or tests so reviewers can see the improvement story.
5. Update `docs/strategy_guide.md`.

## Running tests

```bash
uv run pytest
uv run pytest tests/unit/
uv run pytest -k test_offset
```

## Code style

```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Updating documentation

- Source lives under `docs/`; navigation is `docs/SUMMARY.md` (literate-nav).
- Preview locally: `uv sync --dev --extra docs` then `uv run mkdocs serve`.
- API reference pages are generated — see `scripts/gen_api_pages.py` and `mkdocs.yml` plugins.

## Pull request checklist

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check src/ tests/` is clean
- [ ] New or changed strategy: offset invariant test added
- [ ] New public API: type hints present
- [ ] `CHANGELOG.md` updated under `[Unreleased]` when behaviour is user-visible
- [ ] `docs/` (including `docs/SUMMARY.md` when adding pages) updated if behaviour, configuration, or user-facing docs changed

See `.github/pull_request_template.md` for the same checklist in the PR UI.
