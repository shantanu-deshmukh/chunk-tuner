# Financial earnings transcripts — chunking benchmark

This example shows how **chunktuner** picks an optimized **(strategy, hyperparameters)** pair for financial text. It uses S&P 500 earnings call transcripts from [kurry/sp500_earnings_transcripts](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts) and the same building blocks as the CLI: `build_full_registry()`, `AutoTuner.recommend()`, and retrieval-style metrics from `Evaluator`.

Earnings calls are a useful benchmark because they combine a structured dialogue format (operator introductions, management remarks, analyst Q&A) with dense financial figures — exactly the kind of content where separator choice and chunk size make a measurable difference in retrieval quality.

You will learn how to:
- Ingest multi-speaker dialogue into `Document` objects with metadata prefixes
- Build a domain-specific `EvalDataset` tied to real financial phrases in the transcript body
- Compare chunking strategies with a `rag_qa` scoring profile tuned for financial retrieval
- Export the winning config as JSON and reload it in a production ingest pipeline

> Full documentation — including the Python API reference, strategy guide, and metrics reference — is at **https://shantanu-deshmukh.github.io/chunktuner/**.

## Setup

From inside this directory:

```bash
cd examples/financial_analysis
uv sync
```

`uv sync` reads `pyproject.toml`, creates a local virtual environment, and installs `chunktuner` as an editable path dependency from the repo root alongside `datasets`.

## Run (no Hugging Face — synthetic fixture)

Uses two enriched synthetic transcripts (~1,500 characters each) so chunk boundaries matter; good for CI and offline checks:

```bash
uv run python run_benchmark.py --fixture --num-transcripts 2
```

## Run (live Hugging Face streaming)

Streams the first *N* training rows (keeps memory bounded on the ~1.8 GB corpus):

```bash
uv run python run_benchmark.py --num-transcripts 20 --max-chars 16000
```

Options:

- `--text-mode raw` — verbatim `content` field.
- `--text-mode structured` — join `structured_content` turns only.
- `--text-mode structured_prefixed` (default) — metadata prefix + speaker turns (temporal anchor for embeddings).
- `--no-truncate` — use full text (slow for large *N*).
- `--use-case rag_qa` (default) — retrieval / QA scoring (`mrr`, `token_iou`, …). Use `summarization` only if you intentionally want length-biased weights.
- `--financial-weights` — finance-oriented `ScoreCalculator` weights (higher `token_recall` / `mrr`, stronger duplication penalty); requires `rag_qa`.
- `--parallel` / `--max-workers N` — evaluate grid jobs in parallel (`ProcessPoolExecutor`).
- `--embedding-model text-embedding-3-small` — real embeddings via LiteLLM (incurs API usage; configure provider env vars as for LiteLLM).
- `--export best-chunking.json` — write the full `Recommendation` as JSON.
- `--no-baseline` — skip the quick `fixed_tokens` baseline run (same flag as `chunk-tune recommend --no-baseline`).
- `--strategies fixed_tokens,recursive_character` — override which registered strategies to tune (comma-separated).
- `--all-text-strategies` — tune every strategy that supports `content_type=text` from `build_full_registry()` (e.g. `semantic`, `late_chunking` when those extras are installed). By default **excludes** `agentic` (LLM API calls); add `--include-agentic` to include it.

## What the script compares

**Default:** `fixed_tokens` and `recursive_character`, each on a small demo grid:

- **fixed_tokens** — three token windows (256 / 512 / 1024 max tokens with modest overlap).
- **recursive_character** — three character windows: 512 ~10% overlap, 1024 ~15% overlap with default separators, and 1024 ~15% overlap with **finance-aware** separators (`question-and-answer`, `Operator Instructions`, then paragraph / line / sentence / space).

**Speaker-aware text** is a **preprocessing** choice (`--text-mode`), not a separate strategy name. The `structured_prefixed` mode flattens `structured_content` into `Speaker: text` blocks and prepends `[SYMBOL FYyear Qn]` so the embedding model gets a temporal anchor with every chunk.

The script builds a **financial anchor** eval dataset — queries tied to phrases like "operating margin" or "free cash flow" found in the transcript body — so scores differ meaningfully across strategies. The tuner ranks results using the `rag_qa` profile by default, which weights `token_recall` and `mrr` most heavily.

## Export and reload (production handoff)

Run the tuner and save the full recommendation:

```bash
uv run python run_benchmark.py \
    --fixture \
    --use-case rag_qa \
    --export best_config.json
```

The JSON is a full `Recommendation`. Read it back in your own ingest pipeline:

```python
from pathlib import Path

from chunktuner.chunking import build_full_registry
from chunktuner.models import ChunkConfig, Recommendation

rec = Recommendation.model_validate_json(Path("best_config.json").read_text(encoding="utf-8"))
print(rec.best.strategy_name)
print(rec.best.config.params)

registry = build_full_registry()
strategy = registry.get(rec.best.strategy_name)
chunks = strategy.chunk(doc, ChunkConfig(name=rec.best.strategy_name, params=rec.best.config.params))
```

## Extending the benchmark

The default run evaluates `fixed_tokens` and `recursive_character`. These run without API keys and cover the most common production configurations.

`build_full_registry()` also registers `semantic` and `late_chunking` for `content_type=text`. To include them:

```bash
# Add semantic (no extra deps):
uv run python run_benchmark.py \
    --fixture \
    --strategies fixed_tokens,recursive_character,semantic

# Add all text-compatible strategies (excludes agentic by default):
uv run python run_benchmark.py \
    --fixture \
    --all-text-strategies

# Include agentic (calls an LLM — costs tokens):
uv run python run_benchmark.py \
    --fixture \
    --all-text-strategies \
    --include-agentic \
    --embedding-model text-embedding-3-small
```

To add a custom strategy or parameter grid, extend `financial_param_grid_for()` in `run_benchmark.py` and register the strategy before passing the registry to `AutoTuner`:

```python
from chunktuner.chunking import build_full_registry
from my_package import MyStrategy

registry = build_full_registry()
registry.register(MyStrategy())
tuner = AutoTuner(registry, evaluator, scorer)
```

| Strategy | Default run | Notes |
|----------|-------------|-------|
| `fixed_tokens` | Yes | Baseline-friendly token windows |
| `recursive_character` | Yes | Core demo; finance-aware separator variant in the grid |
| `semantic` | No | Sentence grouping by cosine similarity |
| `late_chunking` | No | Needs a per-token embedding model |
| `agentic` | No | LLM cost; use sparingly |

## Files

| File | Purpose |
|------|---------|
| `run_benchmark.py` | Load data → `Document` → domain `EvalDataset` → `AutoTuner.recommend()` → optional JSON export |
| `pyproject.toml` | uv project config; pins `chunktuner` as an editable path dep and `datasets` |

## CI

`tests/integration/test_financial_example.py` exercises `--fixture` mode so releases do not break this workflow without downloading HF data.
