# Using chunktuner alongside LangChain

chunktuner benchmarks chunking strategies; LangChain handles retrieval, chains, and agents. They compose: run chunktuner to pick `(strategy, params)`, then configure LangChain splitters to match.

## Typical workflow

1. Run `chunk-tune recommend ./my_docs --use-case rag_qa` (add `--output-format yaml` if you want machine-readable output).
2. Parse the `best` entry from the printed YAML/JSON: it matches `Recommendation.model_dump()` — see `chunktuner.models.Recommendation` and `EvalResult` (`best.config` is a `ChunkConfig`: fields `name` and `params`).
3. Map the winning `recursive_character` params to [`RecursiveCharacterTextSplitter`](https://python.langchain.com/docs/how_to/recursive_text_splitter/) (character-based `chunk_size` / `chunk_overlap`).

## Example: `recursive_character` → LangChain

`recursive_character` uses `chunk_size_chars` and `chunk_overlap_chars` in `ChunkConfig.params` (`src/chunktuner/chunking/recursive_character.py`).

```python
import yaml
from pathlib import Path

# LangChain ≥0.2: `langchain-text-splitters`. Older stacks: `langchain.text_splitter`.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Save CLI output first, e.g.:
#   chunk-tune recommend ./my_docs --use-case rag_qa --output-format yaml > recommend.yaml
data = yaml.safe_load(Path("recommend.yaml").read_text())
cfg = data["best"]["config"]
if cfg["name"] != "recursive_character":
    raise SystemExit(f"Expected recursive_character, got {cfg['name']!r}")

p = cfg["params"]
splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(p.get("chunk_size_chars", 1600)),
    chunk_overlap=int(p.get("chunk_overlap_chars", 0)),
    separators=list(p.get("separators", ["\n\n", "\n", ". ", " ", ""])),
)
```

Install LangChain’s text splitter package in **your** app (`langchain-text-splitters`); chunktuner does not declare it as a dependency.

## Using chunktuner chunks directly

```python
from pathlib import Path

from chunktuner import FileIngestor, default_registry
from chunktuner.models import ChunkConfig

docs = FileIngestor().ingest_dir(Path("./my_docs"))
strategy = default_registry.get("recursive_character")
config = ChunkConfig(
    name="recursive_character",
    params={"chunk_size_chars": 1600, "chunk_overlap_chars": 100},
)
chunks = strategy.chunk(docs[0], config)
# Feed chunk.text into your LangChain document / vector store pipeline
```
