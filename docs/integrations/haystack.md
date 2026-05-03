# Using chunktuner alongside Haystack (2.x)

chunktuner ranks chunking strategies on **your** corpus; Haystack runs preprocessing, embedding, and retrieval pipelines. Haystack is not a dependency of chunktuner — install it separately.

## Typical workflow

1. Run `chunk-tune recommend ./my_docs --use-case rag_qa --output-format yaml` and parse `best.config` (`ChunkConfig`: `name`, `params`).
2. Map character-oriented params (e.g. `chunk_size_chars` from `recursive_character`) to Haystack’s [`DocumentSplitter`](https://docs.haystack.deepset.ai/docs/documentsplitter) units (`split_by="word"` uses word counts, not characters). Calibrate `split_length` / `split_overlap` empirically or by approximating words ≈ chars / 5 for rough English prose.

## Example: Haystack `Document` from chunktuner chunks

```python
from pathlib import Path

from chunktuner import FileIngestor, default_registry
from chunktuner.models import ChunkConfig
from haystack import Document

docs = FileIngestor().ingest_dir(Path("./my_docs"))
strategy = default_registry.get("recursive_character")
cfg = ChunkConfig(name="recursive_character", params={"chunk_size_chars": 1600, "chunk_overlap_chars": 100})
chunks = strategy.chunk(docs[0], cfg)

haystack_docs = [
    Document(content=c.text, meta={"doc_id": docs[0].id, "chunk_id": c.id, "start": c.start_offset, "end": c.end_offset})
    for c in chunks
]
```

## Example: `DocumentSplitter` after a rough char → word mapping

```python
from haystack.components.preprocessors import DocumentSplitter

# Illustrative: map a character target to a word budget — validate on your data.
splitter = DocumentSplitter(split_by="word", split_length=300, split_overlap=30)
# result = splitter.run(documents=haystack_docs)
```

For recursive splitting closer to LangChain’s hierarchy, Haystack also documents [`RecursiveDocumentSplitter`](https://docs.haystack.deepset.ai/docs/recursive_documentsplitter) — chunktuner does not emit its separator list today; use chunktuner-produced chunks when you need parity with an evaluation run.
