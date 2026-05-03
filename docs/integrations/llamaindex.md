# Using chunktuner alongside LlamaIndex

chunktuner selects chunk sizes and strategies using retrieval metrics; LlamaIndex continues to own indexing, retrieval, and orchestration.

## Typical workflow

1. Run `chunk-tune recommend ./my_docs --use-case rag_qa --output-format yaml` and capture stdout to a file.
2. Read `best.config` from the YAML (`name` + `params` on `ChunkConfig` in `chunktuner.models`).
3. Configure a LlamaIndex node parser / splitter to approximate the same effective chunk size. **Important:** chunktuner’s `recursive_character` measures `chunk_size_chars` in **characters**; LlamaIndex [`SentenceSplitter`](https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/) defaults are often expressed in **tokens** — convert or calibrate for your tokenizer (see LlamaIndex docs for your version).

## Example: chunktuner chunks → `TextNode`

```python
from pathlib import Path

from chunktuner import FileIngestor, default_registry
from chunktuner.models import ChunkConfig
from llama_index.core import Document as LIDocument
from llama_index.core.schema import TextNode

docs = FileIngestor().ingest_dir(Path("./my_docs"))
ct_doc = docs[0]
strategy = default_registry.get("recursive_character")
cfg = ChunkConfig(name="recursive_character", params={"chunk_size_chars": 1600, "chunk_overlap_chars": 100})
chunks = strategy.chunk(ct_doc, cfg)

li_doc = LIDocument(text=ct_doc.content, doc_id=ct_doc.id, metadata=dict(ct_doc.metadata))
nodes = [
    TextNode(text=c.text, id_=c.id, metadata={**li_doc.metadata, "start_char": c.start_offset, "end_char": c.end_offset})
    for c in chunks
]
```

Install `llama-index-core` in **your** project; it is not a chunktuner dependency.

## Example: approximate mapping to `SentenceSplitter`

Only use this when your winning strategy is token-window-like and you intentionally map chars → tokens (rough rule of thumb: divide character budget by ~4 for English prose, then validate):

```python
from llama_index.core.node_parser import SentenceSplitter

# Illustrative token budget after rough char→token conversion — tune for your corpus.
splitter = SentenceSplitter(chunk_size=400, chunk_overlap=20)
# splitter.get_nodes_from_documents([...])
```

Prefer **reusing `chunk.text` from chunktuner** (previous section) when you need an exact match to evaluated chunk boundaries.
