# Configuration

## `.autochunk.yaml`

Created in the **current directory** by `chunk-tune init`. Merged at runtime with defaults when keys are missing.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | int | `1` | Schema version |
| `provider` | string | `openai` | Provider label (stored for tooling) |
| `embedding_model` | string | `text-embedding-3-small` | Default embedding model id |
| `llm_model` | string | `gpt-4o-mini` | Default LLM for dataset / answer steps |
| `use_case` | string | `rag_qa` | Default scoring profile |
| `max_docs` | int | `100` | Default cap on documents per run |
| `max_tokens_per_run` | int | `250_000` | Budget guard for runs that honor workspace config |
| `top_k` | int | `5` | Default retrieval depth |
| `cache_dir` | string | `~/.cache/chunktuner` | SQLite and cache root |
| `log_level` | string | `INFO` | Intended logging level for apps reading this file |
| `strip_patterns` | list of string | `[]` | Optional regex patterns stripped at ingest |
| `tokenizer_encoding` | string | `cl100k_base` | tiktoken encoding name (workspace model) |

CLI commands accept `--config /path/to/.autochunk.yaml` to point elsewhere.

---

## Environment variables

| Variable | Effect |
|----------|--------|
| `CHUNKTUNER_CACHE_DIR` | Overrides default cache directory (`~/.cache/chunktuner`) when resolving paths; also used by `default_cache_dir()` |
| `CHUNK_TUNER_BASE_DIR` | **MCP / API security**: corpus root; every tool path must resolve under this directory |
| `CHUNKTUNER_SKIP_OFFSET_VALIDATION` | Set to `1` to skip chunk offset validation (debugging only) |
| `OPENAI_API_KEY` / provider keys | Passed through **LiteLLM** for real embeddings and LLM calls |
| `LITELLM_LOG` | LiteLLM logging verbosity |

Logging for the library itself uses the standard `logging` module; configure handlers in your app. The `log_level` field in YAML is for conventions and future tooling.

---

## Workspace init flow

1. Run `chunk-tune init` in the directory where you want `.autochunk.yaml`.
2. Edit `embedding_model`, `cache_dir`, or `use_case` as needed.
3. Run `chunk-tune estimate` before paid `evaluate` / `recommend` with `--embedding-model`.

---

## Provider examples (LiteLLM)

The CLI wires **`LiteLLMEmbeddingFunction`** when `--embedding-model` is set. Examples:

```bash
export OPENAI_API_KEY=sk-...
chunk-tune recommend ./docs --embedding-model text-embedding-3-small --yes
```

For **Ollama** (local), use a LiteLLM-recognized model string such as `ollama/nomic-embed-text` if your LiteLLM build supports it, and ensure the Ollama server is reachable.

For **Cohere**, set `COHERE_API_KEY` and pass the appropriate `cohere/...` model id per LiteLLM docs.

Exact model ids change with providers; verify against [LiteLLM model list](https://docs.litellm.ai/docs/providers).

---

## Related docs

- [MCP setup](mcp_setup.md) — `CHUNK_TUNER_BASE_DIR` in `mcp.json`
- [CLI reference](cli_reference.md)
