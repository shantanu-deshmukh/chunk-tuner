# chunktuner MCP setup (Python / FastMCP)

The MCP server is **pure Python**: it imports the `chunktuner` library directly (no Node.js, no separate HTTP hop for MCP).

## Install

```bash
uv sync --extra mcp
# or
pip install 'chunktuner[mcp]'
```

## Claude Desktop (`.mcp.json`)

```json
{
  "mcpServers": {
    "chunktuner": {
      "command": "uvx",
      "args": ["--from", "chunktuner[mcp]", "chunk-tune-mcp"],
      "env": {
        "CHUNK_TUNER_BASE_DIR": "/absolute/path/to/your/corpus"
      }
    }
  }
}
```

- **`CHUNK_TUNER_BASE_DIR`**: every `path` argument to tools must resolve under this directory (security boundary).
- **`CHUNKTUNER_CACHE_DIR`**: optional override for the SQLite cache directory (default: `~/.cache/chunktuner`).
- **Entry point**: `chunk-tune-mcp` → `chunktuner.mcp.server:run` (stdio JSON-RPC on stdout; **never** `print()` in MCP code).

## Local dev (editable)

```bash
cd /path/to/chunktuner
uv run --extra mcp chunk-tune-mcp
```

## Optional HTTP API

The FastAPI app under `chunktuner.api` is separate from MCP. Start it with:

```bash
uv run uvicorn chunktuner.api.main:app --host 127.0.0.1 --port 8765
```

Use this when you want REST clients; MCP hosts talk to `chunk-tune-mcp` only.

## Tools

| Tool | Purpose |
|------|---------|
| `list_strategies` | Strategies + param schemas |
| `preview_chunks` | Chunk inline text (no embeddings) |
| `evaluate_chunking` | Cost dry-run or full eval (dummy embeddings unless you pass a model + keys) |
| `recommend_config` | Full tuner + ranking |

## Resources & prompts

- Resources: `doc://overview`, `doc://strategy_guidelines`, `doc://metrics_guide`
- Prompts: `explain_chunking_results`, `design_eval_questions`
