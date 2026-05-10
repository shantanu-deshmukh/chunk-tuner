# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.2] - 2026-05-11

## [0.1.0] - 2026-05-02

### Added

- Ten chunking strategies: `fixed_tokens`, `recursive_character`, `semantic`,
  `markdown_semantic`, `pdf_structural`, `structural_semantic`, `late_chunking`,
  `agentic`, `code_ast`, `code_window`
- Retrieval evaluation: token recall, IoU, precision, MRR, NDCG@k,
  duplication ratio, average chunk length
- Optional generation metrics via RAGAS: faithfulness, answer relevancy
- LLM-generated and user-provided evaluation datasets
- SQLite-backed embedding and chunk caches
- AutoTuner with optional parallel grid search (`ProcessPoolExecutor`)
- CLI: `init`, `analyze`, `estimate`, `evaluate`, `recommend`, `compare`, `preview`, `cache`
- Python MCP server (FastMCP, stdio): `list_strategies`, `preview_chunks`,
  `evaluate_chunking`, `recommend_config`
- Optional extras: `docling` (PDF/DOCX), `ragas`, `semantic`, `code`, `mcp`, `all`
