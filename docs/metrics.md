# Evaluation metrics

Each field on `EvalMetrics` is aggregated (usually averaged) over evaluation queries. Higher-is-better unless noted.

## Retrieval metrics

### `token_recall`

- **Definition:** Share of gold answer tokens (from `answer_spans`) that appear in the union of retrieved chunk token sets (within the effective context window).
- **Range:** 0–1
- **Higher is better**
- **Example:** `0.85` means about 85% of answer tokens were covered by at least one retrieved chunk.
- **Use cases:** Strong weight in `rag_qa`, `summarization`, `code_assist`.

### `token_precision`

- **Definition:** Share of retrieved tokens (union of top chunks) that overlap gold answer tokens.
- **Range:** 0–1
- **Higher is better**
- **Example:** `0.6` means 60% of retrieved content is on-topic vs the gold span.
- **Use cases:** Informative for `rag_qa`; often paired with recall.

### `token_iou`

- **Definition:** Token-level IoU between gold and retrieved unions: `|gold ∩ retrieved| / |gold ∪ retrieved|`.
- **Range:** 0–1
- **Higher is better**
- **Example:** Penalizes both missing answer text and noisy extra text.
- **Use cases:** `rag_qa` profile weights IoU alongside recall.

### `recall_at_k` (keys 1, 3, 5)

- **Definition:** For each k, whether any of the top-k retrieved chunks hits a gold token (binary per query), then averaged.
- **Range:** 0–1 per k
- **Higher is better**
- **Example:** `recall_at_k[5] = 0.9` means 90% of queries had a hit in the top five chunks.
- **Use cases:** `search` emphasizes `recall_at_k[1]`; `rag_qa` uses multiple k.

### `mrr` (mean reciprocal rank)

- **Definition:** Average of `1 / rank` of the first relevant chunk (0 if none).
- **Range:** 0–1
- **Higher is better**
- **Example:** `0.5` implies the first hit is often around rank 2 on average.
- **Use cases:** Important for `rag_qa`, `search`, `code_assist`.

### `ndcg_at_k` (keys 1, 3, 5)

- **Definition:** Normalized discounted cumulative gain using binary relevance (chunk hits gold tokens or not).
- **Range:** 0–1 per k
- **Higher is better**
- **Example:** Rewards placing relevant chunks earlier in the ranking.
- **Use cases:** Ranking quality for all retrieval-heavy profiles.

### `duplication_ratio`

- **Definition:** Among token positions covered by retrieved chunks, the fraction of tokens that appear in more than one chunk (within the selected top set).
- **Range:** 0–1
- **Lower is better** (less redundant context)
- **Example:** `0.1` means 10% of covered tokens are duplicated across chunks.
- **Use cases:** Penalized in all profiles; high values waste context budget.

### `avg_chunk_length`

- **Definition:** Mean tiktoken length of all chunks produced for the corpus in this run.
- **Range:** roughly 0–∞ tokens
- **Interpretation:** Neither strictly higher nor lower; profile-dependent
- **Example:** `summarization` weights longer chunks slightly positively (more context per chunk).
- **Use cases:** `summarization`, `code_assist` (with `chunk_length_std`).

### `chunk_length_std`

- **Definition:** Standard deviation of per-chunk token lengths.
- **Range:** 0–∞
- **Lower is often better** for uniform pipelines (`code_assist` applies negative weight).
- **Example:** High variance means some tiny and some huge chunks.

### `avg_tokens_per_query`

- **Definition:** Average total tokens in retrieved chunks per query (after effective-k truncation).
- **Range:** 0–∞
- **Lower is better** for cost/latency (profiles use negative weights).
- **Example:** Tracks how “fat” your retrieval window is.

### `embedding_latency_ms` / `total_embedding_tokens`

- **Definition:** Wall time for batched embeddings in this evaluation; total tokens embedded (chunks + questions).
- **Range:** ms and token count ≥ 0
- **Lower latency / predictable tokens** help operational planning; not all score profiles use them directly.

## Generation metrics (optional, RAGAS)

### `faithfulness`

- **Definition:** RAGAS faithfulness of the generated answer against retrieved contexts (when `enable_generation_metrics` is on).
- **Range:** typically 0–1 when present
- **Higher is better**; **`None`** if RAGAS or LLM path unavailable
- **Use cases:** `rag_qa` default weights include faithfulness when available.

### `answer_relevancy`

- **Definition:** RAGAS answer relevancy vs the question (and reference when used).
- **Range:** typically 0–1 when present
- **Higher is better**; **`None`** if unavailable
- **Use cases:** Complements faithfulness for answer quality.

When generation metrics are disabled or RAGAS fails, these fields stay `None` and composite scores ignore them where weights apply.
