# pi-duckdb-search

DuckDB-powered session transcript search for the [Pi coding agent](https://pi.dev).

Searches your Pi session JSONL files using three modes:
- **BM25 keyword** — exact term matching with Porter stemming
- **Semantic** — ONNX embeddings (all-MiniLM-L6-v2) with cosine similarity
- **Hybrid** — Reciprocal Rank Fusion combining both for the best results

Self-contained: no external APIs, no cloud accounts, no network after first model download (~23 MB).

## Install

```bash
pi install npm:pi-duckdb-search
```

## Tools

Five tools are registered, callable by the LLM during conversation:

| Tool | Mode | Use When |
|------|------|----------|
| `session_search` | BM25 keyword | You know exact words |
| `session_semantic` | Vector cosine | You want meaning-based matches |
| `session_hybrid` | RRF fusion | **Recommended** — best of both |
| `session_read` | Direct read | Read a specific entry by path + line |
| `session_status` | Status | Check index health |

## How It Works

```
~/.pi/agent/sessions/**/*.jsonl
        │
        ▼  read_json_objects(format='newline_delimited', ignore_errors=true)
   DuckDB database (~47 MB at 137 sessions)
   ├── session_entries table (content_text + embedding FLOAT[384])
   ├── FTS index (Porter stemmer, BM25 ranking)
   └── _metadata table (entry_count, has_embeddings, last_built)
        │
        ▼  connection.run(sql, [params]) — prepared statements
   BM25:  fts_main_session_entries.match_bm25(entry_id, ?)
   Cosine: array_cosine_similarity(embedding, ?::FLOAT[384])
   Hybrid: RRF fusion (k=60)
```

The extension opens the database in **read-only mode** — no WAL, no locks. A standalone cron build script creates and maintains the database offline.

## Setup

### 1. Install dependencies

```bash
cd ~/.pi/agent && npm install @duckdb/node-api @huggingface/transformers
```

### 2. Set up the cron build (recommended)

The cron script builds the database offline (FTS + embeddings) daily at 4am:

```bash
# Add to crontab:
0 4 * * * /path/to/node ~/.pi/agent/duckdb-search/duckdb-search-build.cjs >> ~/.pi/agent/duckdb-search/build.log 2>&1
```

Manual build (first time or after changes):

```bash
node ~/.pi/agent/duckdb-search/duckdb-search-build.cjs
```

First build takes ~4 minutes (includes 23 MB model download + ONNX inference on ~17K entries). Subsequent incremental builds take ~2 seconds if no new sessions.

### 3. Restart Pi

```bash
pi  # or /reload in an existing session
```

The extension auto-loads the pre-built database on first search. No rebuild on restart.

## Performance

| Metric | Value |
|--------|-------|
| BM25 search latency | ~19 ms |
| Semantic search latency | ~24 ms |
| Hybrid search latency | ~35 ms |
| Database size | ~47 MB (137 sessions, 17K entries) |
| Full build time | ~4 min (ONNX inference dominates) |
| Incremental build time | ~2 s (FTS rebuild only, embeddings preserved) |
| Model download | ~23 MB (one-time, cached in `~/.cache/huggingface/`) |

## Content Extraction

`content_text` extracts from JSON message content arrays:
- `text` — assistant/user text
- `thinking` — reasoning blocks (captured since v2.2; 60% of previously "empty" entries had thinking content)
- `name` — tool call names

Truncated to 2000 characters. Only error responses with `content: []` have no embedding.

## Security

- All search queries use **prepared statements** (`connection.run(sql, [params])` with `?` placeholders)
- Database opened in **read-only mode** when pre-built by cron
- **SQL injection resistant** — tested with `'; DROP TABLE --` payloads
- 34-assertion test suite covers injection, edge cases, and read-only enforcement

## Test Suite

```bash
NODE_PATH=~/.pi/agent/node_modules node ~/.pi/agent/duckdb-search/duckdb-search-test.cjs
```

34 assertions across 12 test groups:
- Schema & content extraction
- BM25 keyword search
- Semantic cosine search
- Hybrid RRF search
- session_read by path + line
- Metadata persistence
- SQL injection resistance
- Edge cases (empty query, missing file, limit clamping, line beyond EOF)
- FTS overwrite without drop
- ignore_errors with malformed JSON
- Content truncation (5000 chars → 2000 chars)
- Read-only mode enforcement

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **DuckDB over SQLite** | Native `read_json_objects()` reads JSONL directly — no ingestion pipeline |
| **Cron over lazy build** | Predictable, no race conditions, doesn't block first search |
| **all-MiniLM-L6-v2** | 23 MB, 384 dims, sufficient at 17K entries |
| **Brute-force cosine over HNSW** | 13 ms at 17K vectors; HNSW persistence is experimental in DuckDB |
| **Read-only mode** | No WAL, no locks — cron writes, extension reads |
| **Prepared statements** | SQL injection safety for all user-supplied query strings |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `SESSIONS_DIR` | `~/.pi/agent/sessions` | Pi session JSONL directory |
| `DB_PATH` | `~/.pi/agent/duckdb-search/sessions.duckdb` | DuckDB database file |
| `MODEL_ID` | `Xenova/all-MiniLM-L6-v2` | ONNX embedding model |
| `EMBED_DIM` | 384 | Embedding dimensions |
| `EXCLUDE_RECENT_MS` | 300000 (5 min) | Exclude files modified in last N ms |
| `MAX_CONTENT_CHARS` | 2000 | Content truncation limit |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant |

## Dependencies

| Package | Purpose |
|---------|---------|
| `@duckdb/node-api` | DuckDB native binary (FTS + FLOAT[] arrays) |
| `@huggingface/transformers` | ONNX embeddings (all-MiniLM-L6-v2) |

Pi-supplied (declared as peerDependencies, not bundled):
- `@earendil-works/pi-coding-agent`
- `typebox`

## Limitations

1. **FTS full rebuild on incremental builds** — DuckDB v1.5.6 FTS doesn't support incremental index updates. The cron script rebuilds FTS (~1 second) but preserves embeddings.
2. **No snippet windowing** — Snippets start from the beginning of `content_text`, not the matching term.
3. **Active session excluded** — Files modified in last 5 minutes are skipped to avoid indexing while JSONL is being written.
4. **HNSW not used** — Brute-force cosine is 13 ms at 17K vectors. HNSW would matter at 100K+ vectors but DuckDB's vss extension has experimental persistence (data loss risk).

## License

MIT
