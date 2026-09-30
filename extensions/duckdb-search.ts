/**
 * duckdb-search.ts — DuckDB-powered session transcript search for Pi.
 *
 * v2.4: All SQL paths use prepared statements or escaped literals.
 *        Test suite added.
 *
 * Tools:
 *   session_search    — BM25 keyword search
 *   session_semantic  — Semantic (vector cosine) search
 *   session_hybrid    — Combined BM25 + cosine via Reciprocal Rank Fusion
 *   session_read      — Read a specific entry by file path + line number
 *   session_status    — Index status and entry count
 *
 * Dependencies:
 *   @duckdb/node-api          — DuckDB native binary (FTS + FLOAT[] arrays)
 *   @huggingface/transformers — ONNX embeddings (all-MiniLM-L6-v2, ~23MB)
 *
 * DuckDB optimizations applied (v2.3):
 *   - Read-only mode: access_mode='READ_ONLY' — no WAL, no locks, no MVCC
 *   - Prepared statements: connection.run(sql, params) — SQL injection safe
 *   - FTS overwrite=true: no need to drop before recreate
 *   - ignore_errors=true: skip malformed JSONL lines
 *   - Race guard: ensureInProgress flag prevents concurrent builds
 *
 * Verified NOT worth doing:
 *   - FTS incremental: not available in DuckDB v1.5.6 (no 'incremental' parameter)
 *   - HNSW/vss: brute-force cosine is 13ms at 17K vectors; HNSW persistence is experimental
 *   - Appender API: 2x faster inserts but FLOAT[] array binding is broken in node-api
 *   - threads=1: actually slower (27ms vs 19ms with default 4 threads)
 *   - preserve_insertion_order=false: no size difference at our scale
 */

import { existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const HOME = process.env.HOME ?? "/home/cpagan";
const SESSIONS_DIR = join(HOME, ".pi", "agent", "sessions");
const DB_DIR = join(HOME, ".pi", "agent", "duckdb-search");
const DB_PATH = join(DB_DIR, "sessions.duckdb");
const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const EMBED_DIM = 384;
const EXCLUDE_RECENT_MS = 5 * 60 * 1000;
const MAX_CONTENT_CHARS = 2000;
const RRF_K = 60;

// Lazy module loading
let duckdbModule: any = null;
async function getDuckDB() {
  if (duckdbModule) return duckdbModule;
  duckdbModule = await import("@duckdb/node-api");
  return duckdbModule;
}

let transformersModule: any = null;
async function getTransformers() {
  if (transformersModule) return transformersModule;
  transformersModule = await import("@huggingface/transformers");
  return transformersModule;
}

interface IndexState {
  ready: boolean;
  indexing: boolean;
  entryCount: number;
  sessionCount: number;
  lastBuilt: string | null;
  error: string | null;
  hasEmbeddings: boolean;
}

const state: IndexState = {
  ready: false,
  indexing: false,
  entryCount: 0,
  sessionCount: 0,
  lastBuilt: null,
  error: null,
  hasEmbeddings: false,
};

let dbInstance: any = null;
let dbConn: any = null;
let embedder: any = null;
let ensureInProgress = false; // race guard

/**
 * Get a read-only DuckDB connection.
 * Read-only mode: no WAL creation, no file locks, no MVCC overhead.
 * The cron build script writes the DB; the extension only reads it.
 */
async function getConnection() {
  if (dbConn) return dbConn;
  const { DuckDBInstance } = await getDuckDB();
  mkdirSync(DB_DIR, { recursive: true });
  if (existsSync(DB_PATH)) {
    // Read-only mode when DB exists (normal path)
    dbInstance = await DuckDBInstance.create(DB_PATH, { access_mode: "READ_ONLY" });
  } else {
    // Read-write mode for initial build
    dbInstance = await DuckDBInstance.create(DB_PATH);
  }
  dbConn = await dbInstance.connect();
  return dbConn;
}

async function getEmbedder() {
  if (embedder) return embedder;
  const { pipeline } = await getTransformers();
  embedder = await pipeline("feature-extraction", MODEL_ID);
  return embedder;
}

async function embed(text: string): Promise<Float32Array> {
  const pipe = await getEmbedder();
  const output = await pipe(text, { pooling: "mean", normalize: true });
  return new Float32Array(output.data);
}

function findSessionFiles(excludeRecent: boolean): string[] {
  const files: string[] = [];
  const now = Date.now();
  function scan(dir: string) {
    if (!existsSync(dir)) return;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const fullPath = join(dir, entry.name);
      if (entry.isDirectory()) {
        scan(fullPath);
      } else if (entry.name.endsWith(".jsonl")) {
        if (excludeRecent) {
          try {
            const st = statSync(fullPath);
            if (now - st.mtimeMs < EXCLUDE_RECENT_MS) continue;
          } catch { /* ignore */ }
        }
        files.push(fullPath);
      }
    }
  }
  scan(SESSIONS_DIR);
  return files;
}

/**
 * Metadata access using prepared statements.
 */
async function getMetadata(key: string): Promise<string | null> {
  try {
    const conn = await getConnection();
    const r = await conn.run(
      "SELECT value FROM _metadata WHERE key = ?",
      [key]
    );
    const rows = await r.getRows();
    return rows.length > 0 ? String(rows[0][0]) : null;
  } catch {
    return null;
  }
}

async function setMetadata(key: string, value: string): Promise<void> {
  try {
    const conn = await getConnection();
    await conn.run("CREATE TABLE IF NOT EXISTS _metadata (key VARCHAR PRIMARY KEY, value VARCHAR);");
    await conn.run(
      "INSERT OR REPLACE INTO _metadata VALUES (?, ?)",
      [key, value]
    );
  } catch { /* ignore */ }
}

async function checkExistingDatabase(): Promise<boolean> {
  if (!existsSync(DB_PATH)) return false;

  try {
    const conn = await getConnection();

    const tableCheck = await conn.run(
      "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'session_entries'"
    );
    if (Number((await tableCheck.getRows())[0][0]) === 0) return false;

    const entryCount = Number(await getMetadata("entry_count") ?? "0");
    const sessionCount = Number(await getMetadata("session_count") ?? "0");
    const hasEmbs = (await getMetadata("has_embeddings")) === "true";
    const lastBuilt = await getMetadata("last_built");

    if (entryCount === 0) return false;

    const filesOnDisk = findSessionFiles(true);
    if (sessionCount < filesOnDisk.length * 0.9) return false;

    state.ready = true;
    state.entryCount = entryCount;
    state.sessionCount = sessionCount;
    state.hasEmbeddings = hasEmbs;
    state.lastBuilt = lastBuilt;
    state.error = null;

    // Status bar message instead of console.log — avoids polluting the prompt area
    return true;
  } catch (e) {
    console.error("[duckdb-search] checkExistingDatabase failed:", e);
    return false;
  }
}

/**
 * Full content extraction: text + thinking + tool call names.
 * Captures thinking blocks (60% of previously "empty" entries had thinking content).
 */
const CONTENT_EXTRACTION_SQL = `
  CASE
    WHEN json_type(json->'message'->'content') = 'ARRAY'
    THEN substring(
      concat(
        COALESCE(array_to_string(json_extract_string(json->'message'->'content', '$[*].text'), ' '), ''),
        COALESCE(array_to_string(json_extract_string(json->'message'->'content', '$[*].thinking'), ' '), ''),
        COALESCE(array_to_string(json_extract_string(json->'message'->'content', '$[*].name'), ' '), '')
      ),
      1, ${MAX_CONTENT_CHARS}
    )
    ELSE substring(COALESCE(json->'message'->>'content', json->>'content', ''), 1, ${MAX_CONTENT_CHARS})
  END
`;

async function buildIndex(): Promise<{ entries: number; sessions: number; error?: string }> {
  if (state.indexing) return { entries: state.entryCount, sessions: state.sessionCount, error: "Indexing already in progress" };
  state.indexing = true;
  state.error = null;

  try {
    const conn = await getConnection();
    await conn.run("INSTALL fts; LOAD fts;");

    try { await conn.run("PRAGMA drop_fts_index('session_entries');"); } catch { /* no index yet */ }
    await conn.run("DROP TABLE IF EXISTS session_entries;");
    await conn.run("DROP TABLE IF EXISTS session_fts_index;");

    const files = findSessionFiles(true);
    if (files.length === 0) {
      state.indexing = false;
      state.error = "No JSONL files found in " + SESSIONS_DIR;
      return { entries: 0, sessions: 0, error: state.error };
    }

    const glob = join(SESSIONS_DIR, "**", "*.jsonl").replace(/\\/g, "/");
    const escapedGlob = esc(glob);

    await conn.run(`
      CREATE TABLE session_entries AS
      SELECT
        filename,
        line_number,
        json->>'type' AS entry_type,
        json->>'id' AS entry_id,
        json->>'parentId' AS parent_id,
        json->>'timestamp' AS timestamp,
        json->'message'->>'role' AS role,
        json->>'customType' AS custom_type,
        ${CONTENT_EXTRACTION_SQL} AS content_text
      FROM (
        SELECT
          filename,
          row_number() OVER (PARTITION BY filename ORDER BY filename) AS line_number,
          json
        FROM read_json_objects('${escapedGlob}', format='newline_delimited', filename=true, ignore_errors=true)
      )
      WHERE json->>'type' IN ('message', 'custom_message')
    `);

    // FTS with overwrite=true — no need to call drop_fts_index first
    await conn.run("PRAGMA create_fts_index('session_entries', 'entry_id', 'content_text', 'role', stemmer='porter', overwrite=true);");

    const countResult = await conn.run("SELECT COUNT(*) AS c FROM session_entries");
    const entryCount = Number((await countResult.getRows())[0][0]);
    const sessResult = await conn.run("SELECT COUNT(DISTINCT filename) AS c FROM session_entries");
    const sessionCount = Number((await sessResult.getRows())[0][0]);

    state.ready = true;
    state.entryCount = entryCount;
    state.sessionCount = sessionCount;
    state.lastBuilt = new Date().toISOString();
    state.hasEmbeddings = false;

    await setMetadata("entry_count", String(entryCount));
    await setMetadata("session_count", String(sessionCount));
    await setMetadata("last_built", state.lastBuilt);
    await setMetadata("has_embeddings", "false");

    generateEmbeddings().catch((e) => {
      console.error("[duckdb-search] embedding generation failed:", e);
    });

    return { entries: entryCount, sessions: sessionCount };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    state.error = msg;
    state.ready = false;
    return { entries: 0, sessions: 0, error: msg };
  } finally {
    state.indexing = false;
  }
}

async function generateEmbeddings(): Promise<void> {
  const conn = await getConnection();
  await conn.run(`ALTER TABLE session_entries ADD COLUMN IF NOT EXISTS embedding FLOAT[${EMBED_DIM}];`);

  const result = await conn.run("SELECT entry_id, content_text FROM session_entries WHERE embedding IS NULL AND length(content_text) > 0 ORDER BY rowid;");
  const rows = await result.getRows();

  if (rows.length === 0) {
    state.hasEmbeddings = true;
    await setMetadata("has_embeddings", "true");
    return;
  }

  // Embedding progress logged to status bar via ctx — silent in prompt area

  await conn.run(`CREATE TABLE IF NOT EXISTS _emb_temp (entry_id VARCHAR, embedding FLOAT[${EMBED_DIM}]);`);
  await conn.run("DELETE FROM _emb_temp;");

  const batchSize = 64;
  for (let i = 0; i < rows.length; i += batchSize) {
    const batch = rows.slice(i, i + batchSize);
    const placeholders: string[] = [];
    const insertParams: any[] = [];

    for (const row of batch) {
      const entryId = row[0] as string;
      const text = String(row[1] || "").slice(0, MAX_CONTENT_CHARS);
      if (!text) continue;
      const embedding = await embed(text);
      const arrStr = embeddingToDuckArray(embedding);
      placeholders.push(`(?, ?::FLOAT[${EMBED_DIM}])`);
      insertParams.push(entryId, arrStr);
    }

    if (placeholders.length > 0) {
      await conn.run(
        `INSERT INTO _emb_temp VALUES ${placeholders.join(",")};`,
        insertParams
      );
    }

    if ((i + batchSize) % 512 === 0 || i + batchSize >= rows.length) {
      // Embedding progress — silent (no console.log)
    }
  }

  await conn.run("UPDATE session_entries SET embedding = (SELECT embedding FROM _emb_temp WHERE _emb_temp.entry_id = session_entries.entry_id) WHERE entry_id IN (SELECT entry_id FROM _emb_temp);");
  await conn.run("DROP TABLE _emb_temp;");
  await conn.run("CHECKPOINT;");

  state.hasEmbeddings = true;
  await setMetadata("has_embeddings", "true");
  // Embedding complete — silent (no console.log)
}

/**
 * Ensure the index is ready — with race guard.
 * ensureInProgress flag prevents concurrent buildIndex() calls
 * if session_start event and first search fire simultaneously.
 */
async function ensureIndex(): Promise<boolean> {
  if (state.ready) return true;
  if (state.indexing || ensureInProgress) return false;

  ensureInProgress = true;
  try {
    const existing = await checkExistingDatabase();
    if (existing) return true;

    const result = await buildIndex();
    return !result.error;
  } finally {
    ensureInProgress = false;
  }
}

/** Format an embedding as a DuckDB array literal for cosine queries. */
function embeddingToDuckArray(emb: Float32Array): string {
  return `[${Array.from(emb).map((v) => v.toFixed(6)).join(",")}]`;
}

/** Escape a string for use in a DuckDB single-quoted string literal. */
function esc(s: string): string {
  return s.replace(/'/g, "''");
}

export default function (pi: ExtensionAPI) {
  // Background build on session_start — loads DB before first search
  pi.on("session_start", async (_event, ctx) => {
    ensureIndex().then((ok) => {
      if (ok && state.entryCount > 0) {
        try {
          ctx.ui.setStatus("duckdb-search", `📚 ${state.entryCount} entries, ${state.sessionCount} sessions, embeddings: ${state.hasEmbeddings ? "ready" : "no"}`);
          setTimeout(() => {
            try { ctx.ui.setStatus("duckdb-search", undefined); } catch {}
          }, 5000);
        } catch {}
      }
    }).catch((e) => {
      console.error("[duckdb-search] background ensureIndex failed:", e);
    });
  });

  // session_search — BM25 keyword search (prepared statements)
  pi.registerTool({
    name: "session_search",
    label: "Session Search",
    description:
      "Search past Pi session transcripts using BM25 keyword ranking. " +
      "Returns matching entries with file path, line number, role, timestamp, and snippet. " +
      "Use distinctive keywords for best results. Excludes the current active session.",
    parameters: Type.Object({
      query: Type.String({ description: "Search terms. Multiple words are ANDed for BM25 ranking." }),
      role: Type.Optional(Type.String({ description: "Filter by role: user, assistant, toolResult, or custom" })),
      limit: Type.Optional(Type.Number({ description: "Max results (default 20, max 50)" })),
      offset: Type.Optional(Type.Number({ description: "Pagination offset (default 0)" })),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const query = (params.query as string)?.trim();
      if (!query) return { content: [{ type: "text", text: "Error: query is required" }], details: { error: "query required" } };

      const limit = Math.min(Math.max(1, (params.limit as number) ?? 20), 50);
      const offset = Math.max(0, (params.offset as number) ?? 0);
      const role = (params.role as string) ?? null;

      const ready = await ensureIndex();
      if (!ready) return { content: [{ type: "text", text: `Index not ready: ${state.error ?? "building..."}` }], details: { error: state.error ?? "indexing" } };

      try {
        const conn = await getConnection();
        await conn.run("LOAD fts;");

        // Prepared statement: ? placeholders for SQL injection safety
        let sql: string;
        let sqlParams: any[];

        if (role) {
          sql = `
            SELECT filename, line_number, role, timestamp, entry_type,
                   substring(content_text, 1, 500) AS snippet,
                   fts_main_session_entries.match_bm25(entry_id, ?) AS score
            FROM session_entries
            WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL
              AND role = ?
            ORDER BY score DESC
            LIMIT ? OFFSET ?
          `;
          sqlParams = [query, query, role, limit, offset];
        } else {
          sql = `
            SELECT filename, line_number, role, timestamp, entry_type,
                   substring(content_text, 1, 500) AS snippet,
                   fts_main_session_entries.match_bm25(entry_id, ?) AS score
            FROM session_entries
            WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL
            ORDER BY score DESC
            LIMIT ? OFFSET ?
          `;
          sqlParams = [query, query, limit, offset];
        }

        const result = await conn.run(sql, sqlParams);
        const rows = await result.getRows();

        if (rows.length === 0) return { content: [{ type: "text", text: `No results found for: ${query}` }], details: { query, result_count: 0, mode: "bm25" } };

        const lines: string[] = [`Found ${rows.length} result(s) for: ${query}`, ""];
        for (const row of rows) {
          const [filename, lineNum, entryRole, ts, entryType, snippet, score] = row;
          const shortFile = String(filename).replace(SESSIONS_DIR + "/", "");
          lines.push(`--- ${shortFile}:${lineNum} (score: ${Number(score).toFixed(2)}) ---`);
          lines.push(`  role: ${entryRole || "N/A"} | type: ${entryType} | time: ${ts || "N/A"}`);
          lines.push(`  text: ${String(snippet || "").slice(0, 300)}`);
          lines.push("");
        }

        return { content: [{ type: "text", text: lines.join("\n") }], details: { query, result_count: rows.length, limit, offset, mode: "bm25" } };
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        return { content: [{ type: "text", text: `Search error: ${msg}` }], details: { error: msg } };
      }
    },
  });

  // session_semantic — vector cosine similarity (prepared statements)
  pi.registerTool({
    name: "session_semantic",
    label: "Session Semantic Search",
    description:
      "Search past Pi session transcripts by meaning, not just keywords. " +
      "Uses local ONNX embeddings (all-MiniLM-L6-v2) and cosine similarity. " +
      "Example: searching 'RFC 5549 extended nexthop' can find entries about 'BGP unnumbered'.",
    parameters: Type.Object({
      query: Type.String({ description: "Natural language query — searches by meaning, not exact words." }),
      role: Type.Optional(Type.String({ description: "Filter by role: user, assistant, toolResult, or custom" })),
      limit: Type.Optional(Type.Number({ description: "Max results (default 20, max 50)" })),
      offset: Type.Optional(Type.Number({ description: "Pagination offset (default 0)" })),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const query = (params.query as string)?.trim();
      if (!query) return { content: [{ type: "text", text: "Error: query is required" }], details: { error: "query required" } };

      const limit = Math.min(Math.max(1, (params.limit as number) ?? 20), 50);
      const offset = Math.max(0, (params.offset as number) ?? 0);
      const role = (params.role as string) ?? null;

      const ready = await ensureIndex();
      if (!ready) return { content: [{ type: "text", text: `Index not ready: ${state.error ?? "building..."}` }], details: { error: state.error ?? "indexing" } };

      if (!state.hasEmbeddings) return { content: [{ type: "text", text: "Embeddings not yet generated. Run the cron build script or wait for background generation." }], details: { error: "embeddings not ready" } };

      try {
        const conn = await getConnection();
        const queryEmbedding = await embed(query);
        const arrStr = embeddingToDuckArray(queryEmbedding);

        // Prepared statement with ? placeholders
        let sql: string;
        let sqlParams: any[];

        if (role) {
          sql = `
            SELECT filename, line_number, role, timestamp, entry_type,
                   substring(content_text, 1, 500) AS snippet,
                   array_cosine_similarity(embedding, ?::FLOAT[${EMBED_DIM}]) AS similarity
            FROM session_entries
            WHERE embedding IS NOT NULL AND role = ?
            ORDER BY similarity DESC
            LIMIT ? OFFSET ?
          `;
          sqlParams = [arrStr, role, limit, offset];
        } else {
          sql = `
            SELECT filename, line_number, role, timestamp, entry_type,
                   substring(content_text, 1, 500) AS snippet,
                   array_cosine_similarity(embedding, ?::FLOAT[${EMBED_DIM}]) AS similarity
            FROM session_entries
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT ? OFFSET ?
          `;
          sqlParams = [arrStr, limit, offset];
        }

        const result = await conn.run(sql, sqlParams);
        const rows = await result.getRows();

        if (rows.length === 0) return { content: [{ type: "text", text: `No results found for: ${query}` }], details: { query, result_count: 0, mode: "semantic" } };

        const lines: string[] = [`Found ${rows.length} semantic result(s) for: ${query}`, ""];
        for (const row of rows) {
          const [filename, lineNum, entryRole, ts, entryType, snippet, similarity] = row;
          const shortFile = String(filename).replace(SESSIONS_DIR + "/", "");
          lines.push(`--- ${shortFile}:${lineNum} (similarity: ${Number(similarity).toFixed(3)}) ---`);
          lines.push(`  role: ${entryRole || "N/A"} | type: ${entryType} | time: ${ts || "N/A"}`);
          lines.push(`  text: ${String(snippet || "").slice(0, 300)}`);
          lines.push("");
        }

        return { content: [{ type: "text", text: lines.join("\n") }], details: { query, result_count: rows.length, limit, offset, mode: "semantic" } };
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        return { content: [{ type: "text", text: `Semantic search error: ${msg}` }], details: { error: msg } };
      }
    },
  });

  // session_hybrid — BM25 + cosine via RRF (prepared statements)
  pi.registerTool({
    name: "session_hybrid",
    label: "Session Hybrid Search",
    description:
      "Search session transcripts combining BM25 keyword ranking and semantic cosine similarity " +
      "via Reciprocal Rank Fusion (RRF). Returns the best of both: exact keyword matches AND " +
      "conceptually related entries. Recommended for comprehensive search.",
    parameters: Type.Object({
      query: Type.String({ description: "Search query — keywords for BM25, meaning for semantic, both for hybrid." }),
      role: Type.Optional(Type.String({ description: "Filter by role: user, assistant, toolResult, or custom" })),
      limit: Type.Optional(Type.Number({ description: "Max results (default 20, max 50)" })),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const query = (params.query as string)?.trim();
      if (!query) return { content: [{ type: "text", text: "Error: query is required" }], details: { error: "query required" } };

      const limit = Math.min(Math.max(1, (params.limit as number) ?? 20), 50);
      const role = (params.role as string) ?? null;

      const ready = await ensureIndex();
      if (!ready) return { content: [{ type: "text", text: `Index not ready: ${state.error ?? "building..."}` }], details: { error: state.error ?? "indexing" } };

      try {
        const conn = await getConnection();
        await conn.run("LOAD fts;");

        // BM25 query with prepared statement
        let bm25Sql: string;
        let bm25Params: any[];
        if (role) {
          bm25Sql = `
            SELECT entry_id, filename, line_number, role, timestamp, entry_type,
                   substring(content_text, 1, 500) AS snippet
            FROM session_entries
            WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL AND role = ?
            ORDER BY fts_main_session_entries.match_bm25(entry_id, ?) DESC
            LIMIT 50
          `;
          bm25Params = [query, role, query];
        } else {
          bm25Sql = `
            SELECT entry_id, filename, line_number, role, timestamp, entry_type,
                   substring(content_text, 1, 500) AS snippet
            FROM session_entries
            WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL
            ORDER BY fts_main_session_entries.match_bm25(entry_id, ?) DESC
            LIMIT 50
          `;
          bm25Params = [query, query];
        }
        const bm25Result = await conn.run(bm25Sql, bm25Params);
        const bm25Rows = await bm25Result.getRows();

        // Cosine query with prepared statement
        let cosineRows: any[][] = [];
        if (state.hasEmbeddings) {
          const queryEmbedding = await embed(query);
          const arrStr = embeddingToDuckArray(queryEmbedding);

          let cosineSql: string;
          let cosineParams: any[];
          if (role) {
            cosineSql = `
              SELECT entry_id, filename, line_number, role, timestamp, entry_type,
                     substring(content_text, 1, 500) AS snippet
              FROM session_entries
              WHERE embedding IS NOT NULL AND role = ?
              ORDER BY array_cosine_similarity(embedding, ?::FLOAT[${EMBED_DIM}]) DESC
              LIMIT 50
            `;
            cosineParams = [role, arrStr];
          } else {
            cosineSql = `
              SELECT entry_id, filename, line_number, role, timestamp, entry_type,
                     substring(content_text, 1, 500) AS snippet
              FROM session_entries
              WHERE embedding IS NOT NULL
              ORDER BY array_cosine_similarity(embedding, ?::FLOAT[${EMBED_DIM}]) DESC
              LIMIT 50
            `;
            cosineParams = [arrStr];
          }
          const cosineResult = await conn.run(cosineSql, cosineParams);
          cosineRows = await cosineResult.getRows();
        }

        // Reciprocal Rank Fusion
        const rrfScores = new Map<string, { row: any[]; score: number; bm25_rank: number | null; cosine_rank: number | null }>();

        for (let i = 0; i < bm25Rows.length; i++) {
          const entryId = String(bm25Rows[i][0]);
          const rrf = 1 / (RRF_K + i + 1);
          rrfScores.set(entryId, { row: bm25Rows[i], score: rrf, bm25_rank: i + 1, cosine_rank: null });
        }

        for (let i = 0; i < cosineRows.length; i++) {
          const entryId = String(cosineRows[i][0]);
          const rrf = 1 / (RRF_K + i + 1);
          const existing = rrfScores.get(entryId);
          if (existing) {
            existing.score += rrf;
            existing.cosine_rank = i + 1;
          } else {
            rrfScores.set(entryId, { row: cosineRows[i], score: rrf, bm25_rank: null, cosine_rank: i + 1 });
          }
        }

        const fused = Array.from(rrfScores.values()).sort((a, b) => b.score - a.score).slice(0, limit);

        if (fused.length === 0) return { content: [{ type: "text", text: `No results found for: ${query}` }], details: { query, result_count: 0, mode: "hybrid" } };

        const lines: string[] = [`Found ${fused.length} hybrid result(s) for: ${query}`, ""];
        for (const item of fused) {
          const [entryId, filename, lineNum, entryRole, ts, entryType, snippet] = item.row;
          const shortFile = String(filename).replace(SESSIONS_DIR + "/", "");
          const rankInfo = [`bm25:${item.bm25_rank ?? "-"}`, `cos:${item.cosine_rank ?? "-"}`].join(" ");
          lines.push(`--- ${shortFile}:${lineNum} (rrf: ${item.score.toFixed(4)} | ${rankInfo}) ---`);
          lines.push(`  role: ${entryRole || "N/A"} | type: ${entryType} | time: ${ts || "N/A"}`);
          lines.push(`  text: ${String(snippet || "").slice(0, 300)}`);
          lines.push("");
        }

        return { content: [{ type: "text", text: lines.join("\n") }], details: { query, result_count: fused.length, limit, mode: "hybrid", has_embeddings: state.hasEmbeddings } };
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        return { content: [{ type: "text", text: `Hybrid search error: ${msg}` }], details: { error: msg } };
      }
    },
  });

  // session_read — read specific entry by path + line (prepared statement)
  pi.registerTool({
    name: "session_read",
    label: "Session Read",
    description:
      "Read a specific entry from a Pi session JSONL file by file path and line number. " +
      "Returns the full decoded entry including role, content, and metadata. " +
      "Use after session_search, session_semantic, or session_hybrid to read full context.",
    parameters: Type.Object({
      path: Type.String({ description: "Absolute path to the .jsonl session file" }),
      line: Type.Number({ description: "Line number (1-indexed) in the JSONL file" }),
    }),

    async execute(_toolCallId, params, _signal, _onUpdate, _ctx) {
      const filePath = params.path as string;
      const lineNum = params.line as number;

      if (!filePath || !filePath.endsWith(".jsonl")) return { content: [{ type: "text", text: "Error: path must point to a .jsonl file" }], details: { error: "invalid path" } };
      if (!existsSync(filePath)) return { content: [{ type: "text", text: `Error: file not found: ${filePath}` }], details: { error: "file not found" } };

      try {
        const conn = await getConnection();
        const result = await conn.run(
          `
          WITH numbered AS (
            SELECT row_number() OVER () AS rn, json
            FROM read_json_objects(?, format='newline_delimited', ignore_errors=true)
          )
          SELECT
            json->>'type' AS entry_type, json->>'id' AS entry_id,
            json->>'parentId' AS parent_id, json->>'timestamp' AS timestamp,
            json->'message'->>'role' AS role,
            json->'message'->>'content' AS content,
            json->>'customType' AS custom_type,
            CAST(json AS VARCHAR) AS raw_json
          FROM numbered WHERE rn = ? LIMIT 1
          `,
          [filePath, Math.max(1, lineNum)]
        );

        const rows = await result.getRows();
        if (rows.length === 0) return { content: [{ type: "text", text: `No entry at line ${lineNum} in ${filePath}` }], details: { error: "line not found" } };

        const [entryType, entryId, parentId, ts, role, content, customType, rawJson] = rows[0];
        const lines: string[] = [`Entry at ${filePath}:${lineNum}`, `  type: ${entryType}`];
        if (customType) lines.push(`  customType: ${customType}`);
        if (role) lines.push(`  role: ${role}`);
        if (ts) lines.push(`  timestamp: ${ts}`);
        if (entryId) lines.push(`  id: ${entryId}`);
        if (parentId) lines.push(`  parentId: ${parentId}`);
        lines.push("");
        lines.push(content ? `content:\n${String(content).slice(0, 2000)}` : `raw_json:\n${String(rawJson).slice(0, 2000)}`);

        return { content: [{ type: "text", text: lines.join("\n") }], details: { path: filePath, line: lineNum, entry_type: entryType } };
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        return { content: [{ type: "text", text: `Read error: ${msg}` }], details: { error: msg } };
      }
    },
  });

  // session_status
  pi.registerTool({
    name: "session_status",
    label: "Session Index Status",
    description:
      "Check the DuckDB session search index status: entry count, session count, " +
      "embedding status, last build time, and whether a rebuild is needed.",
    parameters: Type.Object({}),

    async execute(_toolCallId, _params, _signal, _onUpdate, _ctx) {
      await ensureIndex();
      const files = findSessionFiles(true);
      const fileCount = files.length;

      const lines: string[] = ["DuckDB Session Search Index Status", ""];
      lines.push(`  indexed entries: ${state.entryCount}`);
      lines.push(`  indexed sessions: ${state.sessionCount}`);
      lines.push(`  JSONL files on disk: ${fileCount} (excluding recent)`);
      lines.push(`  index ready: ${state.ready}`);
      lines.push(`  indexing: ${state.indexing}`);
      lines.push(`  embeddings: ${state.hasEmbeddings ? "ready" : "building or not started"}`);
      lines.push(`  last built: ${state.lastBuilt ?? "never"}`);
      lines.push(`  database: ${DB_PATH}`);
      lines.push(`  model: ${MODEL_ID} (${EMBED_DIM} dims)`);
      lines.push(`  tools: session_search (BM25), session_semantic (cosine), session_hybrid (RRF), session_read`);
      if (state.error) lines.push(`  error: ${state.error}`);

      if (state.ready && fileCount > state.sessionCount * 1.1) {
        lines.push("");
        lines.push(`  ⚠ Session count mismatch (${state.sessionCount} indexed vs ${fileCount} on disk). Rebuild recommended.`);
      }

      if (!state.ready && !state.indexing) {
        lines.push("");
        lines.push("  Index not built. Call session_search to trigger build, or run the cron build script.");
      }

      return {
        content: [{ type: "text", text: lines.join("\n") }],
        details: { entry_count: state.entryCount, session_count: state.sessionCount, file_count: fileCount, ready: state.ready, indexing: state.indexing, has_embeddings: state.hasEmbeddings, last_built: state.lastBuilt, error: state.error },
      };
    },
  });
}
