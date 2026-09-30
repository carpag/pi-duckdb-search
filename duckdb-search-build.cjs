#!/usr/bin/env node
/**
 * duckdb-search-build.cjs — Standalone build script for DuckDB session search.
 *
 * v2.3: Full content extraction, incremental indexing, ignore_errors, FTS overwrite=true.
 *
 * Stores metadata in _metadata table for the extension to read.
 *
 * Run via cron (daily at 4am):
 *   0 4 * * * /home/cpagan/.local/share/pi-node/node-v22.23.0-linux-x64/bin/node \
 *     /home/cpagan/.pi/agent/duckdb-search/duckdb-search-build.cjs >> \
 *     /home/cpagan/.pi/agent/duckdb-search/build.log 2>&1
 */

const { DuckDBInstance } = require("@duckdb/node-api");
const path = require("node:path");
const fs = require("node:fs");

const HOME = process.env.HOME || "/home/cpagan";
const SESSIONS_DIR = path.join(HOME, ".pi", "agent", "sessions");
const DB_DIR = path.join(HOME, ".pi", "agent", "duckdb-search");
const DB_PATH = path.join(DB_DIR, "sessions.duckdb");
const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const EMBED_DIM = 384;
const EXCLUDE_RECENT_MS = 5 * 60 * 1000;
const MAX_CONTENT_CHARS = 2000;

function ts() { return new Date().toISOString(); }
function log(msg) { console.log(`[${ts()}] ${msg}`); }

/** Find all JSONL files, excluding files modified in last 5 minutes. */
function findSessionFiles() {
  const files = [];
  const now = Date.now();
  function scan(dir) {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) scan(full);
      else if (entry.name.endsWith(".jsonl")) {
        try {
          const st = fs.statSync(full);
          if (now - st.mtimeMs < EXCLUDE_RECENT_MS) {
            log(`  skipping recent file: ${entry.name} (modified ${Math.round((now - st.mtimeMs) / 1000)}s ago)`);
            continue;
          }
          files.push({ path: full, mtime: st.mtimeMs });
        } catch { /* ignore */ }
      }
    }
  }
  scan(SESSIONS_DIR);
  return files;
}

async function getMetadata(conn, key) {
  try {
    const r = await conn.run(`SELECT value FROM _metadata WHERE key = '${key.replace(/'/g, "''")}';`);
    const rows = await r.getRows();
    return rows.length > 0 ? String(rows[0][0]) : null;
  } catch { return null; }
}

async function setMetadata(conn, key, value) {
  await conn.run(`CREATE TABLE IF NOT EXISTS _metadata (key VARCHAR PRIMARY KEY, value VARCHAR);`);
  await conn.run(`INSERT OR REPLACE INTO _metadata VALUES ('${key.replace(/'/g, "''")}', '${value.replace(/'/g, "''")}');`);
}

async function loadEmbedder() {
  log("Loading embedding model...");
  const { pipeline } = await import("@huggingface/transformers");
  const embedder = await pipeline("feature-extraction", MODEL_ID);
  log("Model loaded");
  return embedder;
}

async function embed(embedder, text) {
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return new Float32Array(output.data);
}

/** Full rebuild from scratch — used when no existing DB or schema changed. */
async function fullBuild(conn, files) {
  log("Full rebuild (no existing database or schema mismatch)...");

  try { await conn.run("PRAGMA drop_fts_index('session_entries');"); } catch {}
  await conn.run("DROP TABLE IF EXISTS session_entries;");
  await conn.run("DROP TABLE IF EXISTS _emb_temp;");

  const glob = path.join(SESSIONS_DIR, "**", "*.jsonl").replace(/\\/g, "/");

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
      CASE
        WHEN json_type(json->'message'->'content') = 'ARRAY'
        THEN substring(
          concat(
            COALESCE(array_to_string(json_extract_string(json->'message'->'content', '$[*].text'), ' '), ''),
            COALESCE(array_to_string(json_extract_string(json->'message'->'content', '$[*].thinking'), ' '), ''),
            COALESCE(array_to_string(json_extract_string(json->'message'->'content', '$[*].name'), ' '), '')
          ), 1, ${MAX_CONTENT_CHARS}
        )
        ELSE substring(COALESCE(json->'message'->>'content', json->>'content', ''), 1, ${MAX_CONTENT_CHARS})
      END AS content_text
    FROM (
      SELECT
        filename,
        row_number() OVER (PARTITION BY filename ORDER BY filename) AS line_number,
        json
      FROM read_json_objects('${glob}', format='newline_delimited', filename=true, ignore_errors=true)
    )
    WHERE json->>'type' IN ('message', 'custom_message')
  `);

  const entryCount = Number((await (await conn.run("SELECT COUNT(*) FROM session_entries")).getRows())[0][0]);
  const sessionCount = Number((await (await conn.run("SELECT COUNT(DISTINCT filename) FROM session_entries")).getRows())[0][0]);
  log(`Indexed ${entryCount} entries from ${sessionCount} sessions`);

  log("Building FTS index...");
  await conn.run("PRAGMA create_fts_index('session_entries', 'entry_id', 'content_text', 'role', stemmer='porter', overwrite=true);");

  // Add embedding column
  await conn.run(`ALTER TABLE session_entries ADD COLUMN IF NOT EXISTS embedding FLOAT[${EMBED_DIM}];`);

  // Generate all embeddings
  const embedder = await loadEmbedder();
  await generateEmbeddingsForNew(conn, embedder);

  return { entryCount, sessionCount };
}

/** Generate embeddings only for entries where embedding IS NULL. */
async function generateEmbeddingsForNew(conn, embedder) {
  const result = await conn.run("SELECT entry_id, content_text FROM session_entries WHERE embedding IS NULL AND length(content_text) > 0 ORDER BY rowid;");
  const rows = await result.getRows();

  if (rows.length === 0) {
    log("All entries already have embeddings.");
    return 0;
  }

  log(`Generating embeddings for ${rows.length} entries...`);

  await conn.run(`CREATE TABLE IF NOT EXISTS _emb_temp (entry_id VARCHAR, embedding FLOAT[${EMBED_DIM}]);`);
  await conn.run("DELETE FROM _emb_temp;");

  const batchSize = 64;
  let embedded = 0;
  for (let i = 0; i < rows.length; i += batchSize) {
    const batch = rows.slice(i, i + batchSize);
    const placeholders = [];
    const insertParams = [];

    for (const row of batch) {
      const entryId = row[0];
      const text = String(row[1] || "").slice(0, MAX_CONTENT_CHARS);
      if (!text) continue;
      const embedding = await embed(embedder, text);
      const arr = Array.from(embedding).map((v) => v.toFixed(6)).join(",");
      placeholders.push(`(?, ?::FLOAT[${EMBED_DIM}])`);
      insertParams.push(entryId, `[${arr}]`);
    }

    if (placeholders.length > 0) {
      await conn.run(
        `INSERT INTO _emb_temp VALUES ${placeholders.join(",")};`,
        insertParams
      );
    }

    embedded = Math.min(i + batchSize, rows.length);
    if (embedded % 512 < batchSize || embedded >= rows.length) {
      log(`  Embedded ${embedded}/${rows.length}`);
    }
  }

  log("Merging embeddings...");
  await conn.run("UPDATE session_entries SET embedding = (SELECT embedding FROM _emb_temp WHERE _emb_temp.entry_id = session_entries.entry_id) WHERE entry_id IN (SELECT entry_id FROM _emb_temp);");
  await conn.run("DROP TABLE _emb_temp;");

  return rows.length;
}

/** Incremental build — only process new/modified files. */
async function incrementalBuild(conn, files) {
  log("Incremental build...");

  // Get list of already-indexed filenames
  const indexedResult = await conn.run("SELECT DISTINCT filename FROM session_entries;");
  const indexedFiles = new Set((await indexedResult.getRows()).map((r) => String(r[0])));
  log(`  Previously indexed: ${indexedFiles.size} files`);

  // Find new files (on disk but not in DB)
  const newFiles = files.filter((f) => !indexedFiles.has(f.path));
  // Find deleted files (in DB but not on disk)
  const diskPaths = new Set(files.map((f) => f.path));
  const deletedFiles = Array.from(indexedFiles).filter((f) => !diskPaths.has(f));

  log(`  New files: ${newFiles.length}`);
  log(`  Deleted files: ${deletedFiles.length}`);

  // Delete entries from removed files
  for (const filePath of deletedFiles) {
    await conn.run(`DELETE FROM session_entries WHERE filename = '${filePath.replace(/'/g, "''")}';`);
    log(`    Deleted entries for: ${path.basename(filePath)}`);
  }

  // Insert entries for new files
  if (newFiles.length > 0) {
    for (const file of newFiles) {
      const escapedPath = file.path.replace(/'/g, "''").replace(/\\/g, "/");
      await conn.run(`
        INSERT INTO session_entries (filename, line_number, entry_type, entry_id, parent_id, timestamp, role, custom_type, content_text)
        SELECT
          filename,
          line_number,
          json->>'type' AS entry_type,
          json->>'id' AS entry_id,
          json->>'parentId' AS parent_id,
          json->>'timestamp' AS timestamp,
          json->'message'->>'role' AS role,
          json->>'customType' AS custom_type,
          CASE
            WHEN json_type(json->'message'->'content') = 'ARRAY'
            THEN substring(
              concat(
                COALESCE(array_to_string(json_extract_string(json->'message'->'content', '$[*].text'), ' '), ''),
                COALESCE(array_to_string(json_extract_string(json->'message'->'content', '$[*].thinking'), ' '), ''),
                COALESCE(array_to_string(json_extract_string(json->'message'->'content', '$[*].name'), ' '), '')
              ), 1, ${MAX_CONTENT_CHARS}
            )
            ELSE substring(COALESCE(json->'message'->>'content', json->>'content', ''), 1, ${MAX_CONTENT_CHARS})
          END AS content_text
        FROM (
          SELECT
            '${escapedPath}' AS filename,
            row_number() OVER () AS line_number,
            json
          FROM read_json_objects('${escapedPath}', format='newline_delimited', ignore_errors=true)
        )
        WHERE json->>'type' IN ('message', 'custom_message')
      `);
      log(`    Inserted: ${path.basename(file.path)}`);
    }
  }

  // Ensure embedding column exists
  await conn.run(`ALTER TABLE session_entries ADD COLUMN IF NOT EXISTS embedding FLOAT[${EMBED_DIM}];`);

  // Rebuild FTS index (DuckDB FTS doesn't support incremental)
  log("Rebuilding FTS index...");
  await conn.run("PRAGMA create_fts_index('session_entries', 'entry_id', 'content_text', 'role', stemmer='porter', overwrite=true);");

  // Generate embeddings for any entries that don't have them (new files or crash recovery)
  let newEmbeddings = 0;
  const needEmbResult = await conn.run("SELECT COUNT(*) FROM session_entries WHERE embedding IS NULL AND length(content_text) > 0;");
  const needEmb = Number((await needEmbResult.getRows())[0][0]);
  if (needEmb > 0) {
    log(`${needEmb} entries need embeddings — generating...`);
    const embedder = await loadEmbedder();
    newEmbeddings = await generateEmbeddingsForNew(conn, embedder);
  } else {
    log("All entries have embeddings.");
  }

  // Count final state
  const entryCount = Number((await (await conn.run("SELECT COUNT(*) FROM session_entries")).getRows())[0][0]);
  const sessionCount = Number((await (await conn.run("SELECT COUNT(DISTINCT filename) FROM session_entries")).getRows())[0][0]);
  const embCount = Number((await (await conn.run("SELECT COUNT(*) FROM session_entries WHERE embedding IS NOT NULL")).getRows())[0][0]);

  return { entryCount, sessionCount, newEmbeddings, embCount };
}

async function main() {
  log("=== DuckDB session search build started ===");
  fs.mkdirSync(DB_DIR, { recursive: true });

  const files = findSessionFiles();
  log(`Found ${files.length} JSONL files (excluding recent)`);
  if (files.length === 0) { log("No files to index. Exiting."); process.exit(0); }

  const dbExists = fs.existsSync(DB_PATH);
  let conn;

  if (!dbExists) {
    log("No existing database — creating new...");
    const inst = await DuckDBInstance.create(DB_PATH);
    conn = await inst.connect();
    await conn.run("INSTALL fts; LOAD fts;");

    const { entryCount, sessionCount } = await fullBuild(conn, files);
    await conn.run("CHECKPOINT;");

    // Store metadata
    await setMetadata(conn, "entry_count", String(entryCount));
    await setMetadata(conn, "session_count", String(sessionCount));
    await setMetadata(conn, "last_built", ts());
    await setMetadata(conn, "has_embeddings", "true");
    await setMetadata(conn, "build_type", "full");

    const embCount = Number((await (await conn.run("SELECT COUNT(*) FROM session_entries WHERE embedding IS NOT NULL")).getRows())[0][0]);
    log(`Embeddings: ${embCount}/${entryCount}`);

    await conn.run("CHECKPOINT;");
    try { await conn.disconnect(); } catch {}
    log(`Database size: ${(fs.statSync(DB_PATH).size / 1024 / 1024).toFixed(1)} MB`);
    log("=== Build complete (full) ===");
    return;
  }

  // Existing DB — try incremental
  log("Opening existing database...");
  const inst = await DuckDBInstance.create(DB_PATH);
  conn = await inst.connect();
  await conn.run("LOAD fts;");

  // Check if session_entries table exists
  const tableCheck = await conn.run("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'session_entries'");
  const hasTable = Number((await tableCheck.getRows())[0][0]) > 0;

  let result;
  if (!hasTable) {
    result = await fullBuild(conn, files);
    await setMetadata(conn, "build_type", "full");
  } else {
    result = await incrementalBuild(conn, files);
    await setMetadata(conn, "build_type", "incremental");
  }

  await conn.run("CHECKPOINT;");

  // Store metadata
  await setMetadata(conn, "entry_count", String(result.entryCount));
  await setMetadata(conn, "session_count", String(result.sessionCount));
  await setMetadata(conn, "last_built", ts());
  await setMetadata(conn, "has_embeddings", "true");

  // Final stats
  const embCount = Number((await (await conn.run("SELECT COUNT(*) FROM session_entries WHERE embedding IS NOT NULL")).getRows())[0][0]);
  log(`Final: ${result.entryCount} entries, ${result.sessionCount} sessions, ${embCount} embeddings`);

  try { await conn.disconnect(); } catch {}
  log(`Database size: ${(fs.statSync(DB_PATH).size / 1024 / 1024).toFixed(1)} MB`);
  log("=== Build complete ===");
}

main().catch((e) => {
  log(`FATAL: ${e.message}`);
  log(e.stack || "");
  process.exit(1);
});
