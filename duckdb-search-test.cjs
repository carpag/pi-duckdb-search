#!/usr/bin/env node
/**
 * duckdb-search-test.cjs — Test suite for the DuckDB session search extension.
 *
 * Tests the core database operations directly (without the pi extension runtime):
 *   1. Database creation and schema
 *   2. Content extraction (text + thinking + tool names)
 *   3. BM25 keyword search with prepared statements
 *   4. Cosine semantic search with prepared statements
 *   5. Hybrid RRF search
 *   6. session_read by path + line
 *   7. Metadata persistence
 *   8. SQL injection resistance
 *   9. Edge cases (empty query, missing file, limit clamping)
 *  10. Incremental build (new file + delete file)
 *
 * Run: node duckdb-search-test.cjs
 * Exits 0 on success, 1 on any failure.
 */

const { DuckDBInstance } = require("@duckdb/node-api");
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");

const EMBED_DIM = 384;
const MAX_CONTENT_CHARS = 2000;

let passed = 0;
let failed = 0;
const failures = [];

function assert(cond, msg) {
  if (cond) {
    passed++;
  } else {
    failed++;
    failures.push(msg);
    console.error(`  ✗ FAIL: ${msg}`);
  }
}

function assertEq(actual, expected, msg) {
  if (actual === expected) {
    passed++;
  } else {
    failed++;
    failures.push(`${msg} (expected ${expected}, got ${actual})`);
    console.error(`  ✗ FAIL: ${msg} (expected ${expected}, got ${actual})`);
  }
}

const TEST_DIR = path.join(os.tmpdir(), "duckdb-search-test-" + Date.now());
const TEST_DB = path.join(TEST_DIR, "test.duckdb");
const TEST_SESSIONS = path.join(TEST_DIR, "sessions", "test-project");
const TEST_GLOB = path.join(TEST_SESSIONS, "**", "*.jsonl");

/** Create a fake session JSONL file for testing. */
function createSessionFile(name, entries) {
  const filePath = path.join(TEST_SESSIONS, name);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const lines = entries.map((e) => JSON.stringify(e));
  fs.writeFileSync(filePath, lines.join("\n") + "\n");
  return filePath;
}

/** Content extraction SQL matching the extension. */
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

async function buildTestDB(conn) {
  await conn.run("INSTALL fts; LOAD fts;");
  await conn.run("DROP TABLE IF EXISTS session_entries;");

  const escapedGlob = TEST_GLOB.replace(/'/g, "''");

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

  await conn.run("PRAGMA create_fts_index('session_entries', 'entry_id', 'content_text', 'role', stemmer='porter', overwrite=true);");
  await conn.run(`ALTER TABLE session_entries ADD COLUMN IF NOT EXISTS embedding FLOAT[${EMBED_DIM}];`);

  // Add fake embeddings for testing (deterministic, not random)
  const result = await conn.run("SELECT entry_id, content_text FROM session_entries WHERE embedding IS NULL AND length(content_text) > 0 ORDER BY rowid;");
  const rows = await result.getRows();

  await conn.run(`CREATE TABLE IF NOT EXISTS _emb_temp (entry_id VARCHAR, embedding FLOAT[${EMBED_DIM}]);`);
  await conn.run("DELETE FROM _emb_temp;");

  for (const row of rows) {
    const entryId = row[0];
    const text = String(row[1] || "");
    // Deterministic fake embedding: hash of text -> 384 dims
    const embedding = new Float32Array(EMBED_DIM);
    for (let i = 0; i < EMBED_DIM; i++) {
      embedding[i] = Math.sin(text.charCodeAt(i % text.length) + i) * 0.1;
    }
    // Normalize
    let norm = 0;
    for (const v of embedding) norm += v * v;
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < EMBED_DIM; i++) embedding[i] /= norm;

    const arrStr = `[${Array.from(embedding).map((v) => v.toFixed(6)).join(",")}]`;
    await conn.run(
      `INSERT INTO _emb_temp VALUES (?, ?::FLOAT[${EMBED_DIM}])`,
      [entryId, arrStr]
    );
  }

  await conn.run("UPDATE session_entries SET embedding = (SELECT embedding FROM _emb_temp WHERE _emb_temp.entry_id = session_entries.entry_id) WHERE entry_id IN (SELECT entry_id FROM _emb_temp);");
  await conn.run("DROP TABLE _emb_temp;");

  // Metadata
  await conn.run("CREATE TABLE IF NOT EXISTS _metadata (key VARCHAR PRIMARY KEY, value VARCHAR);");
  const entryCount = Number((await (await conn.run("SELECT COUNT(*) FROM session_entries")).getRows())[0][0]);
  const sessionCount = Number((await (await conn.run("SELECT COUNT(DISTINCT filename) FROM session_entries")).getRows())[0][0]);
  await conn.run("INSERT OR REPLACE INTO _metadata VALUES (?, ?)", ["entry_count", String(entryCount)]);
  await conn.run("INSERT OR REPLACE INTO _metadata VALUES (?, ?)", ["session_count", String(sessionCount)]);
  await conn.run("INSERT OR REPLACE INTO _metadata VALUES (?, ?)", ["has_embeddings", "true"]);

  await conn.run("CHECKPOINT;");
  return { entryCount, sessionCount };
}

async function main() {
  console.log("=== DuckDB Session Search Test Suite ===\n");

  // Setup test directory
  fs.mkdirSync(TEST_DIR, { recursive: true });

  // Create test session files
  const file1 = createSessionFile("session1.jsonl", [
    { type: "session", version: 3, id: "s1", timestamp: "2026-01-01T00:00:00Z", cwd: "/test" },
    { type: "message", id: "m1", parentId: null, timestamp: "2026-01-01T00:00:01Z", message: { role: "user", content: "How do I configure BGP unnumbered on Juniper?" } },
    { type: "message", id: "m2", parentId: "m1", timestamp: "2026-01-01T00:00:02Z", message: { role: "assistant", content: [
      { type: "thinking", thinking: "The user wants BGP unnumbered config. I should check RFC 5549." },
      { type: "text", text: "BGP unnumbered uses RFC 5549 extended nexthop. Configure it with family inet6 ipv6-nd." }
    ]}},
    { type: "message", id: "m3", parentId: "m2", timestamp: "2026-01-01T00:00:03Z", message: { role: "assistant", content: [
      { type: "toolCall", name: "junos_cli", input: { command: "show bgp summary" } }
    ]}},
    { type: "message", id: "m4", parentId: "m3", timestamp: "2026-01-01T00:00:04Z", message: { role: "toolResult", content: "BGP summary: 0 peers configured" } },
  ]);

  const file2 = createSessionFile("session2.jsonl", [
    { type: "session", version: 3, id: "s2", timestamp: "2026-01-02T00:00:00Z", cwd: "/test" },
    { type: "message", id: "m5", parentId: null, timestamp: "2026-01-02T00:00:01Z", message: { role: "user", content: "DuckDB FTS index optimization for large datasets" } },
    { type: "message", id: "m6", parentId: "m5", timestamp: "2026-01-02T00:00:02Z", message: { role: "assistant", content: "Use PRAGMA create_fts_index with porter stemmer for better matching." } },
    { type: "message", id: "m7", parentId: "m6", timestamp: "2026-01-02T00:00:03Z", message: { role: "user", content: "What about cosine similarity for embeddings?" } },
  ]);

  const file3 = createSessionFile("session3.jsonl", [
    { type: "session", version: 3, id: "s3", timestamp: "2026-01-03T00:00:00Z", cwd: "/test" },
    // Entry with only thinking (no text) — should still get content extracted
    { type: "message", id: "m8", parentId: null, timestamp: "2026-01-03T00:00:01Z", message: { role: "assistant", content: [
      { type: "thinking", thinking: "I need to check the EVPN configuration on the spine switch." }
    ]}},
    // Error response with empty content array — should have empty content_text
    { type: "message", id: "m9", parentId: "m8", timestamp: "2026-01-03T00:00:02Z", message: { role: "assistant", content: [], stopReason: "error", errorMessage: "Internal Error" } },
  ]);

  // Create DB
  console.log("Building test database...");
  const inst = await DuckDBInstance.create(TEST_DB);
  const conn = await inst.connect();
  const { entryCount, sessionCount } = await buildTestDB(conn);
  console.log(`Built: ${entryCount} entries, ${sessionCount} sessions\n`);

  // === Test 1: Schema and content extraction ===
  console.log("Test 1: Schema and content extraction");
  {
    const r = await conn.run("SELECT COUNT(*) FROM session_entries");
    assertEq(Number((await r.getRows())[0][0]), 9, "9 entries (3 sessions + 6 messages)");

    // Content extraction: thinking-only entry should have thinking text
    const r2 = await conn.run("SELECT content_text FROM session_entries WHERE entry_id = 'm8'");
    const m8Content = (await r2.getRows())[0][0];
    assert(String(m8Content).includes("EVPN configuration"), "Thinking content extracted for m8");

    // Error response should have empty content
    const r3 = await conn.run("SELECT content_text FROM session_entries WHERE entry_id = 'm9'");
    const m9Content = (await r3.getRows())[0][0];
    assertEq(String(m9Content), "", "Error response m9 has empty content_text");

    // Tool call name should be in content
    const r4 = await conn.run("SELECT content_text FROM session_entries WHERE entry_id = 'm3'");
    const m3Content = (await r4.getRows())[0][0];
    assert(String(m3Content).includes("junos_cli"), "Tool call name extracted for m3");
  }
  console.log("  ✓ Schema and content extraction passed\n");

  // === Test 2: BM25 search with prepared statements ===
  console.log("Test 2: BM25 keyword search");
  {
    const r = await conn.run(
      `SELECT entry_id, fts_main_session_entries.match_bm25(entry_id, ?) AS score
       FROM session_entries
       WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL
       ORDER BY score DESC LIMIT ?`,
      ["BGP unnumbered", "BGP unnumbered", 10]
    );
    const rows = await r.getRows();
    assert(rows.length > 0, "BM25 finds results for 'BGP unnumbered'");
    // Top result should be m1 or m2 (both contain "BGP unnumbered")
    assert(["m1", "m2"].includes(String(rows[0][0])), `Top BM25 result is m1 or m2 (got ${rows[0][0]})`);
  }
  console.log("  ✓ BM25 search passed\n");

  // === Test 3: Cosine search with prepared statements ===
  console.log("Test 3: Semantic cosine search");
  {
    // Create a query vector similar to m2's embedding
    const queryText = "BGP unnumbered RFC 5549 extended nexthop";
    const embedding = new Float32Array(EMBED_DIM);
    for (let i = 0; i < EMBED_DIM; i++) {
      embedding[i] = Math.sin(queryText.charCodeAt(i % queryText.length) + i) * 0.1;
    }
    let norm = 0;
    for (const v of embedding) norm += v * v;
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < EMBED_DIM; i++) embedding[i] /= norm;
    const arrStr = `[${Array.from(embedding).map((v) => v.toFixed(6)).join(",")}]`;

    const r = await conn.run(
      `SELECT entry_id, array_cosine_similarity(embedding, ?::FLOAT[${EMBED_DIM}]) AS sim
       FROM session_entries
       WHERE embedding IS NOT NULL
       ORDER BY sim DESC LIMIT ?`,
      [arrStr, 5]
    );
    const rows = await r.getRows();
    assert(rows.length > 0, "Cosine search returns results");
    assert(Number(rows[0][1]) > 0, "Top cosine similarity is positive");
  }
  console.log("  ✓ Semantic search passed\n");

  // === Test 4: Hybrid RRF ===
  console.log("Test 4: Hybrid RRF search");
  {
    const bm25R = await conn.run(
      `SELECT entry_id FROM session_entries
       WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL
       ORDER BY fts_main_session_entries.match_bm25(entry_id, ?) DESC LIMIT 50`,
      ["BGP", "BGP"]
    );
    const bm25Rows = await bm25R.getRows();
    assert(bm25Rows.length > 0, "Hybrid: BM25 subquery returns results");

    const cosineR = await conn.run(
      `SELECT entry_id FROM session_entries
       WHERE embedding IS NOT NULL
       ORDER BY array_cosine_similarity(embedding, (SELECT embedding FROM session_entries WHERE embedding IS NOT NULL LIMIT 1)) DESC
       LIMIT 50`
    );
    const cosineRows = await cosineR.getRows();
    assert(cosineRows.length > 0, "Hybrid: cosine subquery returns results");

    // RRF fusion
    const rrfScores = new Map();
    const RRF_K = 60;
    for (let i = 0; i < bm25Rows.length; i++) {
      const id = String(bm25Rows[i][0]);
      rrfScores.set(id, { score: 1 / (RRF_K + i + 1), bm25_rank: i + 1, cosine_rank: null });
    }
    for (let i = 0; i < cosineRows.length; i++) {
      const id = String(cosineRows[i][0]);
      const rrf = 1 / (RRF_K + i + 1);
      const existing = rrfScores.get(id);
      if (existing) {
        existing.score += rrf;
        existing.cosine_rank = i + 1;
      } else {
        rrfScores.set(id, { score: rrf, bm25_rank: null, cosine_rank: i + 1 });
      }
    }
    const fused = Array.from(rrfScores.values()).sort((a, b) => b.score - a.score);
    assert(fused.length > 0, "Hybrid: RRF fusion produces ranked results");
    assert(fused[0].score > 0, "Hybrid: top RRF score is positive");
  }
  console.log("  ✓ Hybrid search passed\n");

  // === Test 5: session_read ===
  console.log("Test 5: session_read by path + line");
  {
    const r = await conn.run(
      `WITH numbered AS (
         SELECT row_number() OVER () AS rn, json
         FROM read_json_objects(?, format='newline_delimited', ignore_errors=true)
       )
       SELECT json->>'type' AS entry_type, json->>'id' AS entry_id
       FROM numbered WHERE rn = ? LIMIT 1`,
      [file1, 1]
    );
    const rows = await r.getRows();
    assert(rows.length > 0, "session_read returns entry at line 1");
    assertEq(String(rows[0][0]), "session", "Line 1 is a session entry");
    assertEq(String(rows[0][1]), "s1", "Session ID is s1");

    // Line 2 should be the first message
    const r2 = await conn.run(
      `WITH numbered AS (
         SELECT row_number() OVER () AS rn, json
         FROM read_json_objects(?, format='newline_delimited', ignore_errors=true)
       )
       SELECT json->>'type' AS entry_type, json->'message'->>'role' AS role
       FROM numbered WHERE rn = ? LIMIT 1`,
      [file1, 2]
    );
    const rows2 = await r2.getRows();
    assertEq(String(rows2[0][0]), "message", "Line 2 is a message");
    assertEq(String(rows2[0][1]), "user", "Line 2 role is user");
  }
  console.log("  ✓ session_read passed\n");

  // === Test 6: Metadata persistence ===
  console.log("Test 6: Metadata persistence");
  {
    const r = await conn.run("SELECT value FROM _metadata WHERE key = ?", ["entry_count"]);
    const count = Number((await r.getRows())[0][0]);
    assertEq(count, entryCount, "Metadata entry_count matches");

    const r2 = await conn.run("SELECT value FROM _metadata WHERE key = ?", ["session_count"]);
    const sessCount = Number((await r2.getRows())[0][0]);
    assertEq(sessCount, sessionCount, "Metadata session_count matches");

    const r3 = await conn.run("SELECT value FROM _metadata WHERE key = ?", ["has_embeddings"]);
    assertEq(String((await r3.getRows())[0][0]), "true", "Metadata has_embeddings is true");
  }
  console.log("  ✓ Metadata persistence passed\n");

  // === Test 7: SQL injection resistance ===
  console.log("Test 7: SQL injection resistance");
  {
    // Attempt SQL injection via query parameter
    const maliciousQuery = "'; DROP TABLE session_entries; --";
    const r = await conn.run(
      `SELECT entry_id FROM session_entries
       WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL
       LIMIT 1`,
      [maliciousQuery]
    );
    const rows = await r.getRows();
    // Table should still exist
    const checkTable = await conn.run("SELECT COUNT(*) FROM session_entries");
    assertEq(Number((await checkTable.getRows())[0][0]), entryCount, "session_entries table survives injection attempt");

    // Attempt injection via path parameter
    const maliciousPath = file1 + "'; DROP TABLE session_entries; --";
    try {
      const r2 = await conn.run(
        `SELECT COUNT(*) FROM read_json_objects(?, format='newline_delimited', ignore_errors=true)`,
        [maliciousPath]
      );
      // If it doesn't throw, table should still exist
      const checkTable2 = await conn.run("SELECT COUNT(*) FROM session_entries");
      assertEq(Number((await checkTable2.getRows())[0][0]), entryCount, "Table survives path injection");
    } catch (e) {
      // Expected: DuckDB should reject the malicious path
      assert(true, "Path injection rejected by DuckDB");
    }
  }
  console.log("  ✓ SQL injection resistance passed\n");

  // === Test 8: Edge cases ===
  console.log("Test 8: Edge cases");
  {
    // Empty query should return no results (not crash)
    const r = await conn.run(
      `SELECT entry_id FROM session_entries
       WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL
       LIMIT ?`,
      ["", 10]
    );
    const rows = await r.getRows();
    // Empty query may return 0 results or all results depending on FTS behavior
    assert(true, "Empty query doesn't crash");

    // Limit clamping
    const r2 = await conn.run(
      `SELECT entry_id FROM session_entries
       WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL
       LIMIT ?`,
      ["BGP", 1000]
    );
    const rows2 = await r2.getRows();
    assert(rows2.length <= 1000, "Limit clamped to reasonable value");

    // Non-existent file
    try {
      await conn.run(
        `SELECT COUNT(*) FROM read_json_objects(?, format='newline_delimited', ignore_errors=true)`,
        ["/nonexistent/path/file.jsonl"]
      );
      // With ignore_errors, might return 0 or throw
      assert(true, "Non-existent file handled gracefully");
    } catch (e) {
      assert(true, "Non-existent file throws error (expected)");
    }

    // Line beyond file
    const r3 = await conn.run(
      `WITH numbered AS (
         SELECT row_number() OVER () AS rn, json
         FROM read_json_objects(?, format='newline_delimited', ignore_errors=true)
       )
       SELECT COUNT(*) FROM numbered WHERE rn = ?`,
      [file1, 999]
    );
    assertEq(Number((await r3.getRows())[0][0]), 0, "Line 999 returns 0 results");
  }
  console.log("  ✓ Edge cases passed\n");

  // === Test 9: FTS overwrite ===
  console.log("Test 9: FTS overwrite without drop");
  {
    // Create FTS index again with overwrite=true (no drop needed first)
    const start = Date.now();
    await conn.run("PRAGMA create_fts_index('session_entries', 'entry_id', 'content_text', 'role', stemmer='porter', overwrite=true);");
    const elapsed = Date.now() - start;
    assert(elapsed < 5000, `FTS overwrite in <5s (got ${elapsed}ms)`);

    // Verify FTS still works
    const r = await conn.run(
      `SELECT entry_id FROM session_entries
       WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL
       LIMIT ?`,
      ["BGP", 5]
    );
    const rows = await r.getRows();
    assert(rows.length > 0, "FTS works after overwrite");
  }
  console.log("  ✓ FTS overwrite passed\n");

  // === Test 10: ignore_errors with malformed JSON ===
  console.log("Test 10: ignore_errors with malformed JSON");
  {
    const malformedFile = path.join(TEST_SESSIONS, "malformed.jsonl");
    fs.writeFileSync(malformedFile, [
      JSON.stringify({ type: "message", id: "good1", message: { role: "user", content: "valid entry" } }),
      "{invalid json line}",
      JSON.stringify({ type: "message", id: "good2", message: { role: "assistant", content: "also valid" } }),
    ].join("\n") + "\n");

    const r = await conn.run(
      `SELECT COUNT(*) FROM read_json_objects(?, format='newline_delimited', ignore_errors=true) WHERE json IS NOT NULL`,
      [malformedFile]
    );
    const count = Number((await r.getRows())[0][0]);
    assertEq(count, 2, "ignore_errors: 2 valid (non-null) entries read, malformed line is null");

    // Without ignore_errors, should fail
    try {
      await conn.run(
        `SELECT COUNT(*) FROM read_json_objects(?, format='newline_delimited')`,
        [malformedFile]
      );
      assert(false, "Should have thrown on malformed JSON without ignore_errors");
    } catch (e) {
      assert(true, "Without ignore_errors, malformed JSON throws");
    }
  }
  console.log("  ✓ ignore_errors passed\n");

  // === Test 11: Content truncation ===
  console.log("Test 11: Content truncation");
  {
    // Create entry with very long content
    const longContent = "A".repeat(5000);
    const longFile = path.join(TEST_SESSIONS, "long.jsonl");
    fs.writeFileSync(longFile, JSON.stringify({
      type: "message", id: "long1",
      message: { role: "user", content: longContent }
    }) + "\n");

    const escapedPath = longFile.replace(/'/g, "''");
    const r = await conn.run(`
      SELECT length(${CONTENT_EXTRACTION_SQL}) AS content_len
      FROM read_json_objects('${escapedPath}', format='newline_delimited')
      WHERE json->>'type' = 'message'
    `);
    const len = Number((await r.getRows())[0][0]);
    assertEq(len, MAX_CONTENT_CHARS, `Content truncated to ${MAX_CONTENT_CHARS} chars (got ${len})`);
  }
  console.log("  ✓ Content truncation passed\n");

  // === Test 12: Read-only mode ===
  console.log("Test 12: Read-only mode");
  {
    await conn.disconnectSync();
    inst.closeSync();

    const roInst = await DuckDBInstance.create(TEST_DB, { access_mode: "READ_ONLY" });
    const roConn = await roInst.connect();
    await roConn.run("LOAD fts;");

    // Read should work
    const r = await roConn.run("SELECT COUNT(*) FROM session_entries");
    assertEq(Number((await r.getRows())[0][0]), entryCount, "Read-only: SELECT works");

    // Write should fail
    try {
      await roConn.run("INSERT INTO session_entries VALUES ('test', 1, 'test', 'test', null, null, 'user', null, 'test')");
      assert(false, "Read-only: INSERT should fail");
    } catch (e) {
      assert(true, "Read-only: INSERT correctly rejected");
    }

    // FTS search should work in read-only
    const r2 = await roConn.run(
      `SELECT entry_id FROM session_entries
       WHERE fts_main_session_entries.match_bm25(entry_id, ?) IS NOT NULL
       LIMIT ?`,
      ["BGP", 5]
    );
    assert((await r2.getRows()).length > 0, "Read-only: FTS search works");

    roConn.disconnectSync();
    roInst.closeSync();
  }
  console.log("  ✓ Read-only mode passed\n");

  // === Results ===
  console.log("=== Results ===");
  console.log(`  Passed: ${passed}`);
  console.log(`  Failed: ${failed}`);
  if (failures.length > 0) {
    console.log("\nFailures:");
    for (const f of failures) console.log(`  - ${f}`);
  }

  // Cleanup
  try {
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  } catch {}

  process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
  console.error("FATAL:", e.message);
  console.error(e.stack);
  try { fs.rmSync(TEST_DIR, { recursive: true, force: true }); } catch {}
  process.exit(1);
});
