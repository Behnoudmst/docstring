import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { EmbeddingMismatchError, type Chunk } from "@docstring/core";
import { SqliteChunkStore, escapeFts, toBlob } from "./sqlite-store.js";

const DIM = 4;
const dirs: string[] = [];

function dbPath(name = "test.db"): string {
  const dir = mkdtempSync(join(tmpdir(), "docstring-"));
  dirs.push(dir);
  return join(dir, name);
}

afterEach(() => {
  while (dirs.length > 0) rmSync(dirs.pop()!, { recursive: true, force: true });
});

function chunk(over: Partial<Chunk> = {}): Chunk {
  return {
    id: "c1",
    repo: "elegant-menu-front",
    path: "src/lib/getSubDomain.ts",
    language: "typescript",
    symbolName: "getSubDomain",
    symbolKind: "function",
    startLine: 3,
    endLine: 7,
    content: "export function getSubDomain(host: string) { return host.split('.')[0]; }",
    embeddingText: "File: src/lib/getSubDomain.ts\nSymbol: getSubDomain (function)\n---\ncode",
    imports: ["next/headers"],
    gitSha: "a".repeat(40),
    embeddingModel: "nomic-embed-text",
    embeddingDim: DIM,
    ...over,
  };
}

const open = (path: string, model = "nomic-embed-text", dim = DIM) =>
  new SqliteChunkStore({ path, embeddingModel: model, embeddingDim: dim });

describe("escapeFts", () => {
  it("quotes identifiers so FTS5 operators do not blow up", () => {
    expect(escapeFts("getSubDomain")).toBe('"getSubDomain"');
    expect(escapeFts("where is the JWT signed?")).toContain('"JWT"');
  });

  it("survives punctuation-only input", () => {
    expect(() => escapeFts("*(){}")).not.toThrow();
  });
});

describe("toBlob", () => {
  it("packs floats little-endian at 4 bytes each", () => {
    expect(toBlob([1, 0, 0, 0]).byteLength).toBe(16);
  });
});

describe("SqliteChunkStore", () => {
  it("round-trips a chunk with all metadata intact", async () => {
    const s = open(dbPath());
    const c = chunk();
    await s.upsertChunks([c], [[1, 0, 0, 0]]);
    const [back] = await s.getByIds(["c1"]);
    expect(back).toEqual(c);
    await s.close();
  });

  it("ranks by vector distance", async () => {
    const s = open(dbPath());
    await s.upsertChunks(
      [chunk({ id: "a", path: "src/a.ts" }), chunk({ id: "b", path: "src/b.ts" })],
      [[1, 0, 0, 0], [0, 1, 0, 0]],
    );
    const hits = await s.vectorSearch([0.95, 0.05, 0, 0], 2);
    expect(hits[0]?.chunk.path).toBe("src/a.ts");
    expect(hits[0]!.score).toBeGreaterThan(hits[1]!.score);
    await s.close();
  });

  it("finds exact identifiers via full-text search", async () => {
    const s = open(dbPath());
    await s.upsertChunks(
      [
        chunk({ id: "a", symbolName: "getSubDomain" }),
        chunk({ id: "b", path: "src/other.ts", symbolName: "unrelated", content: "nothing here" }),
      ],
      [[1, 0, 0, 0], [0, 1, 0, 0]],
    );
    const hits = await s.keywordSearch("getSubDomain", 5);
    expect(hits[0]?.chunk.symbolName).toBe("getSubDomain");
    await s.close();
  });

  it("applies filters to both search paths", async () => {
    const s = open(dbPath());
    await s.upsertChunks(
      [
        chunk({ id: "a", path: "src/lib/a.ts" }),
        chunk({ id: "b", path: "src/components/b.tsx" }),
      ],
      [[1, 0, 0, 0], [1, 0, 0, 0]],
    );
    const hits = await s.vectorSearch([1, 0, 0, 0], 10, { pathPrefix: "src/lib/" });
    expect(hits.map((h) => h.chunk.path)).toEqual(["src/lib/a.ts"]);
    await s.close();
  });

  it("deletes every chunk of a file from all three tables", async () => {
    const s = open(dbPath());
    await s.upsertChunks(
      [chunk({ id: "a" }), chunk({ id: "b", startLine: 20 })],
      [[1, 0, 0, 0], [0, 1, 0, 0]],
    );
    expect(await s.deleteByFile("elegant-menu-front", "src/lib/getSubDomain.ts")).toBe(2);
    expect(s.stats()).toEqual({ chunks: 0, files: 0, vectors: 0 });
    expect(await s.vectorSearch([1, 0, 0, 0], 5)).toEqual([]);
    await s.close();
  });

  it("upserting the same id twice does not duplicate vectors", async () => {
    const s = open(dbPath());
    await s.upsertChunks([chunk()], [[1, 0, 0, 0]]);
    await s.upsertChunks([chunk({ content: "changed" })], [[0, 1, 0, 0]]);
    expect(s.stats()).toEqual({ chunks: 1, files: 1, vectors: 1 });
    await s.close();
  });

  it("persists across reopen", async () => {
    const path = dbPath();
    const s1 = open(path);
    await s1.upsertChunks([chunk()], [[1, 0, 0, 0]]);
    await s1.close();

    const s2 = open(path);
    expect(s2.stats().chunks).toBe(1);
    expect((await s2.vectorSearch([1, 0, 0, 0], 1))[0]?.chunk.id).toBe("c1");
    await s2.close();
  });

  it("refuses to reopen an index with a different embedding model", async () => {
    const path = dbPath();
    const s1 = open(path);
    await s1.upsertChunks([chunk()], [[1, 0, 0, 0]]);
    await s1.close();
    expect(() => open(path, "text-embedding-3-small", DIM)).toThrow(EmbeddingMismatchError);
  });

  it("refuses a query vector of the wrong dimension", async () => {
    const s = open(dbPath());
    await s.upsertChunks([chunk()], [[1, 0, 0, 0]]);
    await expect(s.vectorSearch([1, 0], 5)).rejects.toBeInstanceOf(EmbeddingMismatchError);
    await s.close();
  });

  it("tracks file hashes for incremental reindexing", async () => {
    const s = open(dbPath());
    s.recordFile("elegant-menu-front", "src/a.ts", "hash1");
    expect(s.fileHashes("elegant-menu-front").get("src/a.ts")).toBe("hash1");
    s.recordFile("elegant-menu-front", "src/a.ts", "hash2");
    expect(s.fileHashes("elegant-menu-front").get("src/a.ts")).toBe("hash2");
    await s.close();
  });
});

describe("first run", () => {
  it("creates the parent directory rather than throwing", async () => {
    const dir = mkdtempSync(join(tmpdir(), "docstring-"));
    dirs.push(dir);
    const nested = join(dir, ".docstring", "index.db");
    const s = open(nested);
    await s.upsertChunks([chunk()], [[1, 0, 0, 0]]);
    expect(s.stats().chunks).toBe(1);
    await s.close();
  });
});

describe("prefix compatibility", () => {
  it("refuses to query an index built with a different document prefix", async () => {
    const path = dbPath();
    const s1 = new SqliteChunkStore({
      path, embeddingModel: "nomic-embed-text", embeddingDim: DIM, embeddingPrefix: "",
    });
    await s1.upsertChunks([chunk()], [[1, 0, 0, 0]]);
    await s1.close();

    expect(() => new SqliteChunkStore({
      path, embeddingModel: "nomic-embed-text", embeddingDim: DIM,
      embeddingPrefix: "search_document: ",
    })).toThrow(/Delete the index file and re-index/);
  });
});

describe("keyword scores", () => {
  it("scores better matches higher, not lower", async () => {
    const s = open(dbPath());
    await s.upsertChunks(
      [
        chunk({ id: "strong", path: "src/lib/getSubDomain.ts", symbolName: "getSubDomain",
                content: "getSubDomain getSubDomain host subdomain routing" }),
        chunk({ id: "weak", path: "src/other.ts", symbolName: "other",
                content: "unrelated code about pricing" }),
      ],
      [[1, 0, 0, 0], [0, 1, 0, 0]],
    );
    const hits = await s.keywordSearch("getSubDomain subdomain routing", 5);
    expect(hits[0]?.chunk.id).toBe("strong");
    // The ordering and the scores must agree.
    for (let i = 1; i < hits.length; i++) {
      expect(hits[i]!.score).toBeLessThanOrEqual(hits[i - 1]!.score);
    }
    await s.close();
  });

  it("keeps scores within (0, 1]", async () => {
    const s = open(dbPath());
    await s.upsertChunks([chunk({ content: "routing routing routing" })], [[1, 0, 0, 0]]);
    for (const h of await s.keywordSearch("routing", 5)) {
      expect(h.score).toBeGreaterThan(0);
      expect(h.score).toBeLessThanOrEqual(1);
    }
    await s.close();
  });
});

describe("hasPathPrefix", () => {
  it("distinguishes a prefix that covers indexed files from one that covers none", async () => {
    const store = open(dbPath());
    await store.upsertChunks([chunk()]);
    expect(store.hasPathPrefix("src/")).toBe(true);
    expect(store.hasPathPrefix("src/lib/")).toBe(true);
    expect(store.hasPathPrefix("app/")).toBe(false);
    await store.close();
  });

  it("treats LIKE wildcards as literal characters", async () => {
    const store = open(dbPath());
    await store.upsertChunks([chunk()]);
    // Unescaped, "%" and "_" would match anything and claim coverage the
    // index does not have.
    expect(store.hasPathPrefix("%")).toBe(false);
    expect(store.hasPathPrefix("_rc/")).toBe(false);
    await store.close();
  });
});
