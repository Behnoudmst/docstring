import { describe, expect, it } from "vitest";
import { EmbeddingMismatchError, type Chunk } from "@docstring/core";
import { MemoryChunkStore, cosine } from "./memory-store.js";

function chunk(path: string, content: string, over: Partial<Chunk> = {}): Chunk {
  return {
    id: `${path}#0`,
    repo: "elegant-menu-front",
    path,
    language: "typescript",
    symbolName: null,
    symbolKind: "file",
    startLine: 1,
    endLine: 20,
    content,
    embeddingText: `File: ${path}\n---\n${content}`,
    imports: [],
    gitSha: "a".repeat(40),
    embeddingModel: "nomic-embed-text",
    embeddingDim: 3,
    ...over,
  };
}

describe("cosine", () => {
  it("is 1 for identical vectors and 0 for orthogonal ones", () => {
    expect(cosine([1, 0, 0], [1, 0, 0])).toBeCloseTo(1);
    expect(cosine([1, 0, 0], [0, 1, 0])).toBeCloseTo(0);
  });

  it("throws on dimension mismatch rather than returning nonsense", () => {
    expect(() => cosine([1, 0], [1, 0, 0])).toThrow(/dimension mismatch/);
  });
});

describe("MemoryChunkStore", () => {
  it("round-trips a chunk", async () => {
    const s = new MemoryChunkStore();
    const c = chunk("src/middleware.ts", "export function middleware() {}");
    await s.upsertChunks([c], [[1, 0, 0]]);
    expect(await s.getByIds([c.id])).toEqual([c]);
  });

  it("ranks by vector similarity", async () => {
    const s = new MemoryChunkStore();
    await s.upsertChunks(
      [chunk("src/a.ts", "a"), chunk("src/b.ts", "b")],
      [[1, 0, 0], [0, 1, 0]],
    );
    const hits = await s.vectorSearch([0.9, 0.1, 0], 2);
    expect(hits[0]?.chunk.path).toBe("src/a.ts");
  });

  it("finds exact identifiers by keyword that vectors would miss", async () => {
    const s = new MemoryChunkStore();
    await s.upsertChunks(
      [chunk("src/lib/getSubDomain.ts", "export function getSubDomain(host: string) {}")],
      [[0, 0, 1]],
    );
    const hits = await s.keywordSearch("getSubDomain", 5);
    expect(hits[0]?.chunk.path).toBe("src/lib/getSubDomain.ts");
  });

  it("applies path prefix filters", async () => {
    const s = new MemoryChunkStore();
    await s.upsertChunks(
      [chunk("src/lib/a.ts", "a"), chunk("src/components/b.tsx", "b")],
      [[1, 0, 0], [1, 0, 0]],
    );
    const hits = await s.vectorSearch([1, 0, 0], 10, { pathPrefix: "src/lib/" });
    expect(hits.map((h) => h.chunk.path)).toEqual(["src/lib/a.ts"]);
  });

  it("deletes every chunk of a file, for incremental reindexing", async () => {
    const s = new MemoryChunkStore();
    await s.upsertChunks(
      [chunk("src/a.ts", "one"), chunk("src/a.ts", "two", { id: "src/a.ts#1" })],
      [[1, 0, 0], [0, 1, 0]],
    );
    expect(await s.deleteByFile("elegant-menu-front", "src/a.ts")).toBe(2);
    expect(s.size).toBe(0);
  });

  it("refuses to mix embedding models in one index", async () => {
    const s = new MemoryChunkStore();
    await s.upsertChunks([chunk("src/a.ts", "a")], [[1, 0, 0]]);
    await expect(
      s.upsertChunks([chunk("src/b.ts", "b", { embeddingModel: "text-embedding-3-small" })], [[1, 0, 0]]),
    ).rejects.toBeInstanceOf(EmbeddingMismatchError);
  });

  it("refuses a query vector of the wrong dimension", async () => {
    const s = new MemoryChunkStore();
    await s.upsertChunks([chunk("src/a.ts", "a")], [[1, 0, 0]]);
    await expect(s.vectorSearch([1, 0], 5)).rejects.toBeInstanceOf(EmbeddingMismatchError);
  });
});
