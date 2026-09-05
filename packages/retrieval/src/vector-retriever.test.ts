import { describe, expect, it } from "vitest";
import type { Chunk, ChunkStore, Embedder, ScoredChunk } from "@docstring/core";
import { KeywordRetriever, VectorRetriever } from "./vector-retriever.js";

const chunk = (path: string): Chunk => ({
  id: path, repo: "r", path, language: "typescript",
  symbolName: null, symbolKind: "function", startLine: 1, endLine: 5,
  content: "code", embeddingText: "text", imports: [],
  gitSha: "a".repeat(40), embeddingModel: "m", embeddingDim: 3,
});

const fakeEmbedder: Embedder = {
  model: "m",
  dim: 3,
  prefixes: { query: "Q: ", document: "D: " },
  async embed(texts) {
    return texts.map((t) => [t.length, 0, 0]);
  },
  async embedDocuments(texts) {
    return this.embed(texts.map((t) => this.prefixes.document + t));
  },
  async embedQuery(text) {
    const [v] = await this.embed([this.prefixes.query + text]);
    return v ?? [];
  },
};

function fakeStore(): ChunkStore & { lastVector?: number[]; lastK?: number } {
  const s: any = {
    async vectorSearch(embedding: number[], k: number): Promise<ScoredChunk[]> {
      s.lastVector = embedding;
      s.lastK = k;
      return [{ chunk: chunk("src/a.ts"), score: 0.9, source: "vector" }];
    },
    async keywordSearch(): Promise<ScoredChunk[]> {
      return [{ chunk: chunk("src/b.ts"), score: 0.5, source: "keyword" }];
    },
    async upsertChunks() {}, async deleteByFile() { return 0; },
    async getByIds() { return []; }, async indexInfo() { return null; }, async close() {},
  };
  return s;
}

describe("VectorRetriever", () => {
  it("embeds the query and passes the vector through to the store", async () => {
    const store = fakeStore();
    const r = new VectorRetriever(fakeEmbedder, store);
    const hits = await r.retrieve("where is the middleware", 10);
    // The query prefix must be applied before embedding, not after.
    expect(store.lastVector).toEqual(["Q: where is the middleware".length, 0, 0]);
    expect(store.lastK).toBe(10);
    expect(hits[0]?.chunk.path).toBe("src/a.ts");
  });

  it("returns nothing rather than throwing when embedding yields nothing", async () => {
    const empty: Embedder = {
      model: "m", dim: 3, prefixes: { query: "", document: "" },
      async embed() { return []; },
      async embedDocuments() { return []; },
      async embedQuery() { return []; },
    };
    const r = new VectorRetriever(empty, fakeStore());
    expect(await r.retrieve("q", 5)).toEqual([]);
  });
});

describe("KeywordRetriever", () => {
  it("goes straight to full-text search with no embedding call", async () => {
    const r = new KeywordRetriever(fakeStore());
    const hits = await r.retrieve("getSubDomain", 5);
    expect(hits[0]?.source).toBe("keyword");
  });
});
