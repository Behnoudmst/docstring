import { describe, expect, it } from "vitest";
import type { Chunk, Retriever, ScoredChunk } from "@docstring/core";
import { HybridRetriever, fuseRrf } from "./hybrid-retriever.js";

const chunk = (id: string): Chunk => ({
  id, repo: "r", path: `src/${id}.ts`, language: "typescript",
  symbolName: id, symbolKind: "function", startLine: 1, endLine: 5,
  content: "code", embeddingText: "text", imports: [],
  gitSha: "a".repeat(40), embeddingModel: "m", embeddingDim: 3,
});

const ranking = (ids: string[], source: ScoredChunk["source"]): ScoredChunk[] =>
  ids.map((id, i) => ({ chunk: chunk(id), score: 1 / (i + 1), source }));

const stub = (name: string, ids: string[], source: ScoredChunk["source"]): Retriever => ({
  name,
  async retrieve(_q, k) {
    return ranking(ids, source).slice(0, k);
  },
});

describe("fuseRrf", () => {
  it("ranks a document found by both sources above one found by either alone", () => {
    const fused = fuseRrf(
      [
        { results: ranking(["a", "b", "c"], "vector"), weight: 1 },
        { results: ranking(["c", "d", "e"], "keyword"), weight: 1 },
      ],
      60,
    );
    expect(fused[0]?.chunk.id).toBe("c");
  });

  it("preserves order within a single ranking", () => {
    const fused = fuseRrf([{ results: ranking(["a", "b", "c"], "vector"), weight: 1 }], 60);
    expect(fused.map((f) => f.chunk.id)).toEqual(["a", "b", "c"]);
  });

  it("deduplicates chunks appearing in both rankings", () => {
    const fused = fuseRrf(
      [
        { results: ranking(["a", "b"], "vector"), weight: 1 },
        { results: ranking(["a", "b"], "keyword"), weight: 1 },
      ],
      60,
    );
    expect(fused).toHaveLength(2);
  });

  it("weights let one source dominate", () => {
    const heavyKeyword = fuseRrf(
      [
        { results: ranking(["a"], "vector"), weight: 0.1 },
        { results: ranking(["b"], "keyword"), weight: 10 },
      ],
      60,
    );
    expect(heavyKeyword[0]?.chunk.id).toBe("b");
  });

  it("uses rank, not score, so incomparable scales do not matter", () => {
    // BM25 values are unbounded and negative; cosine is 0..1. Fusion must ignore both.
    const wild: ScoredChunk[] = [
      { chunk: chunk("x"), score: -9999, source: "keyword" },
      { chunk: chunk("y"), score: -1, source: "keyword" },
    ];
    const fused = fuseRrf([{ results: wild, weight: 1 }], 60);
    expect(fused.map((f) => f.chunk.id)).toEqual(["x", "y"]);
  });

  it("returns nothing when every source is empty", () => {
    expect(fuseRrf([{ results: [], weight: 1 }], 60)).toEqual([]);
  });
});

describe("HybridRetriever", () => {
  it("surfaces results neither source ranked first alone", async () => {
    const r = new HybridRetriever(
      stub("v", ["a", "shared", "b"], "vector"),
      stub("kw", ["c", "shared", "d"], "keyword"),
    );
    const hits = await r.retrieve("q", 3);
    expect(hits[0]?.chunk.id).toBe("shared");
    expect(hits[0]?.source).toBe("fused");
  });

  it("recovers an exact identifier that vector search ranked low", async () => {
    const vector = stub("v", ["noise1", "noise2", "noise3", "getSubDomain"], "vector");
    const keyword = stub("kw", ["getSubDomain"], "keyword");
    const hits = await new HybridRetriever(vector, keyword).retrieve("getSubDomain", 2);
    expect(hits[0]?.chunk.id).toBe("getSubDomain");
  });

  it("still works when one source returns nothing", async () => {
    const hits = await new HybridRetriever(
      stub("v", ["a", "b"], "vector"),
      stub("kw", [], "keyword"),
    ).retrieve("q", 5);
    expect(hits.map((h) => h.chunk.id)).toEqual(["a", "b"]);
  });

  it("fuses over the candidate depth, not the requested k", async () => {
    const deep = Array.from({ length: 50 }, (_, i) => `v${i}`);
    const r = new HybridRetriever(
      stub("v", deep, "vector"),
      stub("kw", ["v49"], "keyword"),
      { candidates: 50 },
    );
    // v49 is last in the vector ranking but first in keyword — fusion should lift it.
    const hits = await r.retrieve("q", 5);
    expect(hits.map((h) => h.chunk.id)).toContain("v49");
  });
});
