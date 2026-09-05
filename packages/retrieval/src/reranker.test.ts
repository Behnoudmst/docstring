import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Chunk, Reranker, Retriever, ScoredChunk } from "@docstring/core";
import { OllamaReranker, RerankingRetriever, parseScores } from "./reranker.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

const chunk = (id: string): Chunk => ({
  id, repo: "r", path: `src/${id}.ts`, language: "typescript",
  symbolName: id, symbolKind: "function", startLine: 1, endLine: 5,
  content: `export function ${id}() {}`, embeddingText: "t", imports: [],
  gitSha: "a".repeat(40), embeddingModel: "m", embeddingDim: 3,
});

const pool = (ids: string[]): ScoredChunk[] =>
  ids.map((id, i) => ({ chunk: chunk(id), score: 1 / (i + 1), source: "fused" as const }));

const baseRetriever = (ids: string[]): Retriever => ({
  name: "base",
  async retrieve(_q, k) {
    return pool(ids).slice(0, k);
  },
});

function mockGenerate(reply: (prompt: string) => string) {
  globalThis.fetch = vi.fn(async (_url: unknown, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as { prompt: string };
    return { ok: true, json: async () => ({ response: reply(body.prompt) }) } as Response;
  }) as unknown as typeof fetch;
}

describe("RerankingRetriever", () => {
  it("fetches the candidate pool, not just k", async () => {
    let askedFor = 0;
    const base: Retriever = {
      name: "base",
      async retrieve(_q, k) { askedFor = k; return pool(["a", "b", "c"]); },
    };
    const identity: Reranker = { name: "id", async rerank(_q, c, k) { return c.slice(0, k); } };
    await new RerankingRetriever(base, identity, 50).retrieve("q", 5);
    expect(askedFor).toBe(50);
  });

  it("returns the reranked order, not the base order", async () => {
    const reverse: Reranker = {
      name: "rev",
      async rerank(_q, c, k) { return [...c].reverse().slice(0, k); },
    };
    const hits = await new RerankingRetriever(baseRetriever(["a", "b", "c"]), reverse).retrieve("q", 3);
    expect(hits.map((h) => h.chunk.id)).toEqual(["c", "b", "a"]);
  });

  it("skips the model entirely when there is nothing to reorder", async () => {
    const never: Reranker = {
      name: "never",
      async rerank() { throw new Error("should not be called"); },
    };
    const hits = await new RerankingRetriever(baseRetriever(["only"]), never).retrieve("q", 5);
    expect(hits).toHaveLength(1);
  });
});

describe("OllamaReranker", () => {
  it("orders by model score", async () => {
    mockGenerate(() => JSON.stringify([
      { id: 0, score: 1 }, { id: 1, score: 9 }, { id: 2, score: 5 },
    ]));
    const hits = await new OllamaReranker({ batchSize: 10 }).rerank("q", pool(["a", "b", "c"]), 3);
    expect(hits.map((h) => h.chunk.id)).toEqual(["b", "c", "a"]);
    expect(hits[0]?.source).toBe("reranked");
  });

  it("survives prose wrapped around the JSON", async () => {
    mockGenerate(() => 'Sure! Here you go:\n```json\n[{"id":0,"score":8},{"id":1,"score":2}]\n```');
    const hits = await new OllamaReranker().rerank("q", pool(["a", "b"]), 2);
    expect(hits[0]?.chunk.id).toBe("a");
  });

  it("falls back to base order when the model returns garbage", async () => {
    mockGenerate(() => "I cannot help with that.");
    const hits = await new OllamaReranker().rerank("q", pool(["a", "b", "c"]), 3);
    expect(hits.map((h) => h.chunk.id)).toEqual(["a", "b", "c"]);
  });

  it("falls back to base order when the model is unreachable", async () => {
    globalThis.fetch = vi.fn(async () => { throw new Error("ECONNREFUSED"); }) as unknown as typeof fetch;
    const hits = await new OllamaReranker().rerank("q", pool(["a", "b"]), 2);
    expect(hits.map((h) => h.chunk.id)).toEqual(["a", "b"]);
  });

  it("ignores out-of-range ids instead of corrupting the ranking", async () => {
    mockGenerate(() => JSON.stringify([{ id: 99, score: 10 }, { id: 1, score: 7 }]));
    const hits = await new OllamaReranker().rerank("q", pool(["a", "b"]), 2);
    expect(hits.map((h) => h.chunk.id)).toEqual(["b", "a"]);
  });

  it("batches large candidate pools", async () => {
    let calls = 0;
    mockGenerate(() => {
      calls++;
      return JSON.stringify(Array.from({ length: 10 }, (_, i) => ({ id: i, score: 10 - i })));
    });
    const ids = Array.from({ length: 50 }, (_, i) => `c${i}`);
    await new OllamaReranker({ batchSize: 10 }).rerank("q", pool(ids), 8);
    expect(calls).toBe(5);
  });

  it("keeps every candidate scoreable across batch boundaries", async () => {
    mockGenerate((prompt) => {
      // Score only the last snippet of each batch highly.
      const n = (prompt.match(/^### \d+$/gm) ?? []).length;
      return JSON.stringify([{ id: n - 1, score: 10 }]);
    });
    const ids = Array.from({ length: 20 }, (_, i) => `c${i}`);
    const hits = await new OllamaReranker({ batchSize: 10 }).rerank("q", pool(ids), 2);
    expect(hits.map((h) => h.chunk.id)).toEqual(["c9", "c19"]);
  });
});

describe("ScoreCache", () => {
  it("returns cached scores without calling the model again", async () => {
    const dir = mkdtempSync(join(tmpdir(), "docstring-cache-"));
    const cachePath = join(dir, "cache.json");
    let calls = 0;
    mockGenerate(() => {
      calls++;
      return JSON.stringify([{ id: 0, score: 2 }, { id: 1, score: 9 }]);
    });

    const first = new OllamaReranker({ cachePath });
    const a = await first.rerank("same query", pool(["a", "b"]), 2);
    expect(calls).toBe(1);

    const second = new OllamaReranker({ cachePath });
    const b = await second.rerank("same query", pool(["a", "b"]), 2);
    expect(calls).toBe(1);
    expect(b.map((h) => h.chunk.id)).toEqual(a.map((h) => h.chunk.id));

    rmSync(dir, { recursive: true, force: true });
  });

  it("misses the cache when the candidate set changes", async () => {
    const dir = mkdtempSync(join(tmpdir(), "docstring-cache-"));
    const cachePath = join(dir, "cache.json");
    let calls = 0;
    mockGenerate(() => {
      calls++;
      return JSON.stringify([{ id: 0, score: 5 }]);
    });
    const r = new OllamaReranker({ cachePath });
    await r.rerank("q", pool(["a"]), 1);
    await r.rerank("q", pool(["b"]), 1);
    expect(calls).toBe(2);
    rmSync(dir, { recursive: true, force: true });
  });

  it("does not cache unparseable responses", async () => {
    const dir = mkdtempSync(join(tmpdir(), "docstring-cache-"));
    const cachePath = join(dir, "cache.json");
    let calls = 0;
    mockGenerate(() => {
      calls++;
      return "nope";
    });
    const r = new OllamaReranker({ cachePath });
    await r.rerank("q", pool(["a"]), 1);
    await r.rerank("q", pool(["a"]), 1);
    expect(calls).toBe(2);
    rmSync(dir, { recursive: true, force: true });
  });
});

describe("OllamaReranker.check", () => {
  it("names the missing model and how to install it", async () => {
    mockGenerate(() => "");
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ models: [{ name: "nomic-embed-text:latest" }] }),
    })) as unknown as typeof fetch;
    await expect(new OllamaReranker({ model: "qwen2.5:3b" }).check()).rejects.toThrow(
      /ollama pull qwen2\.5:3b/,
    );
  });

  it("passes when the model is installed", async () => {
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ models: [{ name: "qwen2.5:3b" }] }),
    })) as unknown as typeof fetch;
    await expect(new OllamaReranker({ model: "qwen2.5:3b" }).check()).resolves.toBeUndefined();
  });

  it("counts failures so a silent fallback is visible", async () => {
    mockGenerate(() => "not json");
    const r = new OllamaReranker();
    await r.rerank("q", pool(["a", "b"]), 2);
    expect(r.telemetry.parsed).toBe(0);
    expect(r.telemetry.parseFailed).toBe(1);
    expect(r.telemetry.sample).toContain("not json");
  });
});


describe("parseScores tolerates what small models actually return", () => {
  const shapes: [string, string][] = [
    ["bare array of objects", '[{"id":0,"score":7},{"id":1,"score":2}]'],
    ["object wrapping an array", '{"scores":[{"id":0,"score":7},{"id":1,"score":2}]}'],
    ["different wrapper key", '{"results":[{"id":0,"score":7},{"id":1,"score":2}]}'],
    ["positional numbers in an object", '{"scores":[7,2]}'],
    ["bare positional array", "[7,2]"],
    ["id to score map", '{"0":7,"1":2}'],
    ["alternative score key", '[{"id":0,"relevance":7},{"id":1,"relevance":2}]'],
    ["fenced json", '```json\n{"scores":[{"id":0,"score":7},{"id":1,"score":2}]}\n```'],
    ["prose then json", 'Here are the scores:\n[{"id":0,"score":7},{"id":1,"score":2}]'],
  ];

  for (const [name, text] of shapes) {
    it(`handles ${name}`, () => {
      const scores = parseScores(text, 2);
      expect(scores).toHaveLength(2);
      expect(scores.find((s) => s.index === 0)?.score).toBe(7);
      expect(scores.find((s) => s.index === 1)?.score).toBe(2);
    });
  }

  it("still rejects genuine refusals", () => {
    expect(parseScores("I cannot help with that.", 2)).toEqual([]);
  });

  it("drops ids outside the batch", () => {
    expect(parseScores('[{"id":99,"score":10},{"id":1,"score":3}]', 2)).toEqual([
      { index: 1, score: 3 },
    ]);
  });
});
