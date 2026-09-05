import { afterEach, describe, expect, it, vi } from "vitest";
import type { Chunk, Retriever, ScoredChunk } from "@docstring/core";
import {
  OllamaQueryRewriter,
  RewritingRetriever,
  parseVariants,
  type QueryRewriter,
} from "./query-rewriter.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

const chunk = (id: string): Chunk => ({
  id, repo: "r", path: `src/${id}.ts`, language: "typescript",
  symbolName: id, symbolKind: "function", startLine: 1, endLine: 5,
  content: "code", embeddingText: "t", imports: [],
  gitSha: "a".repeat(40), embeddingModel: "m", embeddingDim: 3,
});

const ranked = (ids: string[]): ScoredChunk[] =>
  ids.map((id, i) => ({ chunk: chunk(id), score: 1 / (i + 1), source: "vector" as const }));

/** Returns a different ranking per query, so fusion behaviour is observable. */
const perQuery = (map: Record<string, string[]>): Retriever => ({
  name: "base",
  async retrieve(q, k) {
    return ranked(map[q] ?? []).slice(0, k);
  },
});

const stubRewriter = (variants: string[]): QueryRewriter => ({
  name: "rw",
  async rewrite() {
    return variants;
  },
});

describe("parseVariants", () => {
  it("reads the requested shape", () => {
    expect(parseVariants('{"queries":["a","b"]}', 2)).toEqual(["a", "b"]);
  });

  it("accepts a bare array", () => {
    expect(parseVariants('["a","b"]', 2)).toEqual(["a", "b"]);
  });

  it("accepts any array-valued key", () => {
    expect(parseVariants('{"rewrites":["a"]}', 2)).toEqual(["a"]);
  });

  it("survives fences and prose", () => {
    expect(parseVariants('Here:\n```json\n{"queries":["a"]}\n```', 2)).toEqual(["a"]);
  });

  it("respects the variant limit", () => {
    expect(parseVariants('["a","b","c","d"]', 2)).toEqual(["a", "b"]);
  });

  it("returns nothing on unparseable output rather than guessing", () => {
    expect(parseVariants("I cannot help", 2)).toEqual([]);
  });
});

describe("RewritingRetriever", () => {
  const base = perQuery({
    "how does the app decide which menu to show": ["noise1", "noise2", "noise3"],
    "subdomain host header middleware": ["getSubDomain", "middleware", "noise1"],
  });

  it("surfaces results the original question could not reach", async () => {
    const r = new RewritingRetriever(base, stubRewriter(["subdomain host header middleware"]));
    const hits = await r.retrieve("how does the app decide which menu to show", 3);
    expect(hits.map((h) => h.chunk.id)).toContain("getSubDomain");
  });

  it("keeps the original ranking in the fusion", async () => {
    const r = new RewritingRetriever(base, stubRewriter(["subdomain host header middleware"]));
    const hits = await r.retrieve("how does the app decide which menu to show", 10);
    expect(hits.map((h) => h.chunk.id)).toContain("noise1");
  });

  it("weights the original above any single rewrite", async () => {
    const split = perQuery({ original: ["A"], rewrite: ["B"] });
    const r = new RewritingRetriever(split, stubRewriter(["rewrite"]), { originalWeight: 2 });
    const hits = await r.retrieve("original", 2);
    expect(hits[0]?.chunk.id).toBe("A");
  });

  it("falls back to the original when rewriting produces nothing", async () => {
    const r = new RewritingRetriever(base, stubRewriter([]));
    const hits = await r.retrieve("how does the app decide which menu to show", 3);
    expect(hits.map((h) => h.chunk.id)).toEqual(["noise1", "noise2", "noise3"]);
  });
});

describe("OllamaQueryRewriter", () => {
  function mockGenerate(reply: string) {
    globalThis.fetch = vi.fn(async () => ({
      ok: true,
      json: async () => ({ response: reply }),
    })) as unknown as typeof fetch;
  }

  it("returns the rewritten queries", async () => {
    mockGenerate('{"queries":["subdomain host header","resolve venue from hostname"]}');
    expect(await new OllamaQueryRewriter().rewrite("how does routing work")).toEqual([
      "subdomain host header",
      "resolve venue from hostname",
    ]);
  });

  it("caches per query so repeated evaluation is cheap", async () => {
    let calls = 0;
    globalThis.fetch = vi.fn(async () => {
      calls++;
      return { ok: true, json: async () => ({ response: '{"queries":["a"]}' }) };
    }) as unknown as typeof fetch;
    const r = new OllamaQueryRewriter();
    await r.rewrite("q");
    await r.rewrite("q");
    expect(calls).toBe(1);
  });

  it("returns nothing when the model is unreachable, losing no query", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch;
    expect(await new OllamaQueryRewriter().rewrite("q")).toEqual([]);
  });
});
