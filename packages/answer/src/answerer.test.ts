import { describe, expect, it, vi } from "vitest";
import type { Chunk, ScoredChunk } from "@docstring/core";
import { extractQuotes, toSources, validateAnswer } from "./validate.js";
import { Answerer, buildPrompt, parseAnswer, type LlmClient } from "./answerer.js";

const chunk = (over: Partial<Chunk> = {}): Chunk => ({
  id: "c1",
  repo: "elegant-menu-front",
  path: "src/lib/getSubDomain.ts",
  language: "typescript",
  symbolName: "getSubDomain",
  symbolKind: "function",
  startLine: 3,
  endLine: 7,
  content: 'export function getSubDomain(host: string) {\n  const parts = host.split(".");\n  return parts[0];\n}',
  embeddingText: "t",
  imports: [],
  gitSha: "a".repeat(40),
  embeddingModel: "m",
  embeddingDim: 3,
  ...over,
});

const hits = (chunks: Chunk[]): ScoredChunk[] =>
  chunks.map((c, i) => ({ chunk: c, score: 1 / (i + 1), source: "fused" as const }));

const sources = toSources(hits([chunk(), chunk({ id: "c2", path: "src/middleware.ts", symbolName: "middleware", content: "export default function middleware(request: Request) {\n  const host = request.headers.get(\"host\");\n}" })]));

describe("extractQuotes", () => {
  it("pulls out backticked code spans", () => {
    expect(extractQuotes("it calls `host.split(\".\")` first")).toEqual(['host.split(".")']);
  });

  it("pulls out fenced blocks", () => {
    expect(extractQuotes("```ts\nconst a = 1;\n```")).toEqual(["const a = 1;"]);
  });

  it("ignores bare identifiers, which are names rather than quotations", () => {
    expect(extractQuotes("the `getSubDomain` helper")).toEqual([]);
  });

  it("normalises whitespace so formatting differences do not fail a real quote", () => {
    expect(extractQuotes("`const  parts =   host.split(\".\")`")).toEqual([
      'const parts = host.split(".")',
    ]);
  });
});

describe("validateAnswer", () => {
  it("accepts a well-cited answer", () => {
    const report = validateAnswer(
      {
        answer: "The subdomain is taken from the host header.",
        claims: [{ text: "The host is split on dots.", citations: [1] }],
        confidence: 0.8,
      },
      sources,
    );
    expect(report.ok).toBe(true);
    expect(report.citedFraction).toBe(1);
  });

  it("rejects a citation to a source that was never retrieved", () => {
    const report = validateAnswer(
      { answer: "x", claims: [{ text: "something", citations: [9] }], confidence: 0.9 },
      sources,
    );
    expect(report.ok).toBe(false);
    expect(report.errors.map((e) => e.kind)).toContain("unknown-citation");
  });

  it("rejects a claim with no citation at all", () => {
    const report = validateAnswer(
      { answer: "x", claims: [{ text: "an uncited assertion", citations: [] }], confidence: 0.9 },
      sources,
    );
    expect(report.errors.map((e) => e.kind)).toContain("uncited-claim");
  });

  it("accepts a quote that appears literally in the cited source", () => {
    const report = validateAnswer(
      {
        answer: "x",
        claims: [{ text: 'it runs `const parts = host.split(".")`', citations: [1] }],
        confidence: 0.9,
      },
      sources,
    );
    expect(report.ok).toBe(true);
    expect(report.quotesChecked).toBe(1);
  });

  it("rejects a fabricated quote — the case the whole feature exists for", () => {
    const report = validateAnswer(
      {
        answer: "x",
        claims: [{ text: "it runs `const parts = host.split(SEPARATOR)`", citations: [1] }],
        confidence: 0.9,
      },
      sources,
    );
    expect(report.ok).toBe(false);
    expect(report.errors.map((e) => e.kind)).toContain("quote-not-found");
  });

  it("rejects a real quote attributed to the wrong source", () => {
    const report = validateAnswer(
      {
        answer: "x",
        claims: [{ text: 'middleware does `const parts = host.split(".")`', citations: [2] }],
        confidence: 0.9,
      },
      sources,
    );
    expect(report.errors.map((e) => e.kind)).toContain("quote-not-found");
  });

  it("reports partial citation coverage rather than pass or fail alone", () => {
    const report = validateAnswer(
      {
        answer: "x",
        claims: [
          { text: "cited", citations: [1] },
          { text: "uncited", citations: [] },
        ],
        confidence: 0.9,
      },
      sources,
    );
    expect(report.citedFraction).toBe(0.5);
  });
});

describe("parseAnswer", () => {
  it("reads the requested shape", () => {
    const raw = parseAnswer('{"answer":"a","claims":[{"text":"t","citations":[1]}],"confidence":0.7}');
    expect(raw?.claims[0]?.citations).toEqual([1]);
    expect(raw?.confidence).toBe(0.7);
  });

  it("accepts alternative citation keys", () => {
    const raw = parseAnswer('{"answer":"a","claims":[{"text":"t","sources":[2]}],"confidence":0.5}');
    expect(raw?.claims[0]?.citations).toEqual([2]);
  });

  it("survives fences and surrounding prose", () => {
    const raw = parseAnswer('Sure:\n```json\n{"answer":"a","claims":[],"confidence":0.4}\n```');
    expect(raw?.answer).toBe("a");
  });

  it("clamps confidence into range", () => {
    expect(parseAnswer('{"answer":"a","claims":[],"confidence":5}')?.confidence).toBe(1);
  });

  it("returns null on unparseable output", () => {
    expect(parseAnswer("I cannot help with that")).toBeNull();
  });
});

describe("buildPrompt", () => {
  it("numbers the sources and shows their locations", () => {
    const prompt = buildPrompt("how does routing work", sources);
    expect(prompt).toContain("[1] src/lib/getSubDomain.ts:3-7");
    expect(prompt).toContain("[2] src/middleware.ts");
    expect(prompt).toContain("how does routing work");
  });
});

const stubLlm = (responses: string[]): LlmClient => {
  let i = 0;
  return {
    model: "stub",
    async complete() {
      return responses[Math.min(i++, responses.length - 1)]!;
    },
  };
};

describe("Answerer", () => {
  const good = JSON.stringify({
    answer: "The subdomain identifies the venue.",
    claims: [{ text: "The host is split on dots.", citations: [1] }],
    confidence: 0.8,
  });

  const fabricated = JSON.stringify({
    answer: "x",
    claims: [{ text: "it calls `const parts = host.split(SEPARATOR)`", citations: [1] }],
    confidence: 0.9,
  });

  it("returns a validated answer with resolved citations", async () => {
    const result = await new Answerer(stubLlm([good])).answer("q", hits([chunk()]));
    expect(result.kind).toBe("answer");
    expect(result.attempts).toBe(1);
    expect(result.claims[0]?.citations[0]).toMatchObject({
      path: "src/lib/getSubDomain.ts",
      startLine: 3,
      endLine: 7,
    });
  });

  it("retries once with the failure attached, then succeeds", async () => {
    const llm = stubLlm([fabricated, good]);
    const spy = vi.spyOn(llm, "complete");
    const result = await new Answerer(llm).answer("q", hits([chunk()]));
    expect(result.kind).toBe("answer");
    expect(result.attempts).toBe(2);
    expect(spy.mock.calls[1]?.[0]).toContain("failed validation");
  });

  it("refuses rather than returning an answer whose citations never validate", async () => {
    const result = await new Answerer(stubLlm([fabricated])).answer("q", hits([chunk()]));
    expect(result.kind).toBe("refusal");
    expect(result.reason).toMatch(/could not be verified/);
    expect(result.text).toBe("");
  });

  it("treats zero confidence from the model as a refusal, not a failure", async () => {
    const zero = JSON.stringify({
      answer: "The sources do not mention Kubernetes.",
      claims: [],
      confidence: 0,
    });
    const result = await new Answerer(stubLlm([zero])).answer("q", hits([chunk()]));
    expect(result.kind).toBe("refusal");
    expect(result.reason).toContain("Kubernetes");
    expect(result.attempts).toBe(1);
  });

  it("refuses when nothing was retrieved, without calling the model", async () => {
    const llm = stubLlm([good]);
    const spy = vi.spyOn(llm, "complete");
    const result = await new Answerer(llm).answer("q", []);
    expect(result.kind).toBe("refusal");
    expect(spy).not.toHaveBeenCalled();
  });

  it("refuses when the model is unreachable", async () => {
    const broken: LlmClient = {
      model: "stub",
      async complete() {
        throw new Error("ECONNREFUSED");
      },
    };
    const result = await new Answerer(broken).answer("q", hits([chunk()]));
    expect(result.kind).toBe("refusal");
    expect(result.reason).toContain("ECONNREFUSED");
  });

  it("never cites a source outside the retrieved set", async () => {
    const outOfRange = JSON.stringify({
      answer: "x",
      claims: [{ text: "claim", citations: [1, 42] }],
      confidence: 0.9,
    });
    const result = await new Answerer(stubLlm([outOfRange, good])).answer("q", hits([chunk()]));
    for (const claim of result.claims) {
      for (const citation of claim.citations) {
        expect(citation.ref).toBeLessThanOrEqual(result.sources.length);
      }
    }
  });
});

describe("quote extraction boundary", () => {
  it("checks short expressions but not bare names", () => {
    expect(extractQuotes("calls `host.split(\".\")`")).toHaveLength(1);
    expect(extractQuotes("the `middleware` function")).toHaveLength(0);
    expect(extractQuotes("uses `parts[0]`")).toHaveLength(1);
    expect(extractQuotes("sets `x = 1`")).toHaveLength(1);
  });
});
