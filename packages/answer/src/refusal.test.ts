import { describe, expect, it } from "vitest";
import type { Chunk, ScoredChunk } from "@docstring/core";
import {
  buildGroundednessPrompt,
  computeSignals,
  parseGroundedness,
  shouldRefuse,
} from "./refusal.js";

const chunk = (id: string): Chunk => ({
  id, repo: "r", path: `src/${id}.ts`, language: "typescript",
  symbolName: id, symbolKind: "function", startLine: 1, endLine: 5,
  content: `export function ${id}() {}`, embeddingText: "t", imports: [],
  gitSha: "a".repeat(40), embeddingModel: "m", embeddingDim: 3,
});

const hits = (scores: number[]): ScoredChunk[] =>
  scores.map((score, i) => ({ chunk: chunk(`c${i}`), score, source: "fused" as const }));

describe("computeSignals", () => {
  it("reports a sharp peak when one result dominates", () => {
    const s = computeSignals(hits([1.0, 0.2, 0.15, 0.1, 0.05]));
    expect(s.gapRatio).toBeCloseTo(0.8);
    expect(s.aboveHalf).toBe(1);
    expect(s.decay).toBeGreaterThan(2);
  });

  it("reports a flat distribution when nothing matched distinctively", () => {
    const s = computeSignals(hits([0.5, 0.49, 0.48, 0.47, 0.46]));
    expect(s.gapRatio).toBeLessThan(0.05);
    expect(s.aboveHalf).toBe(5);
    expect(s.decay).toBeCloseTo(1, 1);
  });

  it("is scale-free: ratios survive multiplying every score", () => {
    const small = computeSignals(hits([0.01, 0.002, 0.001]));
    const large = computeSignals(hits([100, 20, 10]));
    expect(small.gapRatio).toBeCloseTo(large.gapRatio, 6);
    expect(small.concentration).toBeCloseTo(large.concentration, 6);
    expect(small.decay).toBeCloseTo(large.decay, 6);
  });

  it("handles an empty result without dividing by zero", () => {
    const s = computeSignals([]);
    expect(s.topScore).toBe(0);
    expect(s.concentration).toBe(0);
    expect(s.gapRatio).toBe(0);
  });
});

describe("shouldRefuse", () => {
  const strong = computeSignals(hits([1.0, 0.2, 0.1]));
  const flat = computeSignals(hits([0.5, 0.49, 0.48]));

  it("allows a confident retrieval through", () => {
    expect(shouldRefuse(strong, { minGapRatio: 0.3 }).refuse).toBe(false);
  });

  it("refuses a flat ranking", () => {
    const d = shouldRefuse(flat, { minGapRatio: 0.3 });
    expect(d.refuse).toBe(true);
    expect(d.reasons[0]).toMatch(/too close/);
  });

  it("explains every rule that fired, not just the first", () => {
    const d = shouldRefuse(flat, { minGapRatio: 0.3, minDecay: 2, minConcentration: 0.8 });
    expect(d.reasons).toHaveLength(3);
  });

  it("refuses when almost nothing was retrieved", () => {
    const d = shouldRefuse(computeSignals(hits([0.9])), { minCandidates: 3 });
    expect(d.refuse).toBe(true);
    expect(d.reasons[0]).toMatch(/only 1 candidates/);
  });

  it("an empty policy never refuses", () => {
    expect(shouldRefuse(flat, {}).refuse).toBe(false);
  });
});

describe("groundedness", () => {
  it("asks the model to judge, not to answer", () => {
    const prompt = buildGroundednessPrompt("where is auth", hits([1, 0.5]));
    expect(prompt).toContain("Do not answer the question");
    expect(prompt).toContain("[1] src/c0.ts");
  });

  it("parses the requested shape", () => {
    expect(parseGroundedness('{"answerable": false, "missing": "no schema files"}')).toEqual({
      answerable: false,
      missing: "no schema files",
    });
  });

  it("accepts alternative keys and string booleans", () => {
    expect(parseGroundedness('{"contains_answer": "yes"}')?.answerable).toBe(true);
    expect(parseGroundedness('{"can_answer": "false"}')?.answerable).toBe(false);
  });

  it("survives fenced output", () => {
    expect(parseGroundedness('```json\n{"answerable": true}\n```')?.answerable).toBe(true);
  });

  it("returns null rather than guessing on unparseable output", () => {
    expect(parseGroundedness("I think maybe")).toBeNull();
  });
});
