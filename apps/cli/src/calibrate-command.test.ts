import { describe, expect, it } from "vitest";
import { analyse, type Observation } from "./calibrate-command.js";
import { computeSignals } from "@docstring/answer";
import type { Chunk, ScoredChunk } from "@docstring/core";

const chunk = (id: string): Chunk => ({
  id, repo: "r", path: `src/${id}.ts`, language: "typescript",
  symbolName: id, symbolKind: "function", startLine: 1, endLine: 5,
  content: "code", embeddingText: "t", imports: [],
  gitSha: "a".repeat(40), embeddingModel: "m", embeddingDim: 3,
});

const hits = (scores: number[]): ScoredChunk[] =>
  scores.map((score, i) => ({ chunk: chunk(`c${i}`), score, source: "fused" as const }));

const obs = (id: string, answerable: boolean, scores: number[]): Observation => ({
  id, answerable, nearMiss: false, signals: computeSignals(hits(scores)),
});

describe("analyse", () => {
  it("finds a threshold that separates peaked from flat retrievals", () => {
    const observations = [
      obs("a1", true, [1.0, 0.1, 0.05]),
      obs("a2", true, [1.0, 0.15, 0.05]),
      obs("u1", false, [0.5, 0.49, 0.48]),
      obs("u2", false, [0.5, 0.5, 0.49]),
    ];
    const gap = analyse(observations).find((r) => r.signal === "gapRatio")!;
    expect(gap.separation).toBeGreaterThan(0.5);
    expect(gap.best?.trueRefusalRate).toBe(1);
    expect(gap.best?.falseRefusalRate).toBe(0);
  });

  it("reports near-zero Youden when the groups are indistinguishable", () => {
    const observations = [
      obs("a1", true, [0.5, 0.49, 0.48]),
      obs("a2", true, [0.5, 0.48, 0.47]),
      obs("u1", false, [0.5, 0.49, 0.48]),
      obs("u2", false, [0.5, 0.48, 0.47]),
    ];
    for (const r of analyse(observations)) {
      expect(r.best?.youden ?? 0).toBeLessThan(0.6);
    }
  });

  it("trades true refusal against false refusal monotonically", () => {
    const observations = [
      obs("a1", true, [1.0, 0.1]),
      obs("a2", true, [0.8, 0.3]),
      obs("u1", false, [0.5, 0.45]),
    ];
    const gap = analyse(observations).find((r) => r.signal === "gapRatio")!;
    for (let i = 1; i < gap.sweep.length; i++) {
      expect(gap.sweep[i]!.trueRefusalRate).toBeGreaterThanOrEqual(gap.sweep[i - 1]!.trueRefusalRate);
      expect(gap.sweep[i]!.falseRefusalRate).toBeGreaterThanOrEqual(gap.sweep[i - 1]!.falseRefusalRate);
    }
  });

  it("evaluates every signal, not just the best one", () => {
    const reports = analyse([obs("a", true, [1, 0.1]), obs("u", false, [0.5, 0.49])]);
    expect(reports.map((r) => r.signal).sort()).toEqual([
      "concentration", "decay", "gapRatio", "topScore",
    ]);
  });
});
