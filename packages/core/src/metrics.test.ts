import { describe, expect, it } from "vitest";
import { aggregate, ndcgAtK, precisionAtK, recallAtK, reciprocalRank, scoreQuestion } from "./metrics.js";
import { noisyOracleRetriever, oracleRetriever, randomRetriever } from "./testing/fake-retrievers.js";
import type { GroundTruth } from "./ground-truth.js";
import { validateGroundTruth } from "./ground-truth.js";

const corpus = Array.from({ length: 200 }, (_, i) => `src/file-${i}.ts`);

const gt: GroundTruth = {
  meta: {},
  questions: [
    {
      id: "q1",
      category: "locational",
      difficulty: "easy",
      question: "where is the middleware?",
      answerable: true,
      verified: true,
      expectedFiles: ["src/file-1.ts"],
    },
    {
      id: "q2",
      category: "cross-file",
      difficulty: "hard",
      question: "trace the login flow",
      answerable: true,
      verified: true,
      expectedFiles: ["src/file-2.ts", "src/file-3.ts", "src/file-4.ts"],
    },
    {
      id: "q3",
      category: "explanatory",
      difficulty: "medium",
      question: "how does sharding work?",
      answerable: false,
      nearMiss: false,
      refusalReason: "no sharding",
      expectedFiles: [],
    },
  ],
};

const answerable = gt.questions.filter((q) => q.answerable);

async function run(retriever: { name: string; retrieve: Function }, k = 20) {
  const scores = [];
  for (const q of answerable) {
    const results = await retriever.retrieve(q.question, k);
    scores.push(scoreQuestion(q.id, q.expectedFiles, results));
  }
  return aggregate(retriever.name, scores);
}

describe("primitive metrics", () => {
  it("recall counts expected files found within k", () => {
    expect(recallAtK(["a", "b"], ["a", "x", "b"], 3)).toBe(1);
    expect(recallAtK(["a", "b"], ["a", "x", "b"], 2)).toBe(0.5);
    expect(recallAtK(["a"], ["x", "y"], 5)).toBe(0);
  });

  it("precision measures noise in the top k", () => {
    expect(precisionAtK(["a"], ["a", "x", "y", "z"], 4)).toBe(0.25);
    expect(precisionAtK(["a", "b"], ["a", "b"], 2)).toBe(1);
  });

  it("reciprocal rank rewards ranking the answer first", () => {
    expect(reciprocalRank(["a"], ["a", "b"])).toBe(1);
    expect(reciprocalRank(["a"], ["x", "a"])).toBe(0.5);
    expect(reciprocalRank(["a"], ["x", "y"])).toBe(0);
  });

  it("ndcg is 1 only when all expected files lead the ranking", () => {
    expect(ndcgAtK(["a", "b"], ["a", "b", "x"], 3)).toBeCloseTo(1);
    expect(ndcgAtK(["a", "b"], ["x", "a", "b"], 3)).toBeLessThan(1);
  });

  it("k larger than the result list does not crash", () => {
    expect(recallAtK(["a"], ["a"], 100)).toBe(1);
    expect(precisionAtK(["a"], [], 10)).toBe(0);
  });
});

describe("ceiling: an oracle must score a perfect 1.0", () => {
  it("scores 1.0 on every metric", async () => {
    const agg = await run(oracleRetriever(gt));
    expect(agg.recall[20]).toBe(1);
    expect(agg.precision[20]).toBe(1);
    expect(agg.ndcg[20]).toBe(1);
    expect(agg.mrr).toBe(1);
    expect(agg.totalMisses).toEqual([]);
  });
});

describe("floor: a query-blind retriever must score near zero", () => {
  it("recalls almost nothing from a 200-file corpus", async () => {
    const agg = await run(randomRetriever(corpus));
    expect(agg.recall[20]).toBeLessThan(0.35);
    expect(agg.precision[20]).toBeLessThan(0.2);
  });
});

describe("noise separates recall from precision", () => {
  it("keeps recall at 1.0 while precision falls", async () => {
    const agg = await run(noisyOracleRetriever(gt, corpus));
    expect(agg.recall[20]).toBe(1);
    expect(agg.precision[20]).toBeLessThan(1);
  });
});

describe("ground truth validation", () => {
  it("accepts a well-formed set", () => {
    expect(validateGroundTruth(gt)).toEqual([]);
  });

  it("catches answerable questions with no expected files", () => {
    const broken: GroundTruth = {
      meta: {},
      questions: [{ ...gt.questions[0]!, expectedFiles: [] }, gt.questions[2]!],
    };
    expect(validateGroundTruth(broken)).toContain("q1: answerable but has no expectedFiles");
  });

  it("catches a set with no unanswerable questions", () => {
    const broken: GroundTruth = { meta: {}, questions: answerable };
    expect(validateGroundTruth(broken)).toContain(
      "no unanswerable questions — refusal cannot be measured",
    );
  });
});

describe("recall is monotonic in k", () => {
  it("never decreases as k grows", async () => {
    const ks = [5, 10, 20, 50];
    const results = Array.from({ length: 50 }, (_, i) => ({
      chunk: { path: i === 30 ? "src/file-2.ts" : `src/other-${i}.ts` } as never,
      score: 1 / (i + 1),
      source: "vector" as const,
    }));
    const s = scoreQuestion("q", ["src/file-2.ts"], results, ks);
    for (let i = 1; i < ks.length; i++) {
      expect(s.recall[ks[i]!]!).toBeGreaterThanOrEqual(s.recall[ks[i - 1]!]!);
    }
    expect(s.recall[50]).toBe(1);
    expect(s.recall[20]).toBe(0);
  });

  it("refuses to aggregate at k values the questions were not scored at", () => {
    const s = scoreQuestion("q", ["a"], [], [5, 10]);
    expect(() => aggregate("r", [s], [5, 10, 50])).toThrow(/scored without k=50/);
  });
});
