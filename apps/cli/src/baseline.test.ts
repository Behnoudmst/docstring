import { describe, expect, it } from "vitest";
import type { AggregateScore } from "@docstring/core";
import { compare, toBaseline } from "./baseline.js";

const agg = (over: Partial<AggregateScore> = {}): AggregateScore => ({
  retrieverName: "hybrid",
  questionCount: 62,
  recall: { 5: 0.497, 10: 0.587, 20: 0.724, 50: 0.767 },
  precision: { 5: 0.18, 10: 0.11, 20: 0.07, 50: 0.05 },
  ndcg: { 5: 0.4, 10: 0.45, 20: 0.5, 50: 0.52 },
  mrr: 0.546,
  totalMisses: ["q038", "q061", "q063"],
  ...over,
});

describe("compare", () => {
  const baseline = toBaseline(agg());

  it("passes an unchanged run", () => {
    const result = compare(baseline, agg());
    expect(result.ok).toBe(true);
    expect(result.regressions).toEqual([]);
  });

  it("ignores movement inside the tolerance", () => {
    const result = compare(baseline, agg({ mrr: 0.536 }), 0.02);
    expect(result.ok).toBe(true);
  });

  it("fails on a drop beyond the tolerance", () => {
    const result = compare(baseline, agg({ recall: { 5: 0.497, 10: 0.48, 20: 0.724, 50: 0.767 } }));
    expect(result.ok).toBe(false);
    expect(result.regressions[0]?.metric).toBe("recall@10");
    expect(result.regressions[0]?.delta).toBeLessThan(0);
  });

  it("reports improvements without failing", () => {
    const result = compare(baseline, agg({ mrr: 0.72 }));
    expect(result.ok).toBe(true);
    expect(result.improvements[0]?.metric).toBe("mrr");
  });

  it("names questions that newly broke and newly passed", () => {
    const result = compare(baseline, agg({ totalMisses: ["q038", "q999"] }));
    expect(result.newMisses).toEqual(["q999"]);
    expect(result.fixedMisses).toEqual(["q061", "q063"]);
  });

  it("refuses to compare runs over different question counts", () => {
    const result = compare(baseline, agg({ questionCount: 70 }));
    expect(result.ok).toBe(false);
    expect(result.incomparable).toMatch(/Re-record the baseline/);
  });
});

describe("ask defaults", () => {
  it("uses the measured-best retrieval configuration without flags", async () => {
    const mod = await import("./ask-command.js");
    const src = await import("node:fs/promises").then((fs) =>
      fs.readFile(new URL("./ask-command.ts", import.meta.url), "utf8"),
    );
    expect(typeof mod.runAsk).toBe("function");
    // Guards against a future edit silently reverting the shipped config.
    expect(src).toContain('opts.keywordWeight ?? 0.3');
    expect(src).toContain('opts.rewrite ?? true');
    expect(src).toContain('opts.originalWeight ?? 1.5');
  });
});
