import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import type { AggregateScore } from "@docstring/core";

export interface Baseline {
  recordedAt: string;
  retriever: string;
  questionCount: number;
  recall: Record<string, number>;
  mrr: number;
  /** Question ids that were misses when the baseline was recorded. */
  misses: string[];
  note?: string;
}

export function toBaseline(agg: AggregateScore, note?: string): Baseline {
  return {
    recordedAt: new Date().toISOString(),
    retriever: agg.retrieverName,
    questionCount: agg.questionCount,
    recall: Object.fromEntries(Object.entries(agg.recall).map(([k, v]) => [k, v])),
    mrr: agg.mrr,
    misses: agg.totalMisses,
    ...(note ? { note } : {}),
  };
}

export async function saveBaseline(baseline: Baseline, path: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, JSON.stringify(baseline, null, 2) + "\n");
}

export async function loadBaseline(path: string): Promise<Baseline> {
  return JSON.parse(await readFile(path, "utf8")) as Baseline;
}

export interface Regression {
  metric: string;
  baseline: number;
  current: number;
  delta: number;
}

export interface ComparisonResult {
  ok: boolean;
  regressions: Regression[];
  improvements: Regression[];
  newMisses: string[];
  fixedMisses: string[];
  /** Set when the two runs are not comparable at all. */
  incomparable?: string;
}

/**
 * Compares a run against a recorded baseline.
 *
 * The tolerance exists because retrieval is not perfectly deterministic across
 * model versions and a 1% wobble should not fail a build. Anything beyond it
 * is a real change and should be explained in the commit that caused it.
 */
export function compare(
  baseline: Baseline,
  current: AggregateScore,
  tolerance = 0.02,
): ComparisonResult {
  if (baseline.questionCount !== current.questionCount) {
    return {
      ok: false,
      regressions: [],
      improvements: [],
      newMisses: [],
      fixedMisses: [],
      incomparable:
        `baseline covers ${baseline.questionCount} questions but this run covers ` +
        `${current.questionCount}. Re-record the baseline after changing the question set.`,
    };
  }

  const regressions: Regression[] = [];
  const improvements: Regression[] = [];

  const check = (metric: string, before: number, after: number) => {
    const delta = after - before;
    if (delta < -tolerance) regressions.push({ metric, baseline: before, current: after, delta });
    else if (delta > tolerance) improvements.push({ metric, baseline: before, current: after, delta });
  };

  for (const [k, before] of Object.entries(baseline.recall)) {
    const after = current.recall[Number(k)];
    if (after !== undefined) check(`recall@${k}`, before, after);
  }
  check("mrr", baseline.mrr, current.mrr);

  const wasMissing = new Set(baseline.misses);
  const isMissing = new Set(current.totalMisses);

  return {
    ok: regressions.length === 0,
    regressions,
    improvements,
    newMisses: current.totalMisses.filter((id) => !wasMissing.has(id)),
    fixedMisses: baseline.misses.filter((id) => !isMissing.has(id)),
  };
}

export function formatComparison(result: ComparisonResult, tolerance: number): string {
  if (result.incomparable) {
    return `\n  cannot compare: ${result.incomparable}\n`;
  }

  const lines: string[] = [``];
  const row = (r: Regression, sign: string) =>
    `    ${r.metric.padEnd(12)} ${(r.baseline * 100).toFixed(1).padStart(6)}% -> ` +
    `${(r.current * 100).toFixed(1).padStart(6)}%  (${sign}${(r.delta * 100).toFixed(1)})`;

  if (result.regressions.length > 0) {
    lines.push(`  REGRESSIONS (beyond ${(tolerance * 100).toFixed(0)}% tolerance)`);
    lines.push(...result.regressions.map((r) => row(r, "")));
    lines.push(``);
  }
  if (result.improvements.length > 0) {
    lines.push(`  improvements`);
    lines.push(...result.improvements.map((r) => row(r, "+")));
    lines.push(``);
  }
  if (result.newMisses.length > 0) {
    lines.push(`  newly missing: ${result.newMisses.join(", ")}`, ``);
  }
  if (result.fixedMisses.length > 0) {
    lines.push(`  now found: ${result.fixedMisses.join(", ")}`, ``);
  }
  if (result.ok && result.improvements.length === 0 && result.newMisses.length === 0) {
    lines.push(`  no change beyond tolerance`, ``);
  }

  return lines.join("\n");
}
