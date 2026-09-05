import type { GroundTruthQuestion, Retriever } from "@docstring/core";
import { computeSignals, type RefusalSignals, type SignalName } from "@docstring/answer";
import { loadGroundTruth } from "./eval.js";
import { openRetriever } from "./search-command.js";

export interface Observation {
  id: string;
  answerable: boolean;
  nearMiss: boolean;
  signals: RefusalSignals;
}

const SIGNALS: SignalName[] = ["topScore", "concentration", "gapRatio", "decay"];

export async function collect(
  questions: GroundTruthQuestion[],
  retriever: Retriever,
  k: number,
  onProgress?: (done: number, total: number) => void,
): Promise<Observation[]> {
  const out: Observation[] = [];
  for (const q of questions) {
    const hits = await retriever.retrieve(q.question, k);
    out.push({
      id: q.id,
      answerable: q.answerable,
      nearMiss: q.nearMiss === true,
      signals: computeSignals(hits, k),
    });
    onProgress?.(out.length, questions.length);
  }
  return out;
}

export interface SweepRow {
  threshold: number;
  /** Unanswerable questions correctly refused. */
  trueRefusalRate: number;
  /** Answerable questions wrongly refused. The one users feel. */
  falseRefusalRate: number;
  /** Distance above the diagonal: 0 means no better than chance. */
  youden: number;
}

export interface SignalReport {
  signal: SignalName;
  meanAnswerable: number;
  meanUnanswerable: number;
  /** Positive means answerable questions score higher, which is the useful direction. */
  separation: number;
  sweep: SweepRow[];
  best: SweepRow | null;
}

export function analyse(observations: Observation[], steps = 20): SignalReport[] {
  const answerable = observations.filter((o) => o.answerable);
  const unanswerable = observations.filter((o) => !o.answerable);

  return SIGNALS.map((signal) => {
    const valuesA = answerable.map((o) => o.signals[signal]);
    const valuesU = unanswerable.map((o) => o.signals[signal]);
    const mean = (xs: number[]) => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);
    const all = [...valuesA, ...valuesU];
    const lo = Math.min(...all);
    const hi = Math.max(...all);

    const sweep: SweepRow[] = [];
    for (let i = 0; i <= steps; i++) {
      const threshold = lo + ((hi - lo) * i) / steps;
      // Refuse when the signal falls below the threshold.
      const trueRefusalRate =
        valuesU.length === 0 ? 0 : valuesU.filter((v) => v < threshold).length / valuesU.length;
      const falseRefusalRate =
        valuesA.length === 0 ? 0 : valuesA.filter((v) => v < threshold).length / valuesA.length;
      sweep.push({
        threshold,
        trueRefusalRate,
        falseRefusalRate,
        youden: trueRefusalRate - falseRefusalRate,
      });
    }

    const best = sweep.reduce<SweepRow | null>(
      (acc, row) => (acc === null || row.youden > acc.youden ? row : acc),
      null,
    );

    return {
      signal,
      meanAnswerable: mean(valuesA),
      meanUnanswerable: mean(valuesU),
      separation: mean(valuesA) - mean(valuesU),
      sweep,
      best,
    };
  });
}

export function formatCalibration(observations: Observation[], reports: SignalReport[]): string {
  const answerable = observations.filter((o) => o.answerable).length;
  const unanswerable = observations.length - answerable;

  const lines = [
    ``,
    `  ${observations.length} questions: ${answerable} answerable, ${unanswerable} unanswerable`,
    ``,
    `  signal          mean(answerable)  mean(unanswerable)  separation   best Youden`,
    `  ------------------------------------------------------------------------------`,
  ];

  for (const r of [...reports].sort((a, b) => (b.best?.youden ?? 0) - (a.best?.youden ?? 0))) {
    lines.push(
      `  ${r.signal.padEnd(15)} ${r.meanAnswerable.toFixed(4).padStart(15)} ` +
        `${r.meanUnanswerable.toFixed(4).padStart(19)} ${r.separation.toFixed(4).padStart(11)} ` +
        `${(r.best?.youden ?? 0).toFixed(3).padStart(12)}`,
    );
  }

  const top = [...reports].sort((a, b) => (b.best?.youden ?? 0) - (a.best?.youden ?? 0))[0];
  if (top?.best) {
    lines.push(
      ``,
      `  best single signal: ${top.signal} at threshold ${top.best.threshold.toFixed(4)}`,
      `    correctly refuses ${(top.best.trueRefusalRate * 100).toFixed(0)}% of unanswerable questions`,
      `    wrongly refuses   ${(top.best.falseRefusalRate * 100).toFixed(0)}% of answerable ones`,
      ``,
      `  full sweep for ${top.signal}:`,
      ``,
      `    threshold      true refusal   false refusal   Youden`,
      `    ------------------------------------------------------`,
      ...top.sweep.map(
        (row) =>
          `    ${row.threshold.toFixed(4).padStart(9)}   ${(row.trueRefusalRate * 100).toFixed(0).padStart(11)}%   ` +
          `${(row.falseRefusalRate * 100).toFixed(0).padStart(12)}%   ${row.youden.toFixed(3).padStart(6)}`,
      ),
      ``,
    );
  }

  if ((top?.best?.youden ?? 0) < 0.2) {
    lines.push(
      `  No signal separates the two groups well. Retrieval scores alone cannot`,
      `  tell "the answer is here" from "something similar is here" — which is`,
      `  exactly what a near-miss question is designed to produce. A groundedness`,
      `  check that reads the retrieved code is the next thing to try.`,
      ``,
    );
  }

  return lines.join("\n");
}

export interface CalibrateOptions {
  db?: string;
  embedding?: string;
  retriever?: string;
  keywordWeight?: number;
  vectorWeight?: number;
  k?: number;
}

export async function runCalibrate(groundTruthPath: string, opts: CalibrateOptions = {}) {
  const gt = await loadGroundTruth(groundTruthPath);
  // Unanswerable questions carry no verified flag: there is nothing to verify.
  const questions = gt.questions.filter((q) => !q.answerable || q.verified === true);

  const handle = await openRetriever({
    db: opts.db,
    embedding: opts.embedding,
    kind: opts.retriever ?? "hybrid",
    keywordWeight: opts.keywordWeight ?? 0.3,
    vectorWeight: opts.vectorWeight,
  });

  try {
    const started = Date.now();
    const observations = await collect(
      questions,
      handle.retriever,
      opts.k ?? 10,
      (done, total) => {
        const elapsed = (Date.now() - started) / 1000;
        process.stdout.write(`\r  ${done}/${total} questions, ${elapsed.toFixed(0)}s    `);
      },
    );
    process.stdout.write("\r".padEnd(60) + "\r");
    return { observations, reports: analyse(observations) };
  } finally {
    await handle.close();
  }
}
