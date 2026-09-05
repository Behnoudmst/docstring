import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import {
  aggregate,
  scoreQuestion,
  validateGroundTruth,
  type AggregateScore,
  type GroundTruth,
  type QuestionScore,
  type Retriever,
} from "@docstring/core";

const KS = [5, 10, 20, 50];

export async function loadGroundTruth(path: string): Promise<GroundTruth> {
  const gt = JSON.parse(await readFile(path, "utf8")) as GroundTruth;
  const problems = validateGroundTruth(gt);
  if (problems.length > 0) {
    throw new Error(`ground truth is invalid:\n  - ${problems.join("\n  - ")}`);
  }
  return gt;
}

export interface EvalRun {
  retriever: string;
  timestamp: string;
  onlyVerified: boolean;
  aggregate: AggregateScore;
  perQuestion: QuestionScore[];
}

export async function runEval(
  gt: GroundTruth,
  retriever: Retriever,
  opts: { onlyVerified?: boolean; limit?: number; onProgress?: (done: number, total: number) => void } = {},
): Promise<EvalRun> {
  const onlyVerified = opts.onlyVerified ?? true;
  let questions = gt.questions.filter(
    (q) => q.answerable && (!onlyVerified || q.verified === true),
  );
  // Deterministic subset: every nth question, so the sample stays spread across
  // categories rather than taking the first N (which are all locational).
  if (opts.limit && opts.limit < questions.length) {
    const step = questions.length / opts.limit;
    questions = Array.from({ length: opts.limit }, (_, i) => questions[Math.floor(i * step)]!);
  }

  if (questions.length === 0) {
    throw new Error(
      onlyVerified
        ? "No verified answerable questions. Set verified:true as you confirm each one, or pass --all."
        : "No answerable questions in the ground truth.",
    );
  }

  const scores: QuestionScore[] = [];
  for (const q of questions) {
    const results = await retriever.retrieve(q.question, Math.max(...KS));
    scores.push(scoreQuestion(q.id, q.expectedFiles, results, KS));
    opts.onProgress?.(scores.length, questions.length);
  }

  return {
    retriever: retriever.name,
    timestamp: new Date().toISOString(),
    onlyVerified,
    aggregate: aggregate(retriever.name, scores, KS),
    perQuestion: scores,
  };
}

export function formatTable(run: EvalRun): string {
  const a = run.aggregate;
  const pct = (n: number) => (n * 100).toFixed(1).padStart(6);
  const lines = [
    ``,
    `  retriever: ${a.retrieverName}   questions: ${a.questionCount}${run.onlyVerified ? " (verified only)" : " (all)"}`,
    ``,
    `  k     recall   precision   nDCG`,
    `  ---------------------------------`,
    ...KS.map((k) => `  ${String(k).padEnd(4)} ${pct(a.recall[k] ?? 0)}%   ${pct(a.precision[k] ?? 0)}%  ${pct(a.ndcg[k] ?? 0)}%`),
    ``,
    `  MRR: ${a.mrr.toFixed(3)}`,
  ];
  if (a.totalMisses.length > 0) {
    lines.push(``, `  total misses (nothing found at k=${Math.max(...KS)}):`, `  ${a.totalMisses.join(", ")}`);
  }
  return lines.join("\n");
}

export async function saveRun(run: EvalRun, dir = "evals/results"): Promise<string> {
  const file = resolve(dir, `${run.timestamp.replace(/[:.]/g, "-")}_${run.retriever}.json`);
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify(run, null, 2));
  return file;
}
