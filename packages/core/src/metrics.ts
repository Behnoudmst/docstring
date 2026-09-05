import type { ScoredChunk } from "./types.js";

/**
 * Retrieved chunks collapsed to the distinct files they came from,
 * in rank order. File-level is the right granularity for ground truth:
 * you know which file holds the answer, not which chunk id.
 */
export function retrievedFiles(results: ScoredChunk[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const r of results) {
    if (!seen.has(r.chunk.path)) {
      seen.add(r.chunk.path);
      out.push(r.chunk.path);
    }
  }
  return out;
}

/** Fraction of expected files that appear in the top k. The metric that matters most. */
export function recallAtK(expected: string[], retrieved: string[], k: number): number {
  if (expected.length === 0) return 1;
  const top = new Set(retrieved.slice(0, k));
  const hits = expected.filter((f) => top.has(f)).length;
  return hits / expected.length;
}

/** Fraction of the top k that was actually wanted. Measures noise. */
export function precisionAtK(expected: string[], retrieved: string[], k: number): number {
  const top = retrieved.slice(0, k);
  if (top.length === 0) return 0;
  const want = new Set(expected);
  return top.filter((f) => want.has(f)).length / top.length;
}

/** Reciprocal rank of the first correct file. Rewards ranking the answer high. */
export function reciprocalRank(expected: string[], retrieved: string[]): number {
  const want = new Set(expected);
  for (let i = 0; i < retrieved.length; i++) {
    const path = retrieved[i];
    if (path !== undefined && want.has(path)) return 1 / (i + 1);
  }
  return 0;
}

/** Binary-relevance nDCG@k. Rewards getting *all* expected files high, not just one. */
export function ndcgAtK(expected: string[], retrieved: string[], k: number): number {
  if (expected.length === 0) return 1;
  const want = new Set(expected);
  let dcg = 0;
  for (let i = 0; i < Math.min(k, retrieved.length); i++) {
    const path = retrieved[i];
    if (path !== undefined && want.has(path)) dcg += 1 / Math.log2(i + 2);
  }
  let idcg = 0;
  for (let i = 0; i < Math.min(k, expected.length); i++) {
    idcg += 1 / Math.log2(i + 2);
  }
  return idcg === 0 ? 0 : dcg / idcg;
}

export interface QuestionScore {
  id: string;
  recall: Record<number, number>;
  precision: Record<number, number>;
  ndcg: Record<number, number>;
  mrr: number;
  expected: string[];
  retrieved: string[];
}

export function scoreQuestion(
  id: string,
  expected: string[],
  results: ScoredChunk[],
  ks: number[] = [5, 10, 20],
): QuestionScore {
  const retrieved = retrievedFiles(results);
  const recall: Record<number, number> = {};
  const precision: Record<number, number> = {};
  const ndcg: Record<number, number> = {};
  for (const k of ks) {
    recall[k] = recallAtK(expected, retrieved, k);
    precision[k] = precisionAtK(expected, retrieved, k);
    ndcg[k] = ndcgAtK(expected, retrieved, k);
  }
  return { id, recall, precision, ndcg, mrr: reciprocalRank(expected, retrieved), expected, retrieved };
}

export interface AggregateScore {
  retrieverName: string;
  questionCount: number;
  recall: Record<number, number>;
  precision: Record<number, number>;
  ndcg: Record<number, number>;
  mrr: number;
  /** Questions where not one expected file was retrieved at the largest k. */
  totalMisses: string[];
}

export function aggregate(
  retrieverName: string,
  scores: QuestionScore[],
  ks: number[] = [5, 10, 20],
): AggregateScore {
  // Guard against scoring at one set of k values and aggregating at another:
  // the missing entries silently read as zero and recall stops being monotonic.
  for (const s of scores) {
    for (const k of ks) {
      if (s.recall[k] === undefined) {
        throw new Error(
          `question ${s.id} was scored without k=${k}. Pass the same ks to scoreQuestion and aggregate.`,
        );
      }
    }
  }
  const mean = (xs: number[]) => (xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length);
  const maxK = Math.max(...ks);
  const recall: Record<number, number> = {};
  const precision: Record<number, number> = {};
  const ndcg: Record<number, number> = {};
  for (const k of ks) {
    recall[k] = mean(scores.map((s) => s.recall[k] ?? 0));
    precision[k] = mean(scores.map((s) => s.precision[k] ?? 0));
    ndcg[k] = mean(scores.map((s) => s.ndcg[k] ?? 0));
  }
  return {
    retrieverName,
    questionCount: scores.length,
    recall,
    precision,
    ndcg,
    mrr: mean(scores.map((s) => s.mrr)),
    totalMisses: scores.filter((s) => (s.recall[maxK] ?? 0) === 0).map((s) => s.id),
  };
}
