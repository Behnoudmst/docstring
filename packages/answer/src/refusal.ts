import type { ScoredChunk } from "@docstring/core";

/**
 * Signals read from the retrieval result, before any generation happens.
 *
 * Deliberately scale-free where possible. Cosine similarity, BM25 and RRF
 * produce values on completely different scales, so a threshold tuned on one
 * retriever is meaningless on another. Ratios transfer; raw scores do not.
 */
export interface RefusalSignals {
  /** Raw top score. Comparable only within one retriever. */
  topScore: number;
  /** Share of the total score held by the best result. High when one chunk dominates. */
  concentration: number;
  /** (s1 - s2) / s1. High when the winner is clearly ahead of the runner-up. */
  gapRatio: number;
  /** s1 / mean(s1..sn). High when the ranking decays sharply rather than staying flat. */
  decay: number;
  /** Chunks scoring at least half the top score. Low when only one thing matched. */
  aboveHalf: number;
  candidateCount: number;
}

export function computeSignals(hits: ScoredChunk[], n = 10): RefusalSignals {
  const scores = hits.slice(0, n).map((h) => h.score);
  const top = scores[0] ?? 0;
  const second = scores[1] ?? 0;
  const total = scores.reduce((a, b) => a + b, 0);
  const mean = scores.length > 0 ? total / scores.length : 0;

  return {
    topScore: top,
    concentration: total > 0 ? top / total : 0,
    gapRatio: top > 0 ? (top - second) / top : 0,
    decay: mean > 0 ? top / mean : 0,
    aboveHalf: scores.filter((s) => s >= top / 2).length,
    candidateCount: hits.length,
  };
}

export type SignalName = "topScore" | "concentration" | "gapRatio" | "decay";

export interface RefusalPolicy {
  /** Refuse when any configured minimum is not met. */
  minTopScore?: number;
  minConcentration?: number;
  minGapRatio?: number;
  minDecay?: number;
  /** Refuse when the retriever returned almost nothing. */
  minCandidates?: number;
}

export interface RefusalDecision {
  refuse: boolean;
  reasons: string[];
}

/**
 * A readable rule, not a model judgement. Someone who gets refused should be
 * able to be told exactly why, and a maintainer should be able to change it
 * without retraining anything.
 */
export function shouldRefuse(signals: RefusalSignals, policy: RefusalPolicy): RefusalDecision {
  const reasons: string[] = [];

  if (policy.minCandidates !== undefined && signals.candidateCount < policy.minCandidates) {
    reasons.push(`only ${signals.candidateCount} candidates were retrieved`);
  }
  if (policy.minTopScore !== undefined && signals.topScore < policy.minTopScore) {
    reasons.push(`best match scored ${signals.topScore.toFixed(3)}, below ${policy.minTopScore}`);
  }
  if (policy.minConcentration !== undefined && signals.concentration < policy.minConcentration) {
    reasons.push(
      `no result stood out (concentration ${signals.concentration.toFixed(3)} < ${policy.minConcentration})`,
    );
  }
  if (policy.minGapRatio !== undefined && signals.gapRatio < policy.minGapRatio) {
    reasons.push(
      `top two results were too close (gap ${signals.gapRatio.toFixed(3)} < ${policy.minGapRatio})`,
    );
  }
  if (policy.minDecay !== undefined && signals.decay < policy.minDecay) {
    reasons.push(
      `scores were flat across the ranking (decay ${signals.decay.toFixed(3)} < ${policy.minDecay})`,
    );
  }

  return { refuse: reasons.length > 0, reasons };
}

/**
 * Asks a model whether the retrieved context contains an answer at all,
 * without asking it to write one. Cheaper than generation and it catches the
 * case validation cannot: a real citation to a plausible but wrong chunk.
 */
export interface GroundednessCheck {
  answerable: boolean;
  missing?: string;
}

export function buildGroundednessPrompt(query: string, hits: ScoredChunk[], snippetChars = 600): string {
  const blocks = hits
    .slice(0, 8)
    .map((h, i) =>
      [
        `[${i + 1}] ${h.chunk.path}:${h.chunk.startLine}-${h.chunk.endLine}`,
        "```",
        h.chunk.content.slice(0, snippetChars),
        "```",
      ].join("\n"),
    )
    .join("\n\n");

  return [
    `Decide whether these code snippets contain enough information to answer a question.`,
    `Do not answer the question. Only judge whether the answer is present.`,
    ``,
    `Question: ${query}`,
    ``,
    `Snippets:`,
    blocks,
    ``,
    `If the snippets are about a related topic but do not contain the answer, that is "no".`,
    ``,
    `Reply with a JSON object only:`,
    `{"answerable": true or false, "missing": "what is absent, if anything"}`,
  ].join("\n");
}

export function parseGroundedness(text: string): GroundednessCheck | null {
  const candidates = [text.trim()];
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced?.[1]) candidates.push(fenced[1].trim());
  const object = text.match(/\{[\s\S]*\}/);
  if (object) candidates.push(object[0]);

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as Record<string, unknown>;
      const raw = parsed["answerable"] ?? parsed["contains_answer"] ?? parsed["can_answer"];
      if (raw === undefined) continue;
      const answerable =
        typeof raw === "boolean" ? raw : String(raw).toLowerCase().startsWith("t") || raw === "yes";
      const missing = typeof parsed["missing"] === "string" ? parsed["missing"] : undefined;
      return { answerable, missing };
    } catch {
      /* try the next shape */
    }
  }
  return null;
}
