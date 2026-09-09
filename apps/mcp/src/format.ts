import type { ScoredChunk } from "@docstring/core";

/**
 * Clamp rather than trust the declared schema bounds. A model can send an
 * out-of-range value, and an unbounded k would pull the whole index into an
 * agent's context. Clamping in the handler makes the bound real.
 */
export function clampK(value: unknown, fallback: number, max: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(1, Math.min(max, Math.trunc(n)));
}

/** Agents parse this. Structured lines beat prose. */
export function formatHits(hits: ScoredChunk[]): string {
  if (hits.length === 0) {
    return "No results. The index may not cover this area, or the wording may not match the code.";
  }
  return hits
    .map((h, i) => {
      const c = h.chunk;
      return [
        `${i + 1}. ${c.path}:${c.startLine}-${c.endLine}`,
        `   symbol: ${c.symbolName ?? "(none)"} (${c.symbolKind ?? "?"})  score: ${h.score.toFixed(3)}`,
        c.content.split("\n").slice(0, 20).map((l) => `   | ${l}`).join("\n"),
      ].join("\n");
    })
    .join("\n\n");
}
