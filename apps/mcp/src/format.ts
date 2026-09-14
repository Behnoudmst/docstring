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

/**
 * An empty result set hides its own cause: a filter that matched nothing looks
 * exactly like a query that matched nothing. Saying which one happened stops
 * the caller rephrasing a query that was never at fault, and stops it reading
 * the emptiness as proof the code does not exist.
 */
export function formatNoPrefixMatch(raw: string, normalized: string): string {
  const shown = raw === normalized ? `'${raw}'` : `'${raw}' (read as '${normalized}')`;
  return (
    `No indexed file is under ${shown}, so path_prefix discarded every result. ` +
    `The search itself was not the problem and this is not evidence the code is ` +
    `absent. Retry without path_prefix, or with a repo-relative prefix such as ` +
    `'app/api/'.`
  );
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
