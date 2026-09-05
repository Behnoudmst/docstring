import type { RetrievalFilter, Retriever, ScoredChunk } from "@docstring/core";

export interface RrfOptions {
  /**
   * RRF's smoothing constant. 60 is the value from the original paper and
   * needs no tuning: it flattens the difference between ranks 1 and 2 enough
   * that a single confident-but-wrong result cannot dominate the fusion.
   */
  k?: number;
  /** How deep to go in each source before fusing. Deeper costs latency, not accuracy. */
  candidates?: number;
  /** Relative trust in each source. Equal by default — change one thing at a time. */
  weights?: { vector?: number; keyword?: number };
}

/**
 * Fuses two rankings by position rather than by score.
 *
 * This matters because cosine similarity and BM25 are not comparable numbers:
 * one is bounded, the other is not, and normalising them requires assumptions
 * that break per query. RRF only reads rank, so no calibration is needed.
 *
 *   score(d) = sum over sources of  weight / (k + rank(d))
 */
export function fuseRrf(
  rankings: { results: ScoredChunk[]; weight: number }[],
  k: number,
): ScoredChunk[] {
  const scores = new Map<string, number>();
  const byId = new Map<string, ScoredChunk>();

  for (const { results, weight } of rankings) {
    results.forEach((hit, index) => {
      const id = hit.chunk.id;
      scores.set(id, (scores.get(id) ?? 0) + weight / (k + index + 1));
      if (!byId.has(id)) byId.set(id, hit);
    });
  }

  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([id, score]) => ({
      chunk: byId.get(id)!.chunk,
      score,
      source: "fused" as const,
    }));
}

/**
 * Vector search finds code that means the same thing; keyword search finds
 * code that is literally named the thing. For a codebase both matter, and
 * they fail on different queries — "what happens when payment fails" needs
 * the first, "getSubDomain" needs the second.
 */
export class HybridRetriever implements Retriever {
  readonly name = "hybrid";
  private readonly k: number;
  private readonly candidates: number;
  private readonly weights: { vector: number; keyword: number };

  constructor(
    private readonly vector: Retriever,
    private readonly keyword: Retriever,
    opts: RrfOptions = {},
  ) {
    this.k = opts.k ?? 60;
    this.candidates = opts.candidates ?? 50;
    this.weights = {
      vector: opts.weights?.vector ?? 1,
      keyword: opts.weights?.keyword ?? 1,
    };
  }

  async retrieve(query: string, k: number, filter?: RetrievalFilter): Promise<ScoredChunk[]> {
    const depth = Math.max(this.candidates, k);
    const [vectorHits, keywordHits] = await Promise.all([
      this.vector.retrieve(query, depth, filter),
      this.keyword.retrieve(query, depth, filter),
    ]);

    return fuseRrf(
      [
        { results: vectorHits, weight: this.weights.vector },
        { results: keywordHits, weight: this.weights.keyword },
      ],
      this.k,
    ).slice(0, k);
  }
}
