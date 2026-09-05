import type { Chunk, Retriever, ScoredChunk } from "../types.js";
import type { GroundTruth } from "../ground-truth.js";

export function fakeChunk(path: string, i = 0): Chunk {
  return {
    id: `${path}#${i}`,
    repo: "test",
    path,
    language: "typescript",
    symbolName: null,
    symbolKind: "file",
    startLine: 1,
    endLine: 10,
    content: `// ${path}`,
    embeddingText: `File: ${path}`,
    imports: [],
    gitSha: "0".repeat(40),
    embeddingModel: "fake",
    embeddingDim: 3,
  };
}

const score = (chunks: Chunk[]): ScoredChunk[] =>
  chunks.map((chunk, i) => ({ chunk, score: 1 / (i + 1), source: "vector" as const }));

/**
 * Deterministic pseudo-random retriever. Ignores the query entirely,
 * so it establishes the floor: whatever this scores is what "no signal" looks like.
 */
export function randomRetriever(corpus: string[], seed = 42): Retriever {
  let state = seed;
  const next = () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return state / 2147483648;
  };
  return {
    name: "random",
    async retrieve(_query, k) {
      const pool = [...corpus].sort(() => next() - 0.5);
      return score(pool.slice(0, k).map((p) => fakeChunk(p)));
    },
  };
}

/**
 * Returns exactly the expected files. Establishes the ceiling:
 * every metric must be 1.0. If it is not, the metric is wrong, not the retriever.
 */
export function oracleRetriever(gt: GroundTruth): Retriever {
  const byQuestion = new Map(gt.questions.map((q) => [q.question, q.expectedFiles]));
  return {
    name: "oracle",
    async retrieve(query, k) {
      const expected = byQuestion.get(query) ?? [];
      return score(expected.slice(0, k).map((p) => fakeChunk(p)));
    },
  };
}

/** Oracle padded with junk after the correct files: recall 1.0, precision below 1.0. */
export function noisyOracleRetriever(gt: GroundTruth, corpus: string[]): Retriever {
  const inner = oracleRetriever(gt);
  return {
    name: "noisy-oracle",
    async retrieve(query, k) {
      const hits = await inner.retrieve(query, k);
      const have = new Set(hits.map((h) => h.chunk.path));
      const filler = corpus.filter((p) => !have.has(p)).slice(0, Math.max(0, k - hits.length));
      return [...hits, ...score(filler.map((p) => fakeChunk(p)))];
    },
  };
}
