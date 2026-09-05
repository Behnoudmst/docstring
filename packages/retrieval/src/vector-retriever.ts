import type {
  ChunkStore,
  Embedder,
  RetrievalFilter,
  Retriever,
  ScoredChunk,
} from "@docstring/core";

/**
 * The naive baseline: embed the question, return the nearest chunks.
 * No keyword search, no fusion, no reranking. Everything added later is
 * measured as a delta against this.
 */
export class VectorRetriever implements Retriever {
  readonly name = "vector";

  constructor(
    private readonly embedder: Embedder,
    private readonly store: ChunkStore,
  ) {}

  async retrieve(query: string, k: number, filter?: RetrievalFilter): Promise<ScoredChunk[]> {
    // embedQuery, not embed: asymmetric models need the query-side prefix here.
    const embedding = await this.embedder.embedQuery(query);
    if (embedding.length === 0) return [];
    return this.store.vectorSearch(embedding, k, filter);
  }
}

/**
 * Lexical-only baseline. Not meant to be used alone, but measuring it
 * separately tells you how much of your retrieval is carried by exact
 * identifier matches — which for code is usually more than people expect.
 */
export class KeywordRetriever implements Retriever {
  readonly name = "keyword";

  constructor(private readonly store: ChunkStore) {}

  async retrieve(query: string, k: number, filter?: RetrievalFilter): Promise<ScoredChunk[]> {
    return this.store.keywordSearch(query, k, filter);
  }
}
