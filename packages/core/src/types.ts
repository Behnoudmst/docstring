/**
 * A single indexed unit of source code.
 * `content` is what gets cited; `embeddingText` is what gets embedded.
 * Keeping them separate is what lets you enrich the embedding with file
 * path and symbol context without corrupting the citation.
 */
export interface Chunk {
  id: string;
  repo: string;
  path: string;
  language: string;
  symbolName: string | null;
  symbolKind: SymbolKind | null;
  startLine: number;
  endLine: number;
  content: string;
  embeddingText: string;
  imports: string[];
  gitSha: string;
  embeddingModel: string;
  embeddingDim: number;
}

export type SymbolKind =
  | "function"
  | "class"
  | "method"
  | "interface"
  | "type"
  | "component"
  | "constant"
  | "file";

export interface ScoredChunk {
  chunk: Chunk;
  score: number;
  /** Which stage produced this score. Useful when debugging hybrid + rerank. */
  source: "vector" | "keyword" | "fused" | "reranked";
}

export interface RetrievalFilter {
  repo?: string;
  pathPrefix?: string;
  language?: string;
  symbolKind?: SymbolKind;
}

/**
 * Persistence boundary. Implement once for sqlite-vec, later for pgvector.
 * Nothing above this interface should know which backend is in use.
 */
export interface ChunkStore {
  upsertChunks(chunks: Chunk[]): Promise<void>;
  deleteByFile(repo: string, path: string): Promise<number>;
  vectorSearch(
    embedding: number[],
    k: number,
    filter?: RetrievalFilter,
  ): Promise<ScoredChunk[]>;
  keywordSearch(
    query: string,
    k: number,
    filter?: RetrievalFilter,
  ): Promise<ScoredChunk[]>;
  getByIds(ids: string[]): Promise<Chunk[]>;
  /** Model + dim the index was built with, or null if empty. */
  indexInfo(): Promise<{ embeddingModel: string; embeddingDim: number } | null>;
  close(): Promise<void>;
}

/**
 * The only thing the eval runner needs. Every retrieval strategy —
 * naive, hybrid, reranked — implements this, so they are directly comparable.
 */
export interface Retriever {
  readonly name: string;
  retrieve(
    query: string,
    k: number,
    filter?: RetrievalFilter,
  ): Promise<ScoredChunk[]>;
}

/**
 * Reorders a candidate set by reading the query and each chunk together.
 * Slower and more accurate than embedding similarity, which compares two
 * vectors computed independently and never sees the pair.
 */
export interface Reranker {
  readonly name: string;
  rerank(query: string, candidates: ScoredChunk[], topK: number): Promise<ScoredChunk[]>;
}

export interface Embedder {
  readonly model: string;
  readonly dim: number;
  embed(texts: string[]): Promise<number[][]>;
  /** Embeds chunks for indexing. May apply a model-specific document prefix. */
  embedDocuments(texts: string[]): Promise<number[][]>;
  /** Embeds a user question. May apply a different, query-specific prefix. */
  embedQuery(text: string): Promise<number[]>;
  /** Task prefixes in use. The document side is recorded with the index. */
  readonly prefixes: { query: string; document: string };
}

/** Thrown when an index built with one model is queried with another. */
export class EmbeddingMismatchError extends Error {
  constructor(
    public readonly indexed: { model: string; dim: number },
    public readonly current: { model: string; dim: number },
  ) {
    super(
      `Index was built with ${indexed.model} (dim ${indexed.dim}) but the current config uses ${current.model} (dim ${current.dim}). Re-index before querying.`,
    );
    this.name = "EmbeddingMismatchError";
  }
}
