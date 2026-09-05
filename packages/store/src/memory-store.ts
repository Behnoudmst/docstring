import {
  EmbeddingMismatchError,
  type Chunk,
  type ChunkStore,
  type RetrievalFilter,
  type ScoredChunk,
} from "@docstring/core";

export function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error(`dimension mismatch: ${a.length} vs ${b.length}`);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

function matches(chunk: Chunk, f?: RetrievalFilter): boolean {
  if (!f) return true;
  if (f.repo && chunk.repo !== f.repo) return false;
  if (f.pathPrefix && !chunk.path.startsWith(f.pathPrefix)) return false;
  if (f.language && chunk.language !== f.language) return false;
  if (f.symbolKind && chunk.symbolKind !== f.symbolKind) return false;
  return true;
}

/**
 * Reference implementation of ChunkStore. Not for production — it exists so
 * retrieval, fusion and reranking can be built and tested before any database
 * is involved, and so the sqlite-vec adapter has something to be checked against.
 */
export class MemoryChunkStore implements ChunkStore {
  private chunks = new Map<string, Chunk>();
  private vectors = new Map<string, number[]>();

  constructor(private readonly embedFn?: (text: string) => number[]) {}

  async upsertChunks(chunks: Chunk[], vectors?: number[][]): Promise<void> {
    const info = await this.indexInfo();
    chunks.forEach((c, i) => {
      if (info && (info.embeddingModel !== c.embeddingModel || info.embeddingDim !== c.embeddingDim)) {
        throw new EmbeddingMismatchError(
          { model: info.embeddingModel, dim: info.embeddingDim },
          { model: c.embeddingModel, dim: c.embeddingDim },
        );
      }
      this.chunks.set(c.id, c);
      const v = vectors?.[i] ?? this.embedFn?.(c.embeddingText);
      if (v) this.vectors.set(c.id, v);
    });
  }

  async deleteByFile(repo: string, path: string): Promise<number> {
    let n = 0;
    for (const [id, c] of this.chunks) {
      if (c.repo === repo && c.path === path) {
        this.chunks.delete(id);
        this.vectors.delete(id);
        n++;
      }
    }
    return n;
  }

  async vectorSearch(embedding: number[], k: number, filter?: RetrievalFilter): Promise<ScoredChunk[]> {
    const info = await this.indexInfo();
    if (info && info.embeddingDim !== embedding.length) {
      throw new EmbeddingMismatchError(
        { model: info.embeddingModel, dim: info.embeddingDim },
        { model: "query", dim: embedding.length },
      );
    }
    const out: ScoredChunk[] = [];
    for (const [id, chunk] of this.chunks) {
      if (!matches(chunk, filter)) continue;
      const v = this.vectors.get(id);
      if (!v) continue;
      out.push({ chunk, score: cosine(embedding, v), source: "vector" });
    }
    return out.sort((a, b) => b.score - a.score).slice(0, k);
  }

  /** Naive token overlap. A real store uses FTS5 / tsvector; this keeps the shape honest. */
  async keywordSearch(query: string, k: number, filter?: RetrievalFilter): Promise<ScoredChunk[]> {
    const terms = query.toLowerCase().split(/\W+/).filter((t) => t.length > 2);
    const out: ScoredChunk[] = [];
    for (const chunk of this.chunks.values()) {
      if (!matches(chunk, filter)) continue;
      const hay = `${chunk.path} ${chunk.symbolName ?? ""} ${chunk.content}`.toLowerCase();
      const hits = terms.filter((t) => hay.includes(t)).length;
      if (hits > 0) out.push({ chunk, score: hits / terms.length, source: "keyword" });
    }
    return out.sort((a, b) => b.score - a.score).slice(0, k);
  }

  async getByIds(ids: string[]): Promise<Chunk[]> {
    return ids.map((id) => this.chunks.get(id)).filter((c): c is Chunk => c !== undefined);
  }

  async indexInfo(): Promise<{ embeddingModel: string; embeddingDim: number } | null> {
    const first = this.chunks.values().next();
    if (first.done) return null;
    return { embeddingModel: first.value.embeddingModel, embeddingDim: first.value.embeddingDim };
  }

  async close(): Promise<void> {
    this.chunks.clear();
    this.vectors.clear();
  }

  get size(): number {
    return this.chunks.size;
  }
}
