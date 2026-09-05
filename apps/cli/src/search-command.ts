import type { Retriever, ScoredChunk } from "@docstring/core";
import {
  HybridRetriever,
  KeywordRetriever,
  OllamaQueryRewriter,
  OllamaReranker,
  RerankingRetriever,
  RewritingRetriever,
  VectorRetriever,
} from "@docstring/retrieval";
import { SqliteChunkStore, createEmbedder } from "@docstring/store";

/** Shared across an eval run so each question is rewritten once. */
const rewriteCache = new Map<string, string[]>();

export interface RetrieverHandle {
  retriever: Retriever;
  reranker?: OllamaReranker;
  close(): Promise<void>;
}

export async function openRetriever(opts: {
  db?: string;
  embedding?: string;
  kind?: string;
  candidates?: number;
  vectorWeight?: number;
  keywordWeight?: number;
  rewrite?: boolean;
  rewriteModel?: string;
  rewriteVariants?: number;
  originalWeight?: number;
  rerank?: boolean;
  rerankModel?: string;
  rerankBatch?: number;
  rerankSnippet?: number;
  rerankConcurrency?: number;
  cachePath?: string;
}): Promise<RetrieverHandle> {
  const embedder = createEmbedder(opts.embedding ?? "ollama:nomic-embed-text");
  const store = new SqliteChunkStore({
    path: opts.db ?? ".docstring/index.db",
    embeddingModel: embedder.model,
    embeddingDim: embedder.dim,
    embeddingPrefix: embedder.prefixes.document,
  });

  if ((await store.indexInfo()) === null) {
    await store.close();
    throw new Error(
      `The index at ${opts.db ?? ".docstring/index.db"} is empty. Run "pnpm index <repo>" first.`,
    );
  }

  const vector = new VectorRetriever(embedder, store);
  const keyword = new KeywordRetriever(store);

  let retriever: Retriever;
  if (opts.kind === "keyword") retriever = keyword;
  else if (opts.kind === "hybrid") {
    retriever = new HybridRetriever(vector, keyword, {
      candidates: opts.candidates,
      weights: { vector: opts.vectorWeight, keyword: opts.keywordWeight },
    });
  } else retriever = vector;

  // Rewriting wraps the base retriever; reranking wraps whatever comes out.
  if (opts.rewrite) {
    retriever = new RewritingRetriever(
      retriever,
      new OllamaQueryRewriter({
        model: opts.rewriteModel,
        variants: opts.rewriteVariants,
        cache: rewriteCache,
      }),
      { originalWeight: opts.originalWeight, candidates: opts.candidates },
    );
  }

  let reranker: OllamaReranker | undefined;
  if (opts.rerank) {
    reranker = new OllamaReranker({
      model: opts.rerankModel,
      batchSize: opts.rerankBatch,
      snippetChars: opts.rerankSnippet,
      concurrency: opts.rerankConcurrency,
      cachePath: opts.cachePath ?? ".docstring/rerank-cache.json",
    });
    // Fail loudly here rather than degrading silently on every call.
    await reranker.check();
    retriever = new RerankingRetriever(
      retriever,
      reranker,
      opts.candidates ?? 50,
    );
  }

  return { retriever, reranker, close: () => store.close() };
}

export function formatHits(query: string, hits: ScoredChunk[]): string {
  if (hits.length === 0) return `\n  no results for "${query}"\n`;
  const lines = [``, `  "${query}"`, ``];
  hits.forEach((h, i) => {
    const c = h.chunk;
    lines.push(
      `  ${String(i + 1).padStart(2)}. ${h.score.toFixed(3)}  ${c.path}:${c.startLine}-${c.endLine}`,
      `      ${c.symbolKind ?? "-"}  ${c.symbolName ?? "(unnamed)"}`,
      ...c.content
        .split("\n")
        .slice(0, 2)
        .map((l) => `      | ${l.slice(0, 90)}`),
      ``,
    );
  });
  return lines.join("\n");
}
