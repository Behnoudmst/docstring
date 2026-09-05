import { indexRepo, type IndexedFile } from "@docstring/chunker";
import { SqliteChunkStore, createEmbedder } from "@docstring/store";
import type { Chunk } from "@docstring/core";

export interface IndexSummary {
  files: number;
  chunks: number;
  byKind: Record<string, number>;
  byLanguage: Record<string, number>;
  unnamed: number;
  largest: { path: string; symbol: string | null; chars: number }[];
  sizeBuckets: Record<string, number>;
  oversized: number;
  medianChars: number;
}

export function summarise(files: IndexedFile[]): IndexSummary {
  const byKind: Record<string, number> = {};
  const byLanguage: Record<string, number> = {};
  const all = files.flatMap((f) => f.chunks);

  for (const c of all) {
    const kind = c.symbolKind ?? "none";
    byKind[kind] = (byKind[kind] ?? 0) + 1;
    byLanguage[c.language] = (byLanguage[c.language] ?? 0) + 1;
  }

  const sizes = all.map((c) => c.content.length).sort((a, b) => a - b);
  const median = sizes.length === 0 ? 0 : (sizes[Math.floor(sizes.length / 2)] ?? 0);
  const buckets: Record<string, number> = {
    "0-500": 0, "500-1500": 0, "1500-3000": 0, "3000-6000": 0, "6000+": 0,
  };
  for (const n of sizes) {
    if (n < 500) buckets["0-500"]!++;
    else if (n < 1500) buckets["500-1500"]!++;
    else if (n < 3000) buckets["1500-3000"]!++;
    else if (n < 6000) buckets["3000-6000"]!++;
    else buckets["6000+"]!++;
  }

  return {
    files: files.length,
    chunks: all.length,
    sizeBuckets: buckets,
    oversized: sizes.filter((n) => n > 3000).length,
    medianChars: median,
    byKind,
    byLanguage,
    unnamed: all.filter((c) => c.symbolName === null).length,
    largest: [...all]
      .sort((a, b) => b.content.length - a.content.length)
      .slice(0, 5)
      .map((c) => ({ path: c.path, symbol: c.symbolName, chars: c.content.length })),
  };
}

export function formatSummary(s: IndexSummary): string {
  const rows = (obj: Record<string, number>) =>
    Object.entries(obj)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `    ${k.padEnd(14)} ${String(v).padStart(5)}`)
      .join("\n");

  return [
    ``,
    `  ${s.files} files -> ${s.chunks} chunks  (${(s.chunks / Math.max(s.files, 1)).toFixed(1)} per file)`,
    ``,
    `  by kind`,
    rows(s.byKind),
    ``,
    `  by language`,
    rows(s.byLanguage),
    ``,
    `  chunk size (chars, median ${s.medianChars})`,
    rows(s.sizeBuckets),
    ``,
    `  oversized (>3000): ${s.oversized}${s.oversized > s.chunks * 0.05 ? "   <- too many, embeddings will be weak" : "   ok"}`,
    ``,
    `  unnamed chunks: ${s.unnamed}${s.unnamed > s.chunks * 0.3 ? "   <- high, check the chunker" : ""}`,
    ``,
    `  largest chunks`,
    ...s.largest.map((l) => `    ${String(l.chars).padStart(6)}ch  ${l.symbol ?? "-"}  ${l.path}`),
    ``,
  ].join("\n");
}

export interface IndexRunOptions {
  db?: string;
  embedding?: string;
  includeUnparsable?: boolean;
  dryRun?: boolean;
  force?: boolean;
  gitSha?: string;
}

export async function runIndex(root: string, opts: IndexRunOptions = {}) {
  const repo = root.replace(/\/$/, "").split("/").pop() ?? "repo";

  // Dry run: chunk only, no embedding model required. Use it to tune the chunker.
  if (opts.dryRun) {
    const files = await indexRepo(root, {
      repo,
      gitSha: opts.gitSha ?? "unset",
      embeddingModel: "none",
      embeddingDim: 0,
      includeUnparsable: opts.includeUnparsable,
    });
    return { summary: summarise(files), persisted: 0, skipped: 0, stats: null };
  }

  const embedder = createEmbedder(opts.embedding ?? "ollama:nomic-embed-text");
  process.stdout.write(`  checking ${embedder.model}... `);
  await embedder.check();
  process.stdout.write(`ok (${embedder.dim}d)\n`);

  const store = new SqliteChunkStore({
    path: opts.db ?? ".docstring/index.db",
    embeddingModel: embedder.model,
    embeddingDim: embedder.dim,
    embeddingPrefix: embedder.prefixes.document,
  });

  const known = opts.force ? new Map<string, string>() : store.fileHashes(repo);

  const files = await indexRepo(root, {
    repo,
    gitSha: opts.gitSha ?? "unset",
    embeddingModel: embedder.model,
    embeddingDim: embedder.dim,
    includeUnparsable: opts.includeUnparsable,
  });

  let persisted = 0;
  let skipped = 0;
  let done = 0;

  for (const file of files) {
    done++;
    if (known.get(file.path) === file.contentHash) {
      skipped++;
      continue;
    }
    await store.deleteByFile(repo, file.path);
    const vectors = await embedder.embedDocuments(file.chunks.map((c: Chunk) => c.embeddingText));
    await store.upsertChunks(file.chunks, vectors);
    store.recordFile(repo, file.path, file.contentHash);
    persisted += file.chunks.length;
    process.stdout.write(
      `\r  embedding ${done}/${files.length} files, ${persisted} chunks written    `,
    );
  }
  process.stdout.write("\n");

  const stats = store.stats();
  await store.close();

  return { summary: summarise(files), persisted, skipped, stats };
}
