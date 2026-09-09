import { McpServer } from "@modelcontextprotocol/server";
import { serveStdio } from "@modelcontextprotocol/server/stdio";
import * as z from "zod/v4";
import { Answerer, OllamaClient } from "@docstring/answer";
import { indexRepo } from "@docstring/chunker";
import type { Chunk } from "@docstring/core";
import {
  HybridRetriever,
  KeywordRetriever,
  OllamaQueryRewriter,
  RewritingRetriever,
  VectorRetriever,
} from "@docstring/retrieval";
import { SqliteChunkStore, createEmbedder } from "@docstring/store";
import { clampK, formatHits } from "./format.js";
import { fallbackDbPath, resolvePaths } from "./paths.js";

/**
 * stdout is the JSON-RPC channel. Anything written there that is not a
 * protocol message corrupts the stream and the host disconnects with a parse
 * error, so every diagnostic goes to stderr.
 */
function log(message: string): void {
  process.stderr.write(`[docstring] ${message}\n`);
}

const EMBEDDING = process.env["DOCSTRING_EMBEDDING"] ?? "ollama:nomic-embed-text";
const ANSWER_MODEL = process.env["DOCSTRING_MODEL"] ?? "qwen2.5:3b";
const paths = resolvePaths();

interface Session {
  store: SqliteChunkStore;
  retriever: RewritingRetriever;
  answerer: Answerer;
  dbPath: string;
}

let session: Session | null = null;

/**
 * Opened once per process, not per call: a host spawns this server once and
 * calls it many times, and reopening SQLite would re-run the embedding
 * dimension check every time.
 */
function open(): Session {
  if (session) return session;

  const embedder = createEmbedder(EMBEDDING);
  let dbPath = paths.dbPath;
  let store: SqliteChunkStore;

  const build = (path: string) =>
    new SqliteChunkStore({
      path,
      embeddingModel: embedder.model,
      embeddingDim: embedder.dim,
      embeddingPrefix: embedder.prefixes.document,
    });

  try {
    store = build(dbPath);
  } catch (err) {
    // A read-only or otherwise unwritable repo should not stop the server:
    // fall back to a per-repo index under the home directory.
    if (paths.explicit) throw err;
    dbPath = fallbackDbPath(paths.repoRoot);
    log(`cannot write ${paths.dbPath}, falling back to ${dbPath}`);
    store = build(dbPath);
  }

  const base = new HybridRetriever(
    new VectorRetriever(embedder, store),
    new KeywordRetriever(store),
    { weights: { vector: 1, keyword: 0.3 } },
  );

  session = {
    store,
    dbPath,
    retriever: new RewritingRetriever(
      base,
      new OllamaQueryRewriter({ model: ANSWER_MODEL }),
      { originalWeight: 1.5 },
    ),
    answerer: new Answerer(new OllamaClient(ANSWER_MODEL)),
  };
  return session;
}

/** Told to the agent, not the human: it is the one who can act on it. */
const NO_INDEX =
  `No index exists for this repository yet. Call index_repo to build one ` +
  `(a few minutes for a medium codebase), then retry. Until then, use grep ` +
  `or file search instead.`;

async function ensureIndexed(): Promise<string | null> {
  const { store } = open();
  return (await store.indexInfo()) === null ? NO_INDEX : null;
}

function errorResult(message: string) {
  return { isError: true, content: [{ type: "text" as const, text: message }] };
}

function textResult(text: string) {
  return { content: [{ type: "text" as const, text }] };
}

function createServer(): McpServer {
  const server = new McpServer({ name: "docstring", version: "1.0.0" });

  server.registerTool(
    "search_code",
    {
      description:
        "Semantic and keyword search over an indexed codebase. Returns code chunks " +
        "with file paths and line ranges. Use it for conceptual questions where you " +
        "do not know the identifier — 'how does authentication work', 'where are " +
        "orders validated'. Prefer grep when you know the exact string or symbol " +
        "name; grep is exact and this is not. Recall is roughly two thirds, so treat " +
        "a miss as inconclusive rather than proof that something does not exist.",
      inputSchema: z.object({
        query: z.string().describe("What to look for, described in plain language"),
        k: z.number().int().min(1).max(30).optional().describe("Results to return (default 8)"),
        path_prefix: z.string().optional().describe("Restrict to a directory, e.g. 'src/lib/'"),
      }),
    },
    async ({ query, k, path_prefix }) => {
      try {
        const missing = await ensureIndexed();
        if (missing) return textResult(missing);
        const { retriever } = open();
        const hits = await retriever.retrieve(
          query,
          clampK(k, 8, 30),
          path_prefix ? { pathPrefix: path_prefix } : undefined,
        );
        return textResult(formatHits(hits));
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log(`search_code failed: ${message}`);
        return errorResult(`Search failed: ${message}`);
      }
    },
  );

  server.registerTool(
    "ask_codebase",
    {
      description:
        "Answer a question about the indexed codebase, with every claim cited to a " +
        "file and line range. Citations are validated in code: a cited chunk must be " +
        "one that was actually retrieved, and quoted code must appear verbatim in the " +
        "source it cites. Returns REFUSED rather than guessing when the retrieved " +
        "code does not support an answer — treat a refusal as a signal to search or " +
        "grep yourself, not as evidence of absence. Slower than search_code; use that " +
        "when you only need locations.",
      inputSchema: z.object({
        question: z.string().describe("A question about how the codebase works"),
        top_k: z.number().int().min(1).max(20).optional().describe("Sources given to the model (default 8)"),
      }),
    },
    async ({ question, top_k }) => {
      try {
        const missing = await ensureIndexed();
        if (missing) return textResult(missing);

        const { retriever, answerer } = open();
        const hits = await retriever.retrieve(question, clampK(top_k, 8, 20));
        const result = await answerer.answer(question, hits);

        if (result.kind === "refusal") {
          const nearby = result.sources
            .slice(0, 3)
            .map((s) => `  ${s.chunk.path}:${s.chunk.startLine}-${s.chunk.endLine}`)
            .join("\n");
          return textResult(
            `REFUSED: ${result.reason}\n\n` +
              (nearby ? `Closest sources considered:\n${nearby}\n\n` : "") +
              `The answer may still exist in the codebase. Try search_code with ` +
              `different wording, or grep for a specific identifier.`,
          );
        }

        const claims = result.claims
          .map(
            (c, i) =>
              `${i + 1}. ${c.text}\n` +
              c.citations
                .map(
                  (cit) =>
                    `   -> ${cit.path}:${cit.startLine}-${cit.endLine}` +
                    (cit.symbolName ? `  ${cit.symbolName}` : ""),
                )
                .join("\n"),
          )
          .join("\n");

        return textResult(
          `${result.text}\n\nClaims and citations:\n${claims}\n\n` +
            `confidence ${result.confidence.toFixed(2)} | ` +
            `${result.validation.quotesChecked} quotes verified | ` +
            `${result.sources.length} sources considered`,
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log(`ask_codebase failed: ${message}`);
        return errorResult(`Answering failed: ${message}`);
      }
    },
  );

  server.registerTool(
    "index_repo",
    {
      description:
        "Build or refresh the search index for this repository. Required once before " +
        "search_code or ask_codebase will return anything, and worth rerunning after " +
        "significant changes — the index is a snapshot, not a live view. Only files " +
        "whose contents changed are re-embedded, so a refresh is much faster than the " +
        "first build. Takes a few minutes on a medium codebase.",
      inputSchema: z.object({
        force: z.boolean().optional().describe("Re-embed every file, ignoring content hashes"),
      }),
    },
    async ({ force }) => {
      try {
        const { store, dbPath } = open();
        const embedder = createEmbedder(EMBEDDING);
        await embedder.check();

        const repo = paths.repoRoot.split("/").pop() ?? "repo";
        const known = force ? new Map<string, string>() : store.fileHashes(repo);

        const files = await indexRepo(paths.repoRoot, {
          repo,
          gitSha: "unset",
          embeddingModel: embedder.model,
          embeddingDim: embedder.dim,
        });

        let persisted = 0;
        let skipped = 0;
        for (const file of files) {
          if (known.get(file.path) === file.contentHash) {
            skipped++;
            continue;
          }
          await store.deleteByFile(repo, file.path);
          const vectors = await embedder.embedDocuments(
            file.chunks.map((c: Chunk) => c.embeddingText),
          );
          await store.upsertChunks(file.chunks, vectors);
          store.recordFile(repo, file.path, file.contentHash);
          persisted += file.chunks.length;
        }

        const stats = store.stats();
        log(`indexed ${paths.repoRoot}: ${persisted} chunks written, ${skipped} files unchanged`);
        return textResult(
          `Indexed ${paths.repoRoot}\n` +
            `  ${persisted} chunks written, ${skipped} files unchanged\n` +
            `  index now holds ${stats.chunks} chunks across ${stats.files} files\n` +
            `  database: ${dbPath}`,
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log(`index_repo failed: ${message}`);
        return errorResult(
          `Indexing failed: ${message}\n\n` +
            `This usually means Ollama is not running or the embedding model is not ` +
            `installed. Try: ollama serve, then ollama pull nomic-embed-text`,
        );
      }
    },
  );

  server.registerTool(
    "index_status",
    {
      description:
        "What is currently indexed: repository path, chunk and file counts, and the " +
        "embedding model. Call this first if search results look empty or stale.",
      inputSchema: z.object({}),
    },
    async () => {
      try {
        const { store, dbPath } = open();
        const stats = store.stats();
        const info = await store.indexInfo();
        return textResult(
          info === null
            ? `Repository: ${paths.repoRoot}\nNo index yet. Call index_repo to build one.`
            : [
                `repository: ${paths.repoRoot}`,
                `database:   ${dbPath}`,
                `files:      ${stats.files}`,
                `chunks:     ${stats.chunks}`,
                `vectors:    ${stats.vectors}`,
                `embedding:  ${info.embeddingModel} (${info.embeddingDim}d)`,
                ``,
                `The index is a snapshot. Call index_repo after significant changes;`,
                `unchanged files are skipped.`,
              ].join("\n"),
        );
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return errorResult(`Could not read the index: ${message}`);
      }
    },
  );

  return server;
}

log(`repo=${paths.repoRoot} db=${paths.dbPath} embedding=${EMBEDDING} model=${ANSWER_MODEL}`);
serveStdio(createServer);
