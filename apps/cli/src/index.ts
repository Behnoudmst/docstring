import { randomRetriever, oracleRetriever } from "@docstring/core";
import { formatTable, loadGroundTruth, runEval, saveRun } from "./eval.js";
import { formatSummary, runIndex } from "./index-command.js";
import { formatHits, openRetriever } from "./search-command.js";
import { formatAnswer, runAsk } from "./ask-command.js";
import { formatCalibration, runCalibrate } from "./calibrate-command.js";
import {
  compare,
  formatComparison,
  loadBaseline,
  saveBaseline,
  toBaseline,
} from "./baseline.js";

const [, , command, ...args] = process.argv;

const USAGE = `usage:
  docstring index <repo-path> [options]
      --db <path>          sqlite index file (default .docstring/index.db)
      --embedding <spec>   ollama:<model> or openai:<model>
      --dry-run            chunk only, no embedding model needed
      --force              re-embed every file, ignoring content hashes
      --include-unparsable index css/md/txt via the text fallback

  docstring search "<query>" [--k 10] [--db <path>]
      --retriever vector|keyword|hybrid    (default vector)
      --candidates <n>     depth fused per source for hybrid (default 50)
      --vector-weight <n>  RRF weight for vector   (default 1)
      --keyword-weight <n> RRF weight for keyword
      --rerank             rerank candidates with a local model
      --rerank-model <m>   Ollama model for reranking (default qwen2.5:3b)  (default 1)
      --rerank             rerank candidates with a local model
      --rerank-model <m>   Ollama model for reranking (default qwen2.5:3b)

  docstring ask "<question>" [options]
      Defaults to the measured-best retrieval configuration.
      --model <m>          answering model (default qwen2.5:3b)
      --top-k <n>          sources shown to the model (default 8)
      --no-rewrite         skip query rewriting (one fewer model call)
      --retriever <kind>   default hybrid
      --db <path>

  docstring calibrate <ground-truth.json> [options]
      sweep refusal thresholds and report true vs false refusal rates
      --k <n>              signals computed over the top n hits (default 10)
      --retriever <kind>   default hybrid
      --db <path>

  docstring eval <ground-truth.json> [options]
      --retriever vector|keyword|hybrid|random|oracle   (default vector)
      --record-baseline <path>  write this run to a baseline file
      --baseline <path>         compare against a baseline and fail on regression
      --tolerance <n>           allowed drop before failing (default 0.02)
      --candidates <n>     depth fused per source for hybrid
      --vector-weight <n>  RRF weight for vector
      --keyword-weight <n> RRF weight for keyword
      --all                include unverified questions
      --db <path>`;

function flag(args: string[], name: string): string | undefined {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
}

function num(args: string[], name: string): number | undefined {
  const v = flag(args, name);
  return v === undefined ? undefined : Number(v);
}

async function main() {
  if (command === "index") {
    const root = args.find((a) => !a.startsWith("--"));
    if (!root) throw new Error("path to the repo is required");
    const result = await runIndex(root, {
      db: flag(args, "db"),
      embedding: flag(args, "embedding"),
      includeUnparsable: args.includes("--include-unparsable"),
      dryRun: args.includes("--dry-run"),
      force: args.includes("--force"),
      gitSha: flag(args, "git-sha"),
    });
    console.log(formatSummary(result.summary));
    if (result.stats) {
      console.log(
        `  persisted ${result.persisted} chunks, skipped ${result.skipped} unchanged files\n` +
          `  index now holds ${result.stats.chunks} chunks / ${result.stats.vectors} vectors ` +
          `across ${result.stats.files} files\n`,
      );
    }
    return;
  }

  if (command === "search") {
    const query = args.find((a) => !a.startsWith("--"));
    if (!query) throw new Error('a query is required, e.g. docstring search "how does routing work"');
    const handle = await openRetriever({
      db: flag(args, "db"),
      embedding: flag(args, "embedding"),
      kind: flag(args, "retriever"),
      candidates: num(args, "candidates"),
      vectorWeight: num(args, "vector-weight"),
      keywordWeight: num(args, "keyword-weight"),
      rewriteModel: flag(args, "rewrite-model"),
      rewriteVariants: num(args, "rewrite-variants"),
      originalWeight: num(args, "original-weight"),
      rerank: args.includes("--rerank"),
      rerankModel: flag(args, "rerank-model"),
      rerankBatch: num(args, "rerank-batch"),
      rerankSnippet: num(args, "rerank-snippet"),
      rerankConcurrency: num(args, "rerank-concurrency"),
    });
    try {
      const hits = await handle.retriever.retrieve(query, Number(flag(args, "k") ?? 10));
      console.log(formatHits(query, hits));
    } finally {
      await handle.close();
    }
    return;
  }

  if (command === "ask") {
    const query = args.find((a) => !a.startsWith("--"));
    if (!query) throw new Error('a question is required, e.g. docstring ask "how does routing work"');
    const result = await runAsk(query, {
      rewrite: args.includes("--no-rewrite") ? false : undefined,
      db: flag(args, "db"),
      embedding: flag(args, "embedding"),
      retriever: flag(args, "retriever"),
      model: flag(args, "model"),
      topK: num(args, "top-k"),
      candidates: num(args, "candidates"),
      vectorWeight: num(args, "vector-weight"),
      keywordWeight: num(args, "keyword-weight"),
      rewriteModel: flag(args, "rewrite-model"),
      rewriteVariants: num(args, "rewrite-variants"),
      originalWeight: num(args, "original-weight"),
      rerank: args.includes("--rerank"),
      rerankModel: flag(args, "rerank-model"),
    });
    console.log(formatAnswer(query, result));
    if (result.kind === "refusal") process.exitCode = 2;
    return;
  }

  if (command === "calibrate") {
    const file = args.find((a) => !a.startsWith("--"));
    if (!file) throw new Error("path to ground truth json is required");
    const { observations, reports } = await runCalibrate(file, {
      db: flag(args, "db"),
      embedding: flag(args, "embedding"),
      retriever: flag(args, "retriever"),
      keywordWeight: num(args, "keyword-weight"),
      vectorWeight: num(args, "vector-weight"),
      k: num(args, "k"),
    });
    console.log(formatCalibration(observations, reports));
    return;
  }

  if (command === "eval") {
    const file = args.find((a) => !a.startsWith("--"));
    if (!file) throw new Error("path to ground truth json is required");
    const kind = flag(args, "retriever") ?? "vector";
    const onlyVerified = !args.includes("--all");
    const gt = await loadGroundTruth(file);

    let retriever;
    let close = async () => {};
    let reranker;
    if (kind === "random" || kind === "oracle") {
      const corpus = [...new Set(gt.questions.flatMap((q) => q.expectedFiles))];
      retriever = kind === "random" ? randomRetriever(corpus) : oracleRetriever(gt);
    } else {
      const handle = await openRetriever({
        db: flag(args, "db"),
        embedding: flag(args, "embedding"),
        kind,
        candidates: num(args, "candidates"),
        vectorWeight: num(args, "vector-weight"),
        keywordWeight: num(args, "keyword-weight"),
        rewrite: args.includes("--rewrite"),
        rewriteModel: flag(args, "rewrite-model"),
        rewriteVariants: num(args, "rewrite-variants"),
        originalWeight: num(args, "original-weight"),
        rerank: args.includes("--rerank"),
        rerankModel: flag(args, "rerank-model"),
        rerankBatch: num(args, "rerank-batch"),
        rerankSnippet: num(args, "rerank-snippet"),
        rerankConcurrency: num(args, "rerank-concurrency"),
      });
      retriever = handle.retriever;
      reranker = handle.reranker;
      close = handle.close;
    }

    try {
      const started = Date.now();
      const run = await runEval(gt, retriever, {
        onlyVerified,
        limit: num(args, "limit"),
        onProgress: (done, total) => {
          const elapsed = (Date.now() - started) / 1000;
          const eta = done > 0 ? (elapsed / done) * (total - done) : 0;
          process.stdout.write(
            `\r  ${done}/${total} questions, ${elapsed.toFixed(0)}s elapsed, ~${eta.toFixed(0)}s left    `,
          );
        },
      });
      process.stdout.write("\r".padEnd(70) + "\r");
      console.log(formatTable(run));
      if (reranker) {
        const t = reranker.telemetry;
        console.log(
          `  rerank calls: ${t.calls} (${t.parsed} scored, ${t.parseFailed} unparseable, ` +
            `${t.httpFailed} request errors), ${t.cached} from cache`,
        );
        if (t.parsed === 0 && t.sample) {
          console.log(`\n  the model never scored anything. first response was:\n`);
          console.log(
            t.sample
              .split("\n")
              .map((l) => `    | ${l}`)
              .join("\n"),
          );
        }
      }
      const baselinePath = flag(args, "baseline");
      const recordPath = flag(args, "record-baseline");
      const tolerance = num(args, "tolerance") ?? 0.02;

      if (recordPath) {
        await saveBaseline(toBaseline(run.aggregate, flag(args, "note")), recordPath);
        console.log(`  baseline written to ${recordPath}`);
      }

      if (baselinePath) {
        const result = compare(await loadBaseline(baselinePath), run.aggregate, tolerance);
        console.log(formatComparison(result, tolerance));
        if (!result.ok) process.exitCode = 1;
      }

      console.log(`  saved: ${await saveRun(run)}\n`);
    } finally {
      await close();
    }
    return;
  }

  console.error(USAGE);
  process.exit(1);
}

main().catch((err) => {
  console.error(`\n  ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
