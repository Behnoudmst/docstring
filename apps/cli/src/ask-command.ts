import { Answerer, OllamaClient, type Answer } from "@docstring/answer";
import { openRetriever, type RetrieverHandle } from "./search-command.js";

export interface AskOptions {
  db?: string;
  embedding?: string;
  retriever?: string;
  model?: string;
  topK?: number;
  candidates?: number;
  vectorWeight?: number;
  keywordWeight?: number;
  rerank?: boolean;
  rerankModel?: string;
  rewrite?: boolean;
  rewriteModel?: string;
  rewriteVariants?: number;
  originalWeight?: number;
}

export async function runAsk(query: string, opts: AskOptions = {}) {
  const llm = new OllamaClient(opts.model ?? "qwen2.5:3b");
  await llm.check();

  // Defaults are the measured-best configuration, not guesses:
  // hybrid fusion with keyword down-weighted to 0.3, plus query rewriting
  // with the original question weighted 1.5. See the results table in README.
  const handle: RetrieverHandle = await openRetriever({
    db: opts.db,
    embedding: opts.embedding,
    kind: opts.retriever ?? "hybrid",
    candidates: opts.candidates,
    vectorWeight: opts.vectorWeight,
    keywordWeight: opts.keywordWeight ?? 0.3,
    rerank: opts.rerank,
    rerankModel: opts.rerankModel,
    rewrite: opts.rewrite ?? true,
    rewriteModel: opts.rewriteModel,
    rewriteVariants: opts.rewriteVariants,
    originalWeight: opts.originalWeight ?? 1.5,
  });

  try {
    const hits = await handle.retriever.retrieve(query, opts.topK ?? 8);
    return await new Answerer(llm).answer(query, hits);
  } finally {
    await handle.close();
  }
}

export function formatAnswer(query: string, result: Answer): string {
  const lines = [``, `  ${query}`, ``];

  if (result.kind === "refusal") {
    lines.push(`  REFUSED: ${result.reason}`, ``);
    if (result.validation.errors.length > 0) {
      lines.push(`  validation failures:`);
      for (const e of result.validation.errors.slice(0, 5)) {
        lines.push(`    - ${e.detail}`);
      }
      lines.push(``);
    }
    if (result.sources.length > 0) {
      lines.push(`  closest sources considered:`);
      for (const s of result.sources.slice(0, 3)) {
        lines.push(
          `    [${s.ref}] ${s.chunk.path}:${s.chunk.startLine}-${s.chunk.endLine}  ${s.chunk.symbolName ?? ""}`,
        );
      }
      lines.push(``);
    }
    return lines.join("\n");
  }

  lines.push(...result.text.split("\n").map((l) => `  ${l}`), ``);

  lines.push(`  claims and citations:`);
  result.claims.forEach((claim, i) => {
    lines.push(`   ${i + 1}. ${claim.text}`);
    for (const c of claim.citations) {
      lines.push(`      -> ${c.path}:${c.startLine}-${c.endLine}  ${c.symbolName ?? ""}`);
    }
  });

  lines.push(
    ``,
    `  confidence ${result.confidence.toFixed(2)} | ` +
      `${result.attempts} attempt${result.attempts === 1 ? "" : "s"} | ` +
      `${result.validation.quotesChecked} quote${result.validation.quotesChecked === 1 ? "" : "s"} verified | ` +
      `${result.sources.length} sources`,
    ``,
  );

  return lines.join("\n");
}
