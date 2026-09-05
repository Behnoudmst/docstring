import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { RetrievalFilter, Reranker, Retriever, ScoredChunk } from "@docstring/core";

/**
 * Reranking is the slow step, and evaluation runs the same questions against
 * the same candidates repeatedly. Caching by (model, query, candidate ids)
 * turns a 15-minute rerun into seconds, which is the difference between
 * iterating on the pipeline and not bothering.
 */
export class ScoreCache {
  private data: Record<string, Scored[]> = {};
  private dirty = false;

  constructor(private readonly path?: string) {
    if (!path) return;
    try {
      this.data = JSON.parse(readFileSync(path, "utf8")) as Record<string, Scored[]>;
    } catch {
      this.data = {};
    }
  }

  static key(model: string, query: string, ids: string[]): string {
    return createHash("sha256").update(`${model}\0${query}\0${ids.join(",")}`).digest("hex").slice(0, 24);
  }

  get(key: string): Scored[] | undefined {
    return this.data[key];
  }

  set(key: string, scores: Scored[]): void {
    this.data[key] = scores;
    this.dirty = true;
  }

  flush(): void {
    if (!this.path || !this.dirty) return;
    mkdirSync(dirname(this.path), { recursive: true });
    writeFileSync(this.path, JSON.stringify(this.data));
    this.dirty = false;
  }
}

/**
 * Wraps a candidate generator with a reranking pass.
 *
 * The division of labour: the base retriever is fast and optimises recall,
 * fetching wide. The reranker is slow and optimises precision, cutting deep.
 * Recall@candidates is therefore the hard ceiling on what this can return —
 * a reranker can only reorder what it was given.
 */
export class RerankingRetriever implements Retriever {
  readonly name: string;

  constructor(
    private readonly base: Retriever,
    private readonly reranker: Reranker,
    private readonly candidates = 50,
  ) {
    this.name = `${base.name}+${reranker.name}`;
  }

  async retrieve(query: string, k: number, filter?: RetrievalFilter): Promise<ScoredChunk[]> {
    const pool = await this.base.retrieve(query, Math.max(this.candidates, k), filter);
    if (pool.length <= 1) return pool.slice(0, k);
    return this.reranker.rerank(query, pool, k);
  }
}

export interface OllamaRerankerOptions {
  model?: string;
  baseUrl?: string;
  /** Candidates scored per model call. Larger is faster but degrades attention. */
  batchSize?: number;
  concurrency?: number;
  /** Characters of each chunk shown to the model. Prompt size dominates latency. */
  snippetChars?: number;
  /** Path to a JSON score cache. Reruns of the same query become instant. */
  cachePath?: string;
}

interface Scored {
  index: number;
  score: number;
}

/**
 * Small models return a different shape almost every call, especially under
 * Ollama's `format: "json"` which forces an object rather than a bare array.
 * Observed shapes, all of which must work:
 *
 *   [{"id":0,"score":7}]              the requested form
 *   {"scores":[{"id":0,"score":7}]}   wrapped in an object
 *   {"results":[7,2,5]}               positional, no ids
 *   {"0":7,"1":2}                     an id->score map
 *   [7, 2, 5]                         a bare positional array
 *
 * Being strict here means throwing away work the model actually did.
 */
export function parseScores(text: string, expected: number): Scored[] {
  const raw = extractJson(text);
  if (raw === undefined) return [];
  return normalise(raw, expected);
}

function extractJson(text: string): unknown {
  const trimmed = text.trim();
  const candidates = [trimmed];
  // Models wrap JSON in fences or prose regardless of instructions.
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced?.[1]) candidates.push(fenced[1].trim());
  const object = trimmed.match(/\{[\s\S]*\}/);
  if (object) candidates.push(object[0]);
  const array = trimmed.match(/\[[\s\S]*\]/);
  if (array) candidates.push(array[0]);

  for (const c of candidates) {
    try {
      return JSON.parse(c);
    } catch {
      /* try the next shape */
    }
  }
  return undefined;
}

function normalise(raw: unknown, expected: number): Scored[] {
  const valid = (r: Scored) =>
    Number.isInteger(r.index) && r.index >= 0 && r.index < expected && Number.isFinite(r.score);

  if (Array.isArray(raw)) {
    // [7, 2, 5] — positional scores with no ids.
    if (raw.every((v) => typeof v === "number")) {
      return raw.map((score, index) => ({ index, score: Number(score) })).filter(valid);
    }
    return raw
      .map((row, i) => {
        const r = row as Record<string, unknown>;
        const idField = r["id"] ?? r["index"] ?? r["snippet"] ?? i;
        const scoreField = r["score"] ?? r["relevance"] ?? r["rating"] ?? r["value"];
        return { index: Number(idField), score: Number(scoreField) };
      })
      .filter(valid);
  }

  if (raw && typeof raw === "object") {
    const obj = raw as Record<string, unknown>;
    // Any array-valued property: {"scores": [...]}, {"results": [...]}
    for (const value of Object.values(obj)) {
      if (Array.isArray(value)) {
        const inner = normalise(value, expected);
        if (inner.length > 0) return inner;
      }
    }
    // {"0": 7, "1": 2} — an id to score map.
    const entries = Object.entries(obj)
      .map(([k, v]) => ({ index: Number(k), score: Number(v) }))
      .filter(valid);
    if (entries.length > 0) return entries;
  }

  return [];
}

async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i]!);
    }
  });
  await Promise.all(workers);
  return out;
}

/**
 * Listwise reranking with a small local model.
 *
 * A proper cross-encoder would be better, but this needs no extra runtime
 * beyond the Ollama already required for embeddings. Candidates are scored in
 * batches: a batch that fails to parse keeps its original order rather than
 * being dropped, so a flaky model degrades toward the base ranking instead of
 * losing results.
 */
export class OllamaReranker implements Reranker {
  readonly name = "rerank";
  private model: string;
  private baseUrl: string;
  private batchSize: number;
  private concurrency: number;
  private snippetChars: number;
  private cache: ScoreCache;

  constructor(opts: OllamaRerankerOptions = {}) {
    this.model = opts.model ?? "qwen2.5:3b";
    this.baseUrl = (opts.baseUrl ?? "http://127.0.0.1:11434").replace(/\/$/, "");
    this.batchSize = opts.batchSize ?? 10;
    // Ollama serialises requests unless OLLAMA_NUM_PARALLEL is raised, so
    // concurrency above 1 buys nothing by default.
    this.concurrency = opts.concurrency ?? 1;
    this.snippetChars = opts.snippetChars ?? 400;
    this.cache = new ScoreCache(opts.cachePath);
  }

  flushCache(): void {
    this.cache.flush();
  }

  /**
   * Verify the model exists before a long run. Without this, a missing model
   * degrades silently to the base ranking and the eval reports "reranking did
   * not help" instead of "reranking never ran".
   */
  async check(): Promise<void> {
    let tags: { models?: { name: string }[] };
    try {
      const res = await fetch(`${this.baseUrl}/api/tags`);
      tags = (await res.json()) as { models?: { name: string }[] };
    } catch {
      throw new Error(`Cannot reach Ollama at ${this.baseUrl}. Is it running? Try: ollama serve`);
    }
    const names = (tags.models ?? []).map((m) => m.name);
    const present = names.some((n) => n === this.model || n.startsWith(`${this.model}:`));
    if (!present) {
      throw new Error(
        `Rerank model "${this.model}" is not installed. Run: ollama pull ${this.model}\n` +
          `Installed: ${names.join(", ") || "(none)"}`,
      );
    }
  }

  /** Counts since construction, so a run can report whether the model actually worked. */
  readonly telemetry = {
    calls: 0,
    parsed: 0,
    httpFailed: 0,
    parseFailed: 0,
    cached: 0,
    /** First unparseable response, so a shape mismatch can be seen, not guessed. */
    sample: undefined as string | undefined,
  };

  private prompt(query: string, batch: ScoredChunk[]): string {
    const entries = batch
      .map((hit, i) => {
        const c = hit.chunk;
        return [
          `### ${i}`,
          `path: ${c.path}:${c.startLine}-${c.endLine}`,
          `symbol: ${c.symbolName ?? "(none)"} (${c.symbolKind ?? "?"})`,
          "```",
          c.content.slice(0, this.snippetChars),
          "```",
        ].join("\n");
      })
      .join("\n\n");

    return [
      `You are ranking code snippets by how well they answer a question about a codebase.`,
      ``,
      `Question: ${query}`,
      ``,
      `Snippets:`,
      entries,
      ``,
      `Score each snippet 0-10 for how directly it answers the question.`,
      `10 = contains the answer. 5 = related but does not answer it. 0 = irrelevant.`,
      `Judge only what is shown. Do not assume code you cannot see.`,
      ``,
      `Reply with a JSON object only, no prose. One entry per snippet:`,
      `{"scores": [{"id": 0, "score": 7}, {"id": 1, "score": 2}]}`,
    ].join("\n");
  }

  private async scoreBatch(query: string, batch: ScoredChunk[]): Promise<Scored[]> {
    const key = ScoreCache.key(this.model, query, batch.map((b) => b.chunk.id));
    const cached = this.cache.get(key);
    if (cached) {
      this.telemetry.cached++;
      return cached;
    }
    this.telemetry.calls++;
    try {
      const res = await fetch(`${this.baseUrl}/api/generate`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: this.model,
          prompt: this.prompt(query, batch),
          stream: false,
          format: "json",
          options: { temperature: 0 },
        }),
      });
      if (!res.ok) {
        this.telemetry.httpFailed++;
        this.telemetry.sample ??= `HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`;
        return [];
      }
      const json = (await res.json()) as { response?: string };
      const scores = parseScores(json.response ?? "", batch.length);
      if (scores.length > 0) {
        this.telemetry.parsed++;
        this.cache.set(key, scores);
      } else {
        this.telemetry.parseFailed++;
        this.telemetry.sample ??= (json.response ?? "(empty response)").slice(0, 400);
      }
      return scores;
    } catch (err) {
      this.telemetry.httpFailed++;
      this.telemetry.sample ??= err instanceof Error ? err.message : String(err);
      return [];
    }
  }

  async rerank(query: string, candidates: ScoredChunk[], topK: number): Promise<ScoredChunk[]> {
    const batches: ScoredChunk[][] = [];
    for (let i = 0; i < candidates.length; i += this.batchSize) {
      batches.push(candidates.slice(i, i + this.batchSize));
    }

    const results = await mapLimit(batches, this.concurrency, async (batch) =>
      this.scoreBatch(query, batch),
    );
    this.cache.flush();

    const scored: { hit: ScoredChunk; score: number; originalRank: number }[] = [];
    batches.forEach((batch, b) => {
      const offset = b * this.batchSize;
      const byIndex = new Map(results[b]!.map((s) => [s.index, s.score]));
      batch.forEach((hit, i) => {
        const modelScore = byIndex.get(i);
        scored.push({
          hit,
          // Unscored batches fall back to the base ranking rather than vanishing.
          score: modelScore ?? -1,
          originalRank: offset + i,
        });
      });
    });

    return scored
      .sort((a, b) => b.score - a.score || a.originalRank - b.originalRank)
      .slice(0, topK)
      .map(({ hit, score }) => ({
        chunk: hit.chunk,
        score: score >= 0 ? score / 10 : hit.score,
        source: "reranked" as const,
      }));
  }
}
