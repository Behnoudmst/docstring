import type { RetrievalFilter, Retriever, ScoredChunk } from "@docstring/core";
import { fuseRrf } from "./hybrid-retriever.js";

/**
 * A question and the code that answers it are written in different vocabularies.
 * "How does the app decide which restaurant menu to show" and
 * `subdomainMiddleware(req)` share no terms, and in a codebase where every file
 * is about menus, the word "menu" carries no discriminative signal at all —
 * every chunk matches it weakly and nothing stands out.
 *
 * Rewriting the question into likely identifiers and technical terms before
 * embedding attacks the query side of that gap, the way header enrichment
 * attacks the document side.
 */
export interface QueryRewriter {
  readonly name: string;
  rewrite(query: string): Promise<string[]>;
}

export interface OllamaRewriterOptions {
  model?: string;
  baseUrl?: string;
  /** Rewrites to generate. Each costs a retrieval pass. */
  variants?: number;
  cache?: Map<string, string[]>;
}

export function parseVariants(text: string, limit: number): string[] {
  const candidates = [text.trim()];
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced?.[1]) candidates.push(fenced[1].trim());
  const object = text.match(/\{[\s\S]*\}/);
  if (object) candidates.push(object[0]);
  const array = text.match(/\[[\s\S]*\]/);
  if (array) candidates.push(array[0]);

  for (const candidate of candidates) {
    try {
      const parsed: unknown = JSON.parse(candidate);
      const list = Array.isArray(parsed)
        ? parsed
        : Object.values(parsed as Record<string, unknown>).find(Array.isArray);
      if (!Array.isArray(list)) continue;
      const out = list
        .filter((v): v is string => typeof v === "string")
        .map((v) => v.trim())
        .filter((v) => v.length > 0)
        .slice(0, limit);
      if (out.length > 0) return out;
    } catch {
      /* try the next shape */
    }
  }
  return [];
}

export class OllamaQueryRewriter implements QueryRewriter {
  readonly name = "rewrite";
  private model: string;
  private baseUrl: string;
  private variants: number;
  private cache: Map<string, string[]>;

  constructor(opts: OllamaRewriterOptions = {}) {
    this.model = opts.model ?? "qwen2.5:3b";
    this.baseUrl = (opts.baseUrl ?? "http://127.0.0.1:11434").replace(/\/$/, "");
    this.variants = opts.variants ?? 2;
    this.cache = opts.cache ?? new Map();
  }

  private prompt(query: string): string {
    return [
      `Rewrite a question about a codebase into search queries a code search engine would match.`,
      ``,
      `Question: ${query}`,
      ``,
      `Produce ${this.variants} short queries using the words that would appear in the code itself:`,
      `likely function and variable names, framework terms, HTTP and routing vocabulary.`,
      `Drop words that describe the product rather than the implementation.`,
      ``,
      `Example: "how does the app decide which restaurant menu to show"`,
      `becomes ["subdomain host header middleware rewrite", "resolve venue from request hostname"].`,
      ``,
      `Reply with a JSON object only: {"queries": ["...", "..."]}`,
    ].join("\n");
  }

  async rewrite(query: string): Promise<string[]> {
    const cached = this.cache.get(query);
    if (cached) return cached;
    try {
      const res = await fetch(`${this.baseUrl}/api/generate`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: this.model,
          prompt: this.prompt(query),
          stream: false,
          format: "json",
          options: { temperature: 0 },
        }),
      });
      if (!res.ok) return [];
      const json = (await res.json()) as { response?: string };
      const variants = parseVariants(json.response ?? "", this.variants);
      if (variants.length > 0) this.cache.set(query, variants);
      return variants;
    } catch {
      // A failed rewrite must not lose the original question.
      return [];
    }
  }
}

export interface RewritingOptions {
  /** RRF constant for fusing the original and rewritten rankings. */
  k?: number;
  /**
   * Weight on the original question relative to each rewrite. Defaults to 1:
   * rewriting exists to rescue queries the original phrasing cannot answer, so
   * over-weighting the original lets its noise outrank the rewrite's real hits
   * and defeats the purpose. The original still participates, which is what
   * stops a bad paraphrase from throwing away a good ranking.
   */
  originalWeight?: number;
  candidates?: number;
}

/**
 * Retrieves for the original question and each rewrite, then fuses.
 *
 * The original always participates: a rewrite can drift, and fusing keeps a
 * good original ranking from being thrown away by a bad paraphrase.
 */
export class RewritingRetriever implements Retriever {
  readonly name: string;
  private k: number;
  private originalWeight: number;
  private candidates: number;

  constructor(
    private readonly base: Retriever,
    private readonly rewriter: QueryRewriter,
    opts: RewritingOptions = {},
  ) {
    this.name = `${base.name}+${rewriter.name}`;
    this.k = opts.k ?? 60;
    this.originalWeight = opts.originalWeight ?? 1;
    this.candidates = opts.candidates ?? 50;
  }

  async retrieve(query: string, k: number, filter?: RetrievalFilter): Promise<ScoredChunk[]> {
    const depth = Math.max(this.candidates, k);
    const variants = await this.rewriter.rewrite(query);

    const rankings = await Promise.all([
      this.base.retrieve(query, depth, filter).then((results) => ({
        results,
        weight: this.originalWeight,
      })),
      ...variants.map((v) =>
        this.base.retrieve(v, depth, filter).then((results) => ({ results, weight: 1 })),
      ),
    ]);

    return fuseRrf(rankings, this.k).slice(0, k);
  }
}
