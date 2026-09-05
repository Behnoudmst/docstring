import type { Embedder } from "@docstring/core";

/** Known dimensions, so a mismatch is caught before indexing rather than after. */
export const KNOWN_DIMS: Record<string, number> = {
  "nomic-embed-text": 768,
  "mxbai-embed-large": 1024,
  "bge-m3": 1024,
  "all-minilm": 384,
  "snowflake-arctic-embed": 1024,
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
};

async function withRetry<T>(fn: () => Promise<T>, attempts = 3): Promise<T> {
  let lastError: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (i < attempts - 1) {
        await new Promise((r) => setTimeout(r, 250 * 2 ** i));
      }
    }
  }
  throw lastError;
}

/**
 * Task prefixes for asymmetric embedding models.
 *
 * nomic-embed-text documents "search_query: " and "search_document: ", and the
 * usual advice is to always apply them. Measured on this codebase they made
 * retrieval worse on every metric (recall@50 76.7% -> 72.6%, MRR 0.546 -> 0.488),
 * so they are off by default. The likely reason: these chunks are not prose.
 * They already carry a "File: / Symbol: / Imports:" header, so a document
 * prefix stacks a second framing on text the model never saw in that shape.
 *
 * Pass `prefixes` explicitly to re-enable, and re-index — the document prefix
 * is recorded with the index and a mismatch is rejected.
 */
export const TASK_PREFIXES: Record<string, { query: string; document: string }> = {
  "nomic-embed-text": { query: "", document: "" },
  "bge-m3": { query: "", document: "" },
  "mxbai-embed-large": { query: "", document: "" },
};

export interface OllamaOptions {
  model?: string;
  baseUrl?: string;
  dim?: number;
  batchSize?: number;
  /** Override the model's default task prefixes. */
  prefixes?: { query: string; document: string };
}

/**
 * Local embeddings via Ollama. Default for a tool that indexes private
 * codebases: nothing leaves the machine.
 */
export class OllamaEmbedder implements Embedder {
  readonly model: string;
  readonly dim: number;
  readonly prefixes: { query: string; document: string };
  private baseUrl: string;
  private batchSize: number;

  constructor(opts: OllamaOptions = {}) {
    this.model = opts.model ?? "nomic-embed-text";
    this.baseUrl = (opts.baseUrl ?? "http://127.0.0.1:11434").replace(/\/$/, "");
    this.dim = opts.dim ?? KNOWN_DIMS[this.model] ?? 768;
    this.batchSize = opts.batchSize ?? 16;
    const base = this.model.split(":")[0] ?? this.model;
    this.prefixes = opts.prefixes ?? TASK_PREFIXES[base] ?? { query: "", document: "" };
  }

  /** Chunks being indexed. */
  async embedDocuments(texts: string[]): Promise<number[][]> {
    return this.embed(texts.map((t) => this.prefixes.document + t));
  }

  /** A user question. Must use the query prefix, not the document one. */
  async embedQuery(text: string): Promise<number[]> {
    const [v] = await this.embed([this.prefixes.query + text]);
    return v ?? [];
  }

  /** Fails early with a readable message rather than mid-index. */
  async check(): Promise<void> {
    let tags: { models?: { name: string }[] };
    try {
      const res = await fetch(`${this.baseUrl}/api/tags`);
      tags = (await res.json()) as { models?: { name: string }[] };
    } catch {
      throw new Error(
        `Cannot reach Ollama at ${this.baseUrl}. Is it running? Try: ollama serve`,
      );
    }
    const names = (tags.models ?? []).map((m) => m.name);
    const present = names.some((n) => n === this.model || n.startsWith(`${this.model}:`));
    if (!present) {
      throw new Error(
        `Model "${this.model}" is not installed. Run: ollama pull ${this.model}\n` +
          `Installed: ${names.join(", ") || "(none)"}`,
      );
    }
    const [probe] = await this.embed(["dimension probe"]);
    if (probe && probe.length !== this.dim) {
      throw new Error(
        `Model "${this.model}" returns ${probe.length} dimensions but config says ${this.dim}. ` +
          `Set the correct dim, or the index will be built wrong.`,
      );
    }
  }

  async embed(texts: string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      const vectors = await withRetry(async () => {
        const res = await fetch(`${this.baseUrl}/api/embed`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model: this.model, input: batch }),
        });
        if (!res.ok) {
          throw new Error(`Ollama returned ${res.status}: ${await res.text()}`);
        }
        const json = (await res.json()) as { embeddings?: number[][] };
        if (!json.embeddings) throw new Error("Ollama response had no embeddings field");
        return json.embeddings;
      });
      out.push(...vectors);
    }
    return out;
  }
}

export interface OpenAIOptions {
  model?: string;
  apiKey?: string;
  baseUrl?: string;
  batchSize?: number;
}

export class OpenAIEmbedder implements Embedder {
  readonly model: string;
  readonly dim: number;
  readonly prefixes = { query: "", document: "" };

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return this.embed(texts);
  }

  async embedQuery(text: string): Promise<number[]> {
    const [v] = await this.embed([text]);
    return v ?? [];
  }

  private apiKey: string;
  private baseUrl: string;
  private batchSize: number;

  constructor(opts: OpenAIOptions = {}) {
    this.model = opts.model ?? "text-embedding-3-small";
    this.dim = KNOWN_DIMS[this.model] ?? 1536;
    this.apiKey = opts.apiKey ?? process.env.OPENAI_API_KEY ?? "";
    this.baseUrl = (opts.baseUrl ?? "https://api.openai.com/v1").replace(/\/$/, "");
    this.batchSize = opts.batchSize ?? 64;
    if (!this.apiKey) throw new Error("OPENAI_API_KEY is not set");
  }

  async check(): Promise<void> {
    const [probe] = await this.embed(["dimension probe"]);
    if (probe && probe.length !== this.dim) {
      throw new Error(`Expected ${this.dim} dimensions, got ${probe.length}`);
    }
  }

  async embed(texts: string[]): Promise<number[][]> {
    const out: number[][] = [];
    for (let i = 0; i < texts.length; i += this.batchSize) {
      const batch = texts.slice(i, i + this.batchSize);
      const vectors = await withRetry(async () => {
        const res = await fetch(`${this.baseUrl}/embeddings`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${this.apiKey}`,
          },
          body: JSON.stringify({ model: this.model, input: batch }),
        });
        if (!res.ok) throw new Error(`OpenAI returned ${res.status}: ${await res.text()}`);
        const json = (await res.json()) as { data: { embedding: number[] }[] };
        return json.data.map((d) => d.embedding);
      });
      out.push(...vectors);
    }
    return out;
  }
}

export function createEmbedder(spec: string): Embedder & { check(): Promise<void> } {
  const [provider, model] = spec.includes(":") ? spec.split(":", 2) : ["ollama", spec];
  if (provider === "openai") return new OpenAIEmbedder({ model });
  if (provider === "ollama") return new OllamaEmbedder({ model });
  throw new Error(`Unknown embedding provider "${provider}". Use ollama:<model> or openai:<model>.`);
}
