import type { ScoredChunk } from "@docstring/core";
import {
  repairInstruction,
  toSources,
  validateAnswer,
  type RawAnswer,
  type Source,
  type ValidationReport,
} from "./validate.js";

export interface Citation {
  ref: number;
  chunkId: string;
  path: string;
  startLine: number;
  endLine: number;
  symbolName: string | null;
}

export interface Claim {
  text: string;
  citations: Citation[];
}

export interface Answer {
  kind: "answer" | "refusal";
  text: string;
  claims: Claim[];
  confidence: number;
  sources: Source[];
  validation: ValidationReport;
  attempts: number;
  /** Present when kind is "refusal". */
  reason?: string;
}

export function buildPrompt(query: string, sources: Source[], snippetChars = 1200): string {
  const blocks = sources
    .map(({ ref, chunk }) =>
      [
        `[${ref}] ${chunk.path}:${chunk.startLine}-${chunk.endLine}`,
        `symbol: ${chunk.symbolName ?? "(none)"} (${chunk.symbolKind ?? "?"})`,
        "```" + chunk.language,
        chunk.content.slice(0, snippetChars),
        "```",
      ].join("\n"),
    )
    .join("\n\n");

  return [
    `Answer a question about a codebase using only the sources below.`,
    ``,
    `Question: ${query}`,
    ``,
    `Sources:`,
    blocks,
    ``,
    `Rules:`,
    `- Use only these sources. Do not use knowledge of other codebases.`,
    `- Break your answer into claims. Every claim must cite the source numbers it came from.`,
    `- Put text in backticks only if it appears character for character in a cited source.`,
    `- If the sources do not answer the question, set confidence to 0 and say what is missing.`,
    ``,
    `Reply with a JSON object only:`,
    `{`,
    `  "answer": "the full prose answer",`,
    `  "claims": [{"text": "one statement", "citations": [1, 3]}],`,
    `  "confidence": 0.0 to 1.0`,
    `}`,
  ].join("\n");
}

export interface LlmClient {
  readonly model: string;
  complete(prompt: string): Promise<string>;
  check?(): Promise<void>;
}

export class OllamaClient implements LlmClient {
  constructor(
    readonly model = "qwen2.5:3b",
    private readonly baseUrl = "http://127.0.0.1:11434",
  ) {
    this.baseUrl = this.baseUrl.replace(/\/$/, "");
  }

  async check(): Promise<void> {
    let tags: { models?: { name: string }[] };
    try {
      const res = await fetch(`${this.baseUrl}/api/tags`);
      tags = (await res.json()) as { models?: { name: string }[] };
    } catch {
      throw new Error(`Cannot reach Ollama at ${this.baseUrl}. Is it running? Try: ollama serve`);
    }
    const names = (tags.models ?? []).map((m) => m.name);
    if (!names.some((n) => n === this.model || n.startsWith(`${this.model}:`))) {
      throw new Error(
        `Model "${this.model}" is not installed. Run: ollama pull ${this.model}\n` +
          `Installed: ${names.join(", ") || "(none)"}`,
      );
    }
  }

  async complete(prompt: string): Promise<string> {
    const res = await fetch(`${this.baseUrl}/api/generate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: this.model,
        prompt,
        stream: false,
        format: "json",
        options: { temperature: 0 },
      }),
    });
    if (!res.ok) throw new Error(`Ollama returned ${res.status}: ${await res.text()}`);
    const json = (await res.json()) as { response?: string };
    return json.response ?? "";
  }
}

export function parseAnswer(text: string): RawAnswer | null {
  const candidates: string[] = [text.trim()];
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenced?.[1]) candidates.push(fenced[1].trim());
  const object = text.match(/\{[\s\S]*\}/);
  if (object) candidates.push(object[0]);

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as Record<string, unknown>;
      const answer = typeof parsed["answer"] === "string" ? parsed["answer"] : "";
      const rawClaims = Array.isArray(parsed["claims"]) ? parsed["claims"] : [];
      const claims = rawClaims
        .map((c) => {
          const row = c as Record<string, unknown>;
          const claimText = typeof row["text"] === "string" ? row["text"] : "";
          const rawCitations = row["citations"] ?? row["sources"] ?? row["refs"];
          const citations = Array.isArray(rawCitations)
            ? rawCitations.map(Number).filter(Number.isInteger)
            : [];
          return { text: claimText, citations };
        })
        .filter((c) => c.text.length > 0);
      const confidence = Number(parsed["confidence"]);
      return {
        answer,
        claims,
        confidence: Number.isFinite(confidence) ? Math.max(0, Math.min(1, confidence)) : 0,
      };
    } catch {
      /* try the next shape */
    }
  }
  return null;
}

function toCitations(refs: number[], sources: Source[]): Citation[] {
  const byRef = new Map(sources.map((s) => [s.ref, s.chunk]));
  return refs
    .filter((ref) => byRef.has(ref))
    .map((ref) => {
      const c = byRef.get(ref)!;
      return {
        ref,
        chunkId: c.id,
        path: c.path,
        startLine: c.startLine,
        endLine: c.endLine,
        symbolName: c.symbolName,
      };
    });
}

export interface AnswererOptions {
  snippetChars?: number;
  maxSources?: number;
  /** Retries after a failed validation. One is usually enough. */
  maxRetries?: number;
}

/**
 * Generates an answer, then verifies its citations in code and retries once
 * with the failures attached. A second failure produces a refusal rather than
 * an unverified answer: an answer whose citations do not check out is worse
 * than no answer, because it looks trustworthy.
 */
export class Answerer {
  private snippetChars: number;
  private maxSources: number;
  private maxRetries: number;

  constructor(
    private readonly llm: LlmClient,
    opts: AnswererOptions = {},
  ) {
    this.snippetChars = opts.snippetChars ?? 1200;
    this.maxSources = opts.maxSources ?? 8;
    this.maxRetries = opts.maxRetries ?? 1;
  }

  async answer(query: string, hits: ScoredChunk[]): Promise<Answer> {
    const sources = toSources(hits.slice(0, this.maxSources));

    if (sources.length === 0) {
      return {
        kind: "refusal",
        text: "",
        claims: [],
        confidence: 0,
        sources,
        validation: { ok: false, errors: [], citedFraction: 0, quotesChecked: 0 },
        attempts: 0,
        reason: "nothing was retrieved for this question",
      };
    }

    const base = buildPrompt(query, sources, this.snippetChars);
    let lastReport: ValidationReport = {
      ok: false,
      errors: [],
      citedFraction: 0,
      quotesChecked: 0,
    };
    let attempts = 0;
    let prompt = base;

    while (attempts <= this.maxRetries) {
      attempts++;
      let raw: RawAnswer | null;
      try {
        raw = parseAnswer(await this.llm.complete(prompt));
      } catch (err) {
        return this.refuse(
          sources,
          attempts,
          lastReport,
          err instanceof Error ? err.message : "the model call failed",
        );
      }

      if (!raw) {
        lastReport = {
          ok: false,
          errors: [{ kind: "empty-answer", detail: "the model did not return valid JSON" }],
          citedFraction: 0,
          quotesChecked: 0,
        };
        prompt = `${base}\n\n${repairInstruction(lastReport)}`;
        continue;
      }

      // The model itself reporting no confidence is a refusal, not a failure.
      if (raw.confidence === 0) {
        return this.refuse(
          sources,
          attempts,
          { ok: true, errors: [], citedFraction: 0, quotesChecked: 0 },
          raw.answer || "the sources do not contain the answer",
        );
      }

      const report = validateAnswer(raw, sources);
      lastReport = report;

      if (report.ok) {
        return {
          kind: "answer",
          text: raw.answer,
          claims: raw.claims.map((c) => ({
            text: c.text,
            citations: toCitations(c.citations, sources),
          })),
          confidence: raw.confidence,
          sources,
          validation: report,
          attempts,
        };
      }

      prompt = `${base}\n\n${repairInstruction(report)}`;
    }

    return this.refuse(
      sources,
      attempts,
      lastReport,
      `citations could not be verified after ${attempts} attempts`,
    );
  }

  private refuse(
    sources: Source[],
    attempts: number,
    validation: ValidationReport,
    reason: string,
  ): Answer {
    return {
      kind: "refusal",
      text: "",
      claims: [],
      confidence: 0,
      sources,
      validation,
      attempts,
      reason,
    };
  }
}
