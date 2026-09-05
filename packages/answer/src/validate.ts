import type { Chunk, ScoredChunk } from "@docstring/core";

/** A retrieved chunk with the short reference the model sees in the prompt. */
export interface Source {
  ref: number;
  chunk: Chunk;
}

export interface RawClaim {
  text: string;
  citations: number[];
}

export interface RawAnswer {
  answer: string;
  claims: RawClaim[];
  confidence: number;
}

export interface ValidationError {
  kind:
    | "unknown-citation"
    | "uncited-claim"
    | "quote-not-found"
    | "empty-answer"
    | "no-claims";
  detail: string;
  claimIndex?: number;
}

export interface ValidationReport {
  ok: boolean;
  errors: ValidationError[];
  /** Fraction of claims carrying at least one valid citation. */
  citedFraction: number;
  quotesChecked: number;
}

export function toSources(hits: ScoredChunk[]): Source[] {
  return hits.map((h, i) => ({ ref: i + 1, chunk: h.chunk }));
}

/** Whitespace and quote style vary; the code identity should not depend on them. */
function normalise(text: string): string {
  return text
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Spans the model presented as literal source: backtick code spans and
 * fenced blocks. These are the claims that must be checkable character by
 * character, because a model that paraphrases inside backticks is inventing
 * code that looks quoted.
 */
export function extractQuotes(text: string): string[] {
  const quotes: string[] = [];
  for (const m of text.matchAll(/```[a-z]*\n([\s\S]*?)```/g)) {
    if (m[1]) quotes.push(m[1]);
  }
  const withoutFences = text.replace(/```[\s\S]*?```/g, " ");
  for (const m of withoutFences.matchAll(/`([^`\n]+)`/g)) {
    if (m[1]) quotes.push(m[1]);
  }
  // A bare identifier is a name, not a quotation: checking `getSubDomain`
  // produces noise without catching fabrication. Anything with structure —
  // a call, an operator, a string — is a real quote and must be verified.
  const bareIdentifier = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
  return quotes
    .map(normalise)
    .filter((q) => q.length > 0 && !bareIdentifier.test(q));
}

/**
 * The enforcement step. Asking a model to cite is a request; checking the
 * citations in code is a guarantee. Everything here is deliberately pure so
 * it can be tested without a model.
 */
export function validateAnswer(raw: RawAnswer, sources: Source[]): ValidationReport {
  const errors: ValidationError[] = [];
  const byRef = new Map(sources.map((s) => [s.ref, s.chunk]));

  if (normalise(raw.answer).length === 0) {
    errors.push({ kind: "empty-answer", detail: "the answer was empty" });
  }

  if (raw.claims.length === 0) {
    errors.push({ kind: "no-claims", detail: "no claims were returned" });
  }

  let citedClaims = 0;
  let quotesChecked = 0;

  raw.claims.forEach((claim, claimIndex) => {
    const known = claim.citations.filter((ref) => byRef.has(ref));
    const unknown = claim.citations.filter((ref) => !byRef.has(ref));

    for (const ref of unknown) {
      errors.push({
        kind: "unknown-citation",
        claimIndex,
        detail: `claim ${claimIndex + 1} cites [${ref}], which was not among the sources`,
      });
    }

    if (known.length === 0) {
      errors.push({
        kind: "uncited-claim",
        claimIndex,
        detail: `claim ${claimIndex + 1} has no valid citation: "${claim.text.slice(0, 80)}"`,
      });
    } else {
      citedClaims++;
    }

    // A quoted span must appear literally in at least one cited chunk.
    const cited = known.map((ref) => normalise(byRef.get(ref)!.content));
    for (const quote of extractQuotes(claim.text)) {
      quotesChecked++;
      if (!cited.some((content) => content.includes(quote))) {
        errors.push({
          kind: "quote-not-found",
          claimIndex,
          detail: `claim ${claimIndex + 1} quotes code that is not in its cited sources: "${quote.slice(0, 60)}"`,
        });
      }
    }
  });

  return {
    ok: errors.length === 0,
    errors,
    citedFraction: raw.claims.length === 0 ? 0 : citedClaims / raw.claims.length,
    quotesChecked,
  };
}

/** Feedback appended to the retry prompt, so the model knows what to fix. */
export function repairInstruction(report: ValidationReport): string {
  const lines = report.errors.slice(0, 6).map((e) => `- ${e.detail}`);
  return [
    `Your previous answer failed validation:`,
    ...lines,
    ``,
    `Rewrite it. Cite only the numbered sources given. Put text in backticks only`,
    `if it appears character for character in a source you cite. If the sources do`,
    `not contain the answer, say so instead of guessing.`,
  ].join("\n");
}
