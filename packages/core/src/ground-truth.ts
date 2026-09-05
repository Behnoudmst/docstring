export interface GroundTruthQuestion {
  id: string;
  category: "locational" | "explanatory" | "cross-file" | "behavioural";
  difficulty: "easy" | "medium" | "hard";
  question: string;
  answerable: boolean;
  verified?: boolean;
  nearMiss?: boolean;
  refusalReason?: string;
  expectedFiles: string[];
  expectedSymbols?: string[];
  answerSketch?: string[];
  tags?: string[];
  notes?: string;
}

export interface GroundTruth {
  meta: Record<string, unknown>;
  questions: GroundTruthQuestion[];
}

/**
 * Fails loudly on the mistakes that silently corrupt measurement:
 * duplicate ids, answerable questions with no expected files, and
 * unanswerable questions that were given expected files by accident.
 */
export function validateGroundTruth(gt: GroundTruth): string[] {
  const problems: string[] = [];
  const seen = new Set<string>();

  for (const q of gt.questions) {
    if (seen.has(q.id)) problems.push(`duplicate id: ${q.id}`);
    seen.add(q.id);

    if (q.answerable && q.expectedFiles.length === 0) {
      problems.push(`${q.id}: answerable but has no expectedFiles`);
    }
    if (!q.answerable && q.expectedFiles.length > 0) {
      problems.push(`${q.id}: unanswerable but has expectedFiles`);
    }
  }

  const answerable = gt.questions.filter((q) => q.answerable);
  if (answerable.length === 0) problems.push("no answerable questions");
  if (gt.questions.length - answerable.length === 0) {
    problems.push("no unanswerable questions — refusal cannot be measured");
  }
  return problems;
}

/** Questions you have actually confirmed against the source. */
export function verifiedOnly(gt: GroundTruth): GroundTruthQuestion[] {
  return gt.questions.filter((q) => q.verified === true || !q.answerable);
}
