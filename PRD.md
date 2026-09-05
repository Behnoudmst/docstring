# PRD: Docstring

A self-hosted RAG CLI + web tool for codebases and docs, with citation-verified
answers and confidence-based refusal. Open source, TypeScript monorepo.

## 1. Problem

Standard RAG tools return an answer with a vague "sources" list. Users can't
tell which specific claim came from which specific place, and the model will
happily answer even when retrieval quality is poor. This produces confident,
unverifiable, sometimes-wrong answers about a codebase.

## 2. Goal

Build a tool that:
- Indexes a repo's Markdown docs (v1 scope — code comments/docstrings are v2)
- Answers questions about that repo
- Attaches a `[file:line]` citation to every factual claim in the answer
- Verifies each citation actually corresponds to a retrieved chunk (no
  invented citations)
- Refuses to answer ("insufficient context") when retrieval confidence is low,
  instead of guessing

## 3. Non-goals (v1)

- No multi-repo / cross-repo search
- No code-comment/docstring parsing (Markdown docs only)
- No auth, multi-user, or hosted/SaaS mode — local/self-hosted only
- No auto-reindex on file change (manual `ingest` command is fine)
- No fine-tuning or custom embedding models

## 4. Users

Developers who want to ask questions about a codebase's documentation and
trust the answer enough to act on it without manually re-verifying every claim.

## 5. Architecture

pnpm monorepo, three packages:

```
packages/
  core/   # framework-free: ingestion, embedding, retrieval, citation logic
  cli/    # thin CLI wrapper around core (commander/yargs)
  web/    # Next.js App Router: route handlers call core directly, simple UI
```

`cli` and `web` both depend on `core` directly — no HTTP calls between them.
`core` has no framework dependencies so it stays testable and reusable.

**Stack:**
- TypeScript throughout
- Vercel AI SDK (`ai`, `@ai-sdk/anthropic` or `@ai-sdk/openai`) for generation
- Embeddings: local model via `@xenova/transformers` (e.g. `all-MiniLM-L6-v2`)
  for v1, to avoid API costs during development
- Storage: SQLite + `sqlite-vec` for v1 (zero infra); Postgres + `pgvector`
  as a documented alternative
- Markdown parsing: `remark`/`unified`
- File discovery: `fast-glob`

## 6. Functional requirements

### 6.1 Ingestion (`core`, exposed via `docstring ingest <path>`)
- Walk the given repo path for `*.md` / `*.mdx` files, skipping
  `node_modules`, `.git`, and any `.gitignore`-matched paths
- Parse each file and chunk by heading boundary (not fixed character count)
- Each chunk stores: `file_path`, `start_line`, `end_line`, `content`,
  `embedding`, `repo_id`
- Embed each chunk with the local embedding model
- Persist to SQLite (`sqlite-vec` extension for vector search)
- Re-running `ingest` on the same path should replace, not duplicate, that
  repo's existing chunks

### 6.2 Retrieval + answer generation (`core`, exposed via `docstring ask
<question>` and `POST /api/ask`)
- Embed the question with the same model used for ingestion
- Retrieve top-k (default 5) chunks by cosine similarity
- If the top result's similarity is below a configurable threshold, skip
  generation and return a refusal message: "I don't have enough context on
  this in the indexed docs."
- Otherwise, build a prompt that labels each retrieved chunk with an ID and
  instructs the model to cite `[chunk_id]` after every factual claim, and to
  omit anything not supported by the provided chunks
- Call the model via the Vercel AI SDK, streaming the response
- Post-process the streamed output: extract `[chunk_id]` citations, resolve
  each to `file_path:start_line-end_line`, and validate that every cited ID
  was actually in the retrieved set
- Any sentence with a claim but no valid citation is stripped or visibly
  flagged (not silently presented as fact) in the final output

### 6.3 CLI (`packages/cli`)
- `docstring ingest <path>` — run ingestion, print chunk count on completion
- `docstring ask "<question>" [--repo <id>]` — run retrieval + generation,
  print the answer with inline citations resolved to file paths

### 6.4 Web UI (`packages/web`, Next.js App Router)
- `app/api/ask/route.ts` — Route Handler, imports `core` directly, streams
  the response using the AI SDK's streaming helpers
- `app/page.tsx` — a single page: text input for the question, streamed
  answer display, citations rendered as clickable references showing
  file path + line range
- No auth, no persistence of chat history in v1

### 6.5 Eval harness (`packages/core/evals` or similar)
- A YAML/JSON fixture file of `{ question, expectedSourceFile }` pairs
  against a known repo
- A script that runs each question through the `ask` function, checks
  whether returned citations include the expected file, and prints a
  pass/fail summary with an overall score
- Run manually via `pnpm eval` — no CI requirement for v1

## 7. Success criteria

- `docstring ingest` on a real OSS repo's `/docs` folder completes without
  error and produces a sensible chunk count
- `docstring ask` on an in-scope question returns an answer where every
  claim has a citation that manually checks out against the source file
- An out-of-scope question triggers the refusal path, not a hallucinated
  answer
- Eval harness runs and reports a numeric score against the fixture set

## 8. Open questions for implementation

- Exact similarity threshold for refusal — start with a placeholder,
  tune empirically once eval fixtures exist
- Whether SQLite's `sqlite-vec` performs acceptably at the target repo size,
  or whether Postgres/pgvector should be the v1 default instead
- Citation format in the web UI when the indexed repo isn't hosted on
  GitHub (no line-linkable URL to point to)

## 9. Build order

1. Monorepo scaffolding (pnpm workspaces, `core`/`cli`/`web` packages)
2. Ingestion pipeline
3. Retrieval + citation enforcement
4. Confidence/refusal logic
5. CLI commands
6. Web UI
7. Eval harness
8. README, license, contributing guide

Each phase should be functionally checkpointed before moving to the next —
in particular, citation enforcement (phase 3) is the core differentiator and
should not be shortcut to reach the UI faster.