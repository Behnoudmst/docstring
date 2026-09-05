# Docstring

Self-hosted question answering over a codebase, with citations verified in code and refusal when the sources don't support an answer.

Ask a question, get an answer where every claim points at a file and line range — and where any code shown in backticks has been checked character-for-character against the source it cites. If the citations don't hold up, you get a refusal instead of a plausible-sounding answer.

Everything runs locally. Your code never leaves the machine.

```
$ pnpm ask "how does menu styling work"

  The menu styling is implemented through a combination of TypeScript interfaces and React components in the codebase. The `MenuStyleColorSettings` component handles the color settings for the menu, allowing users to change colors via a picker interface. This component receives props including `menuStyle` (which presumably holds the current style) and an `onColorChange` function that updates this state when a user selects a new color from the picker. The `MenuStyleColorPickerItem` component is responsible for rendering each individual color field, displaying its label and allowing users to change the color by clicking on it. This color can be updated via the `onChange` prop passed down from the parent component (`onColorChange`).

  claims and citations:
   1. The MenuStyleColorSettings component handles the color settings for the menu.
      -> src/components/MenuStyleColorSettings.tsx:28-47  MenuStyleColorSettings
   2. The MenuStyleColorPickerItem component is responsible for rendering each individual color field, displaying its label and allowing users to change the color by clicking on it.
      -> src/components/MenuStyleColorPickerItem.tsx:15-44  MenuStyleColorPickerItem

  confidence 1.00 | 1 attempt | 0 quotes verified | 8 sources
```

And a refusal, on a question the codebase cannot answer:

```
$ pnpm ask "which file defines the MongoDB schema for orders"

 REFUSED: The MongoDB schema for orders is not defined in any of the provided codebases.

  closest sources considered:
    [1] src/app/[locale]/(root)/profile/public-orders/page.tsx:14-135  PublicOrders
    [2] src/actions.ts:22-40  createOrderAction
    [3] src/hooks/useOrders.ts:10-24  useTodayOrders
```

---

**What this is not:** a hosted service, a Copilot competitor, or production-hardened
software. It is a working tool and a record of what actually moves retrieval
quality on a real codebase.

## Why this exists

RAG systems fail quietly. You get a fluent, confident, wrong answer with no way to tell whether the problem was chunking, embedding, retrieval, ranking, or the prompt. Most projects build the whole pipeline, get mediocre results, and then tune by intuition.

This one was built measurement-first. A ground-truth set of 80 questions came before any retrieval code, and every design decision after that is a recorded delta against a baseline.

Eight decisions were measured. Three helped. Four made things worse, and every one of those four is something standard RAG advice recommends. One is unresolved.

## Results

Measured on a 181-file Next.js frontend: 632 chunks, median 400 characters. 62 hand-verified answerable questions plus 15 unanswerable ones.

| retriever | recall@5 | recall@10 | recall@20 | recall@50 | MRR |
|---|---|---|---|---|---|
| random (floor) | ~2% | ~4% | ~6% | — | 0.03 |
| keyword only (BM25) | 21.1% | 29.6% | 42.1% | 56.0% | 0.249 |
| vector only | **50.3%** | **61.8%** | 64.9% | — | **0.586** |
| hybrid, equal weight | 42.6% | 54.8% | 67.1% | **77.3%** | 0.502 |
| hybrid, keyword 0.5 | 42.6% | 56.9% | 72.4% | 76.7% | 0.526 |
| hybrid, keyword 0.3 | 49.7% | 58.7% | **72.4%** | 76.7% | 0.546 |
| hybrid + LLM rerank | 34.3% | 46.6% | 61.7% | 77.3% | 0.415 |
| hybrid + rewrite, ow 1.0 | 45.3% | **62.6%** | 71.5% | **79.0%** | 0.539 |
| **hybrid + rewrite, ow 1.5** | 44.0% | **62.6%** | 71.6% | 77.7% | **0.549** |
| hybrid + rewrite, ow 2.0 | 46.9% | **62.6%** | 71.6% | 76.9% | 0.540 |
| oracle (ceiling) | 100% | 100% | 100% | 100% | 1.000 |

The shipped defaults are the bottom-highlighted row — hybrid fusion with keyword
down-weighted to 0.3, plus query rewriting with the original question weighted
1.5. `pnpm ask` uses them with no flags.

Note that recall@10 is identical to three significant figures across all three
rewrite weights. RRF's `1/(60 + rank)` is flat enough that doubling a weight
barely reorders the top ten — robust by design, and hard to tune as a result.

The random and oracle rows exist to prove the metrics are correct before trusting them. A retriever that cheats must score 1.0 on everything; one that ignores the query must score near zero. Both were verified before any real retriever was written.

## What I measured

### Structural chunking beats fixed-size splitting

Chunks are cut on syntax-tree boundaries with tree-sitter, not every N characters. A function stays whole, with its file path, symbol name and imports prepended to the embedded text but kept out of the cited content.

Getting this right on a React codebase took three rounds. A large component initially became one 20,000-character chunk, because the splitter only handled classes. Splitting it into an outline, the individual handlers, the hook bodies, and the render body took chunks-per-file from 1.8 to 3.5 and eliminated every chunk over 6,000 characters.

The second bug was worse: `useEffect` and `useCallback` bodies were summarised to their first line in the outline and dropped, so that code existed in no chunk at all and could never be retrieved or cited. There is now a test asserting that every meaningful source line lands in some chunk.

Recognising `forwardRef`-wrapped components and exported data literals recovered another 22 files that were being indexed anonymously.

### Hybrid retrieval fixes coverage; weighting fixes ranking

Vector search finds code that *means* the same thing. Keyword search finds code that is literally *named* the thing. Fused with Reciprocal Rank Fusion — by rank, not score, since cosine and BM25 aren't comparable numbers — total misses dropped from 8 to 3 and recall@50 rose to 77.3%.

But equal-weight RRF cost 7 points of recall@10. It gives keyword's rank-1 result the same weight as vector's, and keyword has less than half the MRR, so its confident-but-wrong hits displace correct ones. Down-weighting keyword to 0.3 recovers the top-5 ranking while keeping the coverage.

### Query rewriting closes the vocabulary gap

A question and the code that answers it are written in different vocabularies.
Measured by hand:

    "how does the app decide which restaurant menu to show"
      -> ten results scoring 0.518 down to 0.502, none of them the answer

    "subdomain host header routing"
      -> subdomainMiddleware, getSubDomain, middleware at ranks 1, 2, 3

The answer is indexed and retrievable. The natural-language phrasing cannot
reach it. Worse, in a repository where every file concerns menus, the word
"menu" carries no discriminative signal — every chunk matches it weakly and
nothing stands out. The top hit for that question was a page title reading
"Free Qr restaurant menu creator": the best surface match, and useless.

Sending the question through a model first, turning it into likely identifiers
and framework terms, then fusing those rankings with the original, lifted
recall@10 from 58.7% to 62.6% and recall@50 to 79.0%. This attacks the query
side of the gap the way header enrichment attacks the document side.

The original question always participates in the fusion. Over-weighting it lets
its own noise outrank the rewrite's real hits, defeating the purpose; including
it stops a bad paraphrase from discarding a good ranking. Rewriting is not free:
it fixed q061 (an expired-subscription question) and broke q057 (a printer
question that had been working).

### Negative: reranking with a small local model made things worse

The standard advice is that reranking is the single biggest quality jump available. That advice comes from work using purpose-built cross-encoders.

Substituting a general instruct model — the thing anyone self-hosting will actually reach for — cost 8 points of recall@5 and recall@10 against its own input, at both 1.5b and 3b. recall@50 was unchanged at 77.3%, confirming it reorders rather than losing candidates. Scale didn't help, so this is not a capacity problem: listwise relevance scoring is a poor fit for models not trained on it.

Purpose-built cross-encoders like `bge-reranker-v2-m3` are untested here because they don't run through Ollama, and adding a second runtime undercuts the self-hosting story. `--rerank` remains available and off by default.

### Negative: task prefixes made things worse

`nomic-embed-text` documents `search_query: ` and `search_document: ` prefixes, and the usual advice is to always apply them.

| hybrid, keyword 0.3 | recall@5 | recall@10 | recall@20 | recall@50 | MRR |
|---|---|---|---|---|---|
| no prefixes | **49.7%** | **58.7%** | **72.4%** | **76.7%** | **0.546** |
| with prefixes | 45.4% | 57.0% | 65.3% | 72.6% | 0.488 |

Worse on every metric, and total misses doubled. The likely cause is that these chunks aren't prose — they already carry a `File: / Symbol: / Imports:` header, so a document prefix stacks a second framing onto text the model never saw in that shape.

Off by default, mechanism retained. The document prefix is recorded with the index and a mismatch is rejected, because vectors built one way cannot be queried the other way.

### Negative: indexing non-code files costs more than it gains

Including CSS, Markdown and text via the unstructured fallback fixed one question and cost 5.3 points of recall@10. Unstructured chunks have no symbol names and match many queries weakly, displacing real answers. The right fix is structure-aware chunking for those formats, not dumping them into the same index.

### Negative: retrieval scores cannot drive refusal

Four scale-free signals — top score, concentration, gap ratio, decay — swept against 62 answerable and 15 unanswerable questions:

| retriever | best signal | true refusal | false refusal | Youden |
|---|---|---|---|---|
| hybrid | topScore | 80% | 37% | 0.429 |
| vector | topScore | 80% | 37% | 0.429 |

Identical on both, despite RRF and cosine living on entirely different scales. Mean top score was 0.0203 for answerable questions against 0.0191 for unanswerable ones — the distributions almost completely overlap. `gapRatio` was *anti*-correlated: unanswerable questions produced a *larger* gap between first and second result, plausibly because a near-miss question has one obvious wrong match sitting clear of everything else while a genuine multi-file question spreads its score.

Refusing 37% of answerable questions to catch 80% of unanswerable ones is not a usable trade. Embedding similarity measures whether a chunk is *about* a question, not whether it *contains the answer*, and 11 of the 15 unanswerable questions are deliberate near-misses built to make those diverge.

Note also that Youden weights both error types equally, which a product shouldn't: users feel every false refusal and never see a correct one. At a lower threshold the trade is 53% / 11% — worse by Youden, better in practice.

Refusal currently fires on citation-validation failure and on the model self-reporting zero confidence. A groundedness check that reads the retrieved code is implemented but not yet measured.

## Citation enforcement

The part that isn't standard RAG.

The model receives the top 8 chunks as numbered sources and returns claims, each citing source numbers. Then, **in code**:

- every cited number must be one of the sources actually retrieved
- every claim must carry at least one valid citation
- every span in backticks must appear character-for-character in a chunk **that claim cites** — a real quote attributed to the wrong file fails

A failure sends the specific errors back to the model for one retry. A second failure produces a refusal.

Asking a model to cite is a request. Checking in code is a guarantee.

Bare identifiers are exempt: `` `middleware` `` is a name, not a quotation. Anything with structure — `` `host.split(".")` ``, `` `x = 1` `` — is verified.

**Known limit:** validation catches fabricated citations. It does not catch a *real* citation to a plausible-but-wrong chunk. That is what refusal is for, and refusal is the weakest part of the system.

## Quick start

Requires **Node 22.5+** (for `node:sqlite`) and [Ollama](https://ollama.com).

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b

pnpm install
pnpm -r build

pnpm index /path/to/your/repo
pnpm ask "how does authentication work"
pnpm run search "getSubDomain"
```

No native compilation anywhere: SQLite ships with Node, and sqlite-vec provides a prebuilt loadable extension.

## Commands

All commands run through pnpm from the repo root. There is no global binary
yet — `pnpm link --global` from `apps/cli` would give you one, but cloning and
running is the intended path for now.

```
pnpm index <repo-path>
    --db <path>            index file (default .docstring/index.db)
    --embedding <spec>     ollama:<model> or openai:<model>
    --dry-run              chunk only, no model needed — use to tune the chunker
    --force                re-embed everything, ignoring content hashes
    --include-unparsable   index css/md/txt via the text fallback

pnpm run search "<query>"      # note: `pnpm run`, not `pnpm search`
    --k <n>                results to show (default 10)
    --retriever vector|keyword|hybrid
    --keyword-weight <n>   RRF weight (default 1)
    --rewrite              rewrite the question into code vocabulary first
    --original-weight <n>  RRF weight on the original question (default 1)
    --rerank               rerank candidates with a local model

pnpm ask "<question>"
    Defaults to the measured-best retrieval configuration above.
    --model <m>            answering model (default qwen2.5:3b)
    --top-k <n>            sources shown to the model (default 8)
    --no-rewrite           skip query rewriting (one fewer model call)

pnpm eval <ground-truth.json>
    --retriever <kind>     including random and oracle for calibration
    --record-baseline <p>  write this run to a baseline file
    --baseline <p>         compare and exit non-zero on regression
    --tolerance <n>        allowed drop before failing (default 0.02)
    --limit <n>            evaluate a spread subset

pnpm calibrate <ground-truth.json>
    sweep refusal thresholds, report true vs false refusal rates
```

Re-running `pnpm index` only embeds files whose content hash changed.

`pnpm search` is pnpm's own command and will query the npm registry, so the
search command needs `pnpm run search`.

## Evaluation

`evals/ground-truth-*.json` holds 80 questions written by hand against a real repository. For each answerable one, the files that *must* be retrieved. Fifteen are unanswerable, eleven of those deliberate near-misses — questions the codebase almost answers.

```json
{
  "id": "q068",
  "question": "Which file defines the MongoDB schema for orders?",
  "answerable": false,
  "nearMiss": true,
  "refusalReason": "No database layer here. src/types/types.ts holds TS types and is a very strong distractor."
}
```

That question set is the most valuable artifact in the repository. It took a day to write and every number depends on it.

Retrieval quality is a build-failing test. `.github/workflows/ci.yml` installs Ollama, indexes a fixture repo, evaluates, and compares against a recorded baseline — failing the build if recall or MRR drops beyond tolerance, naming which metric moved and which questions newly broke.

Comparing runs over different question counts is refused rather than silently misleading.

## How it works

**Indexing.** Walk the repo, skipping dependencies, build output, lockfiles and binaries. Parse each file with tree-sitter — TSX and TypeScript are separate grammars, routed by extension. Emit a chunk per function, class, interface, type, component or exported data literal, splitting oversized ones structurally. Prepend path, symbol and imports to the embedded text. Embed and store in SQLite with sqlite-vec for vectors and FTS5 for keywords.

**Querying.** Embed the question, search vectors and full text in parallel, fuse by Reciprocal Rank Fusion, optionally rerank, take the top 8.

**Answering.** Generate structured claims with citations, validate them in code, retry once on failure, refuse on the second.

```
packages/core        types, ground-truth schema, retrieval metrics
packages/chunker     tree-sitter parsing, structural chunking, repo walking
packages/store       ChunkStore (sqlite-vec + FTS5, and in-memory), embedders
packages/retrieval   vector, keyword, hybrid, reranking — all one interface
packages/answer      prompting, citation validation, refusal signals
apps/cli             index, search, ask, eval, calibrate
evals                ground truth, baseline, saved runs
```

`packages/core` has no database or HTTP dependencies, which is what lets the eval harness drive retrieval directly.

## Limitations

- **Refusal is weak.** Score-based signals don't separate answerable from unanswerable questions. The groundedness check is built but unmeasured.
- **Recall@10 is 62.6%**, not 90%. Roughly one question in three doesn't get its answer into the top 10.
- **Behavioural questions are the hardest.** "What happens if X fails" has no lexical or semantic anchor when the answer is a `catch` block, or when there is no handling at all. Query rewriting helps some of them and hurts others.
- **Vocabulary gaps are unsolvable by tuning.** One question asks about "bot verification"; the file is `Turnstil.tsx` — misspelled, and never uses those words. Neither retriever finds it.
- **One repository, one language.** Python and Go grammars are wired up but untested against real ground truth.
- **`node:sqlite` is experimental** and prints a warning on start.

## Requirements

Node 22.5+, pnpm, Ollama. Roughly 3 MB of index per 1,000 chunks.

Delete `.docstring/` and re-index to start clean — required when changing embedding model, since the vector dimension is fixed at index creation.
