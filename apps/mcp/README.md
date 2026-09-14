# docstring-mcp

Semantic code search and cited answers over your codebase, for coding agents.
Runs entirely locally — no code leaves your machine.

## Setup

One prerequisite: [Ollama](https://ollama.com) with an embedding model.

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b       # only needed for ask_codebase
```

Install the server, then find its absolute path:

```bash
npm install -g docstring-mcp
which docstring-mcp
```

**VS Code / Copilot** — `.vscode/mcp.json` in the repo you want to search:

```json
{
  "servers": {
    "docstring": {
      "type": "stdio",
      "command": "/absolute/path/from/which/docstring-mcp"
    }
  }
}
```

**Claude Desktop / Cursor** — same, but the top-level key is `mcpServers` and
`type` is inferred rather than required.

Restart your editor after adding the config. Saving the file is often not
enough to make it reload.

Use an absolute path rather than `npx docstring-mcp`. Editors launched from
Finder or the Dock do not inherit your shell PATH, so a version manager like
nvm, fnm or volta puts `npx` somewhere the editor cannot see, and the server
fails to start with `spawn npx ENOENT`.

That is the whole configuration. The server detects the repository from the
directory your editor launches it in, and keeps the index at
`<repo>/.docstring/index.db`. Add that to `.gitignore`.

## Use

Ask your agent to index once:

> index this repo with docstring

Then ask questions normally. The agent calls `search_code` when it needs to find
code by description, and `ask_codebase` when it wants a cited explanation.

## Tools

- **`search_code(query, k?, path_prefix?)`** — chunks with paths and line ranges
- **`ask_codebase(question, top_k?)`** — a cited answer, or a refusal
- **`index_repo(force?)`** — build or refresh the index; unchanged files are skipped
- **`index_status()`** — what is indexed, and with which model

## What it is good at, and what it is not

Semantic search finds code by description when you do not know the identifier —
"where is rate limiting" in a codebase that calls it `throttle`. That is the case
it exists for.

**Grep is better when you know the exact string.** It is faster, exact, and never
stale. This tool measured **62.6% recall@10** against a hand-written 62-question
ground truth: roughly one question in three does not get its answer into the top
ten. A miss is inconclusive, not proof of absence, and both tool descriptions say
so to the agent.

Citations are verified in code rather than requested in a prompt: every cited
chunk must be one that was actually retrieved, and any quoted code must appear
character for character in the source it cites. When validation fails twice, the
tool refuses instead of returning an answer that looks trustworthy.

The full method, results table, and four measured negative results are in the
[main repository](https://github.com/Behnoudmst/docstring).

## Configuration

Everything is optional.

| Variable | Default |
|---|---|
| `DOCSTRING_DB` | `<repo>/.docstring/index.db` |
| `DOCSTRING_REPO` | nearest ancestor containing `.git` |
| `DOCSTRING_EMBEDDING` | `ollama:nomic-embed-text` |
| `DOCSTRING_MODEL` | `qwen2.5:3b` |

### Where to put them

In the `env` block of your editor's MCP config — the same file where you
registered the server. **A `.env` file in your project will not work.** Your
editor spawns this server as a subprocess, so it only sees the variables that
config hands it; it never reads `.env`, and an unset variable silently falls
back to the default above.

**VS Code / Copilot** — `.vscode/mcp.json`:

```json
{
  "servers": {
    "docstring": {
      "type": "stdio",
      "command": "/absolute/path/from/which/docstring-mcp",
      "env": {
        "DOCSTRING_EMBEDDING": "ollama:qwen3-embedding:4b"
      }
    }
  }
}
```

**Claude Desktop / Cursor** — same `env` block, under `mcpServers`.

Restart your editor afterwards. The first stderr line
(**Output → MCP**) reports the models actually in use — check it there rather
than assuming the variable was picked up.

### Choosing an embedding model

The default is `nomic-embed-text` (768d, 274MB) because it runs on any machine,
including laptops without a GPU. It is the floor, not the ceiling.

If your machine has the memory to spare, `qwen3-embedding:4b` (2560d, 2.5GB) is
the recommended upgrade, and any larger model works too. The published results
above were measured on the default, and the embedding model was not itself part
of that comparison — a bigger embedder is the obvious first thing to try if
recall matters more to you than footprint.

```bash
ollama pull qwen3-embedding:4b
```

```json
"env": { "DOCSTRING_EMBEDDING": "ollama:qwen3-embedding:4b" }
```

Note the `ollama:` prefix is still required for a tagged model, so the value
carries two colons.

`DOCSTRING_EMBEDDING` also accepts `openai:text-embedding-3-small` with
`OPENAI_API_KEY` set.

**Changing the embedding model requires deleting the index and re-indexing** —
the vector dimension is fixed when the index is created. Delete
`<repo>/.docstring/index.db`, then ask your agent to index again.

If the repository is not writable, the index falls back to `~/.docstring/`.

Requires Node 22.5+ (for `node:sqlite`). No native compilation.

## Troubleshooting

Diagnostics go to stderr, which VS Code shows under **Output → MCP**. The first
line reports the detected repo, database path, and models.

`Indexing failed` almost always means Ollama is not running (`ollama serve`) or
the model is not installed (`ollama pull nomic-embed-text`).
