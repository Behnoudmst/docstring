import { existsSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { homedir } from "node:os";

/**
 * Walk up from a directory looking for a repository root marker.
 *
 * A host spawns this server in the workspace root, so `process.cwd()` is
 * usually the answer already — but a monorepo package or a nested launch
 * directory would otherwise index the wrong tree.
 */
export function findRepoRoot(start: string = process.cwd()): string | null {
  const markers = [".git", ".hg", ".svn"];
  let dir = resolve(start);
  const stop = resolve("/");

  while (true) {
    for (const marker of markers) {
      const candidate = join(dir, marker);
      if (existsSync(candidate)) return dir;
    }
    const parent = dirname(dir);
    if (parent === dir || dir === stop) return null;
    dir = parent;
  }
}

export interface ResolvedPaths {
  /** The repository being served. */
  repoRoot: string;
  /** Where the index lives. */
  dbPath: string;
  /** True when the caller pinned the db explicitly and we should not guess. */
  explicit: boolean;
}

/**
 * Zero-config by default: index lives inside the repo it describes, so a
 * developer never types a path and two repos never share an index.
 *
 * DOCSTRING_DB still wins when set — a pinned path is an explicit choice.
 */
export function resolvePaths(env: NodeJS.ProcessEnv = process.env): ResolvedPaths {
  const explicitDb = env["DOCSTRING_DB"];
  const repoRoot = env["DOCSTRING_REPO"] ?? findRepoRoot() ?? process.cwd();

  if (explicitDb) {
    return { repoRoot, dbPath: resolve(explicitDb), explicit: true };
  }
  return { repoRoot, dbPath: join(repoRoot, ".docstring", "index.db"), explicit: false };
}

export interface IndexHealth {
  exists: boolean;
  /** Files changed on disk since the index was written, capped for speed. */
  staleFiles: number;
  indexedAt: Date | null;
}

/**
 * Cheap staleness check: compare file mtimes against the index mtime rather
 * than re-hashing every file. Wrong at the margins, fast enough to run on
 * every startup, and it only drives a hint — reindexing is still explicit.
 */
export function checkStaleness(dbPath: string, files: string[]): IndexHealth {
  if (!existsSync(dbPath)) return { exists: false, staleFiles: 0, indexedAt: null };
  const indexedAt = statSync(dbPath).mtime;
  let staleFiles = 0;
  for (const file of files) {
    try {
      if (statSync(file).mtime > indexedAt) staleFiles++;
      if (staleFiles > 50) break;
    } catch {
      /* deleted since the walk; not worth failing over */
    }
  }
  return { exists: true, staleFiles, indexedAt };
}

/** Where a globally-installed run should keep indexes when the repo is read-only. */
export function fallbackDbPath(repoRoot: string): string {
  const slug = repoRoot.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(-80);
  return join(homedir(), ".docstring", `${slug}.db`);
}
