import { readFile, readdir, stat } from "node:fs/promises";
import { join, relative, sep } from "node:path";
import { createHash } from "node:crypto";
import type { Chunk } from "@docstring/core";
import { chunkFile, type ChunkOptions } from "./chunk-file.js";
import { languageForPath } from "./parser.js";

/**
 * Indexing generated files is the most common cause of garbage retrieval:
 * a minified bundle or a lockfile will match almost any query weakly and
 * crowd out the real answer.
 */
export const DEFAULT_IGNORE_DIRS = new Set([
  "node_modules", ".git", ".next", "dist", "build", "out", "coverage",
  ".turbo", ".vercel", "__snapshots__", ".pnpm-store",
]);

export const DEFAULT_IGNORE_FILES = [
  /\.min\.(js|css)$/, /-lock\.(json|yaml)$/, /\.lock$/, /\.d\.ts$/,
  /\.(png|jpe?g|gif|webp|svg|ico|wav|mp3|mp4|woff2?|ttf|eot|pdf|zip|gz)$/i,
];

export interface WalkOptions extends ChunkOptions {
  /** Index files with no grammar (css, md, yaml) via the text fallback. */
  includeUnparsable?: boolean;
  maxFileBytes?: number;
  ignoreDirs?: Set<string>;
  ignoreFiles?: RegExp[];
}

export interface IndexedFile {
  path: string;
  contentHash: string;
  chunks: Chunk[];
}

export function shouldIndex(relPath: string, opts: WalkOptions): boolean {
  const parts = relPath.split(sep);
  const ignoreDirs = opts.ignoreDirs ?? DEFAULT_IGNORE_DIRS;
  if (parts.some((p) => ignoreDirs.has(p) || (p.startsWith(".") && p.length > 1 && !p.includes(".")))) {
    return false;
  }
  const ignoreFiles = opts.ignoreFiles ?? DEFAULT_IGNORE_FILES;
  if (ignoreFiles.some((re) => re.test(relPath))) return false;
  if (languageForPath(relPath) === null && !opts.includeUnparsable) return false;
  return true;
}

export async function* walkFiles(root: string, opts: WalkOptions): AsyncGenerator<string> {
  const stack = [root];
  while (stack.length > 0) {
    const dir = stack.pop()!;
    let entries;
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const full = join(dir, entry.name);
      const rel = relative(root, full);
      if (entry.isDirectory()) {
        const ignoreDirs = opts.ignoreDirs ?? DEFAULT_IGNORE_DIRS;
        if (!ignoreDirs.has(entry.name)) stack.push(full);
      } else if (entry.isFile() && shouldIndex(rel, opts)) {
        yield rel;
      }
    }
  }
}

/**
 * Content hash drives incremental reindexing: unchanged files are skipped,
 * changed files get deleteByFile then upsert. Building this in from the start
 * is what keeps iteration fast once the repo is large.
 */
export function hashContent(source: string): string {
  return createHash("sha256").update(source).digest("hex").slice(0, 16);
}

export async function indexRepo(
  root: string,
  opts: WalkOptions,
  onFile?: (file: IndexedFile, index: number) => void,
): Promise<IndexedFile[]> {
  const maxBytes = opts.maxFileBytes ?? 400_000;
  const out: IndexedFile[] = [];
  let i = 0;
  for await (const rel of walkFiles(root, opts)) {
    const full = join(root, rel);
    const info = await stat(full);
    if (info.size > maxBytes) continue;
    const source = await readFile(full, "utf8");
    const chunks = await chunkFile(rel, source, opts);
    if (chunks.length === 0) continue;
    const file: IndexedFile = { path: rel, contentHash: hashContent(source), chunks };
    out.push(file);
    onFile?.(file, i++);
  }
  return out;
}
