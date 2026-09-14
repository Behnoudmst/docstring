import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { getLoadablePath } from "sqlite-vec";
import {
  EmbeddingMismatchError,
  type Chunk,
  type ChunkStore,
  type RetrievalFilter,
  type ScoredChunk,
  type SymbolKind,
} from "@docstring/core";

export function toBlob(vector: number[]): Uint8Array {
  return new Uint8Array(new Float32Array(vector).buffer);
}

interface Row {
  id: string;
  repo: string;
  path: string;
  language: string;
  symbol_name: string | null;
  symbol_kind: string | null;
  start_line: number;
  end_line: number;
  content: string;
  embedding_text: string;
  imports: string;
  git_sha: string;
  embedding_model: string;
  embedding_dim: number;
}

function toChunk(r: Row): Chunk {
  return {
    id: r.id,
    repo: r.repo,
    path: r.path,
    language: r.language,
    symbolName: r.symbol_name,
    symbolKind: (r.symbol_kind as SymbolKind | null) ?? null,
    startLine: r.start_line,
    endLine: r.end_line,
    content: r.content,
    embeddingText: r.embedding_text,
    imports: JSON.parse(r.imports) as string[],
    gitSha: r.git_sha,
    embeddingModel: r.embedding_model,
    embeddingDim: r.embedding_dim,
  };
}

function matches(chunk: Chunk, f?: RetrievalFilter): boolean {
  if (!f) return true;
  if (f.repo && chunk.repo !== f.repo) return false;
  if (f.pathPrefix && !chunk.path.startsWith(f.pathPrefix)) return false;
  if (f.language && chunk.language !== f.language) return false;
  if (f.symbolKind && chunk.symbolKind !== f.symbolKind) return false;
  return true;
}

/** FTS5 treats several characters as operators; a raw code identifier will throw. */
export function escapeFts(query: string): string {
  const terms = query
    .split(/[^A-Za-z0-9_]+/)
    .filter((t) => t.length > 1)
    .map((t) => `"${t}"`);
  return terms.length > 0 ? terms.join(" OR ") : '""';
}

export interface SqliteStoreOptions {
  path: string;
  embeddingModel: string;
  embeddingDim: number;
  /**
   * Document-side task prefix used at index time. Recorded because an index
   * built without a prefix cannot be queried with one: the vectors sit in a
   * different region of the space and every result is quietly wrong.
   */
  embeddingPrefix?: string;
}

/**
 * Persistent store on node:sqlite. No native compilation: SQLite ships with
 * Node 22.5+, and sqlite-vec provides a prebuilt loadable extension.
 *
 * The vec0 virtual table fixes its dimension at creation, so the index is
 * bound to one embedding model. Opening it with a different one throws rather
 * than silently returning meaningless neighbours.
 */
export class SqliteChunkStore implements ChunkStore {
  private db: DatabaseSync;
  readonly embeddingModel: string;
  readonly embeddingDim: number;
  readonly embeddingPrefix: string;

  constructor(opts: SqliteStoreOptions) {
    this.embeddingModel = opts.embeddingModel;
    this.embeddingDim = opts.embeddingDim;
    this.embeddingPrefix = opts.embeddingPrefix ?? "";
    if (opts.path !== ":memory:") {
      mkdirSync(dirname(opts.path), { recursive: true });
    }
    this.db = new DatabaseSync(opts.path, { allowExtension: true });
    this.db.loadExtension(getLoadablePath());
    this.db.exec("pragma journal_mode = WAL");
    this.db.exec("pragma synchronous = NORMAL");
    this.migrate();
  }

  private migrate(): void {
    this.db.exec(`
      create table if not exists meta (key text primary key, value text not null);
      create table if not exists chunks (
        id text primary key,
        repo text not null,
        path text not null,
        language text not null,
        symbol_name text,
        symbol_kind text,
        start_line integer not null,
        end_line integer not null,
        content text not null,
        embedding_text text not null,
        imports text not null,
        git_sha text not null,
        embedding_model text not null,
        embedding_dim integer not null
      );
      create index if not exists chunks_repo_path on chunks(repo, path);
      create table if not exists files (
        repo text not null,
        path text not null,
        content_hash text not null,
        primary key (repo, path)
      );
    `);

    const stored = this.db.prepare("select value from meta where key = ?").get("embedding_model") as
      | { value: string }
      | undefined;
    const storedDim = this.db.prepare("select value from meta where key = ?").get("embedding_dim") as
      | { value: string }
      | undefined;

    const storedPrefix = this.db.prepare("select value from meta where key = ?").get("embedding_prefix") as
      | { value: string }
      | undefined;

    if (stored && storedDim) {
      if (stored.value !== this.embeddingModel || Number(storedDim.value) !== this.embeddingDim) {
        throw new EmbeddingMismatchError(
          { model: stored.value, dim: Number(storedDim.value) },
          { model: this.embeddingModel, dim: this.embeddingDim },
        );
      }
      if ((storedPrefix?.value ?? "") !== this.embeddingPrefix) {
        throw new Error(
          `This index was built with document prefix ${JSON.stringify(storedPrefix?.value ?? "")} ` +
            `but the current config uses ${JSON.stringify(this.embeddingPrefix)}. ` +
            `Delete the index file and re-index.`,
        );
      }
    } else {
      this.db
        .prepare("insert or replace into meta(key, value) values (?, ?)")
        .run("embedding_model", this.embeddingModel);
      this.db
        .prepare("insert or replace into meta(key, value) values (?, ?)")
        .run("embedding_dim", String(this.embeddingDim));
      this.db
        .prepare("insert or replace into meta(key, value) values (?, ?)")
        .run("embedding_prefix", this.embeddingPrefix);
    }

    // Dimension is baked into the virtual table definition.
    this.db.exec(
      `create virtual table if not exists vec_chunks using vec0(chunk_id text primary key, embedding float[${this.embeddingDim}])`,
    );
    this.db.exec(
      `create virtual table if not exists fts_chunks using fts5(chunk_id unindexed, path, symbol_name, content, tokenize='unicode61')`,
    );
  }

  async upsertChunks(chunks: Chunk[], vectors?: number[][]): Promise<void> {
    if (chunks.length === 0) return;
    const insertChunk = this.db.prepare(`
      insert or replace into chunks
      (id, repo, path, language, symbol_name, symbol_kind, start_line, end_line,
       content, embedding_text, imports, git_sha, embedding_model, embedding_dim)
      values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const delVec = this.db.prepare("delete from vec_chunks where chunk_id = ?");
    const insVec = this.db.prepare("insert into vec_chunks(chunk_id, embedding) values (?, ?)");
    const delFts = this.db.prepare("delete from fts_chunks where chunk_id = ?");
    const insFts = this.db.prepare(
      "insert into fts_chunks(chunk_id, path, symbol_name, content) values (?, ?, ?, ?)",
    );

    this.db.exec("begin");
    try {
      chunks.forEach((c, i) => {
        if (c.embeddingModel !== this.embeddingModel || c.embeddingDim !== this.embeddingDim) {
          throw new EmbeddingMismatchError(
            { model: this.embeddingModel, dim: this.embeddingDim },
            { model: c.embeddingModel, dim: c.embeddingDim },
          );
        }
        insertChunk.run(
          c.id, c.repo, c.path, c.language, c.symbolName, c.symbolKind,
          c.startLine, c.endLine, c.content, c.embeddingText,
          JSON.stringify(c.imports), c.gitSha, c.embeddingModel, c.embeddingDim,
        );
        const v = vectors?.[i];
        if (v) {
          if (v.length !== this.embeddingDim) {
            throw new EmbeddingMismatchError(
              { model: this.embeddingModel, dim: this.embeddingDim },
              { model: "input", dim: v.length },
            );
          }
          delVec.run(c.id);
          insVec.run(c.id, toBlob(v));
        }
        delFts.run(c.id);
        insFts.run(c.id, c.path, c.symbolName ?? "", c.content);
      });
      this.db.exec("commit");
    } catch (err) {
      this.db.exec("rollback");
      throw err;
    }
  }

  async deleteByFile(repo: string, path: string): Promise<number> {
    const ids = this.db
      .prepare("select id from chunks where repo = ? and path = ?")
      .all(repo, path) as { id: string }[];
    const delVec = this.db.prepare("delete from vec_chunks where chunk_id = ?");
    const delFts = this.db.prepare("delete from fts_chunks where chunk_id = ?");
    this.db.exec("begin");
    try {
      for (const { id } of ids) {
        delVec.run(id);
        delFts.run(id);
      }
      this.db.prepare("delete from chunks where repo = ? and path = ?").run(repo, path);
      this.db.exec("commit");
    } catch (err) {
      this.db.exec("rollback");
      throw err;
    }
    return ids.length;
  }

  /**
   * vec0 KNN cannot express arbitrary joins in its filter, so over-fetch and
   * apply the filter afterwards. Over-fetching by 4x keeps recall intact for
   * the narrow filters this tool uses.
   */
  async vectorSearch(embedding: number[], k: number, filter?: RetrievalFilter): Promise<ScoredChunk[]> {
    if (embedding.length !== this.embeddingDim) {
      throw new EmbeddingMismatchError(
        { model: this.embeddingModel, dim: this.embeddingDim },
        { model: "query", dim: embedding.length },
      );
    }
    const fetch = filter ? k * 4 : k;
    const rows = this.db
      .prepare(`
        select c.*, v.distance as distance
        from vec_chunks v
        join chunks c on c.id = v.chunk_id
        where v.embedding match ? and k = ?
        order by v.distance
      `)
      .all(toBlob(embedding), fetch) as unknown as (Row & { distance: number })[];

    return rows
      .map((r) => ({
        chunk: toChunk(r),
        // vec0 returns L2 distance; smaller is better. Map to a descending score.
        score: 1 / (1 + r.distance),
        source: "vector" as const,
      }))
      .filter((s) => matches(s.chunk, filter))
      .slice(0, k);
  }

  async keywordSearch(query: string, k: number, filter?: RetrievalFilter): Promise<ScoredChunk[]> {
    const fetch = filter ? k * 4 : k;
    let rows: (Row & { rank: number })[];
    try {
      rows = this.db
        .prepare(`
          select c.*, bm25(fts_chunks) as rank
          from fts_chunks
          join chunks c on c.id = fts_chunks.chunk_id
          where fts_chunks match ?
          order by rank
          limit ?
        `)
        .all(escapeFts(query), fetch) as unknown as (Row & { rank: number })[];
    } catch {
      return [];
    }
    // sqlite's bm25() returns negative values where MORE negative is better.
    // Negate to get an ascending-is-better relevance, then squash into (0,1]
    // so a better document always has a HIGHER score, like every other source.
    return rows
      .map((r) => {
        const relevance = Math.max(0, -r.rank);
        return {
          chunk: toChunk(r),
          score: relevance / (1 + relevance),
          source: "keyword" as const,
        };
      })
      .filter((s) => matches(s.chunk, filter))
      .slice(0, k);
  }

  async getByIds(ids: string[]): Promise<Chunk[]> {
    if (ids.length === 0) return [];
    const placeholders = ids.map(() => "?").join(",");
    const rows = this.db
      .prepare(`select * from chunks where id in (${placeholders})`)
      .all(...ids) as unknown as Row[];
    return rows.map(toChunk);
  }

  /**
   * Read from `meta`, not from the configured values: this reports what the
   * index was actually built with. The constructor keeps the two in step, so
   * they normally agree — but answering from config would make this method
   * echo the caller's own setting back at it, which is worthless precisely
   * when someone is trying to find out why a model change had no effect.
   */
  async indexInfo(): Promise<{ embeddingModel: string; embeddingDim: number } | null> {
    const count = this.db.prepare("select count(*) as n from chunks").get() as { n: number };
    if (count.n === 0) return null;
    const read = (key: string) =>
      (this.db.prepare("select value from meta where key = ?").get(key) as
        | { value: string }
        | undefined)?.value;
    const model = read("embedding_model");
    const dim = read("embedding_dim");
    // An index with chunks but no meta rows predates the meta table; fall back
    // rather than claim the index is empty.
    return {
      embeddingModel: model ?? this.embeddingModel,
      embeddingDim: dim === undefined ? this.embeddingDim : Number(dim),
    };
  }

  /**
   * Whether any indexed chunk lives under a path prefix. Lets a caller tell a
   * filter that matches nothing from a query that matches nothing — the two
   * are indistinguishable in an empty result set, and reporting the wrong one
   * sends the caller off to rephrase a query that was never the problem.
   */
  hasPathPrefix(prefix: string): boolean {
    const pattern = `${prefix.replace(/[\\%_]/g, "\\$&")}%`;
    const row = this.db
      .prepare("select 1 as found from chunks where path like ? escape '\\' limit 1")
      .get(pattern);
    return row !== undefined;
  }

  /** Content hashes of already-indexed files, for skipping unchanged ones. */
  fileHashes(repo: string): Map<string, string> {
    const rows = this.db
      .prepare("select path, content_hash from files where repo = ?")
      .all(repo) as { path: string; content_hash: string }[];
    return new Map(rows.map((r) => [r.path, r.content_hash]));
  }

  recordFile(repo: string, path: string, contentHash: string): void {
    this.db
      .prepare("insert or replace into files(repo, path, content_hash) values (?, ?, ?)")
      .run(repo, path, contentHash);
  }

  stats(): { chunks: number; files: number; vectors: number } {
    const n = (sql: string) => (this.db.prepare(sql).get() as { n: number }).n;
    return {
      chunks: n("select count(*) as n from chunks"),
      files: n("select count(distinct path) as n from chunks"),
      vectors: n("select count(*) as n from vec_chunks"),
    };
  }

  async close(): Promise<void> {
    this.db.close();
  }
}
