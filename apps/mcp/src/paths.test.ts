import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  checkStaleness,
  fallbackDbPath,
  findRepoRoot,
  normalizePathPrefix,
  resolvePaths,
} from "./paths.js";

const dirs: string[] = [];
const tmp = () => {
  const d = mkdtempSync(join(tmpdir(), "docstring-paths-"));
  dirs.push(d);
  return d;
};
afterEach(() => {
  while (dirs.length > 0) rmSync(dirs.pop()!, { recursive: true, force: true });
});

describe("findRepoRoot", () => {
  it("finds the root from a nested directory", () => {
    const root = tmp();
    mkdirSync(join(root, ".git"));
    const nested = join(root, "packages", "core", "src");
    mkdirSync(nested, { recursive: true });
    expect(findRepoRoot(nested)).toBe(root);
  });

  it("returns the directory itself when it is the root", () => {
    const root = tmp();
    mkdirSync(join(root, ".git"));
    expect(findRepoRoot(root)).toBe(root);
  });

  it("returns null rather than guessing when there is no marker", () => {
    expect(findRepoRoot(tmp())).toBeNull();
  });
});

describe("resolvePaths", () => {
  it("puts the index inside the repo it describes", () => {
    const root = tmp();
    mkdirSync(join(root, ".git"));
    const paths = resolvePaths({ DOCSTRING_REPO: root } as NodeJS.ProcessEnv);
    expect(paths.dbPath).toBe(join(root, ".docstring", "index.db"));
    expect(paths.explicit).toBe(false);
  });

  it("lets an explicit DOCSTRING_DB win", () => {
    const paths = resolvePaths({
      DOCSTRING_REPO: "/repo",
      DOCSTRING_DB: "/custom/index.db",
    } as NodeJS.ProcessEnv);
    expect(paths.dbPath).toBe("/custom/index.db");
    expect(paths.explicit).toBe(true);
  });

  it("keeps two repos on separate indexes", () => {
    const a = resolvePaths({ DOCSTRING_REPO: "/one" } as NodeJS.ProcessEnv);
    const b = resolvePaths({ DOCSTRING_REPO: "/two" } as NodeJS.ProcessEnv);
    expect(a.dbPath).not.toBe(b.dbPath);
  });
});

describe("checkStaleness", () => {
  it("reports a missing index", () => {
    expect(checkStaleness(join(tmp(), "none.db"), [])).toMatchObject({ exists: false });
  });

  it("counts files modified after the index was written", () => {
    const dir = tmp();
    const db = join(dir, "index.db");
    const old = join(dir, "old.ts");
    const fresh = join(dir, "fresh.ts");
    writeFileSync(old, "a");
    writeFileSync(db, "x");
    writeFileSync(fresh, "b");

    // Set times explicitly: writes inside the same millisecond make mtime
    // comparison a coin flip.
    const past = new Date(Date.now() - 60_000);
    const future = new Date(Date.now() + 60_000);
    utimesSync(old, past, past);
    utimesSync(fresh, future, future);

    const health = checkStaleness(db, [old, fresh]);
    expect(health.exists).toBe(true);
    expect(health.staleFiles).toBe(1);
  });

  it("ignores files deleted between the walk and the check", () => {
    const dir = tmp();
    const db = join(dir, "index.db");
    writeFileSync(db, "x");
    expect(() => checkStaleness(db, [join(dir, "gone.ts")])).not.toThrow();
  });
});

describe("fallbackDbPath", () => {
  it("derives a distinct, filesystem-safe name per repo", () => {
    const a = fallbackDbPath("/Users/ben/work/elegant-menu-front");
    const b = fallbackDbPath("/Users/ben/work/other-repo");
    expect(a).not.toBe(b);
    expect(a).toMatch(/elegant-menu-front\.db$/);
  });
});

describe("normalizePathPrefix", () => {
  const root = "/Users/ben/work/vekt";

  it("leaves a repo-relative prefix alone", () => {
    expect(normalizePathPrefix("app/api/", root)).toBe("app/api/");
  });

  it("converts the absolute path an agent actually sends", () => {
    expect(normalizePathPrefix(`${root}/app/api`, root)).toBe("app/api");
  });

  it("treats the repo root itself as no filter", () => {
    expect(normalizePathPrefix(root, root)).toBe("");
    expect(normalizePathPrefix(`${root}/`, root)).toBe("");
  });

  it("strips leading ./ and /", () => {
    expect(normalizePathPrefix("./lib", root)).toBe("lib");
    expect(normalizePathPrefix("/lib", root)).toBe("lib");
  });

  it("keeps a path outside the repo intact rather than silently rewriting it", () => {
    expect(normalizePathPrefix("/etc/passwd", root)).toBe("etc/passwd");
  });

  it("does not strip a sibling directory that merely shares the root's name", () => {
    expect(normalizePathPrefix(`${root}-old/app`, root)).toBe(`${root}-old/app`.replace(/^\//, ""));
  });
});
