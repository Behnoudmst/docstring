import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm"],
  target: "node22",
  platform: "node",
  bundle: true,
  clean: true,
  outDir: "bundle",
  // Workspace packages are bundled in — they are not published separately,
  // so a consumer installing from npm could not resolve them otherwise.
  noExternal: [/^@docstring\//],
  // Native and wasm-bearing dependencies must stay external: they resolve
  // their binaries relative to their own package directory, which bundling
  // would break.
  external: [
    "@modelcontextprotocol/server",
    "web-tree-sitter",
    "tree-sitter-wasms",
    "sqlite-vec",
    "zod",
  ],
  banner: { js: "#!/usr/bin/env node" },
  shims: false,

  /**
   * esbuild treats `node:`-prefixed imports as external before any plugin
   * runs, and normalises the specifier against its own builtin list. That
   * list predates `node:sqlite`, so the prefix is stripped and the published
   * bundle imports a nonexistent "sqlite" package. A resolve plugin cannot
   * intercept it; restoring the specifier afterwards can.
   */
  async onSuccess() {
    const { readFile, writeFile } = await import("node:fs/promises");
    const file = "bundle/index.js";
    const source = await readFile(file, "utf8");
    const fixed = source.replace(/from\s*"sqlite"/g, 'from "node:sqlite"');
    if (fixed === source && source.includes('"sqlite"')) {
      throw new Error("expected to restore the node:sqlite specifier but found nothing to replace");
    }
    await writeFile(file, fixed);
  },
});
