import { createRequire } from "node:module";
import { Language, Parser } from "web-tree-sitter";

const require = createRequire(import.meta.url);

export type SupportedLanguage = "typescript" | "tsx" | "javascript" | "python" | "go";

const GRAMMAR: Record<SupportedLanguage, string> = {
  typescript: "tree-sitter-typescript.wasm",
  tsx: "tree-sitter-tsx.wasm",
  javascript: "tree-sitter-javascript.wasm",
  python: "tree-sitter-python.wasm",
  go: "tree-sitter-go.wasm",
};

/**
 * TSX and TypeScript are DIFFERENT grammars. Parsing a .tsx file with the
 * typescript grammar silently mangles every JSX block, which in a Next.js
 * repo is most of the codebase. Always route by extension.
 */
export function languageForPath(path: string): SupportedLanguage | null {
  if (path.endsWith(".tsx")) return "tsx";
  if (path.endsWith(".ts") || path.endsWith(".mts") || path.endsWith(".cts")) return "typescript";
  if (path.endsWith(".jsx")) return "tsx";
  if (path.endsWith(".js") || path.endsWith(".mjs") || path.endsWith(".cjs")) return "javascript";
  if (path.endsWith(".py")) return "python";
  if (path.endsWith(".go")) return "go";
  return null;
}

let initialised = false;
const cache = new Map<SupportedLanguage, Language>();

export async function getParser(lang: SupportedLanguage): Promise<Parser> {
  if (!initialised) {
    await Parser.init();
    initialised = true;
  }
  let language = cache.get(lang);
  if (!language) {
    const wasmPath = require.resolve(`tree-sitter-wasms/out/${GRAMMAR[lang]}`);
    language = await Language.load(wasmPath);
    cache.set(lang, language);
  }
  const parser = new Parser();
  parser.setLanguage(language);
  return parser;
}
