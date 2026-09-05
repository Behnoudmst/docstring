import { createHash } from "node:crypto";
import type { Chunk, SymbolKind } from "@docstring/core";
import type { Node } from "web-tree-sitter";
import { getParser, languageForPath, type SupportedLanguage } from "./parser.js";

export interface ChunkOptions {
  repo: string;
  gitSha: string;
  embeddingModel: string;
  embeddingDim: number;
  /** Chunks shorter than this get merged with neighbours. */
  minChars?: number;
  /** Classes longer than this are split into per-method chunks plus an outline. */
  maxChars?: number;
}

const DEFAULTS = { minChars: 120, maxChars: 3000 };

/** Declaration node types worth becoming their own chunk, per language. */
const DECLARATIONS: Record<SupportedLanguage, Record<string, SymbolKind>> = {
  typescript: {
    function_declaration: "function",
    generator_function_declaration: "function",
    class_declaration: "class",
    interface_declaration: "interface",
    type_alias_declaration: "type",
    lexical_declaration: "function",
    enum_declaration: "type",
  },
  tsx: {
    function_declaration: "function",
    generator_function_declaration: "function",
    class_declaration: "class",
    interface_declaration: "interface",
    type_alias_declaration: "type",
    lexical_declaration: "function",
    enum_declaration: "type",
  },
  javascript: {
    function_declaration: "function",
    generator_function_declaration: "function",
    class_declaration: "class",
    lexical_declaration: "function",
  },
  python: { function_definition: "function", class_definition: "class" },
  go: { function_declaration: "function", method_declaration: "method", type_declaration: "type" },
};

function nameOf(node: Node): string | null {
  const direct = node.childForFieldName("name");
  if (direct) return direct.text;
  // const Foo = () => {}  →  lexical_declaration > variable_declarator > name
  const declarator = node.namedChildren.find((c) => c?.type === "variable_declarator");
  return declarator?.childForFieldName("name")?.text ?? null;
}

/**
 * A `const` is worth chunking if it holds a function, a wrapped component
 * (forwardRef, memo, styled), or exported data. `const MAX = 5` is noise;
 * `const CURRENCIES = [...]` answers "where is the currency list?".
 */
function declarationValue(node: Node): Node | undefined {
  const declarator = node.namedChildren.find((c) => c?.type === "variable_declarator");
  return declarator?.childForFieldName("value") ?? undefined;
}

function isFunctionValued(node: Node): boolean {
  if (node.type !== "lexical_declaration") return true;
  const value = declarationValue(node);
  if (!value) return false;
  if (value.type === "arrow_function" || value.type === "function_expression") return true;
  // React.forwardRef(...), memo(...), styled(...)(...) — a call taking a function.
  if (value.type === "call_expression" && hasFunctionArgument(value)) return true;
  return false;
}

/** Exported data: arrays, objects, template literals worth their own chunk. */
function isDataValued(node: Node, isExported: boolean): boolean {
  if (node.type !== "lexical_declaration" || !isExported) return false;
  const value = declarationValue(node);
  if (!value) return false;
  return ["array", "object", "template_string", "as_expression", "satisfies_expression"].includes(
    value.type,
  );
}

/** A .tsx function whose name is PascalCase is almost certainly a component. */
function refineKind(kind: SymbolKind, name: string | null, lang: SupportedLanguage): SymbolKind {
  if (lang === "tsx" && kind === "function" && name && /^[A-Z]/.test(name)) return "component";
  return kind;
}

function collectImports(root: Node): string[] {
  const out: string[] = [];
  for (const child of root.namedChildren) {
    if (!child) continue;
    if (child.type === "import_statement" || child.type === "import_from_statement") {
      const source = child.childForFieldName("source")?.text ?? child.text;
      out.push(source.replace(/['"]/g, ""));
    }
  }
  return out;
}

function chunkId(repo: string, path: string, startLine: number, content: string): string {
  return createHash("sha256")
    .update(`${repo}\0${path}\0${startLine}\0${content}`)
    .digest("hex")
    .slice(0, 16);
}

/**
 * Do not embed raw source. Prefixing path, symbol and imports is the single
 * largest retrieval improvement available, because it makes file location and
 * naming part of what the embedding model actually sees.
 */
export function buildEmbeddingText(params: {
  path: string;
  symbolName: string | null;
  symbolKind: SymbolKind | null;
  imports: string[];
  content: string;
}): string {
  const header = [`File: ${params.path}`];
  if (params.symbolName) {
    header.push(`Symbol: ${params.symbolName} (${params.symbolKind ?? "unknown"})`);
  }
  if (params.imports.length > 0) {
    header.push(`Imports: ${params.imports.slice(0, 12).join(", ")}`);
  }
  return `${header.join("\n")}\n---\n${params.content}`;
}

interface RawChunk {
  content: string;
  startLine: number;
  endLine: number;
  symbolName: string | null;
  symbolKind: SymbolKind | null;
}

function signatureOf(node: Node): string {
  const body = node.childForFieldName("body");
  if (!body) return node.text.split("\n")[0] ?? node.text.slice(0, 120);
  return node.text.slice(0, body.startIndex - node.startIndex).trim();
}

/** Does this node contain a function or arrow as a direct argument? (useEffect, useCallback, ...) */
function hasFunctionArgument(node: Node): boolean {
  const call = node.type === "call_expression" ? node : node.descendantsOfType("call_expression")[0];
  if (!call) return false;
  const args = call.childForFieldName("arguments");
  if (!args) return false;
  return args.namedChildren.some(
    (a) => a?.type === "arrow_function" || a?.type === "function_expression",
  );
}

function calleeName(node: Node): string | null {
  const call = node.type === "call_expression" ? node : node.descendantsOfType("call_expression")[0];
  return call?.childForFieldName("function")?.text ?? null;
}

/**
 * Inner declarations worth lifting out of an oversized function body:
 * event handlers, callbacks, nested helpers, and hook bodies. Hook calls
 * matter because a useEffect body is real logic — if it is only summarised
 * in the outline, that code exists in no chunk and can never be cited.
 */
function innerDeclarations(body: Node): Node[] {
  const out: Node[] = [];
  for (const stmt of body.namedChildren) {
    if (!stmt) continue;
    if (stmt.type === "function_declaration") {
      out.push(stmt);
    } else if (stmt.type === "lexical_declaration") {
      if (isFunctionValued(stmt) || hasFunctionArgument(stmt)) out.push(stmt);
    } else if (stmt.type === "expression_statement" && hasFunctionArgument(stmt)) {
      out.push(stmt);
    }
  }
  return out;
}

/** Last-resort split for a piece that is still too big: fixed line windows. */
function windowSplit(
  content: string,
  startLine: number,
  symbolName: string | null,
  kind: SymbolKind | null,
  maxChars: number,
): RawChunk[] {
  const lines = content.split("\n");
  const out: RawChunk[] = [];
  let buf: string[] = [];
  let offset = 0;
  let part = 1;
  const flush = () => {
    if (buf.length === 0) return;
    out.push({
      content: buf.join("\n"),
      startLine: startLine + offset,
      endLine: startLine + offset + buf.length - 1,
      symbolName: symbolName ? `${symbolName} (part ${part})` : null,
      symbolKind: kind,
    });
    offset += buf.length;
    part++;
    buf = [];
  };
  for (const line of lines) {
    // A single minified or long JSX line can exceed the limit on its own.
    if (line.length > maxChars) {
      flush();
      for (let i = 0; i < line.length; i += maxChars) {
        buf = [line.slice(i, i + maxChars)];
        flush();
      }
      continue;
    }
    if (buf.length > 0 && buf.join("\n").length + line.length + 1 > maxChars) flush();
    buf.push(line);
  }
  flush();
  return out;
}

/**
 * A 20k-character React component embeds into one averaged vector that matches
 * everything weakly and nothing precisely. Split it into an outline plus the
 * handlers and the render body, so each piece is separately retrievable.
 */
function splitLargeFunction(
  node: Node,
  outerNode: Node,
  name: string | null,
  kind: SymbolKind,
  maxChars: number,
): RawChunk[] {
  if (outerNode.text.length <= maxChars) return [];

  const body =
    node.childForFieldName("body") ??
    node.namedChildren
      .find((c) => c?.type === "variable_declarator")
      ?.childForFieldName("value")
      ?.childForFieldName("body");
  if (!body) return [];

  const inners = innerDeclarations(body);
  const innerRanges = inners.map((n) => [n.startIndex, n.endIndex] as const);
  const isInsideInner = (n: Node) =>
    innerRanges.some(([s, e]) => n.startIndex >= s && n.endIndex <= e);

  const returnStmt = body.namedChildren.find(
    (c): c is Node => c !== null && c.type === "return_statement" && !isInsideInner(c),
  );

  // Statements that are neither lifted handlers nor the render body: hooks, state, setup.
  const setup = body.namedChildren.filter(
    (c): c is Node =>
      c !== null && !isInsideInner(c) && c !== returnStmt && !inners.includes(c),
  );

  const chunks: RawChunk[] = [];

  const outline = [
    // Take the signature from the outer node so `export default` is preserved.
    outerNode.text.slice(0, body.startIndex - outerNode.startIndex + 1).trim(),
    ...setup.map((s) => `  ${s.text.split("\n")[0]}`),
    ...inners.map((i) => `  ${signatureOf(i) || (i.text.split("\n")[0] ?? "")}`),
    returnStmt ? "  return ( ... )" : "",
    "}",
  ]
    .filter(Boolean)
    .join("\n");

  chunks.push({
    content: outline,
    startLine: outerNode.startPosition.row + 1,
    endLine: outerNode.endPosition.row + 1,
    symbolName: name,
    symbolKind: kind,
  });

  for (const inner of inners) {
    const innerName = nameOf(inner) ?? calleeName(inner);
    const qualified = name && innerName ? `${name}.${innerName}` : innerName;
    const piece: RawChunk = {
      content: inner.text,
      startLine: inner.startPosition.row + 1,
      endLine: inner.endPosition.row + 1,
      symbolName: qualified,
      symbolKind: "method",
    };
    chunks.push(
      ...(piece.content.length > maxChars
        ? windowSplit(piece.content, piece.startLine, qualified, "method", maxChars)
        : [piece]),
    );
  }

  if (returnStmt) {
    const renderName = name ? `${name} (render)` : null;
    const piece: RawChunk = {
      content: returnStmt.text,
      startLine: returnStmt.startPosition.row + 1,
      endLine: returnStmt.endPosition.row + 1,
      symbolName: renderName,
      symbolKind: kind,
    };
    chunks.push(
      ...(piece.content.length > maxChars
        ? windowSplit(piece.content, piece.startLine, renderName, kind, maxChars)
        : [piece]),
    );
  }

  return chunks;
}

/**
 * A 6000-character currency array or allergen map has no blank lines, so the
 * text fallback cannot split it. Split on the literal's own elements instead,
 * packing them up to the size limit.
 */
function splitDataLiteral(
  node: Node,
  outerNode: Node,
  name: string | null,
  maxChars: number,
): RawChunk[] {
  if (outerNode.text.length <= maxChars) return [];
  const value = declarationValue(node);
  const elements = value?.namedChildren.filter((c): c is Node => c !== null) ?? [];
  if (elements.length < 2) return [];

  const chunks: RawChunk[] = [];
  let buf: Node[] = [];
  let part = 1;
  const flush = () => {
    if (buf.length === 0) return;
    const first = buf[0]!;
    const last = buf[buf.length - 1]!;
    chunks.push({
      content: `${name ?? "data"} [part ${part}]\n${buf.map((n) => n.text).join(",\n")}`,
      startLine: first.startPosition.row + 1,
      endLine: last.endPosition.row + 1,
      symbolName: name ? `${name} (part ${part})` : null,
      symbolKind: "constant",
    });
    part++;
    buf = [];
  };
  for (const el of elements) {
    if (buf.reduce((n, b) => n + b.text.length, 0) + el.text.length > maxChars) flush();
    buf.push(el);
  }
  flush();
  return chunks;
}

function splitLargeClass(node: Node, lang: SupportedLanguage, maxChars: number): RawChunk[] {
  const body = node.childForFieldName("body");
  const methods = body
    ? body.namedChildren.filter(
        (c): c is Node =>
          c !== null && (c.type === "method_definition" || c.type === "function_definition"),
      )
    : [];
  if (methods.length === 0 || node.text.length <= maxChars) return [];

  const className = nameOf(node);
  const outline = methods
    .map((m) => `  ${nameOf(m) ?? "?"}(...)`)
    .join("\n");

  const chunks: RawChunk[] = [
    {
      content: `class ${className ?? ""} {\n${outline}\n}`,
      startLine: node.startPosition.row + 1,
      endLine: node.endPosition.row + 1,
      symbolName: className,
      symbolKind: "class",
    },
  ];
  for (const m of methods) {
    chunks.push({
      content: m.text,
      startLine: m.startPosition.row + 1,
      endLine: m.endPosition.row + 1,
      symbolName: className ? `${className}.${nameOf(m) ?? "?"}` : nameOf(m),
      symbolKind: "method",
    });
  }
  return chunks;
}

/** Files with no grammar (md, css, yaml) still get indexed, split on blank lines. */
export function fallbackChunks(source: string, maxChars: number): RawChunk[] {
  const lines = source.split("\n");
  const out: RawChunk[] = [];
  let buf: string[] = [];
  let start = 1;
  const flush = (end: number) => {
    const content = buf.join("\n").trim();
    if (content.length > 0) {
      out.push({ content, startLine: start, endLine: end, symbolName: null, symbolKind: "file" });
    }
    buf = [];
  };
  lines.forEach((line, i) => {
    buf.push(line);
    const size = buf.join("\n").length;
    // Prefer a blank-line boundary, but never exceed the limit waiting for one.
    if (size >= maxChars && (line.trim() === "" || size >= maxChars * 1.2)) {
      flush(i + 1);
      start = i + 2;
    }
  });
  flush(lines.length);
  return out.flatMap((c) =>
    c.content.length > maxChars * 1.2
      ? windowSplit(c.content, c.startLine, null, "file", maxChars)
      : [c],
  );
}

export async function chunkFile(
  path: string,
  source: string,
  opts: ChunkOptions,
): Promise<Chunk[]> {
  const minChars = opts.minChars ?? DEFAULTS.minChars;
  const maxChars = opts.maxChars ?? DEFAULTS.maxChars;
  const lang = languageForPath(path);

  let raw: RawChunk[];
  let imports: string[] = [];

  if (lang === null) {
    raw = fallbackChunks(source, maxChars);
  } else {
    const parser = await getParser(lang);
    const tree = parser.parse(source);
    if (!tree) return [];
    const root = tree.rootNode;
    imports = collectImports(root);
    raw = [];

    for (const top of root.namedChildren) {
      if (!top) continue;
      // `export function foo()` parses as export_statement wrapping the declaration.
      // Keep the outer node's range so the `export` keyword stays in the chunk.
      const isExport = top.type === "export_statement";
      const decl = isExport
        ? top.namedChildren.find((c) => c !== null && c.type in DECLARATIONS[lang])
        : top;
      if (!decl || !(decl.type in DECLARATIONS[lang])) continue;

      const isData = isDataValued(decl, isExport);
      if (!isFunctionValued(decl) && !isData) continue;

      const kindBase = isData ? "constant" : DECLARATIONS[lang][decl.type]!;
      const name = nameOf(decl);
      const kind = isData ? "constant" : refineKind(kindBase, name, lang);

      const outer = isExport ? top : decl;

      if (isData) {
        const dataSplit = splitDataLiteral(decl, outer, name, maxChars);
        if (dataSplit.length > 0) {
          raw.push(...dataSplit);
          continue;
        }
        raw.push({
          content: outer.text,
          startLine: outer.startPosition.row + 1,
          endLine: outer.endPosition.row + 1,
          symbolName: name,
          symbolKind: "constant",
        });
        continue;
      }

      const split = splitLargeClass(decl, lang, maxChars);
      if (split.length > 0) {
        raw.push(...split);
        continue;
      }

      const fnSplit = splitLargeFunction(decl, outer, name, kind, maxChars);
      if (fnSplit.length > 0) {
        raw.push(...fnSplit);
        continue;
      }

      const content = outer.text;
      const startLine = outer.startPosition.row + 1;

      // Nothing structural left to split on, but still oversized.
      if (content.length > maxChars) {
        raw.push(...windowSplit(content, startLine, name, kind, maxChars));
        continue;
      }

      raw.push({
        content,
        startLine,
        endLine: outer.endPosition.row + 1,
        symbolName: name,
        symbolKind: kind,
      });
    }

    // A file of only imports and JSX, or one that failed to parse usefully.
    if (raw.length === 0) raw = fallbackChunks(source, maxChars);
  }

  // Merge runs of tiny UNNAMED fragments so a stylesheet does not become 40 vectors.
  // Named declarations are never merged: symbolName is what makes a citation precise,
  // and "getSubDomain, isSupportedLang" points at neither function properly.
  const mergeable = (c: RawChunk) => c.symbolName === null && c.content.length < minChars;
  const merged: RawChunk[] = [];
  for (const c of raw) {
    const prev = merged[merged.length - 1];
    if (prev && mergeable(prev) && mergeable(c)) {
      prev.content = `${prev.content}\n\n${c.content}`;
      prev.endLine = c.endLine;
    } else {
      merged.push({ ...c });
    }
  }

  return merged.map((c) => ({
    id: chunkId(opts.repo, path, c.startLine, c.content),
    repo: opts.repo,
    path,
    language: lang ?? "text",
    symbolName: c.symbolName,
    symbolKind: c.symbolKind,
    startLine: c.startLine,
    endLine: c.endLine,
    content: c.content,
    embeddingText: buildEmbeddingText({
      path,
      symbolName: c.symbolName,
      symbolKind: c.symbolKind,
      imports,
      content: c.content,
    }),
    imports,
    gitSha: opts.gitSha,
    embeddingModel: opts.embeddingModel,
    embeddingDim: opts.embeddingDim,
  }));
}
