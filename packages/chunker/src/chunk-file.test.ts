import { describe, expect, it } from "vitest";
import { chunkFile, buildEmbeddingText } from "./chunk-file.js";
import { languageForPath } from "./parser.js";

const opts = {
  repo: "elegant-menu-front",
  gitSha: "a".repeat(40),
  embeddingModel: "nomic-embed-text",
  embeddingDim: 768,
};

describe("languageForPath", () => {
  it("routes tsx and ts to different grammars", () => {
    expect(languageForPath("src/components/SexyItem.tsx")).toBe("tsx");
    expect(languageForPath("src/lib/getSubDomain.ts")).toBe("typescript");
  });

  it("returns null for files with no grammar", () => {
    expect(languageForPath("src/app/globals.css")).toBeNull();
    expect(languageForPath("README.md")).toBeNull();
  });
});

describe("chunking a lib file", () => {
  const source = `import { headers } from "next/headers";

export function getSubDomain(host: string) {
  return host.split(".")[0];
}

export function isSupportedLang(lang: string): boolean {
  return ["en", "it", "fa"].includes(lang);
}
`;

  it("emits one chunk per exported function", async () => {
    const chunks = await chunkFile("src/lib/getSubDomain.ts", source, opts);
    expect(chunks.map((c) => c.symbolName)).toEqual(["getSubDomain", "isSupportedLang"]);
  });

  it("keeps the export keyword in the chunk content", async () => {
    const chunks = await chunkFile("src/lib/getSubDomain.ts", source, opts);
    expect(chunks[0]?.content.startsWith("export function")).toBe(true);
  });

  it("records line numbers that match the source", async () => {
    const chunks = await chunkFile("src/lib/getSubDomain.ts", source, opts);
    const lines = source.split("\n");
    const first = chunks[0]!;
    expect(lines[first.startLine - 1]).toContain("getSubDomain");
    expect(lines[first.endLine - 1]).toContain("}");
  });

  it("captures file imports on every chunk", async () => {
    const chunks = await chunkFile("src/lib/getSubDomain.ts", source, opts);
    expect(chunks[0]?.imports).toContain("next/headers");
  });
});

describe("chunking a tsx component file", () => {
  const source = `import { useState } from "react";
import { useOrders } from "@/hooks/useOrders";

export default function OrderCardKitchen({ order }: Props) {
  const [open, setOpen] = useState(false);
  const { refetch } = useOrders();
  return <div onClick={() => setOpen(!open)}>{order.id}</div>;
}

export const ItemPrice = ({ value }: { value: number }) => {
  return <span className="price">{value.toFixed(2)}</span>;
};

const PRICE_LOCALE = "it-IT";
`;

  it("labels PascalCase tsx functions as components", async () => {
    const chunks = await chunkFile("src/components/OrderCardKitchen.tsx", source, opts);
    const kinds = new Map(chunks.map((c) => [c.symbolName, c.symbolKind]));
    expect(kinds.get("OrderCardKitchen")).toBe("component");
  });

  it("picks up arrow-function components assigned to const", async () => {
    const chunks = await chunkFile("src/components/OrderCardKitchen.tsx", source, opts);
    expect(chunks.map((c) => c.symbolName)).toContain("ItemPrice");
  });

  it("ignores plain constants that hold no function", async () => {
    const chunks = await chunkFile("src/components/OrderCardKitchen.tsx", source, opts);
    expect(chunks.map((c) => c.symbolName)).not.toContain("PRICE_LOCALE");
  });

  it("does not mangle jsx — chunks contain the markup", async () => {
    const chunks = await chunkFile("src/components/OrderCardKitchen.tsx", source, opts);
    expect(chunks.some((c) => c.content.includes("<span className=\"price\">"))).toBe(true);
  });
});

describe("embedding text enrichment", () => {
  it("prefixes path, symbol and imports before the source", () => {
    const text = buildEmbeddingText({
      path: "src/lib/getSubDomain.ts",
      symbolName: "getSubDomain",
      symbolKind: "function",
      imports: ["next/headers"],
      content: "export function getSubDomain() {}",
    });
    expect(text).toContain("File: src/lib/getSubDomain.ts");
    expect(text).toContain("Symbol: getSubDomain (function)");
    expect(text).toContain("Imports: next/headers");
    expect(text.indexOf("File:")).toBeLessThan(text.indexOf("export function"));
  });

  it("leaves content untouched so citations quote the real source", async () => {
    const src = `export function a() { return 1; }\n`;
    const chunks = await chunkFile("src/a.ts", src, opts);
    expect(chunks[0]?.content).not.toContain("File:");
  });
});

describe("resilience", () => {
  it("still returns chunks from a file with a syntax error", async () => {
    const broken = `export function ok() { return 1; }\nexport function broken( {{{ \n`;
    const chunks = await chunkFile("src/broken.ts", broken, opts);
    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks.some((c) => c.symbolName === "ok")).toBe(true);
  });

  it("falls back to text splitting for files with no grammar", async () => {
    const css = `.a { color: red; }\n\n.b { color: blue; }\n`;
    const chunks = await chunkFile("src/app/globals.css", css, opts);
    expect(chunks.length).toBeGreaterThan(0);
    expect(chunks[0]?.language).toBe("text");
  });

  it("gives every chunk a stable, unique id", async () => {
    const src = `export function a() { return 1; }\nexport function b() { return 2; }\n`;
    const one = await chunkFile("src/a.ts", src, opts);
    const two = await chunkFile("src/a.ts", src, opts);
    expect(one.map((c) => c.id)).toEqual(two.map((c) => c.id));
    expect(new Set(one.map((c) => c.id)).size).toBe(one.length);
  });
});

describe("oversized components", () => {
  const bigComponent = `"use client";
import { useState, useEffect, useCallback } from "react";

export default function CreateOrderDialogue({ venueId, onClose }: Props) {
  const [items, setItems] = useState<Item[]>([]);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!venueId) return;
    loadDraft();
  }, [venueId]);

  const loadDraft = useCallback(async () => {
    const raw = window.localStorage.getItem("draft");
    if (raw) setItems(JSON.parse(raw));
  }, [venueId]);

  function addItem(item: Item) {
    setItems((prev) => [...prev, item]);
  }

  const handleSubmit = async () => {
    setSubmitting(true);
    await createOrder({ venueId, items });
    onClose();
  };

  return (
    <div className="dialogue">
      <button disabled={submitting} onClick={handleSubmit}>Send</button>
    </div>
  );
}
`;
  const small = { ...opts, maxChars: 400 };

  it("splits a large component instead of emitting one huge chunk", async () => {
    const chunks = await chunkFile("src/components/CreateOrderDialogue.tsx", bigComponent, small);
    expect(chunks.length).toBeGreaterThan(3);
    for (const c of chunks) expect(c.content.length).toBeLessThanOrEqual(400 * 1.5);
  });

  it("keeps an outline chunk under the component's own name", async () => {
    const chunks = await chunkFile("src/components/CreateOrderDialogue.tsx", bigComponent, small);
    const outline = chunks.find((c) => c.symbolName === "CreateOrderDialogue");
    expect(outline).toBeDefined();
    expect(outline!.content).toContain("function CreateOrderDialogue");
  });

  it("lifts handlers into separately retrievable chunks", async () => {
    const chunks = await chunkFile("src/components/CreateOrderDialogue.tsx", bigComponent, small);
    const names = chunks.map((c) => c.symbolName);
    expect(names).toContain("CreateOrderDialogue.handleSubmit");
    expect(names).toContain("CreateOrderDialogue.addItem");
  });

  it("lifts hook bodies so their code is not lost", async () => {
    const chunks = await chunkFile("src/components/CreateOrderDialogue.tsx", bigComponent, small);
    const names = chunks.map((c) => c.symbolName);
    expect(names).toContain("CreateOrderDialogue.useEffect");
    expect(names).toContain("CreateOrderDialogue.loadDraft");
  });

  it("separates the render body from the logic", async () => {
    const chunks = await chunkFile("src/components/CreateOrderDialogue.tsx", bigComponent, small);
    const render = chunks.find((c) => c.symbolName?.includes("(render)"));
    expect(render?.content).toContain("<div className=\"dialogue\">");
  });

  it("loses no meaningful source line — every statement lands in some chunk", async () => {
    const chunks = await chunkFile("src/components/CreateOrderDialogue.tsx", bigComponent, small);
    const covered = chunks.map((c) => c.content).join("\n");
    const meaningful = bigComponent
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 12 && !l.startsWith("import") && !l.startsWith('"use client"'));
    const missing = meaningful.filter((l) => !covered.includes(l));
    expect(missing).toEqual([]);
  });
});

describe("data constants and wrapped components", () => {
  it("indexes exported data by name instead of dropping it", async () => {
    const src = `export const CURRENCIES = ["EUR", "USD", "GBP"];\nconst INTERNAL = 5;\n`;
    const chunks = await chunkFile("src/currencyCodeList.ts", src, opts);
    expect(chunks.map((c) => c.symbolName)).toContain("CURRENCIES");
    expect(chunks.map((c) => c.symbolName)).not.toContain("INTERNAL");
  });

  it("splits a large data literal on its own elements", async () => {
    const items = Array.from({ length: 40 }, (_, i) => `  { code: "C${i}", name: "Currency ${i}" }`);
    const src = `export const CURRENCIES = [\n${items.join(",\n")}\n];\n`;
    const chunks = await chunkFile("src/currencyCodeList.ts", src, { ...opts, maxChars: 400 });
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) expect(c.content.length).toBeLessThan(700);
    expect(chunks[0]?.symbolName).toContain("CURRENCIES");
  });

  it("recognises forwardRef-wrapped components", async () => {
    const src = `import * as React from "react";
export const SelectTrigger = React.forwardRef<Ref, Props>(({ className }, ref) => (
  <Trigger ref={ref} className={className} />
));
`;
    const chunks = await chunkFile("src/components/ui/select.tsx", src, opts);
    const t = chunks.find((c) => c.symbolName === "SelectTrigger");
    expect(t).toBeDefined();
    expect(t?.symbolKind).toBe("component");
  });

  it("never emits a chunk far beyond maxChars, even from one long line", async () => {
    const longLine = `export const BLOB = "${"x".repeat(9000)}";`;
    const chunks = await chunkFile("src/blob.ts", longLine, { ...opts, maxChars: 1000 });
    for (const c of chunks) expect(c.content.length).toBeLessThanOrEqual(1300);
  });
});
