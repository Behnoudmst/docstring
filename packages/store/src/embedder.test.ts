import { afterEach, describe, expect, it, vi } from "vitest";
import { KNOWN_DIMS, OllamaEmbedder, createEmbedder } from "./embedder.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

function mockFetch(handler: (url: string, init?: RequestInit) => unknown) {
  globalThis.fetch = vi.fn(async (url: string | URL | Request, init?: RequestInit) => {
    const body = handler(String(url), init);
    return { ok: true, status: 200, json: async () => body, text: async () => "" } as Response;
  }) as unknown as typeof fetch;
}

describe("createEmbedder", () => {
  it("defaults to ollama when no provider is given", () => {
    expect(createEmbedder("nomic-embed-text").model).toBe("nomic-embed-text");
  });

  it("rejects unknown providers", () => {
    expect(() => createEmbedder("cohere:embed-v3")).toThrow(/Unknown embedding provider/);
  });

  it("knows the dimension of common models", () => {
    expect(KNOWN_DIMS["nomic-embed-text"]).toBe(768);
    expect(new OllamaEmbedder({ model: "bge-m3" }).dim).toBe(1024);
  });
});

describe("OllamaEmbedder", () => {
  it("batches requests rather than sending one call per chunk", async () => {
    let calls = 0;
    mockFetch(() => {
      calls++;
      return { embeddings: Array.from({ length: 16 }, () => [1, 2, 3]) };
    });
    const e = new OllamaEmbedder({ batchSize: 16 });
    await e.embed(Array.from({ length: 32 }, (_, i) => `text ${i}`));
    expect(calls).toBe(2);
  });

  it("gives an actionable error when ollama is unreachable", async () => {
    globalThis.fetch = vi.fn(async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch;
    await expect(new OllamaEmbedder().check()).rejects.toThrow(/Is it running\? Try: ollama serve/);
  });

  it("tells you to pull the model when it is missing", async () => {
    mockFetch(() => ({ models: [{ name: "llama3:latest" }] }));
    await expect(new OllamaEmbedder({ model: "bge-m3" }).check()).rejects.toThrow(
      /ollama pull bge-m3/,
    );
  });

  it("catches a model whose real dimension differs from the configured one", async () => {
    mockFetch((url) =>
      url.endsWith("/api/tags")
        ? { models: [{ name: "nomic-embed-text:latest" }] }
        : { embeddings: [[1, 2, 3]] },
    );
    await expect(new OllamaEmbedder({ dim: 768 }).check()).rejects.toThrow(
      /returns 3 dimensions but config says 768/,
    );
  });

  it("passes check when the model is present and the dimension matches", async () => {
    mockFetch((url) =>
      url.endsWith("/api/tags")
        ? { models: [{ name: "nomic-embed-text:latest" }] }
        : { embeddings: [Array.from({ length: 768 }, () => 0.1)] },
    );
    await expect(new OllamaEmbedder().check()).resolves.toBeUndefined();
  });
});

describe("task prefixes", () => {
  it("defaults to no prefix for nomic — measured as worse on code", () => {
    const e = new OllamaEmbedder({ model: "nomic-embed-text" });
    expect(e.prefixes.query).toBe("");
    expect(e.prefixes.document).toBe("");
  });

  it("matches a tagged model name to its base entry", () => {
    expect(new OllamaEmbedder({ model: "nomic-embed-text:latest" }).prefixes).toEqual({
      query: "", document: "",
    });
  });

  it("sends the document prefix when one is configured", async () => {
    const seen: string[] = [];
    mockFetch((_url, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as { input: string[] };
      seen.push(...body.input);
      return { embeddings: body.input.map(() => [1, 2, 3]) };
    });
    const e = new OllamaEmbedder({ prefixes: { query: "q: ", document: "d: " } });
    await e.embedDocuments(["export function a() {}"]);
    expect(seen[0]).toBe("d: export function a() {}");
  });

  it("sends the query prefix when one is configured", async () => {
    const seen: string[] = [];
    mockFetch((_url, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as { input: string[] };
      seen.push(...body.input);
      return { embeddings: body.input.map(() => [1, 2, 3]) };
    });
    const e = new OllamaEmbedder({ prefixes: { query: "q: ", document: "d: " } });
    await e.embedQuery("where is routing");
    expect(seen[0]).toBe("q: where is routing");
  });

  it("passes text through unchanged when no prefix is set", async () => {
    const seen: string[] = [];
    mockFetch((_url, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as { input: string[] };
      seen.push(...body.input);
      return { embeddings: body.input.map(() => [1, 2, 3]) };
    });
    await new OllamaEmbedder().embedQuery("where is routing");
    expect(seen[0]).toBe("where is routing");
  });
});
