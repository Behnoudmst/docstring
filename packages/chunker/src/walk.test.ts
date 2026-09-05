import { describe, expect, it } from "vitest";
import { shouldIndex, hashContent } from "./walk.js";

const base = {
  repo: "r",
  gitSha: "a".repeat(40),
  embeddingModel: "m",
  embeddingDim: 3,
};

describe("shouldIndex", () => {
  it("indexes source files", () => {
    expect(shouldIndex("src/lib/getSubDomain.ts", base)).toBe(true);
    expect(shouldIndex("src/components/SexyItem.tsx", base)).toBe(true);
  });

  it("skips dependency and build output directories", () => {
    expect(shouldIndex("node_modules/react/index.js", base)).toBe(false);
    expect(shouldIndex(".next/static/chunk.js", base)).toBe(false);
    expect(shouldIndex("dist/index.js", base)).toBe(false);
  });

  it("skips generated and binary files", () => {
    expect(shouldIndex("public/bg-2.webp", base)).toBe(false);
    expect(shouldIndex("public/sound/bell.wav", base)).toBe(false);
    expect(shouldIndex("pnpm-lock.yaml", base)).toBe(false);
    expect(shouldIndex("src/types/generated.d.ts", base)).toBe(false);
    expect(shouldIndex("vendor/jquery.min.js", base)).toBe(false);
  });

  it("skips unparsable files by default but includes them on request", () => {
    expect(shouldIndex("src/app/globals.css", base)).toBe(false);
    expect(shouldIndex("src/app/globals.css", { ...base, includeUnparsable: true })).toBe(true);
  });

  it("keeps root-level source outside src/", () => {
    expect(shouldIndex("styleObject.ts", base)).toBe(true);
    expect(shouldIndex("loader.js", base)).toBe(true);
  });
});

describe("hashContent", () => {
  it("is stable and changes with content", () => {
    expect(hashContent("a")).toBe(hashContent("a"));
    expect(hashContent("a")).not.toBe(hashContent("b"));
  });
});
