import { describe, expect, it } from "vitest";
import { clampK } from "./format.js";

describe("clampK", () => {
  it("passes valid values through", () => {
    expect(clampK(10, 8, 30)).toBe(10);
  });

  it("falls back when absent or unparseable", () => {
    expect(clampK(undefined, 8, 30)).toBe(8);
    expect(clampK("many", 8, 30)).toBe(8);
  });

  it("caps values above the maximum so an agent cannot pull the whole index", () => {
    expect(clampK(99, 8, 30)).toBe(30);
    expect(clampK(1e9, 8, 20)).toBe(20);
  });

  it("floors at 1 rather than returning zero or negative results", () => {
    expect(clampK(0, 8, 30)).toBe(1);
    expect(clampK(-5, 8, 30)).toBe(1);
  });

  it("truncates fractional values", () => {
    expect(clampK(7.9, 8, 30)).toBe(7);
  });
});
