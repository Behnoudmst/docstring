import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["packages/*/src/**/*.test.ts", "apps/*/src/**/*.test.ts"],
    // apps/mcp/src/index.ts calls serveStdio at module scope; importing it under
    // vitest would hang waiting on stdin. Its pure helpers live in format.ts.
    exclude: ["**/node_modules/**", "apps/mcp/src/index.test.ts"],
  },
});
