<!-- Copilot / AI agent instructions for contributors and coding agents -->
# Context7 — AI Coding Agent Instructions

Purpose: Help AI coding agents be immediately productive in this repository by summarizing architecture, workflows, conventions, and concrete commands.

1) Big picture
- Monorepo using `pnpm` workspaces (`packages/*`). Major packages:
  - packages/mcp — the MCP server (see [packages/mcp/src/index.ts](packages/mcp/src/index.ts#L1-L40)).
  - packages/sdk — client library for Context7 API ([packages/sdk/src/client.ts](packages/sdk/src/client.ts#L1-L40)).
  - packages/tools-ai-sdk — AI tooling (see `packages/tools-ai-sdk/src`).
- The MCP server acts as a bridge between MCP clients and the Context7 API. Key design points: tool registration via `server.registerTool`, per-request context via `AsyncLocalStorage`, and two transport modes: `stdio` and `http`.

2) Build / test / run (concrete)
- Install: `pnpm install` at repo root.
- Build everything: `pnpm build` (runs `pnpm -r run build`).
- Run all tests: `pnpm test` (runs tests in each package). Use `pnpm --filter <pkg> test` for a single package.
- Build & run the MCP server locally:
  - Build: `pnpm --filter @upstash/context7-mcp build`
  - Run HTTP server: `node packages/mcp/dist/index.js --transport http --port 3000`
  - Run stdio server (for editor integrations): `node packages/mcp/dist/index.js --transport stdio`
  - Dev (TypeScript watch): `pnpm --filter @upstash/context7-mcp dev`

3) Important runtime conventions
- MCP tools are registered in [packages/mcp/src/index.ts](packages/mcp/src/index.ts#L1-L160). Follow the same pattern: register a tool with `title`, `description`, `inputSchema`, and return structured `content` arrays.
- Always prefer `resolve-library-id` before `get-library-docs` (this is enforced in the user-facing helpers and promised behavior in the MCP tool descriptions).
- Per-request metadata (client IP, API key) is provided via `AsyncLocalStorage` in the MCP server; in stdio mode a `globalApiKey` fallback is used — pass API keys via env `CONTEXT7_API_KEY` or as request headers when using HTTP.

4) APIs & integration points
- Context7 backend API base: `https://context7.com/api` — used by `packages/mcp/src/lib/api.ts` and `packages/sdk`.
- Proxy support: `HTTPS_PROXY`/`HTTP_PROXY` env vars are respected (see `packages/mcp/src/lib/api.ts`).
- HTTP auth: MCP server accepts `Authorization: Bearer <token>` and various `Context7-API-Key` / `X-API-Key` header variants.

5) Code & style conventions
- TypeScript, ESM (`type: "module"`) across packages. Use `tsc` for typechecking.
- Run formatting and lint via: `pnpm format` and `pnpm lint` at repo root (delegates to packages).
- Tests: some packages use vitest; others may have no tests. Use package-level `package.json` to confirm.

6) Quick patterns for contributors
- If adding a new MCP tool, mirror `server.registerTool(...)` shape and ensure input zod schemas are strict and descriptive (see examples in [packages/mcp/src/index.ts](packages/mcp/src/index.ts#L40-L140)).
- When calling Context7 API from server code, use `generateHeaders(...)` helpers and preserve client IP where available (see [packages/mcp/src/lib/api.ts](packages/mcp/src/lib/api.ts#L1-L40)).
- SDKs expect an API key that usually starts with `ctx7sk` — `packages/sdk/src/client.ts` warns if missing prefix.

7) When you need to ask the user for clarification
- If the user asks for docs for a library, request either:
  - the package name (e.g. `react`) so you can call `resolve-library-id`, or
  - the exact Context7-compatible ID (`/org/project` or `/org/project/version`).

8) Files to inspect for examples
- Main entry and tool definitions: [packages/mcp/src/index.ts](packages/mcp/src/index.ts)
- API integration and error handling: [packages/mcp/src/lib/api.ts](packages/mcp/src/lib/api.ts)
- SDK client patterns and API-key behavior: [packages/sdk/src/client.ts](packages/sdk/src/client.ts)
- Root build/test commands: [package.json](package.json)

If any section is unclear or you want more examples (e.g., a sample MCP tool or a test harness), tell me which area to expand and I will iterate.
