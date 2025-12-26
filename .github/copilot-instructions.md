<!-- Copilot / AI agent instructions for contributors and coding agents -->
# Context7 — AI Coding Agent Instructions

Purpose: Help AI coding agents be immediately productive in this repository by summarizing architecture, workflows, conventions, and concrete commands.

## 1) Big picture

Context7 is an MCP (Model Context Protocol) server that fetches up-to-date, version-specific documentation and code examples for any library and injects them into LLM prompts. It's a monorepo using `pnpm` workspaces with three major packages:

- **packages/mcp** — The Model Context Protocol server bridging MCP clients (Cursor, VS Code, Windsurf, Claude Code) with the Context7 backend API. Registers tools for library search and documentation retrieval.
- **packages/sdk** — TypeScript/JavaScript client library for the Context7 API (`https://context7.com/api`). Exports `Context7` class with `searchLibrary()` and `getDocs()` methods. Requires API key starting with `ctx7sk`.
- **packages/tools-ai-sdk** — Vercel AI SDK integration exporting `Context7Agent` and tools (`resolveLibrary`, `getLibraryDocs`) for use in AI agent frameworks.

**Key architectural insight:** The MCP server uses `AsyncLocalStorage` to carry per-request context (client IP, API key) to tool handlers. For stdio transport (editor integrations), falls back to `globalApiKey`. Supports two transports: `stdio` (for editors) and `http` (for general HTTP clients with header-based auth).

## 2) Build / test / run (concrete commands)

```bash
pnpm install                                      # Install all dependencies
pnpm build                                        # Build all packages (tsc + tsup)
pnpm test                                         # Run all tests (vitest for sdk and tools-ai-sdk)
pnpm lint && pnpm format                         # Check/fix lint and formatting

# MCP server specific
pnpm --filter @upstash/context7-mcp build        # Build MCP server
pnpm --filter @upstash/context7-mcp dev          # Dev watch (TypeScript)
node packages/mcp/dist/index.js --transport http --port 3000   # Run HTTP server
node packages/mcp/dist/index.js --transport stdio              # Run stdio server (editor mode)

# SDK and tools-ai-sdk
pnpm --filter @upstash/context7-sdk test
pnpm --filter @upstash/context7-tools-ai-sdk test
```

## 3) Important runtime conventions

**MCP Tool Registration:** Tools are registered in [packages/mcp/src/index.ts](packages/mcp/src/index.ts) following this pattern:
- Use `server.registerTool(title, description, inputSchema, handler)` where `inputSchema` is a Zod schema
- Handler receives tool input and must return `{ content: [{ type: "text", text: "..." }] }`
- The `resolve-library-id` tool must always be preferred before `get-library-docs` (this ordering is enforced in SDK helpers)

**API Key & Authentication Flow:**
- Stdio mode (editor integrations): API key passed via `CONTEXT7_API_KEY` env var, stored in `globalApiKey`
- HTTP mode: API key passed via `Authorization: Bearer <token>` header or `Context7-API-Key` / `X-API-Key` variants
- SDK client throws error if no API key provided; warns if prefix is not `ctx7sk`

**Request Context Propagation:** Use `requestContext.run()` in HTTP handlers to propagate client IP and API key through async call stacks.

## 4) APIs & integration points

- **Backend API:** `https://context7.com/api` (base URL in [packages/sdk/src/client.ts](packages/sdk/src/client.ts#L13))
- **SDK Endpoints:** `searchLibrary(query)` and `getDocs(libraryId, options)` with optional `topic`, `mode` ("code" or "info"), `format` ("json" or "txt")
- **Proxy Support:** `HTTPS_PROXY` / `HTTP_PROXY` environment variables are respected by SDK's HTTP client
- **Error Handling:** SDK throws `Context7Error` with descriptive messages; MCP server catches and returns error content blocks

## 5) Code & style conventions

- **Language:** TypeScript with ESM (`type: "module"` in all package.json files)
- **Typechecking:** `pnpm typecheck` (runs `tsc --noEmit` per-package)
- **Testing:** SDK and tools-ai-sdk use Vitest; MCP server has no unit tests yet (echo placeholder in package.json)
- **Formatting & Linting:** Prettier + ESLint configured at repo root, delegated to packages via `pnpm format` / `pnpm lint`
- **Distribution:** SDK uses `tsup` for dual CJS/ESM outputs; MCP server uses raw `tsc` output with `chmod 755`

## 6) Quick patterns for contributors

**Adding a new MCP tool:** Mirror the existing `resolve-library-id` or `get-library-docs` patterns in [packages/mcp/src/index.ts](packages/mcp/src/index.ts):
1. Define Zod schema for inputs (be strict and descriptive)
2. Register with `server.registerTool()`
3. Handler calls SDK methods and returns `{ content: [...] }`

**Calling Context7 API from MCP server:** Use `searchLibraries()` or `fetchLibraryDocumentation()` from [packages/mcp/src/lib/api.ts](packages/mcp/src/lib/api.ts) — they handle header generation and client IP extraction.

**SDK Usage:** Always instantiate `Context7` with an API key; prefer the two-step flow (`searchLibrary()` then `getDocs()`) over guessing library IDs.

**Export patterns (tools-ai-sdk):** Export both SDK wrapper tools and a higher-level `Context7Agent` for AI frameworks. Peer dependencies are `ai`, `zod`, and `@upstash/context7-sdk`.

## 7) When you need to ask the user for clarification

- If they ask for library docs without specifying the library clearly, request the **package name** (e.g. `react`) so you can call `resolve-library-id`, or the exact **Context7-compatible ID** (`/org/project` or `/org/project/version`).
- If modifying MCP tools, clarify whether the change affects the public tool interface (title, description, inputs) or internal implementation.

## 8) Key files for reference

- [packages/mcp/src/index.ts](packages/mcp/src/index.ts) — MCP server entry, tool registration, transport setup
- [packages/mcp/src/lib/api.ts](packages/mcp/src/lib/api.ts) — API client helpers, header generation, error handling
- [packages/sdk/src/client.ts](packages/sdk/src/client.ts) — SDK `Context7` class, API key validation, method signatures
- [packages/tools-ai-sdk/src/agents/index.ts](packages/tools-ai-sdk/src/agents/index.ts) — `Context7Agent` for Vercel AI SDK
- [package.json](package.json) — Root workspace scripts and dependencies
- [pnpm-workspace.yaml](pnpm-workspace.yaml) — Workspace configuration

If any section is unclear or you want more examples, let me know which area to expand.
