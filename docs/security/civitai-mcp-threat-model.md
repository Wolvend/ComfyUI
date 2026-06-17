# Civitai FastMCP Threat Model

## Scope

This repository adds a local Civitai FastMCP server under `tools/civitai_mcp/` and supporting tests and docs. The server runs on a desktop machine alongside ComfyUI and can read from the local filesystem, query the public Civitai API, and optionally download or install assets into ComfyUI model folders.

## Assets That Matter

- Local ComfyUI installation and its `models/` tree
- Model binaries, workflow files, cache files, and downloaded assets
- Any Civitai API credentials configured in the environment
- The integrity of tool outputs used by downstream agents
- The user’s expectation that read-only tools do not write files

## Trust Boundaries

- **Local filesystem boundary:** The server can read and, for opt-in tools, write inside the local ComfyUI checkout. Paths must stay inside the intended ComfyUI root unless the user explicitly provides an alternate destination.
- **Network boundary:** All Civitai data comes from a remote public service. Responses, metadata, and filenames must be treated as untrusted input.
- **MCP boundary:** Tool inputs may be driven by other agents or clients. Every tool must validate parameters before performing filesystem or network actions.
- **Cache boundary:** Cached metadata must never become a source of truth for destructive actions; it is only an accelerator for read operations.

## Attacker-Controlled Inputs

- Search terms, tags, usernames, sort keys, and pagination values passed to query tools
- Model IDs, version IDs, file IDs, hashes, and download URLs
- Destination folder paths and filenames for install/download tools
- Workflow JSON or file paths used by future workflow-scanning helpers
- HTTP responses from Civitai, including filenames, metadata, and hash fields

## Security Invariants

- Read-only tools must not write files or mutate local state.
- Download/install tools must be opt-in and must clearly report their target path before writing.
- Default install behavior should remain dry-run or plan-only.
- Path resolution must not escape the configured ComfyUI root unless the user explicitly asks for an external path.
- File writes should never overwrite existing files silently.
- Hash lookups and duplicate detection should be used to reduce risk, not to bypass validation.
- Tool outputs should preserve structured data and summaries without leaking secrets or raw sensitive paths unnecessarily.

## Main Failure Modes

- A remote Civitai response injects a misleading filename or path-like string that causes a bad local install target.
- A user or agent passes a path traversal payload that escapes the ComfyUI directory.
- A download tool writes to the wrong folder because asset-type inference is too aggressive.
- A cached response hides a newer model or version and causes stale recommendations.
- A write tool overwrites a local file without an explicit overwrite decision.
- A future workflow-scanning tool reports missing assets incorrectly and directs the user to install the wrong file.
- A tool returns raw API payloads that include more information than downstream callers need, increasing accidental exposure.

## Required Controls

- Normalize and validate all paths before writing.
- Keep dry-run as the default for any write-capable tool.
- Require explicit destinations for non-ComfyUI installs.
- Prefer ComfyUI-aware folder mapping, but let users override the target path explicitly.
- Validate hashes and file identity before suggesting duplicates or matching downloads.
- Bound cache lifetimes and treat cache misses as normal.
- Return structured error objects with the exact failing operation and a safe next step.

## Repository-Wide Security Posture

The server is intended to be local-first and user-controlled. The main security objective is preventing accidental or agent-driven filesystem damage while preserving a useful automation surface for model discovery and installation. The codebase should favor explicitness, reversible actions, and clear reports over hidden convenience.

