# Civitai FastMCP Handoff Notes

## Summary

The Civitai FastMCP foundation now ships with a broader, ComfyUI-aware toolset for local model discovery, file inspection, safe install planning, workflow scanning, cache inspection, report export, and agent-facing prompts.

## What Changed

- Added a repo-scoped threat model for the Civitai MCP surface.
- Expanded the server with:
  - top/trending model discovery
  - model file-only inspection
  - creator profile summaries
  - compatibility hints for ComfyUI model placement
  - batch install planning
  - workflow missing-asset scanning
  - cache status and explicit cache clearing
  - model report export in Markdown and JSON
  - MCP prompts for common discovery and install tasks
- Hardened downloads:
  - explicit download URLs are restricted to Civitai domains
  - filenames are sanitized to basenames before local writes
- Updated the package README and tests.

## Verification

- `C:\Users\situa\Desktop\ComfyUI_git\venv\Scripts\python.exe -m unittest tests.test_civitai_mcp -v`
- Live Civitai smoke checks for:
  - top model discovery
  - model file inspection
  - creator profile summaries
  - model report export
- Local server start check:
  - `C:\Users\situa\Desktop\ComfyUI_git\venv\Scripts\python.exe -m tools.civitai_mcp`
  - `C:\Users\situa\Desktop\ComfyUI_git\venv\Scripts\python.exe -m tools.civitai_mcp --transport streamable-http --host 127.0.0.1 --port 8000`
- Docker smoke check:
  - `C:\Users\situa\Desktop\ComfyUI_git\tools\civitai_mcp\docker\run_smoke.ps1`

## GitHub Notes

- The connected GitHub app reports pull-only permissions for `Comfy-Org/ComfyUI` in this session.
- `gh` is not installed in the current environment.
- To publish this work upstream, we will need either a fork with push access or a session with write permissions.
