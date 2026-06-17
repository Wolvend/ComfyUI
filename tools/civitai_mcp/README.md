# Civitai FastMCP

Local, read-first MCP server for Civitai with ComfyUI-aware path resolution and install planning.

## Quickstart

Run from `C:\Users\situa\Desktop\ComfyUI_git`:

```powershell
.\venv\Scripts\python.exe -m tools.civitai_mcp
```

Use HTTP transport when a client needs it:

```powershell
.\venv\Scripts\python.exe -m tools.civitai_mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

Optional configuration:

- `CIVITAI_API_KEY` - Civitai API key
- `CIVITAI_MCP_BASE_URL` - Civitai API base URL override
- `CIVITAI_MCP_COMFYUI_ROOT` - ComfyUI root override
- `CIVITAI_MCP_CACHE_DIR` - cache directory override
- `CIVITAI_MCP_TIMEOUT` - request timeout in seconds
- `CIVITAI_MCP_PORT` - streamable HTTP port when that transport is enabled

## What it covers

- Search models, creators, tags, and images
- Inspect model versions, files, and model reports
- Get ComfyUI folder hints for downloads and installs
- Resolve local ComfyUI model folders and detect duplicates
- Scan workflows for likely missing local assets
- Inspect cache state and clear the cache explicitly
- Plan downloads and installs without writing by default

## Minimal examples

- Find a model: `civitai_search_models`
- Check a folder hint: `civitai_get_compatibility_hint`
- Plan a safe install: `civitai_plan_safe_install`
- Scan a workflow: `civitai_scan_workflow_assets`

## Expected folders

The resolver maps common asset types into the local `models/` tree, including:

- `models/checkpoints/`
- `models/loras/`
- `models/vae/`
- `models/embeddings/`
- `models/controlnet/`
- `models/upscale_models/`
- `models/text_encoders/`
- `models/llm/`

## Safety notes

- Read-only tools stay read-only.
- `civitai_download_asset` and `civitai_install_asset` default to `dry_run=True`.
- `civitai_clear_cache` requires `confirm=true`.
- Download URLs are limited to Civitai-owned domains.
- Filenames are sanitized to basenames before writes.
- Explicit destination folders must resolve inside the ComfyUI `models/` tree. Relative folders are treated as ComfyUI-root-relative before validation.
- Workflow scanning reports likely missing assets only; it does not change files.

## Verification

Run the focused package tests:

```powershell
.\venv\Scripts\python.exe -m unittest tests.test_civitai_mcp -v
```

If you run the server over HTTP, verify it starts cleanly on the chosen port:

```powershell
.\venv\Scripts\python.exe -m tools.civitai_mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

Docker smoke harness:

```powershell
.\tools\civitai_mcp\docker\run_smoke.ps1
```

Or run the image directly:

```powershell
docker build -f .\tools\civitai_mcp\docker\Dockerfile.smoke -t civitai-mcp-smoke .
docker run --rm civitai-mcp-smoke
```
