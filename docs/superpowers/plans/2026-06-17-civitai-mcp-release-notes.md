# Civitai FastMCP Release Notes

## What changed

- Added `civitai_plan_download` as a read-only planner for safe download workflows.
- The new planner detects an existing `.part` file and reports whether the destination looks like a resume candidate, an already present file, or a fresh download target.
- `civitai_export_model_report` now uses the newest model version when building JSON and Markdown reports.
- `civitai_get_model_files` now uses the newest model version when a caller asks by `model_id`.
- `tools/civitai_mcp/README.md` now mentions the new download planner in the quickstart section.

## Behavior notes

- Download planning remains read-only.
- Resume support is still not implemented by the built-in downloader. The planner only reports when a partial file exists and tells the caller to continue with an external resumable downloader if needed.
- The server still keeps install and download paths inside the ComfyUI models tree unless an explicit ComfyUI-root-relative path is provided and validated.

## Verification

- `.\venv\Scripts\python.exe -m unittest tests.test_civitai_mcp -v`
- `.\tools\civitai_mcp\docker\run_smoke.ps1`
