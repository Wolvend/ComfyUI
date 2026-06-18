from __future__ import annotations

import argparse
import html
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from .cache import JsonCache
from .client import CivitaiAPIError, CivitaiClient
from .paths import ComfyUIPaths


class ToolError(BaseModel):
    code: str
    message: str
    suggestion: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    status: Literal["ok", "dry_run", "downloaded", "installed", "cached", "error"] = "ok"
    summary: str
    query: dict[str, Any] | None = None
    data: Any = None
    warnings: list[str] = Field(default_factory=list)
    error: ToolError | None = None


@dataclass(slots=True)
class ServerConfig:
    comfyui_root: Path
    cache_dir: Path
    api_key: str | None = None
    base_url: str = "https://civitai.com/api/v1"
    timeout: float = 30.0
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls, **overrides: Any) -> "ServerConfig":
        comfyui_root = Path(
            overrides.get("comfyui_root")
            or os.getenv("CIVITAI_MCP_COMFYUI_ROOT")
            or os.getenv("COMFYUI_ROOT")
            or Path(__file__).resolve().parents[2]
        ).expanduser().resolve()
        cache_dir = Path(
            overrides.get("cache_dir")
            or os.getenv("CIVITAI_MCP_CACHE_DIR")
            or (comfyui_root / "storage" / "civitai_mcp")
        ).expanduser().resolve()
        api_key = overrides.get("api_key") or os.getenv("CIVITAI_API_KEY") or None
        base_url = overrides.get("base_url") or os.getenv("CIVITAI_API_BASE_URL") or "https://civitai.com/api/v1"
        timeout_value = overrides.get("timeout")
        if timeout_value is None:
            timeout_env = os.getenv("CIVITAI_MCP_TIMEOUT")
            timeout_value = float(timeout_env) if timeout_env and timeout_env.strip() else 30.0
        return cls(
            comfyui_root=comfyui_root,
            cache_dir=cache_dir,
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout_value),
            debug=bool(overrides.get("debug", False)),
            host=str(overrides.get("host") or os.getenv("CIVITAI_MCP_HOST", "127.0.0.1")),
            port=int(overrides.get("port") or os.getenv("CIVITAI_MCP_PORT", 8000)),
        )


def create_server(config: ServerConfig | None = None) -> FastMCP:
    config = config or ServerConfig.from_env()
    cache = JsonCache(config.cache_dir / "cache.json")
    paths = ComfyUIPaths(config.comfyui_root)
    client = CivitaiClient(
        base_url=config.base_url,
        api_key=config.api_key,
        cache=cache,
        timeout=config.timeout,
    )

    app = FastMCP(
        name="Civitai FastMCP Foundation",
        instructions=(
            "Use these tools to search Civitai, inspect models, resolve ComfyUI model folders, "
            "and prepare safe installs. Prefer read-only tools first. "
            "Install tools default to dry-run mode until explicitly disabled."
        ),
        debug=config.debug,
        log_level="DEBUG" if config.debug else "INFO",
        host=config.host,
        port=config.port,
    )

    def envelope(
        *,
        status: str,
        summary: str,
        data: Any = None,
        query: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        error: ToolError | None = None,
    ) -> ToolResult:
        return ToolResult(
            status=status, summary=summary, data=data, query=query, warnings=warnings or [], error=error
        )

    def normalize_html(text: str | None) -> str | None:
        if not text:
            return text
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return html.unescape(cleaned).strip()

    def model_summary(model: dict[str, Any]) -> dict[str, Any]:
        versions = model.get("modelVersions") or []
        latest = latest_model_version(versions)
        return {
            "id": model.get("id"),
            "name": model.get("name"),
            "type": model.get("type"),
            "creator": model.get("creator"),
            "tags": [tag.get("name") for tag in model.get("tags", []) if isinstance(tag, dict)],
            "stats": model.get("stats"),
            "nsfwLevel": model.get("nsfwLevel"),
            "availability": model.get("availability"),
            "supportsGeneration": model.get("supportsGeneration"),
            "description": normalize_html(model.get("description")),
            "url": f"https://civitai.com/models/{model.get('id')}" if model.get("id") else None,
            "versionCount": len(versions),
            "latestVersion": version_summary(latest) if latest else None,
        }

    def latest_model_version(versions: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not versions:
            return None

        def sort_key(version: dict[str, Any]) -> tuple[str, int, int]:
            published = str(version.get("publishedAt") or "")
            index = version.get("index")
            try:
                index_value = int(index)
            except (TypeError, ValueError):
                index_value = -1
            try:
                version_id = int(version.get("id"))
            except (TypeError, ValueError):
                version_id = -1
            return (published, index_value, version_id)

        return max(versions, key=sort_key)

    def version_summary(version: dict[str, Any] | None) -> dict[str, Any] | None:
        if not version:
            return None
        files = version.get("files") or []
        primary = next((item for item in files if item.get("primary")), files[0] if files else None)
        return {
            "id": version.get("id"),
            "name": version.get("name"),
            "index": version.get("index"),
            "baseModel": version.get("baseModel"),
            "baseModelType": version.get("baseModelType"),
            "publishedAt": version.get("publishedAt"),
            "status": version.get("status"),
            "availability": version.get("availability"),
            "trainedWords": version.get("trainedWords") or [],
            "vaeId": version.get("vaeId"),
            "stats": version.get("stats"),
            "downloadUrl": version.get("downloadUrl"),
            "fileCount": len(files),
            "imageCount": len(version.get("images") or []),
            "primaryFile": file_summary(primary) if primary else None,
        }

    def file_summary(file_entry: dict[str, Any] | None) -> dict[str, Any] | None:
        if not file_entry:
            return None
        return {
            "id": file_entry.get("id"),
            "name": file_entry.get("name"),
            "type": file_entry.get("type"),
            "sizeKB": file_entry.get("sizeKB"),
            "primary": file_entry.get("primary"),
            "downloadUrl": file_entry.get("downloadUrl"),
            "hashes": file_entry.get("hashes"),
            "scan": {
                "pickle": file_entry.get("pickleScanResult"),
                "virus": file_entry.get("virusScanResult"),
                "scannedAt": file_entry.get("scannedAt"),
            },
        }

    def image_summary(image: dict[str, Any]) -> dict[str, Any]:
        meta = image.get("meta") or image.get("metadata")
        prompt = None
        negative_prompt = None
        if isinstance(meta, dict):
            prompt = meta.get("prompt")
            negative_prompt = meta.get("negativePrompt")
        return {
            "id": image.get("id"),
            "url": image.get("url"),
            "width": image.get("width"),
            "height": image.get("height"),
            "nsfw": image.get("nsfw"),
            "nsfwLevel": image.get("nsfwLevel"),
            "type": image.get("type"),
            "createdAt": image.get("createdAt"),
            "postId": image.get("postId"),
            "username": image.get("username"),
            "hasMeta": image.get("hasMeta"),
            "hasPositivePrompt": image.get("hasPositivePrompt"),
            "prompt": prompt,
            "negativePrompt": negative_prompt,
            "meta": meta,
            "stats": image.get("stats"),
        }

    def creator_summary(creator: dict[str, Any]) -> dict[str, Any]:
        return {
            "username": creator.get("username"),
            "modelCount": creator.get("modelCount"),
            "link": creator.get("link"),
        }

    def tag_summary(tag: dict[str, Any]) -> dict[str, Any]:
        return {"name": tag.get("name"), "modelCount": tag.get("modelCount"), "link": tag.get("link")}

    def clean_query(**params: Any) -> dict[str, Any]:
        return {key: value for key, value in params.items() if value is not None and value != ""}

    def normalize_limit(limit: int | None, default: int, maximum: int = 200) -> int:
        value = default if limit is None else limit
        if value < 0:
            raise ValueError("limit must be non-negative")
        if value > maximum:
            raise ValueError(f"limit must be <= {maximum}")
        return value

    def normalize_path(path_value: str | None) -> Path | None:
        if not path_value:
            return None
        candidate = Path(path_value).expanduser()
        if not candidate.is_absolute():
            candidate = config.comfyui_root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(paths.models_root)
        except ValueError as exc:
            raise ValueError(
                f"destination_folder must resolve inside the ComfyUI models tree ({paths.models_root})."
            ) from exc
        return resolved

    def read_workflow_text(path_value: str) -> str:
        workflow_path = Path(path_value).expanduser().resolve()
        if workflow_path.suffix.lower() != ".json":
            raise ValueError("workflow_path must point to a .json workflow file.")
        if workflow_path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("workflow_path is too large; provide workflow_json for explicit inline scanning.")
        return workflow_path.read_text(encoding="utf-8")

    def safe_relative(path_value: Path) -> str | None:
        try:
            return path_value.relative_to(config.comfyui_root).as_posix()
        except ValueError:
            return None

    def error_result(operation: str, exc: Exception) -> ToolResult:
        if isinstance(exc, CivitaiAPIError):
            return envelope(
                status="error",
                summary=f"{operation} failed against Civitai.",
                error=ToolError(
                    code="http_error",
                    message=str(exc),
                    suggestion="Check the Civitai API key, rate limits, and whether the endpoint still exists.",
                    details={"url": exc.url, "status_code": exc.status_code, "operation": exc.operation},
                ),
            )
        return envelope(
            status="error",
            summary=f"{operation} failed.",
            error=ToolError(
                code=exc.__class__.__name__,
                message=str(exc),
                suggestion="Inspect the arguments and retry the tool with a smaller, explicit request.",
            ),
        )

    def model_search_params(
        query: str | None,
        tag: str | None,
        username: str | None,
        sort: str | None,
        period: str | None,
        limit: int | None,
        cursor: str | None,
        page: int | None,
        base_model: str | None,
        model_type: str | None,
        nsfw: bool | None,
        include_nsfw: bool | None,
    ) -> dict[str, Any]:
        params = clean_query(
            query=query,
            tag=tag,
            username=username,
            sort=sort,
            period=period,
            limit=normalize_limit(limit, default=20),
            cursor=cursor,
            page=page,
            baseModel=base_model,
            type=model_type,
            nsfw=nsfw,
            includeNSFW=include_nsfw,
        )
        return params

    def image_search_params(
        *,
        limit: int | None,
        page: int | None,
        cursor: str | None,
        post_id: int | None,
        model_id: int | None,
        model_version_id: int | None,
        username: str | None,
        nsfw: bool | None,
        sort: str | None,
        period: str | None,
    ) -> dict[str, Any]:
        return clean_query(
            limit=normalize_limit(limit, default=100),
            page=page,
            cursor=cursor,
            postId=post_id,
            modelId=model_id,
            modelVersionId=model_version_id,
            username=username,
            nsfw=nsfw,
            sort=sort,
            period=period,
        )

    def install_plan(
        *,
        download_url: str | None = None,
        model_version_id: int | None = None,
        file_hash: str | None = None,
        file_id: int | None = None,
        asset_type: str | None = None,
        destination_folder: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        resolved_download_url, source = client.resolve_download_url(
            download_url=download_url,
            model_version_id=model_version_id,
            file_hash=file_hash,
            file_id=file_id,
        )
        version = None
        if model_version_id is not None:
            version = client.get_model_version(model_version_id)
        elif file_hash:
            version = client.get_model_version_by_hash(file_hash)
        file_entry = None
        if version:
            if file_id is not None:
                file_entry = next((item for item in version.get("files", []) if str(item.get("id")) == str(file_id)), None)
            else:
                file_entry = next((item for item in version.get("files", []) if item.get("primary")), None)
                if file_entry is None and version.get("files"):
                    file_entry = version["files"][0]
        source_file = file_summary(file_entry) if file_entry else None
        expected_size_kb = file_entry.get("sizeKB") if file_entry else None
        expected_size_bytes: int | None = None
        if expected_size_kb is not None:
            try:
                expected_size_bytes = int(float(expected_size_kb) * 1024)
            except (TypeError, ValueError):
                expected_size_bytes = None
        url_filename = Path(re.sub(r"[?#].*$", "", Path(urlparse(resolved_download_url).path).name or "")).name
        raw_filename = filename or (file_entry or {}).get("name") or url_filename or "download.bin"
        chosen_filename = Path(str(raw_filename)).name or "download.bin"
        resolved_destination_folder = normalize_path(destination_folder)
        if resolved_destination_folder is None:
            version_base_model = version.get("baseModel") if version else None
            resolved_destination_folder = paths.resolve_model_folder(asset_type, chosen_filename, version_base_model)
        return {
            "source": source,
            "download_url": resolved_download_url,
            "filename": chosen_filename,
            "filename_sanitized": chosen_filename != str(raw_filename),
            "destination_folder": str(resolved_destination_folder),
            "destination_path": str(resolved_destination_folder / chosen_filename),
            "resolved_folder": str(resolved_destination_folder),
            "resolved_folder_relative": safe_relative(resolved_destination_folder),
            "source_model_version": version_summary(version) if version else None,
            "source_file": source_file,
            "expected_size_kb": expected_size_kb,
            "expected_size_bytes": expected_size_bytes,
            "existing_path": str((resolved_destination_folder / chosen_filename)) if (resolved_destination_folder / chosen_filename).exists() else None,
        }

    def download_plan(
        *,
        download_url: str | None = None,
        model_version_id: int | None = None,
        file_hash: str | None = None,
        file_id: int | None = None,
        asset_type: str | None = None,
        destination_folder: str | None = None,
        destination_filename: str | None = None,
    ) -> dict[str, Any]:
        plan = install_plan(
            download_url=download_url,
            model_version_id=model_version_id,
            file_hash=file_hash,
            file_id=file_id,
            asset_type=asset_type,
            destination_folder=destination_folder,
            filename=destination_filename,
        )
        destination_path = Path(plan["destination_path"])
        partial_path = destination_path.with_name(f"{destination_path.name}.part")
        expected_size_bytes = plan.get("expected_size_bytes")
        destination_exists = destination_path.exists()
        partial_exists = partial_path.exists()
        existing_size_bytes = destination_path.stat().st_size if destination_exists else None
        partial_size_bytes = partial_path.stat().st_size if partial_exists else None

        download_state = "ready_to_download"
        if destination_exists and expected_size_bytes is not None and existing_size_bytes is not None and existing_size_bytes >= expected_size_bytes:
            download_state = "already_present"
        elif partial_exists:
            if expected_size_bytes is not None and partial_size_bytes is not None and partial_size_bytes < expected_size_bytes:
                download_state = "resume_candidate"
            else:
                download_state = "partial_present"
        elif expected_size_bytes is None:
            download_state = "size_unknown"

        if download_state == "already_present":
            next_step = "No download is needed unless you want overwrite=true for a fresh copy."
        elif download_state == "resume_candidate":
            next_step = (
                "A partial file exists, but the built-in downloader does not resume range requests. "
                "Keep the partial file intact and use an external resumable downloader if you want to continue safely."
            )
        elif download_state == "partial_present":
            next_step = (
                "A partial file exists, but the final file size is unknown or already matches. "
                "Inspect the partial file before writing anything new."
            )
        elif download_state == "size_unknown":
            next_step = "The source file size was not exposed by metadata, so verify the destination manually before writing."
        else:
            next_step = "Call civitai_download_asset with dry_run=false to write the file."

        return {
            **plan,
            "destination_exists": destination_exists,
            "existing_size_bytes": existing_size_bytes,
            "partial_path": str(partial_path),
            "partial_exists": partial_exists,
            "partial_size_bytes": partial_size_bytes,
            "download_state": download_state,
            "resume_supported": False,
            "next_step": next_step,
        }

    def infer_asset_type(filename: str | None, explicit_asset_type: str | None, model_type: str | None, base_model: str | None) -> tuple[str, list[str]]:
        reasons: list[str] = []
        if explicit_asset_type:
            reasons.append(f"explicit asset_type={explicit_asset_type}")
            return explicit_asset_type, reasons

        if not filename:
            return model_type or base_model or "checkpoint", ["no filename given"]

        lower = filename.lower()
        if lower.endswith(".gguf"):
            if any(hint in lower for hint in ("clip", "text", "encoder", "ltx", "gemma", "qwen", "llama", "mistral", "phi")):
                reasons.append("gguf text/encoder filename")
                return "text_encoder", reasons
            reasons.append("gguf model filename")
            return "llm", reasons
        if "lora" in lower or "lycoris" in lower:
            reasons.append("filename indicates lora")
            return "lora", reasons
        if "vae" in lower:
            reasons.append("filename indicates vae")
            return "vae", reasons
        if "embed" in lower or "textual" in lower:
            reasons.append("filename indicates embeddings")
            return "embedding", reasons
        if model_type:
            reasons.append(f"model_type={model_type}")
            return model_type, reasons
        if base_model:
            reasons.append(f"base_model={base_model}")
            return base_model, reasons
        return "checkpoint", ["default checkpoint fallback"]

    def compatibility_hint(
        *,
        filename: str | None,
        asset_type: str | None,
        base_model: str | None,
        model_type: str | None,
    ) -> dict[str, Any]:
        inferred_type, reasons = infer_asset_type(filename, asset_type, model_type, base_model)
        folder = paths.resolve_model_folder(inferred_type, filename, base_model)
        confidence = 0.95 if asset_type else 0.8 if filename else 0.5
        notes = [
            "Use the suggested ComfyUI folder unless the workflow explicitly expects a different location.",
            "When the file is a GGUF encoder or text projection, prefer models/text_encoders.",
        ]
        if folder.name == "llm":
            notes.append("Plain GGUF model files usually belong under models/llm.")
        if folder.name == "text_encoders":
            notes.append("Encoder-style GGUF files usually belong under models/text_encoders.")
        return {
            "inferred_asset_type": inferred_type,
            "recommended_folder": str(folder),
            "recommended_folder_relative": safe_relative(folder),
            "confidence": confidence,
            "reasons": reasons,
            "notes": notes,
        }

    def local_asset_matches(filename: str, root: Path | None = None) -> list[str]:
        base = root or paths.models_root
        if not base.exists():
            return []
        target_name = Path(filename).name
        try:
            matches = [str(path) for path in base.rglob("*") if path.is_file() and path.name == target_name]
        except OSError:
            matches = []
        return matches[:20]

    def extract_workflow_asset_candidates(workflow: Any) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []

        def walk(value: Any, key_path: list[str]) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    walk(item, key_path + [str(key)])
                return
            if isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, key_path + [str(index)])
                return
            if not isinstance(value, str):
                return

            lowered = value.lower()
            key_blob = "/".join(key_path).lower()
            looks_like_file = bool(
                re.search(r"\.(safetensors|ckpt|pt|bin|gguf|vae|onnx|pth|sft|ggml)$", lowered)
                or re.search(r"(models?[\\/])|([\\/](lora|loras|vae|clip|text_encoders|llm|upscale|checkpoint)s?[\\/])", lowered)
                or any(token in key_blob for token in ("model", "vae", "clip", "lora", "checkpoint", "upscale", "encoder", "weight", "file", "path"))
            )
            if not looks_like_file:
                return

            filename = Path(re.sub(r"[?#].*$", "", value)).name
            if not filename:
                return
            asset_type_hint, reasons = infer_asset_type(filename, None, None, None)
            folder = paths.resolve_model_folder(asset_type_hint, filename, None)
            matches = local_asset_matches(filename, folder)
            candidates.append(
                {
                    "source_value": value,
                    "filename": filename,
                    "key_path": "/".join(key_path),
                    "asset_type_hint": asset_type_hint,
                    "recommended_folder": str(folder),
                    "recommended_folder_relative": safe_relative(folder),
                    "local_matches": matches,
                    "missing": len(matches) == 0,
                    "reasons": reasons,
                }
            )

        walk(workflow, [])
        return candidates

    def batch_install_plan(items: list[dict[str, Any]], defaults: dict[str, Any]) -> dict[str, Any]:
        planned: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            merged = {**defaults, **item}
            if "destination_filename" in merged and "filename" not in merged:
                merged["filename"] = merged.pop("destination_filename")
            else:
                merged.pop("destination_filename", None)
            planned.append({"index": index, "plan": install_plan(**merged)})
        return {
            "item_count": len(items),
            "planned_count": len(planned),
            "plans": planned,
        }

    def render_model_report(model: dict[str, Any]) -> dict[str, Any]:
        latest = latest_model_version(model.get("modelVersions") or [])
        latest_summary = version_summary(latest) if latest else None
        file_rows = [file_summary(item) for item in (latest.get("files", []) if latest else [])]
        markdown_lines = [
            f"# {model.get('name')}",
            "",
            f"- Model ID: {model.get('id')}",
            f"- Type: {model.get('type')}",
            f"- Creator: {model.get('creator', {}).get('username') if isinstance(model.get('creator'), dict) else model.get('creator')}",
            f"- Tags: {', '.join(tag.get('name') for tag in model.get('tags', []) if isinstance(tag, dict)) or 'none'}",
            f"- URL: https://civitai.com/models/{model.get('id')}",
            f"- Description: {normalize_html(model.get('description')) or 'n/a'}",
        ]
        if latest_summary:
            markdown_lines.extend(
                [
                    "",
                    "## Latest Version",
                    "",
                    f"- Version ID: {latest_summary.get('id')}",
                    f"- Name: {latest_summary.get('name')}",
                    f"- Base Model: {latest_summary.get('baseModel')}",
                    f"- Published: {latest_summary.get('publishedAt')}",
                    f"- Trained Words: {', '.join(latest_summary.get('trainedWords') or []) or 'none'}",
                ]
            )
        if file_rows:
            markdown_lines.extend(
                [
                    "",
                    "## Files",
                    "",
                    "| Name | Size KB | Primary | SHA256 |",
                    "| --- | ---: | --- | --- |",
                ]
            )
            for row in file_rows:
                sha256 = (row.get("hashes") or {}).get("SHA256", "")
                markdown_lines.append(
                    f"| {row.get('name')} | {row.get('sizeKB')} | {bool(row.get('primary'))} | {sha256} |"
                )
        return {
            "json": {
                "model": model_summary(model),
                "latest_version": latest_summary,
                "files": file_rows,
            },
            "markdown": "\n".join(markdown_lines).strip(),
        }

    @app.tool(
        name="civitai_search_models",
        description="Search Civitai models with filters and return normalized results plus pagination metadata.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_search_models(
        query: str | None = None,
        tag: str | None = None,
        username: str | None = None,
        sort: str | None = None,
        period: str | None = None,
        limit: int | None = 20,
        cursor: str | None = None,
        page: int | None = None,
        base_model: str | None = None,
        model_type: str | None = None,
        nsfw: bool | None = None,
        include_nsfw: bool | None = None,
        include_raw: bool = False,
    ) -> ToolResult:
        try:
            params = model_search_params(query, tag, username, sort, period, limit, cursor, page, base_model, model_type, nsfw, include_nsfw)
            data = client.search_models(**params)
            items = [model_summary(item) for item in data.get("items", [])]
            payload: dict[str, Any] = {"items": items, "metadata": data.get("metadata", {})}
            if include_raw:
                payload["raw"] = data
            return envelope(
                status="ok",
                summary=f"Found {len(items)} models on Civitai.",
                query=params,
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Search models", exc)

    @app.tool(
        name="civitai_get_top_models",
        description="Get top Civitai models using a high-signal sorting and time window.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_get_top_models(
        limit: int | None = 20,
        sort: str = "Most Downloaded",
        period: str = "AllTime",
        include_raw: bool = False,
    ) -> ToolResult:
        try:
            params = clean_query(limit=normalize_limit(limit, default=20), sort=sort, period=period)
            data = client.search_models(**params)
            items = [model_summary(item) for item in data.get("items", [])]
            payload: dict[str, Any] = {"items": items, "metadata": data.get("metadata", {}), "sort": sort, "period": period}
            if include_raw:
                payload["raw"] = data
            return envelope(
                status="ok",
                summary=f"Found {len(items)} top models using {sort} over {period}.",
                query=params,
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Get top models", exc)

    @app.tool(
        name="civitai_get_model_files",
        description="Inspect only the files for a model version, including hashes and scan metadata.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_get_model_files(
        model_id: int | None = None,
        version_id: int | None = None,
        file_hash: str | None = None,
        include_raw: bool = False,
    ) -> ToolResult:
        try:
            if file_hash:
                data = client.get_model_version_by_hash(file_hash)
            elif version_id is not None:
                data = client.get_model_version(version_id)
            elif model_id is not None:
                model = client.get_model(model_id)
                versions = model.get("modelVersions") or []
                if not versions:
                    raise ValueError(f"Model {model_id} does not expose any versions.")
                data = latest_model_version(versions) or versions[0]
            else:
                raise ValueError("Provide model_id, version_id, or file_hash.")

            files = [file_summary(item) for item in data.get("files", [])]
            payload: dict[str, Any] = {
                "version": version_summary(data),
                "files": files,
                "file_count": len(files),
            }
            if include_raw:
                payload["raw"] = data
            return envelope(
                status="ok",
                summary=f"Found {len(files)} files for model version {data.get('id')}.",
                query={"model_id": model_id, "version_id": version_id, "file_hash": file_hash},
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Get model files", exc)

    @app.tool(
        name="civitai_get_creator_profile",
        description="Summarize a Civitai creator profile using their creator listing and recent models.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_get_creator_profile(
        username: str,
        model_limit: int | None = 10,
        include_raw: bool = False,
    ) -> ToolResult:
        try:
            creator_results = client.search_creators(query=username, limit=10)
            model_results = client.search_models(username=username, limit=normalize_limit(model_limit, default=10), sort="Most Downloaded")
            creator_match = next((item for item in creator_results.get("items", []) if item.get("username", "").lower() == username.lower()), None)
            payload: dict[str, Any] = {
                "creator": creator_summary(creator_match) if creator_match else {"username": username},
                "creators": [creator_summary(item) for item in creator_results.get("items", [])],
                "models": [model_summary(item) for item in model_results.get("items", [])],
                "creator_metadata": creator_results.get("metadata", {}),
                "model_metadata": model_results.get("metadata", {}),
            }
            if include_raw:
                payload["raw"] = {"creators": creator_results, "models": model_results}
            return envelope(
                status="ok",
                summary=f"Loaded creator profile for {username}.",
                query={"username": username, "model_limit": model_limit},
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Get creator profile", exc)

    @app.tool(
        name="civitai_get_compatibility_hint",
        description="Infer the best ComfyUI folder and usage notes for a Civitai asset.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    def civitai_get_compatibility_hint(
        filename: str | None = None,
        asset_type: str | None = None,
        base_model: str | None = None,
        model_type: str | None = None,
    ) -> ToolResult:
        try:
            payload = compatibility_hint(
                filename=filename,
                asset_type=asset_type,
                base_model=base_model,
                model_type=model_type,
            )
            return envelope(
                status="ok",
                summary=f"Suggested {payload['recommended_folder_relative']} for {filename or asset_type or 'asset'}.",
                query={"filename": filename, "asset_type": asset_type, "base_model": base_model, "model_type": model_type},
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Get compatibility hint", exc)

    @app.tool(
        name="civitai_plan_batch_install",
        description="Plan multiple Civitai installs in one pass without writing files.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_plan_batch_install(
        items_json: str,
        default_asset_type: str | None = None,
        default_destination_folder: str | None = None,
        dry_run: bool = True,
    ) -> ToolResult:
        try:
            raw_items = json.loads(items_json)
            if not isinstance(raw_items, list):
                raise ValueError("items_json must decode to a JSON list of install items.")
            items: list[dict[str, Any]] = []
            for item in raw_items:
                if not isinstance(item, dict):
                    raise ValueError("Each batch install item must be a JSON object.")
                items.append(item)
            defaults = {
                "asset_type": default_asset_type,
                "destination_folder": default_destination_folder,
            }
            payload = batch_install_plan(items, defaults)
            payload["dry_run"] = dry_run
            return envelope(
                status="dry_run" if dry_run else "ok",
                summary=f"Planned {payload['planned_count']} install items.",
                query={"item_count": len(items), "default_asset_type": default_asset_type, "default_destination_folder": default_destination_folder},
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Plan batch install", exc)

    @app.tool(
        name="civitai_scan_workflow_assets",
        description="Scan a ComfyUI workflow for local asset references and likely-missing model files.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    def civitai_scan_workflow_assets(
        workflow_json: str | None = None,
        workflow_path: str | None = None,
    ) -> ToolResult:
        try:
            if workflow_path:
                workflow_text = read_workflow_text(workflow_path)
            elif workflow_json is not None:
                workflow_text = workflow_json
            else:
                raise ValueError("Provide workflow_json or workflow_path.")

            workflow = json.loads(workflow_text)
            candidates = extract_workflow_asset_candidates(workflow)
            missing = [item for item in candidates if item["missing"]]
            payload = {
                "candidate_count": len(candidates),
                "missing_count": len(missing),
                "candidates": candidates,
                "missing": missing,
            }
            return envelope(
                status="ok",
                summary=f"Scanned workflow and found {len(missing)} likely-missing assets.",
                query={"workflow_path": workflow_path},
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Scan workflow assets", exc)

    @app.tool(
        name="civitai_cache_status",
        description="Inspect the local Civitai metadata cache.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    def civitai_cache_status() -> ToolResult:
        try:
            return envelope(
                status="ok",
                summary="Loaded cache status.",
                data=cache.stats(),
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Cache status", exc)

    @app.tool(
        name="civitai_clear_cache",
        description="Clear the local Civitai metadata cache after explicit confirmation.",
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False),
        structured_output=True,
    )
    def civitai_clear_cache(confirm: bool = False) -> ToolResult:
        try:
            if not confirm:
                return envelope(
                    status="dry_run",
                    summary="Cache clear not executed because confirm=false.",
                    data={"cache": cache.stats(), "next_step": "Call again with confirm=true to clear the cache."},
                )
            cache.clear()
            return envelope(
                status="ok",
                summary="Cleared the local Civitai metadata cache.",
                data={"cleared": True},
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Clear cache", exc)

    @app.tool(
        name="civitai_get_model",
        description="Get a single Civitai model with versions, files, and image summaries.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_get_model(model_id: int, include_raw: bool = False) -> ToolResult:
        try:
            data = client.get_model(model_id)
            payload = {
                "model": model_summary(data),
                "versions": [version_summary(version) for version in data.get("modelVersions", [])],
            }
            if include_raw:
                payload["raw"] = data
            return envelope(
                status="ok",
                summary=f"Loaded model {data.get('name')} ({model_id}).",
                query={"model_id": model_id},
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Get model", exc)

    @app.tool(
        name="civitai_export_model_report",
        description="Export a model report in Markdown and JSON for agent review or offline saving.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_export_model_report(
        model_id: int,
        format: Literal["both", "json", "markdown"] = "both",
        include_raw: bool = False,
    ) -> ToolResult:
        try:
            data = client.get_model(model_id)
            report = render_model_report(data)
            payload: dict[str, Any] = {}
            if format in ("both", "json"):
                payload["json"] = report["json"]
                if include_raw:
                    payload["raw"] = data
            if format in ("both", "markdown"):
                payload["markdown"] = report["markdown"]
            return envelope(
                status="ok",
                summary=f"Generated model report for {data.get('name')} in {format} format.",
                query={"model_id": model_id, "format": format},
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Export model report", exc)

    @app.tool(
        name="civitai_get_model_version",
        description="Get one Civitai model version with file and image details.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_get_model_version(version_id: int, include_raw: bool = False) -> ToolResult:
        try:
            data = client.get_model_version(version_id)
            payload = {"version": version_summary(data), "files": [file_summary(item) for item in data.get("files", [])], "images": [image_summary(item) for item in data.get("images", [])]}
            if include_raw:
                payload["raw"] = data
            return envelope(
                status="ok",
                summary=f"Loaded model version {version_id}.",
                query={"version_id": version_id},
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Get model version", exc)

    @app.tool(
        name="civitai_get_model_version_by_hash",
        description="Look up a model version by file hash and return the matching version data.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_get_model_version_by_hash(file_hash: str, include_raw: bool = False) -> ToolResult:
        try:
            data = client.get_model_version_by_hash(file_hash)
            payload = {"version": version_summary(data), "files": [file_summary(item) for item in data.get("files", [])], "images": [image_summary(item) for item in data.get("images", [])]}
            if include_raw:
                payload["raw"] = data
            return envelope(
                status="ok",
                summary=f"Resolved hash {file_hash[:12]} to model version {data.get('id')}.",
                query={"file_hash": file_hash},
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Hash lookup", exc)

    @app.tool(
        name="civitai_search_images",
        description="Search Civitai images and return prompt metadata when available.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_search_images(
        limit: int | None = 100,
        page: int | None = None,
        cursor: str | None = None,
        post_id: int | None = None,
        model_id: int | None = None,
        model_version_id: int | None = None,
        username: str | None = None,
        nsfw: bool | None = None,
        sort: str | None = None,
        period: str | None = None,
        include_raw: bool = False,
    ) -> ToolResult:
        try:
            params = image_search_params(
                limit=limit,
                page=page,
                cursor=cursor,
                post_id=post_id,
                model_id=model_id,
                model_version_id=model_version_id,
                username=username,
                nsfw=nsfw,
                sort=sort,
                period=period,
            )
            data = client.search_images(**params)
            items = [image_summary(item) for item in data.get("items", [])]
            payload: dict[str, Any] = {"items": items, "metadata": data.get("metadata", {})}
            if include_raw:
                payload["raw"] = data
            return envelope(
                status="ok",
                summary=f"Found {len(items)} images on Civitai.",
                query=params,
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Search images", exc)

    @app.tool(
        name="civitai_get_top_images",
        description="Convenience wrapper for the most active Civitai images and their metadata.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_get_top_images(
        limit: int | None = 20,
        sort: str = "Most Reactions",
        period: str = "AllTime",
        nsfw: bool | None = None,
    ) -> ToolResult:
        return civitai_search_images(limit=limit, sort=sort, period=period, nsfw=nsfw)

    @app.tool(
        name="civitai_search_creators",
        description="Search Civitai creators and return their model counts and profile links.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_search_creators(query: str | None = None, limit: int | None = 20, page: int | None = None, include_raw: bool = False) -> ToolResult:
        try:
            params = clean_query(query=query, limit=normalize_limit(limit, default=20), page=page)
            data = client.search_creators(**params)
            items = [creator_summary(item) for item in data.get("items", [])]
            payload: dict[str, Any] = {"items": items, "metadata": data.get("metadata", {})}
            if include_raw:
                payload["raw"] = data
            return envelope(
                status="ok",
                summary=f"Found {len(items)} creators on Civitai.",
                query=params,
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Search creators", exc)

    @app.tool(
        name="civitai_search_tags",
        description="Search Civitai tags and return the matching tag names and model links.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_search_tags(query: str | None = None, limit: int | None = 20, page: int | None = None, include_raw: bool = False) -> ToolResult:
        try:
            params = clean_query(query=query, limit=normalize_limit(limit, default=20), page=page)
            data = client.search_tags(**params)
            items = [tag_summary(item) for item in data.get("items", [])]
            payload: dict[str, Any] = {"items": items, "metadata": data.get("metadata", {})}
            if include_raw:
                payload["raw"] = data
            return envelope(status="ok", summary=f"Found {len(items)} tags on Civitai.", query=params, data=payload)
        except Exception as exc:  # noqa: BLE001
            return error_result("Search tags", exc)

    @app.tool(
        name="civitai_list_comfyui_model_folders",
        description="List the model folders currently visible in the local ComfyUI installation.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    def civitai_list_comfyui_model_folders() -> ToolResult:
        try:
            folders = paths.list_model_folders()
            return envelope(
                status="ok",
                summary=f"Found {len(folders)} model folders under ComfyUI models/.",
                data={"comfyui_root": str(config.comfyui_root), "models_root": str(paths.models_root), "folders": folders},
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("List ComfyUI model folders", exc)

    @app.tool(
        name="civitai_resolve_comfyui_model_path",
        description="Resolve the best local ComfyUI destination folder for a Civitai asset.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    def civitai_resolve_comfyui_model_path(
        asset_type: str | None = None,
        filename: str | None = None,
        base_model: str | None = None,
    ) -> ToolResult:
        try:
            folder = paths.resolve_model_folder(asset_type, filename, base_model)
            return envelope(
                status="ok",
                summary=f"Resolved {asset_type or 'asset'} to {folder}.",
                query={"asset_type": asset_type, "filename": filename, "base_model": base_model},
                data={
                    "folder": str(folder),
                    "folder_relative": safe_relative(folder),
                    "comfyui_root": str(config.comfyui_root),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Resolve ComfyUI path", exc)

    @app.tool(
        name="civitai_detect_local_duplicates",
        description="Scan a local ComfyUI folder for duplicate files using size and content hashes.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    def civitai_detect_local_duplicates(folder: str | None = None) -> ToolResult:
        try:
            scan_folder = normalize_path(folder) if folder else paths.models_root
            duplicates = paths.scan_duplicate_files(scan_folder)
            return envelope(
                status="ok",
                summary=f"Found {len(duplicates)} duplicate hash groups under {scan_folder}.",
                query={"folder": str(scan_folder)},
                data={"folder": str(scan_folder), "duplicate_groups": duplicates},
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Detect duplicates", exc)

    def write_asset(
        *,
        download_url: str | None = None,
        model_version_id: int | None = None,
        file_hash: str | None = None,
        file_id: int | None = None,
        destination_folder: str | None = None,
        destination_filename: str | None = None,
        asset_type: str | None = None,
        dry_run: bool = True,
        overwrite: bool = False,
    ) -> ToolResult:
        plan = install_plan(
            download_url=download_url,
            model_version_id=model_version_id,
            file_hash=file_hash,
            file_id=file_id,
            asset_type=asset_type,
            destination_folder=destination_folder,
            filename=destination_filename,
        )
        if dry_run:
            return envelope(status="dry_run", summary="Install plan prepared; no files were written.", data=plan)

        resolved_download_url = plan["download_url"]
        destination_path = Path(plan["destination_path"])
        if destination_path.exists() and not overwrite:
            return envelope(
                status="cached",
                summary=f"File already exists at {destination_path}.",
                data={**plan, "existing_path": str(destination_path)},
            )
        downloaded_path = client.download(resolved_download_url, destination_path)
        return envelope(
            status="downloaded" if destination_folder is None else "installed",
            summary=f"Wrote asset to {downloaded_path}.",
            data={**plan, "written_path": str(downloaded_path)},
        )

    @app.tool(
        name="civitai_download_asset",
        description="Download a Civitai file to temp or a chosen folder. Dry-run is the default.",
        annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, openWorldHint=True),
        structured_output=True,
    )
    def civitai_download_asset(
        download_url: str | None = None,
        model_version_id: int | None = None,
        file_hash: str | None = None,
        file_id: int | None = None,
        destination_folder: str | None = None,
        destination_filename: str | None = None,
        asset_type: str | None = None,
        dry_run: bool = True,
        overwrite: bool = False,
    ) -> ToolResult:
        try:
            return write_asset(
                download_url=download_url,
                model_version_id=model_version_id,
                file_hash=file_hash,
                file_id=file_id,
                destination_folder=destination_folder,
                destination_filename=destination_filename,
                asset_type=asset_type,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Download asset", exc)

    @app.tool(
        name="civitai_install_asset",
        description="Install a Civitai file directly into a chosen ComfyUI model folder. Dry-run is the default.",
        annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, openWorldHint=True),
        structured_output=True,
    )
    def civitai_install_asset(
        download_url: str | None = None,
        model_version_id: int | None = None,
        file_hash: str | None = None,
        file_id: int | None = None,
        destination_folder: str | None = None,
        destination_filename: str | None = None,
        asset_type: str | None = None,
        dry_run: bool = True,
        overwrite: bool = False,
    ) -> ToolResult:
        try:
            return write_asset(
                download_url=download_url,
                model_version_id=model_version_id,
                file_hash=file_hash,
                file_id=file_id,
                destination_folder=destination_folder,
                destination_filename=destination_filename,
                asset_type=asset_type,
                dry_run=dry_run,
                overwrite=overwrite,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Install asset", exc)

    @app.tool(
        name="civitai_plan_download",
        description="Plan a safe download into ComfyUI and report whether a partial file could be resumed safely.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True),
        structured_output=True,
    )
    def civitai_plan_download(
        download_url: str | None = None,
        model_version_id: int | None = None,
        file_hash: str | None = None,
        file_id: int | None = None,
        destination_folder: str | None = None,
        destination_filename: str | None = None,
        asset_type: str | None = None,
    ) -> ToolResult:
        try:
            payload = download_plan(
                download_url=download_url,
                model_version_id=model_version_id,
                file_hash=file_hash,
                file_id=file_id,
                destination_folder=destination_folder,
                destination_filename=destination_filename,
                asset_type=asset_type,
            )
            return envelope(
                status="ok",
                summary=f"Planned download for {payload['filename']} into {payload['resolved_folder_relative'] or payload['resolved_folder']}.",
                query={
                    "download_url": download_url,
                    "model_version_id": model_version_id,
                    "file_hash": file_hash,
                    "file_id": file_id,
                    "destination_folder": destination_folder,
                    "destination_filename": destination_filename,
                    "asset_type": asset_type,
                },
                data=payload,
            )
        except Exception as exc:  # noqa: BLE001
            return error_result("Plan download", exc)

    @app.tool(
        name="civitai_server_status",
        description="Report the local Civitai MCP server configuration and ComfyUI path state.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
        structured_output=True,
    )
    def civitai_server_status() -> ToolResult:
        return envelope(
            status="ok",
            summary="Civitai MCP server is configured.",
            data={
                "comfyui_root": str(config.comfyui_root),
                "models_root": str(paths.models_root),
                "cache_dir": str(config.cache_dir),
                "api_base_url": config.base_url,
                "api_key_configured": bool(config.api_key),
            },
        )

    @app.prompt(
        name="civitai_find_best_model",
        description="Prompt to find the best Civitai model for a ComfyUI use case.",
    )
    def civitai_find_best_model(use_case: str, base_model: str | None = None) -> str:
        return (
            f"Find the best Civitai model for this ComfyUI use case: {use_case}. "
            f"Prefer local-safe options, summarize top candidates, explain tradeoffs, and suggest a ComfyUI folder. "
            f"{f'Base model target: {base_model}.' if base_model else ''} "
            "Use read-only tools first, then only propose explicit installs if needed."
        )

    @app.prompt(
        name="civitai_plan_safe_install",
        description="Prompt to plan a safe Civitai download or install into ComfyUI.",
    )
    def civitai_plan_safe_install(asset_ref: str, destination_hint: str | None = None) -> str:
        return (
            f"Plan a safe install for this asset reference: {asset_ref}. "
            f"{f'Destination hint: {destination_hint}. ' if destination_hint else ''}"
            "Use compatibility hints, resolve the local ComfyUI folder, detect duplicates, and produce a dry-run first. "
            "Only suggest a write step after the install plan is explicit and safe."
        )

    @app.prompt(
        name="civitai_scan_workflow",
        description="Prompt to inspect a ComfyUI workflow for missing assets and model placement issues.",
    )
    def civitai_scan_workflow(workflow_path: str) -> str:
        return (
            f"Inspect this ComfyUI workflow for missing assets and model placement issues: {workflow_path}. "
            "Summarize likely missing files, suggest local ComfyUI folders, and avoid changing the workflow unless asked."
        )

    return app


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Civitai FastMCP foundation server.")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http", "sse"])
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--comfyui-root", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    config = ServerConfig.from_env(
        comfyui_root=args.comfyui_root,
        cache_dir=args.cache_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        debug=args.debug,
        host=args.host,
        port=args.port,
    )
    server = create_server(config)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
