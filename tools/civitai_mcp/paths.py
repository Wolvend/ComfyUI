from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FOLDER_MAP = {
    "checkpoint": "checkpoints",
    "checkpoints": "checkpoints",
    "model": "checkpoints",
    "lora": "loras",
    "lycoris": "loras",
    "loras": "loras",
    "vae": "vae",
    "vae_approx": "vae_approx",
    "embedding": "embeddings",
    "embeddings": "embeddings",
    "textualinversion": "embeddings",
    "hypernetwork": "hypernetworks",
    "controlnet": "controlnet",
    "upscaler": "upscale_models",
    "upscale": "upscale_models",
    "upscale_model": "upscale_models",
    "clip": "text_encoders",
    "text_encoder": "text_encoders",
    "textencoders": "text_encoders",
    "llm": "llm",
    "gguf": "llm",
    "diffusers": "diffusers",
    "unet": "unet",
    "style": "style_models",
    "style_model": "style_models",
    "photomaker": "photomaker",
    "detection": "detection",
    "ipadapter": "ipadapter",
    "clip_vision": "clip_vision",
    "audio_encoder": "audio_encoders",
    "audio_encoders": "audio_encoders",
    "motion": "frame_interpolation",
    "optical_flow": "optical_flow",
    "tensorrt": "tensorrt",
}


GGUF_TEXT_ENCODER_HINTS = (
    "ltx",
    "clip",
    "text",
    "encoder",
    "t5",
    "gemma",
    "qwen",
    "llama",
    "mistral",
    "phi",
)


@dataclass(slots=True)
class ComfyUIPaths:
    root: Path

    @property
    def models_root(self) -> Path:
        return self.root / "models"

    @classmethod
    def from_env(cls, root: str | None = None) -> "ComfyUIPaths":
        if root:
            return cls(Path(root).expanduser().resolve())
        from os import getenv

        for candidate in ("CIVITAI_MCP_COMFYUI_ROOT", "COMFYUI_ROOT"):
            value = getenv(candidate)
            if value:
                return cls(Path(value).expanduser().resolve())
        return cls(Path(__file__).resolve().parents[2])

    def list_model_folders(self) -> list[dict[str, Any]]:
        folders: list[dict[str, Any]] = []
        if not self.models_root.exists():
            return folders
        for top in sorted(p for p in self.models_root.iterdir() if p.is_dir()):
            folders.append(self._folder_entry(top))
            for child in sorted(p for p in top.iterdir() if p.is_dir()):
                folders.append(self._folder_entry(child))
        return folders

    def _folder_entry(self, folder: Path) -> dict[str, Any]:
        file_count = sum(1 for item in folder.iterdir() if item.is_file())
        return {
            "name": folder.name,
            "path": str(folder),
            "relative_path": folder.relative_to(self.root).as_posix(),
            "file_count": file_count,
        }

    def resolve_model_folder(
        self,
        asset_type: str | None = None,
        filename: str | None = None,
        base_model: str | None = None,
    ) -> Path:
        folder_name = self._resolve_folder_name(asset_type, filename, base_model)
        target = self.models_root / folder_name
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _resolve_folder_name(
        self,
        asset_type: str | None,
        filename: str | None = None,
        base_model: str | None = None,
    ) -> str:
        normalized = self._normalize(asset_type)
        if filename:
            guessed = self._guess_from_filename(filename)
            if guessed:
                return guessed
        if normalized in DEFAULT_FOLDER_MAP:
            return DEFAULT_FOLDER_MAP[normalized]
        if base_model:
            base_model_name = self._normalize(base_model)
            if base_model_name in DEFAULT_FOLDER_MAP:
                return DEFAULT_FOLDER_MAP[base_model_name]
        return "checkpoints"

    def _guess_from_filename(self, filename: str) -> str | None:
        lower = filename.lower()
        if any(hint in lower for hint in GGUF_TEXT_ENCODER_HINTS):
            return "text_encoders"
        if lower.endswith(".gguf"):
            return "llm"
        if lower.endswith((".safetensors", ".ckpt", ".pt", ".bin")):
            if "lora" in lower or "lycoris" in lower:
                return "loras"
            if "vae" in lower:
                return "vae"
            if "emb" in lower or "textual" in lower:
                return "embeddings"
        return None

    @staticmethod
    def _normalize(value: str | None) -> str:
        return "".join(ch for ch in (value or "").lower() if ch.isalnum())

    def scan_duplicate_files(self, folder: Path | None = None) -> list[dict[str, Any]]:
        base = Path(folder or self.models_root)
        if not base.exists():
            return []
        candidates: list[Path] = [p for p in base.rglob("*") if p.is_file()]
        size_groups: dict[int, list[Path]] = {}
        for path in candidates:
            try:
                size_groups.setdefault(path.stat().st_size, []).append(path)
            except OSError:
                continue

        groups: list[dict[str, Any]] = []
        for size, paths in size_groups.items():
            if len(paths) < 2:
                continue
            hash_groups: dict[str, list[Path]] = {}
            for path in paths:
                digest = self._file_digest(path)
                if digest:
                    hash_groups.setdefault(digest, []).append(path)
            for digest, digest_paths in hash_groups.items():
                if len(digest_paths) < 2:
                    continue
                groups.append(
                    {
                        "hash": digest,
                        "size_bytes": size,
                        "count": len(digest_paths),
                        "paths": [str(path) for path in sorted(digest_paths)],
                    }
                )
        groups.sort(key=lambda item: (-item["count"], item["size_bytes"], item["hash"]))
        return groups

    @staticmethod
    def _file_digest(path: Path) -> str | None:
        try:
            from blake3 import blake3

            hasher = blake3()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            try:
                hasher = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(chunk)
                return hasher.hexdigest()
            except Exception:
                return None
