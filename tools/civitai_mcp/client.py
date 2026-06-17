from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from .cache import JsonCache


class CivitaiAPIError(RuntimeError):
    def __init__(self, operation: str, url: str, status_code: int, response_text: str):
        self.operation = operation
        self.url = url
        self.status_code = status_code
        self.response_text = response_text
        message = f"{operation} failed with HTTP {status_code} for {url}: {response_text[:400]}"
        super().__init__(message)


@dataclass
class CivitaiClient:
    base_url: str = "https://civitai.com/api/v1"
    api_key: str | None = None
    cache: JsonCache | None = None
    timeout: float = 30.0
    transport: httpx.BaseTransport | None = None

    def __post_init__(self) -> None:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            headers=headers,
            timeout=self.timeout,
            follow_redirects=True,
            transport=self.transport,
        )

    def close(self) -> None:
        self._client.close()

    def _request_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        cache_key: str | None = None,
        cache_ttl: int = 0,
    ) -> dict[str, Any]:
        if cache_key and self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        response = self._client.get(path, params=self._clean_params(params))
        if response.status_code >= 400:
            raise CivitaiAPIError(path, str(response.request.url), response.status_code, response.text)
        data = response.json()
        if cache_key and self.cache and cache_ttl > 0:
            self.cache.set(cache_key, data, cache_ttl)
        return data

    def _request_stream(self, url: str, destination: Path) -> Path:
        self._validate_civitai_url(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._client.stream("GET", url) as response:
            if response.status_code >= 400:
                raise CivitaiAPIError("download", str(response.request.url), response.status_code, response.text)
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        return destination

    def search_models(self, **params: Any) -> dict[str, Any]:
        return self._request_json(
            "/models",
            params,
            cache_key=self._cache_key("models", params),
            cache_ttl=300,
        )

    def get_model(self, model_id: int | str) -> dict[str, Any]:
        return self._request_json(
            f"/models/{model_id}",
            cache_key=self._cache_key("model", model_id),
            cache_ttl=3600,
        )

    def get_model_version(self, version_id: int | str) -> dict[str, Any]:
        return self._request_json(
            f"/model-versions/{version_id}",
            cache_key=self._cache_key("model-version", version_id),
            cache_ttl=3600,
        )

    def get_model_version_by_hash(self, file_hash: str) -> dict[str, Any]:
        safe_hash = quote(file_hash, safe="")
        return self._request_json(
            f"/model-versions/by-hash/{safe_hash}",
            cache_key=self._cache_key("model-version-hash", file_hash),
            cache_ttl=86400,
        )

    def search_images(self, **params: Any) -> dict[str, Any]:
        return self._request_json(
            "/images",
            params,
            cache_key=self._cache_key("images", params),
            cache_ttl=300,
        )

    def search_creators(self, **params: Any) -> dict[str, Any]:
        return self._request_json(
            "/creators",
            params,
            cache_key=self._cache_key("creators", params),
            cache_ttl=300,
        )

    def search_tags(self, **params: Any) -> dict[str, Any]:
        return self._request_json(
            "/tags",
            params,
            cache_key=self._cache_key("tags", params),
            cache_ttl=300,
        )

    def download(self, url: str, destination: Path) -> Path:
        return self._request_stream(url, destination)

    @staticmethod
    def _validate_civitai_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"https"}:
            raise ValueError("Only https download URLs are allowed.")
        host = (parsed.hostname or "").lower()
        allowed_hosts = {"civitai.com", "www.civitai.com", "image.civitai.com", "files.civitai.com"}
        if host not in allowed_hosts and not host.endswith(".civitai.com"):
            raise ValueError(f"Unsupported download host: {host}")

    def resolve_download_url(
        self,
        *,
        download_url: str | None = None,
        model_version_id: int | str | None = None,
        file_hash: str | None = None,
        file_id: int | str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if download_url:
            self._validate_civitai_url(download_url)
            return download_url, {"source": "explicit_download_url"}

        version: dict[str, Any] | None = None
        source = {}
        if file_hash:
            version = self.get_model_version_by_hash(file_hash)
            source = {"source": "hash_lookup", "hash": file_hash}
        elif model_version_id is not None:
            version = self.get_model_version(model_version_id)
            source = {"source": "version_lookup", "model_version_id": int(model_version_id)}
        else:
            raise ValueError("Provide download_url, model_version_id, or file_hash.")

        if file_id is not None:
            for file_entry in version.get("files", []):
                if str(file_entry.get("id")) == str(file_id):
                    return file_entry["downloadUrl"], {**source, "file_id": int(file_id)}
            raise ValueError(f"File id {file_id} was not found on the selected model version.")

        for file_entry in version.get("files", []):
            if file_entry.get("primary"):
                return file_entry["downloadUrl"], {
                    **source,
                    "file_id": file_entry.get("id"),
                    "file_name": file_entry.get("name"),
                }

        files = version.get("files", [])
        if not files:
            raise ValueError("The selected model version does not expose any downloadable files.")
        file_entry = files[0]
        return file_entry["downloadUrl"], {
            **source,
            "file_id": file_entry.get("id"),
            "file_name": file_entry.get("name"),
        }

    def default_temp_download_dir(self) -> Path:
        return Path(tempfile.gettempdir()) / "civitai-mcp"

    @staticmethod
    def _clean_params(params: dict[str, Any] | None) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in (params or {}).items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                if not value:
                    continue
                clean[key] = ",".join(str(item) for item in value)
                continue
            if value == "":
                continue
            clean[key] = value
        return clean

    @staticmethod
    def _cache_key(namespace: str, params: Any) -> str:
        if isinstance(params, dict):
            items = tuple(sorted((str(key), params[key]) for key in params))
        else:
            items = params
        return JsonCache.key(namespace, items)
