from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest import mock

import httpx
import unittest

from tools.civitai_mcp.cache import JsonCache
from tools.civitai_mcp.client import CivitaiClient
from tools.civitai_mcp.paths import ComfyUIPaths
from tools.civitai_mcp.server import ServerConfig, create_server


class TestCivitaiMCP(unittest.TestCase):
    def test_civitai_client_builds_search_request(self) -> None:
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["headers"] = dict(request.headers)
            return httpx.Response(
                200,
                json={
                    "items": [],
                    "metadata": {
                        "nextCursor": "abc123",
                        "nextPage": "https://civitai.com/api/v1/models?cursor=abc123",
                    },
                },
            )

        with tempfile_directory() as tmp_path:
            client = CivitaiClient(
                cache=JsonCache(tmp_path / "cache.json"),
                transport=httpx.MockTransport(handler),
            )
            try:
                data = client.search_models(query="ltx", tag="anime", username="civitai", limit=5, baseModel="Flux")
            finally:
                client.close()

        self.assertEqual(data["metadata"]["nextCursor"], "abc123")
        self.assertIn("/api/v1/models?", seen["url"])
        self.assertIn("query=ltx", seen["url"])
        self.assertIn("tag=anime", seen["url"])
        self.assertIn("username=civitai", seen["url"])
        self.assertIn("limit=5", seen["url"])
        self.assertIn("baseModel=Flux", seen["url"])

    def test_path_resolution_prefers_known_ltx_and_gguf_folders(self) -> None:
        with tempfile_directory() as tmp_path:
            root = tmp_path / "ComfyUI"
            (root / "models" / "text_encoders").mkdir(parents=True)
            (root / "models" / "llm").mkdir(parents=True)
            (root / "models" / "loras").mkdir(parents=True)
            paths = ComfyUIPaths(root)

            self.assertEqual(paths.resolve_model_folder("LoRA", "foo.safetensors").name, "loras")
            self.assertEqual(paths.resolve_model_folder(None, "ltx-2.3_text_projection_bf16.safetensors").name, "text_encoders")
            self.assertEqual(paths.resolve_model_folder(None, "gemma_3_12b_it_fp4_mixed.gguf").name, "text_encoders")
            self.assertEqual(paths.resolve_model_folder(None, "something_else.gguf").name, "llm")

    def test_list_model_folders_reports_existing_dirs(self) -> None:
        with tempfile_directory() as tmp_path:
            root = tmp_path / "ComfyUI"
            (root / "models" / "loras" / "Ideogram4").mkdir(parents=True)
            (root / "models" / "loras" / "Ideogram4" / "example.safetensors").write_text("x", encoding="utf-8")
            paths = ComfyUIPaths(root)

            folders = paths.list_model_folders()
            rel_paths = {item["relative_path"] for item in folders}
            self.assertIn("models/loras", rel_paths)
            self.assertIn("models/loras/Ideogram4", rel_paths)

    def test_duplicate_detection_groups_same_content(self) -> None:
        with tempfile_directory() as tmp_path:
            root = tmp_path / "ComfyUI"
            folder = root / "models" / "loras"
            folder.mkdir(parents=True)
            (folder / "a.safetensors").write_bytes(b"same")
            (folder / "b.safetensors").write_bytes(b"same")
            (folder / "c.safetensors").write_bytes(b"different")
            paths = ComfyUIPaths(root)

            groups = paths.scan_duplicate_files(folder)
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["count"], 2)
            self.assertEqual(sorted(Path(p).name for p in groups[0]["paths"]), ["a.safetensors", "b.safetensors"])

    def test_download_asset_writes_expected_payload(self) -> None:
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(
                200,
                content=b"payload",
                headers={"content-disposition": 'attachment; filename="demo.safetensors"'},
            )

        with tempfile_directory() as tmp_path:
            client = CivitaiClient(
                cache=JsonCache(tmp_path / "cache.json"),
                transport=httpx.MockTransport(handler),
            )
            try:
                out_path = tmp_path / "demo.safetensors"
                downloaded = client.download("https://civitai.com/api/download/models/1", out_path)
            finally:
                client.close()

            self.assertEqual(downloaded.read_bytes(), b"payload")
            self.assertIn("https://civitai.com/api/download/models/1", seen["url"])

    def test_download_url_rejects_non_civitai_hosts(self) -> None:
        with tempfile_directory() as tmp_path:
            client = CivitaiClient(cache=JsonCache(tmp_path / "cache.json"))
            try:
                with self.assertRaises(ValueError):
                    client.resolve_download_url(download_url="https://example.com/file.bin")
            finally:
                client.close()

    def test_server_registers_expected_tools(self) -> None:
        with tempfile_directory() as tmp_path:
            config = ServerConfig.from_env(
                comfyui_root=tmp_path / "ComfyUI",
                cache_dir=tmp_path / "cache",
                api_key=None,
                debug=False,
                host="127.0.0.1",
                port=8000,
            )
            server = create_server(config)
            tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
            prompts = {prompt.name: prompt for prompt in asyncio.run(server.list_prompts())}

            self.assertIn("civitai_search_models", tools)
            self.assertIn("civitai_install_asset", tools)
            self.assertIn("civitai_plan_download", tools)
            self.assertIn("civitai_find_best_model", prompts)
            self.assertIn("civitai_plan_safe_install", prompts)
            self.assertTrue(tools["civitai_search_models"].annotations.readOnlyHint)
            self.assertFalse(tools["civitai_install_asset"].annotations.readOnlyHint)

    def test_server_config_uses_default_timeout_for_blank_env_value(self) -> None:
        with mock.patch.dict(os.environ, {"CIVITAI_MCP_TIMEOUT": ""}, clear=False):
            config = ServerConfig.from_env(
                comfyui_root=Path("C:/tmp/ComfyUI"),
                cache_dir=Path("C:/tmp/cache"),
                api_key=None,
                debug=False,
                host="127.0.0.1",
                port=8000,
            )

        self.assertEqual(config.timeout, 30.0)

    def test_top_models_and_file_only_tools_use_normalized_payloads(self) -> None:
        class FakeClient:
            def search_models(self, **params):
                self.params = params
                return {
                    "items": [
                        {
                            "id": 10,
                            "name": "Demo Model",
                            "creator": {"username": "demo"},
                            "tags": [{"name": "test"}],
                            "type": "Checkpoint",
                            "nsfwLevel": 0,
                            "availability": "Published",
                            "supportsGeneration": True,
                            "description": "<p>hello</p>",
                            "modelVersions": [
                                {
                                    "id": 99,
                                    "name": "v1",
                                    "index": 0,
                                    "baseModel": "Flux",
                                    "baseModelType": "Standard",
                                    "publishedAt": "2026-01-01T00:00:00Z",
                                    "status": "Published",
                                    "availability": "Published",
                                    "trainedWords": ["demo"],
                                    "vaeId": 1,
                                    "stats": {},
                                    "downloadUrl": "https://example.invalid/download",
                                    "files": [
                                        {
                                            "id": 11,
                                            "name": "demo.safetensors",
                                            "type": "Model",
                                            "sizeKB": 123,
                                            "primary": True,
                                            "downloadUrl": "https://example.invalid/download",
                                            "hashes": {"SHA256": "abc"},
                                            "pickleScanResult": "Success",
                                            "virusScanResult": "Success",
                                            "scannedAt": "2026-01-01T00:00:00Z",
                                        }
                                    ],
                                    "images": [],
                                }
                            ],
                        }
                    ],
                    "metadata": {"nextCursor": None},
                }

            def get_model(self, model_id):
                self.model_id = model_id
                return self.search_models()["items"][0]

            def get_model_version(self, version_id):
                self.version_id = version_id
                return self.search_models()["items"][0]["modelVersions"][0]

            def get_model_version_by_hash(self, file_hash):
                self.file_hash = file_hash
                return self.search_models()["items"][0]["modelVersions"][0]

            def search_creators(self, **kwargs):
                return {"items": [{"username": "demo", "modelCount": 1, "link": "https://civitai.com"}], "metadata": {}}

            def search_images(self, **kwargs):
                return {"items": [], "metadata": {}}

            def search_tags(self, **kwargs):
                return {"items": [], "metadata": {}}

            def resolve_download_url(self, **kwargs):
                return "https://example.invalid/download", {"source": "stub"}

            def download(self, url, destination):
                destination.write_bytes(b"payload")
                return destination

            def close(self):
                pass

        with tempfile_directory() as tmp_path:
            fake_client = FakeClient()
            with mock.patch("tools.civitai_mcp.server.CivitaiClient", return_value=fake_client):
                config = ServerConfig.from_env(
                    comfyui_root=tmp_path / "ComfyUI",
                    cache_dir=tmp_path / "cache",
                    api_key=None,
                    debug=False,
                    host="127.0.0.1",
                    port=8000,
                )
                server = create_server(config)

                top_result = asyncio.run(server.call_tool("civitai_get_top_models", {"limit": 1}))
                report_result = asyncio.run(server.call_tool("civitai_export_model_report", {"model_id": 10}))
                files_result = asyncio.run(server.call_tool("civitai_get_model_files", {"model_id": 10}))
                hint_result = asyncio.run(
                    server.call_tool(
                        "civitai_get_compatibility_hint",
                        {"filename": "ltx-2.3_text_projection_bf16.safetensors"},
                    )
                )
                creator_result = asyncio.run(server.call_tool("civitai_get_creator_profile", {"username": "demo"}))

            self.assertEqual(top_result[1]["status"], "ok")
            self.assertEqual(top_result[1]["data"]["items"][0]["name"], "Demo Model")
            self.assertIn("# Demo Model", report_result[1]["data"]["markdown"])
            self.assertEqual(report_result[1]["data"]["json"]["model"]["id"], 10)
            self.assertEqual(files_result[1]["data"]["file_count"], 1)
            self.assertEqual(files_result[1]["data"]["files"][0]["name"], "demo.safetensors")
            self.assertEqual(hint_result[1]["data"]["recommended_folder_relative"], "models/text_encoders")
            self.assertEqual(creator_result[1]["data"]["creator"]["username"], "demo")

    def test_get_model_uses_newest_version_metadata(self) -> None:
        class FakeClient:
            def get_model(self, model_id):
                self.model_id = model_id
                return {
                    "id": model_id,
                    "name": "Demo Model",
                    "creator": {"username": "demo"},
                    "tags": [],
                    "type": "Checkpoint",
                    "nsfwLevel": 0,
                    "availability": "Published",
                    "supportsGeneration": True,
                    "description": "<p>hello</p>",
                    "modelVersions": [
                        {
                            "id": 1,
                            "name": "old",
                            "index": 0,
                            "baseModel": "Flux",
                            "baseModelType": "Standard",
                            "publishedAt": "2025-01-01T00:00:00Z",
                            "status": "Published",
                            "availability": "Published",
                            "trainedWords": [],
                            "vaeId": None,
                            "stats": {},
                            "downloadUrl": "https://example.invalid/download-old",
                            "files": [],
                            "images": [],
                        },
                        {
                            "id": 2,
                            "name": "new",
                            "index": 1,
                            "baseModel": "Flux",
                            "baseModelType": "Standard",
                            "publishedAt": "2026-01-01T00:00:00Z",
                            "status": "Published",
                            "availability": "Published",
                            "trainedWords": [],
                            "vaeId": None,
                            "stats": {},
                            "downloadUrl": "https://example.invalid/download-new",
                            "files": [],
                            "images": [],
                        },
                    ],
                }

            def search_models(self, **kwargs):
                return {"items": [], "metadata": {}}

            def get_model_version(self, version_id):
                return {"files": [], "images": []}

            def get_model_version_by_hash(self, file_hash):
                return {"files": [], "images": []}

            def search_creators(self, **kwargs):
                return {"items": [], "metadata": {}}

            def search_images(self, **kwargs):
                return {"items": [], "metadata": {}}

            def search_tags(self, **kwargs):
                return {"items": [], "metadata": {}}

            def resolve_download_url(self, **kwargs):
                return "https://example.invalid/download", {"source": "stub"}

            def download(self, url, destination):
                destination.write_bytes(b"payload")
                return destination

            def close(self):
                pass

        with tempfile_directory() as tmp_path:
            fake_client = FakeClient()
            with mock.patch("tools.civitai_mcp.server.CivitaiClient", return_value=fake_client):
                config = ServerConfig.from_env(
                    comfyui_root=tmp_path / "ComfyUI",
                    cache_dir=tmp_path / "cache",
                    api_key=None,
                    debug=False,
                    host="127.0.0.1",
                    port=8000,
                )
                server = create_server(config)
                result = asyncio.run(server.call_tool("civitai_get_model", {"model_id": 10}))

        self.assertEqual(result[1]["data"]["model"]["latestVersion"]["id"], 2)

    def test_export_model_report_uses_latest_version_in_markdown_and_json(self) -> None:
        class FakeClient:
            def get_model(self, model_id):
                self.model_id = model_id
                return {
                    "id": model_id,
                    "name": "Demo Model",
                    "creator": {"username": "demo"},
                    "tags": [],
                    "type": "Checkpoint",
                    "nsfwLevel": 0,
                    "availability": "Published",
                    "supportsGeneration": True,
                    "description": "<p>hello</p>",
                    "modelVersions": [
                        {
                            "id": 1,
                            "name": "old",
                            "index": 0,
                            "baseModel": "Flux",
                            "baseModelType": "Standard",
                            "publishedAt": "2025-01-01T00:00:00Z",
                            "status": "Published",
                            "availability": "Published",
                            "trainedWords": [],
                            "vaeId": None,
                            "stats": {},
                            "downloadUrl": "https://example.invalid/download-old",
                            "files": [{"id": 1, "name": "old.safetensors", "type": "Model", "sizeKB": 1, "primary": True, "downloadUrl": "https://example.invalid/download-old", "hashes": {}}],
                            "images": [],
                        },
                        {
                            "id": 2,
                            "name": "new",
                            "index": 1,
                            "baseModel": "Flux",
                            "baseModelType": "Standard",
                            "publishedAt": "2026-01-01T00:00:00Z",
                            "status": "Published",
                            "availability": "Published",
                            "trainedWords": [],
                            "vaeId": None,
                            "stats": {},
                            "downloadUrl": "https://example.invalid/download-new",
                            "files": [{"id": 2, "name": "new.safetensors", "type": "Model", "sizeKB": 2, "primary": True, "downloadUrl": "https://example.invalid/download-new", "hashes": {}}],
                            "images": [],
                        },
                    ],
                }

            def search_models(self, **kwargs):
                return {"items": [], "metadata": {}}

            def get_model_version(self, version_id):
                return {"files": [], "images": []}

            def get_model_version_by_hash(self, file_hash):
                return {"files": [], "images": []}

            def search_creators(self, **kwargs):
                return {"items": [], "metadata": {}}

            def search_images(self, **kwargs):
                return {"items": [], "metadata": {}}

            def search_tags(self, **kwargs):
                return {"items": [], "metadata": {}}

            def resolve_download_url(self, **kwargs):
                return "https://example.invalid/download", {"source": "stub"}

            def download(self, url, destination):
                destination.write_bytes(b"payload")
                return destination

            def close(self):
                pass

        with tempfile_directory() as tmp_path:
            fake_client = FakeClient()
            with mock.patch("tools.civitai_mcp.server.CivitaiClient", return_value=fake_client):
                config = ServerConfig.from_env(
                    comfyui_root=tmp_path / "ComfyUI",
                    cache_dir=tmp_path / "cache",
                    api_key=None,
                    debug=False,
                    host="127.0.0.1",
                    port=8000,
                )
                server = create_server(config)
                result = asyncio.run(server.call_tool("civitai_export_model_report", {"model_id": 10}))

        self.assertEqual(result[1]["status"], "ok")
        self.assertEqual(result[1]["data"]["json"]["model"]["latestVersion"]["id"], 2)
        self.assertIn("Version ID: 2", result[1]["data"]["markdown"])

    def test_plan_download_reports_resume_candidate_without_writing(self) -> None:
        class FakeClient:
            def get_model(self, model_id):
                return {"modelVersions": []}

            def search_models(self, **kwargs):
                return {"items": [], "metadata": {}}

            def get_model_version(self, version_id):
                return {
                    "id": version_id,
                    "name": "demo",
                    "index": 0,
                    "baseModel": "Flux",
                    "baseModelType": "Standard",
                    "publishedAt": "2026-01-01T00:00:00Z",
                    "status": "Published",
                    "availability": "Published",
                    "trainedWords": [],
                    "vaeId": None,
                    "stats": {},
                    "downloadUrl": "https://civitai.com/api/download/models/1",
                    "files": [
                        {
                            "id": 11,
                            "name": "demo.safetensors",
                            "type": "Model",
                            "sizeKB": 2,
                            "primary": True,
                            "downloadUrl": "https://civitai.com/api/download/models/1",
                            "hashes": {"SHA256": "abc"},
                            "pickleScanResult": "Success",
                            "virusScanResult": "Success",
                            "scannedAt": "2026-01-01T00:00:00Z",
                        }
                    ],
                    "images": [],
                }

            def get_model_version_by_hash(self, file_hash):
                return self.get_model_version(99)

            def search_creators(self, **kwargs):
                return {"items": [], "metadata": {}}

            def search_images(self, **kwargs):
                return {"items": [], "metadata": {}}

            def search_tags(self, **kwargs):
                return {"items": [], "metadata": {}}

            def resolve_download_url(self, **kwargs):
                return "https://civitai.com/api/download/models/1", {"source": "stub"}

            def download(self, url, destination):
                destination.write_bytes(b"payload")
                return destination

            def close(self):
                pass

        with tempfile_directory() as tmp_path:
            comfyui_root = tmp_path / "ComfyUI"
            target_folder = comfyui_root / "models" / "loras"
            target_folder.mkdir(parents=True)
            partial_path = target_folder / "demo.safetensors.part"
            partial_path.write_bytes(b"partial")
            fake_client = FakeClient()
            with mock.patch("tools.civitai_mcp.server.CivitaiClient", return_value=fake_client):
                config = ServerConfig.from_env(
                    comfyui_root=comfyui_root,
                    cache_dir=tmp_path / "cache",
                    api_key=None,
                    debug=False,
                    host="127.0.0.1",
                    port=8000,
                )
                server = create_server(config)
                result = asyncio.run(
                    server.call_tool(
                        "civitai_plan_download",
                        {
                            "model_version_id": 99,
                            "asset_type": "LoRA",
                        },
                    )
                )

        self.assertEqual(result[1]["status"], "ok")
        self.assertEqual(result[1]["data"]["download_state"], "resume_candidate")
        self.assertFalse((target_folder / "demo.safetensors").exists())
        self.assertTrue(result[1]["data"]["partial_exists"])
        self.assertFalse(result[1]["data"]["resume_supported"])

    def test_workflow_scan_cache_and_batch_plan_tools(self) -> None:
        with tempfile_directory() as tmp_path:
            workflow = {
                "nodes": [
                    {"inputs": {"ckpt_name": "missing_model.safetensors"}},
                    {"inputs": {"vae_name": "existing_vae.safetensors"}},
                ]
            }
            (tmp_path / "workflow.json").write_text(json.dumps(workflow), encoding="utf-8")
            comfyui_root = tmp_path / "ComfyUI"
            (comfyui_root / "models" / "vae").mkdir(parents=True)
            (comfyui_root / "models" / "vae" / "existing_vae.safetensors").write_text("ok", encoding="utf-8")

            fake_client = type(
                "FakeClient",
                (),
                {
                    "search_models": lambda self, **kwargs: {"items": [], "metadata": {}},
                    "get_model": lambda self, model_id: {"modelVersions": []},
                    "get_model_version": lambda self, version_id: {"files": [], "images": []},
                    "get_model_version_by_hash": lambda self, file_hash: {"files": [], "images": []},
                    "search_creators": lambda self, **kwargs: {"items": [], "metadata": {}},
                    "search_images": lambda self, **kwargs: {"items": [], "metadata": {}},
                    "search_tags": lambda self, **kwargs: {"items": [], "metadata": {}},
                    "resolve_download_url": lambda self, **kwargs: ("https://example.invalid/download", {"source": "stub"}),
                    "download": lambda self, url, destination: destination,
                    "close": lambda self: None,
                },
            )()

            with mock.patch("tools.civitai_mcp.server.CivitaiClient", return_value=fake_client):
                config = ServerConfig.from_env(
                    comfyui_root=comfyui_root,
                    cache_dir=tmp_path / "cache",
                    api_key=None,
                    debug=False,
                    host="127.0.0.1",
                    port=8000,
                )
                server = create_server(config)
                scan_result = asyncio.run(server.call_tool("civitai_scan_workflow_assets", {"workflow_path": str(tmp_path / "workflow.json")}))
                cache_status = asyncio.run(server.call_tool("civitai_cache_status", {}))
                clear_dry_run = asyncio.run(server.call_tool("civitai_clear_cache", {"confirm": False}))
                batch_result = asyncio.run(
                    server.call_tool(
                        "civitai_plan_batch_install",
                        {
                            "items_json": json.dumps(
                                [
                                    {"model_version_id": 99, "asset_type": "LoRA"},
                                    {"file_hash": "abc", "destination_filename": "../../demo.safetensors"},
                                ]
                            ),
                            "default_destination_folder": str(comfyui_root / "models" / "loras"),
                        },
                    )
                )

            self.assertEqual(scan_result[1]["data"]["missing_count"], 1)
            self.assertEqual(scan_result[1]["data"]["missing"][0]["filename"], "missing_model.safetensors")
            self.assertGreaterEqual(cache_status[1]["data"]["entry_count"], 0)
            self.assertEqual(clear_dry_run[1]["status"], "dry_run")
            self.assertEqual(batch_result[1]["data"]["planned_count"], 2)
            self.assertEqual(batch_result[1]["data"]["plans"][1]["plan"]["filename"], "demo.safetensors")
            self.assertTrue(batch_result[1]["data"]["plans"][1]["plan"]["filename_sanitized"])

    def test_download_asset_rejects_destination_outside_comfyui_models_tree(self) -> None:
        class FakeClient:
            def search_models(self, **kwargs):
                return {"items": [], "metadata": {}}

            def get_model(self, model_id):
                return {"modelVersions": []}

            def get_model_version(self, version_id):
                return {"files": [], "images": []}

            def get_model_version_by_hash(self, file_hash):
                return {"files": [], "images": []}

            def search_creators(self, **kwargs):
                return {"items": [], "metadata": {}}

            def search_images(self, **kwargs):
                return {"items": [], "metadata": {}}

            def search_tags(self, **kwargs):
                return {"items": [], "metadata": {}}

            def resolve_download_url(self, **kwargs):
                return "https://civitai.com/api/download/models/1", {"source": "stub"}

            def download(self, url, destination):
                destination.write_bytes(b"payload")
                return destination

            def close(self):
                pass

        with tempfile_directory() as tmp_path:
            comfyui_root = tmp_path / "ComfyUI"
            (comfyui_root / "models" / "loras").mkdir(parents=True)
            fake_client = FakeClient()
            with mock.patch("tools.civitai_mcp.server.CivitaiClient", return_value=fake_client):
                config = ServerConfig.from_env(
                    comfyui_root=comfyui_root,
                    cache_dir=tmp_path / "cache",
                    api_key=None,
                    debug=False,
                    host="127.0.0.1",
                    port=8000,
                )
                server = create_server(config)
                result = asyncio.run(
                    server.call_tool(
                        "civitai_download_asset",
                        {
                            "download_url": "https://civitai.com/api/download/models/1",
                            "destination_folder": str(tmp_path / "outside"),
                            "dry_run": True,
                        },
                    )
                )

        self.assertEqual(result[1]["status"], "error")
        self.assertIn("models tree", result[1]["error"]["message"])


def tempfile_directory():
    from contextlib import contextmanager
    from tempfile import TemporaryDirectory

    @contextmanager
    def _tempdir():
        with TemporaryDirectory() as tmp:
            yield Path(tmp)

    return _tempdir()
