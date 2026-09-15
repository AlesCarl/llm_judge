"""Sandbox manifest helpers (no NIKA orchestration imports)."""

from __future__ import annotations

import json
from pathlib import Path

from agent.sandbox.constants import MANIFEST_FILENAME


def load_sandbox_manifest(workspace_dir: str | Path) -> dict:
    """Load ``sandbox_manifest.json`` from a sandbox workspace."""
    path = Path(workspace_dir).resolve() / MANIFEST_FILENAME
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_mcp_servers(workspace_dir: str | Path) -> dict | None:
    """Return pre-baked MCP HTTP config from the manifest, if present."""
    servers = load_sandbox_manifest(workspace_dir).get("mcp_servers")
    if isinstance(servers, dict) and servers:
        return servers
    return None
