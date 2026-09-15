"""Session fields for SDK agents running inside a sandbox workspace."""

from __future__ import annotations

import os

from agent.sandbox.config import ENV_SESSION_DIR
from agent.sandbox.manifest import load_sandbox_manifest
from agent.sandbox.session_dir import resolve_agent_session_dir


def resolve_sdk_session_fields(session_id: str) -> tuple[str, str]:
    """Return ``(session_dir, scenario_name)`` from sandbox env + manifest."""
    workspace = os.environ.get(ENV_SESSION_DIR, "").strip() or "."
    session_dir = resolve_agent_session_dir(workspace)
    manifest = load_sandbox_manifest(session_dir)
    scenario_name = str(manifest.get("scenario_name") or "").strip()
    return session_dir, scenario_name
