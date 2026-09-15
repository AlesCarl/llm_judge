"""SDK mixin kit paths for native sbx shell agents."""

from __future__ import annotations

from pathlib import Path

from nika.config import _REPO_ROOT

SBX_KIT_DIR = _REPO_ROOT / "src" / "agent" / "sandbox" / "sbx" / "kit"
SBX_SDK_MIXIN_DIR = SBX_KIT_DIR / "sdk-mixin"


def sdk_mixin_dir() -> Path:
    return SBX_SDK_MIXIN_DIR
