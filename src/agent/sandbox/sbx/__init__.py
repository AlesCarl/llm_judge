"""Docker Sandboxes (sbx) backend for NIKA agent execution."""

__all__ = [
    "SbxSandboxManager",
    "SbxSandboxRunResult",
    "sbx_available",
]


def __getattr__(name: str):
    if name == "sbx_available":
        from agent.sandbox.sbx.client import sbx_available

        return sbx_available
    if name == "SbxSandboxManager":
        from agent.sandbox.sbx.manager import SbxSandboxManager

        return SbxSandboxManager
    if name == "SbxSandboxRunResult":
        from agent.sandbox.sbx.manager import SbxSandboxRunResult

        return SbxSandboxRunResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
