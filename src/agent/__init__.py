"""Troubleshooting agent implementations for NIKA."""

__all__ = ["create_agent"]


def __getattr__(name: str):
    if name == "create_agent":
        from agent.registry import create_agent

        return create_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
