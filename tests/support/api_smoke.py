from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any


class ApiSmokeMixin:
    """Mixin that records API smoke calls and fails on parse/runtime errors."""

    def smoke(
        self,
        label: str,
        fn: Callable[[], Any],
        *,
        expect_type: type | tuple[type, ...] | None = None,
        min_len: int = 0,
    ) -> Any:
        try:
            result = fn()
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{label}: JSON parse error: {exc}") from exc
        except (ValueError, RuntimeError, TypeError) as exc:
            raise AssertionError(f"{label}: {type(exc).__name__}: {exc}") from exc

        if expect_type is not None:
            assert isinstance(
                result,
                expect_type,
            ), f"{label}: expected {expect_type}, got {type(result)}"
        if min_len > 0:
            text = "" if result is None else str(result)
            assert len(text) >= min_len, (
                f"{label}: unexpected empty result ({result!r})"
            )
        return result

    def smoke_async(
        self,
        label: str,
        fn: Callable[[], Any],
        *,
        min_len: int = 0,
    ) -> Any:
        return self.smoke(label, lambda: asyncio.run(fn()), min_len=min_len)


def assert_json_payload(label: str, payload: str) -> dict:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{label}: invalid JSON: {exc}\n{payload!r}") from exc
    assert isinstance(parsed, dict), f"{label}: JSON root must be an object"
    return parsed
