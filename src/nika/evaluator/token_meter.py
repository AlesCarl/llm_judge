"""Token/cost accounting for the LLM judges.

Thin wrapper around LangChain's ``UsageMetadataCallbackHandler`` so every judge
(single / multi / multi_role) can record how many tokens its evaluation
consumed and dump a per-judge sidecar JSON next to its output. This enables a
cost-vs-accuracy comparison across the three judge systems.

Usage:
    meter = new_meter()
    cfg = meter_config(meter)          # pass as config=cfg to every .invoke()
    ...                                # run the judge
    dump_cost(meter, save_path, judge="multi", filename="multi_cost.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler


logger = logging.getLogger(__name__)


def new_meter() -> UsageMetadataCallbackHandler:
    """Create a fresh usage handler that accumulates tokens across .invoke()s."""
    return UsageMetadataCallbackHandler()


def meter_config(handler: UsageMetadataCallbackHandler | None) -> dict | None:
    """Build the invoke config that routes token usage to ``handler``.

    Returns None when handler is None, so callers can pass it unconditionally
    (``config=None`` is a no-op for LangChain's .invoke()).
    """
    if handler is None:
        return None
    return {"callbacks": [handler]}


def dump_cost(
    handler: UsageMetadataCallbackHandler,
    save_path: str,
    judge: str,
    filename: str,
) -> dict:
    """Write a per-judge token-cost sidecar next to ``save_path``.

    ``handler.usage_metadata`` is ``{model_name: {input_tokens, output_tokens,
    total_tokens, ...}}``. We sum across models into flat totals and keep the
    per-model breakdown.

    Args:
        handler: The usage handler passed to the judge's .invoke() calls.
        save_path: The judge's main output path; the sidecar is written in the
            same directory.
        judge: Tag identifying the judge ("single" / "multi" / "multi_role").
        filename: Sidecar filename (distinct per judge to avoid collisions in
            the shared run directory).

    Returns:
        The payload dict that was written (also handy for logging/tests).
    """
    per_model = dict(handler.usage_metadata or {})
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for usage in per_model.values():
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)

    payload = {
        "judge": judge,
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
        "total_tokens": totals["total_tokens"],
        "per_model": per_model,
    }

    out = Path(save_path).with_name(filename)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "Token cost (%s): %d in + %d out = %d total → %s",
        judge,
        totals["input_tokens"],
        totals["output_tokens"],
        totals["total_tokens"],
        out,
    )
    return payload
