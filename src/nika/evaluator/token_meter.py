"""Token accounting for the judges, to compare cost against accuracy.

    meter = new_meter()
    cfg = meter_config(meter)  # pass as config=cfg to every .invoke()
    ...
    dump_cost(meter, save_path, judge="multi", filename="multi_cost.json")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler


logger = logging.getLogger(__name__)


def new_meter() -> UsageMetadataCallbackHandler:
    """New handler that accumulates tokens across .invoke() calls."""
    return UsageMetadataCallbackHandler()


def meter_config(handler: UsageMetadataCallbackHandler | None) -> dict | None:
    """Invoke config that routes token usage to ``handler`` (None if no handler)."""
    if handler is None:
        return None
    return {"callbacks": [handler]}


def dump_cost(
    handler: UsageMetadataCallbackHandler,
    save_path: str,
    judge: str,
    filename: str,
) -> dict:
    """Write the token totals (plus per-model breakdown) next to ``save_path`` and return them.

    ``filename`` differs per judge, since they share the session directory.
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
