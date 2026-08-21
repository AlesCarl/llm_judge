"""Start a Kathara lab for one scenario and persist a new session."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from nika.net_env.net_env_pool import get_net_env_instance, scenario_requires_topo_tier
from nika.utils.logger import bind_session_dir, log_event, refresh_logger
from nika.utils.session import Session


def _dump_topology(net_env, session_dir: str) -> None:
    """Persist the topology's device inventory (names only) for the post-hoc judge."""
    names = sorted(net_env.lab.machines.keys())

    def _kind(n: str) -> str:
        for k in ("super_spine", "spine", "leaf", "dns", "webserver", "client", "pc"):
            if k in n:
                return k
        return "other"

    by_kind: dict[str, list[str]] = {}
    for n in names:
        by_kind.setdefault(_kind(n), []).append(n)
    payload = {
        "lab": net_env.lab.name,
        "device_count": len(names),
        "devices": names,
        "by_kind": by_kind,
    }
    with open(Path(session_dir) / "topology.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _normalize_topo_tier(raw: str | None) -> Literal["s", "m", "l"] | None:
    """Return ``None`` for missing/blank input; otherwise validate ``s``/``m``/``l``."""
    if raw is None or raw == "":
        return None
    if raw not in ("s", "m", "l"):
        raise ValueError("Topology tier must be one of: s, m, l.")
    return raw  # type: ignore[return-value]


def start_net_env(
    scenario: str,
    topo_size: str | None,
    *,
    redeploy: bool = True,
    instance_tag: str | None = None,
) -> str:
    """Deploy the lab for ``scenario`` and create a new runtime session."""
    tier = _normalize_topo_tier(topo_size)
    if scenario_requires_topo_tier(scenario) and tier is None:
        raise ValueError(f"Scenario '{scenario}' requires an explicit topology tier (-t s|m|l).")
    if not scenario_requires_topo_tier(scenario) and tier is not None:
        raise ValueError(f"Scenario '{scenario}' does not use topology tiers; omit -t/--tier.")

    refresh_logger()
    tag = instance_tag or f"{datetime.now().strftime('%m%d%H%M%S')}-{uuid4().hex[:6]}"
    lab_name = f"{scenario}__{tag}"
    net_env = get_net_env_instance(scenario, topo_size=tier, lab_name=lab_name)
    if net_env.lab_exists() and redeploy:
        net_env.undeploy()
        net_env.deploy()
    elif not net_env.lab_exists():
        net_env.deploy()

    # Time-based session ID: YYYYMMDD-HHMMSS-{6hex}
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6]

    session = Session()
    scenario_params: dict = {"lab_name": net_env.lab.name}
    if tier is not None:
        scenario_params["topo_size"] = tier
    session.init_session(
        session_id=session_id,
        scenario_name=scenario,
        lab_name=net_env.lab.name,
        scenario_topo_size=tier,
        scenario_params=scenario_params,
    )
    bind_session_dir(session.session_dir)
    _dump_topology(net_env, session.session_dir)
    log_event(
        "env_start",
        f"Started network environment: {scenario} (size={tier}) — session {session_id}, lab {net_env.lab.name}",
        scenario=scenario,
        topo_size=tier,
        session_id=session_id,
        lab_name=net_env.lab.name,
    )
    return session_id
