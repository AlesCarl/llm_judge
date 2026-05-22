"""Start a Kathara lab for one scenario and persist a new session."""

from datetime import datetime
from uuid import uuid4
from typing import Literal

from nika.net_env.net_env_pool import get_net_env_instance, scenario_requires_topo_tier
from nika.utils.logger import refresh_logger, system_logger
from nika.utils.session import Session


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

    session = Session()
    scenario_params = {"lab_name": net_env.lab.name}
    if tier is not None:
        scenario_params["topo_size"] = tier
    session.init_session(
        session_id=net_env.lab.hash,
        scenario_name=scenario,
        lab_name=net_env.lab.name,
        scenario_topo_size=tier,
        scenario_params=scenario_params,
    )
    system_logger.info(
        f"Started network environment: {scenario} with size {tier} in session {session.session_id}, lab {net_env.lab.name}"
    )
    return session.session_id
