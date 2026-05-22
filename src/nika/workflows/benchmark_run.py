"""Batch or single-case benchmark runs (env → inject → agent → eval)."""

from __future__ import annotations

import csv
from pathlib import Path

from nika.config import BASE_DIR
from nika.net_env.net_env_pool import get_net_env_instance, scenario_requires_topo_tier
from nika.workflows.agent_run import start_agent
from nika.workflows.failure_inject import inject_failure
from nika.workflows.net_env_start import start_net_env
from nika.workflows.session_eval import eval_results


def default_benchmark_csv_path() -> str:
    if not BASE_DIR:
        raise RuntimeError("BASE_DIR is not set; configure it in your environment or .env file.")
    return str(Path(BASE_DIR) / "benchmark" / "benchmark_selected.csv")


def run_single_benchmark(
    problem: str,
    scenario: str,
    topo_size: str,
    agent_type: str,
    llm_backend: str,
    model: str,
    max_steps: int,
    judge_llm_backend: str,
    judge_model: str,
    destroy_env: bool,
) -> None:
    """
    Run a single benchmark case.

    Args:
        problem: Name of the failure/problem to inject
        scenario: Network scenario name
        topo_size: Topology tier as string s/m/l (empty when the scenario has no tiers)
        agent_type: Agent type (e.g. react)
        llm_backend: The LLM backend to use (e.g. openai, ollama, deepseek)
        model: LLM backend model
        max_steps: Maximum agent steps
        judge_llm_backend: LLM backend used for evaluation
        judge_model: Model used for evaluation
        destroy_env: Whether to destroy the network environment after evaluation
    """
    print(f"Running benchmark for Problem: {problem}, Scenario: {scenario}, Topo Size: {topo_size}")

    tier = topo_size if topo_size else None
    if scenario_requires_topo_tier(scenario) and not tier:
        raise ValueError(f"Scenario '{scenario}' requires a non-empty topology tier (-t s|m|l).")
    if not scenario_requires_topo_tier(scenario):
        tier = None

    session_id = start_net_env(scenario, tier, redeploy=True)

    inject_failure(problem_names=[problem], session_id=session_id)

    start_agent(
        agent_type=agent_type,
        llm_backend=llm_backend,
        model=model,
        max_steps=max_steps,
        session_id=session_id,
    )

    eval_results(
        judge_llm_backend=judge_llm_backend,
        judge_model=judge_model,
        destroy_env=destroy_env,
        session_id=session_id,
    )

    if destroy_env:
        net_env_kwargs: dict = {}
        if tier is not None:
            net_env_kwargs["topo_size"] = tier
        net_env = get_net_env_instance(scenario, **net_env_kwargs)
        if net_env.lab_exists():
            net_env.undeploy()


def run_benchmark_from_csv(
    benchmark_file: str,
    agent_type: str,
    llm_backend: str,
    model: str,
    max_steps: int,
    judge_llm_backend: str,
    judge_model: str,
    destroy_env: bool,
) -> None:
    """
    Run benchmark cases defined in a CSV file.

    The CSV file must contain the following columns:
    - problem
    - scenario
    - topo_size (same values as ``nika env run -t``: s, m, l, or empty)
    """
    with open(benchmark_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            run_single_benchmark(
                problem=row["problem"],
                scenario=row["scenario"],
                topo_size=row.get("topo_size") or "",
                agent_type=agent_type,
                llm_backend=llm_backend,
                model=model,
                max_steps=max_steps,
                judge_llm_backend=judge_llm_backend,
                judge_model=judge_model,
                destroy_env=destroy_env,
            )
