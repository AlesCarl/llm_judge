"""Commands for running diagnosis agents."""

import typer

from agent.cli.codex_worker import REASONING_EFFORT_LEVELS

SUPPORTED_AGENT_TYPES = ("react", "mock", "cli")
SUPPORTED_LLM_BACKENDS = ("openai", "ollama", "deepseek")

agent_app = typer.Typer(help="Troubleshooting agents.")


@agent_app.command("list")
def agent_list() -> None:
    """Print supported agent types and LLM backends."""
    typer.echo("agent_types:")
    for agent_type in SUPPORTED_AGENT_TYPES:
        typer.echo(f"  {agent_type}")
    typer.echo("llm_backends:")
    for backend in SUPPORTED_LLM_BACKENDS:
        typer.echo(f"  {backend}")
    typer.echo("reasoning_effort (cli only):")
    for level in REASONING_EFFORT_LEVELS:
        typer.echo(f"  {level}")


@agent_app.command("run")
def agent_run(
    agent_type: str = typer.Option("react", "-a", "--agent", help="Agent implementation."),
    llm_backend: str = typer.Option("openai", "-b", "--backend", help="LLM provider (openai, ollama, deepseek)."),
    model: str = typer.Option("gpt-5-mini", "-m", "--model", help="Model id for the chosen backend."),
    max_steps: int = typer.Option(
        20,
        "-n",
        "--max-steps",
        help="Max ReAct steps (react and mock only; ignored for cli).",
    ),
    max_loops: int = typer.Option(
        1,
        "-l",
        "--max-loops",
        help="Max retry loops with judge feedback (react only). 1 = no loop (baseline).",
    ),
    judge_backend: str = typer.Option(
        "ollama",
        "--judge-backend",
        help="LLM provider for the in-loop judge (used only when --max-loops > 1).",
    ),
    judge_model: str = typer.Option(
        "qwen3.6:35b",
        "--judge-model",
        help="Model id for the in-loop judge (used only when --max-loops > 1).",
    ),
    retry_on_timeout: bool = typer.Option(
        False,
        "--retry-on-timeout",
        help="Also retry when the agent runs out of steps without submitting (react loop only).",
    ),
    verifier_tools: bool = typer.Option(
        True,
        "--verifier-tools/--no-verifier-tools",
        help=(
            "In-loop coach verifies the agent's claims with read-only network tools "
            "(GT-free). --no-verifier-tools = critique-only ablation arm."
        ),
    ),
    reasoning_effort: str | None = typer.Option(
        None,
        "-e",
        "--reasoning-effort",
        help="Codex model_reasoning_effort (cli only): none, minimal, low, medium, high, xhigh.",
    ),
    session_id: str | None = typer.Option(None, "--session-id", help="Target session id (lab_hash)."),
) -> None:
    """Run the agent on the current session task."""
    from nika.workflows.agent.run import start_agent

    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORT_LEVELS:
        raise typer.BadParameter(
            f"reasoning_effort must be one of {', '.join(REASONING_EFFORT_LEVELS)}"
        )
    if max_loops < 1:
        raise typer.BadParameter("--max-loops must be >= 1.")

    try:
        start_agent(
            agent_type,
            llm_backend,
            model,
            max_steps,
            max_loops=max_loops,
            judge_llm_backend=judge_backend,
            judge_model=judge_model,
            retry_on_timeout=retry_on_timeout,
            verifier_tools=verifier_tools,
            session_id=session_id,
            reasoning_effort=reasoning_effort,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
