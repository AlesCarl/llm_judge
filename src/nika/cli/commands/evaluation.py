"""Commands for offline evaluation (metrics, judge, publish)."""

import typer

eval_app = typer.Typer(help="Evaluate a completed agent session.")


@eval_app.command("metrics")
def eval_metrics(
    session_id: str | None = typer.Option(None, "--session-id", help="Target session id (lab_hash)."),
) -> None:
    """Compute rule-based scores and trace stats; write eval_metrics.json."""
    from nika.workflows.session_eval import run_eval_metrics

    try:
        run_eval_metrics(session_id=session_id)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@eval_app.command("judge")
def eval_judge(
    judge_backend: str = typer.Option(
        ...,
        "-b",
        "--backend",
        help="LLM provider for the judge (openai, ollama, deepseek).",
    ),
    judge_model: str = typer.Option(..., "-m", "--model", help="Judge model id."),

        judge_type: str = typer.Option(
        "single",
        "--judge-type",
        help=(
            "Judge strategy: 'single' (LLMJudge), 'multi' "
            "(Critic/Advocate debate with consensus + synthesis), or "
            "'multi_role' (ChatEval-style N-role sequential debate)."
        ),
    ),

    session_id: str | None = typer.Option(None, "--session-id", help="Target session id (lab_hash)."),
) -> None:
    """Run LLM-as-judge only; write llm_judge.json."""
    from nika.workflows.session_eval import run_llm_judge

    try:
        run_llm_judge(judge_backend, judge_model, judge_type=judge_type, session_id=session_id)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@eval_app.command("publish")
def eval_publish(
    no_destroy: bool = typer.Option(
        False,
        "--no-destroy",
        help="Leave the Kathara lab running after publishing.",
    ),
    session_id: str | None = typer.Option(None, "--session-id", help="Target session id (lab_hash)."),
) -> None:
    """Merge eval artifacts, append one CSV row, then undeploy and clear the session."""
    from nika.workflows.session_eval import publish_session_eval

    try:
        publish_session_eval(destroy_env=not no_destroy, session_id=session_id)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
