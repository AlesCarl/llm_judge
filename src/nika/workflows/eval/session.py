"""Session evaluation: numeric metrics and LLM judge on closed sessions."""

import json
import os
import time
from pathlib import Path

from langchain_core.exceptions import OutputParserException

from nika.config import resolve_results_root
from nika.evaluator.llm_judge import LLMJudge
from nika.evaluator.result_log import EVAL_METRICS_FILENAME, MESSAGES_FILENAME
from nika.evaluator.trace_parser import AgentTraceParser
from nika.evaluator.scoring import (
    score_detection,
    score_rca_v2,
)
from nika.utils.logger import bind_session_dir, log_event, system_logger
from nika.utils.session import Session
from nika.utils.session_artifacts import (
    RUN_FILENAME,
    is_finished_session,
    iter_session_dirs,
)
from nika.utils.session_store import SessionStore
from nika.workflows.session.close import close_session

logger = system_logger


def _format_judge_ground_truth(gt: dict) -> str:
    causes = gt.get("root_causes") or []
    if not causes:
        return "No structured root causes (healthy or unlabeled session)."
    lines = ["Structured root causes (resource + fault_type):"]
    for item in causes:
        resource = item.get("resource") or {}
        resource_id = item.get("resource_id") or resource.get("id") or resource
        lines.append(f"- {resource_id} type={item.get('fault_type')}")
    return "\n".join(lines)


def _session_is_still_running(session_id: str) -> bool:
    try:
        return SessionStore().get_session(session_id).get("status") == "running"
    except FileNotFoundError:
        return False


def _iter_eval_session_ids(
    *,
    session_id: str | None = None,
    result_dir: str | Path | None = None,
) -> list[str]:
    """Return session ids to evaluate under *result_dir* (or the default results root)."""
    if session_id is not None:
        return [session_id]

    results_root = resolve_results_root(result_dir)
    candidates: list[str] = []
    for session_dir in iter_session_dirs(results_root):
        run_meta = json.loads((session_dir / RUN_FILENAME).read_text(encoding="utf-8"))
        if not is_finished_session(run_meta):
            continue
        sid = run_meta.get("session_id") or session_dir.name
        if _session_is_still_running(sid):
            continue
        candidates.append(sid)

    if not candidates:
        raise FileNotFoundError(
            f"No closed session found under {results_root}/. "
            "Close a session with `nika session close` first."
        )
    if result_dir is None and len(candidates) > 1:
        raise ValueError(
            "Multiple closed sessions found under results/. Please pass --session_id to select one."
        )
    return candidates


def generic_eval(gt, submission):
    """Score detection and pair-based RCA from structured ``gt`` and ``submission``."""
    detection_score = score_detection(submission, gt)
    return {
        "detection_score": detection_score,
        **score_rca_v2(submission, gt),
    }


def build_eval_metrics_payload(
    *,
    gt: dict,
    submission: dict | None,
    trace_metrics: dict,
) -> dict:
    """Build the persisted rule-based metrics payload from session artifacts."""
    if submission is not None:
        scores = generic_eval(gt, submission)
    else:
        scores = {
            "detection_score": -1.0,
            "localization_accuracy": -1.0,
            "localization_precision": -1.0,
            "localization_recall": -1.0,
            "localization_f1": -1.0,
            "rca_accuracy": -1.0,
            "rca_precision": -1.0,
            "rca_recall": -1.0,
            "rca_f1": -1.0,
            "fault_type_precision": -1.0,
            "fault_type_recall": -1.0,
            "fault_type_f1": -1.0,
        }

    return {
        **scores,
        "in_tokens": trace_metrics.get("in_tokens"),
        "out_tokens": trace_metrics.get("out_tokens"),
        "steps": trace_metrics.get("steps"),
        "tool_calls": trace_metrics.get("tool_calls"),
        "tool_errors": trace_metrics.get("tool_errors"),
    }


def run_eval_metrics(
    *,
    session_id: str | None = None,
    result_dir: str | Path | None = None,
) -> None:
    """Compute rule-based scores and trace stats; write ``eval_metrics.json`` under each session dir."""
    for sid in _iter_eval_session_ids(session_id=session_id, result_dir=result_dir):
        _run_eval_metrics_one(session_id=sid, result_dir=result_dir)


def _run_eval_metrics_one(
    *,
    session_id: str,
    result_dir: str | Path | None = None,
) -> None:
    session = Session()
    session.load_closed_session(session_id=session_id, result_dir=result_dir)
    bind_session_dir(session.session_dir)

    gt_path = Path(session.session_dir) / "ground_truth.json"
    gt = json.loads(gt_path.read_text())

    submission_path = Path(session.session_dir) / "submission.json"
    submission = (
        json.loads(submission_path.read_text()) if submission_path.exists() else None
    )
    if submission is None:
        logger.error(f"Submission file not found: {submission_path}")

    trace_path = os.path.join(session.session_dir, MESSAGES_FILENAME)
    trace_metrics = AgentTraceParser(trace_path=trace_path).parse_trace()
    payload = build_eval_metrics_payload(
        gt=gt,
        submission=submission,
        trace_metrics=trace_metrics,
    )
    out_path = Path(session.session_dir) / EVAL_METRICS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    session.update_run_meta("eval_metrics", payload)
    log_event(
        "eval_metrics_saved",
        f"Wrote numeric eval metrics to {out_path}",
        session_id=session.session_id,
    )
    log_event(
        "eval_publish",
        f"Published evaluation for session {session.session_id} (scenario {session.scenario_name}).",
        session_id=session.session_id,
        scenario=session.scenario_name,
    )


def run_llm_judge(
    judge_llm_provider: str,
    judge_model: str,
    *,
    judge_type: str = "single",
    session_id: str | None = None,
    result_dir: str | Path | None = None,
) -> None:
    """Run LLM-as-judge only; writes ``llm_judge.json`` under each selected session dir.

    Args:
        judge_llm_provider: LLM provider (openai, deepseek, anthropic, custom).
        judge_model: Model id for the chosen provider.
        judge_type: ``single`` for LLMJudge, ``multi`` for MultiAgentJudge
            (Critic/Advocate debate; the final judge is unbounded and the
            debater order is shuffled), ``multi_role`` for MultiRoleDebateJudge
            (ChatEval-style N-role sequential debate with numeric aggregation),
            or ``agent`` for AgentAsJudge (tool-grounded verification of the
            session's own trace, then GT-aware scoring).
        session_id: Target session id (auto-detected when only one closed).
        result_dir: Results parent directory (default: run config result dir).
    """
    for sid in _iter_eval_session_ids(session_id=session_id, result_dir=result_dir):
        _run_llm_judge_one(
            judge_llm_provider,
            judge_model,
            judge_type=judge_type,
            session_id=sid,
            result_dir=result_dir,
        )


def _build_judge(judge_type: str, judge_llm_provider: str, judge_model: str):
    # Imported lazily so `nika eval metrics` does not load the debate/agent judges.
    if judge_type == "multi":
        from nika.evaluator.multi_agent_judge import MultiAgentJudge

        return MultiAgentJudge(judge_llm_backend=judge_llm_provider, judge_model=judge_model)
    if judge_type == "multi_role":
        from nika.evaluator.multi_role_debate.multi_role_debate_judge import MultiRoleDebateJudge

        return MultiRoleDebateJudge(judge_llm_backend=judge_llm_provider, judge_model=judge_model)
    if judge_type == "agent":
        from nika.evaluator.agent_judge import AgentAsJudge

        return AgentAsJudge(judge_llm_backend=judge_llm_provider, judge_model=judge_model)
    if judge_type == "single":
        return LLMJudge(judge_llm_provider=judge_llm_provider, judge_model=judge_model)
    raise ValueError(
        f"Unknown judge_type {judge_type!r}; expected single, multi, multi_role or agent."
    )


def _run_llm_judge_one(
    judge_llm_provider: str,
    judge_model: str,
    *,
    judge_type: str = "single",
    session_id: str,
    result_dir: str | Path | None = None,
) -> None:
    session = Session()
    session.load_closed_session(session_id=session_id, result_dir=result_dir)
    bind_session_dir(session.session_dir)

    gt_path = Path(session.session_dir) / "ground_truth.json"
    gt = json.loads(gt_path.read_text())

    trace_path = os.path.join(session.session_dir, MESSAGES_FILENAME)
    logger.info(
        f"Evaluating session {session.session_id} using LLM-as-Judge (judge_type={judge_type})."
    )

    llm_judge = _build_judge(judge_type, judge_llm_provider, judge_model)
    start_time = time.time()
    try:
        llm_judge.evaluate_agent(
            ground_truth=_format_judge_ground_truth(gt),
            trace_path=trace_path,
            save_path=f"{session.session_dir}/llm_judge.json",
        )
        eval_time = round(time.time() - start_time, 2)
        logger.info(f"LLM Judge eval time: {eval_time}s")

        judge_path = Path(session.session_dir) / "llm_judge.json"
        if judge_path.exists():
            data = json.loads(judge_path.read_text(encoding="utf-8"))
            data["eval_time"] = eval_time
            judge_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            session.update_run_meta("llm_judge", data)
    except OutputParserException as exc:
        eval_time = round(time.time() - start_time, 2)
        logger.warning(f"LLM Judge output not parseable as JSON; saved as .md. ({eval_time}s)")
        md_path = Path(session.session_dir) / "llm_judge.md"
        md_path.write_text(exc.llm_output, encoding="utf-8")


def eval_results(
    *,
    destroy_env: bool = True,
    session_id: str | None = None,
) -> None:
    """Close the session, then write rule-based ``eval_metrics.json``.

    LLM judge and CSV summary are offline steps via ``nika eval judge`` /
    ``nika eval summary``.
    """
    resolved_session_id = session_id
    try:
        session = Session()
        session.load_running_session(session_id=session_id)
        resolved_session_id = session.session_id
        close_session(session_id=resolved_session_id, undeploy=destroy_env)
    except FileNotFoundError:
        # Session may already have been closed by another process while a long
        # agent run was still in flight; evaluate from results artifacts.
        if resolved_session_id is None:
            raise
    run_eval_metrics(session_id=resolved_session_id)
