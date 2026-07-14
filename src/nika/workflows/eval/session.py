"""Session evaluation: numeric metrics, LLM judge, and publish on closed sessions."""

import json
import os
import shutil
import textwrap
import time
from pathlib import Path

from langchain_core.exceptions import OutputParserException

from nika.evaluator.generic_eval import generic_eval
from nika.evaluator.llm_judge import LLMJudge
from nika.evaluator.multi_agent_judge import MultiAgentJudge
from nika.evaluator.multi_role_debate.multi_role_debate_judge import MultiRoleDebateJudge
from nika.evaluator.result_log import EVAL_METRICS_FILENAME, MESSAGES_FILENAME
from nika.evaluator.trace_parser import AgentTraceParser
from nika.utils.logger import bind_session_dir, log_event, system_logger
from nika.utils.session import Session
from nika.workflows.session.close import close_session

logger = system_logger

# ``generic_eval`` now lives in ``nika.evaluator.generic_eval`` so the in-loop
# agent can reuse it without importing this module (which pulls in the LLM
# judges). Re-exported here to keep existing call sites working.


def run_eval_metrics(*, session_id: str | None = None) -> None:
    """Compute rule-based scores and trace stats; write ``eval_metrics.json`` under the session dir."""
    session = Session()
    session.load_closed_session(session_id=session_id)
    bind_session_dir(session.session_dir)

    gt_path = Path(session.session_dir) / "ground_truth.json"
    gt = json.loads(gt_path.read_text())

    submission_path = Path(session.session_dir) / "submission.json"
    if submission_path.exists():
        submission = json.loads(submission_path.read_text())
        (
            detection_score,
            loc_acc,
            loc_prec,
            loc_rec,
            loc_f1,
            rca_acc,
            rca_prec,
            rca_rec,
            rca_f1,
        ) = generic_eval(gt, submission)
    else:
        logger.error(f"Submission file not found: {submission_path}")
        detection_score = -1.0
        loc_acc = loc_prec = loc_rec = loc_f1 = -1.0
        rca_acc = rca_prec = rca_rec = rca_f1 = -1.0

    trace_path = os.path.join(session.session_dir, MESSAGES_FILENAME)
    trace_metrics = AgentTraceParser(trace_path=trace_path).parse_trace()

    payload = {
        "detection_score": detection_score,
        "localization_accuracy": loc_acc,
        "localization_precision": loc_prec,
        "localization_recall": loc_rec,
        "localization_f1": loc_f1,
        "rca_accuracy": rca_acc,
        "rca_precision": rca_prec,
        "rca_recall": rca_rec,
        "rca_f1": rca_f1,
        "in_tokens": trace_metrics.get("in_tokens"),
        "out_tokens": trace_metrics.get("out_tokens"),
        "steps": trace_metrics.get("steps"),
        "tool_calls": trace_metrics.get("tool_calls"),
        "tool_errors": trace_metrics.get("tool_errors"),
    }
    out_path = Path(session.session_dir) / EVAL_METRICS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    session.update_run_meta("eval_metrics", payload)
    log_event("eval_metrics_saved", f"Wrote numeric eval metrics to {out_path}", session_id=session.session_id)


def _first_attempt_trace(session_dir: str) -> str | None:
    """Write an attempt-1-only slice of ``messages.jsonl`` for the PRE-feedback judge.

    The diagnosis node writes an ``attempt_start`` marker at the top of every
    attempt (loop 0 included). Cutting the trace just before the 2nd marker
    leaves exactly the first attempt — whose final submission is the
    pre-feedback answer. Returns the slice path, or ``None`` when the run had a
    single attempt (no feedback happened → PRE == POST, caller copies instead).
    """
    src = Path(session_dir) / MESSAGES_FILENAME
    if not src.exists():
        return None
    lines = src.read_text(encoding="utf-8").splitlines()
    marks = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            if json.loads(line).get("event") == "attempt_start":
                marks.append(i)
        except Exception:
            continue
    if len(marks) < 2:
        return None
    out = Path(session_dir) / "messages_attempt1.jsonl"
    out.write_text("\n".join(lines[: marks[1]]), encoding="utf-8")
    return str(out)


def run_llm_judge(
    judge_llm_backend: str,
    judge_model: str,
    *,
    judge_type: str = "single",
    session_id: str | None = None,
) -> None:
    """Run LLM-as-judge only; writes ``llm_judge.json`` under the session dir.

    Args:
        judge_llm_backend: LLM provider (openai, ollama, deepseek).
        judge_model: Model id for the chosen backend.
        judge_type: ``single`` for LLMJudge, ``multi`` for MultiAgentJudge
            (Critic/Advocate debate with consensus + synthesis), or
            ``multi_role`` for MultiRoleDebateJudge (ChatEval-style N-role
            sequential debate with numeric aggregation).
        session_id: Target session id (auto-detected when only one closed).
    """
    session = Session()
    session.load_closed_session(session_id=session_id)
    bind_session_dir(session.session_dir)

    gt_path = Path(session.session_dir) / "ground_truth.json"
    gt = json.loads(gt_path.read_text())

    trace_path = os.path.join(session.session_dir, MESSAGES_FILENAME)
    logger.info(f"Evaluating session {session.session_id} using LLM-as-Judge (judge_type={judge_type}).")

    gt_text = textwrap.dedent(
        f"""\
            The root cause is {gt["root_cause_name"]}.
            The faulty devices are: {", ".join(gt["faulty_devices"])}.
        """
    )

    def _make_judge():
        # A fresh instance per call: ``LLMJudge.evaluate_agent`` mutates
        # ``self.prompt`` in place, so the same object cannot judge two traces.
        if judge_type == "multi":
            return MultiAgentJudge(judge_llm_backend=judge_llm_backend, judge_model=judge_model)
        if judge_type == "multi_role":
            return MultiRoleDebateJudge(judge_llm_backend=judge_llm_backend, judge_model=judge_model)
        return LLMJudge(judge_llm_backend=judge_llm_backend, judge_model=judge_model)

    def _judge_trace(trace: str, filename: str, *, run_meta_key: str | None = None) -> None:
        """Judge one trace into ``filename`` (offline); stamp eval_time; log parse errors."""
        start_time = time.time()
        try:
            _make_judge().evaluate_agent(
                ground_truth=gt_text,
                trace_path=trace,
                save_path=f"{session.session_dir}/{filename}",
            )
            eval_time = round(time.time() - start_time, 2)
            logger.info(f"LLM Judge ({filename}) eval time: {eval_time}s")

            judge_path = Path(session.session_dir) / filename
            if judge_path.exists():
                data = json.loads(judge_path.read_text(encoding="utf-8"))
                data["eval_time"] = eval_time
                judge_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                if run_meta_key:
                    session.update_run_meta(run_meta_key, data)
        except OutputParserException as exc:
            eval_time = round(time.time() - start_time, 2)
            logger.warning(f"LLM Judge ({filename}) output not parseable as JSON; saved as .md. ({eval_time}s)")
            md_path = Path(session.session_dir) / filename.replace(".json", ".md")
            md_path.write_text(exc.llm_output, encoding="utf-8")

    # POST-feedback: the full trace (all attempts → final submission).
    _judge_trace(trace_path, "llm_judge.json", run_meta_key="llm_judge")

    # PRE-feedback: attempt-1 slice (its last submission is the pre-feedback
    # answer). Computed fully offline from the same artifacts. Single-attempt
    # runs had no feedback, so PRE just mirrors POST.
    pre_trace = _first_attempt_trace(session.session_dir)
    pre_path = Path(session.session_dir) / "PRE_llm_judge.json"
    if pre_trace:
        _judge_trace(pre_trace, "PRE_llm_judge.json")
    else:
        post_path = Path(session.session_dir) / "llm_judge.json"
        if post_path.exists():
            shutil.copyfile(post_path, pre_path)


def publish_session_eval(*, session_id: str | None = None) -> None:
    """Validate eval artifacts on a closed session and record publish completion."""
    session = Session()
    session.load_closed_session(session_id=session_id)
    bind_session_dir(session.session_dir)

    metrics_path = Path(session.session_dir) / EVAL_METRICS_FILENAME
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"eval_metrics.json not found under {session.session_dir}. Run `nika eval metrics` first."
        )

    log_event(
        "eval_publish",
        f"Published evaluation for session {session.session_id} (scenario {session.scenario_name}).",
        session_id=session.session_id,
        scenario=session.scenario_name,
    )


def eval_results(
    *,
    destroy_env: bool = True,
    session_id: str | None = None,
    run_judge: bool = False,
    judge_llm_backend: str | None = None,
    judge_model: str | None = None,
    judge_type: str = "single",
) -> None:
    """Close the session, then run metrics and publish; LLM judge runs only when ``run_judge`` is set."""
    if run_judge and (not judge_llm_backend or not judge_model):
        raise ValueError("--judge-backend and --judge-model are required when run_judge is enabled.")

    session = Session()
    session.load_running_session(session_id=session_id)
    resolved_session_id = session.session_id
    close_session(session_id=resolved_session_id, undeploy=destroy_env)
    run_eval_metrics(session_id=resolved_session_id)
    if run_judge:
        run_llm_judge(
            judge_llm_backend, judge_model, judge_type=judge_type, session_id=resolved_session_id
        )
    publish_session_eval(session_id=resolved_session_id)
