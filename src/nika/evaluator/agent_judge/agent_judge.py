"""AgentAsJudge for NIKA.

Fourth judge type, structurally different from the three text-only judges
(single/multi/multi_role): the judge is itself a tool-using agent that
audits the diagnosis agent's OWN trace instead of just reading a final
report and trusting it (the "beauty bias" the text judges are exposed to).

Two phases:

  Phase A (GT-free) — a ReAct agent 
  audits the agent's submitted claims against that SAME session's recorded
  trace, within a hard tool-call budget, and produces a per-claim
  CONFIRMED / REFUTED / COULD-NOT-CHECK verification report. It never sees
  the ground truth.

  Phase B (GT-aware) — a single structured-output call scores the standard
  five criteria using the ground truth AND the Phase-A report; every comment
  must cite the report's verdicts, so the final JudgeResponse stays
  anchored to the active verification instead of collapsing into a plain
  text judge.

Both phases run on a CLOSED session: Kathara is already torn down by the
time ``nika eval judge`` runs, so "tool use" here means reading saved
artifacts (messages.jsonl, submission.json), never live network commands.
This is a deliberate, documented adaptation of the Agent-as-a-Judge paradigm
to post-hoc evaluation, not an oversight.
"""

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from langsmith import tracing_context

from agent.llm.model_factory import load_model
from nika.evaluator.agent_judge.prompts import SCORING_PROMPT, SCORING_SYSTEM, VERIFIER_SYSTEM
from nika.evaluator.agent_judge.verify_tools import build_verify_tools, check_taxonomy
from nika.evaluator.base_judge import BaseJudge
from nika.evaluator.result_log import SUBMISSION_FILENAME
from nika.evaluator.schemas import JudgeResponse
from nika.evaluator.token_meter import dump_cost, meter_config, new_meter

load_dotenv()

logger = logging.getLogger(__name__)

# Tool-call budget for the Phase-A verifier (recursion_limit); keeps the
# judge's cost bounded and comparable to the other three judges.
_VERIFY_BUDGET = 15


# Match a "CLAIM: <claim> — <VERDICT> — <evidence>" line. The verdict token
# anchors the split so an em-dash inside the claim/evidence doesn't break it;
# leading and surrounding markdown bold (``**``) is tolerated because the judge
# model sometimes emits the line in bold, and the evidence tail is optional.
_CLAIM_RE = re.compile(
    r"CLAIM:\s*\**\s*(?P<claim>.*?)\s*[—-]+\s*\**\s*"
    r"(?P<verdict>COULD-NOT-CHECK|CONFIRMED|REFUTED)\b"
    r"\**\s*(?:[—-]+\s*(?P<evidence>.*))?",
    re.IGNORECASE,
)


def _claim_key(claim: str) -> str:
    """Normalise a claim to dedup restatements (bold header vs final line,
    ``=`` vs ``:`` separators) so the same claim is counted once."""
    return re.sub(r"[\s'\"\[\]=:]+", "", claim).lower()


def _parse_claim_verdicts(report: str) -> list[dict]:
    """Extract the verifier's per-claim CLAIM lines into structured records.

    Phase A ends with one ``CLAIM: <claim> — <VERDICT> — <evidence>`` line per
    claim. Parsing them into ``{claim, verdict, evidence}`` gives a
    machine-readable audit field (so downstream analysis can count
    CONFIRMED/REFUTED/COULD-NOT-CHECK per session without regex-scraping the
    free-text report). The model may restate a claim (an inline bold summary
    plus a final line); dedup by normalised claim, last occurrence winning so
    the authoritative final line prevails. Best-effort: unparseable reports
    (e.g. budget exhausted) simply yield an empty list.
    """
    seen: dict[str, dict] = {}
    order: list[str] = []
    for line in report.splitlines():
        if "CLAIM:" not in line.upper():
            continue
        m = _CLAIM_RE.search(line)
        if not m:
            continue
        key = _claim_key(m.group("claim"))
        if key not in seen:
            order.append(key)
        seen[key] = {
            "claim": m.group("claim").strip().strip("*").strip(),
            "verdict": m.group("verdict").upper(),
            "evidence": (m.group("evidence") or "").strip().strip("*").strip(),
        }
    return [seen[k] for k in order]


def _serialize_messages(messages) -> list[dict]:
    """Flatten a LangGraph message list into a JSON-serialisable audit trail."""
    out = []
    for m in messages:
        entry = {
            "type": getattr(m, "type", m.__class__.__name__),
            "content": str(getattr(m, "content", "")),
        }
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
            ]
        out.append(entry)
    return out


class AgentAsJudge(BaseJudge):
    """Agent-as-a-Judge: tool-grounded verification (Phase A) then scoring (Phase B)."""

    def __init__(
        self,
        judge_llm_backend: str = "openai",
        judge_model: str = "gpt-5-mini",
        verify_budget: int = _VERIFY_BUDGET,
    ) -> None:
        self.judge_llm_backend = judge_llm_backend
        self.judge_model = judge_model
        self.verify_budget = verify_budget

        self._scoring_llm = load_model(
            llm_backend=judge_llm_backend, model=judge_model
        ).with_structured_output(JudgeResponse)

    @staticmethod
    def _read_submission(session_dir: str) -> dict:
        path = Path(session_dir) / SUBMISSION_FILENAME
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _run_verifier(self, session_dir: str, submission: dict, meter) -> tuple[str, list[dict]]:
        """Phase A: GT-free ReAct audit of the submission against this session's trace."""
        verifier_llm = load_model(llm_backend=self.judge_llm_backend, model=self.judge_model)
        verifier_agent = create_react_agent(
            model=verifier_llm,
            tools=build_verify_tools(session_dir),
            prompt=VERIFIER_SYSTEM,
        )

        if submission:
            request = (
                "Claims to audit (the agent's own final submission — NOT the ground truth):\n"
                f"  anomaly present : {submission.get('is_anomaly')}\n"
                f"  root cause      : {submission.get('root_cause_name', [])}\n"
                f"  accused devices : {submission.get('faulty_devices', [])}\n\n"
                "Use the tools to audit these claims against this session's own recorded "
                "trace, then report per-claim verdicts."
            )
        else:
            request = (
                "The agent produced NO final submission this session (submission.json "
                "is absent). There are no claims to confirm — do not treat blank fields "
                "as a 'no anomaly' claim. Use the tools to check whether the recorded "
                "trace shows a fault the agent should have caught, and report that no "
                "submission was made."
            )

        config: dict = {"recursion_limit": self.verify_budget}
        mc = meter_config(meter)
        if mc:
            config["callbacks"] = mc["callbacks"]

        try:
            with tracing_context(enabled=False):
                result = verifier_agent.invoke(
                    {"messages": [HumanMessage(content=request)]}, config=config
                )
            report = str(result["messages"][-1].content)
            if "need more steps" in report.lower():
                report = "(verification ran out of budget before completing)\n" + report
            audit_trace = _serialize_messages(result["messages"])
        except GraphRecursionError:
            report = "(verification ran out of budget before completing)"
            audit_trace = []
        except Exception as exc:  # verification must never kill the eval
            logger.warning("AgentAsJudge — verification pass failed: %s", exc)
            report = f"(verification failed: {exc})"
            audit_trace = []

        return report, audit_trace

    def evaluate_agent(self, ground_truth: str, trace_path: str, save_path: str) -> JudgeResponse:
        """Evaluate the agent through tool-grounded verification then scoring.

        Args:
            ground_truth: The expected correct answer for the task.
            trace_path: Path to the agent's action trace log (messages.jsonl).
            save_path: Path where the resulting JudgeResponse JSON will be saved.

        Returns:
            JudgeResponse: Structured evaluation, schema identical to the other judges.
        """
        session_dir = os.path.dirname(trace_path)

        with open(trace_path, "r") as f:
            raw_trace = f.read()
        trace = self._parse_trace(raw_trace)

        meter = new_meter()

        submission = self._read_submission(session_dir)

        logger.info("AgentAsJudge — Phase A: tool-grounded verification (GT-free)")
        verification_report, audit_trace = self._run_verifier(session_dir, submission, meter)

        taxonomy = check_taxonomy(submission.get("root_cause_name") or [])

        logger.info("AgentAsJudge — Phase B: scoring (GT-aware)")
        scoring_prompt = SCORING_PROMPT.format(
            ground_truth=ground_truth,
            trace=trace,
            verification_report=verification_report,
            taxonomy_check=json.dumps(taxonomy),
        )
        with tracing_context(enabled=False):
            evaluation: JudgeResponse = self._scoring_llm.invoke(
                [SystemMessage(content=SCORING_SYSTEM), HumanMessage(content=scoring_prompt)],
                config=meter_config(meter),
            )

        with open(save_path, "w+") as f:
            f.write(evaluation.model_dump_json(indent=2))

        audit_path = save_path.replace("llm_judge.json", "agent_judge_trace.json")
        with open(audit_path, "w+") as f:
            json.dump(
                {
                    "verification_report": verification_report,
                    "claim_verdicts": _parse_claim_verdicts(verification_report),
                    "taxonomy_check": taxonomy,
                    "tool_call_trace": audit_trace,
                },
                f,
                indent=2,
            )

        dump_cost(meter, save_path, judge="agent", filename="agent_cost.json")

        return evaluation
