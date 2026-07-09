import asyncio
import json
import logging
import os
import shutil
from pathlib import Path

import langsmith as ls
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import Field, ValidationError
from typing_extensions import TypedDict

from agent.langgraph.domain_agents.diagnosis_agent import DiagnosisAgent
from agent.langgraph.domain_agents.submission_agent import SubmissionAgent
from agent.langgraph.loop_feedback import (
    VerifierCoach,
    attempt_digest,
    compose_feedback,
    merge_case_file,
)
from agent.llm.model_factory import load_model
from agent.utils.loggers import AgentCallbackLogger, MessageLogger
from nika.utils.logger import system_logger
from nika.utils.session import Session

load_dotenv()


logging.basicConfig(level=logging.INFO)



class AgentState(TypedDict):
    """The state of the agent."""

    messages: list[BaseMessage]
    diagnosis_report: str = Field(
        default="",
        description="The diagnosis report of the network state after analysis.",
    )
    is_max_steps_reached: bool = Field(
        default=False,
        description="Indicates whether the agent has reached the maximum number of steps allowed.",
    )
    # --- retry-loop fields (used only when max_loops > 1) ---
    task_description: str = Field(
        default="",
        description="Original task text, re-presented to the agent on each retry.",
    )
    loop_count: int = Field(
        default=0,
        description="Number of completed diagnosis→submission→judge attempts.",
    )
    resolved: bool = Field(
        default=False,
        description="Whether the deterministic stop check marked the problem solved.",
    )
    judge_feedback: str = Field(
        default="",
        description="Coach review (verdicts + hint) injected into the next diagnosis attempt.",
    )
    attempt_findings: str = Field(
        default="",
        description="Agent's own diagnosis report from the previous attempt (for retry continuity).",
    )
    case_file: list[str] = Field(
        default=[],
        description="Cross-attempt evidence ledger: coach-extracted facts (no conclusions).",
    )
    loop_stop: bool = Field(
        default=False,
        description="Hard stop from the judge node (e.g. submission converged) independent of 'resolved'.",
    )


class BasicReActAgent:

    def __init__(
        self,
        session_id: str,
        llm_backend: str = "openai",
        model: str = "gpt-5-mini",
        max_steps: int = 20,
        max_loops: int = 1,
        judge_llm_backend: str = "ollama",
        judge_model: str = "qwen3.6:35b",
        retry_on_timeout: bool = False,
        verifier_tools: bool = True,
    ):
        self.session_id = session_id
        self.max_steps = max_steps
        self.max_loops = max_loops
        self.retry_on_timeout = retry_on_timeout
        self.session = Session()
        self.session.load_running_session(session_id=session_id)
        self.session_dir = self.session.session_dir
        # Persist the step budget so each run.json is self-documenting.
        self.session.update_session("max_steps", max_steps)
        self.session.update_session("max_loops", max_loops)

        # Set up Langfuse callback handler

        # Initialize Langfuse client
        langfuse = get_client()

        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        self.langfuse_handler = CallbackHandler()

        if langfuse.auth_check():
            system_logger.info("Authentication to Langfuse successful.")
        else:
            system_logger.warning("Authentication to Langfuse failed. Please check your LANGFUSE_API_KEY.")

        # load agent and tools
        diagnosis_agent = DiagnosisAgent(
            session_id=session_id,
            llm_backend=llm_backend,
            model=model,
            scenario_name=self.session.scenario_name,
            problem_names=self.session.problem_names,
        )
        asyncio.run(diagnosis_agent.load_tools())
        self.diagnosis_agent = diagnosis_agent.get_agent()


        # load agent and tools
        submission_agent = SubmissionAgent(session_id=session_id, llm_backend=llm_backend, model=model)
        asyncio.run(submission_agent.load_tools())
        self.submission_agent = submission_agent.get_agent()

        # GT-free in-loop coach — only needed for the retry loop. The ground
        # truth is NEVER read here: the coach judges the agent's answer from
        # evidence only (its trace + optional live-network verification); the
        # GT is used exclusively by the offline evaluator after the run.
        # Left unloaded when max_loops == 1 so single-shot runs stay identical
        # (and don't pin the judge model in VRAM).
        self.coach = None
        if self.max_loops > 1:
            judge_llm = load_model(llm_backend=judge_llm_backend, model=judge_model, temperature=0.1)
            # verifier_tools=True → the coach audits the agent's claims with the
            # same read-only diagnostic MCP tools (grounded verdicts).
            # verifier_tools=False → critique-only ablation arm (no tools).
            self.coach = VerifierCoach(
                llm=judge_llm,
                tools=diagnosis_agent.tools if verifier_tools else None,
                session_dir=self.session_dir,
            )
            self.session.update_session("verifier_tools", bool(verifier_tools))

        
        ###  build the state graph
        self.graph = self._build_loop_graph() if self.max_loops > 1 else self._build_single_graph()






    def _build_single_graph(self):
        """Baseline linear graph (unchanged): diagnosis → submission → END."""
        worker_builder = StateGraph(AgentState)
        worker_builder.add_node("diagnosis_agent", self.diagnosis_agent_builder)
        worker_builder.add_node("submission_agent", self.submission_agent_builder)

        worker_builder.add_edge(START, "diagnosis_agent")
        worker_builder.add_conditional_edges(
            "diagnosis_agent",
            lambda state: state.get("is_max_steps_reached", False),
            {
                True: END,
                False: "submission_agent",
            },
        )
        worker_builder.add_edge("submission_agent", END)
        return worker_builder.compile()


    def _build_loop_graph(self):
        """Retry loop: diagnosis → submission → judge → (retry | END)."""
        worker_builder = StateGraph(AgentState)
        worker_builder.add_node("diagnosis_agent", self.diagnosis_agent_builder)
        worker_builder.add_node("submission_agent", self.submission_agent_builder)
        worker_builder.add_node("judge", self.judge_builder)

        worker_builder.add_edge(START, "diagnosis_agent")

        # Out-of-steps: retry via judge (hint) only if enabled, else END as today.
        timeout_target = "judge" if self.retry_on_timeout else END
        worker_builder.add_conditional_edges(
            "diagnosis_agent",
            lambda state: "timeout" if state.get("is_max_steps_reached", False) else "submit",
            {
                "timeout": timeout_target,
                "submit": "submission_agent",
            },
        )
        worker_builder.add_edge("submission_agent", "judge")
        worker_builder.add_conditional_edges(
            "judge",
            self._judge_router,
            {
                "retry": "diagnosis_agent",
                "end": END,
            },
        )
        return worker_builder.compile()


    def _judge_router(self, state: AgentState) -> str:
        if (
            state.get("resolved", False)
            or state.get("loop_stop", False)
            or state.get("loop_count", 0) >= self.max_loops
        ):
            return "end"
        return "retry"


    def _attempt_budget(self, loop_count: int) -> int:
        """Step budget (recursion_limit) for a diagnosis attempt.

        A retry is NOT a fresh problem on equal footing with the first attempt.
        It restarts from a blank slate ([task, digest, hint]) with no memory of
        the concrete findings the first attempt gathered, so it must first
        RE-DISCOVER the network state (re-run the same tools) BEFORE it can act
        on the feedback. FABRIC traces showed retries doing 2-7x the tool calls
        of the first attempt and hitting the recursion wall (~13 turns at budget
        40) *before* they could submit — the guided work was then thrown away.
        Give retries a larger budget so re-discovery + the feedback-directed
        investigation both fit; the first attempt keeps the baseline budget.
        """
        if loop_count == 0:
            return self.max_steps
        return self.max_steps * 2


    async def run(self, task_description: str): ##
        config = {"callbacks": [self.langfuse_handler]}
        if self.max_loops > 1:
            # Outer-graph safety net against runaway loops (mirrors the inner
            # ReAct recursion_limit); ~4 super-steps per attempt plus margin.
            config["recursion_limit"] = self.max_loops * 4 + 10

        with ls.tracing_context(
            project_name=os.getenv("LANGSMITH_PROJECT", "NIKA"),
            metadata={
                "scenario": self.session.scenario_name,
                "problem": self.session.problem_names[0],
                "topo_size": self.session.scenario_topo_size,
                "model": self.session.model,
            },
        ):
            # Keep-last: submission.json naturally holds the final attempt's
            # answer (no GT is available in-loop to rank attempts; the per-
            # attempt copies let the offline evaluator compute an oracle-best
            # comparison afterwards).
            return await self.graph.ainvoke(
                {
                    "messages": [HumanMessage(content=task_description)],
                    "task_description": task_description,
                },
                config=config,
            )


    @staticmethod
    def _normalize_submission(submission: dict) -> tuple:
        """Order-insensitive fingerprint of a submission (for convergence check)."""
        return (
            bool(submission.get("is_anomaly")),
            tuple(sorted(str(d).lower() for d in (submission.get("faulty_devices") or []))),
            tuple(sorted(str(r).lower() for r in (submission.get("root_cause_name") or []))),
        )


    def _submission_converged(self, submission: dict, loop_count: int) -> bool:
        """True when this attempt's submission equals the previous attempt's.

        A retry that reproduces the same answer despite the coach's challenge
        means further loops would only burn budget — stop early.
        """
        if loop_count < 2:
            return False
        prev_path = Path(self.session_dir) / f"submission_attempt_{loop_count - 1}.json"
        if not prev_path.exists():
            return False
        try:
            prev = json.loads(prev_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return self._normalize_submission(prev) == self._normalize_submission(submission)




    async def diagnosis_agent_builder(self, state: AgentState):
        # Per-attempt marker: lets attempt_digest() slice messages.jsonl to
        # the current attempt only (the file accumulates across attempts).
        MessageLogger(agent="system", session_dir=self.session_dir).log(
            "attempt_start", {"loop": state.get("loop_count", 0)}
        )
        # On a retry the coach has left an evidence-based review: reset the
        # conversation to [task] + [protocol] + [case file] + [own findings] +
        # [review] instead of replaying the full (rabbit-hole) history.
        feedback = state.get("judge_feedback", "")
        if feedback:
            case_file = state.get("case_file") or []
            # Retry context: a short imperative protocol frames the review.
            # The verdicts come from an evidence-based reviewer (no answer
            # key), so SUPPORTED means "well backed", not "guaranteed right".
            protocol_lines = [
                "[THIS IS A RETRY — an independent reviewer graded your previous answer "
                "on evidence; it does NOT know the correct answer]",
                "1. Dimensions graded SUPPORTED: keep them; re-investigate only if new "
                "evidence clearly contradicts them.",
                "2. Dimensions graded WEAK or UNSUPPORTED: re-open them — gather the "
                "missing evidence with your tools; do not resubmit the same answer "
                "unverified.",
            ]
            if case_file:
                protocol_lines.append(
                    "3. Trust the CASE FILE facts below — do not spend steps re-running "
                    "checks that already established them."
                )
            protocol_lines.append(
                f"{len(protocol_lines)}. Conclude with a full submission: kept dimensions "
                "unchanged + the re-worked one(s) updated — always submit your best "
                "hypothesis, never leave it empty."
            )
            blocks = [
                HumanMessage(content=state.get("task_description", "")),
                HumanMessage(content="\n".join(protocol_lines)),
            ]
            if case_file:
                facts = "\n".join(f"- {f}" for f in case_file)
                blocks.append(
                    HumanMessage(content=f"[CASE FILE — facts established in previous attempts]\n{facts}")
                )
            findings = (state.get("attempt_findings") or "").strip()[:2500]
            if findings:
                blocks.append(HumanMessage(content=f"[WHAT YOU FOUND LAST TIME]\n{findings}"))
            blocks.append(HumanMessage(content=feedback))
            messages = blocks
        else:
            messages = state["messages"]

        budget = self._attempt_budget(state.get("loop_count", 0))
        try:
            cb = AgentCallbackLogger(agent="diagnosis_agent", session_dir=self.session_dir)
            diagnosis_report = await self.diagnosis_agent.ainvoke(
                {"messages": messages},
                config={
                    "callbacks": [cb],
                    "recursion_limit": budget,
                },
                debug=True,
            )
            return {"diagnosis_report": [diagnosis_report["messages"][-1].content], "is_max_steps_reached": False}
        except ValidationError as e:
            AgentCallbackLogger(agent="diagnosis_agent", session_dir=self.session_dir)._log(
                "error", {"message": f"Validation error: {e}"}
            )
            return {
                "messages": [HumanMessage(content=f"Error: {e}")],
                "diagnosis_report": ["ERROR_VALIDATION"],
                "is_max_steps_reached": False,
            }
        except GraphRecursionError:
            AgentCallbackLogger(agent="diagnosis_agent", session_dir=self.session_dir)._log(
                "error",
                {"message": "Diagnosis agent reached max recursion limit."},
            )
            return {
                "messages": [HumanMessage(content="Error: diagnosis did not finish within max steps.")],
                "diagnosis_report": ["ERROR_MAX_STEPS_REACHED"],
                "is_max_steps_reached": True,
            }




    async def submission_agent_builder(self, state: AgentState):
        diag_text = state["diagnosis_report"][-1]
        result = await self.submission_agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Based on the diagnosis report: {diag_text}, please provide the submission. Do not submit if no report available."
                    ),
                ]
            },
            config={
                "callbacks": [AgentCallbackLogger(agent="submission_agent", session_dir=self.session_dir)],
                "recursion_limit": self.max_steps,
            },
            debug=True,
        )
        return {
            "messages": result["messages"],
        }


    async def judge_builder(self, state: AgentState):
        """GT-free in-loop reviewer: coach verdict + redirect feedback.

        The ground truth is NEVER read here. Stop signals, in order:
        budget exhausted → converged (same submission twice) → coach approved
        (every dimension graded SUPPORTED from evidence).
        """
        loop_count = state.get("loop_count", 0) + 1
        sub_path = Path(self.session_dir) / "submission.json"
        no_submission = not (sub_path.exists() and sub_path.stat().st_size > 0)

        submission: dict = {}
        converged = False
        if not no_submission:
            submission = json.loads(sub_path.read_text(encoding="utf-8"))
            shutil.copyfile(sub_path, Path(self.session_dir) / f"submission_attempt_{loop_count}.json")
            converged = self._submission_converged(submission, loop_count)

        # Out of loop budget → stop without spending the coach LLM (the final
        # attempt's quality is judged offline, with the GT, after the run).
        if loop_count >= self.max_loops:
            self._log_loop_attempt(loop_count, no_submission, stop_reason="budget_exhausted")
            return {"loop_count": loop_count, "resolved": False}

        # Same answer as the previous attempt despite the challenge → another
        # loop would only repeat itself; stop early.
        if converged:
            self._log_loop_attempt(loop_count, no_submission, stop_reason="converged")
            return {"loop_count": loop_count, "resolved": False, "loop_stop": True}

        diagnosis_report = (state.get("diagnosis_report") or [""])[-1]
        review = await self.coach.review(
            task_description=state.get("task_description", ""),
            submission=submission,
            diagnosis_report=diagnosis_report,
            digest=attempt_digest(self.session_dir),
            no_submission=no_submission,
        )

        resolved = (not no_submission) and review.approved
        self._log_loop_attempt(
            loop_count,
            no_submission,
            stop_reason="coach_approved" if resolved else None,
            review=review,
        )
        if resolved:
            return {"loop_count": loop_count, "resolved": True}

        feedback = compose_feedback(
            review=review,
            submission=submission,
            loop_count=loop_count,
            no_submission=no_submission,
        )
        return {
            "loop_count": loop_count,
            "resolved": False,
            "judge_feedback": feedback,
            "attempt_findings": diagnosis_report,
            "case_file": merge_case_file(state.get("case_file") or [], review.new_facts),
        }


    def _log_loop_attempt(self, loop_count, no_submission, stop_reason=None, review=None):
        """Append a per-attempt record to loop_log.json (dataset for iteration curves).

        GT scores are no longer recorded here (no GT in the loop); the offline
        evaluator recomputes them from submission_attempt_N.json + GT.
        """
        log_path = Path(self.session_dir) / "loop_log.json"
        history = []
        if log_path.exists():
            try:
                history = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                history = []
        record = {
            "attempt": loop_count,
            "resolved": stop_reason == "coach_approved",
            "no_submission": no_submission,
            "stop_reason": stop_reason,
        }
        if review is not None:
            record["coach_verdicts"] = {dim: status for dim, (status, _) in review.statuses.items()}
            record["suspected_family"] = review.suspected_family
            record["verified"] = review.verified
        history.append(record)
        log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
