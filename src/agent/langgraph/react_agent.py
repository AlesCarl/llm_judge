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
from agent.langgraph.loop_feedback import generate_feedback, is_resolved
from agent.llm.model_factory import load_model
from agent.utils.loggers import AgentCallbackLogger
from nika.evaluator.generic_eval import generic_eval
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
        description="Sanitized redirect hint injected into the next diagnosis attempt.",
    )
    attempt_findings: str = Field(
        default="",
        description="Agent's own diagnosis report from the previous attempt (leak-free; for retry continuity).",
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

        # Judge side of the leak firewall — only needed for the retry loop.
        # Left unloaded when max_loops == 1 so single-shot runs stay identical
        # (and don't pin the judge model in VRAM).
        self.judge_llm = None
        self._gt = {}
        self.fault_family = ""
        # Best-attempt tracking: the final submission.json is the highest-scoring
        # attempt, so a retry that regresses can never make the loop worse than
        # the baseline (keep-best, not keep-last).
        self._best_score = -1.0
        self._best_submission = None
        if self.max_loops > 1:
            self.judge_llm = load_model(llm_backend=judge_llm_backend, model=judge_model)
            gt_path = Path(self.session_dir) / "ground_truth.json"
            if gt_path.exists():
                self._gt = json.loads(gt_path.read_text(encoding="utf-8"))
            self.fault_family = getattr(self.session, "root_cause_category", "") or ""

        
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
        if state.get("resolved", False) or state.get("loop_count", 0) >= self.max_loops:
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
            result = await self.graph.ainvoke(
                {
                    "messages": [HumanMessage(content=task_description)],
                    "task_description": task_description,
                },
                config=config,
            )
            if self.max_loops > 1:
                self._finalize_best_submission()
            return result


    def _submission_score(self, submission: dict) -> float:
        """Scalar ranking of a submission (invalid/-1 dims count as 0)."""
        s = generic_eval(self._gt, submission)
        det, loc_f1, rca_f1 = s[0], s[4], s[8]
        return max(det, 0.0) + max(loc_f1, 0.0) + max(rca_f1, 0.0)


    def _finalize_best_submission(self):
        """Ensure submission.json holds the best-scoring attempt (keep-best)."""
        sub_path = Path(self.session_dir) / "submission.json"
        # A final attempt may have written a submission the judge never scored
        # (e.g. it submitted then timed out): consider it too.
        if sub_path.exists() and sub_path.stat().st_size > 0:
            try:
                current = json.loads(sub_path.read_text(encoding="utf-8"))
                if self._submission_score(current) > self._best_score:
                    self._best_score = self._submission_score(current)
                    self._best_submission = current
            except Exception:
                pass
        if self._best_submission is not None:
            sub_path.write_text(json.dumps(self._best_submission), encoding="utf-8")




    async def diagnosis_agent_builder(self, state: AgentState):
        # On a retry the judge has left a sanitized hint: reset the conversation
        # to [task] + [previous-attempt digest] + [hint] instead of replaying the
        # full (rabbit-hole) history.
        feedback = state.get("judge_feedback", "")
        if feedback:
            # Retry context: a short imperative protocol frames the agent's OWN
            # prior findings (so it can KEEP the confirmed dimensions instead of
            # re-deriving and losing them) plus the keep/fix verdict + coach hint
            # (``feedback`` already carries those). The tool digest is NOT shown
            # to the agent anymore — it stays on the coach side only.
            protocol = (
                "[THIS IS A RETRY — refine, do NOT restart from scratch]\n"
                "1. KEEP the confirmed dimensions: reuse your findings below, do not re-investigate them.\n"
                "2. Use your tools ONLY on the to-fix dimension.\n"
                "3. Submit the confirmed dimensions unchanged + the fixed one updated."
            )
            blocks = [
                HumanMessage(content=state.get("task_description", "")),
                HumanMessage(content=protocol),
            ]
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
        """In-loop verifier: deterministic stop check + (if wrong) redirect hint.

        Reads the ground truth locally and writes ONLY sanitized outputs into
        the shared state — the GT never crosses to the agent side.
        """
        loop_count = state.get("loop_count", 0) + 1
        sub_path = Path(self.session_dir) / "submission.json"
        no_submission = not (sub_path.exists() and sub_path.stat().st_size > 0)

        resolved = False
        scores = None
        submission: dict = {}
        if not no_submission:
            submission = json.loads(sub_path.read_text(encoding="utf-8"))
            shutil.copyfile(sub_path, Path(self.session_dir) / f"submission_attempt_{loop_count}.json")
            scores = generic_eval(self._gt, submission)
            det, loc_f1, rca_f1 = scores[0], scores[4], scores[8]
            resolved = is_resolved(det, loc_f1, rca_f1)

            # Keep-best: remember the highest-scoring submission seen so far.
            total = max(det, 0.0) + max(loc_f1, 0.0) + max(rca_f1, 0.0)
            if total > self._best_score:
                self._best_score = total
                self._best_submission = submission

        self._log_loop_attempt(loop_count, resolved, no_submission, scores)

        # Solved, or out of loop budget → stop without spending the judge LLM.
        if resolved or loop_count >= self.max_loops:
            return {"loop_count": loop_count, "resolved": resolved}

        hint = generate_feedback(
            session_dir=self.session_dir,
            fault_family=self.fault_family,
            gt=self._gt,
            submission=submission,
            scores=scores,
            llm=self.judge_llm,
            no_submission=no_submission,
            loop_count=loop_count,
        )
        return {
            "loop_count": loop_count,
            "resolved": False,
            "judge_feedback": hint,
            "attempt_findings": (state.get("diagnosis_report") or [""])[-1],
        }


    def _log_loop_attempt(self, loop_count, resolved, no_submission, scores):
        """Append a per-attempt record to loop_log.json (dataset for iteration curves)."""
        log_path = Path(self.session_dir) / "loop_log.json"
        history = []
        if log_path.exists():
            try:
                history = json.loads(log_path.read_text(encoding="utf-8"))
            except Exception:
                history = []
        record = {
            "attempt": loop_count,
            "resolved": resolved,
            "no_submission": no_submission,
        }
        if scores is not None:
            record["detection_score"] = scores[0]
            record["localization_f1"] = scores[4]
            record["rca_f1"] = scores[8]
        history.append(record)
        log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
