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
from agent.langgraph.loop_feedback import attempt_digest, generate_feedback, is_resolved
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
    attempt_summary: str = Field(
        default="",
        description="Deterministic digest of the previous attempt's tool activity.",
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
        if self.max_loops > 1:
            self.judge_llm = load_model(llm_backend=judge_llm_backend, model=judge_model)
            gt_path = Path(self.session_dir) / "ground_truth.json"
            if gt_path.exists():
                self._gt = json.loads(gt_path.read_text(encoding="utf-8"))
            self.fault_family = getattr(self.session, "root_cause_category", "") or ""

        # build the state graph
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
        """Step budget schedule: full on the first attempt, reduced on retries."""
        if loop_count == 0:
            return self.max_steps
        return max(10, self.max_steps // 2)


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
            return result




    async def diagnosis_agent_builder(self, state: AgentState):
        # On a retry the judge has left a sanitized hint: reset the conversation
        # to [task] + [previous-attempt digest] + [hint] instead of replaying the
        # full (rabbit-hole) history.
        feedback = state.get("judge_feedback", "")
        if feedback:
            messages = [
                HumanMessage(content=state.get("task_description", "")),
                HumanMessage(content=f"[SUMMARY OF YOUR PREVIOUS ATTEMPT]\n{state.get('attempt_summary', '')}"),
                HumanMessage(
                    content=(
                        "[FEEDBACK FROM REVIEW — you were not correct last time]\n"
                        f"{feedback}\n"
                        "Re-investigate accordingly, then conclude with a fresh submission."
                    )
                ),
            ]
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
        wrong_dims: list[str] = []
        scores = None
        if not no_submission:
            submission = json.loads(sub_path.read_text(encoding="utf-8"))
            shutil.copyfile(sub_path, Path(self.session_dir) / f"submission_attempt_{loop_count}.json")
            scores = generic_eval(self._gt, submission)
            det, loc_f1, rca_f1 = scores[0], scores[4], scores[8]
            resolved = is_resolved(det, loc_f1, rca_f1)
            if det != 1.0:
                wrong_dims.append("detection")
            if not (loc_f1 >= 1.0 - 1e-9):
                wrong_dims.append("localization")
            if not (rca_f1 >= 1.0 - 1e-9):
                wrong_dims.append("root cause")

        self._log_loop_attempt(loop_count, resolved, no_submission, scores)

        # Solved, or out of loop budget → stop without spending the judge LLM.
        if resolved or loop_count >= self.max_loops:
            return {"loop_count": loop_count, "resolved": resolved}

        hint = generate_feedback(
            session_dir=self.session_dir,
            fault_family=self.fault_family,
            gt=self._gt,
            wrong_dims=wrong_dims,
            llm=self.judge_llm,
            no_submission=no_submission,
        )
        return {
            "loop_count": loop_count,
            "resolved": False,
            "judge_feedback": hint,
            "attempt_summary": attempt_digest(self.session_dir),
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
