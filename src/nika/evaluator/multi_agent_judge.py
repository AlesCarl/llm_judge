
##
## - Implementa `MultiAgentJudge` estendendo `BaseJudge`
## - Contiene tutta la logica del debate (player, round, moderator)


import json
import os

from dotenv import load_dotenv
from langsmith import tracing_context
from pydantic import BaseModel, Field

# from agent.llm.langchain_deepseek import DeepSeekLLM
from agent.llm.model_factory import load_model
from agent.utils.template import LLM_JUDGE_PROMPT_TEMPLATE
from nika.config import RESULTS_DIR
from nika.orchestrator.problems.prob_pool import get_problem_instance


"""
MultiAgentJudge for NIKA.

Multiple LLM agents deliberate over the agent's action trace; a moderator
checks for consensus after each round; a final synthesizer produces a
JudgeResponse identical in schema to the one returned by LLMJudge.

Public interface mirrors LLMJudge so the two are drop-in replacements.
"""

import json
import logging

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import tracing_context

from nika.evaluator.base_judge import BaseJudge
from nika.evaluator.schemas import JudgeResponse




from agent.llm.model_factory import load_model
from agent.utils.template import (
    DEBATER_SYSTEM_PROMPT,
    INITIAL_EVALUATION_PROMPT,
    MODERATOR_PROMPT,
    MODERATOR_SYSTEM_PROMPT,
    REBUTTAL_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)
from nika.evaluator.llm_judge import JudgeResponse



load_dotenv()

logger = logging.getLogger(__name__)



# ─── Debate player ────────────────────────────────────────────────────────────

""" 
NOTA:  MAD Agent (classe base) aveva già tutti quei metodi: add_event, ask ..  —> quindi DebatePlayer ereditava tutto 
    In NIKA no
"""


class DebatePlayer:
    """A single debate participant backed by a LangChain chat model."""

    def __init__(self, llm: BaseChatModel, name: str) -> None:
        self.llm = llm
        self.name = name
        self._messages: list = []

    def set_system_prompt(self, prompt: str) -> None:
        self._messages = [SystemMessage(content=prompt)]

    def add_user_message(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))

    def add_assistant_message(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))

    def speak(self) -> str:
        with tracing_context(enabled=False):
            response: AIMessage = self.llm.invoke(self._messages)
        answer = str(response.content)
        self.add_assistant_message(answer)
        logger.debug("[%s]\n%s", self.name, answer)
        return answer


# ─── Multi-agent judge ────────────────────────────────────────────────────────

class MultiAgentJudge(BaseJudge):
    """
    Debate-based judge for NIKA.

    Creates `num_debaters` LLM agents that independently evaluate the trace,
    then exchange rebuttals for up to `max_rounds` rounds. A moderator checks
    for consensus after each round. A final synthesis step produces a
    JudgeResponse using structured output.
    """

    def __init__(
        self,
        judge_llm_backend: str = "openai",
        judge_model: str = "gpt-4o-mini",
        num_debaters: int = 2,
        max_rounds: int = 3,
    ) -> None:
        
        """
        Args:
            judge_llm_backend: Backend name for model_factory ("openai", "ollama", "deepseek").
            judge_model: Model identifier for the chosen backend.
            num_debaters: Number of debating agents (≥2 recommended).
            max_rounds: Maximum debate rounds before forcing a synthesis.
        """
        self.judge_llm_backend = judge_llm_backend
        self.judge_model = judge_model
        self.num_debaters = num_debaters
        self.max_rounds = max_rounds

        # Plain LLM for debaters and moderator 
        self._llm: BaseChatModel = load_model(llm_backend=judge_llm_backend, model=judge_model)
        # Structured LLM for the final synthesis step
        self._synthesis_llm: BaseChatModel = load_model(
            llm_backend=judge_llm_backend, model=judge_model
        ).with_structured_output(JudgeResponse)




    #### helpers: 

    def _create_debaters(self) -> list[DebatePlayer]:
        players = []
        for i in range(self.num_debaters):
            player = DebatePlayer(llm=self._llm, name=f"Judge-{i + 1}")
            player.set_system_prompt(DEBATER_SYSTEM_PROMPT)
            players.append(player)
        return players

    def _create_moderator(self) -> DebatePlayer:
        moderator = DebatePlayer(llm=self._llm, name="Moderator")
        moderator.set_system_prompt(MODERATOR_SYSTEM_PROMPT)
        return moderator

    def _check_consensus(
        self, moderator: DebatePlayer, assessments: list[str]
    ) -> tuple[bool, str]:
        """Ask the moderator whether consensus has been reached.

        Returns:
            (consensus: bool, summary: str)
        """
        joined = "\n\n---\n\n".join(
            f"[Judge-{i + 1}]\n{text}" for i, text in enumerate(assessments)
        )
        moderator.add_user_message(MODERATOR_PROMPT.format(assessments=joined))
        raw = moderator.speak()

        # Strip possible markdown fences before JSON parsing
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            parsed = json.loads(clean)
            return bool(parsed.get("consensus", False)), parsed.get("summary", "")
        except json.JSONDecodeError:
            logger.warning("Moderator returned non-JSON; treating as no consensus.")
            return False, raw

    def _synthesize(self, debate_transcript: str) -> JudgeResponse:
        """Produce the final structured JudgeResponse from the full debate transcript."""
        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=SYNTHESIS_PROMPT.format(debate_transcript=debate_transcript)),
        ]
        with tracing_context(enabled=False):
            result: JudgeResponse = self._synthesis_llm.invoke(messages)
        return result

    def _build_transcript(self, rounds: list[dict]) -> str:
        lines = []
        for r in rounds:
            lines.append(f"=== Round {r['round']} ===")
            for name, text in r["assessments"].items():
                lines.append(f"[{name}]\n{text}")
            if r.get("moderator_summary"):
                lines.append(f"[Moderator]\n{r['moderator_summary']}")
            lines.append("")
        return "\n".join(lines)

    
    ### Public API *** 

    def evaluate_agent(
        self, ground_truth: str, trace_path: str, save_path: str
    ) -> JudgeResponse:
        """Evaluate the agent through multi-agent debate.

        ## Più LLM dibattono per round, moderatore verifica consensus, synthesizer produce il JudgeResponse

        Args:
            ground_truth: The expected correct answer / ground truth for the task.
            trace_path: Path to the agent's action trace log file.
            save_path: Path where the resulting JudgeResponse JSON will be saved.

        Returns:
            JudgeResponse: Structured evaluation identical in schema to LLMJudge's output.
        """

        with open(trace_path, "r") as f:
            raw_trace = f.read()
        trace = self._parse_trace(raw_trace)

        debaters = self._create_debaters()
        moderator = self._create_moderator()

        rounds: list[dict] = []
        current_assessments: list[str] = []

        ## Round 1: each debater evaluates independently
        logger.info("MultiAgentJudge — Round 1: independent evaluations")
        initial_prompt = INITIAL_EVALUATION_PROMPT.format(
            ground_truth=ground_truth, trace=trace
        )
        for debater in debaters:
            debater.add_user_message(initial_prompt)
            current_assessments.append(debater.speak())

        consensus, summary = self._check_consensus(moderator, current_assessments)
        rounds.append({
            "round": 1,
            "assessments": {d.name: a for d, a in zip(debaters, current_assessments)},
            "moderator_summary": summary,
        })

        ## Rounds 2-N: rebuttals until consensus or max_rounds
        for round_num in range(2, self.max_rounds + 1):
            if consensus:
                logger.info(
                    "MultiAgentJudge — Consensus reached after round %d", round_num - 1
                )
                break

            logger.info("MultiAgentJudge — Round %d: rebuttals", round_num)
            new_assessments: list[str] = []

            for i, debater in enumerate(debaters):
                other_text = "\n\n---\n\n".join(
                    f"[{debaters[j].name}]\n{a}"
                    for j, a in enumerate(current_assessments)
                    if j != i
                )
                debater.add_user_message(
                    REBUTTAL_PROMPT.format(other_assessments=other_text)
                )
                new_assessments.append(debater.speak())

            current_assessments = new_assessments
            consensus, summary = self._check_consensus(moderator, current_assessments)
            rounds.append({
                "round": round_num,
                "assessments": {d.name: a for d, a in zip(debaters, current_assessments)},
                "moderator_summary": summary,
            })

        # Final synthesis → JudgeResponse
        logger.info("MultiAgentJudge — Synthesizing final JudgeResponse")
        transcript = self._build_transcript(rounds)
        evaluation: JudgeResponse = self._synthesize(transcript)

        with open(save_path, "w+") as f:
            f.write(evaluation.model_dump_json(indent=2))
        
        ## NEW: debug 
        transcript_path = save_path.replace("llm_judge.json", "debate_transcript.txt")
        with open(transcript_path, "w+") as f:
            f.write(transcript)

        return evaluation