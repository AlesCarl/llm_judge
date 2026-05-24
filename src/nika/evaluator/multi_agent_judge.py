
##
## - Implementa `MultiAgentJudge` estendendo `BaseJudge`
## - Contiene tutta la logica del debate (player, round, moderator)


"""
MultiAgentJudge for NIKA.

Two debate players with opposing roles (Critic vs Advocate) deliberate over the
agent's action trace. Each player produces a structured DebaterResponse (scores +
reasoning). A hybrid consensus check first compares scores numerically; only when
scores diverge beyond the threshold does the moderator LLM check reasoning alignment.
A final synthesizer produces a JudgeResponse identical in schema to LLMJudge.

"""

import json
import logging

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import tracing_context

from nika.evaluator.base_judge import BaseJudge
from nika.evaluator.schemas import DebaterResponse, JudgeResponse


from agent.llm.model_factory import load_model
from agent.utils.template import (
    ADVOCATE_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    INITIAL_EVALUATION_PROMPT,
    MODERATOR_PROMPT,
    MODERATOR_SYSTEM_PROMPT,
    REBUTTAL_PROMPT,
    SYNTHESIS_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)


load_dotenv()

logger = logging.getLogger(__name__)

# Criteria evaluated by debaters (must match Scores field names)
_CRITERIA = ["relevance", "correctness", "efficiency", "clarity", "final_outcome", "overall_score"]

# Score gap threshold above which a criterion is considered divergent
_CONSENSUS_THRESHOLD = 1

# temperatures for the two opposing roles
_CRITIC_TEMPERATURE = 0.2
_ADVOCATE_TEMPERATURE = 0.6


##  Debate player 

class DebatePlayer:
    """A single debate participant backed by a LangChain chat model.

    When use_structured_output() is called, speak() invokes the structured LLM
    and returns the JSON-serialised DebaterResponse (stored in message history
    so subsequent rounds have context). Otherwise it returns plain text.
    """

    def __init__(self, llm: BaseChatModel, name: str) -> None:
        self.llm = llm
        self.name = name
        self._messages: list = []
        self._structured_llm: BaseChatModel | None = None

    def use_structured_output(self, schema) -> None:
        self._structured_llm = self.llm.with_structured_output(schema)

    def set_system_prompt(self, prompt: str) -> None:
        self._messages = [SystemMessage(content=prompt)]

    def add_user_message(self, content: str) -> None:
        self._messages.append(HumanMessage(content=content))

    def add_assistant_message(self, content: str) -> None:
        self._messages.append(AIMessage(content=content))

    def speak(self) -> str:
        with tracing_context(enabled=False):
            if self._structured_llm is not None:
                response = self._structured_llm.invoke(self._messages)
                answer = response.model_dump_json(indent=2)
            else:
                response: AIMessage = self.llm.invoke(self._messages)
                answer = str(response.content)
        self.add_assistant_message(answer)
        logger.debug("[%s]\n%s", self.name, answer)
        return answer


## Multi-agent judge 

class MultiAgentJudge(BaseJudge):
    """

    A Critic (low temperature, strict) and an Advocate (higher temperature, lenient)
    independently evaluate the trace, then exchange rebuttals for up to max_rounds.
    
    Consensus is checked with a hybrid strategy:
      1. Programmatic: if all scores are within _CONSENSUS_THRESHOLD → consensus.
      2. LLM moderator: only called when scores diverge, to check reasoning alignment.
    
    A final synthesis step produces a JudgeResponse using structured output.
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
            judge_llm_backend: Backend name ("openai", "ollama", "deepseek").
            judge_model: Model identifier for the chosen backend.
            num_debaters: Kept for API compatibility; only Critic+Advocate are used.
            max_rounds: Maximum debate rounds before forcing synthesis.
        """
        self.judge_llm_backend = judge_llm_backend
        self.judge_model = judge_model
        self.num_debaters = num_debaters
        self.max_rounds = max_rounds

        # Structured LLM for the final synthesis step
        self._synthesis_llm: BaseChatModel = load_model(
            llm_backend=judge_llm_backend, model=judge_model
        ).with_structured_output(JudgeResponse)


    #  helpers

    def _create_debaters(self) -> list[DebatePlayer]:
        """Create Critic (strict, low temp) and Advocate (lenient, higher temp)."""
        
        critic_llm = load_model(
            llm_backend=self.judge_llm_backend,
            model=self.judge_model,
            temperature=_CRITIC_TEMPERATURE,
        )
        advocate_llm = load_model(
            llm_backend=self.judge_llm_backend,
            model=self.judge_model,
            temperature=_ADVOCATE_TEMPERATURE,
        )

        critic = DebatePlayer(llm=critic_llm, name="Critic")
        critic.set_system_prompt(CRITIC_SYSTEM_PROMPT)
        critic.use_structured_output(DebaterResponse)

        advocate = DebatePlayer(llm=advocate_llm, name="Advocate")
        advocate.set_system_prompt(ADVOCATE_SYSTEM_PROMPT)
        advocate.use_structured_output(DebaterResponse)

        return [critic, advocate]

    def _create_moderator(self) -> DebatePlayer:
        moderator = DebatePlayer(
            llm=load_model(llm_backend=self.judge_llm_backend, model=self.judge_model),
            name="Moderator",
        )
        moderator.set_system_prompt(MODERATOR_SYSTEM_PROMPT)
        return moderator

    def _extract_scores(self, assessment_json: str) -> dict[str, int] | None:
        """Parse a DebaterResponse JSON string and return {criterion: score}."""
        try:
            data = json.loads(assessment_json)
            return {c: data["scores"][c]["score"] for c in _CRITERIA}
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


    def _check_consensus(
        self,
        moderator: DebatePlayer,
        debaters: list[DebatePlayer],
        assessments: list[str],
    ) -> tuple[bool, str]:
        
        """Hybrid consensus check.

        Step 1 — Programmatic: parse scores and check all criteria are within
        _CONSENSUS_THRESHOLD. If so, return True immediately (no LLM call).

        Step 2 — LLM moderator: called only for divergent criteria to check
        whether the reasoning is fundamentally aligned.

        Returns:
            (consensus: bool, summary: str)
        """
        parsed = [self._extract_scores(a) for a in assessments]

        if all(p is not None for p in parsed):
            divergent_lines = []
            for criterion in _CRITERIA:
                scores = [p[criterion] for p in parsed]
                if max(scores) - min(scores) > _CONSENSUS_THRESHOLD:
                    scores_info = ", ".join(
                        f"{d.name}={s}" for d, s in zip(debaters, scores)
                    )
                    divergent_lines.append(f"- {criterion}: {scores_info}")

            if not divergent_lines:
                logger.info("MultiAgentJudge — Programmatic consensus: all scores within threshold.")
                return True, "Numerical consensus: all scores within threshold."

            divergent_criteria = "\n".join(divergent_lines)
            logger.info(
                "MultiAgentJudge — Score divergence detected; consulting moderator.\n%s",
                divergent_criteria,
            )
        else:
            # Fallback: could not parse structured scores
            logger.warning("MultiAgentJudge — Could not parse debater scores; consulting moderator.")
            divergent_criteria = "(Could not parse structured scores — inspect full assessments below)"

        # LLM moderator checks reasoning alignment for divergent criteria
        joined = "\n\n---\n\n".join(
            f"[{d.name}]\n{a}" for d, a in zip(debaters, assessments)
        )
        moderator.add_user_message(
            MODERATOR_PROMPT.format(divergent_criteria=divergent_criteria, assessments=joined)
        )
        raw = moderator.speak()

        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            parsed_mod = json.loads(clean)
            return bool(parsed_mod.get("consensus", False)), parsed_mod.get("summary", "")
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


    ## main 

    def evaluate_agent(
        self, ground_truth: str, trace_path: str, save_path: str
    ) -> JudgeResponse:
        """Evaluate the agent through Critic vs Advocate debate.

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


        # Round 1: Critic and Advocate evaluate independently
        logger.info("MultiAgentJudge — Round 1: independent evaluations (Critic vs Advocate)")
        initial_prompt = INITIAL_EVALUATION_PROMPT.format(
            ground_truth=ground_truth, trace=trace
        )
        for debater in debaters:
            debater.add_user_message(initial_prompt)
            current_assessments.append(debater.speak())

        consensus, summary = self._check_consensus(moderator, debaters, current_assessments)
        rounds.append({
            "round": 1,
            "assessments": {d.name: a for d, a in zip(debaters, current_assessments)},
            "moderator_summary": summary,
        })



        # Rounds 2-N: rebuttals until consensus or max_rounds
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
                debater.add_user_message(REBUTTAL_PROMPT.format(other_assessments=other_text))
                new_assessments.append(debater.speak())

            current_assessments = new_assessments
            consensus, summary = self._check_consensus(moderator, debaters, current_assessments)
            rounds.append({
                "round": round_num,
                "assessments": {d.name: a for d, a in zip(debaters, current_assessments)},
                "moderator_summary": summary,
            })
            

        # Final synthesis - JudgeResponse
        logger.info("MultiAgentJudge — Synthesizing final JudgeResponse")
        transcript = self._build_transcript(rounds)
        evaluation: JudgeResponse = self._synthesize(transcript)

        with open(save_path, "w+") as f:
            f.write(evaluation.model_dump_json(indent=2))

        transcript_path = save_path.replace("llm_judge.json", "debate_transcript.txt")
        with open(transcript_path, "w+") as f:
            f.write(transcript)

        return evaluation

