
##
## - Implementa `MultiAgentJudge` estendendo `BaseJudge`
## - Contiene tutta la logica del debate (player, round, giudice finale)


"""
MultiAgentJudge for NIKA.

Two debate players with opposing roles (Critic vs Advocate) deliberate over the
agent's action trace. Each player produces a structured DebaterResponse (scores +
reasoning). Consensus is decided purely numerically — the debate continues (up to
max_rounds) until every criterion is within _CONSENSUS_THRESHOLD, with no moderator
LLM.

A final evidence-grounded judge then produces the JudgeResponse. Unlike a plain
CONFIRM/ARBITRATE synthesiser, this judge is never bounded to the debaters'
[min, max] band: their scores are evidence, not a constraint, and the judge may
overturn either or both when the trace supports it (SYNTHESIS_FREE_INSTRUCTION).
The debater order shown to the judge is shuffled once per session and recorded
to ``debate_order.json``, so a positional preference can be told apart from a
preference for one of the two personas. Output schema is identical to LLMJudge.
"""

import json
import logging
import random

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import tracing_context

from nika.evaluator.base_judge import BaseJudge
from nika.evaluator.schemas import DebaterResponse, JudgeResponse
from nika.evaluator.token_meter import dump_cost, meter_config, new_meter


from agent.llm.model_factory import load_model
from agent.utils.template import (
    ADVOCATE_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    INITIAL_EVALUATION_PROMPT,
    REBUTTAL_PROMPT,
    SYNTHESIS_FREE_INSTRUCTION,
    SYNTHESIS_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
)


load_dotenv()

logger = logging.getLogger(__name__)

# Criteria evaluated by debaters (must match Scores field names)
_CRITERIA = ["relevance", "correctness", "efficiency", "clarity", "final_outcome"]

# Score gap threshold above which a criterion is considered divergent
_CONSENSUS_THRESHOLD = 1

# temperatures for the two opposing roles
_CRITIC_TEMPERATURE = 0.2
_ADVOCATE_TEMPERATURE = 0.2


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
        # Optional invoke config (e.g. token-usage callback); set by the judge.
        self.invoke_config: dict | None = None

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
                response = self._structured_llm.invoke(
                    self._messages, config=self.invoke_config
                )
                answer = response.model_dump_json(indent=2)
            else:
                response: AIMessage = self.llm.invoke(
                    self._messages, config=self.invoke_config
                )
                answer = str(response.content)
        self.add_assistant_message(answer)
        logger.debug("[%s]\n%s", self.name, answer)
        return answer


## Multi-agent judge 

class MultiAgentJudge(BaseJudge):
    """

    A Critic (strict) and an Advocate ( lenient)
    independently evaluate the trace, then exchange rebuttals for up to max_rounds.

    Consensus is purely numerical: the debate stops as soon as every criterion is
    within _CONSENSUS_THRESHOLD (no moderator LLM). If max_rounds is reached without
    consensus, the debate stops anyway.

    A final evidence-grounded judge then produces the JudgeResponse. It is never
    bounded to the debaters' [min, max] band — their scores are evidence, not a
    constraint — and it may overturn either or both when the trace supports it.
    The debater order it sees is shuffled once per session and recorded, so a
    positional preference can be told apart from a preference for one of the two
    personas. Output uses structured output, schema identical to LLMJudge.
    """

    _ROUNDS_FILENAME = "debate_rounds.json"
    _TRANSCRIPT_FILENAME = "debate_transcript.txt"
    _COST_FILENAME = "multi_cost.json"
    _ORDER_FILENAME = "debate_order.json"
    _JUDGE_TAG = "multi"

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

        # Drawn once per instance; run_llm_judge() builds a fresh judge for every
        # session, so this is effectively one random order per session.
        self._order: list[int] = random.sample(range(self.num_debaters), self.num_debaters)
        # Debater names in their canonical order; captured during synthesis so
        # evaluate_agent() can record what the judge actually saw.
        self._names_seen: list[str] = []


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

    def _extract_scores(self, assessment_json: str) -> dict[str, int] | None:
        """Parse a DebaterResponse JSON string and return {criterion: score}."""
        try:
            data = json.loads(assessment_json)
            return {c: data["scores"][c]["score"] for c in _CRITERIA}
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


    def _parse_round_scores(self, assessments: list[str]) -> list[dict[str, int]]:
        """Parse every debater's per-criterion scores for a round.

        Fail-fast: if any debater's structured output cannot be parsed, raise.
        The caller aborts this multi evaluation (no llm_judge.json produced) and
        the batch moves on to the next case — better a flagged missing data
        point than a silently degraded one.
        """
        parsed = [self._extract_scores(a) for a in assessments]
        if any(p is None for p in parsed):
            raise ValueError(
                "MultiAgentJudge — could not parse debater scores; aborting multi eval."
            )
        return parsed

    @staticmethod
    def _is_consensus(parsed: list[dict[str, int]]) -> bool:
        """Numerical consensus: every criterion within _CONSENSUS_THRESHOLD.

        Pure code check — no LLM. This is the ONLY arbiter of consensus; the
        debate continues (up to max_rounds) until all criteria are within the
        threshold or the round budget is exhausted.
        """
        for criterion in _CRITERIA:
            scores = [p[criterion] for p in parsed]
            if max(scores) - min(scores) > _CONSENSUS_THRESHOLD:
                return False
        return True

    def _format_debater_votes(
        self, debater_names: list[str], parsed: list[dict[str, int]]
    ) -> str:
        """Render the debaters' final per-criterion scores in the shuffled order.

        Also records which names were shown, so evaluate_agent() can persist
        what the judge actually saw.
        """
        self._names_seen = list(debater_names)
        lines = []
        for idx in self._order:
            if idx >= len(debater_names) or idx >= len(parsed):
                continue
            name, scores = debater_names[idx], parsed[idx]
            crit = ", ".join(f"{c}={scores[c]}" for c in _CRITERIA)
            lines.append(f"{name}: {crit}")
        return "\n".join(lines)

    def _synthesize(
        self,
        ground_truth: str,
        trace: str,
        debate_transcript: str,
        debater_names: list[str],
        final_parsed: list[dict[str, int]],
        consensus: bool,
        invoke_config: dict | None = None,
    ) -> JudgeResponse:
        """Produce the final JudgeResponse as an evidence-grounded, unconstrained judge.

        ``consensus`` is accepted for signature compatibility but deliberately
        ignored: whether the debaters agreed must not change the judge's mandate.
        The judge sees the ground truth, the trace, the shuffled-order transcript
        AND the debaters' explicit final scores, and decides its own verdict —
        it is never bounded to the debaters' [min, max] band.
        """
        debater_votes = self._format_debater_votes(debater_names, final_parsed)
        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(
                content=SYNTHESIS_PROMPT.format(
                    ground_truth=ground_truth,
                    trace=trace,
                    debate_transcript=debate_transcript,
                    debater_votes=debater_votes,
                    mode_instruction=SYNTHESIS_FREE_INSTRUCTION,
                )
            ),
        ]
        with tracing_context(enabled=False):
            result: JudgeResponse = self._synthesis_llm.invoke(
                messages, config=invoke_config
            )
        return result

    @staticmethod
    def _strip_reasoning(assessment_json: str) -> str:
        """Drop the top-level `reasoning` field before exposing an assessment
        to the other debater in a rebuttal round (keeps only per-criterion
        scores/comments — the actual arguable content)."""
        data = json.loads(assessment_json)
        data.pop("reasoning", None)
        return json.dumps(data, indent=2)

    def _build_transcript(self, rounds: list[dict]) -> str:
        """Render the transcript with the debaters in the same shuffled order
        used for the votes, so the judge sees a single consistent ordering."""
        lines = []
        for r in rounds:
            lines.append(f"=== Round {r['round']} ===")
            items = list(r["assessments"].items())
            for idx in self._order:
                if idx >= len(items):
                    continue
                name, text = items[idx]
                lines.append(f"[{name}]\n{text}")
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

        meter = new_meter()
        invoke_config = meter_config(meter)

        debaters = self._create_debaters()
        for d in debaters:
            d.invoke_config = invoke_config
        debater_names = [d.name for d in debaters]

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

        final_parsed = self._parse_round_scores(current_assessments)
        consensus = self._is_consensus(final_parsed)
        rounds.append({
            "round": 1,
            "assessments": {d.name: a for d, a in zip(debaters, current_assessments)},
        })


        # Rounds 2-N: rebuttals until numerical consensus or max_rounds.
        # Consensus is decided purely by the score gap (no moderator LLM).
        for round_num in range(2, self.max_rounds + 1):
            if consensus:
                logger.info(
                    "MultiAgentJudge — Numerical consensus reached after round %d", round_num - 1
                )
                break

            logger.info("MultiAgentJudge — Round %d: rebuttals", round_num)
            new_assessments: list[str] = []

            for i, debater in enumerate(debaters):
                other_text = "\n\n---\n\n".join(
                    self._strip_reasoning(a)
                    for j, a in enumerate(current_assessments)
                    if j != i
                )
                debater.add_user_message(REBUTTAL_PROMPT.format(other_assessments=other_text))
                new_assessments.append(debater.speak())

            current_assessments = new_assessments
            final_parsed = self._parse_round_scores(current_assessments)
            consensus = self._is_consensus(final_parsed)
            rounds.append({
                "round": round_num,
                "assessments": {d.name: a for d, a in zip(debaters, current_assessments)},
            })

        if not consensus:
            logger.info(
                "MultiAgentJudge — No consensus after %d rounds; judge will arbitrate.",
                self.max_rounds,
            )


        # Final judgement — evidence-grounded, unconstrained judge (sees GT + trace + votes).
        logger.info("MultiAgentJudge — Final judgement, debater order %s", self._order)
        transcript = self._build_transcript(rounds)


        ## METRICS: save debate rounds for ** analysis

        rounds_path = save_path.replace("llm_judge.json", self._ROUNDS_FILENAME)
        with open(rounds_path, "w+") as f:
            json.dump(rounds, f, indent=2)


        evaluation: JudgeResponse = self._synthesize(
            ground_truth=ground_truth,
            trace=trace,
            debate_transcript=transcript,
            debater_names=debater_names,
            final_parsed=final_parsed,
            consensus=consensus,
            invoke_config=invoke_config,
        )

        with open(save_path, "w+") as f:
            f.write(evaluation.model_dump_json(indent=2))

        transcript_path = save_path.replace("llm_judge.json", self._TRANSCRIPT_FILENAME)
        with open(transcript_path, "w+") as f:
            f.write(transcript)

        shown = [self._names_seen[i] for i in self._order if i < len(self._names_seen)]
        order_path = save_path.replace("llm_judge.json", self._ORDER_FILENAME)
        with open(order_path, "w+") as f:
            json.dump(
                {
                    "order": self._order,
                    "canonical_names": self._names_seen,
                    "shown_to_judge": shown,
                    "first_shown": shown[0] if shown else None,
                },
                f,
                indent=2,
            )

        dump_cost(
            meter, save_path, judge=self._JUDGE_TAG, filename=self._COST_FILENAME
        )

        return evaluation

