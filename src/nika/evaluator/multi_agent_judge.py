"""MultiAgentJudge: Critic vs Advocate debate, then a final unconstrained judge.

Both debaters score the trace independently (round 1), then exchange rebuttals
until every criterion is within _CONSENSUS_THRESHOLD or max_rounds is reached.
Consensus is a numeric check, not an LLM call.

The final judge sees ground truth, trace, transcript and the debaters' scores,
and may overturn both. The order in which the debaters are shown is shuffled
once per session and saved to debate_order.json, to separate positional bias
from persona bias.
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

# Must match the Scores field names.
_CRITERIA = ["relevance", "correctness", "efficiency", "clarity", "final_outcome"]

# Largest score gap still counted as agreement.
_CONSENSUS_THRESHOLD = 1

_CRITIC_TEMPERATURE = 0.2
_ADVOCATE_TEMPERATURE = 0.2


class DebatePlayer:
    """One debater: a chat model plus its own message history.

    With structured output enabled, speak() returns the DebaterResponse as JSON.
    """

    def __init__(self, llm: BaseChatModel, name: str) -> None:
        self.llm = llm
        self.name = name
        self._messages: list = []
        self._structured_llm: BaseChatModel | None = None
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


class MultiAgentJudge(BaseJudge):
    """Critic (strict) vs Advocate (lenient) debate with a final evidence-grounded judge."""

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
        """num_debaters is kept for API compatibility; only Critic and Advocate are used."""
        self.judge_llm_backend = judge_llm_backend
        self.judge_model = judge_model
        self.num_debaters = num_debaters
        self.max_rounds = max_rounds

        self._synthesis_llm: BaseChatModel = load_model(
            llm_backend=judge_llm_backend, model=judge_model
        ).with_structured_output(JudgeResponse)

        # run_llm_judge() builds a new judge per session: one random order per session.
        self._order: list[int] = random.sample(range(self.num_debaters), self.num_debaters)
        # Filled during synthesis, saved by evaluate_agent().
        self._names_seen: list[str] = []


    def _create_debaters(self) -> list[DebatePlayer]:
        """Create the Critic and the Advocate, both with structured output."""
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
        """Return {criterion: score} from a DebaterResponse JSON, or None if unparseable."""
        try:
            data = json.loads(assessment_json)
            return {c: data["scores"][c]["score"] for c in _CRITERIA}
        except (json.JSONDecodeError, KeyError, TypeError):
            return None


    def _parse_round_scores(self, assessments: list[str]) -> list[dict[str, int]]:
        """Parse every debater's scores; raise if any is unparseable.

        Aborting the evaluation is intended: a missing data point is better than a degraded one.
        """
        parsed = [self._extract_scores(a) for a in assessments]
        if any(p is None for p in parsed):
            raise ValueError(
                "MultiAgentJudge — could not parse debater scores; aborting multi eval."
            )
        return parsed

    @staticmethod
    def _is_consensus(parsed: list[dict[str, int]]) -> bool:
        """True if every criterion is within _CONSENSUS_THRESHOLD. Pure code, no LLM."""
        for criterion in _CRITERIA:
            scores = [p[criterion] for p in parsed]
            if max(scores) - min(scores) > _CONSENSUS_THRESHOLD:
                return False
        return True

    def _format_debater_votes(
        self, debater_names: list[str], parsed: list[dict[str, int]]
    ) -> str:
        """Render the final scores in the shuffled order and remember the names shown."""
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
        """Final verdict, not bounded to the debaters' score range.

        ``consensus`` is ignored on purpose: agreement must not change the judge's mandate.
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
        """Drop the top-level `reasoning` before showing an assessment to the other debater."""
        data = json.loads(assessment_json)
        data.pop("reasoning", None)
        return json.dumps(data, indent=2)

    def _build_transcript(self, rounds: list[dict]) -> str:
        """Render the transcript in the same shuffled order used for the votes."""
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

    def evaluate_agent(
        self, ground_truth: str, trace_path: str, save_path: str
    ) -> JudgeResponse:
        """Run the debate and save the JudgeResponse plus rounds, transcript, order and cost."""
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

        # Round 1: independent evaluations
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


        # Rounds 2..N: rebuttals until consensus or max_rounds.
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


        # Final judgement
        logger.info("MultiAgentJudge — Final judgement, debater order %s", self._order)
        transcript = self._build_transcript(rounds)

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

