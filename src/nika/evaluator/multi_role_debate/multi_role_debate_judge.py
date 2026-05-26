"""Multi-Role Debate orchestrator .

Drives N RoleDebater participants through sequential rounds of free
discussion followed by a final structured-scoring round. Returns the
full transcript and the parsed DebaterResponse from each debater.

"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import ValidationError

from agent.llm.model_factory import load_model

from nika.evaluator.base_judge import BaseJudge
from nika.evaluator.schemas import DebaterResponse, JudgeResponse
from nika.evaluator.multi_role_debate.debater import RoleDebater

from nika.evaluator.multi_role_debate.aggregator import aggregate_responses
from nika.evaluator.multi_role_debate.roles_config import (
    DEFAULT_DEBATE_CONFIG,
    DebateConfig,
    RoleConfig,
)




logger = logging.getLogger(__name__)



_CONTINUATION_PROMPT = """\
The other referees have spoken (their statements are visible above).
Consider their points: refine, defend, or update your position based on
the evidence in the trace. Keep it short and focused, ${agent_name}.

${final_prompt}
"""


class MultiRoleDebateJudge(BaseJudge):
    """Multi-role debate judge for NIKA.

    Workflow:
      1. Build N RoleDebater players from a DebateConfig.
      2. Each debater is primed with its role_description (system
         prompt). Round 1 receives the full task prompt (ground truth
         + trace); later rounds receive a short continuation prompt.
      3. Run num_rounds rounds sequentially. Within a round, debaters
         speak in config order; each sees the previous statements
         (current round so far + all prior rounds) as visible context.
      4. In the final round, structured output is activated so every
         debater emits a DebaterResponse JSON with per-criterion scores.
      5. run_debate() returns (transcript, [DebaterResponse|None,...]).
      6. evaluate_agent() runs run_debate, saves artefacts, and applies
         the aggregator.
    """

    def __init__(
        self,
        judge_llm_backend: str = "openai",
        judge_model: str = "gpt-4o-mini",
        debate_config: DebateConfig | None = None,
    ) -> None:
        
        """Args:
            judge_llm_backend: LLM provider ("openai", "ollama", "deepseek").
            judge_model: Default model id (overridable per role).
            debate_config: Roles + debate parameters. Defaults to
                DEFAULT_DEBATE_CONFIG.
        """
        self.judge_llm_backend = judge_llm_backend
        self.judge_model = judge_model
        self.config = debate_config or DEFAULT_DEBATE_CONFIG


    ### setup

    def _build_debater(self, role: RoleConfig) -> RoleDebater:
        """Instantiate one RoleDebater from its RoleConfig."""
        llm: BaseChatModel = load_model(
            llm_backend=self.judge_llm_backend,
            model=role.model or self.judge_model,
            temperature=role.temperature,
        )
        debater = RoleDebater(llm=llm, name=role.name)
        debater.set_system_prompt(role.role_description)
        return debater
    

    ### prompts

    def _initial_user_prompt(
        self, role: RoleConfig, ground_truth: str, trace: str
    ) -> str:
        """Build the round-1 user message (full template, empty final_prompt)."""
        return Template(self.config.prompt_template).safe_substitute(
            ground_truth=ground_truth,
            trace=trace,
            role_description=role.role_description,
            agent_name=role.name,
            final_prompt="",
        )

    def _continuation_prompt(self, role: RoleConfig, is_final: bool) -> str:
        """Build the per-turn message for rounds 2..N (short)."""
        final_prompt = ""
        if is_final:
            final_prompt = role.final_prompt or self.config.final_prompt
        return Template(_CONTINUATION_PROMPT).safe_substitute(
            agent_name=role.name,
            final_prompt=final_prompt,
        )
    



    ### debate ### 

    def run_debate(
        self, ground_truth: str, trace: str
    ) -> tuple[str, list[DebaterResponse | None]]:
        """Run the full multi-role debate and return its outcome.

        Returns:
            (transcript, final_responses)
              transcript: human-readable text of all rounds.
              final_responses: list aligned with self.config.roles.
                  Each entry is the parsed DebaterResponse from the
                  final round, or None if parsing failed.
        """
        roles = self.config.roles
        num_rounds = self.config.num_rounds
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        if not roles:
            raise ValueError("DebateConfig.roles is empty")

        debaters = [self._build_debater(r) for r in roles]
        # statements[round_idx][role_idx] = raw reply (text or JSON string)
        statements: list[list[str]] = []

        for round_idx in range(num_rounds):
            is_final = round_idx == num_rounds - 1
            logger.info(
                "MultiRoleDebateJudge — Round %d/%d%s",
                round_idx + 1,
                num_rounds,
                " (FINAL, structured scoring)" if is_final else "",
            )
            round_statements: list[str] = []



            for i, (role, debater) in enumerate(zip(roles, debaters)):
                # 1. Inject peer statements not yet seen by this debater.
                self._inject_pending_peers(
                    debater=debater,
                    roles=roles,
                    statements=statements,
                    round_statements=round_statements,
                    self_idx=i,
                    round_idx=round_idx,
                )


                # 2. Add the per-turn task prompt.
                if round_idx == 0:
                    user_msg = self._initial_user_prompt(role, ground_truth, trace)
                else:
                    user_msg = self._continuation_prompt(role, is_final=is_final)
                debater.add_user_message(user_msg)


                # 3. Switch to structured output for the final round.
                if is_final:
                    debater.use_structured_output(DebaterResponse)


                # 4. Speak — wrapped so a single debater failure doesn't
                try:
                    reply = debater.speak()
                except Exception as e:
                    reply = f"[ERROR — {type(e).__name__}: {e}]"
                    logger.error(
                        "  [%s] speak() failed (round %d): %s",
                        role.name, round_idx + 1, e,
                    )

                round_statements.append(reply)

                #_sep = "─" * 60
                #print(f"\n{_sep}")
                #print(f"  Round {round_idx + 1}/{num_rounds}  │  {role.name}")
                #print(_sep)
                #print(reply)

            statements.append(round_statements)


            

        transcript = self._build_transcript(roles, statements)
        final_responses = self._parse_final_responses(statements[-1])
        return transcript, final_responses


    ### peer feed

    @staticmethod
    def _inject_pending_peers(
        debater: RoleDebater,
        roles: list[RoleConfig],
        statements: list[list[str]],
        round_statements: list[str],
        self_idx: int,
        round_idx: int,
    ) -> None:
        """Append every peer statement that's new to `debater`.

        - Round 0: peers with index < self_idx in the current round.
        - Round r>0: peers with index >= self_idx from round r-1 (they
          spoke AFTER this debater's last turn in round r-1), then
          peers with index < self_idx from the current round.
        """
        if round_idx > 0:
            prev_round = statements[round_idx - 1]
            for j in range(self_idx, len(roles)):
                debater.add_peer_message(roles[j].name, prev_round[j])

        for j in range(self_idx):
            debater.add_peer_message(roles[j].name, round_statements[j])


    ### transcript

    @staticmethod
    def _build_transcript(
        roles: list[RoleConfig], statements: list[list[str]]
    ) -> str:
        lines: list[str] = []
        for r_idx, round_statements in enumerate(statements):
            lines.append(f"=== Round {r_idx + 1} ===")
            for role, text in zip(roles, round_statements):
                lines.append(f"[{role.name}]\n{text}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _parse_final_responses(
        final_round: list[str],
    ) -> list[DebaterResponse | None]:
        """Parse each final-round reply (expected DebaterResponse JSON).

        Tolerant: returns None for entries that don't parse so the
        aggregator can skip / warn instead of crashing the whole eval.
        """
        parsed: list[DebaterResponse | None] = []
        for text in final_round:
            try:
                parsed.append(DebaterResponse.model_validate_json(text))
            except (ValidationError, ValueError) as e:
                logger.warning("Failed to parse DebaterResponse: %s", e)
                parsed.append(None)
        return parsed


    ### API

    def evaluate_agent(
        self, ground_truth: str, trace_path: str, save_path: str
    ) -> JudgeResponse:
        """Full BaseJudge entry point.

        Runs the debate, persists transcript + per-debater responses,
        then aggregates into a JudgeResponse (placeholder in Step 3).
        """
        with open(trace_path, "r") as f:
            raw_trace = f.read()
        trace = self._parse_trace(raw_trace)

        transcript, final_responses = self.run_debate(ground_truth, trace)

        # Persist artefacts (transcript + per-debater raw responses).
        save = Path(save_path)
        transcript_path = save.with_name("debate_transcript.txt")
        responses_path = save.with_name("debate_responses.json")
        transcript_path.write_text(transcript, encoding="utf-8")
        responses_path.write_text(
            json.dumps(
                [r.model_dump() if r else None for r in final_responses],
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Saved transcript → %s", transcript_path)
        logger.info("Saved debater responses → %s", responses_path)

       # Aggregate per-debater scores into a single JudgeResponse
       # numeric averaging across the panel
        role_names = [r.name for r in self.config.roles]
        evaluation = aggregate_responses(final_responses, role_names)

        with open(save_path, "w+") as f:
            f.write(evaluation.model_dump_json(indent=2))

        return evaluation