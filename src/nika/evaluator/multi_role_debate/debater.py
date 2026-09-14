"""Single debate participant for the Multi-Role Debate judge.

Every turn the debater receives the
shared conversation (its own past statements + other debaters' past
statements as visible context) and produces a new statement.

Two output modes:
  - free-form text (debate rounds): used for discussion / rebuttals
  - structured DebaterResponse JSON (final round): forced by switching
    to the structured LLM right before the call

The debater is "dumb" about turn ordering: the orchestrator drives when speak() is called 
and what user prompt is injected.
"""

from __future__ import annotations

import logging
from string import Template
from typing import Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import tracing_context
from pydantic import BaseModel


logger = logging.getLogger(__name__)



class RoleDebater:
    """A single role-based debate participant.

    Stateful: holds its own chronological message list. The orchestrator
    grows that list by calling:
      - set_system_prompt(...) once at init
      - add_user_message(...) for the per-turn task prompt
      - add_peer_message(...) for statements from other debaters
      - speak() to invoke the LLM and append the assistant reply
    """

    def __init__(self, llm: BaseChatModel, name: str) -> None:
        self.llm = llm
        self.name = name
        self._messages: list = []
        self._structured_llm: BaseChatModel | None = None
        # Optional invoke config (e.g. token-usage callback); set by the judge.
        self.invoke_config: dict | None = None
        # Position of the round-1 task prompt in _messages. Set by
        # add_user_message(is_initial=True), used by collapse_initial().
        self._initial_idx: int | None = None




   ### setup

    def set_system_prompt(self, prompt: str) -> None:
        """Initialize the message list with the role/system prompt."""
        self._messages = [SystemMessage(content=prompt)]
        # The history is rebuilt from scratch, so any recorded position into
        # the old one is stale.
        self._initial_idx = None

    def use_structured_output(self, schema: Type[BaseModel]) -> None:
        """Enable structured-output mode for the *next* speak() calls.

        In multi-role debate this is typically activated only for the
        final round, so debaters output a DebaterResponse JSON.
        """
        self._structured_llm = self.llm.with_structured_output(schema)




    ### message ops

    def add_user_message(self, content: str, is_initial: bool = False) -> None:
        """Append a HumanMessage (e.g. the per-turn task prompt).

        `is_initial` marks the round-1 task prompt so collapse_initial() can
        find it later. Its position is not fixed: peer statements are injected
        before it, so it lands at a different index for each debater.
        """
        if is_initial:
            self._initial_idx = len(self._messages)
        self._messages.append(HumanMessage(content=content))

    def add_peer_message(
        self, peer_name: str, content: str, template: str | None = None
    ) -> None:
        """Inject another debater's statement as a visible HumanMessage.

        ChatEval's "visibility: all" semantics: every debater sees the
        full chain of statements from the panel.

        The statement travels on the same channel as the task prompt, so
        `template` (DebateConfig.peer_message_template) frames it as a peer's
        opinion to be checked against the trace rather than as an instruction.
        Without a template the statement is injected bare, prefixed with the
        peer's name — the original behaviour.
        """
        if template is None:
            body = f"[{peer_name}]\n{content}"
        else:
            body = Template(template).safe_substitute(
                peer_name=peer_name, content=content
            )
        self._messages.append(HumanMessage(content=body))

    def add_assistant_message(self, content: str) -> None:
        """Append an AIMessage (the debater's own reply)."""
        self._messages.append(AIMessage(content=content))

    def collapse_initial(self, text: str) -> None:
        """Replace the round-1 task prompt with a short stub, in place.

        The round-1 prompt carries ground truth, trace and rubric, and the
        whole history is resent on every call, so that block is paid for on
        every turn. When the final-round prompt re-presents the same evidence
        there is no reason to keep the older copy. No-op when no initial
        prompt is registered (num_rounds == 1, where it is the only copy).

        Destructive: the original text is not kept, so after this call the
        message list no longer reflects the history as it was first built.
        The saved debate transcript is unaffected — the judge builds it from
        the statements, not from this list.
        """
        if self._initial_idx is None:
            return
        self._messages[self._initial_idx] = HumanMessage(content=text)




    ### main

    def speak(self) -> str:
        """Invoke the *LLM* on the current message list and return the reply.

        If structured-output mode is active (use_structured_output was
        called) the reply is the JSON serialization of the parsed
        pydantic object; otherwise it's the raw text.

        The reply is appended to the message list as an AIMessage so the
        next turn has it in context.
        """
        with tracing_context(enabled=False):
            if self._structured_llm is not None:
                parsed: BaseModel = self._structured_llm.invoke(
                    self._messages, config=self.invoke_config
                )
                answer = parsed.model_dump_json(indent=2)
            else:
                response: AIMessage = self.llm.invoke(
                    self._messages, config=self.invoke_config
                )
                answer = str(response.content)

        self.add_assistant_message(answer)
        logger.debug("[%s]\n%s", self.name, answer)
        return answer