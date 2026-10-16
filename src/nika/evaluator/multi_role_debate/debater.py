"""One participant of the Multi-Role Debate judge.

Output is free text in the discussion rounds and a structured DebaterResponse
in the final round. The orchestrator decides when speak() is called.
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
    """Debater with its own message history, grown by the orchestrator."""

    def __init__(self, llm: BaseChatModel, name: str) -> None:
        self.llm = llm
        self.name = name
        self._messages: list = []
        self._structured_llm: BaseChatModel | None = None
        self.invoke_config: dict | None = None
        # Index of the round-1 task prompt, used by collapse_initial().
        self._initial_idx: int | None = None

    def set_system_prompt(self, prompt: str) -> None:
        """Reset the history to the role prompt."""
        self._messages = [SystemMessage(content=prompt)]
        self._initial_idx = None

    def use_structured_output(self, schema: Type[BaseModel]) -> None:
        """Make the next speak() calls return `schema` (used only in the final round)."""
        self._structured_llm = self.llm.with_structured_output(schema)

    def add_user_message(self, content: str, is_initial: bool = False) -> None:
        """Append a task prompt. `is_initial` marks the round-1 prompt for collapse_initial()."""
        if is_initial:
            self._initial_idx = len(self._messages)
        self._messages.append(HumanMessage(content=content))

    def add_peer_message(
        self, peer_name: str, content: str, template: str | None = None
    ) -> None:
        """Show another debater's statement.

        `template` frames it as a peer's opinion, not as an instruction.
        Without it the statement is prefixed with the peer's name only.
        """
        if template is None:
            body = f"[{peer_name}]\n{content}"
        else:
            body = Template(template).safe_substitute(
                peer_name=peer_name, content=content
            )
        self._messages.append(HumanMessage(content=body))

    def add_assistant_message(self, content: str) -> None:
        """Append the debater's own reply."""
        self._messages.append(AIMessage(content=content))

    def collapse_initial(self, text: str) -> None:
        """Replace the round-1 prompt with a short stub to save tokens.

        The history is resent on every call and the final-round prompt repeats
        the same evidence. No-op if there is no initial prompt. The saved
        transcript is unaffected.
        """
        if self._initial_idx is None:
            return
        self._messages[self._initial_idx] = HumanMessage(content=text)

    def speak(self) -> str:
        """Call the LLM on the history, append the reply to it and return it (JSON if structured)."""
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