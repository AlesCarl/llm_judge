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


   

   ### setup

    def set_system_prompt(self, prompt: str) -> None:
        """Initialize the message list with the role/system prompt."""
        self._messages = [SystemMessage(content=prompt)]

    def use_structured_output(self, schema: Type[BaseModel]) -> None:
        """Enable structured-output mode for the *next* speak() calls.

        In multi-role debate this is typically activated only for the
        final round, so debaters output a DebaterResponse JSON.
        """
        self._structured_llm = self.llm.with_structured_output(schema)

    def disable_structured_output(self) -> None:
        """Revert to free-form text generation."""
        self._structured_llm = None




    ### message ops

    def add_user_message(self, content: str) -> None:
        """Append a HumanMessage (e.g. the per-turn task prompt)."""
        self._messages.append(HumanMessage(content=content))

    def add_peer_message(self, peer_name: str, content: str) -> None:
        """Inject another debater's statement as a visible HumanMessage.

        ChatEval's "visibility: all" semantics: every debater sees the
        full chain of statements from the panel. 
        """
        self._messages.append(
            HumanMessage(content=f"[{peer_name}]\n{content}")
        )

    def add_assistant_message(self, content: str) -> None:
        """Append an AIMessage (the debater's own reply)."""
        self._messages.append(AIMessage(content=content))




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
                parsed: BaseModel = self._structured_llm.invoke(self._messages)
                answer = parsed.model_dump_json(indent=2)
            else:
                response: AIMessage = self.llm.invoke(self._messages)
                answer = str(response.content)

        self.add_assistant_message(answer)
        logger.debug("[%s]\n%s", self.name, answer)
        return answer



    ### introspection

    @property
    def transcript(self) -> list:
        """Return a shallow copy of this debater's full message list."""
        return list(self._messages)