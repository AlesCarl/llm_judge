"""Message summarization utility for the ReAct agent.

Called via pre_model_hook before each LLM call. When the conversation
history exceeds `char_threshold` characters, compresses the older messages
into a concise summary, keeping only the most recent `keep_last` messages
intact.

This prevents Ollama context-window overflows during long agent runs.
"""

from __future__ import annotations

import logging
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM = (
    "You are a concise assistant. Summarize the following network "
    "troubleshooting conversation into 3-5 sentences. Focus on: "
    "what was investigated, what tools were called, key findings, "
    "and conclusions reached so far."
)


def _total_chars(messages: list[BaseMessage]) -> int:
    return sum(len(str(m.content)) for m in messages)


def needs_summarization(messages: list[BaseMessage], char_threshold: int) -> bool:
    return _total_chars(messages) > char_threshold


def summarize_messages(
    messages: list[BaseMessage],
    llm,
    keep_last: int = 6,
    char_threshold: int = 30_000,
) -> list[BaseMessage]:
    """Compress older messages into a summary, keep recent `keep_last` messages.

    Message layout after compression:
        [SystemMessage (original, if present)]
        [HumanMessage  (summary of old history)]
        [last `keep_last` messages]
    """
    if not needs_summarization(messages, char_threshold):
        return messages

    # Split system vs non-system
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system  = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) <= keep_last:
        return messages

    to_compress = non_system[:-keep_last]
    recent      = non_system[-keep_last:]

    # Format old messages for summarization (truncate each to avoid recursion)
    formatted = "\n".join(
        f"[{type(m).__name__}]: {str(m.content)[:400]}"
        for m in to_compress
    )

    summary_response = llm.invoke([
        SystemMessage(content=_SUMMARIZE_SYSTEM),
        HumanMessage(content=f"Conversation to summarize:\n\n{formatted}"),
    ])
    summary_text = str(summary_response.content)

    logger.info(
        "Summarizer: compressed %d messages → summary (%d chars). Keeping last %d.",
        len(to_compress), len(summary_text), keep_last,
    )

    summary_msg = HumanMessage(
        content=f"[PREVIOUS CONVERSATION SUMMARY]\n{summary_text}\n[END SUMMARY]"
    )

    return system_msgs + [summary_msg] + recent