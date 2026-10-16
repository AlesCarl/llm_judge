import json
from abc import ABC, abstractmethod

from nika.evaluator.schemas import JudgeResponse


class BaseJudge(ABC):
    """Common interface of all judges."""

    @abstractmethod
    def evaluate_agent(self, ground_truth: str, trace_path: str, save_path: str) -> JudgeResponse:
        pass

    def _parse_trace(self, trace: str) -> str:
        """Turn the raw messages.jsonl into a compact JSON string for the prompt."""
        new_trace = []
        for line in trace.splitlines():
            line = json.loads(line)
            if "event" in line:
                if line["event"] == "llm_start":
                    new_trace.append({
                        "timestamp": line.get("timestamp", ""),
                        "event": "LLM Prompt",
                        "payload": line.get("prompts", ""),
                    })
                elif line["event"] == "llm_end":
                    new_trace.append({
                        "timestamp": line.get("timestamp", ""),
                        "event": "LLM Response",
                        "payload": line.get("text", ""),
                    })
                else:
                    new_trace.append(line)
        return json.dumps(new_trace, ensure_ascii=False)
