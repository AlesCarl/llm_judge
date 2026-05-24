
##
## Solo l'interfaccia astratta con il metodo evaluate_agent()



from abc import ABC, abstractmethod
import json

# from nika.evaluator.llm_judge import JudgeResponse

from nika.evaluator.schemas import JudgeResponse


class BaseJudge(ABC):

    @abstractmethod
    def evaluate_agent(self, ground_truth: str, trace_path: str, save_path: str) -> JudgeResponse:
        pass


    def _parse_trace(self, trace: str) -> str:
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
##
## Solo l'interfaccia astratta con il metodo evaluate_agent()

