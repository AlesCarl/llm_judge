import os

from dotenv import load_dotenv
from langsmith import tracing_context

from agent.llm.model_factory import load_model
from agent.utils.template import LLM_JUDGE_PROMPT_TEMPLATE
from nika.config import RESULTS_DIR
from nika.evaluator.base_judge import BaseJudge
from nika.evaluator.schemas import JudgeResponse
from nika.evaluator.token_meter import dump_cost, meter_config, new_meter
from nika.orchestrator.problems.prob_pool import get_problem_instance

load_dotenv()


class LLMJudge(BaseJudge):
    """Baseline judge: one LLM call reads ground truth and trace and returns a JudgeResponse."""

    def __init__(self, judge_llm_backend: str = "openai", judge_model: str = "gpt-5-mini"):
        self.llm = load_model(llm_backend=judge_llm_backend, model=judge_model)
        self.llm = self.llm.with_structured_output(JudgeResponse)
        self.prompt = LLM_JUDGE_PROMPT_TEMPLATE

    def evaluate_agent(self, ground_truth: str, trace_path: str, save_path: str = None) -> JudgeResponse:
        with open(trace_path, "r") as f:
            trace = self._parse_trace(f.read())

        self.prompt = self.prompt.format(ground_truth=ground_truth, trace=trace)
        meter = new_meter()
        with tracing_context(enabled=False):
            evaluation: JudgeResponse = self.llm.invoke(self.prompt, config=meter_config(meter))

        if save_path:
            with open(save_path, "w+") as f:
                f.write(evaluation.model_dump_json(indent=2))
            dump_cost(meter, save_path, judge="single", filename="single_cost.json")

        return evaluation


if __name__ == "__main__":
    judge = LLMJudge()
    session_id = "20251113090058"
    root_cause_name = "frr_down_localization"
    eval_model = "gpt-oss:20b"
    problem_instance = get_problem_instance(root_cause_name)
    problem_description = problem_instance.META.description

    trace_file = os.path.join(RESULTS_DIR, root_cause_name, f"{session_id}_{eval_model}_conversation.log")

    evaluation_content = judge.evaluate_agent(
        problem_description,
        trace_file,
        save_path=os.path.join(RESULTS_DIR, root_cause_name, f"{session_id}_{eval_model}_llm_judge.log"),
    )
    print("Evaluation Result:", evaluation_content)
