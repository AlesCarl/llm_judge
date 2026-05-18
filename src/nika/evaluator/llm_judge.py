import json
import os

from dotenv import load_dotenv
from langsmith import tracing_context
from pydantic import BaseModel, Field

# from agent.llm.langchain_deepseek import DeepSeekLLM
from agent.llm.model_factory import load_model
from agent.utils.template import LLM_JUDGE_PROMPT_TEMPLATE
from nika.config import RESULTS_DIR
from nika.orchestrator.problems.prob_pool import get_problem_instance

from nika.evaluator.base_judge import BaseJudge
from nika.evaluator.schemas import JudgeResponse, Score, Scores


load_dotenv()


''' 
class Score(BaseModel):
    score: int = Field(..., ge=1, le=5, description="Score from 1 to 5.")
    comment: str = Field(..., description="Comment explaining the rationale for the score.")


class Scores(BaseModel):
    relevance: Score = Field(..., description="How relevant the agent's actions were to the problem.")
    correctness: Score = Field(..., description="How correct the tools/commands and actions were.")
    efficiency: Score = Field(..., description="How efficient and well-ordered the agent’s actions were.")
    clarity: Score = Field(..., description="How clear and well-explained the agent’s reasoning was.")
    final_outcome: Score = Field(..., description="Whether the final outcome existed and matched the ground truth.")
    overall_score: Score = Field(..., description="Overall final score summarizing the total performance.")



class JudgeResponse(BaseModel):
    scores: Scores = Field(..., description="Per-criterion scores and evaluator comments.")
    overall_evaluation: str = Field(..., description="High-level summary of strengths and weaknesses.")
    reasoning_for_overall_score: str = Field(..., description="Explanation of why this overall score was given.")
    eval_time: float = 0.0   ##

'''


class LLMJudge(BaseJudge):

    def __init__(self, judge_llm_backend: str = "openai", judge_model: str = "gpt-5-mini"):

        self.llm = load_model(llm_backend=judge_llm_backend, model=judge_model)
        

        self.llm = self.llm.with_structured_output(JudgeResponse) #
        self.prompt = LLM_JUDGE_PROMPT_TEMPLATE




    def evaluate_agent(self, ground_truth: str, trace_path: str, save_path: str) -> str:
        """Evaluate the agent's performance based on the problem description, network environment info, and action history.

        ### Un singolo LLM legge trace+ground_truth e produce direttamente il JudgeResponse

        Args:
            problem_description: Description of the problem.
            net_env_info: Information about the network environment.
            trace_path: Path to the file containing the agent's action history.
            save_path: Path to save the evaluation result.

        Returns:
            str: The evaluation result from the judge model.
            int: The score extracted from the evaluation result.
        """
        with open(trace_path, "r") as f:
            trace = f.read()
        trace = self._parse_trace(trace)

        self.prompt = self.prompt.format(
            ground_truth=ground_truth,
            trace=trace,
        )
        with tracing_context(enabled=False):
            evaluation: JudgeResponse = self.llm.invoke(self.prompt)

        # Save evaluation result to file
        if save_path:
          with open(save_path, "w+") as f:
            f.write(evaluation.model_dump_json(indent=2))


        return evaluation


if __name__ == "__main__":
    judge = LLMJudge()
    session_id = "20251113090058"
    root_cause_name = "frr_down_localization"
    eval_model = "gpt-oss:20b"
    problem_instance = get_problem_instance(root_cause_name)
    problem_description = problem_instance.META.description
    net_env_info = problem_instance.net_env.get_info()

    trace_file = os.path.join(RESULTS_DIR, root_cause_name, f"{session_id}_{eval_model}_conversation.log")

    evaluation_content, score = judge.evaluate_agent(
        problem_description,
        net_env_info,
        trace_file,
        save_path=os.path.join(RESULTS_DIR, root_cause_name, f"{session_id}_{eval_model}_llm_judge.log"),
    )
    print("Evaluation Result:", evaluation_content)
    print("Evaluation Score:", score)
