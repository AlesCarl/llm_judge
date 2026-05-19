

from pydantic import BaseModel, Field


# QUI METTO JudgeResponse



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
    eval_time: float = 0.0


class DebaterResponse(BaseModel):
    """Output produced by each debate participant (critic & advocate)"""
    scores: Scores = Field(..., description="Per-criterion scores with justification.")
    reasoning: str = Field(..., description="Overall reasoning summary supporting the scores.")
    # JudgeResponse invece rimane l'output finale

