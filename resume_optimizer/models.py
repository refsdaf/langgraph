from typing import List, Dict, TypedDict, Optional
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    resume: Dict
    job_description: str
    user_feedback: List[str]
    keywords: List[str]
    optimized_resume: Dict
    ai_recommendations: List[str]
    changes: str
    rag_query: str
    evaluation: Optional[Dict]
    final_output: Optional[Dict] # The final JSON output for the user


class Keywords(BaseModel):
    """A list of keywords, skills, and technologies from a job description."""
    keywords: List[str] = Field(
        description="A list of the most important keywords, skills, and technologies found in the job description."
    )


class Evaluation(BaseModel):
    """The evaluation of the optimized resume."""
    is_credible: bool = Field(description="Is the rewritten content believable and not overly exaggerated?")
    is_relevant: bool = Field(description="Does the optimized resume effectively address the job description's requirements?")
    is_grounded: bool = Field(description="Is the optimized resume factually consistent with the original resume?")
    ai_recommendations: List[str] = Field(description="A list of specific, actionable recommendations for improvement if any checks fail.")


class RagDecision(BaseModel):
    """The decision on whether to use RAG or not."""
    decision: str = Field(description="The decision, which must be 'rag' or 'no_rag'.")


class OptimizedResume(BaseModel):
    """The optimized resume and a summary of the changes made."""
    optimized_resume: Dict = Field(description="The full, optimized resume in JSON format.")
    changes: str = Field(description="A detailed summary of the specific changes made to the resume and the reasoning behind them.")
