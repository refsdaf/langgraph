from typing import List, Dict, TypedDict, Optional

# Pydantic models for structured output
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


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


def process_inputs(state: GraphState) -> Dict:
    """
    Processes the initial inputs and initializes the state.
    """
    state['user_feedback'] = state.get('user_feedback', [])
    state['keywords'] = state.get('keywords', [])
    state['optimized_resume'] = state.get('optimized_resume', {})
    state['ai_recommendations'] = state.get('ai_recommendations', [])
    state['changes'] = state.get('changes', "")
    state['rag_query'] = state.get('rag_query', "")
    state['evaluation'] = state.get('evaluation', None)
    state['final_output'] = state.get('final_output', None)
    return state


class Evaluation(BaseModel):
    """The evaluation of the optimized resume."""
    is_credible: bool = Field(description="Is the rewritten content believable and not overly exaggerated?")
    is_relevant: bool = Field(description="Does the optimized resume effectively address the job description's requirements?")
    is_grounded: bool = Field(description="Is the optimized resume factually consistent with the original resume?")
    ai_recommendations: List[str] = Field(description="A list of specific, actionable recommendations for improvement if any checks fail.")


def evaluate_resume(state: GraphState) -> Dict:
    """
    Evaluates the optimized resume against the original resume and job description.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Evaluation)

    system_prompt = """You are an expert hiring manager and career coach. Your task is to evaluate a rewritten resume.

You will be given the original resume, the rewritten 'optimized' resume, and the target job description.

Please evaluate the optimized resume based on three criteria:
1.  **Grounded:** Is the optimized resume factually consistent with the original resume? It should not invent new work experiences or skills that were not present in some form in the original. (Boolean: true/false)
2.  **Relevant:** Does the optimized resume effectively highlight the skills and experiences that are most relevant to the target job description? (Boolean: true/false)
3.  **Credible:** Is the optimized resume believable? Are the quantified achievements reasonable? Does it sound authentic? (Boolean: true/false)

Based on your evaluation, provide a list of specific, actionable recommendations for improvement. If the resume is excellent, the list should be empty."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original Resume:\n{original_resume}\n\nOptimized Resume:\n{optimized_resume}\n\nJob Description:\n{job_description}\n\nEvaluation:"),
    ])

    chain = prompt | structured_llm

    result = chain.invoke({
        "original_resume": state["resume"],
        "optimized_resume": state["optimized_resume"],
        "job_description": state["job_description"],
    })

    state['evaluation'] = result
    state['ai_recommendations'] = result.ai_recommendations
    return state


def output_formatter(state: GraphState) -> Dict:
    """
    Formats the final output for the user into the 'final_output' state key.
    """
    final_output = {
        "optimized_resume": state.get("optimized_resume"),
        "ai_recommendations": state.get("ai_recommendations"),
        "changes": state.get("changes"),
    }
    state["final_output"] = final_output
    return state


# --- Conditional Edge Functions ---

def decide_to_rewrite(state: GraphState) -> str:
    """
    Determines whether to proceed to the user or to loop back for another rewrite.
    This is the self-correction loop.
    """
    print("--- Evaluating resume for self-correction ---")
    evaluation = state.get("evaluation")

    if evaluation and evaluation.is_grounded and evaluation.is_relevant and evaluation.is_credible:
        print("--- Decision: Resume is good. Proceeding to user presentation. ---")
        return "present_to_user"
    else:
        print("--- Decision: Resume needs self-correction. Looping back to rewrite. ---")
        return "rewrite_resume"


def handle_user_feedback(state: GraphState) -> str:
    """
    Determines whether to loop for another rewrite based on user feedback.
    """
    print("--- Checking for user feedback ---")
    if state.get("user_feedback"):
        print("--- Decision: User feedback found. Looping back to rewrite. ---")
        return "rewrite_resume"
    else:
        print("--- Decision: No user feedback. Ending process. ---")
        return "end"


# --- Placeholder Node for Interruption ---

def present_to_user(state: GraphState) -> Dict:
    """
    A placeholder node that represents presenting the output to the user.
    The graph will interrupt after this node.
    """
    print("--- Presenting to user (graph will pause) ---")
    return {}


# --- Graph Assembly ---
from langgraph.graph import StateGraph, END, START

def create_graph() -> StateGraph:
    """
    Creates and compiles the resume optimization agent graph.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("process_inputs", process_inputs)
    workflow.add_node("extract_keywords", extract_keywords)
    workflow.add_node("rewrite_resume", rewrite_resume)
    workflow.add_node("evaluate_resume", evaluate_resume)
    workflow.add_node("present_to_user", present_to_user)
    workflow.add_node("output_formatter", output_formatter)

    # Define edges
    workflow.set_entry_point("process_inputs")
    workflow.add_edge("process_inputs", "extract_keywords")
    # Temporarily bypass RAG nodes
    workflow.add_edge("extract_keywords", "rewrite_resume")
    workflow.add_edge("rewrite_resume", "evaluate_resume")

    # Self-correction loop
    workflow.add_conditional_edges(
        "evaluate_resume",
        decide_to_rewrite,
        {
            "rewrite_resume": "rewrite_resume",
            "present_to_user": "present_to_user",
        }
    )

    # User feedback loop (after interruption)
    workflow.add_conditional_edges(
        "present_to_user",
        handle_user_feedback,
        {
            "rewrite_resume": "rewrite_resume",
            "end": "output_formatter",
        }
    )

    workflow.add_edge("output_formatter", END)

    # Compile the graph with interruption
    app = workflow.compile(interrupt_after=["present_to_user"])

    return app


def extract_keywords(state: GraphState) -> Dict:
    """
    Extracts keywords from the job description using an LLM.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Keywords)
    system_prompt = """You are an expert recruiter. Your task is to extract the most important keywords, skills, \
and technologies from the given job description. Focus on the core requirements and qualifications. \
Provide a concise list of these terms."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Here is the job description:\n\n{job_description}"),
    ])
    chain = prompt | structured_llm
    job_description = state["job_description"]
    result = chain.invoke({"job_description": job_description})
    state['keywords'] = result.keywords
    return state


def transform_query(state: GraphState) -> Dict:
    """
    Transforms the keywords and job description into a better query for RAG.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system_prompt = """You are a query optimization expert. Your task is to take a list of keywords and a job description \
and transform them into a single, concise, and semantic query that is ideal for retrieving relevant documents from a \
vector database. The query should capture the core essence of the job role."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Job Description:\n{job_description}\n\nKeywords:\n{keywords}\n\nOptimized Query:"),
    ])
    chain = prompt | llm | StrOutputParser()
    job_description = state["job_description"]
    keywords = state["keywords"]
    rag_query = chain.invoke({
        "job_description": job_description,
        "keywords": ", ".join(keywords)
    })
    state['rag_query'] = rag_query
    return state


class OptimizedResume(BaseModel):
    """The optimized resume and a summary of the changes made."""
    optimized_resume: Dict = Field(description="The full, optimized resume in JSON format.")
    changes: str = Field(description="A detailed summary of the specific changes made to the resume and the reasoning behind them.")


def rewrite_resume(state: GraphState) -> Dict:
    """
    Rewrites the resume based on the job description and keywords.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    structured_llm = llm.with_structured_output(OptimizedResume)

    system_prompt = """You are an expert career coach and resume writer. Your task is to optimize a client's resume to perfectly match a target job description.

You must:
1.  Analyze the provided resume (in JSON format), job description, and list of keywords.
2.  Rewrite the resume's sections (especially 'work_experience' and 'projects') to align with the job description.
3.  Incorporate the provided keywords naturally and effectively into the experience and project descriptions.
4.  Apply the STAR (Situation, Task, Action, Result) method to rephrase bullet points. Focus on quantifiable achievements and impact.
5.  Maintain a professional and confident tone. Do not fabricate experience, but creatively rephrase existing information to highlight the most relevant skills.
6.  Return the entire optimized resume in the original JSON structure.
7.  Provide a separate, detailed summary of the changes you made and why you made them."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Original Resume:\n{resume}\n\nJob Description:\n{job_description}\n\nKeywords to focus on:\n{keywords}"),
    ])

    chain = prompt | structured_llm

    result = chain.invoke({
        "resume": state["resume"],
        "job_description": state["job_description"],
        "keywords": ", ".join(state["keywords"])
    })

    state['optimized_resume'] = result.optimized_resume
    state['changes'] = result.changes
    return state
