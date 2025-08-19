import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser

from .models import GraphState, Evaluation, Keywords, OptimizedResume
from .prompts import (
    EVALUATE_RESUME_SYSTEM_PROMPT,
    EXTRACT_KEYWORDS_SYSTEM_PROMPT,
    REWRITE_RESUME_SYSTEM_PROMPT,
    TRANSFORM_QUERY_SYSTEM_PROMPT,
)


def process_inputs(state: GraphState) -> GraphState:
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


def evaluate_resume(state: GraphState) -> GraphState:
    """
    Evaluates the optimized resume against the original resume and job description.
    """
    llm = ChatDeepSeek(model="deepseek-chat", api_key=os.environ.get("DEEPSEEK_API_KEY"), temperature=0)
    structured_llm = llm.with_structured_output(Evaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATE_RESUME_SYSTEM_PROMPT),
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


def output_formatter(state: GraphState) -> GraphState:
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


def extract_keywords(state: GraphState) -> GraphState:
    """
    Extracts keywords from the job description using an LLM.
    """
    llm = ChatDeepSeek(model="deepseek-chat", api_key=os.environ.get("DEEPSEEK_API_KEY"), temperature=0)
    structured_llm = llm.with_structured_output(Keywords)
    prompt = ChatPromptTemplate.from_messages([
        ("system", EXTRACT_KEYWORDS_SYSTEM_PROMPT),
        ("human", "Here is the job description:\n\n{job_description}"),
    ])
    chain = prompt | structured_llm
    job_description = state["job_description"]
    result = chain.invoke({"job_description": job_description})
    state['keywords'] = result.keywords
    return state


def transform_query(state: GraphState) -> GraphState:
    """
    Transforms the keywords and job description into a better query for RAG.
    """
    llm = ChatDeepSeek(model="deepseek-chat", api_key=os.environ.get("DEEPSEEK_API_KEY"), temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", TRANSFORM_QUERY_SYSTEM_PROMPT),
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


def rewrite_resume(state: GraphState) -> GraphState:
    """
    Rewrites the resume based on the job description and keywords.
    """
    llm = ChatDeepSeek(model="deepseek-chat", api_key=os.environ.get("DEEPSEEK_API_KEY"), temperature=0.3)
    structured_llm = llm.with_structured_output(OptimizedResume)

    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_RESUME_SYSTEM_PROMPT),
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


def present_to_user(state: GraphState) -> dict:
    """
    A placeholder node that represents presenting the output to the user.
    The graph will interrupt after this node.
    """
    print("--- Presenting to user (graph will pause) ---")
    return {}
