import os
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

from .models import GraphState, RagDecision
from . import nodes
from . import tools
from .prompts import SHOULD_USE_RAG_SYSTEM_PROMPT


def should_use_rag(state: GraphState) -> str:
    """
    Determines whether to use the RAG pipeline or go directly to rewriting.
    """
    print("--- Deciding whether to use RAG ---")
    llm = ChatDeepSeek(model="deepseek-chat", api_key=os.environ.get("DEEPSEEK_API_KEY"), temperature=0)
    structured_llm = llm.with_structured_output(RagDecision)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SHOULD_USE_RAG_SYSTEM_PROMPT),
        ("human", "Job Description:\n{job_description}\n\nDecision:"),
    ])

    chain = prompt | structured_llm

    result = chain.invoke({"job_description": state["job_description"]})

    if result.decision == "rag":
        print("--- Decision: RAG required for keyword enhancement. ---")
        return "transform_query"
    else:
        print("--- Decision: RAG not required. Proceeding to rewrite. ---")
        return "rewrite_resume"


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


def create_graph():
    """
    Creates and compiles the resume optimization agent graph.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("process_inputs", nodes.process_inputs)
    workflow.add_node("extract_keywords", nodes.extract_keywords)
    workflow.add_node("transform_query", nodes.transform_query)
    workflow.add_node("rag_keyword_enhancer", tools.rag_keyword_enhancer)
    workflow.add_node("rewrite_resume", nodes.rewrite_resume)
    workflow.add_node("evaluate_resume", nodes.evaluate_resume)
    workflow.add_node("present_to_user", nodes.present_to_user)
    workflow.add_node("output_formatter", nodes.output_formatter)

    # Define edges
    workflow.set_entry_point("process_inputs")
    workflow.add_edge("process_inputs", "extract_keywords")

    # Add the new RAG routing logic
    workflow.add_conditional_edges(
        "extract_keywords",
        should_use_rag,
        {
            "transform_query": "transform_query",
            "rewrite_resume": "rewrite_resume",
        }
    )
    workflow.add_edge("transform_query", "rag_keyword_enhancer")
    workflow.add_edge("rag_keyword_enhancer", "rewrite_resume")

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

    # Compile the graph with interruption and a memory saver
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer, interrupt_after=["present_to_user"])

    return app
