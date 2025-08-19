import pytest
import os
from langgraph.graph import StateGraph
from resume_optimizer.models import GraphState, Keywords, OptimizedResume, Evaluation, RagDecision
from resume_optimizer.nodes import (
    process_inputs, extract_keywords, rewrite_resume,
    evaluate_resume, output_formatter
)
from resume_optimizer.graph import (
    decide_to_rewrite, handle_user_feedback, create_graph, should_use_rag
)

def test_graph_state_initialization():
    """
    Tests if the StateGraph can be initialized with the GraphState TypedDict.
    """
    try:
        workflow = StateGraph(GraphState)
        assert workflow is not None
        assert set(workflow.channels.keys()) == {
            "resume", "job_description", "user_feedback", "keywords",
            "optimized_resume", "ai_recommendations", "changes", "rag_query",
            "evaluation", "final_output",
        }
    except Exception as e:
        pytest.fail(f"StateGraph initialization failed with GraphState: {e}")

def test_process_inputs_node():
    """
    Tests the process_inputs node to ensure it correctly initializes the state.
    """
    initial_state = {"resume": {}, "job_description": ""}
    output_state = process_inputs(initial_state)
    assert output_state.get("user_feedback") == []
    assert output_state.get("keywords") == []
    assert output_state.get("rag_query") == ""
    assert output_state.get("evaluation") is None
    assert output_state.get("final_output") is None

def test_extract_keywords_node(mocker):
    """Tests the extract_keywords node."""
    mocker.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key"})
    mocker.patch("langchain_core.runnables.base.RunnableSequence.invoke", return_value=Keywords(keywords=["Python"]))
    state = {"job_description": "Need Python."}
    output_state = extract_keywords(state)
    assert output_state["keywords"] == ["Python"]

def test_rewrite_resume_node(mocker):
    """Tests the rewrite_resume node."""
    mocker.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key"})
    mock_output = OptimizedResume(optimized_resume={"exp": "new"}, changes="...")
    mocker.patch("langchain_core.runnables.base.RunnableSequence.invoke", return_value=mock_output)
    state = {"resume": {}, "job_description": "", "keywords": []}
    output_state = rewrite_resume(state)
    assert output_state["optimized_resume"]["exp"] == "new"

def test_evaluate_resume_node(mocker):
    """Tests the evaluate_resume node."""
    mocker.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key"})
    mock_output = Evaluation(is_credible=True, is_relevant=True, is_grounded=True, ai_recommendations=["Good job"])
    mocker.patch("langchain_core.runnables.base.RunnableSequence.invoke", return_value=mock_output)
    state = {"resume": {}, "optimized_resume": {}, "job_description": ""}
    output_state = evaluate_resume(state)
    assert output_state["evaluation"].is_credible is True
    assert output_state["ai_recommendations"] == ["Good job"]

def test_output_formatter_node():
    """Tests the output_formatter node."""
    state = {"optimized_resume": {"exp": "final"}, "ai_recommendations": [], "changes": "final"}
    output_state = output_formatter(state)
    assert output_state["final_output"]["optimized_resume"]["exp"] == "final"

def test_decide_to_rewrite_edge():
    """Tests the decide_to_rewrite conditional edge."""
    good_state = {"evaluation": Evaluation(is_credible=True, is_relevant=True, is_grounded=True, ai_recommendations=[])}
    bad_state = {"evaluation": Evaluation(is_credible=True, is_relevant=False, is_grounded=True, ai_recommendations=[])}
    assert decide_to_rewrite(good_state) == "present_to_user"
    assert decide_to_rewrite(bad_state) == "rewrite_resume"

def test_handle_user_feedback_edge():
    """Tests the handle_user_feedback conditional edge."""
    assert handle_user_feedback({"user_feedback": ["more professional"]}) == "rewrite_resume"
    assert handle_user_feedback({"user_feedback": []}) == "end"

def test_graph_assembly_and_happy_path(mocker):
    """Tests the graph assembly and a happy path execution."""
    mocker.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key"})
    # Mock the chain invokes. Add a mock for the new RAG decision node.
    mocker.patch("langchain_core.runnables.base.RunnableSequence.invoke", side_effect=[
        Keywords(keywords=["Python"]),
        RagDecision(decision="no_rag"), # 1. Mock the RAG decision
        OptimizedResume(optimized_resume={"exp": "Optimized"}, changes="..."),
        Evaluation(is_credible=True, is_relevant=True, is_grounded=True, ai_recommendations=[])
    ])
    app = create_graph()
    config = {"configurable": {"thread_id": "test-thread-1"}}
    final_state = app.invoke({"resume": {}, "job_description": ""}, config)
    assert final_state["optimized_resume"]["exp"] == "Optimized"

def test_graph_self_correction_loop(mocker):
    """Tests the self-correction loop."""
    mocker.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key"})
    mocker.patch("langchain_core.runnables.base.RunnableSequence.invoke", side_effect=[
        Keywords(keywords=["Python"]),
        RagDecision(decision="no_rag"), # Mock the RAG decision
        OptimizedResume(optimized_resume={"exp": "Bad"}, changes="..."),
        Evaluation(is_credible=False, is_relevant=True, is_grounded=True, ai_recommendations=[]),
        OptimizedResume(optimized_resume={"exp": "Good"}, changes="..."),
        Evaluation(is_credible=True, is_relevant=True, is_grounded=True, ai_recommendations=[])
    ])
    app = create_graph()
    config = {"configurable": {"thread_id": "test-thread-2"}}
    final_state = app.invoke({"resume": {}, "job_description": "..."}, config)
    assert final_state["optimized_resume"]["exp"] == "Good"

# The user feedback loop is complex to test with mocks due to the
# checkpointer state. The happy path and self-correction loop tests
# provide sufficient coverage for the graph's logic for now.
# The user feedback loop can be tested manually with the run_agent.py script.


def test_rag_keyword_enhancer_node(mocker):
    """
    Tests the RAG keyword enhancement node by mocking the API call.
    """
    import httpx
    from resume_optimizer.tools import rag_keyword_enhancer

    # Mock the httpx.post call
    mock_response = mocker.MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "keywords": ["Enhanced Python", "Cloud Services", "Advanced SQL"]
    }
    mocker.patch("httpx.post", return_value=mock_response)

    # Input state
    initial_state = {
        "rag_query": "software engineer with Python and SQL",
        "keywords": ["Python", "SQL"], # Pass original keywords to ensure they get updated
    }

    # Execute the node
    output_state = rag_keyword_enhancer(initial_state)

    # Assertions
    assert "keywords" in output_state
    assert output_state["keywords"] == ["Enhanced Python", "Cloud Services", "Advanced SQL"]


def test_should_use_rag_edge(mocker):
    """
    Tests the conditional edge that decides whether to use RAG.
    """
    # Mock the LLM to return "rag"
    mocker.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key"})
    mocker.patch(
        "langchain_core.runnables.base.RunnableSequence.invoke",
        return_value=RagDecision(decision="rag")
    )

    state_for_rag = {"job_description": "Requires esoteric knowledge in quantum blockchain."}
    assert should_use_rag(state_for_rag) == "transform_query" # Path to RAG

    # Mock the LLM to return "no_rag"
    mocker.patch(
        "langchain_core.runnables.base.RunnableSequence.invoke",
        return_value=RagDecision(decision="no_rag")
    )

    state_for_no_rag = {"job_description": "Standard software engineer job."}
    assert should_use_rag(state_for_no_rag) == "rewrite_resume" # Path to bypass RAG
