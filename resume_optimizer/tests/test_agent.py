import pytest
import os
from langgraph.graph import StateGraph
from resume_optimizer.agent import GraphState, process_inputs, extract_keywords, transform_query, Keywords

def test_graph_state_initialization():
    """
    Tests if the StateGraph can be initialized with the GraphState TypedDict.
    """
    try:
        workflow = StateGraph(GraphState)
        assert workflow is not None
        assert set(workflow.channels.keys()) == {
            "resume",
            "job_description",
            "user_feedback",
            "keywords",
            "optimized_resume",
            "ai_recommendations",
            "changes",
            "rag_query",
            "evaluation",
            "final_output",
        }
    except Exception as e:
        pytest.fail(f"StateGraph initialization failed with GraphState: {e}")


def test_process_inputs_node():
    """
    Tests the process_inputs node to ensure it correctly initializes the state.
    """
    initial_state = {
        "resume": {"experience": "some experience"},
        "job_description": "a job description",
    }
    output_state = process_inputs(initial_state)
    assert output_state.get("user_feedback") == []
    assert output_state.get("keywords") == []
    assert output_state.get("optimized_resume") == {}
    assert output_state.get("ai_recommendations") == []
    assert output_state.get("changes") == ""
    assert output_state.get("rag_query") == ""
    assert output_state.get("evaluation") is None
    assert output_state.get("final_output") is None


def test_extract_keywords_node(mocker):
    """
    Tests the extract_keywords node.
    """
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    mocker.patch(
        "langchain_core.runnables.base.RunnableSequence.invoke",
        return_value=Keywords(keywords=["Python", "Langchain", "SQL"])
    )
    initial_state = {
        "job_description": "We are looking for a software engineer with experience in Python, Langchain, and SQL."
    }
    output_state = extract_keywords(initial_state)
    assert output_state["keywords"] == ["Python", "Langchain", "SQL"]


def test_transform_query_node(mocker):
    """
    Tests the transform_query node.
    """
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    mocker.patch(
        "langchain_core.runnables.base.RunnableSequence.invoke",
        return_value="semantic query for software engineer with Python and SQL"
    )
    initial_state = {
        "job_description": "We need Python and SQL.",
        "keywords": ["Python", "SQL", "Software Engineer"],
    }
    output_state = transform_query(initial_state)
    assert output_state["rag_query"] == "semantic query for software engineer with Python and SQL"


def test_rewrite_resume_node(mocker):
    """
    Tests the rewrite_resume node.
    It should take a resume and keywords and return an optimized resume and changes.
    """
    from resume_optimizer.agent import rewrite_resume, OptimizedResume

    # Mock the LLM call
    mock_output = OptimizedResume(
        optimized_resume={"experience": "Optimized experience using Python and SQL."},
        changes="Rewrote experience section to highlight Python and SQL skills."
    )
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    mocker.patch(
        "langchain_core.runnables.base.RunnableSequence.invoke",
        return_value=mock_output
    )

    # Input state
    initial_state = {
        "resume": {"experience": "I worked with Python and SQL."},
        "job_description": "We need Python and SQL.",
        "keywords": ["Python", "SQL"],
    }

    # Execute the node
    output_state = rewrite_resume(initial_state)

    # Assertions
    assert "optimized_resume" in output_state
    assert "changes" in output_state
    assert output_state["optimized_resume"] == mock_output.optimized_resume
    assert output_state["changes"] == mock_output.changes


def test_evaluate_resume_node(mocker):
    """
    Tests the evaluate_resume node.
    It should take original and optimized resumes and return an evaluation.
    """
    from resume_optimizer.agent import evaluate_resume, Evaluation

    # Mock the LLM call
    mock_output = Evaluation(
        is_credible=True,
        is_relevant=True,
        is_grounded=True,
        ai_recommendations=["The resume looks great!"]
    )
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})
    mocker.patch(
        "langchain_core.runnables.base.RunnableSequence.invoke",
        return_value=mock_output
    )

    # Input state
    initial_state = {
        "resume": {"experience": "Original."},
        "optimized_resume": {"experience": "Optimized."},
        "job_description": "A job description.",
    }

    # Execute the node
    output_state = evaluate_resume(initial_state)

    # Assertions
    assert "evaluation" in output_state
    assert output_state["evaluation"].is_credible is True
    assert output_state["evaluation"].is_relevant is True
    assert "ai_recommendations" in output_state
    assert output_state["ai_recommendations"] == ["The resume looks great!"]


def test_output_formatter_node():
    """
    Tests the output_formatter node.
    It should take the final state and format it into the required output dict.
    """
    from resume_optimizer.agent import output_formatter

    # Input state
    initial_state = {
        "optimized_resume": {"experience": "Final optimized experience."},
        "ai_recommendations": ["Recommendation 1", "Recommendation 2"],
        "changes": "Final summary of changes."
    }

    # Execute the node
    output_state = output_formatter(initial_state)

    # Assertions
    assert "final_output" in output_state
    final_output = output_state["final_output"]
    assert isinstance(final_output, dict)
    assert final_output.get("optimized_resume") == initial_state["optimized_resume"]
    assert final_output.get("ai_recommendations") == initial_state["ai_recommendations"]
    assert final_output.get("changes") == initial_state["changes"]


def test_decide_to_rewrite_edge(mocker):
    """
    Tests the conditional edge that decides whether to rewrite or present to the user.
    """
    from resume_optimizer.agent import decide_to_rewrite, Evaluation

    # Case 1: Evaluation is good, should proceed to present
    good_state = {
        "evaluation": Evaluation(is_credible=True, is_relevant=True, is_grounded=True, ai_recommendations=[])
    }
    assert decide_to_rewrite(good_state) == "present_to_user"

    # Case 2: Evaluation is not grounded, should rewrite
    bad_state_grounded = {
        "evaluation": Evaluation(is_credible=True, is_relevant=True, is_grounded=False, ai_recommendations=["Invented skills"])
    }
    assert decide_to_rewrite(bad_state_grounded) == "rewrite_resume"

    # Case 3: Evaluation is not relevant, should rewrite
    bad_state_relevant = {
        "evaluation": Evaluation(is_credible=True, is_relevant=False, is_grounded=True, ai_recommendations=["Not tailored"])
    }
    assert decide_to_rewrite(bad_state_relevant) == "rewrite_resume"


def test_handle_user_feedback_edge():
    """
    Tests the conditional edge that decides whether to loop based on user feedback.
    """
    from resume_optimizer.agent import handle_user_feedback

    # Case 1: User provides feedback, should loop to rewrite
    state_with_feedback = {"user_feedback": ["Make it more professional."]}
    assert handle_user_feedback(state_with_feedback) == "rewrite_resume"

    # Case 2: User provides no feedback, should end
    state_no_feedback = {"user_feedback": []}
    assert handle_user_feedback(state_no_feedback) == "end"


def test_graph_assembly_and_happy_path(mocker):
    """
    Tests the graph assembly and a happy path execution.
    """
    from resume_optimizer.agent import create_graph, Keywords, OptimizedResume, Evaluation

    # --- Mock all external calls ---
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})

    # Mock the 'invoke' method for all RunnableSequence chains.
    # The side_effect list provides the return values in the order they are called.
    mocker.patch(
        "langchain_core.runnables.base.RunnableSequence.invoke",
        side_effect=[
            Keywords(keywords=["Python", "SQL"]),      # 1. from extract_keywords
            OptimizedResume(                           # 2. from rewrite_resume
                optimized_resume={"experience": "Optimized"},
                changes="Rewrote everything."
            ),
            Evaluation(                                # 3. from evaluate_resume (good eval)
                is_credible=True, is_relevant=True, is_grounded=True, ai_recommendations=[]
            )
        ]
    )

    # --- Run the graph ---
    app = create_graph()
    initial_input = {
        "resume": {"experience": "Original"},
        "job_description": "A job description"
    }

    # Invoke the graph. It should stop at the 'present_to_user' interrupt.
    final_state = app.invoke(initial_input)

    # --- Assertions ---
    # Check that the state has been populated by the nodes
    assert final_state["keywords"] == ["Python", "SQL"]
    assert final_state["optimized_resume"]["experience"] == "Optimized"
    assert final_state["changes"] == "Rewrote everything."
    assert final_state["evaluation"].is_credible is True


def test_graph_self_correction_loop(mocker):
    """
    Tests that the graph correctly loops back to rewrite the resume
    if the initial evaluation is negative.
    """
    from resume_optimizer.agent import create_graph, Keywords, OptimizedResume, Evaluation

    # --- Mock all external calls ---
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"})

    # Mock the chain invokes with side_effect to simulate the loop
    mocker.patch(
        "langchain_core.runnables.base.RunnableSequence.invoke",
        side_effect=[
            Keywords(keywords=["Python", "SQL"]),      # 1. extract_keywords
            OptimizedResume(                           # 2. rewrite_resume (first attempt)
                optimized_resume={"experience": "Bad rewrite"},
                changes="First attempt."
            ),
            Evaluation(                                # 3. evaluate_resume (first attempt - fails)
                is_credible=True, is_relevant=False, is_grounded=True, ai_recommendations=["Needs more keywords"]
            ),
            OptimizedResume(                           # 4. rewrite_resume (second attempt)
                optimized_resume={"experience": "Good rewrite"},
                changes="Second attempt."
            ),
            Evaluation(                                # 5. evaluate_resume (second attempt - succeeds)
                is_credible=True, is_relevant=True, is_grounded=True, ai_recommendations=[]
            ),
        ]
    )

    # --- Run the graph ---
    app = create_graph()
    initial_input = {"resume": {}, "job_description": "..."}
    final_state = app.invoke(initial_input)

    # --- Assertions ---
    # The final state should reflect the *second* successful rewrite.
    assert final_state["optimized_resume"]["experience"] == "Good rewrite"
    assert final_state["changes"] == "Second attempt."
