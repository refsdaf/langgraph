import os
import json
from pprint import pprint
from resume_optimizer.agent import create_graph

def run_test():
    """
    Runs a real test of the resume optimizer agent using the Gemini API.
    """
    # Set the API key for the DeepSeek model
    os.environ["DEEPSEEK_API_KEY"] = "sk-1a1e059d7da9442a9de5dcf195b66a8c"

    # Sample input data
    sample_resume = {
        "personal_info": {
            "name": "John Doe",
            "email": "john.doe@email.com"
        },
        "work_experience": [
            {
                "title": "Software Engineer",
                "company": "Tech Solutions Inc.",
                "duration": "Jan 2022 - Present",
                "description": [
                    "Developed a Python-based data processing pipeline.",
                    "Worked with SQL databases to store and retrieve user data.",
                    "Helped build a customer-facing web application."
                ]
            }
        ],
        "skills": ["Python", "SQL", "JavaScript"]
    }

    sample_job_description = """
    We are seeking a Senior Python Developer to join our dynamic team.
    The ideal candidate will have extensive experience in building scalable applications using Python.
    Responsibilities include designing and implementing backend services, working with large-scale databases (SQL),
    and collaborating with front-end developers. Experience with cloud platforms and building AI-powered
    applications with LangChain is a strong plus.
    """

    initial_input = {
        "resume": sample_resume,
        "job_description": sample_job_description,
    }

    # Create and run the graph
    app = create_graph()
    print("--- Invoking Agent ---")

    # The graph will run until it hits the first interruption point
    final_state = None
    for output in app.stream(initial_input):
        for key, value in output.items():
            pprint(f"--- Node: {key} ---")
            # The 'value' is the entire state dictionary at that node
            if isinstance(value, dict):
                final_state = value
            pprint(value)
            print("\n")

    print("\n--- Agent Paused for User Feedback ---")
    if final_state:
        pprint("Optimized Resume:")
        pprint(final_state.get("optimized_resume"))
        print("\nChanges Made:")
        pprint(final_state.get("changes"))
        print("\nAI Recommendations:")
        pprint(final_state.get("ai_recommendations"))


if __name__ == "__main__":
    run_test()
