import os
import httpx
from .models import GraphState


def rag_keyword_enhancer(state: GraphState) -> GraphState:
    """
    Enhances keywords by making an API call to a LightRAG server.
    (This is a mock implementation as we don't have credentials)
    """
    lightrag_url = os.environ.get("LIGHTRAG_API_URL", "http://localhost:9621/api/query")
    print(f"--- Calling LightRAG API at {lightrag_url} ---")

    try:
        payload = {"query": state["rag_query"], "mode": "hybrid"}
        response = httpx.post(lightrag_url, json=payload, timeout=60)
        response.raise_for_status()

        response_json = response.json()
        enhanced_keywords = response_json.get("keywords", [])

        if enhanced_keywords:
            print("--- RAG call successful, updating keywords ---")
            state['keywords'] = enhanced_keywords
        else:
            print("--- RAG call did not return new keywords, keeping original ---")

    except httpx.RequestError as e:
        print(f"--- LightRAG API call failed: {e}. Keeping original keywords. ---")

    return state
