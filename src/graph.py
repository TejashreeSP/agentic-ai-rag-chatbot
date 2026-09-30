from typing import TypedDict
import os

import streamlit as st
from langgraph.graph import StateGraph, START, END
from pinecone import Pinecone
from dotenv import load_dotenv
from google import genai


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()


def get_config(key):
    """
    Get configuration from environment variables first.
    If running on Streamlit Cloud, use Streamlit secrets.
    """

    value = os.getenv(key)

    if value:
        return value

    try:
        return st.secrets.get(key)
    except Exception:
        return None


PINECONE_API_KEY = get_config("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_config("PINECONE_INDEX_NAME")
GEMINI_API_KEY = get_config("GEMINI_API_KEY")


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# LANGGRAPH STATE
# =========================================================

class AgentState(TypedDict):
    question: str
    context: list
    answer: str
    score: float


# =========================================================
# RETRIEVAL NODE
# =========================================================

def retrieve(state: AgentState):

    pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

    index = pc.Index(
        PINECONE_INDEX_NAME
    )

    # Search Pinecone using the question
    results = index.search(
        namespace="agentic-ai",
        query={
            "inputs": {
                "text": state["question"]
            },
            "top_k": 5
        }
    )

    contexts = []

    # Extract retrieved chunks
    for result in results["result"]["hits"]:

        text = result["fields"].get(
            "text",
            ""
        ).strip()

        if text:

            contexts.append({
                "text": text,

                "source": result["fields"].get(
                    "source",
                    ""
                ),

                "page": result["fields"].get(
                    "page",
                    0
                ),

                "score": float(
                    result.get(
                        "_score",
                        0
                    )
                )
            })

    return {
        "context": contexts
    }


# =========================================================
# GENERATION NODE
# =========================================================

def generate(state: AgentState):

    context = state.get(
        "context",
        []
    )

    # -----------------------------------------------------
    # No retrieved context
    # -----------------------------------------------------

    if not context:

        return {
            "context": [],
            "answer": (
                "I could not find this information in the "
                "Agentic AI document."
            ),
            "score": 0.0
        }

    # -----------------------------------------------------
    # Find highest retrieval score
    # -----------------------------------------------------

    best_score = max(
        float(item.get("score", 0))
        for item in context
    )

    # -----------------------------------------------------
    # Reject unrelated questions
    # -----------------------------------------------------

    if best_score < 0.45:

        return {
            "context": [],
            "answer": (
                "I could not find this information in the "
                "Agentic AI document."
            ),
            "score": best_score
        }

    # -----------------------------------------------------
    # Prepare retrieved context
    # -----------------------------------------------------

    context_text = "\n\n".join(
        f"Page {item.get('page', 0)}:\n"
        f"{item.get('text', '')}"
        for item in context
    )

    # -----------------------------------------------------
    # Grounded RAG prompt
    # -----------------------------------------------------

    prompt = f"""
You are an Agentic AI RAG chatbot.

Answer the user's question ONLY using the retrieved
content from the Agentic AI eBook.

STRICT RULES:

1. Use only the retrieved document content.
2. Do not use outside knowledge.
3. Do not invent or assume information.
4. Answer the exact question asked.
5. Summarize the information in your own words.
6. Do not copy the PDF text word-for-word.
7. Keep the answer clear, natural, and concise.
8. If there are multiple points, use bullet points.
9. If the question asks for a definition, give a definition.
10. If the question asks for benefits, give only benefits
    supported by the document.
11. If the question asks for capabilities, give only
    capabilities supported by the document.
12. If the question asks for challenges, give only
    challenges supported by the document.
13. If the question asks for a comparison, compare only
    information supported by the document.
14. If the answer is not available in the retrieved
    document content, respond exactly with:

"I could not find this information in the Agentic AI document."

RETRIEVED DOCUMENT CONTENT:

{context_text}

USER QUESTION:

{state["question"]}

Give only the final answer to the user's question.
"""

    # -----------------------------------------------------
    # Gemini generation
    # -----------------------------------------------------

    try:

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        answer = response.text.strip()

        if answer:

            return {
                "answer": answer,
                "score": best_score
            }

    except Exception as e:

        print(
            "\n================ GEMINI ERROR ================"
        )

        print(
            repr(e)
        )

        print(
            "================================================\n"
        )

    # -----------------------------------------------------
    # Gemini unavailable
    # -----------------------------------------------------

    return {
        "answer": (
            "The answer-generation service is temporarily "
            "unavailable. Please try again."
        ),
        "score": best_score
    }


# =========================================================
# LANGGRAPH WORKFLOW
# =========================================================

graph_builder = StateGraph(
    AgentState
)


# Add nodes

graph_builder.add_node(
    "retrieve",
    retrieve
)

graph_builder.add_node(
    "generate",
    generate
)


# START → RETRIEVE

graph_builder.add_edge(
    START,
    "retrieve"
)


# RETRIEVE → GENERATE

graph_builder.add_edge(
    "retrieve",
    "generate"
)


# GENERATE → END

graph_builder.add_edge(
    "generate",
    END
)


# Compile LangGraph

graph = graph_builder.compile()