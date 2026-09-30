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
    value = os.getenv(key)

    if value:
        return value

    try:
        return st.secrets[key]
    except Exception:
        return None


PINECONE_API_KEY = get_config("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_config("PINECONE_INDEX_NAME")
GEMINI_API_KEY = get_config("GEMINI_API_KEY")


# =========================================================
# CLIENTS
# =========================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(
    PINECONE_INDEX_NAME
)

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
# RETRIEVAL
# =========================================================

def retrieve(state: AgentState):

    question = state["question"]

    try:

        results = index.search(
            namespace="agentic-ai",
            query={
                "inputs": {
                    "text": question
                },
                "top_k": 5
            }
        )

        contexts = []

        # -------------------------------------------------
        # Get hits safely
        # -------------------------------------------------

        if isinstance(results, dict):

            result_data = results.get(
                "result",
                {}
            )

            hits = result_data.get(
                "hits",
                []
            )

        else:

            result_data = getattr(
                results,
                "result",
                None
            )

            if result_data is not None:

                hits = getattr(
                    result_data,
                    "hits",
                    []
                )

            else:

                hits = []

        # -------------------------------------------------
        # Process retrieved chunks
        # -------------------------------------------------

        for hit in hits:

            if isinstance(hit, dict):

                fields = hit.get(
                    "fields",
                    {}
                )

                score = hit.get(
                    "_score",
                    0
                )

            else:

                fields = getattr(
                    hit,
                    "fields",
                    {}
                )

                score = getattr(
                    hit,
                    "_score",
                    0
                )

            # Make fields dictionary if necessary
            if not isinstance(fields, dict):

                try:
                    fields = dict(fields)
                except Exception:
                    fields = {}

            text = str(
                fields.get(
                    "text",
                    ""
                )
            ).strip()

            if text:

                contexts.append({
                    "text": text,

                    "source": str(
                        fields.get(
                            "source",
                            ""
                        )
                    ),

                    "page": fields.get(
                        "page",
                        0
                    ),

                    "score": float(
                        score
                    )
                })

        # -------------------------------------------------
        # DEBUG INFORMATION
        # -------------------------------------------------

        print(
            "QUESTION:",
            question
        )

        print(
            "NUMBER OF HITS:",
            len(hits)
        )

        print(
            "NUMBER OF CONTEXTS:",
            len(contexts)
        )

        if contexts:

            print(
                "BEST SCORE:",
                max(
                    item["score"]
                    for item in contexts
                )
            )

        return {
            "context": contexts
        }

    except Exception as e:

        print(
            "================ PINECONE ERROR ================"
        )

        print(
            repr(e)
        )

        print(
            "================================================="
        )

        return {
            "context": []
        }


# =========================================================
# GENERATION
# =========================================================

def generate(state: AgentState):

    context = state.get(
        "context",
        []
    )

    # -----------------------------------------------------
    # No context
    # -----------------------------------------------------

    if not context:

        return {
            "answer": (
                "I could not find this information in the "
                "Agentic AI document."
            ),
            "score": 0.0
        }

    # -----------------------------------------------------
    # Best score
    # -----------------------------------------------------

    best_score = max(
        float(item["score"])
        for item in context
    )

    # -----------------------------------------------------
    # Minimum relevance
    # -----------------------------------------------------

    if best_score < 0.45:

        return {
            "answer": (
                "I could not find this information in the "
                "Agentic AI document."
            ),
            "score": best_score
        }

    # -----------------------------------------------------
    # Prepare context
    # -----------------------------------------------------

    context_text = "\n\n".join(
        f"Page {item.get('page', 0)}:\n"
        f"{item.get('text', '')}"
        for item in context
    )

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = f"""
You are an Agentic AI RAG chatbot.

Answer the user's question ONLY using the retrieved
content from the Agentic AI eBook.

STRICT RULES:

1. Use only the retrieved document content.
2. Do not use outside knowledge.
3. Do not invent information.
4. Answer the exact question.
5. Summarize in your own words.
6. Keep the answer clear and concise.
7. Use bullet points when appropriate.
8. If the information is not available in the
   retrieved content, say:

"I could not find this information in the Agentic AI document."

RETRIEVED DOCUMENT CONTENT:

{context_text}

USER QUESTION:

{state["question"]}

Give only the final answer.
"""

    # -----------------------------------------------------
    # Gemini
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
            "================ GEMINI ERROR ================"
        )

        print(
            repr(e)
        )

        print(
            "================================================"
        )

    return {
        "answer": (
            "The answer-generation service is temporarily "
            "unavailable. Please try again."
        ),
        "score": best_score
    }


# =========================================================
# LANGGRAPH
# =========================================================

graph_builder = StateGraph(
    AgentState
)

graph_builder.add_node(
    "retrieve",
    retrieve
)

graph_builder.add_node(
    "generate",
    generate
)

graph_builder.add_edge(
    START,
    "retrieve"
)

graph_builder.add_edge(
    "retrieve",
    "generate"
)

graph_builder.add_edge(
    "generate",
    END
)

graph = graph_builder.compile()