import streamlit as st
from src.graph import graph


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Agentic AI RAG Chatbot")

st.write(
    "Ask questions based on the Agentic AI eBook."
)


# =========================================================
# QUESTION INPUT
# =========================================================

query = st.text_input(
    "Enter your question:",
    placeholder="What is Agentic AI?"
)


# =========================================================
# ASK BUTTON
# =========================================================

if st.button("Ask"):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching the Agentic AI knowledge base..."
        ):

            try:

                result = graph.invoke(
                    {
                        "question": query,
                        "context": [],
                        "answer": "",
                        "score": 0.0
                    }
                )

                # -----------------------------------------
                # ANSWER
                # -----------------------------------------

                st.subheader("Answer")

                answer = result.get(
                    "answer",
                    ""
                )

                if answer:

                    st.write(answer)

                else:

                    st.write(
                        "No answer was generated."
                    )


                # -----------------------------------------
                # CONFIDENCE SCORE
                # -----------------------------------------

                st.subheader(
                    "Confidence Score"
                )

                score = result.get(
                    "score",
                    0.0
                )

                st.write(
                    round(
                        float(score),
                        3
                    )
                )


                # -----------------------------------------
                # RETRIEVED CONTEXT
                # -----------------------------------------

                context = result.get(
                    "context",
                    []
                )

                if context:

                    st.subheader(
                        "Retrieved Context"
                    )

                    for i, item in enumerate(
                        context,
                        1
                    ):

                        page = item.get(
                            "page",
                            0
                        )

                        text = item.get(
                            "text",
                            ""
                        )

                        score = item.get(
                            "score",
                            0
                        )

                        with st.expander(
                            f"Chunk {i} - Page {page}"
                        ):

                            st.write(text)

                            st.caption(
                                f"Retrieval score: "
                                f"{float(score):.3f}"
                            )

                else:

                    st.info(
                        "No relevant information was "
                        "found in the Agentic AI document."
                    )


            except Exception as e:

                st.error(
                    "An error occurred while processing "
                    "your question."
                )

                st.exception(e)