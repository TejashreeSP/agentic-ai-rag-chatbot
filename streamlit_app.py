import streamlit as st
from src.graph import graph

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 Agentic AI RAG Chatbot")
st.write("Ask questions based on the Agentic AI eBook.")

query = st.text_input(
    "Enter your question:",
    placeholder="What is Agentic AI?"
)

if st.button("Ask") and query.strip():

    with st.spinner("Searching the Agentic AI knowledge base..."):

        result = graph.invoke({
            "question": query,
            "context": [],
            "answer": "",
            "score": 0.0
        })

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Confidence Score")
    st.write(round(result["score"], 3))

    if result.get("context"):

        st.subheader("Retrieved Context")

        for i, item in enumerate(result["context"], 1):

            with st.expander(
                f"Chunk {i} - Page {item.get('page', 0)}"
            ):
                st.write(item.get("text", ""))

    else:
        st.info(
            "No relevant information was found in the Agentic AI document."
        )