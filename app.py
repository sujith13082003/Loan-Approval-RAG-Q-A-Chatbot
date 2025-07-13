# ✅ app.py — RAG Q&A Chatbot using Hugging Face

import streamlit as st
from retriever import FAISSRetriever
from generator import generate_answer

# 🧭 Page configuration
st.set_page_config(page_title="📊 Loan Approval RAG Q&A Chatbot (HF)", layout="wide")
st.title("📊 Loan Approval RAG Q&A Chatbot")
st.markdown(
    """
    Ask a question about loan approvals based on the dataset.  
    The chatbot retrieves similar cases and uses a Hugging Face model to generate answers.
    """
)

# 🧠 Load Retriever
@st.cache_resource
def load_retriever():
    return FAISSRetriever(data_path="Training Dataset.csv")

retriever = load_retriever()

# 🔍 Input Question
question = st.text_input("🔍 Enter your question:")

# 💬 Generate Answer
if question:
    with st.spinner("🔎 Retrieving relevant data..."):
        retrieved_docs = retriever.retrieve(question, top_k=5)
        context = "\n\n".join(retrieved_docs)

    with st.spinner("🧠 Generating answer using Hugging Face model..."):
        answer = generate_answer(question, context)

    st.markdown("### ✅ Answer:")
    st.success(answer)

    with st.expander("📄 View Retrieved Documents"):
        for i, doc in enumerate(retrieved_docs, 1):
            st.markdown(f"**Doc {i}:** {doc}")
