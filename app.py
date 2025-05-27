import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # or HuggingFaceHub
from langchain.prompts import PromptTemplate

# Load FAISS and embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings)

# Create retrieval-based QA chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),  # You can also use HuggingFaceHub for local models
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# Streamlit UI
st.title("🔍 Simple RAG Q&A Bot")
query = st.text_input("Ask a question:")

if query:
    result = qa_chain(query)
    st.write("### Answer:")
    st.write(result["result"])

    # Optional: show source documents
    with st.expander("📄 Sources"):
        for doc in result["source_documents"]:
            st.write(doc.page_content)
