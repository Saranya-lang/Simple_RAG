import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Initialize LLM (Groq - Mixtral or LLaMA3)
llm = ChatGroq(
    groq_api_key="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",  # Set this via env var or directly
    model_name="llama3-8b-8192",   # or "llama3-70b-8192"
)

# Load FAISS and embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

# Set up retriever and RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

# Streamlit UI
st.title("🔍 Simple RAG App with Groq")
query = st.text_input("Ask a question:")

if query:
    result = qa_chain(query)
    st.subheader("Answer")
    st.write(result["result"])

    # Optional: show source docs
    with st.expander("📄 Sources"):
        for doc in result["source_documents"]:
            st.markdown(doc.page_content)
