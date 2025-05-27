from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Load documents
loader = TextLoader("data\data.txt")
documents = loader.load()

# Split documents into chunks

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create and save vector store
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")
