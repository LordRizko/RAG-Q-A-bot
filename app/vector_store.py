from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def create_vector_store(chunks):
    """
    Creates a Chroma vector store from document chunks using a sentence-transformer model.
    """
    # Using a local sentence-transformer model for embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(chunks, embedding_function)
    return vectorstore
