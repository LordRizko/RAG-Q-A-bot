import streamlit as st
import os
import shutil
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.chat_models import ChatOpenAI

# --- Page Configuration ---
st.set_page_config(
    page_title="Doc Q&A",
    page_icon="📄",
    layout="wide"
)

# Load environment variables
load_dotenv()

# --- Backend Logic ---

def load_and_chunk_documents(file_path):
    """
    Loads a PDF, splits it into chunks.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    return chunks

def create_vector_store(chunks):
    """
    Creates a Chroma vector store from document chunks using a sentence-transformer model.
    """
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(chunks, embedding_function)
    return vectorstore

def create_qa_pipeline(vectorstore):
    """
    Creates a Question-Answering pipeline using an OpenRouter model with conversational memory.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

    llm = ChatOpenAI(
        model_name="mistralai/mistral-7b-instruct",
        temperature=0,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
    return qa_chain

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_pipeline" not in st.session_state:
    st.session_state.qa_pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar for PDF Upload ---
with st.sidebar:
    st.header("Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF... This may take a moment."):
                # Save the uploaded file temporarily
                temp_dir = "temp_data"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    chunks = load_and_chunk_documents(file_path)
                    vector_store = create_vector_store(chunks)
                    st.session_state.qa_pipeline = create_qa_pipeline(vector_store)
                    
                    st.success("PDF processed successfully!")
                    st.session_state.messages = []
                    st.session_state.chat_history = []
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(file_path):
                        shutil.rmtree(temp_dir)


    if st.session_state.qa_pipeline:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

# --- Main Chat Interface ---
st.title("📄 Document Q&A with RAG")
st.write("Upload a PDF in the sidebar, then ask questions about its content.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                     st.write(f"- Page {source.get('page', 'N/A')}")

# Chat input
if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.qa_pipeline:
        st.warning("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_pipeline({
                        "question": prompt, 
                        "chat_history": st.session_state.chat_history
                    })
                    answer = result.get("answer", "No answer found.")
                    sources = result.get("source_documents", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.write(f"- Page {source.get('page', 'N/A')}")
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                    st.session_state.chat_history.append((prompt, answer))

                except Exception as e:
                    st.error(f"An error occurred: {e}")

