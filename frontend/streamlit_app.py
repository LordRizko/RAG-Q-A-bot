import streamlit as st
import requests
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Doc Q&A",
    page_icon="📄",
    layout="wide"
)

# --- Backend URL ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar for PDF Upload ---
with st.sidebar:
    st.header("Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF... This may take a moment."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload-and-process-pdf/", files=files)
                    if response.status_code == 200:
                        st.success("PDF processed successfully!")
                        st.session_state.pdf_processed = True
                        st.session_state.messages = [] # Clear previous messages
                        st.session_state.chat_history = [] # Clear chat history
                    else:
                        st.error(f"Error: {response.json().get('message', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

    if st.session_state.pdf_processed:
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
    if not st.session_state.pdf_processed:
        st.warning("Please upload and process a PDF first.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer from backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    payload = {
                        "question": prompt,
                        "chat_history": st.session_state.chat_history
                    }
                    response = requests.post(f"{BACKEND_URL}/ask/", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result.get("answer", "No answer found.")
                        sources = result.get("source_documents", [])
                        
                        st.markdown(answer)
                        
                        if sources:
                            with st.expander("View Sources"):
                                for source in sources:
                                    st.write(f"- Page {source.get('page', 'N/A')}")
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                        
                        # Update chat history
                        st.session_state.chat_history.append((prompt, answer))

                    else:
                        st.error(f"Error: {response.json().get('message', 'Unknown error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection error: {e}")

