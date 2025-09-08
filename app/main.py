from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple
import os
from dotenv import load_dotenv
import shutil
from app.data_ingestion import load_and_chunk_documents
from app.vector_store import create_vector_store
from app.qa_pipeline import create_qa_pipeline

# Load environment variables
load_dotenv()

app = FastAPI()

# In-memory storage
vector_store = None
qa_pipeline = None
chat_history = []

class AskRequest(BaseModel):
    question: str
    chat_history: List[Tuple[str, str]]

@app.post("/upload-and-process-pdf/")
async def upload_and_process_pdf(file: UploadFile = File(...)):
    """
    Uploads a PDF, processes it, and creates a QA pipeline.
    """
    global vector_store, qa_pipeline, chat_history
    
    if not file.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"message": "Please upload a PDF file."})

    # Reset chat history
    chat_history = []

    # Save the uploaded file temporarily
    file_path = os.path.join("data", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Process the document
        chunks = load_and_chunk_documents(file_path)
        vector_store = create_vector_store(chunks)
        qa_pipeline = create_qa_pipeline(vector_store)
        
        return {"message": f"Successfully processed {file.filename} and created QA pipeline."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})
    finally:
        # Clean up the saved file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/ask/")
async def ask_question(request: AskRequest):
    """
    Receives a question and chat history, returns an answer from the QA pipeline.
    """
    global chat_history
    if not qa_pipeline:
        return JSONResponse(status_code=400, content={"message": "QA pipeline not initialized. Please upload a document first."})

    try:
        result = qa_pipeline({"question": request.question, "chat_history": request.chat_history})
        
        # Update server-side chat history
        chat_history.append((request.question, result["answer"]))
        
        return {
            "answer": result["answer"],
            "source_documents": [doc.metadata for doc in result["source_documents"]]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

@app.get("/")
def read_root():
    return {"Hello": "World"}

