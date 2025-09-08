from langchain.chains import ConversationalRetrievalChain
from langchain_openai.chat_models import ChatOpenAI
import os

def create_qa_pipeline(vectorstore):
    """
    Creates a Question-Answering pipeline using an OpenRouter model with conversational memory.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

    llm = ChatOpenAI(
        model_name="mistralai/mistral-7b-instruct", # A reliable free model on OpenRouter
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
