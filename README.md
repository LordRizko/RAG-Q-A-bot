<div align="center">

# 📄 RAG Q&A Bot

Ask questions about the contents of a PDF using a Retrieval-Augmented Generation (RAG) pipeline. 

FastAPI backend + Chroma vector store + HuggingFace embeddings + OpenRouter (ChatOpenAI compatible) LLM + Streamlit frontend.

</div>

---

## 🚀 Features
- Upload a PDF and automatically chunk & index it (LangChain + `RecursiveCharacterTextSplitter`).
- Vector similarity search using Chroma + `sentence-transformers/all-MiniLM-L6-v2` embeddings (local, no remote call).
- Conversational Q&A with chat history (LangChain `ConversationalRetrievalChain`).
- Source transparency: shows page numbers for retrieved chunks.
- Lightweight: no external DB required (in‑memory Chroma instance).

## 🗂 Project Structure
```
app/
	main.py              # FastAPI app + endpoints
	data_ingestion.py    # PDF loading & chunking
	vector_store.py      # Embedding + Chroma creation
	qa_pipeline.py       # Conversational retrieval chain
frontend/
	streamlit_app.py     # UI for upload + chat
data/                  # Temporary PDF storage (cleaned after processing)
requirements.txt       # Python dependencies
```

## ⚙️ Requirements
- Python 3.11+ (3.13 visible from pyc; ensure libs support your version)
- An OpenRouter API key (acts like an OpenAI-compatible endpoint)
- (Windows) PowerShell or CMD

## 🔐 Environment Variables
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_key_here
# Optional override (default http://127.0.0.1:8000)
BACKEND_URL=http://127.0.0.1:8000
```

Grab a key at: https://openrouter.ai/

## 🧪 Quick Start
```powershell
# 1. (Optional) Create & activate a virtual env
python -m venv .venv
.\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run backend (FastAPI)
uvicorn app.main:app --reload

# 4. In a second terminal run Streamlit frontend
streamlit run frontend/streamlit_app.py
```

Open the Streamlit app URL (shown in console, usually http://localhost:8501) and:
1. Upload a PDF in the sidebar.
2. Wait for processing success message.
3. Ask questions in the chat box.
4. Expand “View Sources” to see page references.

## 🧵 API Endpoints (Backend)
| Method | Path | Description |
|--------|------|-------------|
| GET    | `/`                         | Health test (Hello World) |
| POST   | `/upload-and-process-pdf/`  | Upload one PDF, build vector store + QA pipeline |
| POST   | `/ask/`                     | Body: `{ question: str, chat_history: [[q,a], ...] }` → answer + sources |

Sample ask payload:
```json
{
	"question": "What is the main topic?",
	"chat_history": [["Previous question", "Previous answer"]]
}
```

## 🧠 How It Works
1. PDF → pages via `PyPDFLoader`.
2. Pages → overlapping chunks (500 chars, 50 overlap).
3. Chunks embedded with MiniLM → stored in transient Chroma index.
4. Each question: retrieve similar chunks → pass (with chat history) to LLM.
5. Response + source metadata returned to UI.

## ❗ Limitations
- Single-document, in-memory session (uploading a new PDF resets chat & index).
- No persistence across restarts (Chroma not saved to disk here).
- Basic source display (only page metadata).
- No auth / rate limiting.

## 🛣 Possible Improvements
- Multi-file ingestion & persistent vector store (Chroma persist_directory or another DB).
- Add model selection & temperature controls in UI.
- Caching layer for repeated questions.
- Better source snippet display (highlight text extracts).
- Dockerfile + Compose for reproducible deployment.
- Add tests (unit for ingestion, mock LLM for QA chain).

## 🧪 Minimal Test Idea (Future)
- Verify `/upload-and-process-pdf/` with a tiny synthetic PDF returns 200.
- Mock vector store + ensure `/ask/` rejects before upload (400) and answers after (200).

## 🧾 License
Add a license file (e.g., MIT) if you intend to share publicly.

## 🙋 Troubleshooting
| Issue | Fix |
|-------|-----|
| `OPENROUTER_API_KEY not found` | Create `.env`, restart terminal, ensure `python-dotenv` loads (it does in `main.py`). |
| Frontend can’t reach backend | Confirm backend runs on 127.0.0.1:8000; set BACKEND_URL accordingly. |
| Slow first answer | Model cold start + embedding build time; subsequent queries faster. |
| Unicode / PDF parsing oddities | Try re-saving PDF or use different extractor model. |

## 🤝 Contributing
Feel free to open issues / PRs: tests, Dockerization, multi-doc support.

---

Happy querying! 🎯

