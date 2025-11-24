# RAG Document Uploader & AI Chat Application

## About the App

This is a **Retrieval-Augmented Generation (RAG)** application that combines document management with AI-powered question answering. The app allows you to:

- **Upload and process documents** (PDF, DOCX, TXT) into a knowledge base
- **Search documents** using vector similarity and BM25 retrieval
- **Get answers** to questions based on your documents using:
  - Gemini AI with direct generation
  - Hybrid retrieval (combining vector and BM25 search)
  - Vector-only retrieval
  - BM25-only retrieval
- **Rerank results** using LLM-based reranking for better relevance
- **Chat with memory** - Maintain conversation history with context awareness
- **Voice input support** - Ask questions using voice with real-time transcription
- **WhatsApp-style UI** - Modern dark-mode chat interface with emoji support

### Key Features

- **Multi-format Support**: Process PDF, DOCX, and TXT files
- **Advanced Retrieval**: Hybrid search combining vector embeddings and keyword matching
- **Memory Management**: Persistent session management for multi-turn conversations
- **Metadata Filtering**: Filter results by document source
- **LLM Reranking**: Improve relevance using Gemini's reranking
- **Voice Input**: Speak to query your documents
- **Beautiful UI**: Streamlit-based frontend with WhatsApp-style chat bubbles

---

## Installation & Setup Instructions

### Step 1: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 2: Create `.env` File

Create a `.env` file in the project root directory and add your API keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
```

**How to get API keys:**
- **OpenAI API Key**: Visit [platform.openai.com](https://platform.openai.com) → API keys → Create new secret key
- **Gemini API Key**: Visit [ai.google.dev](https://ai.google.dev) → Get API key

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the FastAPI Backend (Uvicorn)

```bash
python main.py
```

This will start the FastAPI server on `http://127.0.0.1:8000`

The server will be running with auto-reload enabled. You should see output like:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Step 5: Run the Streamlit Frontend

Open a **new terminal** (keep the Uvicorn server running) and run:

```bash
streamlit run frontend/streamlit_app.py
```

This will start the Streamlit app and open it in your browser at `http://localhost:8501`

---

## Project Structure

```
RAG_MODULAR/
├── main.py                          # Entry point - Uvicorn server
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (API keys)
├── README.md                        # This file
│
├── app/
│   └── api.py                       # FastAPI application & endpoints
│
├── data_ingestion/
│   ├── loader.py                    # File loading (PDF, DOCX, TXT, CSV)
│   └── preprocessor.py              # Text preprocessing & chunking
│
├── embeddings/
│   ├── base_embedder.py             # Base embedding class
│   ├── openai_embedder.py           # OpenAI embeddings
│   └── sentence_transformer.py      # Sentence Transformers embeddings
│
├── vector_Store/
│   ├── base_store.py                # Base vector store interface
│   ├── faiss_Store.py               # FAISS vector store
│   └── chromdb_store.py             # ChromaDB vector store
│
├── retrievers/
│   ├── bm25_retrievers.py           # BM25 keyword search
│   └── hybrid_retriever.py          # Hybrid (vector + BM25) retrieval
│
├── rerank/
│   ├── reranker.py                  # Base reranker
│   └── llm_reranker.py              # Gemini LLM-based reranking
│
├── metadata/
│   ├── metadata_Store.py            # Metadata storage management
│   └── metadata.json                # Stored metadata
│
├── memory/
│   └── memory_manager.py            # Session & conversation memory
│
├── filters/
│   └── metadata_filter.py           # Metadata-based filtering
│
├── frontend/
│   └── streamlit_app.py             # Streamlit UI application
│
└── uploads_files/                   # Uploaded documents storage
```

---

## API Endpoints

The FastAPI backend provides the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/uploadfile/` | Upload and process a document |
| GET | `/documents/` | List all uploaded documents |
| DELETE | `/documents/{doc_id}` | Delete a document |
| GET | `/search/` | Search documents (vector search) |
| GET | `/query/` | Query with RAG (hybrid/vector/bm25 modes) |
| GET | `/generate_gemini/` | Get Gemini answer without RAG |
| POST | `/chat/` | Chat with memory & context |

---

## Usage Guide

### 1. Upload Documents

1. Go to the "Upload Document" tab
2. Choose a file (PDF, DOCX, or TXT)
3. Click "Upload and Process"
4. The document will be chunked and indexed

### 2. Search Documents

1. Go to the "Search" tab
2. Enter your search query
3. View retrieved chunks with metadata

### 3. Ask Questions (Direct Gemini)

1. Go to the "Question & Answer (Gemini Direct)" tab
2. Type your question
3. Get an answer with retrieved context

### 4. Ask Questions (RAG Modes)

1. Go to the "Question & Answer (RAG Modes)" tab
2. Select retrieval mode: `hybrid`, `vector`, or `bm25`
3. Type your question
4. Get an answer with the selected retrieval method

### 5. Chat with Memory

1. Go to the "WhatsApp-Style AI Chat" tab
2. Type or use voice input to ask questions
3. Select retrieval mode and enable reranking if desired
4. Chat history is maintained within the session
5. Click "Reset Chat" to start a new conversation

---

## Troubleshooting

### 1. "No module named 'openai'"
```bash
pip install openai
```

### 2. "Connection refused" (API not responding)
Make sure the FastAPI server is running:
```bash
python main.py
```

### 3. ".env file not found"
Ensure you've created the `.env` file in the project root with your API keys.

### 4. "Streamlit not found"
```bash
pip install streamlit
```

### 5. Port 8000 already in use
Change the port in `main.py`:
```python
uvicorn.run("app.api:app", host="127.0.0.1", port=8001, reload=True)
```

---

## Requirements

- Python 3.8+
- FastAPI & Uvicorn (Backend)
- Streamlit (Frontend)
- OpenAI API key
- Google Gemini API key

See `requirements.txt` for complete dependency list.

---

## Technologies Used

- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Vector Embeddings**: Sentence Transformers, OpenAI
- **Vector Store**: FAISS, ChromaDB
- **Retrieval**: BM25, Hybrid Search
- **LLM**: OpenAI GPT, Google Gemini
- **Memory**: In-memory session management
- **Voice**: Streamlit Mic Recorder

---

## License

This project is open source and available under the MIT License.

---

## Support

For issues or questions, please create an issue in the repository.

Happy exploring with your RAG application! 🚀
