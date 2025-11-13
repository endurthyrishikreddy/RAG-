from fastapi import FastAPI,UploadFile,File,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from data_ingestion.loader import Loader
from data_ingestion.preprocessor import TextPreprocessor
from embeddings.openai_embedder import OpenAIEmbedder
from embeddings.sentence_transformer import SentenceTransformerEmbedder
from vector_Store.faiss_Store import FaissStore
# from vector_Store.chromadb_store import ChromaDBStore
from openai import OpenAI
from google import genai
from dotenv import load_dotenv
import asyncio
from metadata.metadata_Store import MetadataStore
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.bm25_retrievers import BM25Retriever





app=FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#intilize modules
loader=Loader()
preprocessor=TextPreprocessor(chunk_size=400,overlap=100)
# OpenAIEmbedderembedder = OpenAIEmbedder(model_name="text-embedding-3-small")
sentence_embedder = SentenceTransformerEmbedder(model_name='all-MiniLM-L6-v2')
# faiss_store = FaissStore(dimension=384)
# chromadb_store = ChromaDBStore(persist_dir="chroma_db")

# Initialize OpenAI client
# openai_client = OpenAI()

# Initialize Gemini client
load_dotenv()  # Load environment variables from .env file

gemini_client = genai.Client()

vector_store = None
VECTOR_STORE_PATH = "vector_store/faiss_index"
store_lock = asyncio.Lock()  # To ensure thread-safe access to the vector store

metadata_store = MetadataStore()
bm25_retriever = BM25Retriever(text_chunks=[])
hybrid_retriever = None


@app.on_event("startup")
def load_faiss_index():
    """Load FAISS index once at startup if exists."""
    global vector_store, bm25_retriever, hybrid_retriever
    vector_store = FaissStore(dimension=384)

    if os.path.exists(VECTOR_STORE_PATH + ".index"):
        try:
            vector_store.load(VECTOR_STORE_PATH)
            print("✅ FAISS index loaded successfully at startup.")
        except Exception as e:
            print(f"⚠️ Failed to load FAISS index: {e}")
    else:
        print("ℹ️ No existing FAISS index found. Will create a new one.")

    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever, alpha=0.5)   

@app.get("/")
async def root():
    return {"message":"Welcome to RAG API"}

@app.post("/uploadfile/")
async def upload_file(file:UploadFile=File(...), background_tasks: BackgroundTasks =None):
    "upload a file and return the chunks"
    global vector_store
    #save the file
    file_location=os.path.join("uploads_files",file.filename)
    os.makedirs(loader.upload_dir, exist_ok=True)

    content = await file.read()

    with open(file_location,"wb") as f:
        f.write(content)
    #load the file
    raw_text = await asyncio.to_thread(loader.load_files, file_location)
    # raw_text=loader.load_files(file_location)
    #clean and chunk
    chunks = await asyncio.to_thread(preprocessor.chunk_text, raw_text)
    #embed the chunks
    # embeddings = OpenAIEmbedderembedder.embed(chunks)
    sentence_embeddings = await asyncio.to_thread(sentence_embedder.embed,chunks)

    #store the chunks and embeddings
    metadata = [{"source": file.filename, "chunk_index": i} for i in range(len(chunks))]
    # faiss_store.add(chunks, sentence_embeddings, metadata)
    # vector_store.add(chunks, sentence_embeddings, metadata)
    # vector_store.save(VECTOR_STORE_PATH)
    async with store_lock:
        start_idx = len(vector_store.texts)
        vector_store.add(chunks,sentence_embeddings, metadata)
        bm25_retriever.add_documents(chunks)
        end_idx = len(vector_store.texts)
        background_tasks.add_task(vector_store.save, VECTOR_STORE_PATH)

    # chromadb_store.add(chunks, sentence_embeddings, metadata)

    #save the stores
    # faiss_store.save("faiss_store/faiss_index")
    # faiss_store.save("C:/temp_faiss/faiss_index")

    doc_id = metadata_store.add_document(
        filename=file.filename,
        num_chunks=len(chunks),
        path=file_location,
        start_idx=start_idx,
        end_idx=end_idx,
    )

    return {
        "message": "✅ File processed and stored successfully.",
        "filename": file.filename,
        "doc_id": doc_id,
        "num_chunks": len(chunks),
    }

@app.get("/documents/")
async def list_documents():
    """List all uploaded documents and their metadata."""
    return metadata_store.list_documents()

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document metadata (does not remove from FAISS yet)."""
    metadata_store.delete_document(doc_id)
    return {"message": f"🗑️ Document {doc_id} metadata deleted."}

@app.get("/search/")
async def search_KB(query:str):
    "Query the FAISS store and return similar chunks"
    #embed the query
    #load the store if not already loaded"
    global vector_store
    if vector_store is None:
        vector_store = FaissStore(dimension=384)
        if os.path.exists(VECTOR_STORE_PATH + ".index"):
            try:
                vector_store.load(VECTOR_STORE_PATH)
                print("✅ FAISS index loaded successfully.")
            except Exception as e:
                return {"error": f"Failed to load FAISS store: {str(e)}"}
        else:
            return {"error": "No FAISS index found. Please upload a file first."}
    #embed the query   
    "search the KB for similar chunks"
    query_embedding = sentence_embedder.embed([query])[0]
    results, metadata = vector_store.search(query_embedding, top_k=1)
    return {
        "query": query,
        "results": results,
        "metadata": metadata
    }

# @app.get("/generate/")
# async def generate_response(query:str):
#     global vector_store
#     if vector_store is None:
#         vector_store = FaissStore(dimension=384)
#         if os.path.exists(VECTOR_STORE_PATH + ".index"):
#             try:
#                 vector_store.load(VECTOR_STORE_PATH)
#                 print("✅ FAISS index loaded successfully.")
#             except Exception as e:
#                 return {"error": f"Failed to load FAISS store: {str(e)}"}
#         else:
#             return {"error": "No FAISS index found. Please upload a file first."}
        
#     query_embedding = sentence_embedder.embed([query])[0]
#     results, metadata = vector_store.search(query_embedding, top_k=3)

#     context = "\n".join(results)
#     prompt = f"""
#     You are an intelligent assistant. Use the following context to answer the question.
#     If the context does not contain the answer, say "I'm not sure based on the provided data."

#     Context:
#     {context}

#     Question:
#     {query}

#     Answer:
#     """
#     completion = OpenAI.chats.generate_content(
#         model="gemini-2.5-flash",
#         messages=[
#             {"role": "system", "content": "You are a helpful AI assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.3,
#     )

#     answer = completion.choices[0].message.content.strip()

#     return {
#         "query": query,
#         "answer": answer,
#         "retrieved_context": results
#     }

@app.get("/generate_gemini/")
async def generate_response(query:str):
    global vector_store
    if vector_store is None:
        vector_store = FaissStore(dimension=384)
        if os.path.exists(VECTOR_STORE_PATH + ".index"):
            try:
                vector_store.load(VECTOR_STORE_PATH)
                print("✅ FAISS index loaded successfully.")
            except Exception as e:
                return {"error": f"Failed to load FAISS store: {str(e)}"}
        else:
            return {"error": "No FAISS index found. Please upload a file first."}
        
    # Assuming sentence_embedder is initialized correctly
    query_embedding = await asyncio.to_thread(sentence_embedder.embed,[query])
    query_embedding = query_embedding[0]  # Get the first (and only) embedding
    async with store_lock:
        results, metadata = await asyncio.to_thread(vector_store.search,query_embedding, top_k=3)

    context = "\n".join(results)
    prompt = f"""
    You are an intelligent assistant. Use the following context to answer the question.
    If the context does not contain the answer, say "I'm not sure based on the provided data."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    
    # --- CORRECTED GEMINI API CALL ---
    completion = await asyncio.to_thread(gemini_client.models.generate_content,
        model="gemini-2.5-flash",
        # 1. Use 'contents' instead of 'messages'
        contents=prompt, 
        # 2. Pass temperature and other parameters via 'config'
        config={"temperature": 0.3}
    )


    # 3. Access the response text via the '.text' property
    answer = completion.text.strip()

    return {
        "query": query,
        "answer": answer,
        "retrieved_context": results
    }


@app.get("/query/")
async def query_rag(query: str, mode: str = "hybrid"):
    global vector_store, bm25_retriever
    if vector_store is None or not vector_store.texts:
        return {"error": "No vector store or documents available. Please upload a file first."}

    query_emb = await asyncio.to_thread(sentence_embedder.embed, [query])
    query_emb = query_emb[0]

    async with store_lock:
        if mode == "vector":
            # Semantic search
            top_chunks, _ = await asyncio.to_thread(vector_store.search, query_emb, top_k=3)
        
        elif mode == "bm25":
            # Lexical search
            top_chunks, _ = await asyncio.to_thread(bm25_retriever.retrieve, query, top_k=3)
        
        elif mode == "hybrid":
            # Hybrid search
            top_chunks = await asyncio.to_thread(hybrid_retriever.retrieve, query_emb, query, top_k=3)
        
        else:
            return {"error": "Invalid mode. Choose vector, bm25, or hybrid."}

    context = "\n\n".join(top_chunks)
    prompt = f"""
    You are an intelligent assistant. Use the following context to answer the question.
    If the context does not contain the answer, say "I'm not sure based on the provided data."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    
    # --- CORRECTED GEMINI API CALL ---
    completion = await asyncio.to_thread(gemini_client.models.generate_content,
        model="gemini-2.5-flash",
        # 1. Use 'contents' instead of 'messages'
        contents=prompt, 
        # 2. Pass temperature and other parameters via 'config'
        config={"temperature": 0.3}
    )


    # 3. Access the response text via the '.text' property
    answer = completion.text.strip()

    return {
        "mode": mode,
        "query": query,
        "answer": answer,
        "retrieved_context": top_chunks
    }

