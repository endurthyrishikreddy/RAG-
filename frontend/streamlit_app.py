import streamlit as st
import requests
import os
import pandas as pd

# Define the base URL for the FastAPI backend
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Document Uploader", page_icon="📄", layout="wide")

st.title("📄 RAG Document Uploader")
st.caption("Upload your documents to build a knowledge base for retrieval-augmented generation (RAG).")

tabs = st.tabs(["Upload Document", "View Documents", "Search", "Question & Answer", "Hybrid Q&A"])

with tabs[0]:
    st.header("📤 Upload a Document")
    uploaded_file = st.file_uploader("Choose a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    
    if uploaded_file is not None:
        if st.button("Upload and Process"):
            with st.spinner("Processing document..."):
                # FIX 1 & 2: Include the MIME type and ensure the key is 'file'
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                # FIX 1: Use the correct FastAPI endpoint: /uploadfile/
                response = requests.post(f"{API_URL}/uploadfile/", files=files)
                
                # Use the result to determine the success message and details
                if response.status_code == 200:
                    result = response.json()
                    st.success("Document uploaded and processed successfully!")
                    st.write("**Filename:**", result.get("filename"))
                    st.write("**Document ID:**", result.get("doc_id"))
                    st.write("**Number of Chunks:**", result.get("num_chunks"))
                    # REMOVED: st.write("**File Path:**", result["path"]) <-- This key is missing
                else:
                    try:
                        error_data = response.json()
                        st.error(f"Upload failed (Status {response.status_code}): {error_data.get('detail', response.text)}")
                    except requests.exceptions.JSONDecodeError:
                         st.error(f"Upload failed (Status {response.status_code}): Could not decode response.")


with tabs[1]:
    st.header("📂 View Uploaded Documents")
    
    # Cache the API call to prevent continuous reloading
    @st.cache_data(show_spinner=False)
    def fetch_documents():
        return requests.get(f"{API_URL}/documents/")
        
    response = fetch_documents()
    
    if response.status_code == 200:
        documents = response.json()
        if documents:
            
            # Format documents into a DataFrame for clean display
            df = pd.DataFrame(documents)
            df = df.rename(columns={
                "id": "Document ID", 
                "filename": "Filename", 
                "num_chunks": "Chunks",
                "timestamp": "Uploaded At"
            })
            df = df[['Filename', 'Chunks', 'Document ID', 'Uploaded At']]
            
            st.dataframe(df, use_container_width=True)

            # Deletion UI (needs to trigger a full refresh to update the list)
            st.subheader("Delete Document")
            
            doc_to_delete = st.selectbox(
                "Select document to delete:", 
                options=[""] + list(df['Filename']),
                key='delete_select'
            )
            
            if st.button(f"Delete Selected Document", disabled=(doc_to_delete == "")):
                doc_id_to_delete = df[df['Filename'] == doc_to_delete]['Document ID'].iloc[0]
                
                delete_response = requests.delete(f"{API_URL}/documents/{doc_id_to_delete}")
                if delete_response.status_code == 200:
                    st.success(f"{doc_to_delete} deleted successfully! Refreshing list...")
                    st.cache_data.clear() # Clear cache to force list refresh
                    st.rerun()
                else:
                    st.error("Failed to delete document. Please try again.")

        else:
            st.info("No documents uploaded yet.")
    else:
        st.error("Failed to fetch documents from the API.")

with tabs[2]:
    st.header("🔍 Search the Knowledge Base")
    query = st.text_input("Enter your search query:", key='search_query')
    
    if st.button("Search KB"):
        if query:
            with st.spinner("Searching vector store..."):
                response = requests.get(f"{API_URL}/search/", params={"query": query})
            
            if response.status_code == 200:
                results = response.json()
                st.subheader("📚 Retrieved Chunks")
                
                if results["results"]:
                    for idx, (text, meta) in enumerate(zip(results["results"], results["metadata"])):
                        with st.expander(f"Chunk {idx + 1} (Source: {meta.get('source', 'N/A')} | Index: {meta.get('chunk_index', 'N/A')})"):
                            st.write(text)
                            st.markdown(f"**Metadata:** `{meta}`")
                else:
                    st.info("No relevant results found in the knowledge base.")
            else:
                st.error(f"Failed to perform search (Status {response.status_code}): {response.text}")
        else:
            st.warning("Please enter a search query.") 

            
# --- TAB 3: Gemini Direct QA ---
with tabs[3]:
    st.header("❓ Question & Answer (Gemini Direct)")
    
    question_gemini = st.text_input(
        "Ask a question based on your documents:", 
        key='qa_question_gemini'
    )

    if st.button("Get Answer (Gemini)", key="button_gemini_qna"):
        if not question_gemini.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                response = requests.get(f"{API_URL}/generate_gemini/", params={"query": question_gemini})
            
            if response.status_code == 200:
                data = response.json()
                st.markdown("### 🤖 **Answer:**")
                st.success(data.get("answer", "No answer found."))

                with st.expander("🔍 Retrieved Context"):
                    # Use the correct response key 'retrieved_context'
                    for i, chunk in enumerate(data.get("retrieved_context", [])):
                        st.markdown(f"**Chunk {i+1}:** {chunk}")
            else:
                st.error(f"❌ Error: {response.text}")

# --- TAB 4: Hybrid / Vector / BM25 QA ---
with tabs[4]:
    st.header("❓ Question & Answer (RAG Modes)")
    
    question_rag = st.text_input(
        "Ask a question based on your documents:", 
        key='qa_question_rag'
    )

    mode = st.selectbox(
        "Select retrieval mode:", 
        ["hybrid", "vector", "bm25"], 
        key="retrieval_mode_select"
    )

    if st.button("Get Answer (RAG)", key="button_rag_qna"):
        if not question_rag.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                params = {"query": question_rag, "mode": mode}
                response = requests.get(f"{API_URL}/query/", params=params)

            if response.status_code == 200:
                data = response.json()
                st.markdown("### 🤖 **Answer:**")
                st.success(data.get("answer", "No answer found."))

                with st.expander("🔍 Retrieved Context"):
                    for i, chunk in enumerate(data.get("retrieved_context", [])):
                        st.markdown(f"**Chunk {i+1}:** {chunk}")
                
                st.markdown(f"**Mode Used:** `{data.get('mode', 'unknown')}`")

            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
