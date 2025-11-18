import streamlit as st
import requests
import os
import pandas as pd
import uuid

# Define the base URL for the FastAPI backend
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Document Uploader", page_icon="📄", layout="wide")

st.title("📄 RAG Document Uploader")
st.caption("Upload your documents to build a knowledge base for retrieval-augmented generation (RAG).")

tabs = st.tabs(["Upload Document", "View Documents", "Search", "Question & Answer", "Hybrid Q&A","Chat with Memory","proper UI"])

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


# with tabs[5]:
#     st.header("💬 Chat with Memory")

#     # persist session ID in Streamlit (local memory)
#     if 'session_id' not in st.session_state:
#         st.session_state['session_id'] = str(uuid.uuid4())

#     st.write(f"**Session ID:** `{st.session_state['session_id']}`")

#     #chat history storage
#     if 'chat_history' not in st.session_state:
#         st.session_state['chat_history'] = []

#     #display chat history
#     for entry in st.session_state['chat_history']:
#         role = entry['role']
#         content = entry['content']
#         if role == 'user':
#             st.markdown(f"**You:** {content}")
#         else:
#             st.markdown(f"**Bot:** {content}")  

#     st.divider()

#     user_msg = st.text_input("Type your message:", key='chat_input')

#     col1, col2 ,col3 = st.columns(3)

#     with col1:
#         mode = st.selectbox("Select retrieval mode:", ["hybrid", "vector", "bm25"], key="chat_retrieval_mode")
#     with col2:
#         rerank = st.checkbox("Enable Reranking", key="chat_rerank", value=True)    
#     with col3:
#         filter_source =st.text_input("Filter by source (optional):", key="chat_filter_source")


#     if st.button("Send", key="chat_send_button"): 
#         if user_msg.strip():
#             st.session_state.chat_history.append({"role": "user", "content": user_msg})

#             params = {
#                 "query": user_msg,
#                 "session_id": st.session_state['session_id'],
#                 "mode": mode,
#                 "rerank": rerank,
#                 "filter_source": filter_source if filter_source.strip() else None,
#             }

#             res = requests.get(f"{API_URL}/chat/", params=params)

#             if res.status_code == 200:
#                 data = res.json()
#                 answer = data["answer"]
#                 st.session_state.chat_history.append({"role": "assistant", "content": answer})
#                 st.rerun()

#             else:
#                 st.error(f"Error: {res.text}")
#         else:
#             st.warning("Please enter a message")   
# 



# # --------------- TAB 5 — WhatsApp Style Chat with Memory ---------------------
# with tabs[5]:
#     st.header("💬 WhatsApp-Style Chat (Gemini + Memory)")

#     # persistent session
#     if "session_id" not in st.session_state:
#         st.session_state.session_id = str(uuid.uuid4())

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     st.write(f"Session ID: `{st.session_state.session_id}`")

#     # --- Custom CSS for WhatsApp bubbles ---
#     chat_css = """
#     <style>
#     .user-bubble {
#         background-color: #DCF8C6;
#         color: #000;
#         padding: 10px 14px;
#         border-radius: 15px;
#         max-width: 70%;
#         margin-left: auto;
#         margin-bottom: 10px;
#         font-size: 16px;
#         border: 1px solid #cbd1c5;
#     }
#     .assistant-bubble {
#         background-color: #ECECEC;
#         color: #000;
#         padding: 10px 14px;
#         border-radius: 15px;
#         max-width: 70%;
#         margin-right: auto;
#         margin-bottom: 10px;
#         font-size: 16px;
#         border: 1px solid #d3d3d3;
#     }
#     .chat-container {
#         height: 500px;
#         overflow-y: scroll;
#         padding-right: 10px;
#         border: 1px solid #dcdcdc;
#         border-radius: 10px;
#         background: #fafafa;
#         padding: 15px;
#     }
#     </style>
#     """
#     st.markdown(chat_css, unsafe_allow_html=True)

#     # --- Chat Display Container ---
#     with st.container():
#         st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

#         for msg in st.session_state.chat_history:
#             if msg["role"] == "user":
#                 st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

#         st.markdown("</div>", unsafe_allow_html=True)

#     # --- User input ---
#     user_msg = st.text_input("Type a message...")

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         mode = st.selectbox("Mode", ["hybrid", "vector", "bm25"])
#     with col2:
#         rerank = st.checkbox("Rerank", value=True)
#     with col3:
#         filter_source = st.text_input("Filter source (optional)")

#     if st.button("Send"):
#         if user_msg.strip():

#             # Add user bubble
#             st.session_state.chat_history.append({
#                 "role": "user",
#                 "content": user_msg
#             })

#             params = {
#                 "query": user_msg,
#                 "session_id": st.session_state.session_id,
#                 "mode": mode,
#                 "rerank": rerank,
#                 "filter_source": filter_source if filter_source else None
#             }

#             res = requests.post(f"{API_URL}/chat/", params=params)

#             if res.status_code == 200:
#                 data = res.json()
#                 answer = data["answer"]

#                 # Add assistant bubble
#                 st.session_state.chat_history.append({
#                     "role": "assistant",
#                     "content": answer
#                 })

#                 st.rerun()

#             else:
#                 st.error(f"API Error: {res.text}")

#         else:
#             st.warning("Enter a message before sending.")


# ---------------- TAB 5 — ADVANCED WHATSAPP STYLE CHAT -------------------------
with tabs[6]:
    import uuid
    from datetime import datetime
    from streamlit_mic_recorder import mic_recorder, speech_to_text
    import emoji

    st.header("💬 WhatsApp-Style AI Chat (Dark Mode + Memory + Gemini)")

    # --------------- SESSION STATE -------------------
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "assistant_typing" not in st.session_state:
        st.session_state.assistant_typing = False

    # Reset Chat Button
    if st.button("🗑 Reset Chat"):
        st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.success("Chat reset!")
        st.stop()

    # ----------- DARK MODE WHATSAPP CSS -------------
    dark_css = """
    <style>

    body {
        background-color: #111 !important;
    }

    .chat-container {
        height: 520px;
        overflow-y: auto;
        padding: 15px;
        border-radius: 12px;
        background-color: #0b141a;
        border: 1px solid #233138;
    }

    .user-bubble {
        background-color: #005c4b;
        color: white;
        padding: 12px 15px;
        border-radius: 14px;
        margin-left: auto;
        max-width: 65%;
        margin-bottom: 12px;
        animation: fadeIn 0.4s ease-in-out;
    }

    .assistant-bubble {
        background-color: #202c33;
        color: white;
        padding: 12px 15px;
        border-radius: 14px;
        margin-right: auto;
        max-width: 65%;
        margin-bottom: 12px;
        animation: fadeIn 0.4s ease-in-out;
    }

    /* Profile icons */
    .avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        vertical-align: middle;
        margin-right: 8px;
    }

    /* Timestamp */
    .timestamp {
        font-size: 11px;
        color: #aaaaaa;
        margin-top: 4px;
        text-align: right;
    }

    /* Typing indicator */
    .typing-dot {
        height: 8px;
        width: 8px;
        background-color: #aaa;
        border-radius: 50%;
        display: inline-block;
        margin-right: 4px;
        animation: blink 1.4s infinite both;
    }

    @keyframes blink {
        0% { opacity: .2; }
        20% { opacity: 1; }
        100% { opacity: .2; }
    }

    /* Fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    </style>
    """

    st.markdown(dark_css, unsafe_allow_html=True)

    # ---------------- CHAT VIEW ----------------------
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        bubble = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        avatar = (
            "https://i.imgur.com/KXhYQ.png" if msg["role"] == "user"
            else "https://i.imgur.com/0hQZ4.png"
        )
        st.markdown(
            f"""
            <div class="{bubble}">
                <img src="{avatar}" class="avatar"/>
                {msg["content"]}
                <div class="timestamp">{msg["time"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Typing indicator
    if st.session_state.assistant_typing:
        st.markdown(
            """
            <div class='assistant-bubble'>
              <span class="typing-dot"></span>
              <span class="typing-dot"></span>
              <span class="typing-dot"></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- USER INPUT AREA ---------------------
    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        user_msg = st.text_input("Type your message", key="chat_input")

    with col2:
        emoji_btn = st.button("😊")
        if emoji_btn:
            st.session_state.chat_input = st.session_state.chat_input + emoji.emojize(":smile:")
            st.rerun()

    with col3:
        audio = mic_recorder(
            start_prompt="🎤",
            stop_prompt="🛑",
            just_once=True,
            use_container_width=True,
            key="voice_input"
        )

        if audio:
            text = speech_to_text(audio["bytes"])
            st.session_state.chat_input = text
            st.rerun()

    # retrieval options
    mode = st.selectbox("Mode", ["hybrid", "vector", "bm25"])
    rerank = st.checkbox("Rerank", value=True)
    filter_source = st.text_input("Metadata filter (optional)")

    # ---------------- SEND LOGIC --------------------
    if st.button("Send"):
        if user_msg.strip():

            now = datetime.now().strftime("%I:%M %p")

            # Add user bubble
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_msg,
                "time": now
            })

            st.session_state.assistant_typing = True
            st.rerun()

            # --- API CALL ---
            params = {
                "query": user_msg,
                "session_id": st.session_state.session_id,
                "mode": mode,
                "rerank": rerank,
                "filter_source": filter_source if filter_source else None
            }

            res = requests.post(f"{API_URL}/chat/", params=params)

            st.session_state.assistant_typing = False

            if res.status_code == 200:
                data = res.json()
                answer = data["answer"]

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "time": datetime.now().strftime("%I:%M %p")
                })

                st.rerun()

            else:
                st.error(f"API Error: {res.text}")



