import streamlit as st
import requests
import time

# Define the future URL of our FastAPI backend
BACKEND_URL = "http://localhost:8000"

# ==========================================
# API CLIENT FUNCTIONS (Mocked for now)
# ==========================================
def api_upload_document(uploaded_file):
    """Sends a POST request to FastAPI to ingest the file."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    try:
        response = requests.post(f"{BACKEND_URL}/ingest", files=files)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        st.error("Backend is offline! Is FastAPI running?")
        return False

def api_ask_question(query):
    """Sends a POST request to FastAPI to run the RAG pipeline."""
    payload = {"query": query}
    try:
        response = requests.post(f"{BACKEND_URL}/ask", json=payload)
        if response.status_code == 200:
            return response.json().get("answer", "No answer found.")
        return f"Error: Backend returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not reach backend. Is FastAPI running?"
# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
# Track if the system is ready (file uploaded) and store chat history
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set up the page layout
st.set_page_config(page_title="Thesis RAG Pipeline", page_icon="🔬", layout="wide")

# ==========================================
# VIEW 1: THE LANDING PAGE (Pre-Upload)
# ==========================================
if not st.session_state.is_ready:
    # Center the upload interface using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("GUS.ai - Your Research Assistant")
        st.markdown("### Welcome! Please upload a foundational document to begin.")
        st.write("The system needs a document to build the initial vector space before chatting.")
        
        # Big central uploader
        uploaded_file = st.file_uploader("Upload PDF or Text file", type=["pdf", "txt"], label_visibility="hidden")
        
        if st.button("Ingest & Start Chatting", use_container_width=True):
            if uploaded_file is not None:
                with st.spinner('Sending to backend for chunking and embedding...'):
                    success = api_upload_document(uploaded_file)
                    if success:
                        # Change the state and reload the page
                        st.session_state.is_ready = True
                        st.rerun()
                    else:
                        st.error("Backend error during ingestion.")
            else:
                st.warning("Please select a file first.")

# ==========================================
# VIEW 2: THE CHAT INTERFACE (Post-Upload)
# ==========================================
else:
    st.title("GUS.ai - Your Research Assistant")
    
    # --- SIDEBAR: Change File Option ---
    with st.sidebar:
        st.header("Change Context")
        st.write("Upload a new file to overwrite the current vector database.")
        
        new_file = st.file_uploader("Replace current document", type=["pdf", "txt"])
        
        if st.button("Update Vector DB"):
            if new_file is not None:
                with st.spinner('Processing new document...'):
                    success = api_upload_document(new_file)
                    if success:
                        st.success(f"'{new_file.name}' ingested!")
                        # Optional: Clear old chat history when a new file is uploaded
                        st.session_state.messages = []
                        st.rerun()
            else:
                st.warning("Select a new file to update.")
                
        # Optional: A button to totally reset the app back to the landing page
        st.divider()
        if st.button("Reset Entire Session"):
            st.session_state.is_ready = False
            st.session_state.messages = []
            st.rerun()

    # --- MAIN CHAT WINDOW ---
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        
        # 1. Display User Message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Display Assistant Loading State & Call "API"
        with st.chat_message("assistant"):
            with st.spinner("Fetching hybrid search results and generating answer..."):
                answer = api_ask_question(prompt)
                st.markdown(answer)
                
        st.session_state.messages.append({"role": "assistant", "content": answer})