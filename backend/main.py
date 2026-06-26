from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import shutil
import os
import psycopg2
from dotenv import load_dotenv 
from streamlit import cursor
from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display, Markdown
from ingest_postgres import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from prompts import prompt_images, prompt_text
from qwen_vl_utils import process_vision_info
from load_qwen2vl2B import model, processor
from sentence_transformers import SentenceTransformer
import json
import re
from typing import List, Dict
from ingest_postgres import ingest_data
from retrieve import hybrid_search
from grounding_eval import generator_response, condense_query
# ==========================================
# APP INITIALIZATION & CORS SETUP
# ==========================================
app = FastAPI(title="Thesis RAG API", description="Backend for Hybrid Search and LLM Generation")

# We must enable CORS so your Streamlit frontend (running on port 8501) 
# is allowed to talk to this FastAPI backend (running on port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# DATA MODELS
# ==========================================
class QueryRequest(BaseModel):
    query: str
    history: List[Dict[str, str]]  = []

# ==========================================
# ENDPOINT 1: DOCUMENT INGESTION
# ==========================================
@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    try:
        # 1. Save the uploaded file temporarily to disk
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"Received file: {file.filename}. Starting ingestion...")

        # ---------------------------------------------------------
        #INGESTION POSTGRES

        ingest_data(temp_file_path)
        # ---------------------------------------------------------
        
        # Mock ingestion delay
        time.sleep(2) 
        
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
        return {"message": "Document successfully ingested and vectorized.", "filename": file.filename}
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ENDPOINT 2: RAG QUERY (HYBRID SEARCH + LLM)
# ==========================================
@app.post("/ask")
async def ask_question(request: QueryRequest):
    user_query = request.query
    print(f"Received query: {user_query}")
    
    try:
        # ---------------------------------------------------------
        # RETREVAL + GENERATION

        chat_history = request.history
        standalone_query = condense_query(user_query, chat_history)

        results = hybrid_search(standalone_query)

        if results:
            print(f"\n[DEBUG] Sample DB Result: {results[0]}\n")
        filtered_chunks = []
        print(f"\n--- SCORES FOR: {standalone_query} ---")
        for i, result in enumerate(results):
            chunk_text = result[0][1] 
            score = float(result[1]) 
            print(f"Chunk {i} Score: {score}")
            
            # Temporarily accept ALL chunks so the system doesn't crash
            filtered_chunks.append(chunk_text)
        print("-----------------------\n")

        if not filtered_chunks:
            print("Guardrail triggered! All chunks were filtered out as noise.")
            actual_output = "I'm sorry, but the provided document does not contain information to answer this question."
        else:
            actual_output = generator_response(standalone_query, filtered_chunks)
        
        if not actual_output or actual_output.strip() == "":
            actual_output = "The model failed to generate a response."
            
        print(f"\n[Qwen Answer Preview]: {actual_output[:100]}...\n")
        # ---------------------------------------------------------
        
        # Mock RAG process
        time.sleep(1.5)
        #generated_answer = f"This is the official backend response to: '{user_query}'. I have successfully processed your request through the API!"
        
        return {"answer": actual_output}
        
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer.")

# ==========================================
# SERVER RUNNER (For local testing)
# ==========================================
if __name__ == "__main__":
    import uvicorn
    # Runs the server on http://localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)