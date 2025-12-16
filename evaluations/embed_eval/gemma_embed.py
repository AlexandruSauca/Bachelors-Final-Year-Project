import logging
import os
import json
import pathlib
import random
from time import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("google/embeddinggemma-300m")

INPUT_FILE = "processed_rag_data.jsonl"
OUTPUT_FILE = "embedded_gemma.jsonl"

def main():
    print(f"Reading from {INPUT_FILE}...")
    
    # 3. Read All Data
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
        
    data_buffer = []
    
    # 4. Process Loop
    with open(OUTPUT_FILE, 'w') as f_out:
        for line in tqdm(lines, desc="Embedding Chunks"):
            entry = json.loads(line)
            
            text_to_embed = entry['content']
            
            # --- THE CORE STEP ---
            # Encode the text into a vector
            # normalize_embeddings=True is usually recommended for Retrieval
            embedding = model.encode(text_to_embed, normalize_embeddings=True)
            
            # Add vector to the JSON object
            # We must convert numpy array to a standard Python list for JSON serialization
            entry['embedding'] = embedding.tolist()
            
            # Write immediately to output
            f_out.write(json.dumps(entry) + "\n")

    print(f"\nSuccess! Data with embeddings saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()