import json
import os
import time
import random
import google.generativeai as genai
from tqdm import tqdm

os.environ["GOOGLE_API_KEY"] = "your_google_api_key_here"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

MODEL_NAME = 'gemini-2.5-flash'

INPUT_FILE = "processed_rag_data.jsonl"
OUTPUT_DIR = "./synthetic_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- PROMPTS ---

PROMPT_TEXT = """
You are an expert researcher creating a test exam for a retrieval system.
Read the following text chunk from a research paper:
"{content}"

Task: Write ONE specific, technical question that can be answered *only* by this text. 
The question should be self-contained (do not say "this text" or "the author").
Output ONLY the question.
"""

PROMPT_VISUAL = """
You are analyzing a research paper. 
Here is a description of a {type} (Table/Image):
"{content}"

Task: Write ONE specific question that a researcher would ask to find this specific data.
Focus on the specific metrics, trends, or model names mentioned.
Output ONLY the question.
"""

def generate_question(model, content, chunk_type):
    """Sends the content to Gemini and gets a question back."""
    try:
        if chunk_type == "text":
            prompt = PROMPT_TEXT.format(content=content)
        else:
            prompt = PROMPT_VISUAL.format(type=chunk_type, content=content)
            
        response = model.generate_content(prompt)
        
        if not response.text:
            return None
            
        # Clean up: remove quotes or "Question:" prefixes if the model adds them
        question = response.text.strip().replace('"', '').replace("Question:", "").strip()
        return question
    except Exception as e:
        # Rate limit or API error handling
        print(f"Warning: {e}")
        return None
    
def main():
    # 1. Load the Model
    print(f"Loading {MODEL_NAME}...")
    model = genai.GenerativeModel(MODEL_NAME)
    
    # 2. Prepare Output Files
    queries_path = os.path.join(OUTPUT_DIR, "queries.jsonl")
    qrels_path = os.path.join(OUTPUT_DIR, "qrels.tsv") 
    corpus_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")

    # 3. Process Loop
    print(f"Reading from {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r') as f_in, \
         open(queries_path, 'w') as f_queries, \
         open(corpus_path, 'w') as f_corpus, \
         open(qrels_path, 'w') as f_qrels:
        
        # Write header for TSV (BEIR Format)
        f_qrels.write("query-id\tcorpus-id\tscore\n")
        
        lines = f_in.readlines()
        total_generated = 0
        
        for line in tqdm(lines, desc="Processing Chunks"):
            data = json.loads(line)
            
            chunk_id = data['id']
            content = data['content']
            chunk_type = data['type']
            
            # --- A. ALWAYS Save to Corpus (Required for Evaluation) ---
            # Even if we don't generate a question for it, the chunk must be in the DB
            corpus_entry = {
                "_id": chunk_id,
                "text": content,
                "title": data.get("source", ""),
                "metadata": {"type": chunk_type}
            }
            f_corpus.write(json.dumps(corpus_entry) + "\n")
            
            # --- B. Smart Sampling Logic ---
            should_generate = False
            
            # Rule 1: Always keep Tables and Images (The "Gold" data)
            if chunk_type in ["table", "image"]:
                should_generate = True
                
            # Rule 2: Randomly sample Text (The "Filler" data)
            # We keep ~10% of text chunks to ensure balanced evaluation
            elif chunk_type == "text":
                if len(content) > 150: # Only check if text is long enough
                    if random.random() < 0.10: # 10% probability
                        should_generate = True
            
            # Skip if sampling said "No"
            if not should_generate:
                continue

            # --- C. Generate Question ---
            # Simple rate limit prevention
            time.sleep(0.5) 
            
            question_text = generate_question(model, content, chunk_type)
            
            if question_text:
                total_generated += 1
                # Create a unique query ID
                query_id = f"q_{chunk_id}"
                
                # Save Query
                query_entry = {
                    "_id": query_id,
                    "text": question_text,
                    "metadata": {"generated_from": chunk_id}
                }
                f_queries.write(json.dumps(query_entry) + "\n")
                
                # Save QREL (Ground Truth mapping)
                # Format: query_id TAB corpus_id TAB score
                f_qrels.write(f"{query_id}\t{chunk_id}\t1\n")

    print(f"\n--- DONE ---")
    print(f"Total Questions Generated: {total_generated}")
    print(f"Datasets saved to: {OUTPUT_DIR}")
    print("You can now run your BEIR evaluation script.")

if __name__ == "__main__":
    main()