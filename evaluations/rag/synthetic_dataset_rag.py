import psycopg2
import json
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

DB_HOST = "localhost" 
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-2.5-flash"
number_samples = 50
output_file = "test_dataset_rag.json"

def get_random_chunks(limit):
    conn = psycopg2.connect(
        host = DB_HOST,
        database = DB_NAME,
        user = DB_USER,
        password = DB_PASSWORD,
        port = DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute("SELECT id, content FROM chunks ORDER BY RANDOM() LIMIT %s", (limit,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def generate_synthetic_questions(chunk_text):
    prompt = f"""You are an expert evaluator building a test dataset for a technical search engine. 
Read the following text chunk from a scientific machine learning paper. 
Write ONE specific, realistic technical question that a user would ask where this EXACT text is the perfect answer.

RULES:
1. DO NOT use the exact same nouns or specific jargon found in the text. Paraphrase heavily.
2. Formulate it as a question a developer or researcher would ask.
3. If the chunk does not contain enough coherent information to form a good question (e.g., it is just an equation or a single fragmented sentence), simply reply with the exact word: SKIP.
4. Output ONLY the question itself, or the word SKIP. Do not include any introductory text.

--- EXAMPLES ---

Example 1:
Text Chunk: "We employ three types of regularization during training: residual dropout, label smoothing, and weight decay."
Output: What techniques were used to prevent the model from overfitting?

Example 2:
Text Chunk: "Summary: The table compares different parsers on the WSJ 23 dataset. Raw Table: 24.5 25.6 27.8"
Output: SKIP

--- YOUR TURN ---

Text Chunk:
{chunk_text}

Instruction: Output ONLY the question itself, or the word SKIP. Do not include any introductory text.
Output:"""
    
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Wrong API key: {e}")
        return "SKIP"
    
def main():
    print(f"Fetching {number_samples} random chunks from the database...")
    rows = get_random_chunks(number_samples)
    dataset= []
    skipped_count = 0
    for row in rows:
        chunk_id = row[0]
        chunk_text = row[1]
        generate_questions = generate_synthetic_questions(chunk_text)
        if generate_questions == "SKIP":
            skipped_count += 1
            continue
        dataset.append({
            "question": generate_questions,
            "target_chunk_id": chunk_id
        })
        print(f"Question generated for chunk ID {chunk_id}")
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"\n Saved {len(dataset)} question-chunk pairs to {output_file}")
    print(f"Skipped {skipped_count} chunks.")

if __name__ == "__main__":
    main()