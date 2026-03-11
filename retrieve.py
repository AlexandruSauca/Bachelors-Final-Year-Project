import os
import psycopg2
from dotenv import load_dotenv 
from load_models.load_qwen2vl2B import model, processor
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import bm25s
from google import genai
from google.genai.types import GenerateContentConfig

query = 'How does self-attention work in the Transformer architecture?'
query2 = 'scaled dot product self attention Q K V transformer'

load_dotenv()
DB_HOST = "localhost" 
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("DB_PORT")


embed_model = SentenceTransformer("google/embeddinggemma-300m")
query_embedding = embed_model.encode(query)
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = "gemini-2.5-flash"


def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST, 
        database=DB_NAME, 
        user=DB_USER, 
        password=DB_PASSWORD, 
        port=DB_PORT
    )
    return conn

def dense_retrieval(conn, query_embedding, top_k=10):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, content, metadata 
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
                   """, (str(query_embedding), top_k))
    results = cursor.fetchall()
    cursor.close()
    return results

def sparse_retrieval(conn, query, top_k = 10):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, content, metadata
        FROM chunks, plainto_tsquery('english', %s) query
        WHERE to_tsvector('english', content) @@ query
        ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC
        LIMIT %s
                   """, (query, top_k))
    results = cursor.fetchall()
    cursor.close()  
    return results

def rrf(dense_results, sparse_results, k=25):
    scores = {}
    for rank, row in enumerate(dense_results):
        doc_id = row[0]
        scores.setdefault(doc_id, 0)
        scores[doc_id] += 1/(k + rank + 1)

    for rank, row in enumerate(sparse_results):
        doc_id = row[0]
        scores.setdefault(doc_id, 0)
        scores[doc_id] += 1/(k + rank + 1)

    ranked_ids = sorted(scores, key = scores.get, reverse=True)

    merged = []
    lookup = {r[0]: r for r in dense_results + sparse_results}
    for doc_id in ranked_ids:
        merged.append(lookup[doc_id])

    return merged

def rerank_results(query, results, top_k=5):
    pairs = [(query, r[1]) for r in results]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(results, scores),
                    key=lambda x: x[1],
                    reverse=True)

    return [(r[0], r[1]) for r in ranked[:top_k]]  #r[0] is the database row and r[1] is the relevance score

def generate_hyde_document(query):
    prompt = f"""You are an expert researcher and the lead author of a highly technical scientific paper.
Write a short, dense, single-paragraph passage that directly answers the user's question.
Your response MUST use the exact academic terminology, tone, and formal style found in peer-reviewed journals.
Do NOT write introductory filler (e.g., "Here is the answer" or "According to the paper"). Output ONLY the hypothetical passage.

--- EXAMPLES ---

Example 1 (Machine Learning):
Question: What techniques were used to prevent the model from overfitting?
Passage: We employ three types of regularization during training: residual dropout, label smoothing, and weight decay. We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized. We also apply dropout to the sums of the embeddings and the positional encodings.

Example 2 (Biology/Medicine):
Question: How does the drug affect cellular apoptosis in the treated samples?
Passage: Flow cytometry analysis revealed a significant dose-dependent increase in early apoptotic cells following 48 hours of treatment. The compound induced the cleavage of caspase-3 and PARP, confirming the activation of the intrinsic apoptotic pathway, while control cultures exhibited baseline levels of programmed cell death.

--- YOUR TURN ---

Question: {query}
Passage:"""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], return_tensors="pt", padding=True
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=200, temperature = 0.7)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()

def generate_hyde_gemini(query):
    prompt = f"""You are an expert researcher and the lead author of a highly technical scientific paper.
Write a short, dense, single-paragraph passage that directly answers the user's question.
Your response MUST use the exact academic terminology, tone, and formal style found in peer-reviewed journals.
Do NOT write introductory filler (e.g., "Here is the answer" or "According to the paper"). Output ONLY the hypothetical passage.

--- EXAMPLES ---

Example 1 (Machine Learning):
Question: What techniques were used to prevent the model from overfitting?
Passage: We employ three types of regularization during training: residual dropout, label smoothing, and weight decay. We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized. We also apply dropout to the sums of the embeddings and the positional encodings.

Example 2 (Biology/Medicine):
Question: How does the drug affect cellular apoptosis in the treated samples?
Passage: Flow cytometry analysis revealed a significant dose-dependent increase in early apoptotic cells following 48 hours of treatment. The compound induced the cleavage of caspase-3 and PARP, confirming the activation of the intrinsic apoptotic pathway, while control cultures exhibited baseline levels of programmed cell death.

--- YOUR TURN ---

Question: {query}
Passage:"""
    
    try:
        config = GenerateContentConfig(
            temperature=0.7,
        )
        response = client.models.generate_content(model=gemini_model, contents=prompt, config=config)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error on query '{query[:30]}...': {e}")
        return "SKIP"


def hybrid_search(query, top_k_search=75):
    conn = get_db_connection()

    query_embedding = embed_model.encode(query).tolist()

    dense = dense_retrieval(conn, query_embedding, top_k=top_k_search)
    sparse = sparse_retrieval(conn, query, top_k=top_k_search)
    fused = rrf(dense, sparse)
    reranked = rerank_results(query, fused, top_k=5)
    conn.close()

    return reranked

def hybrid_search_with_HyDE(query, top_k_search=75):
    conn = get_db_connection()
    hyde_doc = generate_hyde_document(query)
    print(f"\n[HyDE Document Generated]:\n{hyde_doc}\n")
    hyde_embedding = embed_model.encode(hyde_doc).tolist()
    dense = dense_retrieval(conn, hyde_embedding, top_k=top_k_search)
    sparse = sparse_retrieval(conn, query, top_k=top_k_search)
    fused = rrf(dense, sparse)
    reranked = rerank_results(query, fused, top_k=5)
    conn.close()
    return reranked

def hybrid_search_with_HyDE_Gemini(query, top_k_search=75):
    conn = get_db_connection()
    hyde_doc = generate_hyde_gemini(query)
    print(f"\n[HyDE Document Generated by Gemini]:\n{hyde_doc}\n")
    hyde_embedding = embed_model.encode(hyde_doc).tolist()
    dense = dense_retrieval(conn, hyde_embedding, top_k=top_k_search)
    sparse = sparse_retrieval(conn, query, top_k=top_k_search)
    fused = rrf(dense, sparse)
    reranked = rerank_results(query, fused, top_k=5)
    conn.close()
    return reranked


def main():
    results = hybrid_search(query)  #toggle between hybrid_search , hybrid_search_with_HyDE and hybrid_search_with_HyDE_Gemini to see the difference
    for rank, (row, score) in enumerate(results):
        print(f"\n--- Rank {rank + 1} | Relevance Score: {score:.4f} ---")
        print(row[1][:500])

if __name__ == "__main__":
    main()