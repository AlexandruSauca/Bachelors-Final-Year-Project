import os
import psycopg2
from dotenv import load_dotenv 
from load_models.load_qwen2vl2B import model, processor
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import bm25s

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
        SELECT id, content, metadata, 
               RANK() OVER (ORDER BY embedding <=> %s::vector) as rank_vec
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
                   """, (query_embedding, query_embedding, top_k))
    results = cursor.fetchall()
    cursor.close()
    return results

def sparse_retrieval(conn, query, top_k = 10):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, content, metadata, 
               RANK() OVER (ORDER BY ts_rank_cd(to_tsvector('english', content), query) DESC) as rank_kw
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

    return [r[0] for r in ranked[:top_k]]

def hybrid_search(query):
    conn = get_db_connection()

    query_embedding = embed_model.encode(query).tolist()

    dense = dense_retrieval(conn, query_embedding, top_k=15)
    sparse = sparse_retrieval(conn, query, top_k=15)
    fused = rrf(dense, sparse)
    reranked = rerank_results(query, fused, top_k=5)
    conn.close()

    return reranked

def main():
    results = hybrid_search(query)
    for r in results:
        print("\n---")
        print(r[1][:500])

if __name__ == "__main__":
    main()