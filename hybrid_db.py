import os
import psycopg2
from dotenv import load_dotenv 

load_dotenv()
DB_HOST = "localhost" 
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

def get_db_connection():
    print("Connecting to the database...")
    try:
        conn  = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )

        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cursor.execute("DROP TABLE IF EXISTS chunks;") 
        cursor.execute("""
            CREATE TABLE chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(768),
                metadata JSONB
            );
        """)

        # semantic index
        cursor.execute("""
            CREATE INDEX idx_chunks_embedding 
            ON chunks USING hnsw (embedding vector_cosine_ops);
        """)

        # full text search index
        cursor.execute("""
            CREATE INDEX idx_chunks_content 
            ON chunks USING GIN (to_tsvector('english', content));
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("Connection successful")

    except Exception as e:
        print("An error occurred while connecting to the database:", e)

if __name__ == "__main__":
    get_db_connection()