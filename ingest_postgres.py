import os
import psycopg2
from dotenv import load_dotenv 
from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display, Markdown
from prompts import prompt_images, prompt_text
from qwen_vl_utils import process_vision_info
from load_models.load_qwen2vl2B import model, processor
from sentence_transformers import SentenceTransformer
import json
import re

file = "./Examples/1706.03762v7.pdf"
vector_dim = 768 # output of EmbeddingGemma, need mofification for other models
embed_model = SentenceTransformer("google/embeddinggemma-300m")

load_dotenv()
DB_HOST = "localhost" 
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

def get_db_connection(vector_dim):
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
        cursor.execute(f"""
            CREATE TABLE chunks (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector({vector_dim}), 
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

def get_insert_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

def chunk_pdf(file_path):
    chunks = partition_pdf(
    filename=file_path, # path to the PDF file to be partitioned
    infer_table_structure=True, #enables automatic detection and structuring of tables within the document
    strategy="hi_res", #most accurate, but potentially slowest and most resource-intensive, strategy for analyzing a document's layout and content
    extract_image_block_types=["Image", "Table"], #extracts images and tables locally
    extract_image_block_output_dir="images", #saves images and tables to the specified directory
    extract_image_block_to_payload=True, #metadata with base64
    chunking_strategy="by_title",
    max_characters=2000,
    combine_text_under_n_chars=500,
    new_after_n_chars=1500,
    )
    return chunks

def get_text(chunks):
    texts = []
    for chunk in chunks:
        if 'CompositeElement' in str(type(chunk)):
            texts.append(chunk.text)
    return texts

def get_tables(chunks):
    tables = []
    for chunk in chunks:
        if 'Table' in str(type(chunk)):
            tables.append(chunk)
    return tables

def get_images_base64(chunks):
    image_64=[]
    for chunk in chunks:
        if 'CompositeElement' in str(type(chunk)):
            chunk_el =chunk.metadata.orig_elements
            for el in chunk_el:
                if 'Image' in str(type(el)):
                    image_64.append(el.metadata.image_base64)
    return image_64

def display_image_base64(base64_code):
  image_data= base64.b64decode(base64_code) #decode base64 to binary
  display(Image(data=image_data))

def proccess_images_base64(images_base64, processor, model, prompt = prompt_images, max_tokens=256):
    """
    Generates a text description for a single base64-encoded image using Qwen-VL.

    Args:
        base64_image (str): The raw base64 encoded image string (without the 'data:image/jpeg;base64,' prefix).
        model: The loaded Qwen2VLForConditionalGeneration model.
        processor: The loaded AutoProcessor.
        prompt_text (str): The text prompt to guide the description.
        max_tokens (int): The maximum number of new tokens to generate.

    Returns:
        str: The generated text description.
    """
    message_images = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "data:image;base64," + images_base64},
            {"type": "text", "text": prompt_images},
        ],
    }
    ]
    text = processor.apply_chat_template(
    message_images, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(message_images)
    inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def proccess_text(text_chunk, processor, model, prompt=prompt_text, max_tokens=256):
    final_prompt = prompt.format(element=text_chunk)
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": final_prompt} 
        ],
    }
    ]

    text = processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
    )

    inputs = processor(
    text=[text],
    images=None,  # Explicitly state there are no images
    videos=None,
    padding=True,
    return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=150) # Give it 150 tokens for a summary
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0] 

def filter_noise(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  #for URLs
    text = re.sub(r'\S+@\S+\.\S+', '', text)  #for emails
    #for arxiv references
    text = re.sub(r'arXiv:.*?(?=\s|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'arXiv preprint', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[[0-9,\s]+\]', '', text)  #removing brackets
    text = re.sub(r'\s+', ' ', text).strip()  # for spaces
    text = re.sub(r'(?:[a-zA-Z0-9]\s){5,}', '', text)
    return text

def ingest_data():
    #initialize DB
    get_db_connection(vector_dim)

    #break pdf into chunks
    chunks = chunk_pdf(file)

    conn = get_insert_connection()
    cur = conn.cursor()

    ref_section = False

    print(f"Processing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):

        if ref_section:
            continue
        original_content = str(chunk)
        if original_content.strip().lower().startswith(("references", "reference", "bibliography")):
            ref_section = True
            continue

        clean_content = filter_noise(original_content)

        content_to_embed = ""
        chunk_type = "text" 
        
        metadata_dict = chunk.metadata.to_dict()
        

        raw_orig_elements = chunk.metadata.orig_elements or []

        found_special = False
        
        for el in raw_orig_elements:
            el_type_str = str(type(el)) # This will now say 'Table' or 'Image' correctly

            # table
            if 'Table' in el_type_str:
                chunk_type = "table"
                search_text = ""
                full_text = ""
                print(f"   [{i}] Found Table! Summarizing...")
                
                # Tables behave differently in object vs dict
                if hasattr(el.metadata, 'text_as_html') and el.metadata.text_as_html:
                    table_html = el.metadata.text_as_html
                else:
                    table_html = el.text

                summary = proccess_text(table_html, processor, model, prompt=prompt_text, max_tokens=256)
                search_text = filter_noise(summary)
                full_text = f"Summary: {summary}\n\nRaw Table:\n{original_content}"
                found_special = True
                break 

            # image
            elif 'Image' in el_type_str:
                # In object mode, access metadata.image_base64 directly
                if hasattr(el.metadata, 'image_base64') and el.metadata.image_base64:
                    chunk_type = "image"
                    search_text = ""
                    full_text = ""  
                    print(f"   [{i}]  Found Image! Describing...")
                    
                    base64_img = el.metadata.image_base64
                    desc = proccess_images_base64(base64_img, processor, model, prompt=prompt_images, max_tokens=256)
                    search_text = filter_noise(desc)
                    full_text = f"Image Description: {desc}\n(Image Context: {original_content})"
                    found_special = True
                    break 

        # text
        if not found_special:
            if len(clean_content) < 50:   #skips text chunks that are too short
                continue
            chunk_type = "text"
            search_text = ""
            full_text = ""
            search_text = clean_content
            full_text = clean_content

        metadata_dict["type"] = chunk_type

        # HNSW index for semantic search
        vector = embed_model.encode(search_text).tolist()

        sql = """
            INSERT INTO chunks (content, embedding, metadata)
            VALUES (%s, %s, %s);
        """
        
        #full_text for GIN index for finding keywords
        cur.execute(sql, (full_text, vector, json.dumps(metadata_dict)))
    
    conn.commit()
    cur.close()
    conn.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()