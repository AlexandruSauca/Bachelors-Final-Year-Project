from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display, Markdown
from prompts import prompt_images, prompt_text, prompt_tables
from qwen_vl_utils import process_vision_info
from load_models.load_qwen2vl2B import model, processor
from langchain.document_loaders import ArxivLoader
from langchain.retrievers import ArxivRetriever
import arxiv
import os
import json

def chunk_pdf(file_path):
    chunks = partition_pdf(
    filename=file_path, # path to the PDF file to be partitioned
    infer_table_structure=True, #enables automatic detection and structuring of tables within the document
    strategy="hi_res", #most accurate, but potentially slowest and most resource-intensive, strategy for analyzing a document's layout and content
    extract_image_block_types=["Image", "Tables"], #extracts images and tables locally
    extract_image_block_output_dir="images", #saves images and tables to the specified directory
    extract_image_block_to_payload=True, #metadata with base64
    chunking_strategy="by_title",
    max_characters=10000,
    combine_text_under_n_chars=2000,
    new_after_n_chars=6000,
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

def proccess_images_base64(images_base64, processor, model, prompt = prompt_images, max_tokens=512):
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

def proccess_text(text_chunk, processor, model, prompt=prompt_text, max_tokens=128):
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

def main():
    paper_ids =[
        "2511.20513", #DesignPref: Capturing Personal Preferences in Visual Design Generation
        "2511.20333", #NNGPT: Rethinking AutoML with Large Language Models
        "2511.20227", #HKRAG: Holistic Knowledge Retrieval-Augmented Generation Over Visually-Rich Documents
        "2511.19858", #A Systematic Analysis of Large Language Models with RAG-enabled Dynamic Prompting for Medical Error Detection and Correction
        "2511.19834", #Large Language Model Aided Birt-Hogg-Dube Syndrome Diagnosis with Multimodal Retrieval-Augmented Generation
        "2511.19481", #Quality analysis and evaluation prediction of RAG retrieval based on machine learning algorithms
        "2511.19423", #Beyond Protein Language Models: An Agentic LLM Framework for Mechanistic Enzyme Design
        "2512.10787", #Replace, Don't Expand: Mitigating Context Dilution in Multi-Hop RAG via Fixed-Budget Evidence Assembly
        "2512.10422", #Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers
        "2512.10933", #Anomalous scaling law for the two-dimensional Gaussian free field
        "2512.10897", #Observability inequality for the von Neumann equation in crystals
        "2512.10220", #On Learning-Curve Monotonicity for Maximum Likelihood Estimators
    ]

    client = arxiv.Client()
    search = arxiv.Search(id_list=paper_ids)
    download_dir = "./downloaded_papers"
    output_file = "processed_rag_data.jsonl"
    #os.makedirs(download_dir, exist_ok=True)
    total_chunks = 0
    for result in client.results(search):
        pdf_filename = f"{result.get_short_id()}.pdf"
        file_path = os.path.join(download_dir, pdf_filename)
        # if not os.path.exists(file_path):
        #     result.download_pdf(dirpath=download_dir, filename=f"{result.get_short_id()}.pdf")
        #     print(f"Downloaded {file_path}")


        print(f"Processing {file_path}...")
        chunks = chunk_pdf(file_path)
        #total_chunks += len(chunks)
        print(f"Total Chunks for {result.title}: {len(chunks)}")
        images_base64 = get_images_base64(chunks)
        print(f"Total Images: {len(images_base64)}")
        tables = get_tables(chunks)
        print(f"Total Tables: {len(tables)}")

        paper_data=[]

        print(f"  > Processing {len(chunks)} Text Chunks...")
        for i, chunk in enumerate(chunks):
            # We skip 'Table' and 'Image' types here to avoid duplication
            # We strictly want "CompositeElement" or "Text"
            if "Table" not in str(type(chunk)) and "Image" not in str(type(chunk)):
                
                # Filter out tiny noise (headers/footers < 50 chars)
                if chunk.text and len(chunk.text) > 50:
                    paper_data.append({
                        "id": f"{result.get_short_id()}_text_{i}",
                        "type": "text",
                        "content": chunk.text,  # The actual text content
                        "source": result.title
                    })
        for i, tbl in enumerate(tables):
            if hasattr(tbl.metadata, "image_base64") and tbl.metadata.image_base64:
                print(f"  > Processing Table {i+1}...")
                tbl_b64 = tbl.metadata.image_base64
                description = proccess_images_base64(tbl_b64, processor, model, prompt_tables)
                
                paper_data.append({
                    "id": f"{result.get_short_id()}_table_{i}",
                    "type": "table",
                    "content": description, # Markdown Table + Summary
                    "original_base64": tbl_b64,
                    "source": result.title
                })
        for i, img_b64 in enumerate(images_base64):
            print(f"  > Processing Image {i+1}...")
            description = proccess_images_base64(img_b64, processor, model, prompt_images)
            
            paper_data.append({
                "id": f"{result.get_short_id()}_img_{i}",
                "type": "image",
                "content": description, # Visual Description
                "original_base64": img_b64,
                "source": result.title
            })

        with open(output_file, 'a') as f:
            for entry in paper_data:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {len(paper_data)} total items for {pdf_filename}")
        



    print(f"Total Chunks Processed: {total_chunks}")

    
    

    
    texts = get_text(chunks)[1]
    #print(proccess_images_base64(images_base64[0], processor, model, prompt=prompt_images, max_tokens=128))
    #print(proccess_text(text_chunk=texts, processor=processor, model=model, prompt=prompt_text, max_tokens=128))

if __name__ == "__main__":
    main()