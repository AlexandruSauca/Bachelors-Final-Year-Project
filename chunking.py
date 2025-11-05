from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display, Markdown
from prompts import prompt_images, prompt_text
from qwen_vl_utils import process_vision_info
from load_qwen2vl2B import model, processor

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

def proccess_images_base64(images_base64, processor, model, prompt = prompt_images, max_tokens=128):
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
    file = "./Examples/1706.03762v7.pdf"
    chunks = chunk_pdf(file)
    #print(f"Total Chunks: {len(chunks)}")

    images_base64 = get_images_base64(chunks)
    #print(f"Total Images: {len(images_base64)}")

    
    texts = get_text(chunks)[1]
    #print(proccess_images_base64(images_base64[0], processor, model, prompt=prompt_images, max_tokens=128))
    #print(proccess_text(text_chunk=texts, processor=processor, model=model, prompt=prompt_text, max_tokens=128))

if __name__ == "__main__":
    main()