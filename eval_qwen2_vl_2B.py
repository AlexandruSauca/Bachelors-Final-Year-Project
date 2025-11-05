from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display, Markdown
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
import torch
from qwen_vl_utils import process_vision_info
from google import genai
from dotenv import load_dotenv
import enum
from tqdm import tqdm
import json
import os
import re
from PIL import Image
from evaluate import load
import logging
from google.generativeai import types
from load_qwen2vl2B import model, processor
from prompts import prompt_eval, prompt_images, prompt_text, SUMMARY_PROMPT, prompt_get_num, COMPARISON_PROMPT
from chunking import proccess_text, get_text,chunk_pdf


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv()

client = genai.Client(api_key="your_gemini_api_key")

file = "./Examples/1706.03762v7.pdf"

file_json = "./results/qwen2_vl_mmlb_results.jsonl"

files = [
        "./mmlb_data/summ/gov_K4.jsonl", "./mmlb_data/summ/gov_K8.jsonl",
        "./mmlb_data/summ/lexsum_K4.jsonl", "./mmlb_data/summ/lexsum_K8.jsonl"
    ]

results_path = "./results/"
data_root = "./4_summ_image/mmlb_image/"
output_qwen2 = "./results/qwen2_vl_mmlb_results.jsonl"


class SummaryRating(enum.Enum):
  VERY_GOOD = '5'
  GOOD = '4'
  OK = '3'
  BAD = '2'
  VERY_BAD = '1'

def evaluate_summary(prompt, ai_response):
    """Evaluate the generated summary against the prompt used."""
    chat = client.chats.create(model='gemini-2.0-flash')
    response = chat.send_message(
      message=SUMMARY_PROMPT.format(prompt=prompt, response=ai_response)
    )
    verbose_eval = response.text

    return verbose_eval

def summarize_text_gemini(prompt):
    chat = client.chats.create(model='gemini-2.5-pro')
    response = chat.send_message(
      message=prompt
    )
    summary = response.text
    
    return summary

def evaluate_mmlb(files, results_path, data_root,model, processor, prompt):
    results = []

    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for i,line in enumerate(tqdm(f, desc=f"Processing {os.path.basename(file)}")):
                if i >= 10: # Only process 10 items
                    print("Stopping after 10 items for testing.")
                    break
                entry = json.loads(line)
                image_paths = entry['image_list']
                ref_summary = entry['summary']
                summary_data = entry['summary']
                ref_summary_flatten = "" 

                if isinstance(summary_data, list) and summary_data:
                # Check the type of the first item in the list
                    if isinstance(summary_data[0], dict):
                    # --- Case 1: List of Dictionaries ---
                        ref_summary_parts = []
                        for sec in summary_data:
                            title = sec.get("section_title", "")
                            paragraphs = " ".join(sec.get("paragraphs", []))
                        if title:
                            ref_summary_parts.append(f"{title}\n{paragraphs}")
                        else:
                            ref_summary_parts.append(paragraphs)
                        ref_summary_flatten = "\n\n".join(ref_summary_parts).strip()
        
                    elif isinstance(summary_data[0], str):
                    # --- Case 2: List of Strings ---
                        ref_summary_flatten = "\n\n".join(summary_data).strip()
        
                elif isinstance(summary_data, str):
                # --- Case 3: It's just a single string ---
                    ref_summary_flatten = summary_data.strip()

                else:
                # --- Fallback for other formats ---
                    ref_summary_flatten = str(summary_data)

                images = []

                # --- Load images safely ---
                for image_name in image_paths:
                    # Handle nested folders by ID
                    image_path = os.path.join(data_root, image_name)
                    if os.path.exists(image_path):
                        images.append(Image.open(image_path).convert("RGB"))

                if not images:
                    print(f" Missing all images for {entry['id']}")
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": prompt_eval},
                            ],
                        }
                    ]

                text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                image_inputs, video_inputs = process_vision_info(messages)

                try:
                    # --- Prepare inputs ---
                    inputs = processor(
                        text=[text_prompt],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt"
                    ).to("cuda")

                    # --- Generate summary ---
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=512)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                    pred_summary = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    
                    print(f"  > Evaluating summary for {entry['id']}...")
                    eval_text = evaluate_summary(prompt=prompt_text, ai_response=pred_summary)
                    eval_num = get_num_eval(prompt=prompt_get_num, gemini_response=eval_text)

                    print(f" > Comparing the 2 summaries for {entry['id']}...")
                    comp_summ = compare_summ(reference_summ=ref_summary_flatten, pred_summ=pred_summary)
                    eval_summ = get_num_eval(prompt=prompt_get_num, gemini_response=comp_summ)

                    results.append({
                        "id": entry['id'],
                        "reference": ref_summary,
                        "reference_flatten": ref_summary_flatten,
                        "prediction": pred_summary,
                        "gemini_eval_quality" : eval_num,
                        "gemini_eval_comparison": eval_summ,
                    })
                except Exception as e:
                    print(f"Error during generation for {entry['id']}: {e}")

    # --- Save results ---
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f" Saved {len(results)} results to {results_path}")

    return results

def get_num_eval(prompt, gemini_response):
    response = client.models.generate_content(
      model='gemini-2.0-flash',
      contents=[prompt, gemini_response],
    )
    return response.text
    
def compare_summ(reference_summ, pred_summ):
    chat = client.chats.create(model='gemini-2.0-flash')
    response = chat.send_message(
      message=COMPARISON_PROMPT.format(
        reference = reference_summ,
        prediction = pred_summ
      )
    )
    verbose_eval = response.text

    return verbose_eval
            
def main():

    chunks = chunk_pdf(file)
    texts = get_text(chunks)[1] 
    model_response = proccess_text(text_chunk=texts, processor=processor, model=model, prompt=prompt_text, max_tokens=512)
    print("Qwen2-VL Summary:")
    print(model_response)

    text_eval = evaluate_summary(prompt=prompt_text, ai_response=model_response)
    #print(text_eval) 

    # The google gemini 2.5 pro model evaluates our model's response as being a 2
    # The google gemini 2.0 flash model evaluates our model's response as being a 5
    
    num_eval = get_num_eval(prompt=prompt_get_num, gemini_response=text_eval)
    #print(num_eval)
    
    #results = evaluate_mmlb(files, output_qwen2, data_root, model, processor, prompt_eval)

    all_scores = []
    scores_this_file = []
    with open(file_json, 'r', encoding='utf-8') as f:
            data = json.load(f) 

            for entry in tqdm(data, desc=f"Reading {os.path.basename(file_json)}"):
                eval_comp = entry.get('gemini_eval_comparison', '') 
                
                match = re.search(r'[1-5]', eval_comp)
                
                if match:
                    scores_this_file.append(int(match.group(0)))
            
            if scores_this_file:
                average_this_file = sum(scores_this_file) / len(scores_this_file)
                print(f"  > Average for this file: {average_this_file:.2f} / 5.0")
                all_scores.extend(scores_this_file) 
            else:
                print("  > No valid scores found in this file.")
                
    
    if all_scores:
        final_average = sum(all_scores) / len(all_scores)
        print("\n---")
        print(f"Final Overall Average Score: {final_average:.2f} / 5.0")   # 1.71 / 5.0
    else:
        print("\nNo valid scores were found across any files.")

if __name__ == "__main__":
    main()