from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display, Markdown
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, BertTokenizer, BertModel
import torch
from qwen_vl_utils import process_vision_info
from google import genai
from dotenv import load_dotenv
import enum
from tqdm import tqdm
import json
import os
import time
import re
from pydantic import BaseModel, Field
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from evaluate import load
from bert_score import BERTScorer
import bert_score
import logging
from google.generativeai import types
from google.api_core import exceptions
from load_qwen3_2B_thinking import model, processor
from prompts import prompt_eval, prompt_images, prompt_text, SUMMARY_PROMPT, prompt_get_num, COMPARISON_PROMPT
from chunking import proccess_text, get_text,chunk_pdf


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

load_dotenv()

rouge = load('rouge')
#scorer = BERTScorer(model_type= "bert-base-uncased")

client = genai.Client(api_key="your_api_key")

file = "./Examples/1706.03762v7.pdf"

file_json = "./results/qwen3_2B_thinking_results.jsonl"

files = [
        "./mmlb_data/summ/gov_K4.jsonl", "./mmlb_data/summ/gov_K8.jsonl",
        "./mmlb_data/summ/lexsum_K4.jsonl", "./mmlb_data/summ/lexsum_K8.jsonl"
    ]

results_path = "./results/"
data_root = "./4_summ_image/mmlb_image/"
output_qwen2 = "./results/qwen3_2B_thinking_results.jsonl"
output_all_results_qwen = "./results/all_results.jsonl"


class SummaryRating(enum.Enum):
  VERY_GOOD = '5'
  GOOD = '4'
  OK = '3'
  BAD = '2'
  VERY_BAD = '1'


def evaluate_summary(prompt, ai_response, chat):
    """Evaluate the generated summary against the prompt used."""
    #chat = client.chats.create(model='gemini-2.5-flash-lite')
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chat.send_message(
                message=SUMMARY_PROMPT.format(
                    prompt=prompt, 
                    response=ai_response
                )
            )
            verbose_eval = response.text
            return verbose_eval  # Success, exit the function

        except (exceptions.ServiceUnavailable, exceptions.DeadlineExceeded) as e:
            # This catches 503 UNAVAILABLE and Timeouts
            print(f"  > Gemini Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt + 1 == max_retries:
                print("  > Max retries reached. Failing this item.")
                return f"EVALUATION FAILED"
            
            wait_time = 2 ** attempt  # Exponential backoff (1s, 2s, 4s)
            print(f"  > Retrying in {wait_time} second(s)...")
            time.sleep(wait_time)
        
        except Exception as e:
            print(f"  > Gemini evaluation failed with a non-retryable error: {e}")
            return f"EVALUATION FAILED"

    return "EVALUATION FAILED: Max retries reached"

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
                # if i >= 10: # Only process 10 items
                #     print("Stopping after 10 items for testing.")
                #     break
                entry = json.loads(line)
                image_paths = entry['image_list']
                ref_summary = entry['summary']
                summary_data = entry['summary']
                ref_summary_flatten = "" 

                if isinstance(summary_data, list) and summary_data:
                # --- Case 1: List of Dictionaries ---
                    if isinstance(summary_data[0], dict):
                        ref_summary_parts = []
                        for sec in summary_data:
                            title = sec.get("section_title", "").strip()
                            paragraphs = " ".join(sec.get("paragraphs", [])).strip()
                            if title:
                                ref_summary_parts.append(f"{title}\n{paragraphs}")
                            else:
                                ref_summary_parts.append(paragraphs)
                        ref_summary_flatten = "\n\n".join(ref_summary_parts).strip()

                    # --- Case 2: List of Strings ---
                    elif isinstance(summary_data[0], str):
                        ref_summary_flatten = "\n\n".join(summary_data).strip()

                    # --- Case 3: Single String ---
                elif isinstance(summary_data, str):
                    ref_summary_flatten = summary_data.strip()

                    # --- Fallback for other formats ---
                else:
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
                    
                    chat = client.chats.create(model='gemini-2.5-flash-lite')

                    print(f"  > Evaluating summary for {entry['id']}...")
                    eval_text = evaluate_summary(prompt=prompt_text, ai_response=pred_summary, chat=chat)
                    eval_num = get_num_eval(prompt=prompt_get_num, gemini_response=eval_text)

                    print(f" > Comparing the 2 summaries for {entry['id']}...")
                    comp_summ = compare_summ(reference_summ=ref_summary_flatten, pred_summ=pred_summary, chat=chat)
                    eval_summ = get_num_eval(prompt=prompt_get_num, gemini_response=comp_summ)


                    print(f" >Parsing the rougeLsum score for {entry['id']}...")
                    rouge_scores = rouge.compute(predictions=[pred_summary], references=[ref_summary_flatten])

                    print(f" >Parsing BertScore metrics for {entry['id']}...")
                    P, R, F1 = bert_score.score([pred_summary], [ref_summary_flatten], model_type='bert-base-uncased')

                    # print("  > Pausing 17s for API rate limit...")
                    # time.sleep(17)

                    results.append({
                        "id": entry['id'],
                        "reference": ref_summary,
                        "reference_flatten": ref_summary_flatten,
                        "prediction": pred_summary,
                        "gemini_eval_quality" : eval_num,
                        "gemini_eval_comparison": eval_summ,
                        "rougeLsum_score": rouge_scores.get('rougeLsum', 0.0),
                        "BertScore_P" : P.item(),
                        "BertScore_R" : R.item(),
                        "BertScore_F1" : F1.item(),
                    })
                except Exception as e:
                    print(f"Error during generation for {entry['id']}: {e}")

    # --- Save results ---
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f" Saved {len(results)} results to {results_path}")

    return results

def get_num_eval(prompt, gemini_response):

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=[prompt, gemini_response],
            )
    
            return response.text  # Success, exit the function

        except (exceptions.ServiceUnavailable, exceptions.DeadlineExceeded) as e:
            # This catches 503 UNAVAILABLE and Timeouts
            print(f"  > Gemini Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt + 1 == max_retries:
                print("  > Max retries reached. Failing this item.")
                return f"EVALUATION FAILED"
            
            wait_time = 2 ** attempt  # Exponential backoff (1s, 2s, 4s)
            print(f"  > Retrying in {wait_time} second(s)...")
            time.sleep(wait_time)
        
        except Exception as e:
            print(f"  > Gemini evaluation failed with a non-retryable error: {e}")
            return f"EVALUATION FAILED"

    return "EVALUATION FAILED: Max retries reached"
    
def compare_summ(reference_summ, pred_summ, chat):
    #chat = client.chats.create(model='gemini-2.5-flash-lite')
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chat.send_message(
                message=COMPARISON_PROMPT.format(
                    reference = reference_summ,
                    prediction = pred_summ
                )
            )
            verbose_eval = response.text
            return verbose_eval  # Success, exit the function

        except (exceptions.ServiceUnavailable, exceptions.DeadlineExceeded) as e:
            # This catches 503 UNAVAILABLE and Timeouts
            print(f"  > Gemini Error (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt + 1 == max_retries:
                print("  > Max retries reached. Failing this item.")
                return f"EVALUATION FAILED"
            
            wait_time = 2 ** attempt  # Exponential backoff (1s, 2s, 4s)
            print(f"  > Retrying in {wait_time} second(s)...")
            time.sleep(wait_time)
        
        except Exception as e:
            print(f"  > Gemini evaluation failed with a non-retryable error: {e}")
            return f"EVALUATION FAILED"

    return "EVALUATION FAILED: Max retries reached"


def main():

    # chunks = chunk_pdf(file)
    # texts = get_text(chunks)[1] 
    #model_response = proccess_text(text_chunk=texts, processor=processor, model=model, prompt=prompt_text, max_tokens=512)
    #print("Qwen2-VL Summary:")
    #print(model_response)

    #text_eval = evaluate_summary(prompt=prompt_text, ai_response=model_response)
    #print(text_eval) 

    # The google gemini 2.5 pro model evaluates our model's response as being a 2
    # The google gemini 2.0 flash model evaluates our model's response as being a 5
    
    #num_eval = get_num_eval(prompt=prompt_get_num, gemini_response=text_eval)
    #print(num_eval)
    
    results = evaluate_mmlb(files, output_qwen2, data_root, model, processor, prompt_eval)

    all_scores = []
    scores_this_file = []

    all_rouge = []
    rouge_this_file = []
    
    all_p_bert = []
    all_r_bert = []
    all_f1_bert = []
    
    p_this_file = []
    r_this_file = []
    f1_this_file = []

    with open(file_json, 'r', encoding='utf-8') as f:
            data = json.load(f) 

            for entry in tqdm(data, desc=f"Reading {os.path.basename(file_json)}"):
                eval_comp = entry.get('gemini_eval_comparison', '') 
                eval_rouge = entry.get('rougeLsum_score', '')
                eval_p = entry.get('BertScore_P', '')
                eval_r = entry.get('BertScore_R', '')
                eval_f1 = entry.get('BertScore_F1', '')

                rouge_this_file.append(eval_rouge)
                p_this_file.append(eval_p)
                r_this_file.append(eval_r)
                f1_this_file.append(eval_f1)
                
                match = re.search(r'[1-5]', eval_comp)
                
                if match:
                    scores_this_file.append(int(match.group(0)))
            
            if scores_this_file:
                average_this_file = sum(scores_this_file) / len(scores_this_file)
                print(f"  > Average for this file: {average_this_file:.2f} / 5.0")
                all_scores.extend(scores_this_file) 
            else:
                print("  > No valid scores found in this file.")
            
            if rouge_this_file:
                average_rouge = sum(rouge_this_file) / len(rouge_this_file)
                print(f" > Average rougeLsum score for this file: {average_rouge:.2f}")
                all_rouge.extend(rouge_this_file)
            else:
                print(" > No valid rouge score found in this file.")

            if p_this_file:
                average_bertscore_p = sum(p_this_file) / len(p_this_file)
                print(f" > Average Precision for Bertscore for this file: {average_bertscore_p:.2f}")
                all_p_bert.extend(p_this_file)
            else:
                print(" > No valid Precison for Bertscore found in this file.")

            if r_this_file:
                average_bertscore_r = sum(r_this_file) / len(r_this_file)
                print(f" > Average Recall for Bertscore for this file: {average_bertscore_r:.2f}")
                all_r_bert.extend(r_this_file)
            else:
                print(" > No valid Recall for Bertscore found in this file.")

            if f1_this_file:
                average_bertscore_f1 = sum(f1_this_file) / len(f1_this_file)
                print(f" > Average F1 score for Bertscore for this file: {average_bertscore_f1:.2f}")
                all_f1_bert.extend(f1_this_file)
            else:
                print(" > No valid F1 score for Bertscore found in this file.")
    
    if all_scores:
        final_average = sum(all_scores) / len(all_scores)
        print("\n---")
        print(f"Final Overall Average Score: {final_average:.2f} / 5.0")   # 1.25 / 5.0  really depends on how much the api enters timeout
    else:
        print("\nNo valid scores were found across any files.")

    if all_rouge:
        final_rouge_score = sum(all_rouge) / len(all_rouge)
        final_rouge_100 = final_rouge_score*100
        print("\n---")
        print(f" > Final Average RougeLSum Score: {final_rouge_100:.2f}") # 11.40 similar to publication
    else:
        print(" > No valid rouge score were found in any files.")

    if all_p_bert:
        final_precision_score = sum(all_p_bert) / len(all_p_bert)
        final_p_100 = final_precision_score*100
        print("\n---")
        print(f" > Final Average Precion for  BertScore: {final_p_100:.2f}")  # 42.73
    else:
        print(" > No valid Precision for Bertscore were found in any files.")

    if all_r_bert:
        final_recall_score = sum(all_r_bert) / len(all_r_bert)
        final_R_100 = final_recall_score*100
        print("\n---")
        print(f" > Final Average Recall for  BertScore: {final_R_100:.2f}")  # 52.98
    else:
        print(" > No valid Recall for Bertscore were found in any files.")

    if all_f1_bert:
        final_f1_score = sum(all_f1_bert) / len(all_f1_bert)
        final_f1_100 = final_f1_score*100
        print("\n---")
        print(f" > Final Average F1 score for  BertScore: {final_f1_100:.2f}")  # 47.23
    else:
        print(" > No valid F1 score for Bertscore were found in any files.")

    f_a = round(final_average,2)
    f_rouge = round(final_rouge_100,2)
    f_p = round(final_p_100,2)
    f_r = round(final_R_100,2)
    f_f1 = round(final_f1_100,2)

    all_results = []
    if os.path.exists(output_all_results_qwen):
        try:
            with open(output_all_results_qwen, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            # Make sure it's a list before appending
                if not isinstance(all_results, list):
                    all_results = []
        except json.JSONDecodeError:
            all_results = []

    all_results.append({
    "id" : "Gemma3_4B",
    "Gemini_evaluation" : f_a,
    "RougeLsum" : f_rouge,
    "Precision_BertScore" : f_p,
    "Recall_BertScore" : f_r,
    "F1Score_BertScore" : f_f1,
    })

    with open(output_all_results_qwen, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f" Saved {len(all_results)} results to {output_all_results_qwen}")

    x = np.array(["gemini_evaluation", "rougeLsum", "precision_bertscore", "recall_bertscore", "f1_score_bertscore"])
    y = np.array([f_a, f_rouge, f_p, f_r, f_f1])
    

    plt.figure(figsize=(10, 6)) # Make the figure a bit wider
    bars = plt.bar(x, y, color=['#4285F4', '#DB4437', '#F4B400', '#F4B400', '#0F9D58'])

    plt.title(f"Evaluation Results for Qwen3_2B_Thinking (Scores 0-100)")
    plt.ylabel("Score")
    plt.ylim(0, 100) # Set Y-axis to go from 0 to 100

    plt.bar_label(bars, fmt='%.2f')

    plt.xticks(rotation=15)

    plt.tight_layout() 
    plt.savefig("Qwen3_2B_Thinking_results.png")
    plt.show()

if __name__ == "__main__":
    main()