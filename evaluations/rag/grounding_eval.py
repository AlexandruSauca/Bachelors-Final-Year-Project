from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GeminiModel
from deepeval.evaluate import AsyncConfig
import json
import os
from dotenv import load_dotenv
from retrieve import hybrid_search
from load_models.load_qwen2vl2B import model, processor

load_dotenv()
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "300"
os.environ["DEEPEVAL_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "600"

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
judge = GeminiModel(model= "gemini-2.5-flash")

fluency = GEval(
    name = "Fluency",
    criteria= """
    Evaluate the 'Scientific Fluency' of the response based on the following 4 dimensions:
    1. Grammar & Syntax: The text must be free of grammatical errors, typos, and awkward phrasing.
    2. Academic Register: The tone should be formal, objective, and scholarly. Avoid slang, contractions, or overly conversational 'chatty' language.
    3. Technical Cohesion: Technical terms must be used correctly in context, and transitions between sentences should be logical and clear.
    4. Conciseness: The response should be direct and avoid unnecessary wordiness or repetitive filler.
    """,
    evaluation_params= [LLMTestCaseParams.ACTUAL_OUTPUT],
    model=judge,
    threshold=0.7,
)

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=judge,
)

answer_relevance = AnswerRelevancyMetric(
    threshold=0.7,
    model=judge,
)

def generator_response(query, contexts):
    #combine top 5 results into one block
    context_text = "\n\n".join([f"Source {i+1}: {c}" for i, c in enumerate(contexts)])
    prompt = f"""<|im_start|>system
You are a precise scientific researcher. Your goal is to provide evidence-based answers.
INSTRUCTIONS:
1. Answer the user's question using ONLY the provided [CONTEXT SOURCES].
2. Use a formal, objective, academic tone.
3. If the answer is not present in the sources, say: "The provided context does not contain enough information."
4. Every claim must be cited using the format: [Source X].
5. Do NOT use any external knowledge or common sense facts not found in the sources.<|im_end|>
<|im_start|>user
[CONTEXT SOURCES]:
{context_text}

[QUESTION]:
{query}

Final Academic Answer:<|im_end|>
<|im_start|>assistant
"""
    
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

    generated_ids = model.generate(**inputs, max_new_tokens=400, temperature = 0.1, repetition_penalty=1.15)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0].strip()

def grounding_evaluation():
    with open("test_dataset_rag.json", "r") as f:
        test_data = json.load(f)

    test_cases = []
    for item in test_data:
        query = item["question"]
        results = hybrid_search(query, top_k_search=75)
        retrieved_context =[r[0][1] for r in results]
        actual_output = generator_response(query, retrieved_context)
        if not actual_output or actual_output.strip() == "":
            actual_output = "The model failed to generate a response."
        print(f"\n[Qwen Answer Preview]: {actual_output[:100]}...\n")
        test_case = LLMTestCase(
            input=query,
            retrieval_context=retrieved_context,
            actual_output=actual_output
        )
        test_cases.append(test_case)
    results = evaluate(test_cases, 
             [fluency, faithfulness, answer_relevance],
             async_config=AsyncConfig(run_async=True, max_concurrent=2))
    
    def safe_serializer(obj):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj) 

    export_data = {
        "original_test_cases": test_cases,
        "raw_evaluation_results": results
    }

    with open("grounding_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(export_data, f, default=safe_serializer, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    grounding_evaluation()