import logging
import torch
import gc
import matplotlib.pyplot as plt
import numpy as np
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval import models

logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

# --- CONFIGURATION ---
DATA_PATH = "./synthetic_dataset"
OUTPUT_IMAGE = "benchmark_results.png"

# The competitors
MODEL_LIST = {
    "Granite-278m": "ibm-granite/granite-embedding-278m-multilingual",
    "Gemma-300m": "google/embeddinggemma-300m", 
    "E5-Multilingual": "intfloat/multilingual-e5-large-instruct",
}

def evaluate_single_model(model_name, model_id, corpus, queries, qrels):
    print(f"\n\n==================================================")
    print(f"STARTING EVALUATION: {model_name}")
    print(f"==================================================")
    
    try:
        # Load Model (FP16 for speed/memory safety)
        model = DRES(models.SentenceBERT(model_id, trust_remote_code=True), batch_size=16)
        
        # Initialize Retriever
        retriever = EvaluateRetrieval(model, score_function="cos_sim")
        
        # Run Retrieval
        print(f"Retrieving candidates...")
        results = retriever.retrieve(corpus, queries)
        
        # Calculate Metrics
        print("Calculating metrics...")
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values=[1, 5, 10])
        
        # We focus on the two most important metrics for your report
        scores = {
            "NDCG@10": ndcg['NDCG@10'], 
            "Recall@5": recall['Recall@5']
        }
        
        print(f"--- RESULTS: {model_name} ---")
        print(f"NDCG@10:  {scores['NDCG@10']:.4f}")
        print(f"Recall@5: {scores['Recall@5']:.4f}")
        
        return scores

    except Exception as e:
        print(f"CRITICAL ERROR evaluating {model_name}: {e}")
        return None
    finally:
        # VRAM Cleanup
        if 'model' in locals(): del model
        if 'retriever' in locals(): del retriever
        gc.collect()
        torch.cuda.empty_cache()


def plot_benchmark_results(final_scores, output_file):
    """
    Generates a grouped bar chart for model comparison.
    final_scores example: {'Granite': {'NDCG@10': 0.5, 'Recall@5': 0.6}, ...}
    """
    print(f"\nGeneratng plot to {output_file}...")
    
    models = list(final_scores.keys())
    metrics = list(final_scores[models[0]].keys()) # ['NDCG@10', 'Recall@5']
    
    x = np.arange(len(metrics))  # label locations
    width = 0.25  # width of the bars
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars for each model
    multiplier = 0
    for model in models:
        measurement = [final_scores[model][m] for m in metrics]
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=model)
        ax.bar_label(rects, padding=3, fmt='%.3f') # Add numbers on top of bars
        multiplier += 1

    # Styling
    ax.set_ylabel('Score (0.0 - 1.0)')
    ax.set_title('RAG Model Evaluation: Granite vs Gemma vs E5 Multilingual')
    ax.set_xticks(x + width / 2 * (len(models) - 1)) # Center labels
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper left', ncols=len(models))
    ax.set_ylim(0, 1.15) # Give space for labels
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved successfully!")

def main():

    print("Loading Dataset...")
    try:
        corpus, queries, qrels = GenericDataLoader(data_folder=DATA_PATH).load(split="test")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Ensure 'synthetic_dataset/qrels/test.tsv' exists.")
        return

    
    final_scores = {}
    
    for name, path in MODEL_LIST.items():
        scores = evaluate_single_model(name, path, corpus, queries, qrels)
        if scores:
            final_scores[name] = scores

    if final_scores:
        print("\n\n##################################################")
        print("FINAL LEADERBOARD")
        print("##################################################")
        sorted_models = sorted(final_scores.items(), key=lambda x: x[1]['NDCG@10'], reverse=True)
        
        for name, metrics in sorted_models:
            print(f"{name:<20} | NDCG: {metrics['NDCG@10']:.4f} | Recall: {metrics['Recall@5']:.4f}")
        
        # Generate the chart
        plot_benchmark_results(final_scores, OUTPUT_IMAGE)

if __name__ == "__main__":
    main()