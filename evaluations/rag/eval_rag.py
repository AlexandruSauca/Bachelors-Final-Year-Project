from retrieve import hybrid_search, hybrid_search_with_HyDE, hybrid_search_with_HyDE_Gemini
import json
import matplotlib.pyplot as plt
import numpy as np

def evaluate_rag(dataset = "test_dataset_rag.json"):
    with open(dataset, "r") as f:
        data = json.load(f)
    total_queries = len(data)
    mrr_sum_15 = 0
    hits_at_1_15 = 0
    hits_at_5_15 = 0

    mrr_sum_hyde_15 = 0
    hits_at_1_hyde_15 = 0
    hits_at_5_hyde_15 = 0

    mrr_sum_hyde_gemini_15 = 0
    hits_at_1_hyde_gemini_15 = 0
    hits_at_5_hyde_gemini_15 = 0

    mrr_sum_75 = 0
    hits_at_1_75 = 0
    hits_at_5_75 = 0

    mrr_sum_hyde_75 = 0
    hits_at_1_hyde_75 = 0
    hits_at_5_hyde_75 = 0

    mrr_sum_hyde_gemini_75 = 0
    hits_at_1_hyde_gemini_75 = 0
    hits_at_5_hyde_gemini_75 = 0

    for i, item in enumerate(data):
        query = item["question"]
        target_chunk_id = item["target_chunk_id"]
        results_15 = hybrid_search(query, top_k_search=15)
        results_with_HyDE_15 = hybrid_search_with_HyDE(query, top_k_search=15)
        results_with_HyDE_Gemini_15 = hybrid_search_with_HyDE_Gemini(query, top_k_search=15)
        retrieved_ids_15 = [r[0][0] for r in results_15]  
        retrieved_ids_hyde_15 = [r[0][0] for r in results_with_HyDE_15]  
        retrieved_ids_hyde_gemini_15 = [r[0][0] for r in results_with_HyDE_Gemini_15]
        rank_15 = 0
        if target_chunk_id in retrieved_ids_15:
            rank_15 = retrieved_ids_15.index(target_chunk_id) + 1
            mrr_sum_15 += (1/rank_15)
            if rank_15 == 1:
                hits_at_1_15 += 1
            if rank_15 <= 5:
                hits_at_5_15 += 1
        print(f"Query {i+1}/{total_queries} | Target ID: {target_chunk_id} | Rank Found: {rank_15 if rank_15 > 0 else 'Missed (Not in Top 5)'}")

        rank_hyde_15 = 0
        if target_chunk_id in retrieved_ids_hyde_15:
            rank_hyde_15 = retrieved_ids_hyde_15.index(target_chunk_id) + 1
            mrr_sum_hyde_15 += (1/rank_hyde_15)
            if rank_hyde_15 == 1:
                hits_at_1_hyde_15 += 1
            if rank_hyde_15 <= 5:
                hits_at_5_hyde_15 += 1
        print(f"HyDE Query {i+1}/{total_queries} | Target ID: {target_chunk_id} | Rank Found: {rank_hyde_15 if rank_hyde_15 > 0 else 'Missed (Not in Top 5)'}")

        rank_hyde_gemini_15 = 0
        if target_chunk_id in retrieved_ids_hyde_gemini_15:
            rank_hyde_gemini_15 = retrieved_ids_hyde_gemini_15.index(target_chunk_id) + 1
            mrr_sum_hyde_gemini_15 += (1/rank_hyde_gemini_15)
            if rank_hyde_gemini_15 == 1:
                hits_at_1_hyde_gemini_15 += 1
            if rank_hyde_gemini_15 <= 5:
                hits_at_5_hyde_gemini_15 += 1
        print(f"HyDE Gemini Query {i+1}/{total_queries} | Target ID: {target_chunk_id} | Rank Found: {rank_hyde_gemini_15 if rank_hyde_gemini_15 > 0 else 'Missed (Not in Top 5)'}")


        results_75 = hybrid_search(query, top_k_search=75)
        results_with_HyDE_75 = hybrid_search_with_HyDE(query, top_k_search=75)
        results_with_HyDE_Gemini_75 = hybrid_search_with_HyDE_Gemini(query, top_k_search=75)
        retrieved_ids_75 = [r[0][0] for r in results_75]
        retrieved_ids_hyde_75 = [r[0][0] for r in results_with_HyDE_75]
        retrieved_ids_hyde_gemini_75 = [r[0][0] for r in results_with_HyDE_Gemini_75]

        rank_75 = 0
        if target_chunk_id in retrieved_ids_75:
            rank_75 = retrieved_ids_75.index(target_chunk_id) + 1
            mrr_sum_75 += (1/rank_75)
            if rank_75 == 1:
                hits_at_1_75 += 1
            if rank_75 <= 5:
                hits_at_5_75 += 1
        print(f"HyDE Query {i+1}/{total_queries} | Target ID: {target_chunk_id} | Rank Found: {rank_75 if rank_75 > 0 else 'Missed (Not in Top 5)'}")

        rank_hyde_75 = 0
        if target_chunk_id in retrieved_ids_hyde_75:
            rank_hyde_75 = retrieved_ids_hyde_75.index(target_chunk_id) + 1
            mrr_sum_hyde_75 += (1/rank_hyde_75)
            if rank_hyde_75 == 1:
                hits_at_1_hyde_75 += 1
            if rank_hyde_75 <= 5:
                hits_at_5_hyde_75 += 1
        print(f"HyDE Query {i+1}/{total_queries} | Target ID: {target_chunk_id} | Rank Found: {rank_hyde_75 if rank_hyde_75 > 0 else 'Missed (Not in Top 5)'}")

        rank_hyde_gemini_75 = 0
        if target_chunk_id in retrieved_ids_hyde_gemini_75:
            rank_hyde_gemini_75 = retrieved_ids_hyde_gemini_75.index(target_chunk_id) + 1
            mrr_sum_hyde_gemini_75 += (1/rank_hyde_gemini_75)
            if rank_hyde_gemini_75 == 1:
                hits_at_1_hyde_gemini_75 += 1
            if rank_hyde_gemini_75 <= 5:
                hits_at_5_hyde_gemini_75 += 1
        print(f"HyDE Gemini Query {i+1}/{total_queries} | Target ID: {target_chunk_id} | Rank Found: {rank_hyde_gemini_75 if rank_hyde_gemini_75 > 0 else 'Missed (Not in Top 5)'}")


    mrr = mrr_sum_15 / total_queries
    hit_rate_1 = hits_at_1_15 / total_queries
    hit_rate_5 = hits_at_5_15 / total_queries
    print(f"\nEvaluation Results:")
    print(f"Total Queries Tested : {total_queries}")
    print(f"Hit@1 (Perfect Top 1): {hit_rate_1:.2%} ({hits_at_1_15}/{total_queries})")
    print(f"Hit@5 (In the Top 5) : {hit_rate_5:.2%} ({hits_at_5_15}/{total_queries})")
    print(f"MRR (Mean Recip Rank): {mrr:.4f}")

    mrr_hyde = mrr_sum_hyde_15 / total_queries
    hit_rate_1_hyde = hits_at_1_hyde_15 / total_queries
    hit_rate_5_hyde = hits_at_5_hyde_15 / total_queries
    print(f"\nEvaluation Results with HyDE:")
    print(f"Hit@1 (Perfect Top 1): {hit_rate_1_hyde:.2%} ({hits_at_1_hyde_15}/{total_queries})")
    print(f"Hit@5 (In the Top 5) : {hit_rate_5_hyde:.2%} ({hits_at_5_hyde_15}/{total_queries})")
    print(f"MRR (Mean Recip Rank): {mrr_hyde:.4f}")

    mrr_hyde_gemini = mrr_sum_hyde_gemini_15 / total_queries
    hit_rate_1_hyde_gemini = hits_at_1_hyde_gemini_15 / total_queries
    hit_rate_5_hyde_gemini = hits_at_5_hyde_gemini_15 / total_queries
    print(f"\nEvaluation Results with HyDE Gemini:")
    print(f"Hit@1 (Perfect Top 1): {hit_rate_1_hyde_gemini:.2%} ({hits_at_1_hyde_gemini_15}/{total_queries})")
    print(f"Hit@5 (In the Top 5) : {hit_rate_5_hyde_gemini:.2%} ({hits_at_5_hyde_gemini_15}/{total_queries})")
    print(f"MRR (Mean Recip Rank): {mrr_hyde_gemini:.4f}")

    mrr_75 = mrr_sum_75 / total_queries
    hit_rate_1_75 = hits_at_1_75 / total_queries    
    hit_rate_5_75 = hits_at_5_75 / total_queries
    print(f"\nEvaluation Results (Top 75):")
    print(f"Hit@1 (Perfect Top 1): {hit_rate_1_75:.2%} ({hits_at_1_75}/{total_queries})")
    print(f"Hit@5 (In the Top 5) : {hit_rate_5_75:.2%} ({hits_at_5_75}/{total_queries})")
    print(f"MRR (Mean Recip Rank): {mrr_75:.4f}")

    mrr_hyde_75 = mrr_sum_hyde_75 / total_queries
    hit_rate_1_hyde_75 = hits_at_1_hyde_75 / total_queries
    hit_rate_5_hyde_75 = hits_at_5_hyde_75 / total_queries
    print(f"\nEvaluation Results with HyDE (Top 75):")
    print(f"Hit@1 (Perfect Top 1): {hit_rate_1_hyde_75:.2%} ({hits_at_1_hyde_75}/{total_queries})") 
    print(f"Hit@5 (In the Top 5) : {hit_rate_5_hyde_75:.2%} ({hits_at_5_hyde_75}/{total_queries})")
    print(f"MRR (Mean Recip Rank): {mrr_hyde_75:.4f}")

    mrr_hyde_gemini_75 = mrr_sum_hyde_gemini_75 / total_queries
    hit_rate_1_hyde_gemini_75 = hits_at_1_hyde_gemini_75 / total_queries
    hit_rate_5_hyde_gemini_75 = hits_at_5_hyde_gemini_75 / total_queries
    print(f"\nEvaluation Results with HyDE Gemini (Top 75):")
    print(f"Hit@1 (Perfect Top 1): {hit_rate_1_hyde_gemini_75:.2%} ({hits_at_1_hyde_gemini_75}/{total_queries})")
    print(f"Hit@5 (In the Top 5) : {hit_rate_5_hyde_gemini_75:.2%} ({hits_at_5_hyde_gemini_75}/{total_queries})")
    print(f"MRR (Mean Recip Rank): {mrr_hyde_gemini_75:.4f}")

    methods = ['Standard (Top 15)', 'Standard (Top 75)', 'HyDE (Top 15)', 'HyDE (Top 75)', 'HyDE Gemini (Top 15)', 'HyDE Gemini (Top 75)']
    mrr = [mrr, mrr_75, mrr_hyde, mrr_hyde_75, mrr_hyde_gemini, mrr_hyde_gemini_75]
    hit5 = [hit_rate_5, hit_rate_5_75, hit_rate_5_hyde, hit_rate_5_hyde_75, hit_rate_5_hyde_gemini, hit_rate_5_hyde_gemini_75]
    hit1 = [hit_rate_1, hit_rate_1_75, hit_rate_1_hyde, hit_rate_1_hyde_75, hit_rate_1_hyde_gemini, hit_rate_1_hyde_gemini_75]

    x = np.arange(len(methods))  
    width = 0.25  

    fig, ax = plt.subplots(figsize=(12, 7))

    rects1 = ax.bar(x - width, mrr, width, label='MRR (Mean Recip Rank)', color='#1f77b4')
    rects2 = ax.bar(x, hit5, width, label='Hit@5 (Recall)', color='#2ca02c')
    rects3 = ax.bar(x + width, hit1, width, label='Hit@1 (Precision)', color='#ff7f0e')

    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Retrieval Quality Metrics', fontsize=16, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 0.65) 

    def autolabel(rects, is_percentage=False):
        for rect in rects:
            height = rect.get_height()
            label = f'{height:.1%}' if is_percentage else f'{height:.3f}'
            ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')

    autolabel(rects1, is_percentage=False)
    autolabel(rects2, is_percentage=True)
    autolabel(rects3, is_percentage=True)

    plt.tight_layout()
    #plt.savefig('retrieval_metrics_comparison_w_gemini.png')
    plt.show()
    


if __name__ == "__main__":
    evaluate_rag()