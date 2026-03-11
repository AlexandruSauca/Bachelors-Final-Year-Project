import time
from retrieve import hybrid_search, hybrid_search_with_HyDE, hybrid_search_with_HyDE_Gemini
import matplotlib.pyplot as plt

def measure_latency(func, query, top_k_search, runs=3):
    total_time = 0
    print(f"Testing {func.__name__} (top_k={top_k_search})...")
    
    _ = func(query, top_k_search) 
    
    for i in range(runs):
        start_time = time.time()
        _ = func(query, top_k_search)
        end_time = time.time()
        total_time += (end_time - start_time)
        
    avg_time = total_time / runs
    return avg_time

def main():
    test_query = "What techniques were used to prevent the model from overfitting?"
    
    print("Starting Latency Benchmark...\n")
    
    time_standard_15 = measure_latency(hybrid_search, test_query, top_k_search=15)
    time_standard_75 = measure_latency(hybrid_search, test_query, top_k_search=75)
    time_hyde_15 = measure_latency(hybrid_search_with_HyDE, test_query, top_k_search=15)
    time_hyde_75 = measure_latency(hybrid_search_with_HyDE, test_query, top_k_search=75)
    time_hyde_gemini_15 = measure_latency(hybrid_search_with_HyDE_Gemini, test_query, top_k_search=15)
    time_hyde_gemini_75 = measure_latency(hybrid_search_with_HyDE_Gemini, test_query, top_k_search=75)
    
    print("INFERENCE LATENCY RESULTS (Average over 3 runs) ")
    print(f"1. Standard Hybrid (Top 15): {time_standard_15:.4f} seconds")
    print(f"2. Standard Hybrid (Top 75): {time_standard_75:.4f} seconds")
    print(f"3. HyDE Hybrid     (Top 15): {time_hyde_15:.4f} seconds")
    print(f"4. HyDE Hybrid     (Top 75): {time_hyde_75:.4f} seconds")
    print(f"5. HyDE Gemini Hybrid (Top 15): {time_hyde_gemini_15:.4f} seconds")
    print(f"6. HyDE Gemini Hybrid (Top 75): {time_hyde_gemini_75:.4f} seconds")

    fig = plt.figure(figsize=(10, 6))
    bar = plt.bar(
        ["Standard Top 15", "Standard Top 75", "HyDE Top 15", "HyDE Top 75", "HyDE Gemini Top 15", "HyDE Gemini Top 75"],
        [time_standard_15, time_standard_75, time_hyde_15, time_hyde_75, time_hyde_gemini_15, time_hyde_gemini_75],
        color=['blue', 'blue', 'orange', 'orange']
    )
    plt.title("Average Inference Latency for Hybrid Search Methods")
    plt.ylabel("Latency (seconds)")
    plt.ylim(0, max(time_standard_75, time_hyde_75, time_hyde_gemini_75) * 1.5)
    plt.xticks(rotation=45, ha='right')
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{height:.4f}s', ha='center', va='bottom')
    plt.tight_layout()
    #plt.savefig("latency_comparison_rag_w_gemini.png")
    plt.show()


if __name__ == "__main__":
    main()