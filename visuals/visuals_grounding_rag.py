import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open("grounding_evaluation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

test_results = data["raw_evaluation_results"]["test_results"]

metrics_data = []

for result in test_results:
    if "metrics_data" in result:
        for metric in result["metrics_data"]:
            metric_name = metric["name"].split(" [")[0]
            
            metrics_data.append({
                "Metric": metric_name,
                "Score": metric["score"],
                "Passed": metric["success"]
            })

df = pd.DataFrame(metrics_data)

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'}) 

#bar chart
pass_rates = df.groupby('Metric')['Passed'].mean() * 100

plt.figure(figsize=(8, 5))
ax = sns.barplot(x=pass_rates.index, y=pass_rates.values, hue=pass_rates.index, palette="Blues_d", legend=False)

plt.title("Overall Metric Pass Rates", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Pass Rate (%)", fontsize=14)
plt.xlabel("Evaluation Metric", fontsize=14)
plt.ylim(0, 105) 
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points',
                fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("thesis_chart_1_pass_rates.png",) 
plt.close()

#box plot
plt.figure(figsize=(8, 5))

sns.boxplot(x="Metric", y="Score", data=df, hue="Metric", palette="Set2", width=0.5, legend=False)
sns.stripplot(x="Metric", y="Score", data=df, color=".25", size=6, alpha=0.6, jitter=True) 

plt.title("Distribution of Evaluation Scores across 26 Queries", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("DeepEval Score (0.0 to 1.0)", fontsize=14)
plt.xlabel("Evaluation Metric", fontsize=14)
plt.ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig("thesis_chart_2_score_distribution.png")
plt.close()

