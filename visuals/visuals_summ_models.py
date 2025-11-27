import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

model_name = []
gemini_eval = []
rougeLsum = []
bert_p = []
bert_r = []
bert_f1 = []

file_path = './results/all_results.jsonl'

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)  

for res in data:
    model_name.append(res['id'])
    gemini_eval.append(res["Gemini_evaluation"])
    rougeLsum.append(res["RougeLsum"])
    bert_p.append(res["Precision_BertScore"])
    bert_r.append(res["Recall_BertScore"])
    bert_f1.append(res["F1Score_BertScore"])

plt.figure(figsize=(12, 6))
sns.heatmap(
    data=[gemini_eval, rougeLsum, bert_p, bert_r, bert_f1],
    annot=True,
    fmt=".2f", 
    yticklabels=['Gemini Eval', 'RougeLsum', 'BertScore Precision', 'BertScore Recall', 'BertScore F1'],
    xticklabels=model_name,
    cmap='YlGnBu'
)
plt.title('Model Performance Comparison')
plt.xlabel('Models')
plt.ylabel('Metrics')
plt.tight_layout()
plt.savefig('model_performance_heatmap.png')
plt.show()