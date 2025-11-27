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

fig, ax = plt.subplots(1,2,figsize=(12, 6))
fig.suptitle('Model Performance Comparison')
sns.heatmap(
    data=[rougeLsum, bert_p, bert_r, bert_f1],
    annot=True,
    fmt=".2f", 
    yticklabels=['RougeLsum', 'BertScore Precision', 'BertScore Recall', 'BertScore F1'],
    xticklabels=model_name,
    cmap='rainbow',
    vmin=0,
    vmax =100,
    ax = ax[0]
)
ax[0].set_title('ROUGE and BERTScore Metrics')
sns.heatmap(
    data=[gemini_eval],
    annot=True,
    fmt=".2f",
    yticklabels=['Gemini Evaluation'],
    xticklabels=model_name,
    cmap='twilight',
    vmin=0,
    vmax=5,
    ax = ax[1]
)
ax[1].set_title('Gemini Evaluation Metric')
plt.tight_layout()
#plt.savefig('model_performance_heatmap2.png')
plt.show()