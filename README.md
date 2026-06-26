# Multimodal Hybrid Retrieval-Augmented Generation for Scientific Document Understanding

An Advanced Multimodal RAG system for scientific PDF question answering, built entirely on locally-deployed, quantized open-source Small Language Models (SLMs). The system combines VLM-based multimodal ingestion with hybrid retrieval (HNSW + GIN), Reciprocal Rank Fusion, and Cross-Encoder reranking to deliver accurate, grounded responses from complex scientific documents.

## Architecture

The pipeline is composed of four stages:

1. **Multimodal Data Ingestion** — PDF parsing via [Unstructured](https://unstructured.io/), table/image summarization with Qwen2-VL-2B-Instruct (INT4), and vectorization with EmbeddingGemma-300m
2. **Vector Database** — PostgreSQL + pgvector with dual indexing: HNSW (semantic) and GIN (lexical)
3. **Hybrid Retrieval** — Dense + Sparse search (Top-K=75), RRF fusion (k=25), and Cross-Encoder reranking (ms-marco-MiniLM-L-6-v2) → Top 3 contexts
4. **Response Generation** — Query Condenser for multi-turn coherence + Qwen2-VL-2B-Instruct (INT4, 1.39 GB VRAM) constrained to retrieved context

The system is served through a **FastAPI** backend and a **Streamlit** frontend (GUS.ai).

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (tested on NVIDIA RTX 4060 TI, 8 GB VRAM)
- Docker & Docker Compose
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on host

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlexandruSauca/Bachelors-Final-Year-Project.git
   cd Bachelors-Final-Year-Project
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   Copy the example file and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env`:
   ```env
   POSTGRES_USER=your_user
   POSTGRES_PASSWORD=your_password
   POSTGRES_DB=your_db_name
   DB_PORT=5432
   APP_PORT=8000
   GEMINI_API_KEY=your_gemini_api_key
   ```

4. **Start the database**
   ```bash
   docker-compose up -d
   ```
   This launches PostgreSQL 16 with pgvector and PgAdmin (accessible at `http://localhost:8080`).

5. **Download models**

   The following models are loaded from HuggingFace on first run (or can be pre-downloaded):
   - `Qwen/Qwen2-VL-2B-Instruct` — VLM for summarization and generation
   - `google/embeddinggemma-300m` — embedding model
   - `cross-encoder/ms-marco-MiniLM-L-6-v2` — reranker

## Usage

### Running the Full Application

1. **Start the backend** (FastAPI on port 8000):
   ```bash
   cd backend
   python main.py
   ```

2. **Start the frontend** (Streamlit on port 8501):
   ```bash
   cd frontend
   streamlit run app.py
   ```

3. Open `http://localhost:8501` in your browser, upload a PDF, and start asking questions.

### Running Individual Components

```bash
# Ingest a PDF directly (without the web interface)
python ingest_postgres.py

# Run a standalone hybrid search query
python retrieve.py
```

### Running Evaluations

```bash
# Retrieval evaluation (MRR, Recall@5, Precision@1)
python evaluations/rag/eval_rag.py

# Retrieval latency benchmarking
python evaluations/rag/benchmark_latency_rag.py

# Generation evaluation (Faithfulness, Answer Relevancy, Fluency)
python evaluations/rag/grounding_eval.py

# VLM summarization evaluation (one script per model)
python evaluations/summ_eval/eval_qwen2_vl_2B.py

# Embedding model comparison (NDCG, Recall)
python evaluations/embed_eval/eval_embed_bier.py
```

## Project Structure

```
├── backend/                              # FastAPI application server
│   ├── main.py                           #   API endpoints (/ingest, /ask)
│   ├── retrieve.py                       #   Hybrid retrieval pipeline
│   ├── ingest_postgres.py                #   Document ingestion pipeline
│   ├── grounding_eval.py                 #   Generator + Query Condenser + DeepEval
│   ├── load_qwen2vl2B.py                 #   Model loading with INT4 quantization
│   └── prompts.py                        #   System prompts
│
├── frontend/                             # Streamlit web interface
│   └── app.py                            #   GUS.ai chat UI
│
├── evaluations/                          # Evaluation scripts
│   ├── rag/
│   │   ├── eval_rag.py                   #   Retrieval metrics (MRR, Hit@1, Hit@5)
│   │   ├── grounding_eval.py             #   Generation quality (DeepEval)
│   │   ├── benchmark_latency_rag.py      #   Latency benchmarking
│   │   └── synthetic_dataset_rag.py      #   Synthetic BeIR dataset generation
│   ├── embed_eval/
│   │   ├── eval_embed_bier.py            #   Embedding model comparison (NDCG, Recall)
│   │   ├── creating_dataset.py           #   Dataset creation for embedding eval
│   │   ├── gemma_embed.py                #   EmbeddingGemma evaluation
│   │   ├── granite_embed.py              #   IBM Granite embedding evaluation
│   │   └── question_gen.py               #   Question generation for eval datasets
│   └── summ_eval/                        #   VLM summarization evaluation
│       ├── eval_qwen2_vl_2B.py           #     Qwen2-VL-2B evaluation
│       ├── eval_qwen2.5_3B.py            #     Qwen2.5-VL-3B evaluation
│       ├── eval_gemma_3_4B.py            #     Gemma3-4B evaluation
│       ├── eval_qwen3_2B_thinking.py     #     Qwen3-2B-Thinking evaluation
│       ├── eval_smolVLM_2B.py            #     SmolVLM-2B evaluation
│       └── eval_gemini.py                #     Gemini-2.5-Flash-Lite evaluation
│
├── load_models/                          # Model loading scripts
│   ├── load_qwen2vl2B.py                 #   Qwen2-VL-2B-Instruct (INT4)
│   ├── load_qwen_2_5_3B.py              #   Qwen2.5-VL-3B
│   ├── load_gemma_4b.py                  #   Gemma3-4B
│   ├── load_qwen3_2B_thinking.py         #   Qwen3-2B-Thinking
│   └── load_smolVLM.py                   #   SmolVLM-2B
│
├── visuals/                              # Visualization generation scripts
│   ├── visuals_summ_models.py            #   Summarization model comparison charts
│   └── visuals_grounding_rag.py          #   Grounding evaluation charts
│
├── chunking.py                           # PDF chunking and VLM summarization
├── ingest_postgres.py                    # Standalone ingestion pipeline
├── retrieve.py                           # Standalone hybrid retrieval
├── hybrid_db.py                          # Database schema initialization
├── prompts.py                            # System prompts for summarization/evaluation
├── part.py                               # PDF partitioning utilities
├── part_pdf.ipynb                        # PDF partitioning notebook
├── partition_pdf.ipynb                   # PDF partition exploration notebook
├── docker-compose.yaml                   # PostgreSQL + pgvector + PgAdmin
├── requirements.txt                      # Python dependencies
├── .env.example                          # Environment variable template
├── pdfs/                                 # Sample PDF documents
│   ├── 1706.03762v7.pdf                  #   "Attention Is All You Need" (Vaswani et al.)
│   └── math.pdf                          #   Additional test document
├── results/                              # Raw evaluation outputs
└── images/                              # Evaluation result visualizations
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Upload and ingest a PDF document |
| POST | `/ask` | Send a query with optional chat history |

### Example `/ask` request:
```json
{
  "query": "How does self-attention work in the Transformer?",
  "history": [
    {"role": "user", "content": "What is a Transformer?"},
    {"role": "assistant", "content": "A Transformer is..."}
  ]
}
```

## Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| RRF constant | k=25 | `retrieve.py` |
| Initial retrieval pool | Top-K=75 | `retrieve.py` |
| Reranker output | Top 3 | `backend/retrieve.py` |
| Min chunk length | 50 chars | `ingest_postgres.py` |
| Quantization | INT4 (NF4) | `load_models/load_qwen2vl2B.py` |
| Embedding dimensions | 768 | `ingest_postgres.py` |
| Query Condenser history | Last 4 turns | `backend/grounding_eval.py` |

## Evaluation Results

| Stage | Metric | Score |
|-------|--------|-------|
| Retrieval | MRR (proposed vs Naive-RAG) | 0.340 vs 0.132 (+157%) |
| Retrieval | Recall@5 (Top-K=75) | 0.538 |
| Ingestion | BERTScore F1 (Qwen2-VL-2B) | 55.02 |
| Ingestion | BERTScore F1 (Gemini baseline) | 56.96 |
| Generation | Faithfulness | 88.5% |
| Generation | Answer Relevancy | 80.8% |
| Generation | Fluency | 69.2% |

## Tech Stack

- **VLM / Generator**: Qwen2-VL-2B-Instruct (INT4 quantized via bitsandbytes)
- **Embedding**: EmbeddingGemma-300m (768-dim)
- **Reranker**: ms-marco-MiniLM-L-6-v2 (22.7M params)
- **Database**: PostgreSQL 16 + pgvector (HNSW + GIN)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit (GUS.ai)
- **Evaluation**: DeepEval, BERTScore, ROUGE-LSum, Gemini-2.5-Flash (LLM-as-a-Judge)
- **Parsing**: Unstructured (hi_res strategy) + Tesseract OCR

## License

This project was developed as a Bachelor's thesis at Politehnica University of Bucharest, Faculty of Electronics, Telecommunications and Information Technology (2026).