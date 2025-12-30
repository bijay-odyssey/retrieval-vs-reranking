# RAG Precision: Retrieval vs. Reranking with Noise Injection

This project demonstrates the impact of **Reranking** in a Retrieval-Augmented Generation (RAG) pipeline. While standard vector retrieval is fast, it often struggles when faced with "noisy" or semantically similar distractor documents. This demo highlights how a Cross-Encoder Reranker corrects these mistakes to improve precision.

## The Problem
Standard retrieval (Bi-Encoders) uses cosine similarity which can be "fooled" by long, keyword-heavy documents that aren't actually relevant. 

## The Solution
By adding a **Reranking stage** (Cross-Encoder), the system performs a deeper semantic analysis of the top-k results, ensuring the most relevant "Domain" knowledge is pushed to the top, even when "Noise" documents have high initial similarity scores.

## Key Features
- **Noise Injection:** 20+ "noisy" technical documents designed to challenge the retriever.
- **Dual Evaluation:** Compares Standard Retrieval (FAISS + MiniLM) vs. Reranked Results (BGE-Reranker).
- **Quantitative Metrics:** Tracks Hit Rate@k and Mean Reciprocal Rank (MRR).
- **Interactive Mode:** Test your own queries against the pipeline in real-time.

## Tech Stack
- **Language:** Python 3.10+
- **Embeddings:** `all-MiniLM-L6-v2`
- **Reranker:** `BAAI/bge-reranker-base`
- **Vector DB:** FAISS (Approximate Nearest Neighbor)
- **Framework:** Sentence-Transformers

## Quick Start
1. Clone the repo.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the demo:
   ```bash
  python main.py
