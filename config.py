class Config:
    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Reranking model
    RERANKER_MODEL = "BAAI/bge-reranker-base"
    
    # Retrieval settings
    TOP_K_RETRIEVE = 10  # Stage 1: Fast retrieval
    TOP_K_RERANK = 3     # Stage 2: Precise reranking
    
    # Paths
    DATA_FILE = "data/sample_documents.txt"
    INDEX_PATH = "faiss_index.bin"
    
    # Device
    DEVICE = "cuda"