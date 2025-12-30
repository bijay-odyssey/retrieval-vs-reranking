from rag_pipeline import RAGPipeline
from evaluation import Evaluator
from config import Config
import os

def load_documents(file_path):
    # 1. 'Clean' Domain Documents
    domain_docs = [
        "Python is a high-level programming language known for simplicity.",
        "Machine learning requires large datasets for training models.",
        "RAG combines retrieval and generation by first retrieving relevant documents and then using them during text generation."
        "Vector databases store embeddings efficiently using FAISS.",
        "Reranking improves retrieval precision using cross-encoders.",
        "FAISS enables fast similarity search at scale.",
        "Large language models generate human-like text.",
        "Fine-tuning adapts models to specific tasks efficiently.",
        "Embeddings represent text as dense numerical vectors.",
        "Chunking splits documents into smaller pieces for retrieval.",
    ]

    # 2. 'Noisy' Technical Documents
    noise_docs = [
    # --- Meta / commentary ---
        "Reranking has become an increasingly discussed topic in modern information retrieval research, especially with the rise of large language models.",
        "Vector databases are often compared based on scalability, licensing, and ecosystem support rather than their internal algorithms.",
        "Embedding models are frequently evaluated using benchmark datasets that measure semantic similarity and clustering performance.",

    # --- Historical / background ---
        "FAISS was released by Facebook AI Research and later open-sourced to support similarity search research.",
        "The transformer architecture was introduced in the paper 'Attention Is All You Need' and changed the direction of NLP research.",
        "Early information retrieval systems relied heavily on keyword-based search before neural methods became popular.",

    # --- Adjacent but wrong intent ---
        "Machine learning workflows often include data preprocessing, model training, evaluation, and deployment stages.",
        "Natural language processing includes tasks such as sentiment analysis, translation, and summarization.",
        "Large language models are often deployed using APIs and require careful monitoring in production environments.",

    # --- Vague / non-definitive ---
        "Chunking strategies can vary widely depending on application constraints and system design choices.",
        "Search accuracy is influenced by many factors including data quality, indexing strategies, and query formulation.",
        "Text representation techniques have evolved significantly over the years with advances in neural networks.",

    # --- Tooling / ecosystem noise ---
        "There are many open-source libraries available for building AI-powered search systems.",
        "Developers often experiment with different embedding models to find the best trade-off between speed and accuracy.",
        "GPU acceleration is commonly used to speed up large-scale machine learning workloads.",

    # --- Organizational / process ---
        "Model evaluation is an ongoing process that should be revisited as data distributions change.",
        "AI systems benefit from continuous iteration and performance monitoring over time.",
        "Research in retrieval-augmented generation is advancing rapidly with new techniques proposed each year."
]


    # Combine them into one list of strings for the pipeline
    all_text_docs = domain_docs + noise_docs

    print(f" Loaded {len(domain_docs)} domain docs and {len(noise_docs)} noise docs.")
    return all_text_docs

def main():
    print("\n" + "="*80)
    print("RAG WITH RERANKING - NOISE TOLERANCE DEMO")
    print("="*80)

    # Initialize pipeline
    print("\n Initializing RAG Pipeline...")
    pipeline = RAGPipeline()

    # Load and index documents
    documents = load_documents(Config.DATA_FILE)
    
    print("\n Indexing documents...")
    pipeline.index_documents(documents)

    # Demo 1: Single query comparison
    print("\n" + "="*80)
    print("DEMO 1: NOISE VS DOMAIN RETRIEVAL")
    print("="*80)
    test_query = "How can I improve my search results accuracy?"
    pipeline.compare_methods(test_query, top_k=3)

    # Demo 2: Multiple queries
    print("\n" + "="*80)
    print("DEMO 2: TESTING RERANKER DISCRIMINATION")
    print("="*80)
    test_queries = [
        "What is used to store vectors?",
        "How to make retrieval more accurate?",
        "What breaks text into smaller parts?"
    ]
    for query in test_queries:
        pipeline.compare_methods(query, top_k=3)

    # Demo 3: Evaluation with ground truth
    print("\n" + "="*80)
    print("DEMO 3: QUANTITATIVE EVALUATION (WITH NOISE)")
    print("="*80)
    test_cases = [
        {'query': 'What stores embeddings efficiently?', 'relevant_doc_ids': [3, 5]},
        {'query': 'How to improve retrieval precision?', 'relevant_doc_ids': [4]},
        {'query': 'What splits documents?', 'relevant_doc_ids': [9]},
        {'query': 'What represents text as vectors?', 'relevant_doc_ids': [8]},
        {'query': 'What combines search and generation?', 'relevant_doc_ids': [2]}
    ]

    evaluator = Evaluator(pipeline)
    results = evaluator.evaluate_queries(test_cases)

    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nTry your own queries to see if the Reranker filters the noise!\n")

    while True:
        query = input(" Enter query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n Goodbye!\n")
            break
        if not query:
            continue
        pipeline.compare_methods(query, top_k=3)

if __name__ == "__main__":
    main()