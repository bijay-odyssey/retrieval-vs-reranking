from embedding_model import EmbeddingModel
from reranker import Reranker
from vector_store import VectorStore
from config import Config
import time

class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.reranker = Reranker()
        self.vector_store = VectorStore()
    
    def index_documents(self, documents):
        print("\nIndexing documents...")
        print(f"Total documents: {len(documents)}")
        
        # Generate embeddings
        embeddings = self.embedder.encode(documents, show_progress=True)
        
        # Add to vector store
        metadata = [{"doc_id": i, "length": len(doc)} for i, doc in enumerate(documents)]
        self.vector_store.add_documents(documents, embeddings, metadata)
        
        print("✓ Indexing complete!\n")
    
    def search_without_reranking(self, query, top_k=3):
        start_time = time.time()
        
        # Embed query
        query_embedding = self.embedder.encode(query)
        
        # Search
        results = self.vector_store.search(query_embedding, k=top_k)
        
        elapsed = time.time() - start_time
        
        return {
            'results': results,
            'time': elapsed,
            'method': 'without_reranking'
        }
    
    def search_with_reranking(self, query, top_k_retrieve=10, top_k_rerank=3):
        start_time = time.time()
        
        # Stage 1: Fast retrieval
        query_embedding = self.embedder.encode(query)
        candidates = self.vector_store.search(query_embedding, k=top_k_retrieve)
        
        stage1_time = time.time() - start_time
        
        # Stage 2: Reranking
        rerank_start = time.time()
        reranked = self.reranker.rerank_with_metadata(query, candidates, top_k=top_k_rerank)
        stage2_time = time.time() - rerank_start
        
        total_time = time.time() - start_time
        
        return {
            'results': reranked,
            'time': total_time,
            'stage1_time': stage1_time,
            'stage2_time': stage2_time,
            'method': 'with_reranking'
        }
    
    def compare_methods(self, query, top_k=3):
        print(f"\n Query: '{query}'\n")
        print("="*80)
        
        # Without reranking
        print("\n METHOD 1: Standard Retrieval (No Reranking)")
        print("-"*80)
        result_no_rerank = self.search_without_reranking(query, top_k=top_k)
        
        for i, doc in enumerate(result_no_rerank['results'], 1):
            print(f"\n{i}. [Distance: {doc['distance']:.4f}]")
            print(f"   {doc['text'][:200]}...")
        
        print(f"\n  Time: {result_no_rerank['time']*1000:.2f}ms")
        
        # With reranking
        print("\n\n METHOD 2: Retrieval + Reranking")
        print("-"*80)
        result_rerank = self.search_with_reranking(
            query, 
            top_k_retrieve=10, 
            top_k_rerank=top_k
        )
        
        for i, doc in enumerate(result_rerank['results'], 1):
            print(f"\n{i}. [Rerank Score: {doc['rerank_score']:.4f}]")
            print(f"   {doc['text'][:200]}...")
        
        print(f"\n Time: {result_rerank['time']*1000:.2f}ms")
        print(f"   ├─ Stage 1 (Retrieval): {result_rerank['stage1_time']*1000:.2f}ms")
        print(f"   └─ Stage 2 (Reranking): {result_rerank['stage2_time']*1000:.2f}ms")
        
        print("\n" + "="*80 + "\n")
        
        return {
            'without_reranking': result_no_rerank,
            'with_reranking': result_rerank
        }