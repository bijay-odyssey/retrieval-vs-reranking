from sentence_transformers import CrossEncoder
from config import Config
import numpy as np

class Reranker:
    def __init__(self, model_name=Config.RERANKER_MODEL):
        print(f"Loading reranker model: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
        print("Reranker model loaded")
    
    def rerank(self, query, documents, top_k=None):
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score all pairs
        scores = self.model.predict(pairs)
        
        # Create results with original indices
        results = [
            (doc, float(score), idx) 
            for idx, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def rerank_with_metadata(self, query, documents_with_meta, top_k=None):
        if not documents_with_meta:
            return []
        
        # Extract texts
        texts = [doc['text'] for doc in documents_with_meta]
        
        # Rerank
        reranked = self.rerank(query, texts, top_k=None)
        
        # Merge scores with metadata
        results = []
        for text, score, orig_idx in reranked:
            doc_copy = documents_with_meta[orig_idx].copy()
            doc_copy['rerank_score'] = score
            results.append(doc_copy)
        
        # Sort and return top k
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results