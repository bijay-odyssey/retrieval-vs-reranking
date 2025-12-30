from rag_pipeline import RAGPipeline
import numpy as np

class Evaluator:
    def __init__(self, rag_pipeline):
        self.pipeline = rag_pipeline
    
    def evaluate_queries(self, test_cases):
        print("\n EVALUATION REPORT")
        print("="*80)
        
        results = {
            'without_reranking': {'hits': 0, 'mrr': []},
            'with_reranking': {'hits': 0, 'mrr': []}
        }
        
        for i, test in enumerate(test_cases, 1):
            query = test['query']
            relevant_ids = test['relevant_doc_ids']
            
            print(f"\n Test {i}: {query}")
            print(f"   Relevant docs: {relevant_ids}")
            
            # Without reranking
            result_no_rerank = self.pipeline.search_without_reranking(query, top_k=3)
            retrieved_ids_no_rerank = [r['metadata']['doc_id'] for r in result_no_rerank['results']]
            
            # With reranking
            result_rerank = self.pipeline.search_with_reranking(query, top_k_retrieve=10, top_k_rerank=3)
            retrieved_ids_rerank = [r['metadata']['doc_id'] for r in result_rerank['results']]
            
            print(f"   Without reranking: {retrieved_ids_no_rerank}")
            print(f"   With reranking: {retrieved_ids_rerank}")
            
            # Calculate metrics
            for method, retrieved_ids in [
                ('without_reranking', retrieved_ids_no_rerank),
                ('with_reranking', retrieved_ids_rerank)
            ]:
                # Hit rate (is any relevant doc in top-k?)
                hit = any(doc_id in relevant_ids for doc_id in retrieved_ids)
                if hit:
                    results[method]['hits'] += 1
                
                # MRR (Mean Reciprocal Rank)
                rank = None
                for pos, doc_id in enumerate(retrieved_ids, 1):
                    if doc_id in relevant_ids:
                        rank = pos
                        break
                
                if rank:
                    results[method]['mrr'].append(1.0 / rank)
                    print(f"   {method}:  Found at rank {rank}")
                else:
                    results[method]['mrr'].append(0.0)
                    print(f"   {method}:  Not found")
        
        # Calculate final metrics
        print("\n" + "="*80)
        print("\n FINAL METRICS")
        print("-"*80)
        
        total_queries = len(test_cases)
        
        for method in ['without_reranking', 'with_reranking']:
            hit_rate = results[method]['hits'] / total_queries * 100
            mrr = np.mean(results[method]['mrr'])
            
            method_name = method.replace('_', ' ').title()
            print(f"\n{method_name}:")
            print(f"  Hit Rate@3: {hit_rate:.1f}%")
            print(f"  MRR: {mrr:.3f}")
        
        print("\n" + "="*80 + "\n")
        
        return results