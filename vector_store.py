import faiss
import numpy as np
import pickle
import os
from config import Config

class VectorStore:
    def __init__(self, dimension=Config.EMBEDDING_DIM):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
    
    def add_documents(self, documents, embeddings, metadata=None):
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents
        self.documents.extend(documents)
        
        # Store metadata
        if metadata is None:
            metadata = [{"id": i} for i in range(len(documents))]
        self.metadata.extend(metadata)
        
        print(f"Added {len(documents)} documents (Total: {len(self.documents)})")
    
    def search(self, query_embedding, k=10):
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'distance': float(dist),
                    'metadata': self.metadata[idx],
                    'index': int(idx)
                })
        
        return results
    
    def save(self, index_path=Config.INDEX_PATH):
        """Save index and documents"""
        faiss.write_index(self.index, index_path)
        
        data_path = index_path.replace('.bin', '_data.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'dimension': self.dimension
            }, f)
        
        print(f" Saved index to {index_path}")
    
    def load(self, index_path=Config.INDEX_PATH):
        """Load index and documents"""
        self.index = faiss.read_index(index_path)
        
        data_path = index_path.replace('.bin', '_data.pkl')
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.metadata = data['metadata']
        self.dimension = data['dimension']
        
        print(f" Loaded index from {index_path}")
    
    def get_total_documents(self):
        """Return total number of documents"""
        return len(self.documents)