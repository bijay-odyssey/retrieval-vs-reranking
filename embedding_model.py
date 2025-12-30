from sentence_transformers import SentenceTransformer
import numpy as np
from config import Config

class EmbeddingModel:
    def __init__(self, model_name=Config.EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded")
    
    def encode(self, texts, batch_size=32, show_progress=False):
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings.astype('float32')
    
    def get_dimension(self):
        return self.model.get_sentence_embedding_dimension()