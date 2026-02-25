import numpy as np
from sentence_transformers import SentenceTransformer
from config import Config


class EmbeddingManager:
    def __init__(self, model_name: str = None):
        self.config = Config()
        self.model_name = model_name or self.config.EMBEDDING_MODEL
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model"""
        self.model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def embed_document_chunks(self, chunks: list[str]) -> list[np.ndarray]:
        """Embed document chunks for RAG"""
        return self.embed_texts(chunks)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        if self.model is None:
            self._load_model()
        return self.model.get_sentence_embedding_dimension()
