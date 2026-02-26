import numpy as np
from typing import List, Dict, Any, Optional
from core.embedding_manager import EmbeddingManager
from core.hnsw_storage import HNSWStorageManager
from core.azure_openai import AzureOpenAIClient


class VectorSearchPipelineHNSW:
    """RAG Pipeline using HNSW for efficient vector search"""
    
    def __init__(self, storage_file: str = "embeddings_hnsw.json"):
        self.embedding_manager = EmbeddingManager()
        self.storage_manager = HNSWStorageManager(storage_file)
        self.openai_client = AzureOpenAIClient()

    def query(self, question: str, top_k: int = 5, similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """Main RAG query method using HNSW"""
        if not question.strip():
            raise ValueError("Question cannot be empty")

        query_embedding = self.embedding_manager.embed_text(question)
        similar_docs = self.storage_manager.search_similar(
            query_embedding, top_k, similarity_threshold
        )

        if not similar_docs:
            response = self.openai_client.generate_response(
                f"I don't have relevant context to answer: {question}"
            )
            return {
                "question": question,
                "answer": response,
                "context_documents": [],
                "similarities": []
            }

        context_docs = [doc["content"] for doc in similar_docs]
        similarities = [doc["similarity"] for doc in similar_docs]

        answer = self.openai_client.generate_rag_response(question, context_docs)

        return {
            "question": question,
            "answer": answer,
            "context_documents": similar_docs,
            "similarities": similarities
        }

    def add_documents(self, documents: List[Dict[str, Any]],
                     chunk_size: int = 500, overlap: int = 50) -> List[int]:
        """Add documents to the RAG system"""
        from document_processor_v2 import DocumentProcessor
        
        processor = DocumentProcessor(self.storage_manager.storage_file)
        return processor.process_documents_batch(documents, chunk_size, overlap)

    def search_documents(self, query: str, top_k: int = 5,
                        similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar documents without generating response"""
        query_embedding = self.embedding_manager.embed_text(query)
        return self.storage_manager.search_similar(query_embedding, top_k, similarity_threshold)

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        stats = self.storage_manager.get_stats()
        stats["embedding_dimension"] = self.embedding_manager.get_embedding_dimension()
        return stats
