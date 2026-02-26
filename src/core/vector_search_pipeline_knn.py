from typing import List, Dict, Any, Optional
from core.embedding_manager import EmbeddingManager
from core.json_storage import JSONStorageManager
from core.azure_openai import AzureOpenAIClient


class VectorSearchPipelineKNN:
    def __init__(self, storage_file: str = "embeddings.json"):
        self.embedding_manager = EmbeddingManager()
        self.storage_manager = JSONStorageManager(storage_file)
        self.openai_client = AzureOpenAIClient()

    def query(self, question: str, top_k: int = 5, similarity_threshold: float = 0.0,
              max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """Main RAG query method"""
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
                     chunk_size: int = 500, overlap: int = 50) -> List[List[int]]:
        """Add documents to the RAG system"""
        from document_processor_v2 import DocumentProcessor
        
        processor = DocumentProcessor(self.storage_manager.storage_file)
        return processor.process_documents_batch(documents, chunk_size, overlap)

    def search_documents(self, query: str, top_k: int = 5,
                        similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar documents without generating response"""
        query_embedding = self.embedding_manager.embed_text(query)
        return self.storage_manager.search_similar(query_embedding, top_k, similarity_threshold)

    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID"""
        return self.storage_manager.get_document_by_id(doc_id)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents in storage"""
        return self.storage_manager.get_all_documents()

    def clear_all_documents(self):
        """Clear all documents from storage"""
        self.storage_manager.clear_all_documents()

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        documents = self.storage_manager.get_all_documents()
        return {
            "total_documents": len(documents),
            "storage_file": self.storage_manager.storage_file,
            "embedding_dimension": self.embedding_manager.get_embedding_dimension()
        }
