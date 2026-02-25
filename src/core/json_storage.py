import json
import os
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime


class JSONStorageManager:
    def __init__(self, storage_file: str = "embeddings.json"):
        self.storage_file = storage_file
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"documents": [], "next_id": 1}
        return {"documents": [], "next_id": 1}

    def _save_data(self):
        """Save data to JSON file"""
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def insert_document(self, content: str, embedding: np.ndarray,
                       metadata: Dict[str, Any] = None) -> int:
        """Insert a document with its embedding"""
        doc_id = self.data["next_id"]
        
        document = {
            "id": doc_id,
            "content": content,
            "embedding": embedding.tolist(),
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        
        self.data["documents"].append(document)
        self.data["next_id"] = doc_id + 1
        self._save_data()
        
        return doc_id

    def search_similar(self, query_embedding: np.ndarray,
                      limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity"""
        similarities = []
        
        for doc in self.data["documents"]:
            doc_embedding = np.array(doc["embedding"])
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            if similarity >= threshold:
                similarities.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "similarity": similarity
                })
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:limit]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)

    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific document by ID"""
        for doc in self.data["documents"]:
            if doc["id"] == doc_id:
                return {
                    "id": doc["id"],
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                }
        return None

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents"""
        return self.data["documents"]

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document by ID"""
        original_length = len(self.data["documents"])
        self.data["documents"] = [
            doc for doc in self.data["documents"] if doc["id"] != doc_id
        ]
        
        if len(self.data["documents"]) < original_length:
            self._save_data()
            return True
        return False

    def clear_all_documents(self):
        """Clear all documents"""
        self.data["documents"] = []
        self.data["next_id"] = 1
        self._save_data()
