import os
import json
import numpy as np
import hnswlib
from typing import List, Dict, Any, Optional


class HNSWStorageManager:
    """HNSW-based vector storage for efficient similarity search"""
    
    def __init__(self, storage_file: str = "embeddings_hnsw.json", 
                 index_file: str = "hnsw_index.bin",
                 dim: int = 384,  # Sentence transformer dimension
                 max_elements: int = 10000,
                 ef_construction: int = 50,  # Reduced for small datasets
                 M: int = 8):  # Reduced for small datasets
        self.storage_file = storage_file
        self.index_file = index_file
        self.dim = dim
        self.max_elements = max_elements
        
        # Initialize HNSW index
        self.index = hnswlib.Index(space='cosine', dim=dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M
        )
        
        # Set ef parameter for search (higher = more accurate, slower)
        self.index.set_ef(20)  # Reduced for small datasets
        
        # Load existing data or create new
        self.data = self._load_data()
        self._rebuild_index()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file"""
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"documents": [], "next_id": 1}
    
    def _save_data(self):
        """Save data to JSON file"""
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def _rebuild_index(self):
        """Rebuild HNSW index from stored documents"""
        if self.data["documents"]:
            # Extract embeddings and IDs
            embeddings = []
            ids = []
            
            for doc in self.data["documents"]:
                embedding = np.array(doc["embedding"], dtype=np.float32)
                embeddings.append(embedding)
                ids.append(doc["id"])
            
            embeddings = np.vstack(embeddings)
            ids = np.array(ids, dtype=np.int32)
            
            # Add to HNSW index
            self.index.add_items(embeddings, ids)
    
    def add_document(self, content: str, embedding: np.ndarray, 
                    metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a document to storage and index"""
        doc_id = self.data["next_id"]
        
        document = {
            "id": doc_id,
            "content": content,
            "embedding": embedding.tolist(),
            "metadata": metadata or {}
        }
        
        self.data["documents"].append(document)
        self.data["next_id"] = doc_id + 1
        
        # Add to HNSW index
        embedding_f32 = np.array(embedding, dtype=np.float32)
        self.index.add_items(embedding_f32.reshape(1, -1), np.array([doc_id], dtype=np.int32))
        
        self._save_data()
        return doc_id
    
    def search_similar(self, query_embedding: np.ndarray,
                      limit: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar documents using HNSW"""
        query_embedding_f32 = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search HNSW index (get more results than needed for filtering)
        num_docs = len(self.data["documents"])
        if num_docs == 0:
            return []
        search_limit = min(limit * 3, num_docs)
        labels, distances = self.index.knn_query(query_embedding_f32, k=search_limit)
        
        results = []
        for label, distance in zip(labels[0], distances[0]):
            # Convert distance to similarity (HNSW returns squared L2 distance for cosine)
            similarity = 1 - distance
            
            # Apply threshold filter
            if similarity >= threshold:
                # Find document by ID
                doc = self.get_document_by_id(int(label))
                if doc:
                    results.append({
                        "id": doc["id"],
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "similarity": similarity
                    })
            
            # Stop if we have enough results
            if len(results) >= limit:
                break
        
        return results
    
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
        """Delete a document (note: HNSW doesn't support deletion, so we rebuild)"""
        original_length = len(self.data["documents"])
        self.data["documents"] = [
            doc for doc in self.data["documents"] if doc["id"] != doc_id
        ]
        
        if len(self.data["documents"]) < original_length:
            # Rebuild index after deletion
            self.index = hnswlib.Index(space='cosine', dim=self.dim)
            self.index.init_index(
                max_elements=self.max_elements,
                ef_construction=200,
                M=16
            )
            self.index.set_ef(50)
            self._rebuild_index()
            self._save_data()
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return {
            "total_documents": len(self.data["documents"]),
            "storage_file": self.storage_file,
            "index_file": self.index_file,
            "dimension": self.dim,
            "max_elements": self.max_elements
        }
