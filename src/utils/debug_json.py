import json
import numpy as np
from datetime import datetime

# Test the convert_numpy_types function
def convert_numpy_types(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Test data similar to what the script creates
test_output = {
    "timestamp": "20260224_212703",
    "search_method": "hybrid_rrf",
    "results": [
        {
            "question": "test",
            "answer": "test answer",
            "initial_chunks": [
                {
                    "content": "test content",
                    "score": 0.5,
                    "vector_similarity": 0.8,
                    "bm25_score": 2.1,
                    "selected_by_reranker": True
                }
            ],
            "reranked_chunks": [
                {
                    "content": "test content",
                    "score": 0.5,
                    "vector_similarity": 0.8,
                    "bm25_score": 2.1,
                    "rerank_score": 0.9,
                    "reasoning": "test reasoning"
                }
            ]
        }
    ]
}

try:
    converted = convert_numpy_types(test_output)
    json_str = json.dumps(converted, indent=2, ensure_ascii=False)
    print("JSON serialization successful")
    print("Length:", len(json_str))
except Exception as e:
    print("JSON serialization failed:", e)
    import traceback
    traceback.print_exc()
