#!/usr/bin/env python3
"""
Script to inspect RAG test results JSON files in a readable format.
Usage: python inspect_results.py [json_file]
"""

import json
import sys
import os
from datetime import datetime

def format_score(score, width=8):
    """Format a score with fixed width and color coding."""
    if score is None:
        return " " * width
    score_val = float(score)
    if score_val >= 0.8:
        return f"{score_val:.3f}".rjust(width)
    elif score_val >= 0.5:
        return f"{score_val:.3f}".rjust(width)
    else:
        return f"{score_val:.3f}".rjust(width)

def truncate_text(text, max_len=80):
    """Truncate text to specified length."""
    if text is None:
        return "N/A"
    if len(text) <= max_len:
        return text
    return text[:max_len-3] + "..."

def inspect_results(json_file):
    """Inspect and display RAG results in a readable format."""
    
    # Load JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading {json_file}: {e}")
        return
    
    print("=" * 80)
    print(f"📊 RAG Results Inspector")
    print("=" * 80)
    print(f"📁 File: {json_file}")
    print(f"🕒 Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"🔍 Search Method: {data.get('search_method', 'N/A')}")
    print(f"⚡ Fusion: {data.get('fusion', 'N/A')}")
    print(f"🎯 Top K: {data.get('top_k', 'N/A')}")
    print(f"🤖 Vector Backend: {data.get('vector_backend', 'N/A')}")
    print("=" * 80)
    
    results = data.get('results', [])
    print(f"📋 Total Questions: {len(results)}")
    print()
    
    for i, result in enumerate(results, 1):
        print(f"❓ Question {i}: {truncate_text(result.get('question', 'N/A'), 70)}")
        print("-" * 80)
        
        # Show answer if available
        answer = result.get('answer', '')
        if answer:
            print(f"💡 Answer: {truncate_text(answer, 100)}")
            print()
        
        # Show initial chunks
        initial_chunks = result.get('initial_chunks', [])
        print(f"🔢 Initial Chunks ({len(initial_chunks)}):")
        print(f"{'#':>3} {'Score':>8} {'Vector':>8} {'BM25':>8} {'BM25-N':>8} {'Selected':>9} | Content")
        print("-" * 100)
        
        for j, chunk in enumerate(initial_chunks, 1):
            score = format_score(chunk.get('fusion_score'))
            vector = format_score(chunk.get('vector_similarity'))
            bm25 = format_score(chunk.get('bm25_score'))
            bm25_norm = format_score(chunk.get('bm25_normalized_score'))
            selected = "✅ YES" if chunk.get('selected_by_reranker') else "❌ NO"
            content = truncate_text(chunk.get('content', ''), 40)
            print(f"{j:>3} {score} {vector} {bm25} {bm25_norm} {selected:>9} | {content}")
        
        print()
        
        # Show reranked chunks
        reranked_chunks = result.get('reranked_chunks', [])
        print(f"🏆 Reranked Chunks ({len(reranked_chunks)}):")
        print(f"{'#':>3} {'Score':>8} {'Vector':>8} {'BM25':>8} {'Rerank':>8} | Content & Reasoning")
        print("-" * 80)
        
        for j, chunk in enumerate(reranked_chunks, 1):
            score = format_score(chunk.get('fusion_score'))
            vector = format_score(chunk.get('vector_similarity'))
            bm25 = format_score(chunk.get('bm25_score'))
            rerank = format_score(chunk.get('rerank_score'))
            content = truncate_text(chunk.get('content', ''), 40)
            reasoning = truncate_text(chunk.get('reasoning', ''), 40)
            print(f"{j:>3} {score} {vector} {bm25} {rerank} | {content}")
            print(f"{'':>3} {'':>8} {'':>8} {'':>8} {'':>8} | 🧠 {reasoning}")
        
        print("\n" + "=" * 80 + "\n")

def find_latest_results():
    """Find the most recent results file."""
    # Get project root (parent of scripts directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, "data/test_results")
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory '{results_dir}' not found")
        return None
    
    # Find all JSON files
    json_files = []
    for file in os.listdir(results_dir):
        if file.endswith('.json'):
            file_path = os.path.join(results_dir, file)
            json_files.append((file_path, os.path.getmtime(file_path)))
    
    if not json_files:
        print(f"❌ No JSON files found in '{results_dir}'")
        return None
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda x: x[1], reverse=True)
    return json_files[0][0]

def main():
    """Main function."""
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Find latest results file
        json_file = find_latest_results()
        if json_file:
            print(f"🔍 Using latest results file: {json_file}")
        else:
            print("❌ No results file found. Please specify a file:")
            print("   python inspect_results.py data/test_results/hybrid_results_20260224_214521.json")
            return
    
    inspect_results(json_file)

if __name__ == "__main__":
    main()
