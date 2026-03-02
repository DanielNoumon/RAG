"""Compare different reranker implementations on the same test data."""
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    return f"{seconds:.2f}s"


def compare_rerankers(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int = 5,
    rerankers_to_test: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare multiple rerankers on the same query and chunks.
    
    Args:
        query: The search query
        chunks: List of chunks to rerank
        top_n: Number of top results to return
        rerankers_to_test: List of reranker names to test.
                          Options: 'llm', 'cross_encoder', 'colbert', 'bge'
                          None = test all available
    
    Returns:
        Dictionary with comparison results including timing and rankings
    """
    if rerankers_to_test is None:
        rerankers_to_test = ['cross_encoder', 'bge', 'colbert', 'llm']
    
    results = {}
    
    # Test Cross-Encoder
    if 'cross_encoder' in rerankers_to_test:
        print("\n" + "=" * 70)
        print("Testing Cross-Encoder Reranker")
        print("=" * 70)
        try:
            from retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
            
            reranker = CrossEncoderReranker(top_n=len(chunks))
            start_time = time.time()
            reranked = reranker.rerank(query, chunks, top_n=len(chunks), verbose=False)
            elapsed = time.time() - start_time
            
            results['cross_encoder'] = {
                'reranked_chunks': reranked,
                'latency': elapsed,
                'model': reranker.model_name,
                'status': 'success'
            }
            print(f"✓ Completed in {format_time(elapsed)}")
        except Exception as e:
            results['cross_encoder'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"✗ Error: {e}")
    
    # Test BGE Reranker
    if 'bge' in rerankers_to_test:
        print("\n" + "=" * 70)
        print("Testing BGE Reranker (2B LLM)")
        print("=" * 70)
        try:
            from retrieval.rerankers.bge_reranker import BGEReranker
            
            reranker = BGEReranker(top_n=len(chunks))
            start_time = time.time()
            reranked = reranker.rerank(query, chunks, top_n=len(chunks), verbose=False)
            elapsed = time.time() - start_time
            
            results['bge'] = {
                'reranked_chunks': reranked,
                'latency': elapsed,
                'model': reranker.model_name,
                'status': 'success'
            }
            print(f"✓ Completed in {format_time(elapsed)}")
        except Exception as e:
            results['bge'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"✗ Error: {e}")
    
    # Test ColBERT
    if 'colbert' in rerankers_to_test:
        print("\n" + "=" * 70)
        print("Testing ColBERT Reranker")
        print("=" * 70)
        try:
            from retrieval.rerankers.colbert_reranker import ColBERTReranker
            
            reranker = ColBERTReranker(top_n=len(chunks))
            start_time = time.time()
            reranked = reranker.rerank(query, chunks, top_n=len(chunks), verbose=False)
            elapsed = time.time() - start_time
            
            results['colbert'] = {
                'reranked_chunks': reranked,
                'latency': elapsed,
                'model': reranker.model_name,
                'status': 'success'
            }
            print(f"✓ Completed in {format_time(elapsed)}")
        except Exception as e:
            results['colbert'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"✗ Error: {e}")
    
    # Test LLM Reranker
    if 'llm' in rerankers_to_test:
        print("\n" + "=" * 70)
        print("Testing LLM Reranker (Azure OpenAI)")
        print("=" * 70)
        try:
            from retrieval.rerankers.llm_reranker import Reranker
            
            reranker = Reranker(top_n=len(chunks))
            start_time = time.time()
            reranked = reranker.rerank(query, chunks, top_n=len(chunks), verbose=False)
            elapsed = time.time() - start_time
            
            results['llm'] = {
                'reranked_chunks': reranked,
                'latency': elapsed,
                'model': reranker.model_name,
                'status': 'success'
            }
            print(f"✓ Completed in {format_time(elapsed)}")
        except Exception as e:
            results['llm'] = {
                'status': 'error',
                'error': str(e)
            }
            print(f"✗ Error: {e}")
    
    return results


def save_results(
    results: Dict[str, Any],
    query: str,
    total_chunks: int,
    output_dir: str = None,
) -> str:
    """Save comparison results to JSON file.
    
    Args:
        results: Dictionary with comparison results from compare_rerankers
        query: The search query used
        total_chunks: Total number of chunks processed
        output_dir: Directory to save results (defaults to script directory)
    
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"reranker_comparison_{timestamp}.json"
    
    # Prepare data for export
    export_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "total_chunks_processed": total_chunks,
        },
        "rerankers": {}
    }
    
    # Add results for each reranker
    for reranker_name, result in results.items():
        if result['status'] == 'success':
            # Get all chunks with their scores and rankings
            all_chunks = result['reranked_chunks']
            
            export_data['rerankers'][reranker_name] = {
                "model": result['model'],
                "latency_seconds": result['latency'],
                "latency_formatted": format_time(result['latency']),
                "total_chunks_ranked": len(all_chunks),
                "chunks": [
                    {
                        "rank": idx + 1,
                        "chunk_id": chunk.get('chunk_id'),
                        "score": chunk.get('rerank_score'),
                        "content": chunk.get('content'),
                        "source": chunk.get('source'),
                    }
                    for idx, chunk in enumerate(all_chunks)
                ]
            }
        else:
            export_data['rerankers'][reranker_name] = {
                "status": "error",
                "error": result.get('error', 'Unknown error')
            }
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return str(output_file)


def print_comparison_table(results: Dict[str, Any], query: str, top_n: int = 5):
    """Print a formatted comparison table of reranker results."""
    print("\n" + "=" * 70)
    print("RERANKER COMPARISON RESULTS")
    print("=" * 70)
    print(f"\nQuery: {query}")
    print(f"Top-{top_n} results per reranker\n")
    
    # Latency comparison
    print("-" * 70)
    print("LATENCY COMPARISON")
    print("-" * 70)
    print(f"{'Reranker':<20} {'Model':<35} {'Latency':<15}")
    print("-" * 70)
    
    for name, result in results.items():
        if result['status'] == 'success':
            model = result['model']
            if len(model) > 33:
                model = model[:30] + "..."
            latency = format_time(result['latency'])
            print(f"{name:<20} {model:<35} {latency:<15}")
        else:
            print(f"{name:<20} {'ERROR':<35} {'-':<15}")
    
    # Ranking comparison
    print("\n" + "-" * 70)
    print("RANKING COMPARISON (Top 5 chunks)")
    print("-" * 70)
    
    for name, result in results.items():
        if result['status'] == 'success':
            print(f"\n{name.upper()} Rankings:")
            chunks = result['reranked_chunks']
            for i, chunk in enumerate(chunks[:top_n], 1):
                chunk_id = chunk.get('chunk_id', '?')
                score = chunk.get('rerank_score', 0)
                content = chunk.get('content', '')[:60]
                print(f"  {i}. [ID:{chunk_id}] Score: {score:>8.4f} - {content}...")
        else:
            print(f"\n{name.upper()}: ERROR - {result.get('error', 'Unknown error')}")
    
    # Overlap analysis
    print("\n" + "-" * 70)
    print("OVERLAP ANALYSIS")
    print("-" * 70)
    
    successful_results = {
        name: [c.get('chunk_id') for c in r['reranked_chunks'][:top_n]]
        for name, r in results.items()
        if r['status'] == 'success'
    }
    
    if len(successful_results) >= 2:
        reranker_names = list(successful_results.keys())
        for i, name1 in enumerate(reranker_names):
            for name2 in reranker_names[i+1:]:
                ids1 = set(successful_results[name1])
                ids2 = set(successful_results[name2])
                overlap = len(ids1 & ids2)
                overlap_pct = (overlap / top_n) * 100
                print(f"{name1} ∩ {name2}: {overlap}/{top_n} chunks ({overlap_pct:.0f}% overlap)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("RERANKER COMPARISON DEMO")
    print("=" * 70)
    
    # Sample query
    query = "Hoeveel vakantiedagen krijg ik volgens het document?"
    
    # Create 20 demo chunks (some relevant, some not)
    demo_chunks = []
    
    # Relevant chunks about vacation
    relevant_content = [
        "Bij een fulltime dienstverband heb je recht op 25 vakantiedagen per jaar.",
        "Vakantiedagen moeten minimaal 2 maanden van tevoren worden aangevraagd.",
        "Bij parttime werk worden vakantiedagen naar rato berekend.",
        "De 25 vakantiedagen bestaan uit 20 wettelijke en 5 bovenwettelijke dagen.",
        "Vakantieaanvragen gaan via de HR portal en moeten door de manager worden goedgekeurd."
    ]
    
    # Irrelevant chunks about other topics
    irrelevant_content = [
        "De dresscode wordt beschreven als smart casual.",
        "Remote werken is mogelijk na goedkeuring van je manager.",
        "Er wordt een flexibele werktijdregeling gehanteerd.",
        "Parkeren is mogelijk in de parkeergarage onder het kantoor.",
        "Lunch wordt voorzien in de bedrijfskantine.",
        "Het kantoor is geopend van 8:30 tot 17:30.",
        "Schoonmaak vindt plaats na werktijd.",
        "De IT helpdesk is bereikbaar via intern telefoonnummer 5555.",
        "Brandveiligheidstraining is verplicht voor alle medewerkers.",
        "Nieuwe medewerkers krijgen een inwerkprogramma van 2 weken.",
        "Salaris wordt maandelijks uitbetaald op de 25e.",
        "Pensioenopbouw is geregeld via het bedrijfspensioenfonds.",
        "Reiskosten worden vergoed op basis van woon-werkafstand.",
        "Ziekteverlof moet gemeld worden bij 8:30 uur.",
        "Er wordt jaarlijks een personeelsfeest georganiseerd."
    ]
    
    # Create chunks with IDs
    for i, content in enumerate(relevant_content + irrelevant_content, 1):
        demo_chunks.append({
            "chunk_id": i,
            "content": content,
            "source": f"Document_{i}.pdf"
        })
    
    print(f"\nTotal chunks: {len(demo_chunks)}")
    print(f"Relevant chunks: {len(relevant_content)} (IDs 1-5)")
    print(f"Irrelevant chunks: {len(irrelevant_content)} (IDs 6-20)")
    
    # Choose which rerankers to test
    # Options: 'cross_encoder', 'bge', 'colbert', 'llm'
    # Comment out any you don't want to test
    rerankers_to_test = [
        'cross_encoder',  # Fast, good baseline
        # 'bge',          # Requires FlagEmbedding, 2B LLM (very slow, ~10GB RAM)
        'colbert',      # Requires colbert model download
        'llm',          # Requires Azure OpenAI credentials
    ]
    
    # Run comparison
    results = compare_rerankers(
        query=query,
        chunks=demo_chunks,
        top_n=5,
        rerankers_to_test=rerankers_to_test
    )
    
    # Print comparison table
    print_comparison_table(results, query, top_n=5)
    
    # Save results to JSON file
    output_file = save_results(
        results=results,
        query=query,
        total_chunks=len(demo_chunks)
    )
    
    print(f"\n✓ Results saved to: {output_file}")
