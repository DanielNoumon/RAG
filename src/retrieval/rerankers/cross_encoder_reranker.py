"""Cross-encoder reranker using sentence-transformers for document relevance scoring."""
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model.

    Cross-encoders jointly encode (query, document) pairs and produce a
    relevance score.  They are slower than bi-encoders but significantly
    more accurate for reranking because the model can attend across both
    texts simultaneously.

    Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — a fast,
    lightweight model trained on MS MARCO that works well for general
    passage reranking.

    The public ``rerank()`` method follows the same signature as the
    LLM-based ``Reranker`` so the two can be used interchangeably in
    retrieval pipelines.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        device: Optional[str] = None,
    ):
        """Initialise the cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model identifier.
            top_n: Default number of chunks to keep after reranking.
            device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
                    ``None`` lets sentence-transformers pick automatically.
        """
        self.model_name = model_name
        self.top_n = top_n
        self.model = CrossEncoder(model_name, device=device)
        print(f"Cross-Encoder Reranker loaded: {model_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        content_key: str = "content",
        verbose: bool = True,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks by cross-encoder relevance to the query.

        Args:
            query: The user query.
            chunks: Retrieved chunks; each dict must contain *content_key*.
            top_n: How many to keep (defaults to ``self.top_n``).
            content_key: Key in each chunk dict that holds the text content.
            verbose: Whether to print progress information.
            batch_size: Batch size passed to the cross-encoder ``predict``.

        Returns:
            Top-n chunks sorted descending by cross-encoder relevance score.
        """
        n = top_n or self.top_n
        if not chunks:
            return []

        if verbose:
            print(
                f"Reranking {len(chunks)} chunks with cross-encoder "
                f"({self.model_name}, batch_size={batch_size})..."
            )

        # Build (query, passage) pairs
        pairs = [
            [query, chunk.get(content_key, "")]
            for chunk in chunks
        ]

        # Score all pairs in one call (the model handles internal batching)
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=verbose,
        )

        # Attach scores to chunks
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_with_score = chunk.copy()
            chunk_with_score["rerank_score"] = float(score)
            scored_chunks.append(chunk_with_score)

        # Sort descending by score and keep top_n
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        if verbose:
            print(f"  Reranking complete. Keeping top {n}/{len(chunks)} chunks.")

        return scored_chunks[:n]


if __name__ == "__main__":
    # Demo: Test reranker with sample chunks
    print("=== Cross-Encoder Reranker Demo ===")
    
    # Configuration
    VERBOSE_LOGGING = True  # Set to False to disable progress logging
    
    # Create reranker instance
    reranker = CrossEncoderReranker()
    
    # Sample query
    query = "Hoeveel vakantiedagen staan er in het document en hoe moet je deze aanvragen?"
    
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
    
    print(f"\nQuery: {query}")
    print(f"Total chunks: {len(demo_chunks)}")
    print("\n=== Before Reranking (showing first 5) ===")
    for i, chunk in enumerate(demo_chunks[:5], 1):
        print(f"{i}. [{chunk['chunk_id']}] {chunk['content'][:60]}...")
    
    # Rerank to top 5
    print("\n=== Reranking to top 5 ===")
    reranked = reranker.rerank(query, demo_chunks, top_n=5, verbose=VERBOSE_LOGGING)
    
    print("\n=== After Reranking (top 5) ===")
    for i, chunk in enumerate(reranked, 1):
        print(f"{i}. [ID:{chunk['chunk_id']}] Score: {chunk['rerank_score']:.4f} - {chunk['content'][:60]}...")
    
    print("\n=== Demo Complete ===")
