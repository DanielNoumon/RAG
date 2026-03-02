"""ColBERT-based reranker using late interaction for document relevance scoring."""
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel


class ColBERTReranker:
    """Reranks retrieved chunks using ColBERT late interaction.

    ColBERT (Contextualized Late Interaction over BERT) uses a novel
    late interaction architecture where query and document embeddings
    are computed independently, then combined via MaxSim operation.
    This provides better accuracy than bi-encoders while being more
    efficient than cross-encoders.

    Default model: ``colbert-ir/colbertv2.0`` — state-of-the-art
    retrieval model trained on MS MARCO.

    The public ``rerank()`` method follows the same signature as the
    other rerankers so they can be used interchangeably in retrieval
    pipelines.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        top_n: int = 5,
        device: Optional[str] = None,
    ):
        """Initialise the ColBERT reranker.

        Args:
            model_name: HuggingFace ColBERT model identifier.
            top_n: Default number of chunks to keep after reranking.
            device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
                    ``None`` lets torch pick automatically.
        """
        self.model_name = model_name
        self.top_n = top_n
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        print(f"ColBERT Reranker loaded: {model_name} on {self.device}")

    def _compute_score(self, query: str, passage: str) -> float:
        """Compute ColBERT late interaction score for a query-passage pair."""
        # Tokenize query and passage separately
        query_inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,  # Queries are typically short
        ).to(self.device)
        
        passage_inputs = self.tokenizer(
            passage,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            # Get contextualized embeddings for query and passage
            query_embeddings = self.model(**query_inputs).last_hidden_state
            passage_embeddings = self.model(**passage_inputs).last_hidden_state
            
            # Late interaction: MaxSim operation
            # For each query token, find max similarity with any passage token
            # Shape: [batch, query_len, hidden_dim] x [batch, passage_len, hidden_dim]
            similarity_matrix = torch.matmul(
                query_embeddings, passage_embeddings.transpose(1, 2)
            )  # [batch, query_len, passage_len]
            
            # Max over passage tokens for each query token
            max_sims = similarity_matrix.max(dim=2).values  # [batch, query_len]
            
            # Sum over query tokens (ColBERT scoring)
            score = max_sims.sum(dim=1).item()
        
        return score

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        content_key: str = "content",
        verbose: bool = True,
        batch_size: int = 1,  # ColBERT typically processes one at a time
    ) -> List[Dict[str, Any]]:
        """Rerank chunks by ColBERT late interaction relevance to the query.

        Args:
            query: The user query.
            chunks: Retrieved chunks; each dict must contain *content_key*.
            top_n: How many to keep (defaults to ``self.top_n``).
            content_key: Key in each chunk dict that holds the text content.
            verbose: Whether to print progress information.
            batch_size: Not used for ColBERT (processes sequentially).

        Returns:
            Top-n chunks sorted descending by ColBERT relevance score.
        """
        n = top_n or self.top_n
        if not chunks:
            return []

        if verbose:
            print(
                f"Reranking {len(chunks)} chunks with ColBERT "
                f"({self.model_name})..."
            )

        # Score all chunks
        scored_chunks = []
        for chunk in chunks:
            passage = chunk.get(content_key, "")
            score = self._compute_score(query, passage)
            
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
    print("=== ColBERT Reranker Demo ===")
    
    # Configuration
    VERBOSE_LOGGING = True  # Set to False to disable progress logging
    
    # Create reranker instance
    reranker = ColBERTReranker()
    
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
