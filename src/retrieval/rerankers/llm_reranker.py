"""LLM-based reranker using Azure OpenAI to score document relevance."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator

from core.azure_openai import AzureOpenAIClient
from core.config import Config
from core.prompts import RERANKER_SYSTEM_PROMPT


class DocumentRanking(BaseModel):
    """Individual document ranking from LLM."""
    document_number: int = Field(..., description="Document number (1-indexed)")
    reasoning: str = Field(..., description="Reasoning for relevance score")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0 and 1")
    answer_evidence: Optional[str] = Field("", description="Exact sentence or phrase that answers the query")
    
    @validator('relevance_score')
    def validate_score(cls, v):
        return max(0.0, min(1.0, float(v)))


class RerankerResponse(BaseModel):
    """Complete LLM reranking response."""
    block_rankings: List[DocumentRanking] = Field(..., description="List of document rankings")


class Reranker:
    """Reranks retrieved chunks using Azure OpenAI LLM.
    
    Uses the LLM to score relevance of each chunk to the query.
    More accurate than cross-encoders for domain-specific content,
    especially with Dutch text.
    """
    
    def __init__(self, model_name: str = None, top_n: int = 5):
        """Initialize reranker with Azure OpenAI client.
        
        Args:
            model_name: Azure OpenAI deployment name (uses config default if None)
            top_n: Number of chunks to keep after reranking
        """
        self.config = Config()
        self.model_name = model_name or self.config.AZURE_OPENAI_DEPLOYMENT_NAME
        self.top_n = top_n
        self.client = AzureOpenAIClient()
        print(f"LLM Reranker loaded: {self.model_name}")
    
    def _score_chunk(self, query: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single chunk for relevance to the query."""
        content = chunk.get("content", "")
        
        # Simple prompt for scoring
        prompt = f"""Geef een relevantiescore van 0.0 tot 1.0 voor de volgende tekst ten opzichte van de vraag.

Vraag: "{query}"

Tekst: "{content}"

Antwoord alleen met een JSON object met een "relevance_score" veld:
{{"relevance_score": 0.85}}"""
        
        try:
            response = self.client.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Je bent een expert in het evalueren van documentrelevantie. Geef altijd een JSON response."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            score = float(result.get("relevance_score", 0.5))
            
            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))
            
            return {**chunk, "rerank_score": score}
            
        except Exception as e:
            print(f"Error scoring chunk: {e}")
            return {**chunk, "rerank_score": 0.5}  # Default score
    
    def _score_batch(self, query: str, chunks: List[Dict[str, Any]], start_idx: int, content_key: str = "content") -> List[Dict[str, Any]]:
        """Score a batch of chunks in a single LLM call."""
        # Format chunks with document numbers (relative to this batch)
        formatted_blocks = ""
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get(content_key, "")
            formatted_blocks += f"Document {i}:\n\"\"\"\n{content}\n\"\"\"\n\n"
        
        # Enhanced prompt with structured JSON output
        user_prompt = f"""=== Query ===
"{query}"

=== Instructions ===
You are given {len(chunks)} documents to evaluate for relevance to this query.
Your task is to evaluate each document and return a JSON with EXACTLY this structure:

{{
  "block_rankings": [
    {{
      "document_number": 1,
      "reasoning": "<reasoning for document 1>",
      "relevance_score": <float 0.0-1.0>
    }},
    {{
      "document_number": 2,
      "reasoning": "<reasoning for document 2>",
      "relevance_score": <float 0.0-1.0>
    }},
    ... (one entry per document)
  ]
}}

IMPORTANT: You MUST include "document_number" (1, 2, 3, etc.) in each ranking.
Provide exactly {len(chunks)} rankings.

=== Documents to evaluate ===
{formatted_blocks}"""
        
        max_retries = 3
        result = None
        error_details = None
        for attempt in range(1, max_retries + 1):
            response = self.client.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": RERANKER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
            )

            content = response.choices[0].message.content.strip()

            if not content:
                error_details = f"Empty response from LLM (attempt {attempt})"
                if attempt < max_retries:
                    continue
                raise ValueError(error_details)

            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif content.startswith("```"):
                json_start = content.find("\n") + 1
                json_end = content.rfind("```")
                content = content[json_start:json_end].strip()

            try:
                result = RerankerResponse.parse_raw(content)
                error_details = None
                break
            except Exception as e:
                error_details = f"Invalid LLM response format: {e}\nRaw content: {content}"
                if attempt < max_retries:
                    continue
                raise ValueError(error_details)
        
        # Create mapping from document_number to ranking
        rankings_by_doc_num = {
            rank.document_number: rank for rank in result.block_rankings
        }
        
        # Apply scores to chunks
        scored_chunks = []
        for idx, chunk in enumerate(chunks, start=1):
            rank = rankings_by_doc_num.get(idx)
            
            if rank is None:
                raise ValueError(f"Missing ranking for document {idx}")
            
            chunk_with_score = chunk.copy()
            chunk_with_score["rerank_score"] = rank.relevance_score
            chunk_with_score["reasoning"] = rank.reasoning
            chunk_with_score["answer_evidence"] = rank.answer_evidence
            scored_chunks.append(chunk_with_score)
        
        return scored_chunks
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_n: Optional[int] = None,
        content_key: str = "content",
        verbose: bool = True,
        batch_size: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks by LLM relevance to the query.
        
        Args:
            query: The user query
            chunks: Retrieved chunks, each must have content_key field
            top_n: How many to keep (defaults to self.top_n)
            content_key: Key in each chunk dict that holds text content
            verbose: Whether to print progress (default: True)
            batch_size: Number of chunks to process in parallel (default: 5)
            
        Returns:
            Top-n chunks sorted by LLM relevance score
        """
        n = top_n or self.top_n
        if not chunks:
            return []
        
        if verbose:
            print(f"Reranking {len(chunks)} chunks with LLM (batch size: {batch_size}, parallel processing)...")
        
        # Create batches of chunks
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batches.append((batch, i))  # (chunks, start_idx)
        
        # Process batches in parallel
        all_scored_chunks = []
        with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self._score_batch, query, batch, start_idx, content_key): (batch, start_idx)
                for batch, start_idx in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch, start_idx = future_to_batch[future]
                try:
                    scored_batch = future.result()
                    all_scored_chunks.extend(scored_batch)
                    if verbose:
                        print(f"  Completed batch {start_idx//batch_size + 1}/{len(batches)}")
                except Exception as e:
                    print(f"  Error in batch {start_idx//batch_size + 1}: {e}")
                    raise  # Re-raise the error as requested
        
        # Sort by rerank score and return top_n
        ranked = sorted(all_scored_chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return ranked[:n]


# Backward compatibility alias
LLMReranker = Reranker


if __name__ == "__main__":
    # Demo: Test reranker with sample chunks
    print("=== Reranker Demo ===")
    
    # Configuration
    MODEL_NAME = None  # Azure OpenAI deployment name (None = use config default)
    VERBOSE_LOGGING = True  # Set to False to disable progress logging
    
    # Create reranker instance
    reranker = Reranker(model_name=MODEL_NAME)
    
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
        print(f"{i}. [ID:{chunk['chunk_id']}] Score: {chunk['rerank_score']:.2f} - {chunk['content'][:60]}...")
    
    print("\n=== Demo Complete ===")
