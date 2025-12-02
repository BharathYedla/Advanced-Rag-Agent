"""
Reranker for Advanced RAG System.

Provides reranking functionality to improve retrieval results.
"""

from typing import List, Callable, Optional
from .retriever import RetrievalResult


class Reranker:
    """
    Reranker for improving retrieval results.
    
    Implements various reranking strategies to refine the initial retrieval results:
    - Score-based reranking
    - Diversity-based reranking
    - Custom scoring functions
    """
    
    def __init__(self):
        """Initialize the reranker."""
        pass
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        method: str = "score"
    ) -> List[RetrievalResult]:
        """
        Rerank retrieval results using the specified method.
        
        Args:
            query: Original query string
            results: List of retrieval results to rerank
            method: Reranking method ("score", "diversity", or "length")
            
        Returns:
            Reranked list of RetrievalResult objects
        """
        if method == "score":
            # Already sorted by score, just return
            return results
        elif method == "diversity":
            return self._rerank_for_diversity(results)
        elif method == "length":
            return self._rerank_by_length(results)
        else:
            raise ValueError(f"Unknown reranking method: {method}")
    
    def _rerank_for_diversity(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank results to maximize diversity while maintaining relevance.
        
        Uses Maximal Marginal Relevance (MMR) approach.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Reranked results
        """
        if len(results) <= 1:
            return results
        
        # Simple diversity-based reranking
        # Select results that are different from already selected ones
        reranked = [results[0]]  # Start with highest scoring result
        remaining = results[1:]
        
        while remaining:
            # Find result most different from already selected
            max_min_distance = -1
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Calculate minimum distance to selected results
                min_distance = float('inf')
                for selected in reranked:
                    # Simple text-based diversity metric
                    distance = self._text_distance(candidate.text, selected.text)
                    min_distance = min(min_distance, distance)
                
                # Weight by original score and diversity
                score = candidate.score * 0.5 + min_distance * 0.5
                
                if score > max_min_distance:
                    max_min_distance = score
                    best_idx = i
            
            reranked.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return reranked
    
    def _rerank_by_length(
        self,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank results preferring medium-length texts.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Reranked results
        """
        # Prefer texts around 500 characters
        optimal_length = 500
        
        def length_score(result: RetrievalResult) -> float:
            length = len(result.text)
            length_penalty = abs(length - optimal_length) / optimal_length
            # Combine original score with length preference
            return result.score * (1.0 - 0.3 * length_penalty)
        
        reranked = sorted(results, key=length_score, reverse=True)
        return reranked
    
    def _text_distance(self, text1: str, text2: str) -> float:
        """
        Calculate a simple distance metric between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Distance score (0 = identical, 1 = completely different)
        """
        # Simple word-based Jaccard distance
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 1.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        jaccard_similarity = intersection / union
        return 1.0 - jaccard_similarity
    
    def custom_rerank(
        self,
        results: List[RetrievalResult],
        scoring_function: Callable[[RetrievalResult], float]
    ) -> List[RetrievalResult]:
        """
        Rerank results using a custom scoring function.
        
        Args:
            results: List of retrieval results
            scoring_function: Function that takes a RetrievalResult and returns a score
            
        Returns:
            Reranked results
        """
        reranked = sorted(results, key=scoring_function, reverse=True)
        return reranked
