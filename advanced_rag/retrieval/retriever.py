"""
Retriever for Advanced RAG System.

Handles retrieval of relevant documents from the vector store.
"""

from typing import List, Dict, Any, Optional, Tuple
from ..indexing.vector_store import VectorStore, VectorStoreEntry


class RetrievalResult:
    """Represents a retrieval result with content and metadata."""
    
    def __init__(
        self,
        text: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a retrieval result.
        
        Args:
            text: Retrieved text
            score: Relevance score
            metadata: Optional metadata
        """
        self.text = text
        self.score = score
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"RetrievalResult(score={self.score:.3f}, text_length={len(self.text)})"


class Retriever:
    """
    Retriever for finding relevant documents.
    
    Implements various retrieval strategies:
    - Basic similarity search
    - Hybrid retrieval (combining multiple methods)
    - Filtered retrieval based on metadata
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store: Vector store to retrieve from
            top_k: Number of results to retrieve by default
            similarity_threshold: Minimum similarity score threshold
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            k: Number of results to retrieve (uses default if not provided)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of RetrievalResult objects
        """
        k = k or self.top_k
        
        # Perform similarity search
        raw_results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )
        
        # Convert to RetrievalResult objects and filter by threshold
        results = []
        for entry, score in raw_results:
            if score >= self.similarity_threshold:
                result = RetrievalResult(
                    text=entry.text,
                    score=score,
                    metadata=entry.metadata.copy()
                )
                results.append(result)
        
        return results
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve relevant documents with their scores.
        
        Args:
            query: Query string
            k: Number of results to retrieve
            filter_metadata: Optional metadata filter
            
        Returns:
            List of tuples (text, score)
        """
        results = self.retrieve(query, k, filter_metadata)
        return [(result.text, result.score) for result in results]
    
    def retrieve_contexts(
        self,
        query: str,
        k: Optional[int] = None,
        separator: str = "\n\n"
    ) -> str:
        """
        Retrieve relevant contexts and combine them into a single string.
        
        Args:
            query: Query string
            k: Number of results to retrieve
            separator: Separator between contexts
            
        Returns:
            Combined context string
        """
        results = self.retrieve(query, k)
        contexts = [result.text for result in results]
        return separator.join(contexts)
    
    def hybrid_retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        boost_metadata: Optional[Dict[str, float]] = None
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval that combines similarity search with metadata boosting.
        
        Args:
            query: Query string
            k: Number of results to retrieve
            boost_metadata: Dict mapping metadata keys to boost factors
            
        Returns:
            List of RetrievalResult objects with boosted scores
        """
        results = self.retrieve(query, k)
        
        if boost_metadata:
            for result in results:
                # Apply metadata boosts
                for key, boost_factor in boost_metadata.items():
                    if key in result.metadata and result.metadata[key]:
                        result.score *= boost_factor
            
            # Re-sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
        
        return results
