"""
Vector Store for Advanced RAG System.

Manages storage and retrieval of document embeddings.
"""

from typing import List, Dict, Any, Optional, Tuple
from .embeddings import EmbeddingGenerator


class VectorStoreEntry:
    """Represents an entry in the vector store."""
    
    def __init__(
        self,
        id: str,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a vector store entry.
        
        Args:
            id: Unique identifier for the entry
            text: The text content
            embedding: The embedding vector
            metadata: Optional metadata
        """
        self.id = id
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"VectorStoreEntry(id={self.id}, text_length={len(self.text)})"


class VectorStore:
    """
    Vector store for managing document embeddings and similarity search.
    
    Provides functionality for:
    - Adding documents with embeddings
    - Similarity search
    - Index management
    """
    
    def __init__(self, embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize the vector store.
        
        Args:
            embedding_generator: Optional embedding generator instance
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.entries: Dict[str, VectorStoreEntry] = {}
        self._next_id = 0
    
    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ) -> str:
        """
        Add a text to the vector store.
        
        Args:
            text: Text to add
            metadata: Optional metadata
            id: Optional custom ID (auto-generated if not provided)
            
        Returns:
            The ID of the added entry
        """
        # Generate ID if not provided
        if id is None:
            id = f"doc_{self._next_id}"
            self._next_id += 1
        
        # Generate embedding
        embedding = self.embedding_generator.generate_embedding(text)
        
        # Create and store entry
        entry = VectorStoreEntry(
            id=id,
            text=text,
            embedding=embedding,
            metadata=metadata
        )
        self.entries[id] = entry
        
        return id
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of IDs for the added entries
        """
        if metadatas is None:
            metadatas = [None] * len(texts)
        
        ids = []
        for text, metadata in zip(texts, metadatas):
            id = self.add_text(text, metadata)
            ids.append(id)
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[VectorStoreEntry, float]]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of tuples (entry, similarity_score) sorted by similarity
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Calculate similarities
        results = []
        for entry in self.entries.values():
            # Apply metadata filter if provided
            if filter_metadata:
                if not all(
                    entry.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                ):
                    continue
            
            # Calculate similarity
            similarity = self.embedding_generator.cosine_similarity(
                query_embedding,
                entry.embedding
            )
            results.append((entry, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return results[:k]
    
    def get_by_id(self, id: str) -> Optional[VectorStoreEntry]:
        """
        Retrieve an entry by its ID.
        
        Args:
            id: Entry ID
            
        Returns:
            VectorStoreEntry if found, None otherwise
        """
        return self.entries.get(id)
    
    def delete(self, id: str) -> bool:
        """
        Delete an entry by its ID.
        
        Args:
            id: Entry ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if id in self.entries:
            del self.entries[id]
            return True
        return False
    
    def clear(self):
        """Clear all entries from the vector store."""
        self.entries.clear()
        self._next_id = 0
    
    def size(self) -> int:
        """
        Get the number of entries in the vector store.
        
        Returns:
            Number of entries
        """
        return len(self.entries)
    
    def get_all_ids(self) -> List[str]:
        """
        Get all entry IDs in the vector store.
        
        Returns:
            List of entry IDs
        """
        return list(self.entries.keys())
