"""
Embedding Generator for Advanced RAG System.

Handles generation of embeddings for text chunks.
"""

from typing import List, Optional, Union
import hashlib


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks.
    
    In a production system, this would integrate with actual embedding models
    (e.g., OpenAI, Sentence-Transformers, etc.). This implementation provides
    a simple deterministic embedding for demonstration purposes.
    """
    
    def __init__(self, embedding_dim: int = 384, model_name: str = "default"):
        """
        Initialize the embedding generator.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            model_name: Name of the embedding model to use
        """
        self.embedding_dim = embedding_dim
        self.model_name = model_name
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        # Simple deterministic embedding using hash
        # In production, use actual embedding models
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert hash bytes to floats
        embedding = []
        for i in range(self.embedding_dim):
            # Use modulo to wrap around if needed
            byte_idx = i % len(hash_bytes)
            # Normalize to [-1, 1]
            value = (hash_bytes[byte_idx] / 127.5) - 1.0
            embedding.append(value)
        
        return embedding
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return [self.generate_embedding(text) for text in texts]
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this generator.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimension")
        
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
