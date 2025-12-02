"""
Text Processor for Advanced RAG System.

Handles text preprocessing, chunking, and preparation for indexing.
"""

import re
from typing import List, Dict, Any, Optional
from .document_loader import Document


class TextChunk:
    """Represents a chunk of text with metadata."""
    
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a TextChunk.
        
        Args:
            text: The chunk text
            metadata: Optional metadata about the chunk
        """
        self.text = text
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"TextChunk(length={len(self.text)}, metadata={self.metadata})"


class TextProcessor:
    """
    Text processor for preprocessing and chunking documents.
    
    Implements various chunking strategies for optimal retrieval:
    - Fixed-size chunking with overlap
    - Sentence-based chunking
    - Paragraph-based chunking
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size of a chunk to be kept
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, method: str = "fixed") -> List[TextChunk]:
        """
        Split text into chunks using the specified method.
        
        Args:
            text: Text to chunk
            method: Chunking method ("fixed", "sentence", or "paragraph")
            
        Returns:
            List of TextChunk objects
        """
        if method == "fixed":
            return self._chunk_fixed_size(text)
        elif method == "sentence":
            return self._chunk_by_sentences(text)
        elif method == "paragraph":
            return self._chunk_by_paragraphs(text)
        else:
            raise ValueError(f"Unknown chunking method: {method}")
    
    def _chunk_fixed_size(self, text: str) -> List[TextChunk]:
        """
        Chunk text into fixed-size pieces with overlap.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Only add chunks that meet minimum size
            if len(chunk_text) >= self.min_chunk_size:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'chunk_index': len(chunks),
                        'start_char': start,
                        'end_char': end,
                        'method': 'fixed_size'
                    }
                )
                chunks.append(chunk)
            
            # Move start position with overlap
            # If overlap is too large, ensure we still make progress
            if self.chunk_overlap >= self.chunk_size:
                start = end
            else:
                start = end - self.chunk_overlap
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[TextChunk]:
        """
        Chunk text by sentences, grouping them to approximate chunk_size.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of TextChunk objects
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_size + sentence_len > self.chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk)
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'chunk_index': len(chunks),
                        'method': 'sentence',
                        'sentence_count': len(current_chunk)
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap (keep last sentence)
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1], sentence]
                    current_size = len(current_chunk[-1]) + sentence_len
                else:
                    current_chunk = [sentence]
                    current_size = sentence_len
            else:
                current_chunk.append(sentence)
                current_size += sentence_len
        
        # Add remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'chunk_index': len(chunks),
                        'method': 'sentence',
                        'sentence_count': len(current_chunk)
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str) -> List[TextChunk]:
        """
        Chunk text by paragraphs.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of TextChunk objects
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) >= self.min_chunk_size:
                chunk = TextChunk(
                    text=para,
                    metadata={
                        'chunk_index': i,
                        'method': 'paragraph'
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def process_document(
        self,
        document: Document,
        preprocess: bool = True,
        chunk_method: str = "fixed"
    ) -> List[TextChunk]:
        """
        Process a document: preprocess and chunk the text.
        
        Args:
            document: Document to process
            preprocess: Whether to preprocess the text
            chunk_method: Chunking method to use
            
        Returns:
            List of TextChunk objects
        """
        text = document.content
        
        if preprocess:
            text = self.preprocess_text(text)
        
        chunks = self.chunk_text(text, method=chunk_method)
        
        # Add document metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update({
                'source_document': document.metadata.get('source', 'unknown'),
                'filename': document.metadata.get('filename', 'unknown')
            })
        
        return chunks
