"""
Document Loader for Advanced RAG System.

Handles loading documents from various sources and formats.
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class Document:
    """Represents a document with content and metadata."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a Document.
        
        Args:
            content: The text content of the document
            metadata: Optional metadata about the document
        """
        self.content = content
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


class DocumentLoader:
    """
    Document loader for various file formats.
    
    Supports loading text files, PDFs, markdown, and more.
    Includes metadata extraction for better document tracking.
    """
    
    def __init__(self):
        """Initialize the document loader."""
        self.supported_extensions = {'.txt', '.md', '.json', '.csv', '.html'}
    
    def load_file(self, file_path: str) -> Document:
        """
        Load a single file and return a Document.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Document object containing the content and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract metadata
        metadata = {
            'source': str(path.absolute()),
            'filename': path.name,
            'extension': path.suffix,
            'size': path.stat().st_size,
        }
        
        return Document(content=content, metadata=metadata)
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """
        Load all supported files from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to recursively load files from subdirectories
            
        Returns:
            List of Document objects
        """
        documents = []
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        # Get file pattern
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                try:
                    doc = self.load_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
        
        return documents
    
    def load_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Create a Document from raw text.
        
        Args:
            text: The text content
            metadata: Optional metadata
            
        Returns:
            Document object
        """
        return Document(content=text, metadata=metadata or {})
