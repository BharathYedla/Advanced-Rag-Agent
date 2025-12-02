"""
Advanced RAG Agent - A comprehensive Retrieval-Augmented Generation system.

This package implements an advanced RAG pipeline with modular components for:
- Data Ingestion: Loading and preprocessing documents
- Indexing: Creating and managing vector stores
- Retrieval: Finding relevant information
- Generation: Producing responses with retrieved context
"""

__version__ = "1.0.0"

from .ingestion.document_loader import DocumentLoader
from .ingestion.text_processor import TextProcessor
from .indexing.vector_store import VectorStore
from .indexing.embeddings import EmbeddingGenerator
from .retrieval.retriever import Retriever
from .retrieval.reranker import Reranker
from .generation.generator import Generator
from .generation.prompt_templates import PromptTemplates

__all__ = [
    "DocumentLoader",
    "TextProcessor",
    "VectorStore",
    "EmbeddingGenerator",
    "Retriever",
    "Reranker",
    "Generator",
    "PromptTemplates",
]
