"""Data ingestion module for loading and preprocessing documents."""

from .document_loader import DocumentLoader
from .text_processor import TextProcessor

__all__ = ["DocumentLoader", "TextProcessor"]
