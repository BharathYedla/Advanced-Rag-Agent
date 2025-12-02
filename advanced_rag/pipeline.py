"""
Advanced RAG Pipeline - Main orchestration module.

Combines all components into a complete RAG pipeline.
"""

from typing import List, Optional, Dict, Any
from .ingestion.document_loader import DocumentLoader, Document
from .ingestion.text_processor import TextProcessor, TextChunk
from .indexing.embeddings import EmbeddingGenerator
from .indexing.vector_store import VectorStore
from .retrieval.retriever import Retriever, RetrievalResult
from .retrieval.reranker import Reranker
from .generation.generator import Generator
from .utils.config import Config
from .utils.logger import setup_logger


class AdvancedRAGPipeline:
    """
    Complete Advanced RAG Pipeline.
    
    Orchestrates all components of the RAG system:
    1. Data Ingestion: Load and preprocess documents
    2. Indexing: Create and manage vector store
    3. Retrieval: Find relevant information
    4. Generation: Produce responses with context
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        self.logger = setup_logger()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("Advanced RAG Pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        # Ingestion
        self.document_loader = DocumentLoader()
        self.text_processor = TextProcessor(
            chunk_size=self.config.get('chunk_size'),
            chunk_overlap=self.config.get('chunk_overlap'),
            min_chunk_size=self.config.get('min_chunk_size')
        )
        
        # Indexing
        self.embedding_generator = EmbeddingGenerator(
            embedding_dim=self.config.get('embedding_dim'),
            model_name=self.config.get('embedding_model')
        )
        self.vector_store = VectorStore(self.embedding_generator)
        
        # Retrieval
        self.retriever = Retriever(
            vector_store=self.vector_store,
            top_k=self.config.get('top_k'),
            similarity_threshold=self.config.get('similarity_threshold')
        )
        self.reranker = Reranker()
        
        # Generation
        self.generator = Generator(
            model_name=self.config.get('model_name'),
            temperature=self.config.get('temperature'),
            max_tokens=self.config.get('max_tokens')
        )
    
    def ingest_documents(
        self,
        documents: List[Document],
        chunk_method: Optional[str] = None
    ) -> int:
        """
        Ingest documents into the pipeline.
        
        Args:
            documents: List of documents to ingest
            chunk_method: Optional chunking method override
            
        Returns:
            Number of chunks created and indexed
        """
        chunk_method = chunk_method or self.config.get('chunk_method')
        total_chunks = 0
        
        self.logger.info(f"Ingesting {len(documents)} documents")
        
        for doc in documents:
            # Process document into chunks
            chunks = self.text_processor.process_document(
                document=doc,
                preprocess=True,
                chunk_method=chunk_method
            )
            
            # Add chunks to vector store
            for chunk in chunks:
                self.vector_store.add_text(
                    text=chunk.text,
                    metadata=chunk.metadata
                )
                total_chunks += 1
        
        self.logger.info(f"Ingested {total_chunks} chunks from {len(documents)} documents")
        return total_chunks
    
    def ingest_from_file(self, file_path: str) -> int:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Number of chunks created
        """
        doc = self.document_loader.load_file(file_path)
        return self.ingest_documents([doc])
    
    def ingest_from_directory(
        self,
        directory_path: str,
        recursive: bool = True
    ) -> int:
        """
        Ingest all files from a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to recursively load subdirectories
            
        Returns:
            Number of chunks created
        """
        docs = self.document_loader.load_directory(directory_path, recursive)
        return self.ingest_documents(docs)
    
    def query(
        self,
        question: str,
        use_reranking: bool = False,
        rerank_method: str = "score"
    ) -> str:
        """
        Query the RAG system.
        
        Args:
            question: Question to answer
            use_reranking: Whether to apply reranking
            rerank_method: Reranking method to use
            
        Returns:
            Generated answer
        """
        self.logger.info(f"Processing query: {question}")
        
        # Retrieve relevant contexts
        results = self.retriever.retrieve(question)
        
        if not results:
            self.logger.warning("No relevant contexts found")
            return "I don't have enough information to answer that question."
        
        # Optional reranking
        if use_reranking:
            results = self.reranker.rerank(
                query=question,
                results=results,
                method=rerank_method
            )
        
        # Extract contexts
        contexts = [result.text for result in results]
        
        # Generate response
        response = self.generator.generate_with_contexts(
            question=question,
            contexts=contexts
        )
        
        self.logger.info("Query processed successfully")
        return response
    
    def get_relevant_contexts(
        self,
        question: str,
        k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Get relevant contexts without generating a response.
        
        Args:
            question: Query question
            k: Number of contexts to retrieve
            
        Returns:
            List of retrieval results
        """
        return self.retriever.retrieve(question, k=k)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG system.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_chunks': self.vector_store.size(),
            'embedding_dim': self.embedding_generator.get_embedding_dimension(),
            'config': self.config.get_all()
        }
    
    def clear(self):
        """Clear all data from the pipeline."""
        self.vector_store.clear()
        self.logger.info("Pipeline cleared")
