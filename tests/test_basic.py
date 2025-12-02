"""
Tests for the Advanced RAG System components.

This demonstrates basic testing patterns for the RAG pipeline.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advanced_rag.ingestion.document_loader import DocumentLoader, Document
from advanced_rag.ingestion.text_processor import TextProcessor
from advanced_rag.indexing.embeddings import EmbeddingGenerator
from advanced_rag.indexing.vector_store import VectorStore
from advanced_rag.retrieval.retriever import Retriever
from advanced_rag.retrieval.reranker import Reranker
from advanced_rag.generation.generator import Generator
from advanced_rag.pipeline import AdvancedRAGPipeline
from advanced_rag.utils.config import Config


def test_document_loader():
    """Test document loading functionality."""
    print("Testing DocumentLoader...")
    
    loader = DocumentLoader()
    
    # Test loading text
    doc = loader.load_text("Test content", metadata={'source': 'test'})
    assert doc.content == "Test content"
    assert doc.metadata['source'] == 'test'
    
    print("✓ DocumentLoader tests passed")


def test_text_processor():
    """Test text processing and chunking."""
    print("Testing TextProcessor...")
    
    processor = TextProcessor(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
    
    # Test preprocessing
    text = "This  is   a   test.   Multiple   spaces."
    processed = processor.preprocess_text(text)
    assert "  " not in processed  # No double spaces
    
    # Test chunking with longer text
    long_text = "This is a test sentence. " * 50  # Make text long enough
    chunks = processor.chunk_text(long_text, method="fixed")
    assert len(chunks) > 0
    assert all(len(chunk.text) <= 60 for chunk in chunks)  # chunk_size + margin
    
    print("✓ TextProcessor tests passed")


def test_embeddings():
    """Test embedding generation."""
    print("Testing EmbeddingGenerator...")
    
    generator = EmbeddingGenerator(embedding_dim=128)
    
    # Test single embedding
    embedding = generator.generate_embedding("test text")
    assert len(embedding) == 128
    assert all(isinstance(x, float) for x in embedding)
    
    # Test batch embeddings
    texts = ["text1", "text2", "text3"]
    embeddings = generator.generate_embeddings(texts)
    assert len(embeddings) == 3
    
    # Test similarity
    emb1 = generator.generate_embedding("hello world")
    emb2 = generator.generate_embedding("hello world")
    similarity = generator.cosine_similarity(emb1, emb2)
    assert 0.99 <= similarity <= 1.0  # Should be almost 1.0 (identical)
    
    print("✓ EmbeddingGenerator tests passed")


def test_vector_store():
    """Test vector store operations."""
    print("Testing VectorStore...")
    
    generator = EmbeddingGenerator(embedding_dim=64)
    store = VectorStore(generator)
    
    # Test adding texts
    id1 = store.add_text("First document", metadata={'type': 'doc'})
    id2 = store.add_text("Second document", metadata={'type': 'doc'})
    
    assert store.size() == 2
    
    # Test retrieval
    doc = store.get_by_id(id1)
    assert doc is not None
    assert "First" in doc.text
    
    # Test search
    results = store.similarity_search("First", k=1)
    assert len(results) == 1
    
    # Test deletion
    assert store.delete(id1) == True
    assert store.size() == 1
    
    print("✓ VectorStore tests passed")


def test_retriever():
    """Test retrieval functionality."""
    print("Testing Retriever...")
    
    # Setup
    generator = EmbeddingGenerator()
    store = VectorStore(generator)
    store.add_text("Python is a programming language")
    store.add_text("Machine learning uses algorithms")
    store.add_text("Data science involves statistics")
    
    retriever = Retriever(store, top_k=2)
    
    # Test basic retrieval
    results = retriever.retrieve("programming")
    assert len(results) <= 2
    assert all(result.score >= 0 for result in results)
    
    # Test context retrieval
    context = retriever.retrieve_contexts("programming", k=2)
    assert isinstance(context, str)
    
    print("✓ Retriever tests passed")


def test_reranker():
    """Test reranking functionality."""
    print("Testing Reranker...")
    
    # Setup
    generator = EmbeddingGenerator()
    store = VectorStore(generator)
    store.add_text("First document")
    store.add_text("Second document")
    store.add_text("Third document")
    
    retriever = Retriever(store, top_k=3)
    results = retriever.retrieve("document")
    
    reranker = Reranker()
    
    # Test reranking
    reranked = reranker.rerank("document", results, method="diversity")
    assert len(reranked) == len(results)
    
    print("✓ Reranker tests passed")


def test_generator():
    """Test generation functionality."""
    print("Testing Generator...")
    
    generator = Generator()
    
    # Test basic generation
    response = generator.generate("What is AI?")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test generation with context
    response = generator.generate(
        prompt="What is Python?",
        context="Python is a programming language."
    )
    assert isinstance(response, str)
    
    print("✓ Generator tests passed")


def test_pipeline():
    """Test complete RAG pipeline."""
    print("Testing AdvancedRAGPipeline...")
    
    # Initialize pipeline
    config = Config({'chunk_size': 100, 'top_k': 2, 'min_chunk_size': 10})
    pipeline = AdvancedRAGPipeline(config=config)
    
    # Ingest documents with longer text
    loader = DocumentLoader()
    docs = [
        loader.load_text("Python is great for data science and machine learning. " * 3),
        loader.load_text("Machine learning is a subset of artificial intelligence. " * 3),
    ]
    
    num_chunks = pipeline.ingest_documents(docs)
    assert num_chunks > 0
    
    # Query
    response = pipeline.query("What is Python?")
    assert isinstance(response, str)
    
    # Get stats
    stats = pipeline.get_stats()
    assert stats['total_chunks'] > 0
    
    # Clear
    pipeline.clear()
    assert pipeline.get_stats()['total_chunks'] == 0
    
    print("✓ AdvancedRAGPipeline tests passed")


def test_config():
    """Test configuration management."""
    print("Testing Config...")
    
    # Test default config
    config = Config()
    assert config.get('chunk_size') == 1000
    
    # Test custom config
    config = Config({'chunk_size': 500})
    assert config.get('chunk_size') == 500
    
    # Test set/get
    config.set('test_key', 'test_value')
    assert config.get('test_key') == 'test_value'
    
    print("✓ Config tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Advanced RAG System Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_document_loader,
        test_text_processor,
        test_embeddings,
        test_vector_store,
        test_retriever,
        test_reranker,
        test_generator,
        test_pipeline,
        test_config,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
