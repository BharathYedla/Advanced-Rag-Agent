"""
Example usage of the Advanced RAG System.

Demonstrates how to use the various components of the RAG pipeline.
"""

import sys
import os

# Add parent directory to path to import advanced_rag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advanced_rag.pipeline import AdvancedRAGPipeline
from advanced_rag.utils.config import Config
from advanced_rag.ingestion.document_loader import DocumentLoader


def example_basic_usage():
    """Demonstrate basic RAG pipeline usage."""
    print("=" * 60)
    print("Example 1: Basic RAG Pipeline Usage")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = AdvancedRAGPipeline()
    
    # Create some sample documents
    loader = DocumentLoader()
    doc1 = loader.load_text(
        "Python is a high-level programming language. It is known for its simplicity and readability. "
        "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
        metadata={'source': 'python_intro', 'topic': 'programming'}
    )
    
    doc2 = loader.load_text(
        "Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data. "
        "Common machine learning techniques include supervised learning, unsupervised learning, and reinforcement learning.",
        metadata={'source': 'ml_intro', 'topic': 'ai'}
    )
    
    doc3 = loader.load_text(
        "Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with text generation. "
        "It retrieves relevant documents from a knowledge base and uses them as context for generating responses.",
        metadata={'source': 'rag_intro', 'topic': 'ai'}
    )
    
    # Ingest documents
    print("\nIngesting documents...")
    num_chunks = pipeline.ingest_documents([doc1, doc2, doc3])
    print(f"Created {num_chunks} chunks from 3 documents")
    
    # Query the system
    print("\n" + "-" * 60)
    print("Querying the RAG system:")
    print("-" * 60)
    
    questions = [
        "What is Python?",
        "Tell me about machine learning",
        "What is RAG?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = pipeline.query(question)
        print(f"A: {answer}")
    
    # Get statistics
    print("\n" + "-" * 60)
    print("Pipeline Statistics:")
    print("-" * 60)
    stats = pipeline.get_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")


def example_custom_configuration():
    """Demonstrate custom configuration."""
    print("\n\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = Config({
        'chunk_size': 500,
        'chunk_overlap': 100,
        'top_k': 3,
        'chunk_method': 'sentence'
    })
    
    # Initialize pipeline with custom config
    pipeline = AdvancedRAGPipeline(config=config)
    
    # Load a document
    loader = DocumentLoader()
    doc = loader.load_text(
        "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans "
        "through natural language. NLP techniques include tokenization, part-of-speech tagging, named entity recognition, "
        "sentiment analysis, and machine translation. Modern NLP often uses deep learning models like transformers. "
        "BERT and GPT are examples of transformer-based models that have achieved state-of-the-art results in many NLP tasks.",
        metadata={'source': 'nlp_doc'}
    )
    
    print("\nIngesting document with sentence-based chunking...")
    num_chunks = pipeline.ingest_documents([doc])
    print(f"Created {num_chunks} chunks")
    
    # Query
    question = "What are some NLP techniques?"
    print(f"\nQ: {question}")
    answer = pipeline.query(question)
    print(f"A: {answer}")


def example_retrieval_only():
    """Demonstrate retrieval without generation."""
    print("\n\n" + "=" * 60)
    print("Example 3: Retrieval Only (No Generation)")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = AdvancedRAGPipeline()
    
    # Add documents
    loader = DocumentLoader()
    docs = [
        loader.load_text("The solar system consists of the Sun and the objects that orbit it, including eight planets."),
        loader.load_text("Mars is the fourth planet from the Sun and is often called the Red Planet due to its reddish appearance."),
        loader.load_text("Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined.")
    ]
    
    pipeline.ingest_documents(docs)
    
    # Retrieve relevant contexts
    question = "Tell me about Mars"
    print(f"\nQuery: {question}")
    print("\nRetrieved contexts:")
    print("-" * 60)
    
    results = pipeline.get_relevant_contexts(question, k=2)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   Text: {result.text}")


def example_reranking():
    """Demonstrate reranking functionality."""
    print("\n\n" + "=" * 60)
    print("Example 4: Retrieval with Reranking")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = AdvancedRAGPipeline()
    
    # Add documents
    loader = DocumentLoader()
    docs = [
        loader.load_text("Coffee is a brewed drink prepared from roasted coffee beans."),
        loader.load_text("Tea is an aromatic beverage commonly prepared by pouring hot water over cured or fresh leaves."),
        loader.load_text("Coffee contains caffeine, a stimulant that can help increase alertness."),
        loader.load_text("Green tea is made from Camellia sinensis leaves that have not undergone the same withering and oxidation process.")
    ]
    
    pipeline.ingest_documents(docs)
    
    # Query with reranking
    question = "What is coffee?"
    print(f"\nQuery: {question}")
    
    print("\n1. Without reranking:")
    answer1 = pipeline.query(question, use_reranking=False)
    print(f"Answer: {answer1[:100]}...")
    
    print("\n2. With diversity reranking:")
    answer2 = pipeline.query(question, use_reranking=True, rerank_method="diversity")
    print(f"Answer: {answer2[:100]}...")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ADVANCED RAG SYSTEM - EXAMPLES")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_custom_configuration()
        example_retrieval_only()
        example_reranking()
        
        print("\n\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
