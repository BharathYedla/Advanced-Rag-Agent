# Advanced RAG Agent

A comprehensive implementation of an Advanced Retrieval-Augmented Generation (RAG) system with modular components for data ingestion, indexing, retrieval, and generation.

## Overview

This project demonstrates how to build an advanced RAG pipeline that improves upon naive RAG implementations by providing:

- **Modular Architecture**: Each component (ingestion, indexing, retrieval, generation) is independent and customizable
- **Flexible Data Ingestion**: Support for multiple document formats and preprocessing strategies
- **Advanced Retrieval**: Hybrid retrieval methods with reranking capabilities
- **Configurable Pipeline**: Easy-to-customize configuration for different use cases

## Architecture

The Advanced RAG system consists of four main pipeline stages:

### 1. Data Ingestion
- **Document Loading**: Support for text, markdown, JSON, CSV, and HTML files
- **Text Processing**: Multiple chunking strategies (fixed-size, sentence-based, paragraph-based)
- **Metadata Extraction**: Automatic extraction and preservation of document metadata

### 2. Indexing
- **Embedding Generation**: Converts text chunks into vector representations
- **Vector Store**: Efficient storage and management of document embeddings
- **Index Management**: Add, update, and delete documents from the index

### 3. Retrieval
- **Similarity Search**: Find relevant documents using cosine similarity
- **Hybrid Retrieval**: Combine multiple retrieval methods
- **Reranking**: Improve retrieval quality with diversity-based and custom reranking

### 4. Generation
- **Context-Aware Generation**: Generate responses using retrieved context
- **Prompt Templates**: Pre-built templates for different use cases
- **Flexible Configuration**: Customize generation parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/BharathYedla/Advanced-Rag-Agent.git
cd Advanced-Rag-Agent

# No external dependencies required for basic functionality
# The system uses built-in Python libraries
```

## Quick Start

```python
from advanced_rag.pipeline import AdvancedRAGPipeline
from advanced_rag.ingestion.document_loader import DocumentLoader

# Initialize the RAG pipeline
pipeline = AdvancedRAGPipeline()

# Load and ingest documents
loader = DocumentLoader()
doc = loader.load_text(
    "Your document text here...",
    metadata={'source': 'example'}
)
pipeline.ingest_documents([doc])

# Query the system
question = "Your question here?"
answer = pipeline.query(question)
print(answer)
```

## Detailed Usage

### Custom Configuration

```python
from advanced_rag.pipeline import AdvancedRAGPipeline
from advanced_rag.utils.config import Config

# Create custom configuration
config = Config({
    'chunk_size': 500,
    'chunk_overlap': 100,
    'top_k': 5,
    'chunk_method': 'sentence',
    'similarity_threshold': 0.3
})

# Initialize with custom config
pipeline = AdvancedRAGPipeline(config=config)
```

### Loading Documents

```python
from advanced_rag.ingestion.document_loader import DocumentLoader

loader = DocumentLoader()

# Load from file
doc = loader.load_file('path/to/document.txt')

# Load from directory
docs = loader.load_directory('path/to/documents/', recursive=True)

# Load from text
doc = loader.load_text("Your text content", metadata={'source': 'manual'})
```

### Text Processing and Chunking

```python
from advanced_rag.ingestion.text_processor import TextProcessor

processor = TextProcessor(
    chunk_size=1000,
    chunk_overlap=200
)

# Different chunking methods
fixed_chunks = processor.chunk_text(text, method="fixed")
sentence_chunks = processor.chunk_text(text, method="sentence")
paragraph_chunks = processor.chunk_text(text, method="paragraph")
```

### Advanced Retrieval

```python
# Retrieve with reranking
answer = pipeline.query(
    question="Your question?",
    use_reranking=True,
    rerank_method="diversity"
)

# Get contexts only (no generation)
contexts = pipeline.get_relevant_contexts(
    question="Your question?",
    k=5
)
```

## Project Structure

```
Advanced-Rag-Agent/
├── advanced_rag/
│   ├── __init__.py
│   ├── pipeline.py              # Main RAG pipeline orchestration
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py   # Document loading utilities
│   │   └── text_processor.py    # Text preprocessing and chunking
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Embedding generation
│   │   └── vector_store.py      # Vector storage and search
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py         # Retrieval strategies
│   │   └── reranker.py          # Reranking methods
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generator.py         # Response generation
│   │   └── prompt_templates.py  # Prompt templates
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       └── logger.py             # Logging utilities
├── examples/
│   └── basic_usage.py           # Example usage scripts
├── tests/
│   └── (test files)
├── README.md
└── LICENSE
```

## Key Features

### Modular Design
Each component can be used independently or as part of the complete pipeline:
- Use `DocumentLoader` alone for document loading
- Use `TextProcessor` for text chunking
- Use `VectorStore` for embedding management
- Use `Retriever` for similarity search
- Use `Generator` for response generation

### Multiple Chunking Strategies
- **Fixed-size chunking**: Split text into chunks of fixed size with overlap
- **Sentence-based chunking**: Group sentences to approximate desired chunk size
- **Paragraph-based chunking**: Use natural paragraph boundaries

### Flexible Retrieval
- **Basic similarity search**: Cosine similarity-based retrieval
- **Filtered retrieval**: Filter results by metadata
- **Hybrid retrieval**: Combine multiple retrieval methods with metadata boosting
- **Reranking**: Improve result quality with diversity and custom scoring

### Production Considerations
- **Logging**: Built-in logging for debugging and monitoring
- **Configuration**: Environment variable support and flexible configuration
- **Extensibility**: Easy to extend with custom components
- **Error Handling**: Robust error handling throughout the pipeline

## Examples

Run the example script to see the system in action:

```bash
python examples/basic_usage.py
```

This will demonstrate:
1. Basic RAG pipeline usage
2. Custom configuration
3. Retrieval without generation
4. Reranking functionality

## Advanced Topics

### Custom Embedding Models
To integrate with actual embedding models (e.g., OpenAI, Sentence-Transformers):

```python
from advanced_rag.indexing.embeddings import EmbeddingGenerator

class CustomEmbeddingGenerator(EmbeddingGenerator):
    def generate_embedding(self, text: str):
        # Implement your custom embedding logic
        pass
```

### Custom Generation Models
To integrate with actual LLM APIs:

```python
from advanced_rag.generation.generator import Generator

class CustomGenerator(Generator):
    def _mock_generate(self, prompt: str):
        # Replace with actual LLM API call
        # e.g., OpenAI API, Anthropic API, local models
        pass
```

## Extending the System

The modular architecture makes it easy to extend:

1. **Add new document loaders**: Extend `DocumentLoader` to support more file formats
2. **Implement new chunking strategies**: Add methods to `TextProcessor`
3. **Create custom reranking methods**: Extend `Reranker` with new algorithms
4. **Add new prompt templates**: Extend `PromptTemplates` with custom templates

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This implementation demonstrates advanced RAG concepts including:
- Modular pipeline architecture
- Multiple chunking strategies
- Hybrid retrieval methods
- Reranking for improved accuracy
- Production-ready design patterns
