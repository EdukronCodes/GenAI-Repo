# RAG Pipeline - End-to-End Implementation

A complete **Retrieval-Augmented Generation (RAG)** pipeline built entirely with open-source models and tools. This implementation provides a production-ready framework for building question-answering systems over your own documents.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Model Options](#model-options)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This RAG pipeline enables you to:
- **Ingest** documents in multiple formats (PDF, DOCX, TXT, Markdown)
- **Process** and chunk documents intelligently
- **Generate** embeddings using open-source sentence transformers
- **Store** vectors in a persistent vector database (ChromaDB)
- **Retrieve** relevant context using semantic search
- **Generate** answers using open-source language models

All components use open-source models - no API keys required!

## ✨ Features

- ✅ **Multiple Document Formats**: PDF, DOCX, TXT, Markdown
- ✅ **Open-Source Embeddings**: Sentence Transformers (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- ✅ **Vector Storage**: ChromaDB with persistent storage
- ✅ **Open-Source LLMs**: Transformers library (GPT-2, DialoGPT, Mistral) or Ollama
- ✅ **Semantic Search**: Cosine similarity-based retrieval
- ✅ **Configurable**: Easy to customize chunk sizes, models, and parameters
- ✅ **Production Ready**: Modular design, error handling, logging

## 🏗️ Architecture

```
┌─────────────────┐
│   Documents     │
│  (PDF/DOCX/TXT) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document        │
│ Processor       │
│ (Chunking)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embedding       │
│ Generator       │
│ (Sentence       │
│  Transformers)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Store    │
│ (ChromaDB)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Query         │─────▶│   Retrieve   │
│                 │      │   Top-K      │
└─────────────────┘      └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │   LLM        │
                          │   Generator  │
                          └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │   Answer     │
                          └──────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

```bash
cd RAG
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For GPU acceleration (recommended for LLM), ensure you have CUDA installed and PyTorch with CUDA support.

### Step 3: Verify Installation

```bash
python -c "import sentence_transformers; import chromadb; print('Installation successful!')"
```

## 🚀 Quick Start

### Basic Usage

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gpt2"
)

# Ingest documents
rag.ingest_documents(["./documents"])

# Query
result = rag.query("What is machine learning?")
print(result['answer'])
```

### Run Example

```bash
python example_usage.py
```

This will:
1. Create sample documents
2. Ingest them into the vector store
3. Run example queries
4. Display results

## 🔧 Components

### 1. Document Processor (`document_processor.py`)

Handles loading and chunking of documents.

**Features**:
- Supports PDF, DOCX, TXT, Markdown
- Configurable chunk size and overlap
- Preserves metadata (file name, position)

**Usage**:
```python
from document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
chunks = processor.process_document("document.pdf")
```

### 2. Embedding Generator (`embedding_generator.py`)

Generates embeddings using sentence transformers.

**Models**:
- `all-MiniLM-L6-v2` (default, 384 dims, fast)
- `all-mpnet-base-v2` (768 dims, better quality)
- `paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

**Usage**:
```python
from embedding_generator import EmbeddingGenerator

generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
embedding = generator.generate_embedding("Your text here")
```

### 3. Vector Store (`vector_store.py`)

Manages vector storage using ChromaDB.

**Features**:
- Persistent storage
- Cosine similarity search
- Metadata filtering

**Usage**:
```python
from vector_store import VectorStore

store = VectorStore(collection_name="my_docs")
store.add_chunks(chunks_with_embeddings)
results = store.search(query_embedding, top_k=5)
```

### 4. LLM Generator (`llm_generator.py`)

Generates text using open-source LLMs.

**Options**:
- Transformers library (GPT-2, DialoGPT, Mistral)
- Ollama (Llama2, Mistral, etc.)

**Usage**:
```python
from llm_generator import LLMGenerator

llm = LLMGenerator(model_name="gpt2")
answer = llm.generate("Your prompt")
```

### 5. RAG Pipeline (`rag_pipeline.py`)

Main orchestrator combining all components.

**Usage**:
```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.ingest_documents(["./documents"])
result = rag.query("Your question")
```

## 📝 Usage Examples

### Example 1: Custom Configuration

```python
rag = RAGPipeline(
    embedding_model="all-mpnet-base-v2",  # Better quality embeddings
    llm_model="microsoft/DialoGPT-medium",  # Better LLM
    chunk_size=1500,  # Larger chunks
    chunk_overlap=300,  # More overlap
    vector_store_path="./my_vector_db"
)
```

### Example 2: Using Ollama

```python
rag = RAGPipeline(
    embedding_model="all-MiniLM-L6-v2",
    use_ollama=True,
    ollama_model="llama2"  # Requires Ollama installed locally
)
```

### Example 3: Processing Specific Files

```python
rag = RAGPipeline()
rag.ingest_documents([
    "./documents/report.pdf",
    "./documents/manual.docx",
    "./documents/notes.txt"
])
```

### Example 4: Advanced Querying

```python
result = rag.query(
    question="Explain neural networks",
    top_k=10,  # Retrieve more context
    max_length=1024  # Longer response
)

print(f"Answer: {result['answer']}")
print(f"Retrieved {result['metadata']['retrieved_count']} chunks")
for chunk in result['retrieved_context']:
    print(f"Source: {chunk['metadata']['source_file']}")
```

## ⚙️ Configuration

### Chunking Parameters

- `chunk_size`: Maximum characters per chunk (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

**Recommendations**:
- Small documents: 500-800 chars
- Medium documents: 1000-1500 chars
- Large documents: 1500-2000 chars
- Overlap: 10-20% of chunk_size

### Retrieval Parameters

- `top_k`: Number of chunks to retrieve (default: 5)
  - More chunks = more context but potentially more noise
  - Fewer chunks = faster but may miss relevant info

### Generation Parameters

- `max_length`: Maximum response length (default: 512)
- `temperature`: Sampling temperature (default: 0.7)
  - Lower (0.1-0.3): More deterministic
  - Higher (0.7-1.0): More creative

## 🤖 Model Options

### Embedding Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ⭐⭐ | Fast, general purpose |
| `all-mpnet-base-v2` | 768 | ⚡⚡ | ⭐⭐⭐ | Better quality |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | ⚡⚡ | ⭐⭐ | Multilingual |

### LLM Models (Transformers)

| Model | Size | Quality | Speed | Memory |
|-------|------|----------|-------|--------|
| `gpt2` | 124M | ⭐ | ⚡⚡⚡ | Low |
| `microsoft/DialoGPT-medium` | 345M | ⭐⭐ | ⚡⚡ | Medium |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | ⭐⭐⭐⭐ | ⚡ | High |

**Note**: Larger models require more GPU memory. Use `gpt2` for CPU-only systems.

### Ollama Models

If using Ollama, popular models include:
- `llama2` (7B)
- `mistral` (7B)
- `codellama` (for code)
- `phi` (smaller, faster)

Install Ollama: https://ollama.ai

## 🚀 Performance Tips

1. **GPU Acceleration**: Use GPU for faster embeddings and LLM inference
   ```python
   # Check if CUDA is available
   import torch
   print(torch.cuda.is_available())
   ```

2. **Batch Processing**: Embeddings are automatically batched for efficiency

3. **Model Selection**: 
   - Use smaller models for faster inference
   - Use larger models for better quality

4. **Chunk Size**: 
   - Smaller chunks = more precise retrieval
   - Larger chunks = more context per chunk

5. **Vector Store**: ChromaDB persists data, so re-ingestion is fast

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution**: Use smaller models or reduce batch size
```python
rag = RAGPipeline(
    llm_model="gpt2",  # Smaller model
    embedding_model="all-MiniLM-L6-v2"  # Smaller embeddings
)
```

### Issue: Slow Embedding Generation

**Solution**: Use GPU or smaller model
```python
# Check GPU availability
import torch
print(torch.cuda.is_available())
```

### Issue: Poor Retrieval Quality

**Solutions**:
1. Use better embedding model: `all-mpnet-base-v2`
2. Adjust chunk size (smaller chunks often better)
3. Increase `top_k` for more context
4. Check document quality and preprocessing

### Issue: LLM Generates Unrelated Text

**Solutions**:
1. Use better LLM model
2. Reduce temperature (more deterministic)
3. Improve prompt formatting
4. Ensure retrieved context is relevant

### Issue: ChromaDB Errors

**Solution**: Clear and recreate vector store
```python
rag.clear_vector_store()
rag.ingest_documents(["./documents"])
```

## 📚 Additional Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Ollama Documentation](https://github.com/ollama/ollama)

## 📄 License

This project uses open-source models and libraries. Please check individual model licenses:
- Sentence Transformers: Apache 2.0
- ChromaDB: Apache 2.0
- Transformers: Apache 2.0

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📧 Support

For questions or issues, please open an issue on the repository.

---

**Built with ❤️ using open-source models**
