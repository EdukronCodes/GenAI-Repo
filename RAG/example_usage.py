"""
Example Usage Script
Demonstrates how to use the RAG pipeline
"""

from rag_pipeline import RAGPipeline
import os


def main():
    """Main example function"""
    
    print("=" * 60)
    print("RAG Pipeline - Example Usage")
    print("=" * 60)
    
    # Initialize RAG pipeline
    print("\n1. Initializing RAG Pipeline...")
    rag = RAGPipeline(
        embedding_model="all-MiniLM-L6-v2",  # Fast embedding model
        llm_model="gpt2",  # Lightweight LLM for demo
        chunk_size=500,  # Smaller chunks for demo
        chunk_overlap=100
    )
    
    # Check if documents directory exists
    docs_dir = "./documents"
    if not os.path.exists(docs_dir):
        print(f"\n2. Creating sample documents directory: {docs_dir}")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Create sample documents
        create_sample_documents(docs_dir)
    
    # Ingest documents
    print("\n2. Ingesting documents...")
    rag.ingest_documents([docs_dir])
    
    # Display vector store info
    print("\n3. Vector Store Information:")
    info = rag.get_vector_store_info()
    print(f"   - Collection: {info['name']}")
    print(f"   - Document count: {info['count']}")
    
    # Example queries
    print("\n4. Running example queries...")
    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "What are the main components of a RAG system?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        result = rag.query(query, top_k=3)
        
        print(f"\nAnswer:")
        print(f"{result['answer']}")
        
        print(f"\nRetrieved {result['metadata']['retrieved_count']} relevant chunks:")
        for j, chunk in enumerate(result['retrieved_context'][:2], 1):  # Show first 2
            print(f"\n  Chunk {j} (from {chunk['metadata'].get('source_file', 'unknown')}):")
            print(f"  {chunk['text'][:200]}...")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


def create_sample_documents(docs_dir: str):
    """Create sample documents for demonstration"""
    
    # Sample document 1: Machine Learning
    ml_content = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. 

## Types of Machine Learning

1. **Supervised Learning**: Uses labeled data to train models
   - Classification: Predicting categories
   - Regression: Predicting continuous values

2. **Unsupervised Learning**: Finds patterns in unlabeled data
   - Clustering: Grouping similar data points
   - Dimensionality reduction: Reducing feature space

3. **Reinforcement Learning**: Learns through interaction with environment
   - Agent receives rewards or penalties
   - Learns optimal policy through trial and error

## Key Concepts

- **Features**: Input variables used for prediction
- **Model**: Mathematical representation of patterns
- **Training**: Process of learning from data
- **Inference**: Making predictions on new data
"""
    
    # Sample document 2: Neural Networks
    nn_content = """# Neural Networks Explained

Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers.

## Architecture

A typical neural network has:
- **Input Layer**: Receives input features
- **Hidden Layers**: Process information (can be multiple)
- **Output Layer**: Produces final predictions

## How They Work

1. Forward Propagation: Data flows from input to output
2. Activation Functions: Introduce non-linearity (ReLU, Sigmoid, Tanh)
3. Backpropagation: Updates weights based on errors
4. Gradient Descent: Optimizes the model

## Applications

- Image recognition
- Natural language processing
- Speech recognition
- Autonomous vehicles
- Recommendation systems
"""
    
    # Sample document 3: RAG Systems
    rag_content = """# Retrieval-Augmented Generation (RAG)

RAG combines information retrieval with language generation to produce more accurate and context-aware responses.

## Components

1. **Document Processor**: Loads and chunks documents
   - Handles various formats (PDF, DOCX, TXT, MD)
   - Splits documents into manageable chunks

2. **Embedding Generator**: Creates vector representations
   - Uses sentence transformers
   - Converts text to numerical vectors

3. **Vector Store**: Stores and retrieves embeddings
   - Uses similarity search
   - ChromaDB or FAISS for storage

4. **Retrieval Module**: Finds relevant context
   - Semantic search
   - Returns top-k similar chunks

5. **LLM Generator**: Generates final answer
   - Uses retrieved context
   - Produces coherent responses

## Benefits

- Reduces hallucinations
- Provides citations
- Handles domain-specific knowledge
- More accurate than LLM alone
"""
    
    # Write sample documents
    with open(os.path.join(docs_dir, "machine_learning.md"), "w", encoding="utf-8") as f:
        f.write(ml_content)
    
    with open(os.path.join(docs_dir, "neural_networks.md"), "w", encoding="utf-8") as f:
        f.write(nn_content)
    
    with open(os.path.join(docs_dir, "rag_systems.md"), "w", encoding="utf-8") as f:
        f.write(rag_content)
    
    print(f"Created 3 sample markdown documents in {docs_dir}")


if __name__ == "__main__":
    main()
