"""
RAG Pipeline Module
End-to-end Retrieval-Augmented Generation pipeline
"""

from typing import List, Dict, Any, Optional
from document_processor import DocumentProcessor
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from llm_generator import LLMGenerator
import os


class RAGPipeline:
    """Complete RAG pipeline orchestrating all components"""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "microsoft/DialoGPT-medium",
        use_ollama: bool = False,
        ollama_model: str = "llama2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_store_path: str = "./chroma_db",
        collection_name: str = "rag_documents"
    ):
        """
        Initialize RAG pipeline
        
        Args:
            embedding_model: Sentence transformer model name
            llm_model: HuggingFace model name for LLM
            use_ollama: Use Ollama for LLM instead of transformers
            ollama_model: Ollama model name
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
            vector_store_path: Path for ChromaDB persistence
            collection_name: ChromaDB collection name
        """
        print("Initializing RAG Pipeline...")
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=vector_store_path
        )
        
        self.llm_generator = LLMGenerator(
            model_name=llm_model,
            use_ollama=use_ollama,
            ollama_model=ollama_model
        )
        
        print("RAG Pipeline initialized successfully!")
    
    def ingest_documents(self, file_paths: List[str]) -> None:
        """
        Ingest documents into the vector store
        
        Args:
            file_paths: List of file paths or directory paths to process
        """
        print("\n=== Document Ingestion ===")
        all_chunks = []
        
        for path in file_paths:
            if os.path.isfile(path):
                print(f"Processing file: {path}")
                chunks = self.document_processor.process_document(path)
                all_chunks.extend(chunks)
            elif os.path.isdir(path):
                print(f"Processing directory: {path}")
                chunks = self.document_processor.process_directory(path)
                all_chunks.extend(chunks)
            else:
                print(f"Path not found: {path}")
        
        if not all_chunks:
            print("No chunks to process")
            return
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        chunks_with_embeddings = self.embedding_generator.generate_embeddings_for_chunks(all_chunks)
        
        # Add to vector store
        print("\nAdding to vector store...")
        self.vector_store.add_chunks(chunks_with_embeddings)
        
        print("\nDocument ingestion complete!")
    
    def query(self, question: str, top_k: int = 5, max_length: int = 512) -> Dict[str, Any]:
        """
        Query the RAG pipeline
        
        Args:
            question: User question
            top_k: Number of relevant chunks to retrieve
            max_length: Maximum length of generated response
            
        Returns:
            Dictionary with answer, retrieved context, and metadata
        """
        print(f"\n=== Processing Query: {question} ===")
        
        # Generate query embedding
        print("Generating query embedding...")
        query_embedding = self.embedding_generator.generate_embedding(question)
        
        # Retrieve relevant chunks
        print(f"Retrieving top {top_k} relevant chunks...")
        retrieved_chunks = self.vector_store.search(query_embedding, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                'answer': "No relevant documents found in the knowledge base.",
                'retrieved_context': [],
                'metadata': {'retrieved_count': 0}
            }
        
        print(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Extract context texts
        context_texts = [chunk['text'] for chunk in retrieved_chunks]
        
        # Generate answer using LLM
        print("Generating answer with LLM...")
        answer = self.llm_generator.generate_with_context(
            query=question,
            context=context_texts,
            max_length=max_length
        )
        
        # Prepare result
        result = {
            'answer': answer,
            'retrieved_context': retrieved_chunks,
            'metadata': {
                'retrieved_count': len(retrieved_chunks),
                'query': question
            }
        }
        
        print("Query processing complete!")
        return result
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about the vector store"""
        return self.vector_store.get_collection_info()
    
    def clear_vector_store(self) -> None:
        """Clear all documents from the vector store"""
        self.vector_store.delete_collection()
        # Recreate collection
        self.vector_store = VectorStore(
            collection_name=self.vector_store.collection_name,
            persist_directory=self.vector_store.persist_directory
        )
