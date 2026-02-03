"""
Document Processor Module
Handles loading and chunking of various document types
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import markdown


class DocumentProcessor:
    """Processes various document formats and splits them into chunks"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, file_path: str) -> str:
        """Load text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
        return text
    
    def load_docx(self, file_path: str) -> str:
        """Load text from DOCX file"""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
        return text
    
    def load_txt(self, file_path: str) -> str:
        """Load text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error loading TXT {file_path}: {e}")
            return ""
    
    def load_markdown(self, file_path: str) -> str:
        """Load text from Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                # Convert markdown to plain text
                html = markdown.markdown(md_content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
        except Exception as e:
            print(f"Error loading Markdown {file_path}: {e}")
            return ""
    
    def load_document(self, file_path: str) -> str:
        """
        Load document based on file extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.load_pdf(str(file_path))
        elif extension == '.docx':
            return self.load_docx(str(file_path))
        elif extension == '.txt':
            return self.load_txt(str(file_path))
        elif extension in ['.md', '.markdown']:
            return self.load_markdown(str(file_path))
        else:
            print(f"Unsupported file type: {extension}")
            return ""
    
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to split
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # Extract chunk
            chunk_text = text[start:end]
            
            # Create chunk dictionary
            chunk = {
                'text': chunk_text.strip(),
                'start': start,
                'end': min(end, text_length),
                'chunk_index': len(chunks)
            }
            
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and process a document into chunks
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of processed chunks with metadata
        """
        text = self.load_document(file_path)
        chunks = self.split_text(text)
        
        # Add file metadata to each chunk
        file_name = Path(file_path).name
        for chunk in chunks:
            chunk['source_file'] = file_name
            chunk['file_path'] = str(file_path)
        
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        directory = Path(directory_path)
        
        # Supported extensions
        extensions = ['.pdf', '.docx', '.txt', '.md', '.markdown']
        
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in extensions:
                print(f"Processing: {file_path}")
                chunks = self.process_document(str(file_path))
                all_chunks.extend(chunks)
        
        return all_chunks
