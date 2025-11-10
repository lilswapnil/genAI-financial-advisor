"""
RAG (Retrieval-Augmented Generation) Pipeline

This module handles:
- Document loading and preprocessing
- Text chunking with overlap
- Embedding generation using BERT/Transformers
- Vector database management (Chroma)
- Semantic search and retrieval
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading and preprocessing."""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        separator: str = "\n"
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            separator: Text separator for chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for file_path in file_paths:
            try:
                # Extract metadata from filename
                path_obj = Path(file_path)
                filename = path_obj.name
                
                # Parse filename: CIK_TYPE_ACCESSION.txt
                parts = filename.replace('.txt', '').split('_')
                metadata = {
                    'source': file_path,
                    'filename': filename,
                    'cik': parts[0] if len(parts) > 0 else 'unknown',
                    'filing_type': parts[1] if len(parts) > 1 else 'unknown',
                }
                
                # Load text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Clean content
                content = self._clean_text(content)
                
                if content:
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
                    logger.info(f"Loaded {filename}: {len(content)} characters")
                else:
                    logger.warning(f"Empty content from {filename}")
                    
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        text = '\n'.join(lines)
        
        # Remove special characters but keep important ones
        text = text.replace('\x00', '')
        
        return text
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        chunks = []
        
        for doc in documents:
            # Split document into chunks
            chunk_texts = self.text_splitter.split_text(doc.page_content)
            
            # Create chunked documents with preserved metadata
            for i, chunk_text in enumerate(chunk_texts):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        **doc.metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunk_texts)
                    }
                )
                chunks.append(chunk_doc)
            
            logger.info(f"Chunked {doc.metadata['filename']}: {len(chunk_texts)} chunks")
        
        return chunks


class EmbeddingManager:
    """Manages embedding generation using HuggingFace models."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of the HuggingFace model for embeddings
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Loaded embeddings model: {model_name}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embeddings.embed_query(query)


class VectorStore:
    """Manages vector database operations using Chroma."""
    
    def __init__(self, persist_dir: str = "data/vector_db"):
        """
        Initialize vector store.
        
        Args:
            persist_dir: Directory to persist Chroma database
        """
        self.persist_dir = persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        logger.info(f"Initialized vector store at {persist_dir}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to vector store.
        
        Args:
            documents: List of documents to add
        """
        # Convert documents to format expected by Chroma
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self.db.add_documents(documents)
        self.db.persist()
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.3
    ) -> List[Tuple[Document, float]]:
        """
        Search vector store for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of (Document, similarity_score) tuples
        """
        results = self.db.similarity_search_with_scores(query, k=k)
        
        # Filter by threshold
        filtered = [(doc, score) for doc, score in results if score >= score_threshold]
        
        logger.info(f"Found {len(filtered)} relevant documents for query")
        return filtered
    
    def get_stats(self) -> Dict:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with store statistics
        """
        return {
            'persist_dir': self.persist_dir,
            'collection_count': self.db._collection.count() if hasattr(self.db, '_collection') else 'unknown'
        }


class RAGPipeline:
    """Complete RAG pipeline combining all components."""
    
    def __init__(
        self,
        raw_data_dir: str = "data/raw_reports",
        vector_db_dir: str = "data/vector_db",
        chunk_size: int = 1024,
        chunk_overlap: int = 128
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            raw_data_dir: Directory with raw documents
            vector_db_dir: Directory for vector database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.raw_data_dir = raw_data_dir
        self.vector_db_dir = vector_db_dir
        
        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embeddings = EmbeddingManager()
        self.vector_store = VectorStore(vector_db_dir)
    
    def ingest_documents(self, file_paths: List[str] = None) -> int:
        """
        Ingest documents into RAG pipeline.
        
        Args:
            file_paths: List of file paths. If None, scans raw_data_dir
            
        Returns:
            Number of chunks added
        """
        # Scan directory if no files provided
        if file_paths is None:
            data_dir = Path(self.raw_data_dir)
            file_paths = list(data_dir.glob('*.txt'))
            file_paths = [str(f) for f in file_paths]
        
        if not file_paths:
            logger.warning("No files to ingest")
            return 0
        
        # Load documents
        documents = self.processor.load_documents(file_paths)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Chunk documents
        chunks = self.processor.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        return len(chunks)
    
    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of results with content and metadata
        """
        results = self.vector_store.search(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score)
            })
        
        return formatted_results
    
    def get_context(
        self,
        query: str,
        k: int = 3,
        max_chars: int = 5000
    ) -> str:
        """
        Get context string for LLM prompt.
        
        Args:
            query: Search query
            k: Number of results
            max_chars: Maximum characters in context
            
        Returns:
            Context string
        """
        results = self.search(query, k=k)
        
        context = "RETRIEVED FINANCIAL DOCUMENTS:\n\n"
        total_chars = 0
        
        for i, result in enumerate(results, 1):
            result_text = f"[Document {i}]\n"
            result_text += f"Filing: {result['metadata'].get('filing_type', 'N/A')}\n"
            result_text += f"CIK: {result['metadata'].get('cik', 'N/A')}\n"
            result_text += f"Content:\n{result['content'][:500]}...\n\n"
            
            if total_chars + len(result_text) <= max_chars:
                context += result_text
                total_chars += len(result_text)
            else:
                break
        
        return context
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'vector_store': self.vector_store.get_stats(),
            'raw_data_dir': self.raw_data_dir
        }


if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline()
    
    # Ingest documents (if any exist)
    num_chunks = pipeline.ingest_documents()
    print(f"Ingested {num_chunks} chunks")
    
    # Example search
    if num_chunks > 0:
        query = "What was the total revenue?"
        results = pipeline.search(query, k=3)
        
        print(f"\n\nSearch results for: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"Content: {result['content'][:200]}...")
            print(f"Score: {result['similarity_score']:.3f}")
