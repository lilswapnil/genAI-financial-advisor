"""
Python API Client for Financial Analyst Advisor.

Usage example:
    client = FinancialAnalystClient("http://localhost:8000")
    response = client.analyze("What was Apple's revenue growth?")
    print(response)
"""

import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SearchResult:
    """Search result from document retrieval."""
    content: str
    cik: str
    filing_type: str
    similarity_score: float


@dataclass
class AnalysisResult:
    """Analysis result from financial question."""
    question: str
    answer: str
    confidence: float
    sources: List[Dict]
    timestamp: datetime


class FinancialAnalystClient:
    """Python client for Financial Analyst Advisor API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health.
        
        Returns:
            Health status
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def search(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.3
    ) -> List[SearchResult]:
        """
        Search for relevant financial documents.
        
        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        payload = {
            "query": query,
            "k": k,
            "score_threshold": score_threshold
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/search",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        # Convert to SearchResult objects
        results = []
        for item in data.get('results', []):
            results.append(SearchResult(
                content=item['content'],
                cik=item['metadata'].get('cik', 'N/A'),
                filing_type=item['metadata'].get('filing_type', 'N/A'),
                similarity_score=item['similarity_score']
            ))
        
        return results
    
    def analyze(
        self,
        question: str,
        include_context: bool = True,
        company_filter: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze a financial question.
        
        Args:
            question: Financial question
            include_context: Include retrieved context
            company_filter: Filter by company CIK
            
        Returns:
            Analysis result with answer
        """
        payload = {
            "question": question,
            "include_context": include_context,
        }
        if company_filter:
            payload["company_filter"] = company_filter
        
        response = self.session.post(
            f"{self.base_url}/api/v1/analyze",
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        return AnalysisResult(
            question=data['question'],
            answer=data['answer'],
            confidence=data['confidence'],
            sources=data.get('sources', []),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
    
    def ingest_documents(
        self,
        file_paths: List[str],
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest documents into vector database.
        
        Args:
            file_paths: List of file paths to ingest
            force_reprocess: Force reprocessing
            
        Returns:
            Ingestion status
        """
        payload = {
            "file_paths": file_paths,
            "force_reprocess": force_reprocess
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/ingest",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using fine-tuned model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        params = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/generate",
            params=params
        )
        response.raise_for_status()
        data = response.json()
        
        return data['generated_text']
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            System statistics
        """
        response = self.session.get(f"{self.base_url}/api/v1/stats")
        response.raise_for_status()
        return response.json()


# ==================== Example Usage ====================

def main():
    """Example usage of the client."""
    
    # Initialize client
    client = FinancialAnalystClient("http://localhost:8000")
    
    print("Financial Analyst Advisor - Python Client\n")
    
    # 1. Check health
    print("1. Checking API health...")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   RAG Pipeline: {health['rag_pipeline_loaded']}")
        print(f"   Model: {health['model_loaded']}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
        print("   Make sure the API server is running:")
        print("   python -m src.api.app\n")
        return
    
    # 2. Search documents
    print("2. Searching documents...")
    try:
        results = client.search(
            "What was Apple's revenue in 2023?",
            k=3
        )
        print(f"   Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"   [{i}] {result.filing_type} (Score: {result.similarity_score:.2f})")
            print(f"       {result.content[:80]}...\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 3. Analyze question
    print("3. Analyzing financial question...")
    try:
        question = "What are the main risks facing technology companies?"
        result = client.analyze(
            question,
            include_context=False
        )
        print(f"   Question: {result.question}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Answer: {result.answer[:200]}...\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 4. Generate text
    print("4. Generating text with fine-tuned model...")
    try:
        prompt = "Financial analysis is important because"
        generated = client.generate(
            prompt,
            max_tokens=100,
            temperature=0.7
        )
        print(f"   Generated: {generated}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # 5. Get statistics
    print("5. Getting system statistics...")
    try:
        stats = client.get_stats()
        print(f"   Timestamp: {stats['timestamp']}")
        if 'rag_pipeline' in stats:
            print(f"   RAG Pipeline Stats: {stats['rag_pipeline']}\n")
    except Exception as e:
        print(f"   Error: {e}\n")


if __name__ == "__main__":
    main()
