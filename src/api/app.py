"""
FastAPI REST API for Financial Analyst Chatbot

Provides endpoints for:
- Financial question answering
- Document search
- RAG context retrieval
- Model inference
"""

"""FastAPI REST API for Financial Analyst Chatbot."""

import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Analyst Chatbot API",
    description="RAG-powered financial analysis with fine-tuned LLaMA model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (loaded on startup)
rag_pipeline = None
fine_tuned_model = None


# ==================== Request/Response Models ====================

class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., description="Search query", min_length=1, max_length=500)
    k: int = Field(5, description="Number of results", ge=1, le=20)
    score_threshold: float = Field(0.3, description="Minimum similarity score", ge=0.0, le=1.0)


class SearchResponse(BaseModel):
    """Response model for search results."""
    query: str
    results: List[Dict]
    total_results: int
    timestamp: str


class AnalystQuestion(BaseModel):
    """Request model for financial analyst question."""
    question: str = Field(..., description="Financial question", min_length=5, max_length=500)
    company_filter: Optional[str] = Field(None, description="Filter results by company CIK")
    include_context: bool = Field(True, description="Include retrieved context in response")


class AnalystAnswer(BaseModel):
    """Response model for financial analyst answer."""
    question: str
    answer: str
    confidence: float = Field(..., description="Confidence score 0-1", ge=0.0, le=1.0)
    sources: List[Dict] = Field(default_factory=list, description="Retrieved source documents")
    retrieved_context: Optional[str] = Field(None, description="Retrieved context used")
    timestamp: str


class IngestDocumentsRequest(BaseModel):
    """Request model for document ingestion."""
    file_paths: List[str] = Field(..., description="List of file paths to ingest")
    force_reprocess: bool = Field(False, description="Force reprocessing of existing files")


class IngestDocumentsResponse(BaseModel):
    """Response model for document ingestion."""
    status: str
    chunks_added: int
    files_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    rag_pipeline_loaded: bool
    model_loaded: bool
    timestamp: str


# ==================== Helper Functions ====================

def load_rag_pipeline():
    """Load RAG pipeline on startup."""
    global rag_pipeline
    try:
        from src.rag.rag_pipeline import RAGPipeline
        
        rag_pipeline = RAGPipeline()
        logger.info("RAG pipeline loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load RAG pipeline: {e}")
        return False


def load_fine_tuned_model():
    """Load fine-tuned model on startup."""
    global fine_tuned_model
    try:
        from src.fine_tuning.lora_finetuner import LoRAFineTuner
        
        model_path = "models/llama_finetuned"
        if os.path.exists(model_path):
            fine_tuned_model = LoRAFineTuner()
            fine_tuned_model.load_pretrained(model_path)
            logger.info("Fine-tuned model loaded successfully")
            return True
        else:
            logger.warning(f"Fine-tuned model not found at {model_path}")
            return False
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model: {e}")
        return False


def generate_answer(question: str, context: str) -> Tuple[str, float]:
    """
    Generate answer using fine-tuned model.
    
    Args:
        question: Financial question
        context: Retrieved context
        
    Returns:
        Tuple of (answer, confidence_score)
    """
    if fine_tuned_model is None:
        # Fallback: return a basic response if model not loaded
        return f"Unable to answer: {question}. Model not loaded.", 0.3
    
    try:
        # Create prompt
        prompt = f"""Based on the financial information provided:

{context}

Question: {question}

Answer:"""
        
        # Generate
        answer = fine_tuned_model.generate(
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.7
        )
        
        # Calculate confidence (simplified - would be more sophisticated in production)
        confidence = 0.8
        
        return answer, confidence
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"Error generating answer: {str(e)}", 0.2


# ==================== Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    logger.info("Starting Financial Analyst Chatbot API...")
    load_rag_pipeline()
    load_fine_tuned_model()
    logger.info("API startup complete")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Financial Analyst Chatbot API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        rag_pipeline_loaded=rag_pipeline is not None,
        model_loaded=fine_tuned_model is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/v1/search", response_model=SearchResponse, tags=["RAG"])
async def search_documents(request: SearchRequest):
    """
    Search for relevant financial documents.
    
    Args:
        request: Search request with query and parameters
        
    Returns:
        List of relevant documents with similarity scores
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not loaded")
    
    try:
        results = rag_pipeline.search(
            query=request.query,
            k=request.k
        )
        
        # Filter by score threshold
        filtered_results = [
            r for r in results 
            if r.get('similarity_score', 0) >= request.score_threshold
        ]
        
        return SearchResponse(
            query=request.query,
            results=filtered_results,
            total_results=len(filtered_results),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=400, detail=f"Search error: {str(e)}")


@app.post("/api/v1/analyze", response_model=AnalystAnswer, tags=["Analysis"])
async def analyze_financial_question(request: AnalystQuestion):
    """
    Analyze a financial question using RAG + fine-tuned model.
    
    Args:
        request: Financial question request
        
    Returns:
        Answer with sources and confidence
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not loaded")
    
    try:
        # Retrieve context
        context = rag_pipeline.get_context(
            query=request.question,
            k=5,
            max_chars=3000
        )
        
        # Retrieve search results
        search_results = rag_pipeline.search(request.question, k=5)
        
        # Generate answer
        answer, confidence = generate_answer(request.question, context)
        
        return AnalystAnswer(
            question=request.question,
            answer=answer,
            confidence=confidence,
            sources=search_results[:3],  # Top 3 sources
            retrieved_context=context if request.include_context else None,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=400, detail=f"Analysis error: {str(e)}")


@app.post("/api/v1/ingest", response_model=IngestDocumentsResponse, tags=["Data"])
async def ingest_documents(
    request: IngestDocumentsRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest financial documents into vector database.
    
    Args:
        request: Document paths to ingest
        background_tasks: FastAPI background tasks
        
    Returns:
        Ingestion status
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not loaded")
    
    try:
        # Validate files
        existing_files = []
        for file_path in request.file_paths:
            if os.path.exists(file_path):
                existing_files.append(file_path)
            else:
                logger.warning(f"File not found: {file_path}")
        
        if not existing_files:
            raise HTTPException(status_code=400, detail="No valid files provided")
        
        # Ingest (can be done in background)
        chunks_added = rag_pipeline.ingest_documents(existing_files)
        
        return IngestDocumentsResponse(
            status="success",
            chunks_added=chunks_added,
            files_processed=len(existing_files),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=400, detail=f"Ingestion error: {str(e)}")


@app.post("/api/v1/generate", tags=["Generation"])
async def generate_text(
    prompt: str = Field(..., description="Input prompt", min_length=5, max_length=1000),
    max_tokens: int = Field(256, description="Max tokens to generate", ge=1, le=1000),
    temperature: float = Field(0.7, description="Temperature", ge=0.1, le=2.0)
):
    """
    Generate text using fine-tuned model.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    if fine_tuned_model is None:
        raise HTTPException(status_code=503, detail="Fine-tuned model not loaded")
    
    try:
        generated = fine_tuned_model.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "prompt": prompt,
            "generated_text": generated,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=400, detail=f"Generation error: {str(e)}")


@app.get("/api/v1/stats", tags=["General"])
async def get_statistics():
    """Get system statistics."""
    rag_stats = rag_pipeline.get_stats() if rag_pipeline else {}
    
    return {
        "rag_pipeline": rag_stats,
        "timestamp": datetime.now().isoformat()
    }


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# ==================== Run ====================

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("API_DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        debug=debug,
        log_level="info"
    )
