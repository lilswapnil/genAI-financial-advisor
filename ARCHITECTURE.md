```
╔════════════════════════════════════════════════════════════════════════════╗
║          FINANCIAL ANALYST ADVISOR - SYSTEM ARCHITECTURE                   ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER / CLIENT LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Web UI      │  │  Python      │  │  REST API    │  │  CLI Tools   │    │
│  │  (Frontend)  │  │  Client      │  │  Docs        │  │  (Scripts)   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │                 │            │
│         └──────────────────┼──────────────────┼─────────────────┘            │
│                            │                  │                              │
└────────────────────────────┼──────────────────┼──────────────────────────────┘
                             ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI REST API (Port 8000)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    API ENDPOINTS                                    │    │
│  │  • GET    /health                  - Health check                  │    │
│  │  • POST   /api/v1/search           - Document search             │    │
│  │  • POST   /api/v1/analyze          - Financial analysis          │    │
│  │  • POST   /api/v1/ingest           - Document ingestion          │    │
│  │  • POST   /api/v1/generate         - Text generation             │    │
│  │  • GET    /api/v1/stats            - System statistics           │    │
│  │                                                                    │    │
│  │  • GET    /docs                    - Swagger UI                  │    │
│  │  • GET    /redoc                   - ReDoc documentation         │    │
│  └──────────────┬──────────┬──────────────┬──────────────┬───────────┘    │
│                 │          │              │              │                 │
└─────────────────┼──────────┼──────────────┼──────────────┼─────────────────┘
                  │          │              │              │
            ┌─────▼──┬───────▼────┬─────────▼────┬────────▼────┐
            │        │            │              │             │
            ▼        ▼            ▼              ▼             ▼
        ┌────────────────────────────────────────────────────────────┐
        │              BUSINESS LOGIC LAYER                          │
        ├────────────────────────────────────────────────────────────┤
        │                                                             │
        │  ┌──────────────────────┐  ┌──────────────────────┐       │
        │  │  RAG PIPELINE        │  │  FINE-TUNED MODEL    │       │
        │  ├──────────────────────┤  ├──────────────────────┤       │
        │  │ • DocumentProcessor  │  │ • LoRAFineTuner      │       │
        │  │ • EmbeddingManager   │  │ • Model Inference    │       │
        │  │ • VectorStore        │  │ • Generation         │       │
        │  │ • SemanticSearch     │  │ • LoRA Adapters      │       │
        │  └─────┬────────────────┘  └──────┬───────────────┘       │
        │        │                          │                        │
        └────────┼──────────────────────────┼────────────────────────┘
                 │                          │
        ┌────────▼────────┐        ┌───────▼──────────┐
        │                 │        │                  │
        ▼                 ▼        ▼                  ▼
    ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  VECTOR DB      │  │  LLM BASE MODEL  │  │  LORA WEIGHTS    │
    │  (Chroma)       │  │  (LLaMA 7B)      │  │  (2.3% params)   │
    │                 │  │                  │  │                  │
    │ • Embeddings    │  │ • Quantized      │  │ • q_proj adapt.  │
    │ • Index         │  │ • 4-bit weights  │  │ • v_proj adapt.  │
    │ • Similarity    │  │ • Cached         │  │ • k_proj adapt.  │
    │   Search        │  │                  │  │ • o_proj adapt.  │
    └────────┬────────┘  └────────┬─────────┘  └──────────┬───────┘
             │                    │                       │
             └────────────────────┼───────────────────────┘
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │          DATA LAYER                               │
        ├───────────────────────────────────────────────────┤
        │                                                   │
        │  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
        │  │  Raw     │  │Processed │  │Vector DB     │    │
        │  │  Reports │  │ Datasets │  │Storage       │    │
        │  │          │  │          │  │              │    │
        │  │SEC Filing│  │Training  │  │Chroma Index  │    │
        │  │ Storage  │  │ QA Pairs │  │              │    │
        │  └──────────┘  └──────────┘  └──────────────┘    │
        │                                                   │
        │  Location: data/                                  │
        │  • raw_reports/        - Downloaded SEC filings   │
        │  • processed/          - Generated training data   │
        │  • vector_db/          - Chroma persistence       │
        │                                                   │
        └───────────────────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │     EXTERNAL DATA SOURCES       │
        ├────────────────────────────────┤
        │                                │
        │  ┌──────────────────────┐      │
        │  │   SEC EDGAR          │      │
        │  │   Database           │      │
        │  │                      │      │
        │  │ • 10-K Filings       │      │
        │  │ • 10-Q Reports       │      │
        │  │ • Company Info       │      │
        │  │ • Financial Data     │      │
        │  └──────────────────────┘      │
        │                                │
        │  ┌──────────────────────┐      │
        │  │ HuggingFace Hub      │      │
        │  │                      │      │
        │  │ • Base LLaMA Model   │      │
        │  │ • Embedding Models   │      │
        │  │ • Tokenizers         │      │
        │  └──────────────────────┘      │
        │                                │
        └────────────────────────────────┘


╔════════════════════════════════════════════════════════════════════════════╗
║                          DATA FLOW EXAMPLE                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

USER QUERY: "What was Apple's total revenue in 2023?"
    │
    ├─→ [1] API Request → POST /api/v1/analyze
    │
    ├─→ [2] RAG Pipeline
    │       ├─→ Embed query using sentence-transformers
    │       ├─→ Search vector database for similar chunks
    │       └─→ Retrieve top-5 relevant documents
    │
    ├─→ [3] Context Assembly
    │       └─→ Combine retrieved documents into context string
    │
    ├─→ [4] Fine-tuned Model
    │       ├─→ Load LLaMA with LoRA adapters
    │       ├─→ Construct prompt with context + question
    │       ├─→ Generate response (256 tokens)
    │       └─→ Calculate confidence score
    │
    ├─→ [5] Response Assembly
    │       ├─→ Format answer with metadata
    │       ├─→ Include source documents
    │       └─→ Add timestamp
    │
    └─→ [6] Return to Client
        └─→ JSON response with answer, confidence, sources


╔════════════════════════════════════════════════════════════════════════════╗
║                     FINE-TUNING ARCHITECTURE                               ║
╚════════════════════════════════════════════════════════════════════════════╝

TRAINING PIPELINE:
    │
    ├─→ [1] Data Preparation
    │       ├─→ Download SEC filings (raw documents)
    │       ├─→ Chunk documents into segments
    │       └─→ Generate (Question, Answer, Context) tuples
    │
    ├─→ [2] Dataset Creation
    │       ├─→ Create synthetic Q&A pairs
    │       ├─→ Format for LLaMA chat template
    │       ├─→ Split train/test (80/20)
    │       └─→ Save as JSONL
    │
    ├─→ [3] Model Setup
    │       ├─→ Load LLaMA-7B base model
    │       ├─→ Apply 4-bit quantization
    │       └─→ Initialize LoRA adapters
    │
    ├─→ [4] Training Loop (3 epochs)
    │       ├─→ Forward pass through base + LoRA
    │       ├─→ Compute loss on training data
    │       ├─→ Gradient accumulation (2 steps)
    │       └─→ Update only LoRA weights (~2.3% of params)
    │
    └─→ [5] Model Checkpoint
            ├─→ Save LoRA weights
            ├─→ Save tokenizer
            └─→ Save training config


╔════════════════════════════════════════════════════════════════════════════╗
║                      TECHNOLOGY STACK                                       ║
╚════════════════════════════════════════════════════════════════════════════╝

FRAMEWORK & MODELS:
  • PyTorch         - Deep learning framework
  • Transformers    - Pre-trained model hub (HuggingFace)
  • PEFT            - Parameter-efficient fine-tuning (LoRA)
  • LangChain       - RAG orchestration
  • Sentence-BERT   - Semantic embeddings

VECTOR DATABASE:
  • Chroma          - Open-source vector database (default)
  • Pinecone        - Cloud vector database (optional)

API FRAMEWORK:
  • FastAPI         - Modern REST API framework
  • Uvicorn         - ASGI web server
  • Pydantic        - Data validation

OPTIMIZATION:
  • BitsAndBytes    - 4-bit quantization
  • Xformers        - Flash Attention
  • Accelerate      - Multi-GPU support

DATA SOURCES:
  • SEC EDGAR       - Financial reports
  • BeautifulSoup   - Web scraping
  • Requests        - HTTP client

MONITORING:
  • Weights & Biases - Experiment tracking (optional)
  • Python logging   - Application logging


╔════════════════════════════════════════════════════════════════════════════╗
║                    DEPLOYMENT ARCHITECTURE                                 ║
╚════════════════════════════════════════════════════════════════════════════╝

DEVELOPMENT:
  Local Machine → Python Process → Port 8000

PRODUCTION:
  Client Request
    │
    ├─→ Load Balancer (nginx/HAProxy)
    │
    ├─→ Container Pool (Docker/Kubernetes)
    │   ├─→ API Container 1 (FastAPI)
    │   ├─→ API Container 2 (FastAPI)
    │   └─→ API Container N (FastAPI)
    │
    ├─→ Shared Model Cache (GPU Memory)
    │   └─→ LLaMA + LoRA weights
    │
    ├─→ Vector Database (Cloud)
    │   └─→ Pinecone / Managed Chroma
    │
    └─→ Document Storage
        └─→ S3 / GCS / Azure Blob

MONITORING & LOGGING:
  ├─→ Application Logs → CloudWatch / Datadog
  ├─→ Metrics → Prometheus
  ├─→ Tracing → Jaeger
  └─→ Error Tracking → Sentry
```

## Component Responsibilities

### API Layer (src/api/app.py)
- HTTP request handling
- Request validation
- Response serialization
- Error handling and logging

### RAG Pipeline (src/rag/rag_pipeline.py)
- Document loading and preprocessing
- Semantic chunking
- Embedding generation
- Vector similarity search

### Fine-Tuning Engine (src/fine_tuning/lora_finetuner.py)
- Model loading and quantization
- LoRA adapter application
- Training loop execution
- Model inference

### Dataset Generator (src/fine_tuning/dataset_generator.py)
- QA pair generation from documents
- Training dataset creation
- Train/test splitting
- Format conversion for LLaMA

### SEC Scraper (src/scraping/sec_scraper.py)
- SEC EDGAR API integration
- Filing download and storage
- Rate limiting and error handling

---

**Generated on**: November 2025
**Version**: 1.0.0
