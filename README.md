# 📊 Financial Analyst Advisor - AI-Powered Financial Report Analysis

A cutting-edge **Retrieval-Augmented Generation (RAG)** chatbot powered by a **fine-tuned LLaMA model with LoRA** that intelligently analyzes SEC financial reports (10-K, 10-Q) and answers complex financial questions.

## 🎯 Key Features

- **📥 SEC Data Ingestion**: Automatically scrapes financial reports from SEC EDGAR database
- **🔍 Smart Document Retrieval**: RAG pipeline with semantic search using BERT embeddings
- **🤖 Fine-tuned LLaMA Model**: Custom financial analyst model using LoRA (Low-Rank Adaptation)
- **⚡ Efficient Fine-tuning**: 4-bit quantization and gradient checkpointing for GPU optimization
- **🚀 REST API**: FastAPI endpoints for question answering and document search
- **💾 Vector Database**: Chroma for efficient semantic search with Pinecone support

## 🏗️ Project Architecture

```
Financial Analyst Advisor/
├── data/
│   ├── raw_reports/           # Downloaded SEC filings
│   ├── processed/             # Generated training datasets
│   └── vector_db/             # Chroma vector database
├── src/
│   ├── scraping/
│   │   └── sec_scraper.py     # SEC EDGAR data collection
│   ├── rag/
│   │   └── rag_pipeline.py    # Document chunking & retrieval
│   ├── fine_tuning/
│   │   ├── dataset_generator.py   # QA dataset creation
│   │   └── lora_finetuner.py      # LoRA fine-tuning engine
│   ├── api/
│   │   └── app.py             # FastAPI application
│   └── utils/
│       └── config.py          # Configuration management
├── models/
│   └── llama_finetuned/       # Fine-tuned model weights
├── notebooks/
│   └── demo.ipynb             # Interactive demonstrations
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.9+ (recommended 3.10 or 3.11)
- **GPU**: 16GB+ VRAM (for fine-tuning with 4-bit quantization)
- **HuggingFace Account**: Required for LLaMA model access
- **Storage**: 50GB+ for models and data

### 1. Installation

```bash
# Clone repository
git clone <repo-url>
cd "Financial Analyst Advisor"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your HuggingFace token and preferences
```

### 2. HuggingFace Setup

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login to HuggingFace
huggingface-cli login

# Accept LLaMA license at: https://huggingface.co/meta-llama/Llama-2-7b-hf
```

### 3. Generate Training Data

```bash
python -m src.fine_tuning.dataset_generator

# Output: data/processed/llama_finetuning.jsonl
```

### 4. Prepare RAG Pipeline

```bash
python -c "
from src.rag.rag_pipeline import RAGPipeline
pipeline = RAGPipeline()
# Will initialize vector database for later use
"
```

### 5. Run Fine-tuning (Optional)

```bash
# Full fine-tuning (requires significant GPU memory)
python -c "
from src.fine_tuning.lora_finetuner import LoRAFineTuner

finetuner = LoRAFineTuner()
finetuner.apply_lora()
finetuner.train(
    train_file='data/processed/llama_finetuning.jsonl',
    output_dir='models/llama_finetuned'
)
"
```

### 6. Start API Server

```bash
# Development mode
python -m src.api.app

# Or with custom settings
API_HOST=0.0.0.0 API_PORT=8000 python -m src.api.app

# API will be available at: http://localhost:8000
```

## 📚 API Documentation

### Interactive Docs

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "rag_pipeline_loaded": true,
  "model_loaded": true,
  "timestamp": "2024-11-09T10:30:00"
}
```

#### 2. Search Financial Documents
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Apple total revenue in 2023?",
    "k": 5,
    "score_threshold": 0.3
  }'
```

Response:
```json
{
  "query": "What was Apple total revenue in 2023?",
  "results": [
    {
      "content": "Apple Inc. reported total revenue of $383.3 billion...",
      "metadata": {
        "cik": "0000320193",
        "filing_type": "10-K"
      },
      "similarity_score": 0.87
    }
  ],
  "total_results": 3,
  "timestamp": "2024-11-09T10:30:00"
}
```

#### 3. Financial Analysis (RAG + Fine-tuned Model)
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main business risks identified by Apple?",
    "include_context": true
  }'
```

Response:
```json
{
  "question": "What are the main business risks identified by Apple?",
  "answer": "Based on Apple's 10-K filing, main risks include supply chain disruptions...",
  "confidence": 0.85,
  "sources": [
    {
      "content": "Risk Factors section...",
      "metadata": {"filing_type": "10-K"},
      "similarity_score": 0.92
    }
  ],
  "retrieved_context": "RETRIEVED FINANCIAL DOCUMENTS: ...",
  "timestamp": "2024-11-09T10:30:00"
}
```

#### 4. Ingest Documents
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": [
      "data/raw_reports/AAPL_10K.txt",
      "data/raw_reports/MSFT_10Q.txt"
    ]
  }'
```

#### 5. Generate Text
```bash
curl -X POST "http://localhost:8000/api/v1/generate?prompt=What%20is%20revenue%20recognition&max_tokens=256&temperature=0.7"
```

## 🔧 Module Deep-Dive

### 1. SEC Scraper (`src/scraping/sec_scraper.py`)

Fetches financial reports from SEC EDGAR database.

**Features:**
- Retrieves 10-K and 10-Q filings
- Handles rate limiting
- Stores raw documents

**Usage:**
```python
from src.scraping.sec_scraper import SECScraper

scraper = SECScraper(output_dir="data/raw_reports")

# Get CIK for company
cik = scraper.get_cik("Apple")

# Get recent filings
filings = scraper.get_filings(cik, filing_types=["10-K"], limit=5)

# Download filing
filepath = scraper.download_filing(cik, filing['accession_number'], "10-K")
```

### 2. RAG Pipeline (`src/rag/rag_pipeline.py`)

Implements complete retrieval-augmented generation pipeline.

**Components:**
- **DocumentProcessor**: Chunks documents with semantic overlap
- **EmbeddingManager**: Generates embeddings with sentence-transformers
- **VectorStore**: Manages Chroma vector database
- **RAGPipeline**: Orchestrates all components

**Usage:**
```python
from src.rag.rag_pipeline import RAGPipeline

# Initialize
pipeline = RAGPipeline(
    chunk_size=1024,
    chunk_overlap=128
)

# Ingest documents
num_chunks = pipeline.ingest_documents(["data/raw_reports/AAPL_10K.txt"])

# Search
results = pipeline.search("What was revenue growth?", k=5)

# Get context for LLM
context = pipeline.get_context("What are the risks?", k=3)
```

### 3. Dataset Generator (`src/fine_tuning/dataset_generator.py`)

Creates training data for fine-tuning.

**Features:**
- Generates QA pairs from documents
- Creates train/test splits
- Formats for LLaMA fine-tuning

**Usage:**
```python
from src.fine_tuning.dataset_generator import FineTuningDatasetGenerator

generator = FineTuningDatasetGenerator()

# Generate dataset
dataset_file = generator.generate_dataset_from_documents(
    documents=[...],
    samples_per_document=5
)

# Split train/test
train_file, test_file = generator.create_training_split(dataset_file)

# Format for LLaMA
llama_file = generator.format_for_llama_finetuning(dataset_file)
```

### 4. LoRA Fine-tuner (`src/fine_tuning/lora_finetuner.py`)

Implements efficient fine-tuning using LoRA.

**Key Features:**
- 4-bit quantization support
- Gradient checkpointing
- LoRA parameter-efficient tuning
- Weights & Biases integration

**Usage:**
```python
from src.fine_tuning.lora_finetuner import LoRAFineTuner

# Initialize
finetuner = LoRAFineTuner(
    model_name="meta-llama/Llama-2-7b-hf",
    use_4bit=True
)

# Apply LoRA
finetuner.apply_lora(r=16, lora_alpha=32)

# Train
finetuner.train(
    train_file="data/processed/llama_finetuning.jsonl",
    output_dir="models/llama_finetuned",
    num_epochs=3,
    batch_size=4
)

# Generate
response = finetuner.generate(
    prompt="What was Apple's revenue?",
    max_new_tokens=256
)
```

### 5. FastAPI Application (`src/api/app.py`)

REST API for the entire system.

**Endpoints:**
- `GET /health` - Health check
- `POST /api/v1/search` - Document search
- `POST /api/v1/analyze` - Financial analysis
- `POST /api/v1/ingest` - Document ingestion
- `POST /api/v1/generate` - Text generation
- `GET /api/v1/stats` - System statistics

## 📊 Example Workflows

### Workflow 1: Analyze Apple's Financial Health

```python
from src.rag.rag_pipeline import RAGPipeline
from src.fine_tuning.lora_finetuner import LoRAFineTuner

# Load pipeline
pipeline = RAGPipeline()
finetuner = LoRAFineTuner()
finetuner.load_pretrained("models/llama_finetuned")

# Get context
question = "What is Apple's current financial health?"
context = pipeline.get_context(question, k=5)

# Generate answer
answer = finetuner.generate(
    prompt=f"Based on: {context}\n\nQuestion: {question}",
    max_new_tokens=512
)

print(answer)
```

### Workflow 2: Compare Companies

```python
# Search for revenue comparison
results = pipeline.search("Compare revenue across tech companies")

# Extract company data
companies = {}
for result in results:
    cik = result['metadata']['cik']
    content = result['content']
    # Process and aggregate...
```

## ⚙️ Configuration

Edit `.env` file or set environment variables:

```bash
# Model
BASE_MODEL=meta-llama/Llama-2-7b-hf
HF_TOKEN=hf_xxxxxxxxxxxx

# Fine-tuning
NUM_TRAIN_EPOCHS=3
TRAIN_BATCH_SIZE=4
LORA_R=16

# API
API_PORT=8000
API_DEBUG=false

# Database
VECTOR_DB_TYPE=chroma
```

For advanced configuration, edit `src/utils/config.py`.

## 📈 Performance Tips

### GPU Memory Optimization
```python
# Use 4-bit quantization
finetuner = LoRAFineTuner(use_4bit=True)

# Enable gradient checkpointing
finetuner = LoRAFineTuner(use_gradient_checkpointing=True)

# Reduce batch size and gradient accumulation
# in training arguments
```

### Inference Speed
```python
# Use smaller embedding model
pipeline = RAGPipeline(
    embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v2"
)

# Reduce context window
context = pipeline.get_context(query, k=3, max_chars=2000)
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Test specific module
pytest tests/test_rag_pipeline.py -v
```

## 📋 Supported File Formats

- **Text (.txt)**: Plain text SEC filings
- **JSONL (.jsonl)**: Training dataset format
- **JSON (.json)**: Configuration and metadata

## 🛠️ Troubleshooting

### GPU Memory Error
```bash
# Use smaller model or reduce batch size
TRAIN_BATCH_SIZE=2 python -m src.fine_tuning.lora_finetuner
```

### HuggingFace Authentication
```bash
# Login again
huggingface-cli login

# Or set token directly
export HF_TOKEN="hf_xxxxxxxxxxxx"
```

### Vector Database Issues
```bash
# Reset Chroma database
rm -rf data/vector_db/

# Reingest documents
python -c "from src.rag.rag_pipeline import RAGPipeline; pipeline = RAGPipeline(); pipeline.ingest_documents()"
```

## 📚 Dependencies

**Core ML/AI:**
- `torch` - Deep learning framework
- `transformers` - Pre-trained models
- `peft` - LoRA implementation
- `langchain` - RAG orchestration
- `sentence-transformers` - Embeddings

**Vector DB:**
- `chromadb` - Vector database
- `pinecone-client` - Optional cloud vector DB

**API:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server

**Optimization:**
- `bitsandbytes` - 4-bit quantization
- `xformers` - Flash attention

See `requirements.txt` for complete list.

## 🚢 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "src.api.app"]
```

### Cloud Deployment
- **AWS**: SageMaker endpoint with LLaMA
- **Google Cloud**: Vertex AI with custom container
- **Azure**: Azure Container Instances + Cosmos DB

## 📖 Documentation

- [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) - Financial Reports Database
- [LangChain](https://langchain.readthedocs.io/) - RAG Orchestration
- [PEFT](https://huggingface.co/docs/peft/) - LoRA Implementation
- [Chroma](https://docs.trychroma.com/) - Vector Database
- [FastAPI](https://fastapi.tiangolo.com/) - API Framework

## 📊 Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| Model Size | 7B | LLaMA 2 base |
| Fine-tunable Params | 2.3% | LoRA with r=16 |
| GPU Memory (Fine-tune) | ~16GB | With 4-bit quantization |
| Inference Speed | ~10 tokens/sec | Single GPU |
| Vector DB Latency | <100ms | Semantic search on 1000 docs |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- LLaMA models by Meta AI
- HuggingFace for transformers and model hub
- LangChain for RAG framework
- Chroma for vector database
- SEC for financial data

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review example notebooks

---

**Built with ❤️ for financial AI analysis**
