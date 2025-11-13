# 🏦 Financial Analyst Advisor

> **AI-powered financial analysis using RAG and fine-tuned LLaMA models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent financial analysis system that combines **Retrieval-Augmented Generation (RAG)** with **fine-tuned LLaMA models** to analyze SEC financial reports and answer complex financial questions with high accuracy and context awareness.

## ✨ Features

- 🔍 **SEC Document Analysis**: Automated scraping and analysis of 10-K/10-Q filings
- 🧠 **RAG Pipeline**: Advanced document retrieval with semantic search
- 🎯 **Fine-tuned LLaMA**: LoRA-adapted model for financial expertise
- ⚡ **REST API**: Production-ready FastAPI with comprehensive endpoints
- 📊 **Vector Database**: Efficient similarity search with ChromaDB
- 🛠️ **Developer Tools**: Automated workflows with Makefile and cleanup scripts

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ 
- 16GB+ GPU memory (for fine-tuning)
- HuggingFace account for model access

### Installation

```bash
# Clone and setup
git clone https://github.com/lilswapnil/genAI-financial-advisor.git
cd genAI-financial-advisor

# Install dependencies (using make for convenience)
make install

# Or manually
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your HuggingFace token
# HF_TOKEN=hf_xxxxxxxxxxxx
```

### Quick Demo

```bash
# Generate sample training data
make gen-data

# Start the API server
make run

# Visit http://localhost:8000/docs for interactive API documentation
```

## 🏗️ Architecture

```
src/
├── api/           # FastAPI REST endpoints
├── rag/           # RAG pipeline & vector database
├── fine_tuning/   # LoRA fine-tuning & dataset generation
├── scraping/      # SEC EDGAR data collection  
└── utils/         # Configuration & utilities

data/
├── raw_reports/   # Downloaded SEC filings
├── processed/     # Training datasets
└── vector_db/     # ChromaDB persistence

models/
└── llama_finetuned/  # Fine-tuned model weights
```

## 📚 Usage Examples

### 1. Document Analysis via API

```bash
# Search financial documents
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Apple revenue growth 2023", "k": 5}'

# Get AI-powered analysis
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Apple main business risks?"}'
```

### 2. Python Integration

```python
from src.rag.rag_pipeline import RAGPipeline
from src.fine_tuning.lora_finetuner import LoRAFineTuner

# Initialize components
pipeline = RAGPipeline()
finetuner = LoRAFineTuner()

# Analyze financial question
question = "What was Apple's revenue growth strategy?"
context = pipeline.get_context(question, k=3)
answer = finetuner.generate(f"Context: {context}\nQ: {question}")
```

### 3. Data Collection

```python
from src.scraping.sec_scraper import SECScraper

# Download SEC filings
scraper = SECScraper()
files = scraper.scrape_company("Apple", num_filings=5)
```

## 🛠️ Development

### Available Commands (Makefile)

```bash
make help          # Show all available commands
make install       # Install dependencies
make clean         # Clean cache files and artifacts
make format        # Format code with black
make lint          # Check code quality with flake8
make test          # Run tests with coverage
make run           # Start API server
make examples      # Run demonstration examples
```

### Development Workflow

```bash
# Setup development environment
make dev-setup

# Make code changes...

# Check code quality
make format lint test

# Run locally
make run
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `POST` | `/api/v1/search` | Document similarity search |
| `POST` | `/api/v1/analyze` | AI-powered financial analysis |
| `POST` | `/api/v1/ingest` | Upload documents to vector DB |
| `POST` | `/api/v1/generate` | Direct model text generation |
| `GET` | `/api/v1/stats` | System statistics |

## ⚙️ Configuration

Key configuration options in `.env`:

```bash
# Model Configuration
BASE_MODEL=meta-llama/Llama-2-7b-hf
HF_TOKEN=your_huggingface_token

# API Settings  
API_PORT=8000
API_DEBUG=false

# Fine-tuning Parameters
NUM_TRAIN_EPOCHS=3
TRAIN_BATCH_SIZE=4
LORA_R=16
```

## 🧪 Fine-tuning

Generate training data and fine-tune the model:

```bash
# Generate financial QA dataset
python -c "
from src.fine_tuning.dataset_generator import generate_sample_dataset
generate_sample_dataset()
"

# Fine-tune with LoRA (requires GPU)
python -c "
from src.fine_tuning.lora_finetuner import LoRAFineTuner
finetuner = LoRAFineTuner(use_4bit=True)
finetuner.apply_lora()
finetuner.train('data/processed/llama_finetuning.jsonl')
"
```

## 🚢 Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "src.api.app"]
```

### Production Tips

- Use GPU instances for model inference
- Configure load balancing for multiple API instances  
- Set up monitoring with the `/health` endpoint
- Use persistent volumes for model weights and vector DB

## 📈 Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Model Size | 7B parameters | LLaMA-2 base |
| LoRA Efficiency | 2.3% trainable | 16GB GPU memory |
| API Latency | <2s | Document search + generation |
| Vector Search | <100ms | 1000+ document corpus |

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/new-feature`
3. **Make** changes and run: `make format lint test`
4. **Commit** with clear messages: `git commit -m "Add new feature"`
5. **Push** and create Pull Request

## 🛠️ Troubleshooting

### Common Issues

**GPU Memory Error**
```bash
# Reduce batch size
export TRAIN_BATCH_SIZE=2
```

**HuggingFace Auth**
```bash
# Re-authenticate
huggingface-cli login
```

**Vector DB Reset**
```bash
# Clear and reinitialize
rm -rf data/vector_db/
make gen-data
```

## 📚 Dependencies

**Core Stack:**
- **torch** - Deep learning framework
- **transformers** - Pre-trained models  
- **langchain** - RAG orchestration
- **fastapi** - Web framework
- **chromadb** - Vector database

See `requirements.txt` for complete list.

## 📖 Documentation

- 📘 [API Documentation](http://localhost:8000/docs) - Interactive API explorer
- 📗 [Setup Guide](SETUP.md) - Detailed installation instructions
- 📕 [Architecture](ARCHITECTURE.md) - System design and components
- 📙 [Refactor Summary](REFACTOR_SUMMARY.md) - Recent improvements

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Meta AI** - LLaMA models
- **HuggingFace** - Transformers ecosystem
- **LangChain** - RAG framework
- **ChromaDB** - Vector database
- **SEC** - Financial data access

---

<p align="center">
  <strong>Built with ❤️ for intelligent financial analysis</strong><br>
  <em>Questions? Open an issue or check the documentation!</em>
</p>
