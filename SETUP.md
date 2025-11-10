# ⚙️ Setup & Installation Guide

Complete step-by-step guide to set up and run Financial Analyst Advisor.

## 📋 System Requirements

### Minimum Hardware
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 16GB (32GB for fine-tuning)
- **GPU**: 16GB VRAM (for fine-tuning with 4-bit quantization)
- **Storage**: 50GB+ (for models and data)

### Supported OS
- macOS 11+
- Ubuntu 20.04+
- Windows 10/11 (with WSL2 recommended)

### Software Requirements
- Python 3.9+ (recommended 3.10 or 3.11)
- Git
- CUDA 11.8+ (for GPU acceleration, optional but recommended)

---

## 🚀 Installation Steps

### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/yourrepo/financial-analyst-advisor.git
cd "Financial Analyst Advisor"

# Verify structure
ls -la
```

### Step 2: Set Up Python Environment

#### Option A: Using venv (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show (venv) prefix)
which python
```

#### Option B: Using Conda

```bash
# Create conda environment
conda create -n financial-advisor python=3.11

# Activate environment
conda activate financial-advisor

# Verify activation (should show (financial-advisor) prefix)
python --version
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Step 4: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your settings
nano .env  # or use your preferred editor
```

**Key settings to configure:**

```env
# HuggingFace Token (REQUIRED for LLaMA model)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxx

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Model Configuration
BASE_MODEL=meta-llama/Llama-2-7b-hf
APP_ENV=development

# GPU Configuration
DEVICE=auto
CUDA_DEVICE=0
```

### Step 5: HuggingFace Setup

This is **required** to access LLaMA models.

#### 5a. Create HuggingFace Account

1. Go to https://huggingface.co
2. Click "Sign Up" and create account
3. Verify email

#### 5b. Accept LLaMA License

1. Navigate to https://huggingface.co/meta-llama/Llama-2-7b-hf
2. Click "Agree and access repository"
3. Accept the license terms

#### 5c. Generate Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Read" permission
4. Copy token
5. Add to `.env`: `HF_TOKEN=hf_your_token_here`

#### 5d. Configure HuggingFace CLI

```bash
# Install HF CLI (if not already installed)
pip install huggingface-hub

# Login with token
huggingface-cli login

# Paste your token when prompted
```

Verify:
```bash
huggingface-cli whoami
```

### Step 6: Test Installation

```bash
# Run quick verification
python examples.py

# Output should show:
# ✓ Sample dataset generated
# ✓ Train/test split created
# ✓ LLaMA format file created
# ... etc
```

---

## 🎯 Quick Start (Without Fine-tuning)

For immediate testing without training your own model:

```bash
# 1. Generate sample data
python -c "from src.fine_tuning.dataset_generator import generate_sample_dataset; generate_sample_dataset()"

# 2. Initialize RAG pipeline
python -c "from src.rag.rag_pipeline import RAGPipeline; RAGPipeline()"

# 3. Start API server
python -m src.api.app

# 4. In another terminal, test API
curl http://localhost:8000/health
```

Open browser to: **http://localhost:8000/docs**

---

## 🤖 Full Setup (With Fine-tuning)

For complete setup with model fine-tuning:

### Step 1: Prepare Training Data

```bash
# Generate sample training dataset
python -c "
from src.fine_tuning.dataset_generator import generate_sample_dataset
dataset_file = generate_sample_dataset()
print(f'Dataset: {dataset_file}')
"

# Output: data/processed/finetune_dataset.jsonl
```

Or use your own data:
```bash
# Scrape SEC data (requires configuring companies)
python -c "
from src.scraping.sec_scraper import scrape_multiple_companies
files = scrape_multiple_companies(['Apple', 'Microsoft'], num_filings=3)
"
```

### Step 2: Create Training/Test Split

```bash
python -c "
from src.fine_tuning.dataset_generator import FineTuningDatasetGenerator

generator = FineTuningDatasetGenerator()
train_file, test_file = generator.create_training_split(
    'data/processed/finetune_dataset.jsonl'
)
print(f'Train: {train_file}')
print(f'Test: {test_file}')
"
```

### Step 3: Fine-tune Model

```bash
# This will take 1-4 hours depending on GPU
python -c "
from src.fine_tuning.lora_finetuner import LoRAFineTuner

finetuner = LoRAFineTuner()
finetuner.apply_lora()

finetuner.train(
    train_file='data/processed/llama_finetuning_train.jsonl',
    eval_file='data/processed/llama_finetuning_test.jsonl',
    output_dir='models/llama_finetuned',
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-4
)
"
```

**Note**: First run will download the LLaMA model (~15GB)

### Step 4: Verify Fine-tuned Model

```bash
python -c "
from src.fine_tuning.lora_finetuner import LoRAFineTuner

finetuner = LoRAFineTuner()
finetuner.load_pretrained('models/llama_finetuned')

response = finetuner.generate(
    'What is important about financial analysis?',
    max_new_tokens=100
)
print(response)
"
```

### Step 5: Start API with Fine-tuned Model

```bash
# API will automatically load fine-tuned model if found
python -m src.api.app
```

---

## 🧪 Testing API Endpoints

### Using cURL

```bash
# 1. Health check
curl http://localhost:8000/health | json_pp

# 2. Search documents
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was revenue?",
    "k": 5
  }' | json_pp

# 3. Analyze question
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main risks?"
  }' | json_pp
```

### Using Python Client

```bash
# Start API in terminal 1
python -m src.api.app

# Run client in terminal 2
python api_client.py
```

### Using Swagger UI

Visit: **http://localhost:8000/docs**

- Try out endpoints interactively
- View request/response schemas
- Test with different parameters

---

## 📊 Verify Installation

```bash
# Check all components
python -c "
print('Checking installation...')

# 1. PyTorch
import torch
print(f'✓ PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')

# 2. Transformers
import transformers
print(f'✓ Transformers {transformers.__version__}')

# 3. FastAPI
import fastapi
print(f'✓ FastAPI {fastapi.__version__}')

# 4. ChromaDB
import chromadb
print(f'✓ ChromaDB {chromadb.__version__}')

# 5. LangChain
import langchain
print(f'✓ LangChain {langchain.__version__}')

# 6. PEFT
import peft
print(f'✓ PEFT {peft.__version__}')

print('\\n✓ All dependencies installed!')
"
```

---

## 🔧 Troubleshooting

### Problem: ModuleNotFoundError

```bash
# Solution: Make sure venv is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### Problem: CUDA Out of Memory

```bash
# Solutions:
# 1. Reduce batch size in .env
TRAIN_BATCH_SIZE=2

# 2. Use CPU instead
DEVICE=cpu

# 3. Use smaller model
BASE_MODEL=meta-llama/Llama-2-3b-hf
```

### Problem: HuggingFace Authentication Failed

```bash
# Solution: Re-login
huggingface-cli logout
huggingface-cli login

# Or set token directly
export HF_TOKEN="hf_your_token"
```

### Problem: Port Already in Use

```bash
# Solution: Use different port
API_PORT=8001 python -m src.api.app

# Or kill process using port 8000
# macOS/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Problem: Slow Inference

```bash
# Solutions:
# 1. Use smaller embedding model
# 2. Reduce context window
# 3. Enable batching
# 4. Use GPU for inference
```

---

## 📚 Next Steps After Installation

1. **Explore Documentation**
   - Read `README.md` for overview
   - Review `ARCHITECTURE.md` for system design
   - Check API docs at `/docs` endpoint

2. **Run Examples**
   ```bash
   python examples.py
   ```

3. **Fine-tune Model**
   - Follow "Full Setup" section above
   - Start with sample data
   - Iterate with your own data

4. **Deploy to Production**
   - See `README.md` deployment section
   - Configure for cloud platform
   - Set up monitoring and logging

5. **Customize for Your Use Case**
   - Modify dataset generation
   - Adjust fine-tuning parameters
   - Integrate with your systems

---

## 💻 Development Commands

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/

# Type checking
mypy src/

# Run tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Generate docs
cd docs && make html

# Start in development mode
API_DEBUG=true python -m src.api.app
```

---

## 📝 File Structure

After setup, you should have:

```
Financial Analyst Advisor/
├── venv/                          # Virtual environment
├── data/
│   ├── raw_reports/              # Downloaded SEC filings
│   ├── processed/                 # Generated datasets
│   └── vector_db/                # Chroma database
├── src/
│   ├── scraping/                 # SEC scraper
│   ├── rag/                       # RAG pipeline
│   ├── fine_tuning/              # Fine-tuning engine
│   ├── api/                       # FastAPI app
│   └── utils/                     # Configuration
├── models/
│   └── llama_finetuned/          # Fine-tuned weights
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── .env                           # Configuration (auto-created)
├── requirements.txt               # Dependencies
├── README.md                      # Main documentation
├── ARCHITECTURE.md               # System design
├── SETUP.md                       # This file
├── examples.py                    # Example scripts
├── quickstart.py                  # Interactive setup
└── api_client.py                  # Python client
```

---

## 🎓 Learning Resources

- **LLaMA & Fine-tuning**: https://github.com/meta-llama/llama
- **LoRA**: https://arxiv.org/abs/2106.09685
- **RAG Pattern**: https://arxiv.org/abs/2005.11401
- **FastAPI**: https://fastapi.tiangolo.com/
- **LangChain**: https://python.langchain.com/

---

## ✅ Installation Checklist

- [ ] Python 3.9+ installed
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Requirements installed
- [ ] `.env` file configured
- [ ] HuggingFace token obtained and set
- [ ] Installation verified
- [ ] Example scripts run successfully
- [ ] API starts without errors
- [ ] API endpoints respond at `/health`

---

## 🆘 Need Help?

1. Check the **Troubleshooting** section above
2. Review `README.md` FAQ
3. Check error logs: Enable debug mode
4. Open issue on GitHub with:
   - System info (OS, Python version)
   - Error message
   - Steps to reproduce
   - Output of `python examples.py`

---

**Happy analyzing! 📊**
