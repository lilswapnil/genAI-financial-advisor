# Makefile for Financial Analyst Advisor

.PHONY: help install clean test lint format run dev docs

# Default target
help:
	@echo "Financial Analyst Advisor - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install     Install dependencies"
	@echo "  dev-install Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  clean       Clean cache files and artifacts"
	@echo "  format      Format code with black"
	@echo "  lint        Lint code with flake8"
	@echo "  test        Run tests"
	@echo "  type-check  Run mypy type checking"
	@echo ""
	@echo "Running:"
	@echo "  run         Start the API server"
	@echo "  examples    Run example demonstrations"
	@echo "  quickstart  Run interactive quickstart"
	@echo ""
	@echo "Data & Training:"
	@echo "  gen-data    Generate sample training data"
	@echo "  ingest      Ingest documents to vector DB"
	@echo ""
	@echo "Documentation:"
	@echo "  docs        Open API documentation"

# Installation
install:
	pip install -r requirements.txt

dev-install: install
	pip install pytest pytest-cov black flake8 mypy

# Development
clean:
	python cleanup.py

format:
	black src/ examples.py quickstart.py api_client.py cleanup.py

lint:
	flake8 src/ examples.py quickstart.py api_client.py --max-line-length=88 --extend-ignore=E203,W503

test:
	pytest tests/ -v --cov=src/ --cov-report=html --cov-report=term

type-check:
	mypy src/

# Running
run:
	python -m src.api.app

examples:
	python examples.py

quickstart:
	python quickstart.py

# Data & Training
gen-data:
	python -c "from src.fine_tuning.dataset_generator import generate_sample_dataset; generate_sample_dataset()"

ingest:
	python -c "from src.rag.rag_pipeline import RAGPipeline; pipeline = RAGPipeline(); pipeline.ingest_documents()"

# Documentation
docs:
	@echo "Starting API server and opening documentation..."
	@echo "Visit: http://localhost:8000/docs"
	python -m src.api.app

# Combined targets
dev-setup: clean install dev-install gen-data
	@echo "Development environment setup complete!"

all-checks: format lint type-check test
	@echo "All code quality checks passed!"