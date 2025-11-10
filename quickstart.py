#!/usr/bin/env python3
"""
Quick start script for Financial Analyst Advisor.

This script provides an interactive menu for:
- Setting up the project
- Generating sample data
- Starting the API
- Running examples
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def check_requirements():
    """Check if all requirements are installed."""
    print_header("Checking Requirements")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  GPU Available: {torch.cuda.is_available()}")
        
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
        
        import fastapi
        print(f"✓ FastAPI: {fastapi.__version__}")
        
        import chromadb
        print(f"✓ ChromaDB: {chromadb.__version__}")
        
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall dependencies with:")
        print("  pip install -r requirements.txt")
        return False


def setup_environment():
    """Setup environment files."""
    print_header("Environment Setup")
    
    env_file = Path(".env")
    if env_file.exists():
        print("✓ .env file already exists")
    else:
        # Copy from template
        template_file = Path(".env.example")
        if template_file.exists():
            env_file.write_text(template_file.read_text())
            print("✓ Created .env from template")
            print("\n⚠️  Please edit .env with your configuration:")
            print("   - Set HF_TOKEN for HuggingFace access")
            print("   - Configure other options as needed")
        else:
            print("✗ .env.example not found")


def generate_sample_data():
    """Generate sample training data."""
    print_header("Generating Sample Data")
    
    try:
        from src.fine_tuning.dataset_generator import generate_sample_dataset
        
        print("Generating sample dataset...")
        dataset_file = generate_sample_dataset()
        print(f"✓ Dataset created: {dataset_file}")
        
        # Show sample
        import json
        with open(dataset_file, 'r') as f:
            sample = json.loads(f.readline())
        
        print(f"\nSample QA:")
        print(f"  Q: {sample['question']}")
        print(f"  A: {sample['answer'][:80]}...")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def start_api():
    """Start the FastAPI server."""
    print_header("Starting API Server")
    
    print("Starting server on http://localhost:8000")
    print("Documentation: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop...\n")
    
    try:
        os.system("python -m src.api.app")
    except KeyboardInterrupt:
        print("\n\nServer stopped.")


def run_examples():
    """Run example demonstrations."""
    print_header("Running Examples")
    
    try:
        result = subprocess.run(
            [sys.executable, "examples.py"],
            capture_output=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def show_menu():
    """Display main menu."""
    print_header("Financial Analyst Advisor")
    print("""
1. Check Requirements
2. Setup Environment
3. Generate Sample Data
4. Run Examples
5. Start API Server
6. Run Fine-tuning
7. Exit

What would you like to do? (Enter number 1-7)
    """)


def main():
    """Main menu loop."""
    print("\n🚀 Welcome to Financial Analyst Advisor\n")
    
    while True:
        show_menu()
        choice = input("Enter choice: ").strip()
        
        if choice == "1":
            check_requirements()
        
        elif choice == "2":
            setup_environment()
        
        elif choice == "3":
            generate_sample_data()
        
        elif choice == "4":
            run_examples()
        
        elif choice == "5":
            start_api()
        
        elif choice == "6":
            print_header("Fine-tuning")
            print("""
To fine-tune the model:

1. Generate training data (option 3)
2. Run: python -c "
from src.fine_tuning.lora_finetuner import LoRAFineTuner

finetuner = LoRAFineTuner()
finetuner.apply_lora()
finetuner.train(
    train_file='data/processed/llama_finetuning.jsonl',
    output_dir='models/llama_finetuned'
)
"

Note: Requires 16GB+ GPU VRAM
            """)
        
        elif choice == "7":
            print("\nGoodbye! 👋")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-7.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
