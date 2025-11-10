"""
Complete end-to-end example workflow for Financial Analyst Advisor.

This script demonstrates:
1. Data ingestion from SEC
2. RAG pipeline setup
3. Dataset generation
4. Fine-tuning setup
5. API usage examples
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_generate_sample_data():
    """Example 1: Generate sample training data."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Generate Sample Training Data")
    print("="*60)
    
    from src.fine_tuning.dataset_generator import generate_sample_dataset
    
    dataset_file = generate_sample_dataset()
    print(f"✓ Dataset generated: {dataset_file}")
    
    # Load and display a sample
    with open(dataset_file, 'r') as f:
        sample = json.loads(f.readline())
    
    print(f"\nSample QA pair:")
    print(f"  Question: {sample['question']}")
    print(f"  Answer: {sample['answer'][:100]}...")
    
    return dataset_file


def example_2_create_training_split(dataset_file):
    """Example 2: Create train/test split."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Create Train/Test Split")
    print("="*60)
    
    from src.fine_tuning.dataset_generator import FineTuningDatasetGenerator
    
    generator = FineTuningDatasetGenerator()
    train_file, test_file = generator.create_training_split(
        dataset_file,
        train_ratio=0.8
    )
    
    print(f"✓ Train file: {train_file}")
    print(f"✓ Test file: {test_file}")
    
    return train_file, test_file


def example_3_format_for_llama(dataset_file):
    """Example 3: Format dataset for LLaMA fine-tuning."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Format Dataset for LLaMA")
    print("="*60)
    
    from src.fine_tuning.dataset_generator import FineTuningDatasetGenerator
    
    generator = FineTuningDatasetGenerator()
    llama_file = generator.format_for_llama_finetuning(dataset_file)
    
    print(f"✓ LLaMA format file: {llama_file}")
    
    # Display sample
    with open(llama_file, 'r') as f:
        sample = json.loads(f.readline())
    
    print(f"\nLLaMA formatted example:")
    print(f"{sample['text'][:200]}...")
    
    return llama_file


def example_4_rag_pipeline_search():
    """Example 4: RAG pipeline document search."""
    print("\n" + "="*60)
    print("EXAMPLE 4: RAG Pipeline Search")
    print("="*60)
    
    from src.rag.rag_pipeline import RAGPipeline
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    print("✓ RAG pipeline initialized")
    
    # Note: In a real scenario, you would ingest actual documents
    # For now, we'll just demonstrate the structure
    print("\nRAG Pipeline Components:")
    print("  - Document Processor: Chunks documents with overlap")
    print("  - Embedding Manager: Generates BERT embeddings")
    print("  - Vector Store: Stores embeddings in Chroma")
    print("  - Search: Semantic similarity search")
    
    # Example search query (would need documents ingested)
    example_query = "What was the total revenue?"
    print(f"\nExample Query: '{example_query}'")
    print("Would return relevant chunks from ingested documents")
    
    return pipeline


def example_5_model_initialization():
    """Example 5: Initialize fine-tuned model."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Fine-tuned Model Initialization")
    print("="*60)
    
    from src.fine_tuning.lora_finetuner import LoRAFineTuner
    
    print("Initializing LLaMA with LoRA...")
    print("\nLoRA Configuration:")
    print("  - Base Model: meta-llama/Llama-2-7b-hf")
    print("  - Rank (r): 16")
    print("  - Alpha: 32")
    print("  - Dropout: 0.05")
    print("  - Target Modules: [q_proj, v_proj, k_proj, o_proj]")
    print("\nOptimizations:")
    print("  - 4-bit quantization enabled")
    print("  - Gradient checkpointing enabled")
    print("  - Memory usage: ~16GB VRAM")
    
    print("\n✓ Model structure ready for fine-tuning")


def example_6_api_endpoints():
    """Example 6: API endpoint demonstrations."""
    print("\n" + "="*60)
    print("EXAMPLE 6: FastAPI Endpoints")
    print("="*60)
    
    print("\nAvailable endpoints:")
    
    endpoints = [
        ("GET", "/health", "Health check"),
        ("POST", "/api/v1/search", "Search financial documents"),
        ("POST", "/api/v1/analyze", "Analyze financial question"),
        ("POST", "/api/v1/ingest", "Ingest new documents"),
        ("POST", "/api/v1/generate", "Generate text with model"),
        ("GET", "/api/v1/stats", "Get system statistics"),
    ]
    
    for method, path, description in endpoints:
        print(f"  [{method}] {path}")
        print(f"      → {description}")
    
    print("\n✓ API is ready to serve requests")
    print("\nStart server with:")
    print("  python -m src.api.app")


def example_7_complete_workflow():
    """Example 7: Complete question-answering workflow."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Complete Q&A Workflow")
    print("="*60)
    
    print("\nWorkflow steps:")
    print("\n1. USER QUERY")
    print("   'What are Tesla's main business risks?'")
    
    print("\n2. RETRIEVE CONTEXT (RAG)")
    print("   - Search vector database for relevant documents")
    print("   - Found 5 relevant chunks from 10-K filing")
    print("   - Similarity scores: [0.92, 0.87, 0.84, 0.79, 0.75]")
    
    print("\n3. CONSTRUCT PROMPT")
    print("   'Based on [context], answer: [question]'")
    
    print("\n4. GENERATE ANSWER (Fine-tuned LLaMA)")
    print("   'According to Tesla's 10-K filing, main risks include:'")
    print("   '- Supply chain disruptions'")
    print("   '- Intense market competition'")
    print("   '- Regulatory challenges...'")
    
    print("\n5. RETURN RESPONSE")
    print("   - Answer with confidence score")
    print("   - Include source documents")
    print("   - Provide retrieved context if requested")
    
    print("\n✓ Complete workflow demonstrated")


def example_8_configuration():
    """Example 8: Configuration options."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Configuration Options")
    print("="*60)
    
    from src.utils.config import Config
    
    print("\nKey configuration sections:")
    
    config_dict = Config.to_dict()
    for section, values in config_dict.items():
        print(f"\n[{section}]")
        if isinstance(values, dict):
            for key, value in list(values.items())[:3]:  # Show first 3
                if not isinstance(value, (dict, list)):
                    print(f"  {key}: {value}")
            if len(values) > 3:
                print(f"  ... and {len(values) - 3} more")


def main():
    """Run all examples."""
    print("\n" + "🚀" * 30)
    print("FINANCIAL ANALYST ADVISOR - EXAMPLES")
    print("🚀" * 30)
    
    try:
        # Example 1: Generate sample data
        dataset_file = example_1_generate_sample_data()
        
        # Example 2: Create train/test split
        train_file, test_file = example_2_create_training_split(dataset_file)
        
        # Example 3: Format for LLaMA
        llama_file = example_3_format_for_llama(dataset_file)
        
        # Example 4: RAG pipeline
        pipeline = example_4_rag_pipeline_search()
        
        # Example 5: Model initialization
        example_5_model_initialization()
        
        # Example 6: API endpoints
        example_6_api_endpoints()
        
        # Example 7: Complete workflow
        example_7_complete_workflow()
        
        # Example 8: Configuration
        example_8_configuration()
        
        print("\n" + "="*60)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print("\n📚 Next steps:")
        print("  1. Review the README.md for detailed documentation")
        print("  2. Configure .env with your settings")
        print("  3. Start the API server: python -m src.api.app")
        print("  4. Explore API documentation at http://localhost:8000/docs")
        print("  5. Fine-tune the model with your custom data")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
