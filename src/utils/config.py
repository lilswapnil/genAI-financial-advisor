"""Configuration management for Financial Analyst Advisor."""

import os
from typing import Dict, Any
from pathlib import Path
import json


class Config:
    """Application configuration."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw_reports"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    VECTOR_DB_DIR = DATA_DIR / "vector_db"
    MODELS_DIR = BASE_DIR / "models"
    
    # Create directories if they don't exist
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ==================== SEC Scraper Config ====================
    SEC_SCRAPER = {
        'output_dir': str(RAW_DATA_DIR),
        'rate_limit_delay': 0.1,  # seconds between requests
        'timeout': 15,  # request timeout
    }
    
    # ==================== RAG Pipeline Config ====================
    RAG_PIPELINE = {
        'raw_data_dir': str(RAW_DATA_DIR),
        'vector_db_dir': str(VECTOR_DB_DIR),
        'chunk_size': 1024,
        'chunk_overlap': 128,
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'similarity_threshold': 0.3,
    }
    
    # ==================== Fine-tuning Config ====================
    FINETUNING = {
        'base_model': 'meta-llama/Llama-2-7b-hf',
        'output_dir': str(MODELS_DIR / 'llama_finetuned'),
        'dataset_dir': str(PROCESSED_DATA_DIR),
        'lora': {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        },
        'training': {
            'num_epochs': 3,
            'batch_size': 4,
            'gradient_accumulation_steps': 2,
            'learning_rate': 2e-4,
            'max_length': 512,
            'save_steps': 500,
            'eval_steps': 500,
        },
        'quantization': {
            'use_4bit': True,
            'use_gradient_checkpointing': True,
        },
    }
    
    # ==================== API Config ====================
    API = {
        'host': os.getenv('API_HOST', '0.0.0.0'),
        'port': int(os.getenv('API_PORT', 8000)),
        'debug': os.getenv('API_DEBUG', 'false').lower() == 'true',
        'title': 'Financial Analyst Chatbot API',
        'description': 'RAG-powered financial analysis with fine-tuned LLaMA',
        'version': '1.0.0',
    }
    
    # ==================== Generation Config ====================
    GENERATION = {
        'temperature': 0.7,
        'top_p': 0.9,
        'max_new_tokens': 256,
        'do_sample': True,
    }
    
    # ==================== Logging Config ====================
    LOGGING = {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    }
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation path."""
        keys = key.split('.')
        value = vars(cls).get(keys[0])
        
        for k in keys[1:]:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'SEC_SCRAPER': cls.SEC_SCRAPER,
            'RAG_PIPELINE': cls.RAG_PIPELINE,
            'FINETUNING': cls.FINETUNING,
            'API': cls.API,
            'GENERATION': cls.GENERATION,
            'LOGGING': cls.LOGGING,
        }
    
    @classmethod
    def save(cls, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(cls.to_dict(), f, indent=2)


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    FINETUNING = {
        **Config.FINETUNING,
        'training': {
            **Config.FINETUNING['training'],
            'num_epochs': 5,
            'batch_size': 8,
        }
    }


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True


# Select config based on environment
ENV = os.getenv('APP_ENV', 'development').lower()

if ENV == 'production':
    current_config = ProductionConfig
elif ENV == 'testing':
    current_config = TestingConfig
else:
    current_config = DevelopmentConfig

# Export configuration
config = current_config()
