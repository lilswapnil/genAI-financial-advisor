"""LoRA Fine-Tuning for LLaMA Model.

This module implements Low-Rank Adaptation (LoRA) fine-tuning for the LLaMA model.
LoRA is efficient because it only trains a small number of additional parameters
while keeping the base model frozen, reducing memory and computation requirements.
"""

import os
import json
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import logging

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinancialQADataset(Dataset):
    """Dataset for financial Q&A fine-tuning."""
    
    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize dataset.
        
        Args:
            data_file: Path to JSONL file with training data
            tokenizer: Tokenizer instance
            max_length: Maximum token length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.data)} examples from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }


class LoRAFineTuner:
    """Fine-tunes LLaMA model using LoRA."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "auto",
        use_4bit: bool = True,
        use_gradient_checkpointing: bool = True
    ):
        """
        Initialize LoRA fine-tuner.
        
        Args:
            model_name: Model name or path
            device: Device to use ('auto', 'cuda', 'cpu')
            use_4bit: Whether to use 4-bit quantization
            use_gradient_checkpointing: Whether to use gradient checkpointing
        """
        self.model_name = model_name
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.use_4bit = use_4bit
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load base model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if using 4-bit
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info("Model loaded successfully")
    
    def prepare_lora_config(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None
    ) -> LoraConfig:
        """
        Create LoRA configuration.
        
        Args:
            r: Rank of LoRA adaptation
            lora_alpha: Alpha parameter for LoRA
            lora_dropout: Dropout rate for LoRA
            target_modules: Which modules to apply LoRA to
            
        Returns:
            LoRA configuration
        """
        if target_modules is None:
            # Default target modules for LLaMA
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        logger.info(f"LoRA config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        return config
    
    def apply_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        """
        Apply LoRA to the model.
        
        Args:
            r: Rank of LoRA adaptation
            lora_alpha: Alpha parameter for LoRA
            lora_dropout: Dropout rate for LoRA
        """
        lora_config = self.prepare_lora_config(r, lora_alpha, lora_dropout)
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        logger.info(f"Total params: {total_params:,}")
    
    def train(
        self,
        train_file: str,
        eval_file: Optional[str] = None,
        output_dir: str = "models/llama_finetuned",
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 2,
        learning_rate: float = 2e-4,
        save_steps: int = 500,
        eval_steps: int = 500,
        max_length: int = 512,
        use_wandb: bool = False
    ) -> str:
        """
        Fine-tune the model on financial Q&A data.
        
        Args:
            train_file: Path to training data
            eval_file: Path to evaluation data (optional)
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            max_length: Maximum token length
            use_wandb: Whether to use Weights & Biases
            
        Returns:
            Path to saved model
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        train_dataset = FinancialQADataset(train_file, self.tokenizer, max_length)
        
        eval_dataset = None
        if eval_file and os.path.exists(eval_file):
            eval_dataset = FinancialQADataset(eval_file, self.tokenizer, max_length)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=100,
            logging_steps=50,
            save_steps=save_steps,
            eval_steps=eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=True if eval_dataset else False,
            optim="paged_adamw_32bit",
            report_to=["wandb"] if use_wandb else [],
            push_to_hub=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        config_file = Path(output_dir) / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'max_length': max_length,
                'train_loss': train_result.training_loss
            }, f, indent=2)
        
        logger.info("Training complete!")
        return output_dir
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using the fine-tuned model.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def save_pretrained(self, save_dir: str):
        """
        Save the fine-tuned model.
        
        Args:
            save_dir: Directory to save model
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        logger.info(f"Model saved to {save_dir}")
    
    def load_pretrained(self, model_dir: str):
        """
        Load a fine-tuned model.
        
        Args:
            model_dir: Directory containing saved model
        """
        logger.info(f"Loading model from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        logger.info("Model loaded successfully")


def fine_tune_example():
    """Example fine-tuning pipeline."""
    
    # Initialize fine-tuner
    finetuner = LoRAFineTuner(
        model_name="meta-llama/Llama-2-7b-hf",
        use_4bit=True
    )
    
    # Apply LoRA
    finetuner.apply_lora(r=16, lora_alpha=32)
    
    # Check if training data exists
    train_file = "data/processed/llama_finetuning.jsonl"
    if not os.path.exists(train_file):
        logger.warning(f"Training file not found: {train_file}")
        logger.info("Generate training data first using dataset_generator.py")
        return
    
    # Train
    model_dir = finetuner.train(
        train_file=train_file,
        output_dir="models/llama_finetuned",
        num_epochs=1,  # Use 1 epoch for testing
        batch_size=2,
        learning_rate=2e-4,
        max_length=512
    )
    
    logger.info(f"Fine-tuned model saved to {model_dir}")


if __name__ == "__main__":
    # Note: To run this, you need:
    # 1. huggingface-hub login or HF_TOKEN environment variable
    # 2. Generated training data from dataset_generator.py
    # 3. Sufficient GPU memory (recommended: 16GB+)
    
    logger.warning("This script requires significant GPU memory and LLaMA model access")
    logger.warning("Make sure you have:")
    logger.warning("1. Accepted LLaMA license at https://huggingface.co/meta-llama")
    logger.warning("2. Logged in with huggingface-cli: huggingface-cli login")
    logger.warning("3. Generated training data using dataset_generator.py")
