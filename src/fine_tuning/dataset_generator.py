"""
Fine-tuning Dataset Generator

Creates (Question, Context, Answer) tuples from financial documents
for training the LLAMA model with LoRA adaptation.
"""

import json
import random
from typing import List, Dict, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FineTuningDatasetGenerator:
    """Generates training data for fine-tuning financial analyst model."""
    
    # Templates for generating questions based on document content
    QUESTION_TEMPLATES = {
        'revenue': [
            "What was the total revenue for {period}?",
            "How much revenue did the company generate in {period}?",
            "What are the revenue figures for {period}?",
            "Can you provide the total revenues for {period}?",
        ],
        'expenses': [
            "What were the total operating expenses for {period}?",
            "How much did the company spend on operating expenses in {period}?",
            "What is the breakdown of expenses for {period}?",
        ],
        'profit': [
            "What was the net income for {period}?",
            "How much profit did the company make in {period}?",
            "What are the net profit figures for {period}?",
        ],
        'growth': [
            "What was the year-over-year growth rate?",
            "How much did revenue grow compared to the previous period?",
            "What is the growth percentage for {period}?",
        ],
        'risks': [
            "What are the main risks identified by management?",
            "What risk factors does the company mention?",
            "What are the key business risks?",
            "What potential challenges does the company face?",
        ],
        'strategy': [
            "What is the company's business strategy?",
            "How does the company plan to grow?",
            "What are the company's strategic initiatives?",
            "What is management's outlook for the future?",
        ],
        'segments': [
            "What are the company's business segments?",
            "How is revenue broken down by business segment?",
            "Which segment is most profitable?",
        ],
    }
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize dataset generator.
        
        Args:
            output_dir: Directory to save generated datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_key_phrases(self, text: str, num_phrases: int = 10) -> List[str]:
        """
        Extract key financial phrases from text.
        
        Args:
            text: Document text
            num_phrases: Number of phrases to extract
            
        Returns:
            List of key phrases
        """
        # Financial keywords to look for
        keywords = [
            'revenue', 'income', 'profit', 'loss', 'earnings', 'expense',
            'cash flow', 'assets', 'liabilities', 'equity', 'growth',
            'segment', 'geographic', 'market', 'customers', 'products',
            'risk', 'opportunity', 'strategy', 'management', 'results'
        ]
        
        phrases = []
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword in text_lower:
                phrases.append(keyword)
        
        # Shuffle and return limited set
        random.shuffle(phrases)
        return phrases[:num_phrases]
    
    def chunk_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Document text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitter (can be improved with NLTK)
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?':
                if len(current.strip()) > 20:  # Filter very short sentences
                    sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        return sentences
    
    def generate_qa_pair(
        self,
        chunk: str,
        filing_type: str,
        cik: str
    ) -> Dict[str, str]:
        """
        Generate a (Question, Answer, Context) pair from a document chunk.
        
        Args:
            chunk: Document chunk
            filing_type: Type of filing (10-K, 10-Q)
            cik: Company CIK
            
        Returns:
            Dictionary with question, answer, and context
        """
        # Select a random question template
        category = random.choice(list(self.QUESTION_TEMPLATES.keys()))
        question_template = random.choice(self.QUESTION_TEMPLATES[category])
        
        # Extract key phrases to make question more specific
        phrases = self.extract_key_phrases(chunk, num_phrases=3)
        
        # Generate period reference if needed
        periods = ['2024', '2023', '2022', 'the current quarter', 'the fiscal year']
        period = random.choice(periods)
        
        try:
            question = question_template.format(period=period)
        except KeyError:
            question = question_template
        
        # Answer is a relevant excerpt from the chunk
        sentences = self.chunk_into_sentences(chunk)
        if sentences:
            answer = sentences[0]  # Use first relevant sentence
        else:
            answer = chunk[:200]
        
        return {
            'question': question,
            'context': chunk[:1000],  # Limit context size
            'answer': answer,
            'metadata': {
                'cik': cik,
                'filing_type': filing_type,
                'category': category,
                'chunk_length': len(chunk)
            }
        }
    
    def generate_dataset_from_documents(
        self,
        documents: List[Dict],
        samples_per_document: int = 3,
        output_file: str = "finetune_dataset.jsonl"
    ) -> str:
        """
        Generate dataset from document collection.
        
        Args:
            documents: List of document dictionaries with 'content', 'cik', 'filing_type'
            samples_per_document: Number of QA pairs per document
            output_file: Output filename
            
        Returns:
            Path to generated dataset file
        """
        dataset = []
        
        for doc in documents:
            content = doc.get('content', '')
            cik = doc.get('cik', 'unknown')
            filing_type = doc.get('filing_type', 'unknown')
            
            if not content or len(content) < 100:
                logger.warning(f"Skipping document with insufficient content")
                continue
            
            # Split content into chunks
            chunks = []
            chunk_size = 2000
            for i in range(0, len(content), chunk_size):
                chunks.append(content[i:i+chunk_size])
            
            # Generate QA pairs for this document
            for _ in range(samples_per_document):
                if not chunks:
                    continue
                chunk = random.choice(chunks)
                
                try:
                    qa_pair = self.generate_qa_pair(chunk, filing_type, cik)
                    dataset.append(qa_pair)
                except Exception as e:
                    logger.warning(f"Error generating QA pair: {e}")
                    continue
        
        # Save dataset
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Generated dataset with {len(dataset)} examples")
        logger.info(f"Saved to {output_path}")
        
        return str(output_path)
    
    def load_dataset(self, file_path: str) -> List[Dict]:
        """
        Load dataset from JSONL file.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            List of dataset items
        """
        dataset = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        
        return dataset
    
    def create_training_split(
        self,
        dataset_file: str,
        train_ratio: float = 0.8,
        output_prefix: str = "train"
    ) -> Tuple[str, str]:
        """
        Split dataset into train/test sets.
        
        Args:
            dataset_file: Path to dataset file
            train_ratio: Ratio of training samples
            output_prefix: Prefix for output files
            
        Returns:
            Tuple of (train_file, test_file) paths
        """
        dataset = self.load_dataset(dataset_file)
        
        # Shuffle
        random.shuffle(dataset)
        
        # Split
        split_idx = int(len(dataset) * train_ratio)
        train_data = dataset[:split_idx]
        test_data = dataset[split_idx:]
        
        # Save
        train_file = self.output_dir / f"{output_prefix}_train.jsonl"
        test_file = self.output_dir / f"{output_prefix}_test.jsonl"
        
        with open(train_file, 'w') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        
        with open(test_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Split dataset: {len(train_data)} train, {len(test_data)} test")
        
        return str(train_file), str(test_file)
    
    def format_for_llama_finetuning(
        self,
        dataset_file: str,
        output_file: str = "llama_finetuning.jsonl"
    ) -> str:
        """
        Format dataset for LLaMA fine-tuning with LoRA.
        
        LLAMA expects format:
        {
            "text": "<s>[INST] Question about financial data [/INST] Answer based on context </s>"
        }
        
        Args:
            dataset_file: Path to dataset file
            output_file: Output filename
            
        Returns:
            Path to formatted dataset
        """
        dataset = self.load_dataset(dataset_file)
        formatted = []
        
        for item in dataset:
            question = item.get('question', '')
            context = item.get('context', '')
            answer = item.get('answer', '')
            
            # Format for LLaMA chat template
            text = f"<s>[INST] Based on the following financial information:\n{context}\n\nQuestion: {question} [/INST] {answer} </s>"
            
            formatted.append({
                'text': text,
                'metadata': item.get('metadata', {})
            })
        
        # Save
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            for item in formatted:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Formatted {len(formatted)} examples for LLaMA fine-tuning")
        logger.info(f"Saved to {output_path}")
        
        return str(output_path)


def generate_sample_dataset() -> str:
    """
    Generate a sample dataset for demonstration.
    
    Returns:
        Path to generated dataset
    """
    generator = FineTuningDatasetGenerator()
    
    # Sample documents (in a real scenario, these would come from SEC scraper)
    sample_documents = [
        {
            'content': """Apple Inc. reported total revenue of $383.3 billion for fiscal year 2023, 
            an increase from $394.3 billion in 2022. The company's operating expenses reached $65 billion 
            in 2023. Net income for the period was $96.9 billion. The company operates in five main segments: 
            iPhone, Services, Mac, iPad, and Wearables. The iPhone segment generated 52% of total revenue. 
            Management identified risks related to supply chain disruptions and regulatory challenges in key markets.""",
            'cik': '0000320193',
            'filing_type': '10-K'
        },
        {
            'content': """Microsoft Corporation achieved revenue of $52.9 billion in Q4 2023, 
            representing a 16% increase year-over-year. Operating income grew to $22.3 billion. 
            The company's cloud computing segment, Azure, showed particularly strong growth at 29% YoY. 
            Management highlighted artificial intelligence investments as a key strategic initiative. 
            Geographic diversification provides resilience, with 49% of revenue from international markets.""",
            'cik': '0000789019',
            'filing_type': '10-Q'
        },
        {
            'content': """Tesla Inc. generated revenue of $81.5 billion in 2023, up 19% from 2022. 
            Automotive revenue comprised 82% of total revenue, while energy storage contributed 11%. 
            The company expanded production capacity globally with new facilities in Germany and Texas. 
            Operating challenges include supply chain constraints and intense market competition. 
            The company maintains a strong balance sheet with $29.1 billion in cash.""",
            'cik': '0001652044',
            'filing_type': '10-K'
        }
    ]
    
    # Generate dataset
    dataset_file = generator.generate_dataset_from_documents(
        sample_documents,
        samples_per_document=5
    )
    
    return dataset_file


if __name__ == "__main__":
    # Generate sample dataset
    dataset_file = generate_sample_dataset()
    
    generator = FineTuningDatasetGenerator()
    
    # Create train/test split
    train_file, test_file = generator.create_training_split(dataset_file)
    
    # Format for LLaMA
    llama_file = generator.format_for_llama_finetuning(dataset_file)
    
    print(f"\nDataset generation complete!")
    print(f"Dataset file: {dataset_file}")
    print(f"Train file: {train_file}")
    print(f"Test file: {test_file}")
    print(f"LLaMA format file: {llama_file}")
