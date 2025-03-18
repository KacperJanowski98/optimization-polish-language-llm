"""
Dataset preparation for Polish language model evaluation.

This module provides functionality to load and prepare datasets for evaluating
Polish language models. It supports loading datasets from the KLEJ benchmark,
including DYK, PolEmo2.0, PSC, and CDSC-E datasets.

The main functionality allows for sampling a configurable number of examples
from each dataset to create evaluation sets of manageable size.
"""

import logging
from typing import Dict, List, Optional, Any
import random

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PolishDatasetLoader:
    """
    Loader for Polish language datasets from the KLEJ benchmark.
    
    This class provides methods to load and sample from various Polish language
    datasets for NLP tasks such as sentiment analysis, entailment, and QA.
    """
    
    SUPPORTED_DATASETS = {
        'dyk': {
            'hf_path': 'allegro/klej-dyk',
            'task_type': 'binary_classification',
            'input_cols': ['question', 'answer'],
            'target_col': 'target',
            'metrics': ['f1'],
            'description': 'Question-answer pairs from Polish Wikipedia'
        },
        'polemo2': {
            'hf_path': 'allegro/klej-polemo2-in',
            'task_type': 'multi_classification',
            'input_cols': ['sentence'],
            'target_col': 'target',
            'metrics': ['accuracy'],
            'description': 'Polish sentiment analysis for reviews'
        },
        'psc': {
            'hf_path': 'allegro/klej-psc',
            'task_type': 'binary_classification',
            'input_cols': ['extract_text', 'summary_text'],
            'target_col': 'label',
            'metrics': ['f1'],
            'description': 'Polish Summaries Corpus for text similarity'
        },
        'cdsc': {
            'hf_path': 'allegro/klej-cdsc-e',
            'task_type': 'multi_classification',
            'input_cols': ['sentence_A', 'sentence_B'],
            'target_col': 'entailment_judgment',
            'metrics': ['accuracy'],
            'description': 'Polish sentence pairs for entailment'
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache the downloaded datasets
            seed: Random seed for reproducibility
        """
        self.cache_dir = cache_dir
        self.seed = seed
        self.datasets: Dict[str, Dict[str, Any]] = {}
        random.seed(seed)
        logger.info("PolishDatasetLoader initialized with seed %d", seed)
    
    def load_dataset(self, dataset_name: str) -> DatasetDict:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load (one of the supported datasets)
            
        Returns:
            The loaded dataset
            
        Raises:
            ValueError: If the dataset name is not supported
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not supported. "
                f"Choose from: {', '.join(self.SUPPORTED_DATASETS.keys())}"
            )
        
        dataset_info = self.SUPPORTED_DATASETS[dataset_name]
        logger.info(f"Loading dataset: {dataset_name} ({dataset_info['hf_path']})")
        
        try:
            dataset = load_dataset(
                dataset_info['hf_path'],
                cache_dir=self.cache_dir
            )
            
            # Validate columns for each split
            validated_info = dataset_info.copy()
            first_split = list(dataset.keys())[0]
            
            # Check if the columns specified in the dataset info exist in the dataset
            actual_columns = list(dataset[first_split].features.keys())
            logger.info(f"Columns found in {dataset_name}: {', '.join(actual_columns)}")
            
            # Validate and potentially update column names
            validated_info = validate_dataset_columns(dataset[first_split], dataset_info)
            if validated_info != dataset_info:
                logger.warning(f"Column names were adjusted for dataset {dataset_name}")
            
            # Encode target column for classification tasks
            if 'classification' in validated_info['task_type']:
                try:
                    dataset = dataset.class_encode_column(validated_info['target_col'])
                except Exception as e:
                    logger.warning(f"Could not encode target column: {str(e)}")
                
            self.datasets[dataset_name] = {
                'data': dataset,
                'info': validated_info
            }
            
            logger.info(f"Successfully loaded {dataset_name}")
            return dataset
        
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise
    
    def load_all_datasets(self) -> Dict[str, DatasetDict]:
        """
        Load all supported datasets.
        
        Returns:
            Dictionary of all loaded datasets
        """
        for dataset_name in tqdm(self.SUPPORTED_DATASETS.keys(), desc="Loading datasets"):
            self.load_dataset(dataset_name)
        
        return {name: info['data'] for name, info in self.datasets.items()}
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
            
        Raises:
            ValueError: If the dataset name is not supported
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset {dataset_name} not supported. "
                f"Choose from: {', '.join(self.SUPPORTED_DATASETS.keys())}"
            )
        
        return self.SUPPORTED_DATASETS[dataset_name]
    
    def sample_dataset(
        self, 
        dataset_name: str, 
        split: str = 'test',
        n_samples: int = 100,
        balanced: bool = True
    ) -> Dataset:
        """
        Sample a specified number of examples from a dataset.
        
        Args:
            dataset_name: Name of the dataset to sample from
            split: Dataset split to sample from (train, validation, test)
            n_samples: Number of examples to sample
            balanced: Whether to ensure balanced samples for classification tasks
            
        Returns:
            Sampled dataset
            
        Raises:
            ValueError: If dataset not loaded or split not available
        """
        # Make sure the dataset is loaded
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name)
        
        dataset_dict = self.datasets[dataset_name]['data']
        dataset_info = self.datasets[dataset_name]['info']
        
        if split not in dataset_dict:
            available_splits = list(dataset_dict.keys())
            raise ValueError(
                f"Split '{split}' not available for dataset {dataset_name}. "
                f"Available splits: {', '.join(available_splits)}"
            )
        
        dataset = dataset_dict[split]
        total_examples = len(dataset)
        
        if n_samples >= total_examples:
            logger.warning(
                f"Requested {n_samples} samples, but dataset only has {total_examples}. "
                f"Returning the entire dataset."
            )
            return dataset
        
        # For classification tasks, try to maintain class distribution
        if balanced and 'classification' in dataset_info['task_type']:
            target_col = dataset_info['target_col']
            
            # Get class distribution
            labels = dataset[target_col]
            unique_labels = set(labels)
            
            # Calculate samples per class
            samples_per_class = n_samples // len(unique_labels)
            
            # Create indices for balanced sampling
            indices = []
            
            for label in unique_labels:
                label_indices = [
                    i for i, l in enumerate(labels) if l == label
                ]
                
                # If we don't have enough samples of this class, take all available
                if len(label_indices) <= samples_per_class:
                    indices.extend(label_indices)
                else:
                    indices.extend(random.sample(label_indices, samples_per_class))
            
            # Fill remaining slots randomly
            remaining = n_samples - len(indices)
            if remaining > 0:
                all_indices = set(range(total_examples))
                remaining_indices = list(all_indices - set(indices))
                indices.extend(random.sample(remaining_indices, min(remaining, len(remaining_indices))))
            
            # Shuffle indices for good measure
            random.shuffle(indices)
            
            # Select the examples using the indices
            sampled_dataset = dataset.select(indices[:n_samples])
        
        else:
            # Simple random sampling
            indices = random.sample(range(total_examples), n_samples)
            sampled_dataset = dataset.select(indices)
        
        logger.info(
            f"Sampled {len(sampled_dataset)} examples from {dataset_name} ({split} split)"
        )
        
        return sampled_dataset
    
    def create_evaluation_dataset(
        self, 
        samples_per_dataset: Dict[str, int],
        split: str = 'test'
    ) -> Dict[str, Dataset]:
        """
        Create an evaluation dataset by sampling from multiple datasets.
        
        Args:
            samples_per_dataset: Dictionary mapping dataset names to number of samples
            split: Dataset split to sample from (train, validation, test)
            
        Returns:
            Dictionary of sampled datasets
            
        Raises:
            ValueError: If any dataset name is not supported
        """
        evaluation_datasets = {}
        
        for dataset_name, n_samples in samples_per_dataset.items():
            if dataset_name not in self.SUPPORTED_DATASETS:
                raise ValueError(
                    f"Dataset {dataset_name} not supported. "
                    f"Choose from: {', '.join(self.SUPPORTED_DATASETS.keys())}"
                )
            
            evaluation_datasets[dataset_name] = self.sample_dataset(
                dataset_name=dataset_name,
                split=split,
                n_samples=n_samples
            )
        
        return evaluation_datasets
    
    def dataset_to_pandas(self, dataset: Dataset) -> pd.DataFrame:
        """
        Convert a Hugging Face dataset to a pandas DataFrame.
        
        Args:
            dataset: Hugging Face dataset
            
        Returns:
            Pandas DataFrame representation of the dataset
        """
        return pd.DataFrame(dataset)
    
    def get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get statistics for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset statistics
            
        Raises:
            ValueError: If the dataset is not loaded
        """
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name)
        
        dataset_dict = self.datasets[dataset_name]['data']
        dataset_info = self.datasets[dataset_name]['info']
        
        stats = {
            'name': dataset_name,
            'path': dataset_info['hf_path'],
            'task_type': dataset_info['task_type'],
            'splits': {},
            'description': dataset_info['description']
        }
        
        for split_name, split_data in dataset_dict.items():
            split_stats = {
                'examples': len(split_data),
            }
            
            # Add class distribution for classification tasks
            if 'classification' in dataset_info['task_type']:
                target_col = dataset_info['target_col']
                labels = split_data[target_col]
                
                # Count occurrences of each label
                label_counts = {}
                for label in labels:
                    if label not in label_counts:
                        label_counts[label] = 0
                    label_counts[label] += 1
                
                split_stats['class_distribution'] = label_counts
            
            stats['splits'][split_name] = split_stats
        
        return stats


# Helper functions for more specific tasks

def prepare_datasets_for_evaluation(
    samples_per_dataset: Dict[str, int] = None,
    cache_dir: Optional[str] = None,
    split: str = 'test',
    seed: int = 42
) -> Dict[str, Dataset]:
    """
    Prepare datasets for model evaluation with default settings.
    
    Args:
        samples_per_dataset: Dictionary mapping dataset names to number of samples.
                             If None, uses default values.
        cache_dir: Directory to cache the downloaded datasets
        split: Dataset split to use
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of sampled datasets ready for evaluation
    """
    if samples_per_dataset is None:
        # Default to 100 samples per dataset if not specified
        samples_per_dataset = {
            'dyk': 100,
            'polemo2': 100,
            'psc': 100,
            'cdsc': 100
        }
    
    loader = PolishDatasetLoader(cache_dir=cache_dir, seed=seed)
    return loader.create_evaluation_dataset(
        samples_per_dataset=samples_per_dataset,
        split=split
    )


def validate_dataset_columns(dataset: Dataset, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that all specified columns exist in the dataset.
    
    Args:
        dataset: Dataset to validate
        dataset_info: Information about the dataset with column specifications
        
    Returns:
        Validated dataset_info with corrected column names if needed
        
    Raises:
        ValueError: If required columns cannot be found or matched
    """
    # Create a copy of the dataset info to avoid modifying the original
    validated_info = dataset_info.copy()
    
    # Get the actual column names in the dataset
    actual_columns = list(dataset.features.keys())
    
    # Check input columns
    for i, col in enumerate(dataset_info['input_cols']):
        if col not in actual_columns:
            # Try to find alternative columns that might be semantically similar
            alternatives = [
                c for c in actual_columns 
                if c.lower() in ['text', 'sentence', 'content', 'input', 'question', 'premise', 'hypothesis']
                or col.lower() in c.lower()
            ]
            
            if alternatives:
                logger.warning(
                    f"Column '{col}' not found in dataset. Using '{alternatives[0]}' instead."
                )
                validated_info['input_cols'][i] = alternatives[0]
            else:
                raise ValueError(
                    f"Column '{col}' not found in dataset and no suitable alternative found. "
                    f"Available columns: {', '.join(actual_columns)}"
                )
    
    # Check target column
    target_col = dataset_info['target_col']
    if target_col not in actual_columns:
        # Try to find alternative target columns
        alternatives = [
            c for c in actual_columns 
            if c.lower() in ['target', 'label', 'class', 'output', 'result', 'answer']
            or target_col.lower() in c.lower()
        ]
        
        if alternatives:
            logger.warning(
                f"Target column '{target_col}' not found in dataset. Using '{alternatives[0]}' instead."
            )
            validated_info['target_col'] = alternatives[0]
        else:
            raise ValueError(
                f"Target column '{target_col}' not found in dataset and no suitable alternative found. "
                f"Available columns: {', '.join(actual_columns)}"
            )
    
    return validated_info


def get_dataset_examples(
    dataset: Dataset,
    dataset_info: Dict[str, Any],
    n_examples: int = 5
) -> List[Dict[str, Any]]:
    """
    Get a few examples from a dataset for inspection.
    
    Args:
        dataset: Dataset to get examples from
        dataset_info: Information about the dataset
        n_examples: Number of examples to return
        
    Returns:
        List of examples
    """
    # Validate column names to avoid KeyErrors
    validated_info = validate_dataset_columns(dataset, dataset_info)
    
    examples = []
    
    for i in range(min(n_examples, len(dataset))):
        example = {}
        
        # Add input fields
        for col in validated_info['input_cols']:
            example[col] = dataset[i][col]
        
        # Add target field
        example[validated_info['target_col']] = dataset[i][validated_info['target_col']]
        
        examples.append(example)
    
    return examples
