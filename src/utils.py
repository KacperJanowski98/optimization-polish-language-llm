"""
Utility functions for Polish language model evaluation.

This module provides helper functions used across the project including:
1. Memory optimization utilities
2. Text formatting and cleaning tools
3. File handling and results management
4. Environment setup checks
"""

import os
import json
import torch
import logging
import gc
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_environment_setup():
    """
    Check that the environment is properly set up.
    
    Returns:
        bool: True if environment is properly set up, False otherwise
    """
    checks_passed = True
    
    # Check for HF_TOKEN
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment variables. Add it to your .env file.")
        checks_passed = False
    
    # Check for GPU
    if torch.cuda.is_available():
        logger.info(f"Found GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        
        # Get memory info
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        
        logger.info(f"Total GPU memory: {memory_total:.2f} GB")
        logger.info(f"Reserved GPU memory: {memory_reserved:.2f} GB")
        logger.info(f"Allocated GPU memory: {memory_allocated:.2f} GB")
        logger.info(f"Available GPU memory: {memory_total - memory_reserved:.2f} GB")
    else:
        logger.warning("No GPU found. Models will run on CPU, which may be very slow.")
        checks_passed = False
    
    # Check for required directories
    required_dirs = ['results']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            logger.warning(f"Directory {dir_name} not found. Creating it.")
            os.makedirs(dir_name, exist_ok=True)
    
    return checks_passed


def optimize_memory():
    """
    Optimize memory usage by clearing cache and running garbage collector.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Memory optimized.")


def calculate_gpu_memory_requirement(model_size_b: float) -> Dict[str, float]:
    """
    Calculate approximate GPU memory requirements for a model.
    
    Args:
        model_size_b: Model size in billions of parameters
        
    Returns:
        Dictionary with memory requirements for different precisions
    """
    # Approximate memory requirements
    # FP32: 4 bytes per parameter
    # FP16: 2 bytes per parameter
    # INT8: 1 byte per parameter
    # INT4: 0.5 bytes per parameter
    
    param_count = model_size_b * 1e9
    
    memory_requirements = {
        "fp32": (param_count * 4) / 1e9,  # GB
        "fp16": (param_count * 2) / 1e9,  # GB
        "int8": (param_count * 1) / 1e9,  # GB
        "int4": (param_count * 0.5) / 1e9,  # GB
    }
    
    return memory_requirements


def get_available_gpu_memory() -> float:
    """
    Get available GPU memory in GB.
    
    Returns:
        Available memory in GB, or 0 if no GPU is available
    """
    if not torch.cuda.is_available():
        return 0
    
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    memory_reserved = torch.cuda.memory_reserved(0) / 1e9
    
    return memory_total - memory_reserved


def format_polish_text(text: str) -> str:
    """
    Apply Polish-specific text formatting rules.
    
    Args:
        text: Input text
        
    Returns:
        Formatted text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Polish special characters handling
    special_chars = {
        'a': 'ą', 'c': 'ć', 'e': 'ę', 'l': 'ł', 'n': 'ń',
        'o': 'ó', 's': 'ś', 'z': 'ż', 'x': 'ź'
    }
    
    # Fix common Polish text issues (experimental - may require refinement)
    # This handles cases where someone typed without Polish diacritical marks
    for char, pl_char in special_chars.items():
        # Only replace in certain contexts
        text = text.replace(f" {char} ", f" {pl_char} ")
    
    return text


def plot_benchmark_comparison(results_df: pd.DataFrame, metric: str = 'f1', save_path: Optional[str] = None):
    """
    Create a benchmark comparison plot.
    
    Args:
        results_df: DataFrame with benchmark results
        metric: Metric to plot ('f1', 'accuracy', etc.)
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar plot
    ax = sns.barplot(x='dataset', y=metric, hue='model', data=results_df)
    
    # Customize
    plt.title(f'Model Performance Comparison ({metric.upper()})', fontsize=16)
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    
    plt.show()


def load_results(results_file: str) -> Dict[str, Any]:
    """
    Load evaluation results from a JSON file.
    
    Args:
        results_file: Path to the results file
        
    Returns:
        Dictionary with loaded results
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def create_results_dataframe(results_dir: str) -> pd.DataFrame:
    """
    Create a DataFrame from all results files in a directory.
    
    Args:
        results_dir: Directory containing result JSON files
        
    Returns:
        DataFrame with consolidated results
    """
    all_data = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and 'complete' not in filename:
            file_path = os.path.join(results_dir, filename)
            try:
                results = load_results(file_path)
                
                # Extract key metrics
                row = {
                    'model': results.get('model', 'unknown'),
                    'dataset': results.get('dataset', 'unknown'),
                    'task': results.get('task', 'unknown'),
                    'accuracy': results.get('accuracy', None),
                    'f1': results.get('f1', None),
                    'precision': results.get('precision', None),
                    'samples': results.get('samples_evaluated', 0),
                    'time': results.get('evaluation_time', 0),
                    'file': filename
                }
                
                all_data.append(row)
                
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
    
    return pd.DataFrame(all_data)


def check_gpu_compatibility(model_size_b: float) -> str:
    """
    Check GPU compatibility with a model of a given size.
    
    Args:
        model_size_b: Model size in billions of parameters
        
    Returns:
        String with recommendation on how to load the model
    """
    if not torch.cuda.is_available():
        return "No GPU available. Loading in CPU mode (very slow)."
    
    available_memory = get_available_gpu_memory()
    memory_requirements = calculate_gpu_memory_requirement(model_size_b)
    
    if available_memory >= memory_requirements["fp16"]:
        return f"Can load in FP16 precision. Required: {memory_requirements['fp16']:.2f} GB, Available: {available_memory:.2f} GB"
    elif available_memory >= memory_requirements["int8"]:
        return f"Recommend loading in 8-bit precision. Required: {memory_requirements['int8']:.2f} GB, Available: {available_memory:.2f} GB"
    elif available_memory >= memory_requirements["int4"]:
        return f"Recommend loading in 4-bit precision. Required: {memory_requirements['int4']:.2f} GB, Available: {available_memory:.2f} GB"
    else:
        return f"Not enough GPU memory even for 4-bit quantization. Consider using CPU mode or a GPU with more memory. Required: {memory_requirements['int4']:.2f} GB, Available: {available_memory:.2f} GB"
