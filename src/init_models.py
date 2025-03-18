"""
Model initialization for Polish language model evaluation.

This module provides functionality to initialize the three language models
for evaluation:
1. Bielik-11B-v2.3-Instruct - A specialized Polish language model
2. Google Gemma-3-4B-IT - A multilingual model
3. Microsoft Phi-4-mini-instruct - A multilingual model

The module handles model-specific requirements, memory management strategies,
and proper configuration for each model.
"""

import os
import logging
from typing import Dict, Tuple, Optional, List, Union
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    Pipeline
)
from dotenv import load_dotenv
from huggingface_hub import login

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define model configurations
MODEL_CONFIGS = {
    'bielik': {
        'model_id': 'speakleash/Bielik-11B-v2.3-Instruct',
        'description': 'Specialized Polish language model (11B parameters)',
        'recommended_quantization': True,
        'context_length': 4096,
    },
    'gemma': {
        'model_id': 'google/gemma-3-4b-it',
        'description': 'Google multilingual model (4B parameters)',
        'recommended_quantization': False,
        'context_length': 8192,
    },
    'phi': {
        'model_id': 'microsoft/Phi-4-mini-instruct',
        'description': 'Microsoft multilingual model',
        'recommended_quantization': False,
        'context_length': 4096,
    }
}


def setup_huggingface_auth():
    """
    Set up Hugging Face authentication using the API token from environment.
    
    Raises:
        ValueError: If the HF_TOKEN is not found in environment variables
    """
    # Load from .env file if it exists
    load_dotenv()
    
    # Get Hugging Face token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found in environment variables. "
            "Please add it to your .env file."
        )
    
    # Login to Hugging Face
    login(token=hf_token)
    logger.info("Successfully authenticated with Hugging Face")


def get_model_info(model_key: str) -> Dict:
    """
    Get information about a specific model.
    
    Args:
        model_key: Key identifier for the model ('bielik', 'gemma', or 'phi')
        
    Returns:
        Dictionary with model information
        
    Raises:
        ValueError: If the model_key is not recognized
    """
    if model_key not in MODEL_CONFIGS:
        valid_keys = list(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Invalid model key: {model_key}. "
            f"Valid options are: {', '.join(valid_keys)}"
        )
    
    return MODEL_CONFIGS[model_key]


def initialize_model(
    model_key: str,
    device: str = "auto",
    load_in_8bit: Optional[bool] = None,
    load_in_4bit: bool = False,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    trust_remote_code: bool = True,
    offload_to_cpu: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Initialize a specific model and its tokenizer.
    
    Args:
        model_key: Key identifier for the model ('bielik', 'gemma', or 'phi')
        device: Device to load the model on ('cpu', 'cuda', 'auto')
        load_in_8bit: Whether to use 8-bit quantization (overrides recommended setting if provided)
        load_in_4bit: Whether to use 4-bit quantization (only used if load_in_8bit is False)
        cache_dir: Directory to cache the downloaded model
        revision: Model revision to load
        trust_remote_code: Whether to trust remote code
        offload_to_cpu: Whether to offload some model layers to CPU to save GPU memory
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        ValueError: If the model_key is not recognized
    """
    if model_key not in MODEL_CONFIGS:
        valid_keys = list(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Invalid model key: {model_key}. "
            f"Valid options are: {', '.join(valid_keys)}"
        )
    
    model_config = MODEL_CONFIGS[model_key]
    model_id = model_config['model_id']
    
    logger.info(f"Initializing model: {model_id}")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Determine quantization
    use_8bit = load_in_8bit if load_in_8bit is not None else model_config['recommended_quantization']
    
    # Initialize tokenizer with padding token if needed
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        revision=revision,
        trust_remote_code=trust_remote_code
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model initialization parameters
    model_kwargs = {
        "cache_dir": cache_dir,
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    
    # Flag to track if model is using distributed computation
    is_distributed_model = False
    
    # Setup device mapping for offloading
    if offload_to_cpu and device == "cuda":
        logger.info(f"Setting up CPU offloading for {model_id}")
        # Create a device map that offloads some layers to CPU
        model_kwargs["device_map"] = "auto"
        model_kwargs["offload_folder"] = "offload_folder"
        model_kwargs["offload_state_dict"] = True
        is_distributed_model = True
    else:
        model_kwargs["device_map"] = device if device != "cpu" else None
    
    # Add quantization if needed
    if device == "cuda":
        if use_8bit:
            logger.info(f"Loading {model_id} in 8-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            logger.info(f"Loading {model_id} in 4-bit quantization")
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    # Store the distributed flag as an attribute of the model for later use
    model._is_distributed = is_distributed_model or model_kwargs.get("device_map") == "auto"
    
    logger.info(f"Successfully loaded {model_id}")
    
    return model, tokenizer


def initialize_models(
    model_keys: Optional[List[str]] = None,
    device: str = "auto",
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    cache_dir: Optional[str] = None,
    offload_to_cpu: bool = False
) -> Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """
    Initialize multiple models for evaluation.
    
    Args:
        model_keys: List of model keys to initialize (defaults to all models)
        device: Device to load models on ('cpu', 'cuda', 'auto') 
        load_in_8bit: Whether to use 8-bit quantization for large models
        load_in_4bit: Whether to use 4-bit quantization (if 8-bit is not used)
        cache_dir: Directory to cache the downloaded models
        offload_to_cpu: Whether to offload some model layers to CPU to save GPU memory
        
    Returns:
        Dictionary of model name to (model, tokenizer) pairs
    """
    # Set up authentication
    setup_huggingface_auth()
    
    # Default to all models if none specified
    if model_keys is None:
        model_keys = list(MODEL_CONFIGS.keys())
    
    # Check if keys are valid
    for key in model_keys:
        if key not in MODEL_CONFIGS:
            valid_keys = list(MODEL_CONFIGS.keys())
            raise ValueError(
                f"Invalid model key: {key}. "
                f"Valid options are: {', '.join(valid_keys)}"
            )
    
    # Load models
    models_dict = {}
    
    for key in model_keys:
        model_config = MODEL_CONFIGS[key]
        logger.info(
            f"Loading {key} model: {model_config['model_id']} - {model_config['description']}"
        )
        
        model, tokenizer = initialize_model(
            model_key=key,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            cache_dir=cache_dir,
            offload_to_cpu=offload_to_cpu
        )
        
        models_dict[key] = (model, tokenizer)
    
    return models_dict


def create_model_pipeline(
    model_key: str,
    task: str = "text-generation",
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    device: str = "auto",
    **kwargs
) -> Pipeline:
    """
    Create a pipeline for a specific model and task.
    
    Args:
        model_key: Key identifier for the model ('bielik', 'gemma', or 'phi')
        task: Task for the pipeline (e.g., 'text-generation', 'text-classification')
        model: Optional pre-loaded model (if None, will be loaded)
        tokenizer: Optional pre-loaded tokenizer (if None, will be loaded)
        device: Device to load the model on ('cpu', 'cuda', 'auto')
        **kwargs: Additional arguments to pass to the pipeline
        
    Returns:
        Hugging Face pipeline
        
    Raises:
        ValueError: If the model_key is not recognized
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = initialize_model(model_key=model_key, device=device)
    
    # Check if model is distributed (loaded with device_map="auto" or offloading)
    is_distributed = hasattr(model, "_is_distributed") and model._is_distributed
    
    # For distributed models, don't specify device in the pipeline
    if is_distributed:
        logger.info(f"Creating pipeline for distributed model {model_key} without device specification")
        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            **kwargs
        )
    else:
        # Determine device for non-distributed models
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Creating pipeline for model {model_key} on device {device}")
        pipe = pipeline(
            task,
            model=model,
            tokenizer=tokenizer,
            device=device if device != "cpu" else -1,
            **kwargs
        )
    
    return pipe


def get_generation_config(model_key: str) -> Dict:
    """
    Get recommended generation configuration for a specific model.
    
    Args:
        model_key: Key identifier for the model ('bielik', 'gemma', or 'phi')
        
    Returns:
        Dictionary with recommended generation parameters
        
    Raises:
        ValueError: If the model_key is not recognized
    """
    if model_key not in MODEL_CONFIGS:
        valid_keys = list(MODEL_CONFIGS.keys())
        raise ValueError(
            f"Invalid model key: {model_key}. "
            f"Valid options are: {', '.join(valid_keys)}"
        )
    
    # Common generation parameters
    common_params = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }
    
    # Model-specific adjustments
    if model_key == 'bielik':
        return {
            **common_params,
            "max_new_tokens": 1024,  # Polish model might need more tokens for Polish language
            "repetition_penalty": 1.2,  # Stronger repetition penalty
        }
    elif model_key == 'gemma':
        return {
            **common_params,
            "temperature": 0.8,  # Slightly higher temperature for Gemma
        }
    elif model_key == 'phi':
        return {
            **common_params,
            "temperature": 0.6,  # More conservative for Phi
            "top_p": 0.92,
        }
    
    return common_params