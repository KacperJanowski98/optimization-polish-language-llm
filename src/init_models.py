"""
Model initialization module for Polish language model evaluation.
"""
import os
import logging
from typing import Dict, Tuple, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

from src.utils import authenticate_huggingface, setup_gpu_memory, get_device
from src.config import MODEL_GENERATION_PARAMS, DEFAULT_GENERATION_PARAMS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model identifiers
BIELIK_MODEL_ID = "speakleash/Bielik-11B-v2.3-Instruct"
GEMMA_MODEL_ID = "google/gemma-3-4b-it"
PHI_MODEL_ID = "microsoft/Phi-4-mini-instruct"

def load_model(
    model_id: str,
    use_8bit: bool = False,
    use_4bit: bool = False,
    device: Optional[str] = None,
    token: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and its tokenizer from Hugging Face.
    
    Args:
        model_id (str): Hugging Face model identifier
        use_8bit (bool): Whether to use 8-bit quantization
        use_4bit (bool): Whether to use 4-bit quantization
        device (Optional[str]): Device to load model on (None for auto)
        token (Optional[str]): Hugging Face token
        
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Loaded model and tokenizer
    """
    logger.info(f"Loading model: {model_id}")
    
    # Get device if not specified
    if device is None:
        device = get_device()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True
    )
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure model loading options
    model_kwargs = {
        "device_map": "auto" if device == "cuda" else device,
        "trust_remote_code": True,
        "token": token
    }
    
    # Handle quantization options
    if use_8bit and use_4bit:
        logger.warning("Both 8-bit and 4-bit quantization specified; using 4-bit")
        use_8bit = False
    
    if use_8bit:
        logger.info("Using 8-bit quantization")
        model_kwargs["load_in_8bit"] = True
    elif use_4bit:
        logger.info("Using 4-bit quantization")
        model_kwargs["load_in_4bit"] = True
        model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
    
    # Load the model
    logger.info(f"Loading model with options: {model_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    
    logger.info(f"Successfully loaded model: {model_id}")
    return model, tokenizer

def initialize_models(
    device: str = "cuda",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    token: Optional[str] = None,
    models_to_load: Optional[List[str]] = None
) -> Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]]:
    """
    Initialize all three models for evaluation.
    
    Args:
        device (str): Device to load models on
        load_in_8bit (bool): Whether to use 8-bit quantization
        load_in_4bit (bool): Whether to use 4-bit quantization
        token (Optional[str]): Hugging Face token
        models_to_load (Optional[List[str]]): Specific models to load
            Options: ["bielik", "gemma", "phi"]
            
    Returns:
        Dictionary of model and tokenizer pairs
    """
    # Ensure authentication
    authenticate_huggingface(token)
    
    # Setup GPU memory optimization
    if device == "cuda":
        setup_gpu_memory()
    
    # Determine which models to load
    all_models = ["bielik", "gemma", "phi"]
    if not models_to_load:
        models_to_load = all_models
    else:
        # Validate model names
        for model in models_to_load:
            if model.lower() not in all_models:
                logger.warning(f"Unknown model: {model}, will be skipped")
    
    models = {}
    
    # Map model names to their HF identifiers and loading parameters
    model_config = {
        "bielik": {
            "id": BIELIK_MODEL_ID,
            "use_8bit": load_in_8bit or True,  # Default to 8-bit for Bielik due to size
            "use_4bit": load_in_4bit
        },
        "gemma": {
            "id": GEMMA_MODEL_ID,
            "use_8bit": load_in_8bit,
            "use_4bit": load_in_4bit
        },
        "phi": {
            "id": PHI_MODEL_ID,
            "use_8bit": load_in_8bit,
            "use_4bit": load_in_4bit
        }
    }
    
    # Load each model
    for model_name in models_to_load:
        if model_name.lower() in model_config:
            config = model_config[model_name.lower()]
            try:
                model, tokenizer = load_model(
                    model_id=config["id"],
                    use_8bit=config["use_8bit"],
                    use_4bit=config["use_4bit"],
                    device=device,
                    token=token
                )
                models[model_name] = (model, tokenizer)
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {str(e)}")
    
    return models

def configure_generation_parameters(model_name: str) -> Dict:
    """
    Configure model-specific generation parameters.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Dict: Generation parameters for the model
    """
    # Start with default parameters
    params = DEFAULT_GENERATION_PARAMS.copy()
    
    # Update with model-specific parameters if available
    if model_name.lower() in MODEL_GENERATION_PARAMS:
        params.update(MODEL_GENERATION_PARAMS[model_name.lower()])
    
    return params


if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    
    # Initialize models (for testing)
    print("Testing model initialization...")
    models = initialize_models(token=token, models_to_load=["phi"])  # Start with the smallest model
    
    if models:
        print(f"Successfully loaded {len(models)} models")
        for name, (model, tokenizer) in models.items():
            print(f"Model: {name}")
            print(f"  - Model class: {model.__class__.__name__}")
            print(f"  - Tokenizer class: {tokenizer.__class__.__name__}")
    else:
        print("No models were loaded")
