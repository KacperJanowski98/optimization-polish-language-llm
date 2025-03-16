"""
Utility functions for the Polish language model evaluation project.
"""
import os
import logging
from typing import Optional
from dotenv import load_dotenv
from huggingface_hub import login

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def authenticate_huggingface(token: Optional[str] = None) -> None:
    """
    Authenticate with Hugging Face using a token.
    
    Args:
        token (Optional[str]): HF token to use. If None, loads from .env file.
    
    Raises:
        ValueError: If no token is provided and none is found in .env.
    """
    # If no token is passed, try to load from .env file
    if token is None:
        logger.info("No token provided, attempting to load from .env file")
        load_dotenv()
        token = os.getenv("HF_TOKEN")
    
    if not token:
        raise ValueError(
            "HF_TOKEN not found. Please provide it as an argument or add it to your .env file."
        )
    
    # Log in to Hugging Face
    try:
        login(token=token)
        logger.info("Successfully authenticated with Hugging Face")
    except Exception as e:
        logger.error(f"Failed to authenticate with Hugging Face: {str(e)}")
        raise

def setup_gpu_memory() -> None:
    """
    Configure GPU memory settings for optimal model loading.
    This function sets appropriate PyTorch configurations for handling large models.
    """
    import torch
    
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("Enabled memory efficient SDP attention")
    except (AttributeError, RuntimeError) as e:
        logger.warning(f"Could not enable memory efficient SDP attention: {str(e)}")
    
    # Enable flash attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("Enabled flash attention")
    except (AttributeError, RuntimeError) as e:
        logger.warning(f"Could not enable flash attention: {str(e)}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        
        logger.info(f"Found {device_count} CUDA device(s)")
        logger.info(f"Using CUDA device {current_device}: {device_name}")
        
        # Log available memory
        free_memory, total_memory = torch.cuda.mem_get_info(current_device)
        free_memory_gb = free_memory / (1024**3)
        total_memory_gb = total_memory / (1024**3)
        
        logger.info(f"GPU Memory: {free_memory_gb:.2f}GB free / {total_memory_gb:.2f}GB total")
    else:
        logger.warning("CUDA is not available, using CPU")

def get_device(use_cuda: bool = True) -> str:
    """
    Determine the device to use for model loading.
    
    Args:
        use_cuda (bool): Whether to use CUDA if available.
        
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    import torch
    
    if use_cuda and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Support for M1/M2 Macs
    else:
        return "cpu"
