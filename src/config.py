"""
Configuration settings for model evaluation.
"""

# Model-specific generation parameters
MODEL_GENERATION_PARAMS = {
    "bielik": {
        "temperature": 0.6,
        "repetition_penalty": 1.3,
        "top_p": 0.9,
        "top_k": 50,
    },
    "gemma": {
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "top_k": 50,
    },
    "phi": {
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    }
}

# Default generation parameters
DEFAULT_GENERATION_PARAMS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "num_return_sequences": 1,
}

# Tasks configuration
TASKS_CONFIG = {
    "cdsc-e": {
        "max_new_tokens": 10,
        "metric_names": ["accuracy", "f1_macro"]
    },
    "polemo2-in": {
        "max_new_tokens": 10,
        "metric_names": ["accuracy", "f1_macro"]
    },
    "translation": {
        "max_new_tokens": 150,
        "metric_names": ["bleu", "rouge1", "rouge2", "rougeL"]
    },
    # Add other tasks as needed
}
