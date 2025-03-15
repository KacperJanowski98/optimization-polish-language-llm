# Polish Language Model Evaluation Project

## Overview

This document provides comprehensive instructions for implementing a project to evaluate the performance of three language models on Polish language tasks:

1. **[Bielik-11B-v2.3-Instruct](https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct)** - A specialized Polish language model
2. **[Google Gemma-3-4B-IT](https://huggingface.co/google/gemma-3-4b-it)** - A multilingual model
3. **[Microsoft Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)** - A multilingual model

The project will evaluate how well these models perform on Polish language tasks, comparing the performance of the Polish-specific model against the multilingual models.

## Objectives

- Download and initialize the three language models from Hugging Face
- Create appropriate Polish language test datasets
- Develop code to evaluate models on various Polish language tasks
- Compare and analyze model performance

## Requirements

- Python 3.10+
- PyTorch
- Transformers library
- Hugging Face account with access token
- Sufficient computational resources (GPU recommended)
- Basic knowledge of NLP and language models

## Project Structure

```
polish_lm_evaluation/
├── data/                 # Test datasets 
├── models/               # Model weights (if cached locally)
├── notebooks/            # Jupyter notebooks for evaluation
├── src/                  # Source code
│   ├── init_models.py    # Model initialization code
│   ├── datasets.py       # Dataset preparation
│   ├── evaluation.py     # Evaluation utilities
│   └── utils.py          # Helper functions
├── results/              # Evaluation results
├── .env                  # Environment variables (API keys)
└── .gitignore            # Git ignore file
```

## Implementation Steps

### 1. Environment Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install torch transformers datasets evaluate pandas matplotlib jupyter sentencepiece huggingface_hub python-dotenv
   ```

3. **Create a .env file for storing API keys**:
   ```bash
   touch .env
   ```

4. **Add your Hugging Face token to the .env file**:
   ```
   HF_TOKEN=your_hugging_face_token_here
   ```

### 2. Hugging Face Authentication

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co) if you don't have one
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Store the token in your .env file as shown above
4. Accept the model terms for each model by visiting their respective Hugging Face pages

For authentication in your code using the .env file:

```python
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables. Please add it to your .env file.")

# Login to Hugging Face
login(token=hf_token)
```

### 3. Model Initialization

Create code to download and initialize the three models:

1. **Requirements**:
   - Handle model-specific options (some models may need different parameters)
   - Implement proper memory management for large models
   - Configure generation parameters appropriately

2. **Model-specific considerations**:
   - **Bielik-11B**: Being 11B parameters, consider using quantization (int8)
   - **Gemma-3-4B**: Follow Google's terms of use and licensing
   - **Phi-4-mini**: Check for any Microsoft-specific requirements

3. **Template structure**:
   ```python
   def initialize_models(device="cuda", load_in_8bit=False):
       """
       Initialize all three models for evaluation
       
       Args:
           device: Device to load models on
           load_in_8bit: Whether to use 8-bit quantization
           
       Returns:
           Dictionary of model and tokenizer pairs
       """
       # Initialization code here
   ```

### 4. Dataset Preparation

Create datasets for evaluating Polish language capabilities:

1. **Task categories to consider**:
   - Text completion/generation in Polish
   - Translation (English → Polish)
   - Question answering in Polish
   - Text classification in Polish
   - Summarization of Polish text
   - Grammar and style evaluation

2. **Dataset sources**:
   - Public Polish NLP datasets (e.g., PolEval, Polish GLUE)
   - Manually created test cases
   - Existing multilingual datasets with Polish components
   - HuggingFace datasets with Polish language support

3. **Dataset processing template**:
   ```python
   def prepare_polish_datasets(tasks=["translation", "qa", "generation"]):
       """
       Prepare datasets for Polish language evaluation
       
       Args:
           tasks: List of tasks to prepare datasets for
           
       Returns:
           Dictionary of task-specific datasets
       """
       # Dataset preparation code here
   ```

### 5. Evaluation Framework

Design an evaluation framework that:

1. **Supports multiple evaluation metrics**:
   - BLEU, ROUGE for translation/generation
   - Accuracy for classification
   - F1/Precision/Recall for QA tasks
   - Custom metrics for Polish-specific evaluation

2. **Implements proper prompt engineering**:
   - Design task-specific prompts in Polish
   - Ensure consistency across models
   - Account for different model capabilities

3. **Manages computational resources**:
   - Batch processing where appropriate
   - Memory management for large models
   - Resource cleanup between tests

4. **Template structure**:
   ```python
   def evaluate_models(models, tokenizers, datasets, metrics=["bleu", "rouge"]):
       """
       Evaluate models on Polish language tasks
       
       Args:
           models: Dictionary of initialized models
           tokenizers: Dictionary of tokenizers
           datasets: Task-specific datasets
           metrics: Evaluation metrics to use
           
       Returns:
           Dictionary of evaluation results
       """
       # Evaluation code here
   ```

### 6. Evaluation Notebook

Create a Jupyter notebook for running evaluations and visualizing results:

1. **Structure**:
   - Environment setup and imports
   - Model initialization
   - Dataset loading
   - Model evaluation
   - Results visualization
   - Analysis and conclusions in raport form

2. **Visualization ideas**:
   - Comparative bar charts for metrics
   - Error analysis tables
   - Task-specific performance breakdown
   - Inference time comparison
   - Memory usage comparison

## Code Templates

### Model Initialization with .env Support

```python
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables
load_dotenv()

# Authenticate with Hugging Face
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables")

login(token=hf_token)

def load_model(model_id, use_8bit=False, device="cuda"):
    """Load a model and its tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure model loading options
    model_kwargs = {
        "device_map": "auto", 
        "trust_remote_code": True,
        "token": hf_token
    }
    
    if use_8bit:
        model_kwargs["load_in_8bit"] = True
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    
    return model, tokenizer
```

### Dataset Processing

```python
from datasets import load_dataset, Dataset

def load_polish_qa_dataset(max_samples=100):
    """Load a Polish QA dataset"""
    # Example using a multilingual dataset and filtering for Polish
    dataset = load_dataset("squad_v2", split="validation")
    
    # Filter or transform for Polish
    # This is just a placeholder - you'd need a real Polish dataset
    polish_examples = []
    
    # Return as a HuggingFace Dataset
    return Dataset.from_dict({
        "question": [ex["question"] for ex in polish_examples],
        "context": [ex["context"] for ex in polish_examples],
        "answer": [ex["answer"] for ex in polish_examples]
    })
```

### Evaluation Function

```python
import evaluate

def evaluate_translation(models, tokenizers, dataset, source_lang="en", target_lang="pl"):
    """Evaluate models on translation task"""
    results = {}
    bleu = evaluate.load("bleu")
    
    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]
        translations = []
        references = []
        
        for example in dataset:
            # Create translation prompt
            prompt = f"Translate from {source_lang} to {target_lang}: {example['source']}"
            
            # Generate translation
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=100)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up translation
            translation = translation.replace(prompt, "").strip()
            
            translations.append(translation)
            references.append([example["target"]])
        
        # Calculate BLEU score
        score = bleu.compute(predictions=translations, references=references)
        results[model_name] = {
            "bleu": score["bleu"],
            "translations": translations
        }
    
    return results
```

## Evaluation Tasks

Here are suggested Polish language tasks for evaluation:

1. **Text Generation**:
   - Polish creative writing
   - Polish text completion
   - Following instructions in Polish

2. **Translation**:
   - English → Polish
   - Polish → English
   - Quality and fluency evaluation

3. **Question Answering**:
   - Factual questions in Polish
   - Reading comprehension in Polish

4. **Grammar and Fluency**:
   - Polish grammar correction
   - Polish text coherence
   - Handling Polish-specific linguistic features

## Results Analysis

Design analysis to answer these questions:

1. Does the Polish-specialized model (Bielik-11B) outperform multilingual models?
2. What are the strengths and weaknesses of each model?
3. Which Polish language features are handled well/poorly?
4. How do model sizes affect performance on Polish tasks?
5. Are there specific Polish linguistic phenomena that challenge the models?

## Next Steps and Extensions

Possible extensions to the project:

1. Testing more models or model sizes
2. More comprehensive Polish test sets
3. Human evaluation to complement automatic metrics
4. Fine-tuning experiments for Polish-specific tasks
5. Domain-specific Polish evaluation (legal, medical, etc.)

## Conclusion

This project will provide valuable insights into how well different language models handle the Polish language, comparing a specialized Polish model against multilingual models. The results can guide model selection for Polish language applications and identify areas where current models need improvement.
