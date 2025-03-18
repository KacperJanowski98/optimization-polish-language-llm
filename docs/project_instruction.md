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
   pip install -r requirements-torch.txt
   pip install -r requirements.txt
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

### 3. Dataset Preparation

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


Using existing datasets from Hugging Face:

1. **[allegro/klej-dyk](https://huggingface.co/datasets/allegro/klej-dyk)** - Question-answer pairs obtained from Czy wiesz... section of Polish Wikipedia
2. **[allegro/klej-polemo2-in](https://huggingface.co/datasets/allegro/klej-polemo2-in)** - The PolEmo2.0 is a dataset of online consumer reviews from four domains: medicine, hotels, products, and university. It is human-annotated on a level of full reviews and individual sentences
3. **[allegro/klej-psc](https://huggingface.co/datasets/allegro/klej-psc)** - The Polish Summaries Corpus (PSC) is a dataset of summaries for 569 news articles
4. **[allegro/klej-cdsc-e](https://huggingface.co/datasets/allegro/klej-cdsc-e)** - Polish CDSCorpus consists of 10K Polish sentence pairs which are human-annotated for semantic relatedness (CDSC-R) and entailment (CDSC-E)

#### Evaluation klej-dyk

```python
import random
from pprint import pprint

from datasets import load_dataset, load_metric

dataset = load_dataset("allegro/klej-dyk")
dataset = dataset.class_encode_column("target")
references = dataset["test"]["target"]

# generate random predictions
predictions = [random.randrange(max(references) + 1) for _ in range(len(references))]

acc = load_metric("accuracy")
f1 = load_metric("f1")

acc_score = acc.compute(predictions=predictions, references=references)
f1_score = f1.compute(predictions=predictions, references=references, average="macro")

pprint(acc_score)
pprint(f1_score)

# {'accuracy': 0.5286686103012633}
# {'f1': 0.46700507614213194}
```

**Tasks (input, output, and metrics)**

The task is to predict if the answer to the given question is correct or not.

Input ('question sentence', 'answer' columns): question and answer sentences

Output ('target' column): 1 if the answer is correct, 0 otherwise.

Domain: Wikipedia

Measurements: F1-Score

#### Evaluation klej-polemo2-in

```python
import random
from pprint import pprint

from datasets import load_dataset, load_metric

dataset = load_dataset("allegro/klej-polemo2-in")
dataset = dataset.class_encode_column("target")
references = dataset["test"]["target"]

# generate random predictions
predictions = [random.randrange(max(references) + 1) for _ in range(len(references))]

acc = load_metric("accuracy")
f1 = load_metric("f1")

acc_score = acc.compute(predictions=predictions, references=references)
f1_score = f1.compute(predictions=predictions, references=references, average="macro")

pprint(acc_score)
pprint(f1_score)

# {'accuracy': 0.25069252077562326}
# {'f1': 0.23760962219870274}
```

**Tasks (input, output, and metrics)**

The task is to predict the correct label of the review.

Input ('text' column): sentence

Output ('target' column): label for sentence sentiment ('zero': neutral, 'minus': negative, 'plus': positive, 'amb': ambiguous)

Domain: Online reviews

Measurements: Accuracy

#### Evaluation klej-psc

```python
import random
from pprint import pprint

from datasets import load_dataset, load_metric

dataset = load_dataset("allegro/klej-psc")
dataset = dataset.class_encode_column("label")
references = dataset["test"]["label"]

# generate random predictions
predictions = [random.randrange(max(references) + 1) for _ in range(len(references))]

acc = load_metric("accuracy")
f1 = load_metric("f1")

acc_score = acc.compute(predictions=predictions, references=references)
f1_score = f1.compute(predictions=predictions, references=references, average="macro")

pprint(acc_score)
pprint(f1_score)

# {'accuracy': 0.18588469184890655}
# {'f1': 0.17511412402843068}
```

**Tasks (input, output, and metrics)**

The task is to predict whether the extract text and summary are similar.

Based on PSC, we formulate a text-similarity task. We generate the positive pairs (i.e., referring to the same article) using only those news articles with both extractive and abstractive summaries. We match each extractive summary with two least similar abstractive ones of the same article. To create negative pairs, we follow a similar procedure. We find two most similar abstractive summaries for each extractive summary, but from different articles.

Input ('extract_text', 'summary_text' columns): extract text and summary text sentences

Output ('label' column): label: 1 indicates summary is similar, 0 means that it is not similar

Domain: News articles

Measurements: F1-Score

#### Evaluation klej-cdsc-e
```python
import random
from pprint import pprint

from datasets import load_dataset, load_metric

dataset = load_dataset("allegro/klej-cdsc-e")
dataset = dataset.class_encode_column("entailment_judgment")
references = dataset["test"]["entailment_judgment"]

# generate random predictions
predictions = [random.randrange(max(references) + 1) for _ in range(len(references))]

acc = load_metric("accuracy")
f1 = load_metric("f1")

acc_score = acc.compute(predictions=predictions, references=references)
f1_score = f1.compute(predictions=predictions, references=references, average="macro")

pprint(acc_score)
pprint(f1_score)

# {'accuracy': 0.325}
# {'f1': 0.2736171695141161}
```

**Tasks (input, output, and metrics)**

The entailment relation between two sentences is labeled with entailment, contradiction, or neutral. The task is to predict if the premise entails the hypothesis (entailment), negates the hypothesis (contradiction), or is unrelated (neutral).

b entails a (a wynika z b) – if a situation or an event described by sentence b occurs, it is recognized that a situation or an event described by a occurs as well, i.e., a and b refer to the same event or the same situation;

Input: ('sentence_A', 'sentence_B'): sentence pair

Output ('entailment_judgment' column): one of the possible entailment relations (entailment, contradiction, neutral)

Domain: image captions

Measurements: Accuracy

### 4. Model Initialization
 
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

### 5. Evaluation Framework

Create a comprehensive evaluation framework to compare the three models' performance on Polish language tasks:

1. **Setup evaluation utilities**:
   - Create consistent evaluation methods for all models
   - Ensure comparable generation parameters across models
   - Implement proper error handling and logging

2. **Standardized evaluation pipeline**:

```python
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_metric
from transformers import pipeline

def evaluate_model_on_dataset(model_name, tokenizer, dataset, task_type, device="cuda"):
    """
    Evaluate a model on a specific dataset
    
    Args:
        model_name: Name of the model or model object
        tokenizer: Associated tokenizer
        dataset: HuggingFace dataset
        task_type: Type of task (classification, qa, entailment, etc.)
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    results = {}
    
    # Setup appropriate pipeline based on task type
    if task_type == "classification":
        pipe = pipeline("text-classification", model=model_name, tokenizer=tokenizer, device=device)
    elif task_type == "qa":
        pipe = pipeline("question-answering", model=model_name, tokenizer=tokenizer, device=device)
    elif task_type == "entailment":
        pipe = pipeline("text-classification", model=model_name, tokenizer=tokenizer, device=device)
    
    # Task-specific evaluation logic
    if dataset.info.builder_name == "allegro/klej-dyk":
        results = evaluate_dyk(pipe, dataset, model_name)
    elif dataset.info.builder_name == "allegro/klej-polemo2-in":
        results = evaluate_polemo(pipe, dataset, model_name)
    elif dataset.info.builder_name == "allegro/klej-psc":
        results = evaluate_psc(pipe, dataset, model_name)
    elif dataset.info.builder_name == "allegro/klej-cdsc-e":
        results = evaluate_cdsc_e(pipe, dataset, model_name)
        
    return results
```

3. **Task-specific evaluation functions**:

```python
def evaluate_dyk(pipe, dataset, model_name):
    """Evaluate model on klej-dyk dataset (question-answer correctness)"""
    predictions = []
    references = dataset["test"]["target"]
    
    for i, sample in tqdm(enumerate(dataset["test"]), total=len(dataset["test"])):
        question = sample["question"]
        answer = sample["answer"]
        prompt = f"Pytanie: {question}\nOdpowiedź: {answer}\nCzy ta odpowiedź jest poprawna? (tak/nie)"
        
        # Get model prediction
        result = pipe(prompt)
        # Map prediction to binary label (implementation depends on model output format)
        pred = 1 if result[0]["label"] == "tak" or result[0]["score"] > 0.5 else 0
        predictions.append(pred)
        
    # Calculate metrics
    acc = load_metric("accuracy")
    f1 = load_metric("f1")
    
    acc_score = acc.compute(predictions=predictions, references=references)
    f1_score = f1.compute(predictions=predictions, references=references, average="macro")
    
    return {
        "model": model_name,
        "dataset": "klej-dyk",
        "accuracy": acc_score["accuracy"],
        "f1": f1_score["f1"]
    }

def evaluate_polemo(pipe, dataset, model_name):
    """Evaluate model on klej-polemo2-in dataset (sentiment analysis)"""
    # Implementation for sentiment analysis
    # ...

def evaluate_psc(pipe, dataset, model_name):
    """Evaluate model on klej-psc dataset (text similarity)"""
    # Implementation for text similarity
    # ...

def evaluate_cdsc_e(pipe, dataset, model_name):
    """Evaluate model on klej-cdsc-e dataset (entailment)"""
    # Implementation for entailment
    # ...
```

4. **Run evaluations across all models and datasets**:

```python
def run_full_evaluation(models_dict):
    """
    Run evaluation on all models across all datasets
    
    Args:
        models_dict: Dictionary of model and tokenizer pairs
        
    Returns:
        DataFrame containing all evaluation results
    """
    results = []
    
    datasets_config = [
        {"name": "allegro/klej-dyk", "task": "classification", "metric": "f1"},
        {"name": "allegro/klej-polemo2-in", "task": "classification", "metric": "accuracy"},
        {"name": "allegro/klej-psc", "task": "classification", "metric": "f1"},
        {"name": "allegro/klej-cdsc-e", "task": "entailment", "metric": "accuracy"}
    ]
    
    for model_name, (model, tokenizer) in models_dict.items():
        for dataset_config in datasets_config:
            dataset = load_dataset(dataset_config["name"])
            
            # Run evaluation for this model on this dataset
            result = evaluate_model_on_dataset(
                model, tokenizer, dataset, dataset_config["task"]
            )
            
            results.append(result)
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    return results_df
```

5. **Performance visualization and analysis**:

```python
def visualize_results(results_df):
    """
    Create visualizations to compare model performance
    
    Args:
        results_df: DataFrame containing evaluation results
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the matplotlib figure
    plt.figure(figsize=(14, 10))
    
    # Create a grouped bar chart
    for i, dataset in enumerate(results_df["dataset"].unique()):
        dataset_results = results_df[results_df["dataset"] == dataset]
        
        plt.subplot(2, 2, i+1)
        
        # Get the primary metric for this dataset
        if dataset == "allegro/klej-dyk" or dataset == "allegro/klej-psc":
            metric = "f1"
        else:
            metric = "accuracy"
            
        # Plot the metric for each model
        sns.barplot(x="model", y=metric, data=dataset_results)
        plt.title(f"{dataset} - {metric.upper()}")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("results/performance_comparison.png")
    plt.show()

def generate_summary_report(results_df):
    """
    Generate a summary report comparing model performance
    
    Args:
        results_df: DataFrame containing evaluation results
        
    Returns:
        String containing performance summary
    """
    summary = "# Polish Language Model Performance Summary\n\n"
    
    # Overall average performance across all tasks
    summary += "## Overall Performance\n\n"
    
    # Calculate average performance per model
    overall_avg = results_df.groupby("model")[["accuracy", "f1"]].mean().reset_index()
    summary += f"Average metrics across all tasks:\n\n{overall_avg.to_markdown()}\n\n"
    
    # Per-dataset performance
    summary += "## Performance by Dataset\n\n"
    
    for dataset in results_df["dataset"].unique():
        summary += f"### {dataset}\n\n"
        
        dataset_results = results_df[results_df["dataset"] == dataset]
        summary += f"{dataset_results.to_markdown()}\n\n"
    
    # Determine the best model overall
    best_model_acc = overall_avg.loc[overall_avg["accuracy"].idxmax()]["model"]
    best_model_f1 = overall_avg.loc[overall_avg["f1"].idxmax()]["model"]
    
    summary += "## Conclusion\n\n"
    
    if best_model_acc == best_model_f1:
        summary += f"The best performing model overall is **{best_model_acc}**.\n\n"
    else:
        summary += f"The best performing model for accuracy is **{best_model_acc}**.\n"
        summary += f"The best performing model for F1 score is **{best_model_f1}**.\n\n"
    
    return summary
```

6. **Evaluation metrics interpretation**:

When comparing the models, consider the following:

- **Overall performance**: Average metrics across all tasks
- **Task-specific strengths**: Some models may excel at certain tasks
- **Consistency**: Models with lower variance across tasks may be more reliable
- **Resource efficiency**: Consider performance relative to model size
- **Speed-quality tradeoff**: Measure inference time alongside accuracy/F1
