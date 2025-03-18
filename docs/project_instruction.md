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

## Task

Based on these datasets, create a dataset with the option to choose the amount of data to be retrieved from the sets. This dataset will be used to evaluate the performance of Polish language models.
