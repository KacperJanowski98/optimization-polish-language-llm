# Polish Language Model Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-yellow)](https://huggingface.co/)

A comprehensive framework for evaluating language models on Polish language tasks, comparing specialized Polish models against multilingual models.

## Project Overview

This project evaluates and compares the performance of four language models on Polish NLP tasks:

1. **Bielik-11B-v2.3-Instruct** (11B parameters) - A specialized Polish language model by Speakleash
2. **Google Gemma-3-4B-IT** (4B parameters) - A multilingual model by Google
3. **Microsoft Phi-4-mini-instruct** (3.8B parameters) - A multilingual model by Microsoft
4. **CYFRAGOVPL/Llama-PLLuM-8B-instruct** (8B parameters) - Polish government-backed Llama-based model

The evaluation uses a variety of Polish language datasets from the KLEJ benchmark to assess model capabilities across different linguistic tasks.

## Key Features

- Standardized evaluation framework for comparing different models
- Support for quantization and memory optimization techniques
- Comprehensive metrics for various Polish language tasks
- Analysis of Polish-specific vs multilingual model performance
- Visualization tools for performance comparison

## Evaluated Tasks

The evaluation covers four primary tasks from the KLEJ benchmark:

1. **Question-Answer Correctness** (klej-dyk): Determining if an answer is correct for a given question
2. **Sentiment Analysis** (klej-polemo2-in): Classifying the sentiment of Polish text
3. **Text Similarity** (klej-psc): Assessing similarity between text extracts and summaries
4. **Textual Entailment** (klej-cdsc-e): Determining entailment relationships between sentence pairs

## Project Structure

```
polish-language-model-evaluation/
├── data/                  # Test datasets 
├── models/                # Model weights (if cached locally)
├── notebooks/             # Jupyter notebooks for evaluation
│   ├── dataset_testing.ipynb     # For exploring datasets
│   └── model_evaluation.ipynb    # For running full evaluations
├── src/                   # Source code
│   ├── init_models.py     # Model initialization code
│   ├── datasets.py        # Dataset preparation
│   ├── evaluation.py      # Evaluation utilities
│   └── utils.py           # Helper functions
├── results/               # Evaluation results
├── docs/                  # Documentation
│   └── project_instruction.md    # Detailed project instructions
├── requirements.txt       # Python dependencies
└── requirements-torch.txt # PyTorch dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch
- A Hugging Face account with an access token
- CUDA-capable GPU (strongly recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KacperJanowski98/polish-language-model-evaluation.git
   cd polish-language-model-evaluation
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements-torch.txt
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Hugging Face token:
   ```
   HF_TOKEN=your_hugging_face_token_here
   ```

5. Accept the model terms on Hugging Face for each model:
   - [Bielik-11B-v2.3-Instruct](https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct)
   - [Google Gemma-3-4B-IT](https://huggingface.co/google/gemma-3-4b-it)
   - [Microsoft Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
   - [CYFRAGOVPL/Llama-PLLuM-8B-instruct](https://huggingface.co/CYFRAGOVPL/Llama-PLLuM-8B-instruct)

### Running Evaluations

1. Open the model evaluation notebook:
   ```bash
   jupyter notebook notebooks/model_evaluation.ipynb
   ```

2. Follow the steps in the notebook to:
   - Load and initialize models
   - Prepare evaluation datasets
   - Run evaluations
   - Analyze and visualize results

## Methodology

The evaluation follows these steps:

1. **Model Initialization**: Models are loaded with appropriate quantization (4-bit or 8-bit) and CPU offloading to manage memory efficiently
2. **Dataset Preparation**: Samples from each dataset are selected for evaluation
3. **Task-Specific Evaluation**: Each model is evaluated on each task using appropriate metrics
4. **Results Analysis**: Performance is compared across models, with special attention to:
   - Overall performance metrics
   - Task-specific strengths and weaknesses
   - Polish-specific vs. multilingual performance differences
   - Efficiency relative to model size

## Results

> Note: Results will be added after running the evaluation.

The evaluation results will include:
- Performance metrics for each model on each dataset
- Comparative analysis across models
- Visualizations of model performance
- Insights into Polish-specific vs. multilingual model capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing model hosting and APIs
- [KLEJ Benchmark](https://klejbenchmark.com/) for the Polish language evaluation datasets
- All model creators for making their models available for research
