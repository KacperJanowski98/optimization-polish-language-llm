# Polish Language Model Evaluation

This project evaluates and compares the performance of three language models on Polish language tasks using the Polish GLUE (KLEJ) benchmark:

1. **[Bielik-11B-v2.3-Instruct](https://huggingface.co/speakleash/Bielik-11B-v2.3-Instruct)** - A specialized Polish language model
2. **[Google Gemma-3-4B-IT](https://huggingface.co/google/gemma-3-4b-it)** - A multilingual model
3. **[Microsoft Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)** - A multilingual model

## Project Overview

The main goal of this project is to evaluate how well these models perform on Polish language tasks, comparing the performance of the Polish-specific model against the multilingual models using standardized Polish GLUE benchmark tasks.

## Polish GLUE (KLEJ) Benchmark

The Polish GLUE benchmark (also known as KLEJ - Kompleksowa Lista Ewaluacji Językowych) is used for evaluation. It includes the following tasks:

1. **CDSC-E** - Compositional Distributional Semantics Corpus - Entailment task
2. **CDSC-R** - Compositional Distributional Semantics Corpus - Relatedness task
3. **CBD** - Cyberbullying Detection dataset
4. **PolEmo2-in** - Sentiment analysis (in-domain)
5. **PolEmo2-out** - Sentiment analysis (out-of-domain)
6. **DYK** - Did You Know - Genuine question detection
7. **PSC** - Polish Summaries Corpus
8. **AR** - Allegro Reviews - Sentiment analysis

Additionally, the project evaluates models on:
- Translation (English → Polish)
- Polish text generation

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch
- Hugging Face account with API token

### Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure Hugging Face token:
   - Create a `.env` file in the project root
   - Add your Hugging Face token: `HF_TOKEN=your_token_here`

### Running the Evaluation

1. Open the Jupyter notebook in `notebooks/model_evaluation.ipynb`:
   ```bash
   jupyter notebook notebooks/model_evaluation.ipynb
   ```

2. Follow the steps in the notebook to:
   - Load Polish GLUE datasets
   - Initialize models
   - Run evaluations
   - Analyze and visualize results

## Project Structure

```
optimization-polish-language-llm/
├── data/                 # Test datasets and cached data
├── models/               # Model weights (if cached locally)
├── notebooks/            # Jupyter notebooks for evaluation
├── src/                  # Source code
│   ├── init_models.py    # Model initialization code
│   ├── datasets.py       # Polish GLUE dataset preparation
│   ├── evaluation.py     # Evaluation utilities
│   └── utils.py          # Helper functions
├── results/              # Evaluation results
├── .env                  # Environment variables (API keys)
└── .gitignore            # Git ignore file
```

## Metrics

The evaluation uses various metrics depending on the task:

- **Classification**: Accuracy, F1 score
- **Regression**: Mean Squared Error, Spearman correlation
- **Translation**: BLEU, ROUGE
- **Generation**: Qualitative assessment

## License

This project is open source and available under the [MIT License](LICENSE).

## References

- [Polish GLUE (KLEJ) Benchmark](https://klejbenchmark.com/)
- [KLEJ on Hugging Face](https://huggingface.co/datasets/allegro/klej)
