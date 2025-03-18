"""
Evaluation framework for Polish language model evaluation.

This module provides comprehensive utilities for evaluating the performance of 
different language models on Polish language tasks. It includes:

1. Task-specific evaluation functions for different datasets
2. Metrics calculation and standardization
3. Result aggregation and comparison utilities

The framework ensures fair and consistent evaluation across all models.
"""

import logging
import time
from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from datasets import Dataset
from evaluate import load as load_metric
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    pipeline
)

from src.init_models import create_model_pipeline, get_generation_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Task-specific evaluation functions

def evaluate_dyk(
    model_key: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model on klej-dyk dataset (question-answer correctness).
    
    Args:
        model_key: Key identifier for the model
        model: The model to evaluate
        tokenizer: Model tokenizer
        dataset: The dataset to evaluate on
        device: Device to run evaluation on
        max_samples: Maximum number of samples to evaluate (None for all)
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating {model_key} on DYK dataset (question-answer correctness)")
    
    # Set up text generation pipeline
    generation_config = get_generation_config(model_key)
    pipe = create_model_pipeline(
        model_key=model_key,
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        **generation_config
    )
    
    # Prepare data
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    predictions = []
    references = samples["target"]
    details = []
    
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {model_key} on DYK")):
        question = sample["question"]
        answer = sample["answer"]
        target = sample["target"]
        
        # Create prompt
        prompt = f"""Pytanie: {question}
Odpowiedź: {answer}
Czy ta odpowiedź jest poprawna? Odpowiedz tylko 'tak' lub 'nie'."""
        
        # Get model response
        try:
            result = pipe(prompt, max_new_tokens=10, do_sample=False)
            response = result[0]["generated_text"][len(prompt):].strip().lower()
            
            # Parse response to get prediction
            if "tak" in response:
                pred = 1
            elif "nie" in response:
                pred = 0
            else:
                # If model didn't produce a clear yes/no, fallback to the most likely
                # based on the start of the response
                pred = 1 if response.startswith("t") else 0
            
            # Store prediction details for analysis
            details.append({
                "id": i,
                "question": question,
                "answer": answer,
                "target": target,
                "response": response,
                "prediction": pred,
                "correct": pred == target
            })
            
            predictions.append(pred)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            # Use a fallback prediction (most frequent class)
            fallback_pred = 1  # Assuming 'yes' is more common
            predictions.append(fallback_pred)
            
            details.append({
                "id": i,
                "question": question,
                "answer": answer,
                "target": target,
                "response": "ERROR",
                "prediction": fallback_pred,
                "correct": fallback_pred == target,
                "error": str(e)
            })
    
    end_time = time.time()
    
    # Calculate metrics
    accuracy = load_metric("accuracy")
    f1 = load_metric("f1")
    
    acc_score = accuracy.compute(predictions=predictions, references=references)
    f1_score = f1.compute(predictions=predictions, references=references, average="macro")
    
    # Calculate additional metrics
    precision = sum(d["correct"] for d in details) / len(details)
    
    # Prepare results
    results = {
        "model": model_key,
        "dataset": "dyk",
        "task": "question-answer correctness",
        "accuracy": acc_score["accuracy"],
        "f1": f1_score["f1"],
        "precision": precision,
        "samples_evaluated": len(samples),
        "evaluation_time": end_time - start_time,
        "details": details
    }
    
    logger.info(f"Evaluation results for {model_key} on DYK: Accuracy={acc_score['accuracy']:.4f}, F1={f1_score['f1']:.4f}")
    
    return results


def evaluate_polemo(
    model_key: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model on klej-polemo2-in dataset (sentiment analysis).
    
    Args:
        model_key: Key identifier for the model
        model: The model to evaluate
        tokenizer: Model tokenizer
        dataset: The dataset to evaluate on
        device: Device to run evaluation on
        max_samples: Maximum number of samples to evaluate (None for all)
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating {model_key} on POLEMO dataset (sentiment analysis)")
    
    # Set up text generation pipeline
    generation_config = get_generation_config(model_key)
    pipe = create_model_pipeline(
        model_key=model_key,
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        **generation_config
    )
    
    # Prepare data
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    # Map class labels
    class_mapping = {
        "zero": "neutralny",
        "minus": "negatywny",
        "plus": "pozytywny",
        "amb": "niejednoznaczny"
    }
    
    # Reverse mapping for prediction
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    
    predictions = []
    references = []
    details = []
    
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {model_key} on POLEMO")):
        text = sample["sentence"]
        target = sample["target"]
        
        # Add reference to numeric class index
        references.append(target)
        
        # Create prompt
        prompt = f"""Przeanalizuj sentyment poniższego tekstu i sklasyfikuj go jako "neutralny", "pozytywny", "negatywny" lub "niejednoznaczny".

Tekst: {text}

Sentyment:"""
        
        # Get model response
        try:
            result = pipe(prompt, max_new_tokens=20, do_sample=False)
            response = result[0]["generated_text"][len(prompt):].strip().lower()
            
            # Parse response to get prediction
            pred_class = None
            for class_name, class_term in class_mapping.items():
                if class_term in response:
                    pred_class = class_name
                    break
            
            # If no clear match, use the closest match
            if pred_class is None:
                for class_name, class_term in class_mapping.items():
                    if class_term[:4] in response:  # Match first few chars
                        pred_class = class_name
                        break
            
            # If still no match, default to most frequent class
            if pred_class is None:
                pred_class = "zero"  # Default to neutral
            
            # Convert class name to class index
            class_indices = list(set(references))
            class_indices.sort()  # Ensure consistent mapping
            class_to_idx = {c: i for i, c in enumerate(class_indices)}
            pred = class_to_idx.get(pred_class, 0)  # Default to first class if not found
            
            # Store prediction details
            details.append({
                "id": i,
                "text": text,
                "target_class": list(class_mapping.keys())[target] if target < len(class_mapping) else "unknown",
                "target_idx": target,
                "response": response,
                "prediction_class": pred_class,
                "prediction_idx": pred,
                "correct": pred == target
            })
            
            predictions.append(pred)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            # Use a fallback prediction (most frequent class)
            fallback_pred = 0  # Assuming 'neutral' is most common
            predictions.append(fallback_pred)
            
            details.append({
                "id": i,
                "text": text,
                "target_class": list(class_mapping.keys())[target] if target < len(class_mapping) else "unknown",
                "target_idx": target,
                "response": "ERROR",
                "prediction_class": "unknown",
                "prediction_idx": fallback_pred,
                "correct": fallback_pred == target,
                "error": str(e)
            })
    
    end_time = time.time()
    
    # Calculate metrics
    accuracy = load_metric("accuracy")
    f1 = load_metric("f1")
    
    acc_score = accuracy.compute(predictions=predictions, references=references)
    f1_score = f1.compute(predictions=predictions, references=references, average="macro")
    
    # Calculate additional metrics
    precision = sum(d["correct"] for d in details) / len(details)
    
    # Prepare results
    results = {
        "model": model_key,
        "dataset": "polemo2",
        "task": "sentiment analysis",
        "accuracy": acc_score["accuracy"],
        "f1": f1_score["f1"],
        "precision": precision,
        "samples_evaluated": len(samples),
        "evaluation_time": end_time - start_time,
        "details": details
    }
    
    logger.info(f"Evaluation results for {model_key} on POLEMO: Accuracy={acc_score['accuracy']:.4f}, F1={f1_score['f1']:.4f}")
    
    return results


def evaluate_psc(
    model_key: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model on klej-psc dataset (text similarity).
    
    Args:
        model_key: Key identifier for the model
        model: The model to evaluate
        tokenizer: Model tokenizer
        dataset: The dataset to evaluate on
        device: Device to run evaluation on
        max_samples: Maximum number of samples to evaluate (None for all)
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating {model_key} on PSC dataset (text similarity)")
    
    # Set up text generation pipeline
    generation_config = get_generation_config(model_key)
    pipe = create_model_pipeline(
        model_key=model_key,
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        **generation_config
    )
    
    # Prepare data
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    predictions = []
    references = samples["label"]
    details = []
    
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {model_key} on PSC")):
        extract_text = sample["extract_text"]
        summary_text = sample["summary_text"]
        target = sample["label"]
        
        # Create prompt
        prompt = f"""Porównaj poniższy tekst źródłowy i podsumowanie, a następnie określ, czy podsumowanie prawidłowo odnosi się do tekstu źródłowego. Odpowiedz tylko 'tak' lub 'nie'.

Tekst źródłowy: {extract_text}

Podsumowanie: {summary_text}

Czy podsumowanie prawidłowo odnosi się do tekstu źródłowego?"""
        
        # Get model response
        try:
            result = pipe(prompt, max_new_tokens=10, do_sample=False)
            response = result[0]["generated_text"][len(prompt):].strip().lower()
            
            # Parse response to get prediction
            if "tak" in response:
                pred = 1
            elif "nie" in response:
                pred = 0
            else:
                # If model didn't produce a clear yes/no, fallback to the most likely
                pred = 1 if response.startswith("t") else 0
            
            # Store prediction details
            details.append({
                "id": i,
                "extract_text": extract_text[:100] + "...",  # Truncate for readability
                "summary_text": summary_text,
                "target": target,
                "response": response,
                "prediction": pred,
                "correct": pred == target
            })
            
            predictions.append(pred)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            # Use a fallback prediction (most frequent class)
            fallback_pred = 1  # Assuming 'similar' is more common
            predictions.append(fallback_pred)
            
            details.append({
                "id": i,
                "extract_text": extract_text[:100] + "...",  # Truncate for readability
                "summary_text": summary_text,
                "target": target,
                "response": "ERROR",
                "prediction": fallback_pred,
                "correct": fallback_pred == target,
                "error": str(e)
            })
    
    end_time = time.time()
    
    # Calculate metrics
    accuracy = load_metric("accuracy")
    f1 = load_metric("f1")
    
    acc_score = accuracy.compute(predictions=predictions, references=references)
    f1_score = f1.compute(predictions=predictions, references=references, average="macro")
    
    # Calculate additional metrics
    precision = sum(d["correct"] for d in details) / len(details)
    
    # Prepare results
    results = {
        "model": model_key,
        "dataset": "psc",
        "task": "text similarity",
        "accuracy": acc_score["accuracy"],
        "f1": f1_score["f1"],
        "precision": precision,
        "samples_evaluated": len(samples),
        "evaluation_time": end_time - start_time,
        "details": details
    }
    
    logger.info(f"Evaluation results for {model_key} on PSC: Accuracy={acc_score['accuracy']:.4f}, F1={f1_score['f1']:.4f}")
    
    return results


def evaluate_cdsc(
    model_key: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluate model on klej-cdsc-e dataset (entailment).
    
    Args:
        model_key: Key identifier for the model
        model: The model to evaluate
        tokenizer: Model tokenizer
        dataset: The dataset to evaluate on
        device: Device to run evaluation on
        max_samples: Maximum number of samples to evaluate (None for all)
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating {model_key} on CDSC dataset (entailment)")
    
    # Set up text generation pipeline
    generation_config = get_generation_config(model_key)
    pipe = create_model_pipeline(
        model_key=model_key,
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        **generation_config
    )
    
    # Prepare data
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    
    # Map class labels
    class_mapping = {
        "entailment": "wynikanie",
        "contradiction": "sprzeczność",
        "neutral": "neutralność"
    }
    
    # Reverse mapping for prediction
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    
    predictions = []
    references = []
    details = []
    
    start_time = time.time()
    
    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {model_key} on CDSC")):
        sentence_a = sample["sentence_A"]
        sentence_b = sample["sentence_B"]
        target = sample["entailment_judgment"]
        
        # Add reference to numeric class index
        references.append(target)
        
        # Create prompt
        prompt = f"""Określ relację między dwoma poniższymi zdaniami, wybierając jedną z opcji: "wynikanie" (zdanie B wynika z A), "sprzeczność" (zdania A i B są sprzeczne) lub "neutralność" (zdania A i B nie są ani sprzeczne, ani jedno nie wynika z drugiego).

Zdanie A: {sentence_a}
Zdanie B: {sentence_b}

Relacja:"""
        
        # Get model response
        try:
            result = pipe(prompt, max_new_tokens=20, do_sample=False)
            response = result[0]["generated_text"][len(prompt):].strip().lower()
            
            # Parse response to get prediction
            pred_class = None
            for class_name, class_term in class_mapping.items():
                if class_term in response:
                    pred_class = class_name
                    break
            
            # If no clear match, use the closest match
            if pred_class is None:
                for class_name, class_term in class_mapping.items():
                    if class_term[:4] in response:  # Match first few chars
                        pred_class = class_name
                        break
            
            # If still no match, default to most frequent class
            if pred_class is None:
                pred_class = "neutral"  # Default to neutral
            
            # Convert class name to class index
            class_indices = list(set(references))
            class_indices.sort()  # Ensure consistent mapping
            class_to_idx = {c: i for i, c in enumerate(class_indices)}
            pred = class_to_idx.get(pred_class, 0)  # Default to first class if not found
            
            # Store prediction details
            details.append({
                "id": i,
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "target_class": list(class_mapping.keys())[target] if target < len(class_mapping) else "unknown",
                "target_idx": target,
                "response": response,
                "prediction_class": pred_class,
                "prediction_idx": pred,
                "correct": pred == target
            })
            
            predictions.append(pred)
            
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            # Use a fallback prediction (most frequent class)
            fallback_pred = 0  # Assuming 'neutral' is most common
            predictions.append(fallback_pred)
            
            details.append({
                "id": i,
                "sentence_a": sentence_a,
                "sentence_b": sentence_b,
                "target_class": list(class_mapping.keys())[target] if target < len(class_mapping) else "unknown",
                "target_idx": target,
                "response": "ERROR",
                "prediction_class": "unknown",
                "prediction_idx": fallback_pred,
                "correct": fallback_pred == target,
                "error": str(e)
            })
    
    end_time = time.time()
    
    # Calculate metrics
    accuracy = load_metric("accuracy")
    f1 = load_metric("f1")
    
    acc_score = accuracy.compute(predictions=predictions, references=references)
    f1_score = f1.compute(predictions=predictions, references=references, average="macro")
    
    # Calculate additional metrics
    precision = sum(d["correct"] for d in details) / len(details)
    
    # Prepare results
    results = {
        "model": model_key,
        "dataset": "cdsc",
        "task": "entailment",
        "accuracy": acc_score["accuracy"],
        "f1": f1_score["f1"],
        "precision": precision,
        "samples_evaluated": len(samples),
        "evaluation_time": end_time - start_time,
        "details": details
    }
    
    logger.info(f"Evaluation results for {model_key} on CDSC: Accuracy={acc_score['accuracy']:.4f}, F1={f1_score['f1']:.4f}")
    
    return results


# Evaluation runner and utilities

def run_evaluation(
    model_key: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    datasets: Dict[str, Dataset],
    device: str = "cuda",
    max_samples_per_dataset: Optional[int] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run full evaluation on all datasets for a single model.
    
    Args:
        model_key: Key identifier for the model
        model: The model to evaluate
        tokenizer: Model tokenizer
        datasets: Dictionary of datasets to evaluate on
        device: Device to run evaluation on
        max_samples_per_dataset: Maximum number of samples to evaluate per dataset
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation results for each dataset
    """
    results = {}
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Evaluation functions for each dataset
    evaluation_functions = {
        "dyk": evaluate_dyk,
        "polemo2": evaluate_polemo,
        "psc": evaluate_psc,
        "cdsc": evaluate_cdsc
    }
    
    # Run evaluation for each dataset
    for dataset_name, dataset in datasets.items():
        logger.info(f"Evaluating {model_key} on {dataset_name} dataset")
        
        if dataset_name in evaluation_functions:
            try:
                # Run dataset-specific evaluation
                eval_fn = evaluation_functions[dataset_name]
                dataset_results = eval_fn(
                    model_key=model_key,
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    device=device,
                    max_samples=max_samples_per_dataset
                )
                
                results[dataset_name] = dataset_results
                
                # Save results to file if output directory is provided
                if output_dir:
                    # Create timestamped filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{model_key}_{dataset_name}_{timestamp}.json"
                    file_path = os.path.join(output_dir, filename)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(dataset_results, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Saved evaluation results to {file_path}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_key} on {dataset_name}: {str(e)}")
                results[dataset_name] = {
                    "model": model_key,
                    "dataset": dataset_name,
                    "error": str(e),
                    "status": "failed"
                }
        else:
            logger.warning(f"No evaluation function available for dataset: {dataset_name}")
            results[dataset_name] = {
                "model": model_key,
                "dataset": dataset_name,
                "error": "No evaluation function available",
                "status": "skipped"
            }
    
    return results


def aggregate_results(all_results: Dict[str, Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
    """
    Aggregate evaluation results into a DataFrame for analysis.
    
    Args:
        all_results: Nested dictionary of results organized by model and dataset
        
    Returns:
        DataFrame with aggregated results
    """
    rows = []
    
    for model_key, model_results in all_results.items():
        for dataset_name, dataset_results in model_results.items():
            if "error" in dataset_results:
                # Skip failed evaluations
                continue
            
            # Extract metrics
            row = {
                "model": model_key,
                "dataset": dataset_name,
                "task": dataset_results.get("task", "unknown"),
                "accuracy": dataset_results.get("accuracy", float("nan")),
                "f1": dataset_results.get("f1", float("nan")),
                "precision": dataset_results.get("precision", float("nan")),
                "samples": dataset_results.get("samples_evaluated", 0),
                "time": dataset_results.get("evaluation_time", 0)
            }
            
            rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    return df


def visualize_results(results_df: pd.DataFrame, output_path: Optional[str] = None):
    """
    Create visualizations to compare model performance.
    
    Args:
        results_df: DataFrame containing evaluation results
        output_path: Path to save the visualization (if None, only displays)
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
        if dataset in ["dyk", "psc"]:
            metric = "f1"
        else:
            metric = "accuracy"
            
        # Plot the metric for each model
        sns.barplot(x="model", y=metric, data=dataset_results)
        plt.title(f"{dataset} - {metric.upper()}")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to {output_path}")
    
    plt.show()


def generate_summary_report(results_df: pd.DataFrame) -> str:
    """
    Generate a summary report comparing model performance.
    
    Args:
        results_df: DataFrame containing evaluation results
        
    Returns:
        String containing performance summary in markdown format
    """
    summary = "# Polish Language Model Performance Summary\n\n"
    
    # Overall average performance across all tasks
    summary += "## Overall Performance\n\n"
    
    # Calculate average performance per model
    overall_avg = results_df.groupby("model")[["accuracy", "f1", "precision"]].mean().reset_index()
    summary += f"Average metrics across all tasks:\n\n{overall_avg.to_markdown(index=False)}\n\n"
    
    # Per-dataset performance
    summary += "## Performance by Dataset\n\n"
    
    for dataset in results_df["dataset"].unique():
        summary += f"### {dataset}\n\n"
        
        dataset_results = results_df[results_df["dataset"] == dataset]
        summary += f"{dataset_results.to_markdown(index=False)}\n\n"
    
    # Determine the best model overall
    best_model_acc = overall_avg.loc[overall_avg["accuracy"].idxmax()]["model"]
    best_model_f1 = overall_avg.loc[overall_avg["f1"].idxmax()]["model"]
    
    summary += "## Conclusion\n\n"
    
    if best_model_acc == best_model_f1:
        summary += f"The best performing model overall is **{best_model_acc}**.\n\n"
    else:
        summary += f"The best performing model for accuracy is **{best_model_acc}**.\n"
        summary += f"The best performing model for F1 score is **{best_model_f1}**.\n\n"
    
    # Add evaluation settings and timestamp
    summary += "## Evaluation Details\n\n"
    summary += f"- Evaluation completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    summary += f"- Total datasets evaluated: {len(results_df['dataset'].unique())}\n"
    summary += f"- Models evaluated: {', '.join(results_df['model'].unique())}\n"
    
    return summary


def save_results(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str,
    prefix: Optional[str] = None
):
    """
    Save evaluation results to files.
    
    Args:
        results: Nested dictionary of results
        output_dir: Directory to save results
        prefix: Optional prefix for filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename prefix
    file_prefix = f"{prefix}_{timestamp}" if prefix else timestamp
    
    # Save complete results
    complete_results_path = os.path.join(output_dir, f"{file_prefix}_complete_results.json")
    with open(complete_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved complete results to {complete_results_path}")
    
    # Create and save aggregated results DataFrame
    results_df = aggregate_results(results)
    df_path = os.path.join(output_dir, f"{file_prefix}_results_table.csv")
    results_df.to_csv(df_path, index=False)
    
    logger.info(f"Saved results table to {df_path}")
    
    # Generate and save summary report
    report = generate_summary_report(results_df)
    report_path = os.path.join(output_dir, f"{file_prefix}_summary_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Saved summary report to {report_path}")
    
    # Create and save visualization
    viz_path = os.path.join(output_dir, f"{file_prefix}_performance_comparison.png")
    try:
        visualize_results(results_df, viz_path)
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
