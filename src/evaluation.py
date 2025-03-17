"""
Evaluation module for Polish language model evaluation.
"""
import difflib
import logging
import time
import re
import gc
import json
from typing import Dict, List, Tuple, Any, Optional, Callable
from tqdm.auto import tqdm

import torch
import numpy as np
import evaluate
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.datasets import create_task_specific_prompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_model_prediction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    generation_params: Dict[str, Any] = None
) -> str:
    """
    Process a single prediction with standardized error handling and logging.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for the model
        prompt: Input prompt for the model
        max_new_tokens: Maximum number of tokens to generate
        generation_params: Additional generation parameters
        
    Returns:
        str: Generated text from the model
    """
    if generation_params is None:
        generation_params = {}
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate with timeout to prevent hanging
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **generation_params
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    except (RuntimeError, ValueError) as e:
        if "CUDA out of memory" in str(e):
            logger.error(f"CUDA out of memory during generation: {str(e)}")
            # Try to recover
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "ERROR: GPU memory exhausted"
        else:
            logger.error(f"Error during model generation: {str(e)}")
            return f"ERROR: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error during model generation: {str(e)}")
        return f"ERROR: Unexpected error: {str(e)}"

def evaluate_models(
    models: Dict[str, Tuple[PreTrainedModel, PreTrainedTokenizer]],
    datasets: Dict[str, Dict[str, Any]],
    metrics: Optional[Dict[str, List[str]]] = None,
    max_samples: Optional[int] = None,
    generation_params: Optional[Dict[str, Dict]] = None,
    save_results: bool = True,
    results_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate models on Polish language tasks.
    
    Args:
        models: Dictionary of initialized models and tokenizers
        datasets: Task-specific datasets with metadata
        metrics: Dictionary mapping task types to metrics to use
        max_samples: Maximum number of samples to evaluate
        generation_params: Model-specific generation parameters
        save_results: Whether to save results to disk
        results_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation results
    """
    results = {}
    
    # Validate inputs
    if not models:
        logger.error("No models provided for evaluation")
        return results
    
    if not datasets:
        logger.error("No datasets provided for evaluation")
        return results
    
    # Use default metrics if none provided
    if metrics is None:
        metrics = {
            "classification": ["accuracy", "f1"],
            "regression": ["mse", "spearman"],
            "generation": ["bleu", "rouge"]
        }
    
    # Initialize evaluation metrics
    metric_dict = {}
    for task_type, metric_list in metrics.items():
        metric_dict[task_type] = {}
        for metric_name in metric_list:
            try:
                metric_dict[task_type][metric_name] = evaluate.load(metric_name)
                logger.info(f"Loaded metric for {task_type}: {metric_name}")
            except Exception as e:
                logger.error(f"Failed to load metric {metric_name}: {str(e)}")
    
    # Evaluate each task for each model
    for model_name, (model, tokenizer) in models.items():
        logger.info(f"Evaluating model: {model_name}")
        model_results = {}
        
        # Get model-specific generation parameters or use defaults
        model_gen_params = {}
        if generation_params and model_name in generation_params:
            model_gen_params = generation_params[model_name]
        
        # Evaluate model on each task
        for task_name, task_info in datasets.items():
            logger.info(f"Evaluating {model_name} on task: {task_name}")
            
            dataset = task_info["dataset"]
            task_type = task_info["task_type"]
            task_metric = task_info["metric"]
            
            # Limit sample size if specified
            if max_samples and len(dataset) > max_samples:
                logger.info(f"Limiting evaluation to {max_samples} samples")
                dataset = dataset.select(range(max_samples))
            
            # Get task-specific evaluation function
            eval_fn = get_evaluation_function(task_name)
            
            # Evaluate the model
            task_results = eval_fn(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                metrics=metric_dict.get(task_type, {}),
                generation_params=model_gen_params,
                task_info=task_info
            )
            
            model_results[task_name] = task_results

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleared cache after model evaluation")
            
            # Save interim results after each model
            if save_results and results_dir:
                save_evaluation_results(results, results_dir, f"interim_{model_name}")
            
            # Log partial results
            logger.info(f"Results for {model_name} on {task_name}:")
            for metric_name, score in task_results["metrics"].items():
                logger.info(f"  {metric_name}: {score:.4f}")
        
        results[model_name] = model_results
        
        # Optional: save interim results after each model
        if save_results and results_dir:
            save_evaluation_results(results, results_dir, f"interim_{model_name}")
    
    # Save final results
    if save_results and results_dir:
        save_evaluation_results(results, results_dir)
    
    return results


def get_evaluation_function(task_name: str) -> Callable:
    """
    Get the appropriate evaluation function for a task.
    
    Args:
        task_name: Name of the task
        
    Returns:
        Evaluation function for the task
    """
    # Map task names to evaluation functions
    if task_name.startswith("klej_cdsc-e"):
        return evaluate_text_entailment
    elif task_name.startswith("klej_cdsc-r"):
        return evaluate_semantic_relatedness
    elif task_name.startswith("klej_cbd"):
        return evaluate_binary_classification
    elif task_name.startswith("klej_polemo2"):
        return evaluate_sentiment_classification
    elif task_name.startswith("klej_dyk"):
        return evaluate_binary_classification
    elif task_name.startswith("klej_psc"):
        return evaluate_binary_classification
    elif task_name.startswith("klej_ar"):
        return evaluate_rating_prediction
    elif task_name == "translation":
        return evaluate_translation
    elif task_name == "generation":
        return evaluate_generation
    else:
        # Default to classification
        logger.warning(f"No specific evaluation function for {task_name}, using default")
        return evaluate_classification


def evaluate_text_entailment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metrics: Dict[str, Any],
    generation_params: Dict[str, Any],
    task_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate model on textual entailment task (CDSC-E).
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Entailment dataset
        metrics: Evaluation metrics
        generation_params: Generation parameters
        task_info: Task metadata
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating textual entailment (CDSC-E)")
    
    # Label mapping for entailment
    label_mapping = {
        0: "ENTAILMENT",
        1: "CONTRADICTION",
        2: "NEUTRAL"
    }
    
    # Reverse mapping for prediction processing
    reverse_mapping = {v.lower(): k for k, v in label_mapping.items()}
    
    predictions = []
    references = []
    generated_texts = []
    
    # Process each example
    for example in tqdm(dataset, desc="Evaluating entailment"):
        # Create prompt
        prompt = create_task_specific_prompt("klej_cdsc-e", example)
        
        # Generate prediction using the standardized helper
        generated_text = process_model_prediction(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            generation_params=generation_params
        )
        
        # Skip samples with errors
        if generated_text.startswith("ERROR:"):
            logger.warning("Skipping example due to generation error.")
            continue
            
        generated_texts.append(generated_text)
        
        # Extract prediction from generated text
        # Try to find one of the expected labels in the response
        prediction = None
        generated_lower = generated_text.lower()

        # First try exact matches with word boundaries to avoid substring matches
        for label_text in reverse_mapping.keys():
            pattern = r'\b' + re.escape(label_text) + r'\b'
            if re.search(pattern, generated_lower):
                prediction = reverse_mapping[label_text]
                break

        # If no match found, log this for analysis
        if prediction is None:
            logger.debug(f"Could not extract prediction with exact pattern. Text: {generated_text}")
            
            # As a fallback, try fuzzy matching instead of defaulting
            # Look for closest match among the labels (if any word is 80%+ similar)
            highest_ratio = 0
            best_match = None
            
            for word in generated_lower.split():
                for label_text in reverse_mapping.keys():
                    if len(word) > 3:  # Only consider words of reasonable length
                        ratio = difflib.SequenceMatcher(None, word, label_text).ratio()
                        if ratio > highest_ratio and ratio > 0.8:  # 80% similarity threshold
                            highest_ratio = ratio
                            best_match = label_text
            
            if best_match:
                prediction = reverse_mapping[best_match]
                logger.debug(f"Using fuzzy match: {best_match} (ratio: {highest_ratio:.2f})")
            else:
                # Only default as a last resort
                prediction = 2  # NEUTRAL
                logger.warning(f"Defaulting to neutral for text: {generated_text}")
        
        # Default to neutral if no clear answer found
        if prediction is None:
            prediction = 2  # NEUTRAL
        
        predictions.append(prediction)
        references.append(example["entailment_judgment"])
    
    # Calculate metrics
    metric_results = {}
    
    if "accuracy" in metrics:
        accuracy = metrics["accuracy"].compute(predictions=predictions, references=references)
        metric_results["accuracy"] = accuracy["accuracy"]
    
    if "f1" in metrics:
        f1_scores = []
        for label in set(references):
            label_preds = [1 if p == label else 0 for p in predictions]
            label_refs = [1 if r == label else 0 for r in references]
            f1 = metrics["f1"].compute(predictions=label_preds, references=label_refs, average="binary")
            f1_scores.append(f1["f1"])
        metric_results["f1_macro"] = sum(f1_scores) / len(f1_scores)
    
    # Prepare and return results
    results = {
        "metrics": metric_results,
        "predictions": predictions,
        "references": references,
        "generated_texts": generated_texts
    }
    
    return results


def evaluate_semantic_relatedness(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metrics: Dict[str, Any],
    generation_params: Dict[str, Any],
    task_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate model on semantic relatedness task (CDSC-R).
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Relatedness dataset
        metrics: Evaluation metrics
        generation_params: Generation parameters
        task_info: Task metadata
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating semantic relatedness (CDSC-R)")
    
    predictions = []
    references = []
    generated_texts = []
    
    # Process each example
    for example in tqdm(dataset, desc="Evaluating relatedness"):
        # Create prompt
        prompt = create_task_specific_prompt("klej_cdsc-r", example)
        
        # Generate prediction using the standardized helper
        generated_text = process_model_prediction(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            generation_params=generation_params
        )
        
        # Skip samples with errors
        if generated_text.startswith("ERROR:"):
            logger.warning("Skipping example due to generation error.")
            continue
            
        generated_texts.append(generated_text)
        
        # Extract prediction from generated text
        # Look for a number between 0 and 5
        prediction = None
        numbers = re.findall(r'\b[0-5](?:\.5)?\b', generated_text)
        if numbers:
            try:
                prediction = float(numbers[0])
            except:
                pass
        
        # Default to median if no clear answer found
        if prediction is None:
            prediction = 2.5
        
        predictions.append(prediction)
        references.append(example["relatedness_score"])
    
    # Calculate metrics
    metric_results = {}
    
    if "mse" in metrics:
        mse = metrics["mse"].compute(predictions=predictions, references=references)
        metric_results["mse"] = mse["mse"]
    
    if "spearman" in metrics:
        spearman = metrics["spearman"].compute(predictions=predictions, references=references)
        metric_results["spearman"] = spearman["spearmanr"]
    
    # Prepare and return results
    results = {
        "metrics": metric_results,
        "predictions": predictions,
        "references": references,
        "generated_texts": generated_texts
    }
    
    return results


def evaluate_binary_classification(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metrics: Dict[str, Any],
    generation_params: Dict[str, Any],
    task_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate model on binary classification tasks (CBD, DYK, PSC).
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Classification dataset
        metrics: Evaluation metrics
        generation_params: Generation parameters
        task_info: Task metadata
        
    Returns:
        Evaluation results
    """
    task_name = next((k for k in task_info.keys() if k.startswith("klej_")), "binary_classification")
    logger.info(f"Evaluating binary classification ({task_name})")
    
    predictions = []
    references = []
    generated_texts = []
    
    # Process each example
    for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
        # Create prompt
        prompt = create_task_specific_prompt(task_name, example)
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # Short output expected
                **generation_params
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
        
        # Extract prediction from generated text
        # Look for yes/no, tak/nie answers
        prediction = None
        if any(word in generated_text.lower() for word in ["tak", "yes"]):
            prediction = 1
        elif any(word in generated_text.lower() for word in ["nie", "no"]):
            prediction = 0
        
        # Default if no clear answer found
        if prediction is None:
            prediction = 0  # Default to the most common class
        
        predictions.append(prediction)
        references.append(example["label"])
    
    # Calculate metrics
    metric_results = {}
    
    if "accuracy" in metrics:
        accuracy = metrics["accuracy"].compute(predictions=predictions, references=references)
        metric_results["accuracy"] = accuracy["accuracy"]
    
    if "f1" in metrics:
        f1 = metrics["f1"].compute(predictions=predictions, references=references, average="binary")
        metric_results["f1"] = f1["f1"]
    
    # Prepare and return results
    results = {
        "metrics": metric_results,
        "predictions": predictions,
        "references": references,
        "generated_texts": generated_texts
    }
    
    return results


def evaluate_sentiment_classification(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metrics: Dict[str, Any],
    generation_params: Dict[str, Any],
    task_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate model on sentiment classification task (PolEmo2).
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Sentiment dataset
        metrics: Evaluation metrics
        generation_params: Generation parameters
        task_info: Task metadata
        
    Returns:
        Evaluation results
    """
    task_name = next((k for k in task_info.keys() if k.startswith("klej_polemo2")), "sentiment")
    logger.info(f"Evaluating sentiment classification ({task_name})")
    
    # Label mapping for sentiment
    label_mapping = {
        0: "POZYTYWNY",  # POSITIVE
        1: "NEGATYWNY",  # NEGATIVE
        2: "NEUTRALNY",  # NEUTRAL
        3: "MIESZANY"    # MIXED
    }
    
    # Reverse mapping for prediction processing
    reverse_mapping = {v.lower(): k for k, v in label_mapping.items()}
    
    predictions = []
    references = []
    generated_texts = []
    
    # Process each example
    for example in tqdm(dataset, desc=f"Evaluating {task_name}"):
        # Create prompt
        prompt = create_task_specific_prompt(task_name, example)
        
        # Generate prediction
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # Short output expected
                **generation_params
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
        
        # Extract prediction from generated text
        # Try to find one of the expected labels in the response
        prediction = None
        for label_text in reverse_mapping.keys():
            if label_text in generated_text.lower():
                prediction = reverse_mapping[label_text]
                break
        
        # Default to neutral if no clear answer found
        if prediction is None:
            prediction = 2  # NEUTRAL
        
        predictions.append(prediction)
        references.append(example["label"])
    
    # Calculate metrics
    metric_results = {}
    
    if "accuracy" in metrics:
        accuracy = metrics["accuracy"].compute(predictions=predictions, references=references)
        metric_results["accuracy"] = accuracy["accuracy"]
    
    if "f1" in metrics:
        f1_scores = []
        for label in set(references):
            label_preds = [1 if p == label else 0 for p in predictions]
            label_refs = [1 if r == label else 0 for r in references]
            f1 = metrics["f1"].compute(predictions=label_preds, references=label_refs, average="binary")
            f1_scores.append(f1["f1"])
        metric_results["f1_macro"] = sum(f1_scores) / len(f1_scores)
    
    # Prepare and return results
    results = {
        "metrics": metric_results,
        "predictions": predictions,
        "references": references,
        "generated_texts": generated_texts
    }
    
    return results


def evaluate_rating_prediction(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metrics: Dict[str, Any],
    generation_params: Dict[str, Any],
    task_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate model on rating prediction task (AR).
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Rating dataset
        metrics: Evaluation metrics
        generation_params: Generation parameters
        task_info: Task metadata
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating rating prediction (AR)")
    
    predictions = []
    references = []
    generated_texts = []
    
    # Process each example
    for example in tqdm(dataset, desc="Evaluating rating"):
        # Create prompt
        prompt = create_task_specific_prompt("klej_ar", example)
        
        # Generate prediction using the standardized helper
        generated_text = process_model_prediction(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            generation_params=generation_params
        )
        
        # Skip samples with errors
        if generated_text.startswith("ERROR:"):
            logger.warning("Skipping example due to generation error.")
            continue
            
        generated_texts.append(generated_text)
        
        # Extract prediction from generated text
        # Look for a number between 1 and 5
        prediction = None
        numbers = re.findall(r'\b[1-5]\b', generated_text)
        if numbers:
            try:
                prediction = int(numbers[0])
            except:
                pass
        
        # Default to median if no clear answer found
        if prediction is None:
            prediction = 3
        
        predictions.append(prediction)
        references.append(example["rating"])
    
    # Calculate metrics
    metric_results = {}
    
    if "accuracy" in metrics:
        accuracy = metrics["accuracy"].compute(predictions=predictions, references=references)
        metric_results["accuracy"] = accuracy["accuracy"]
    
    if "mse" in metrics:
        mse = metrics["mse"].compute(predictions=predictions, references=references)
        metric_results["mse"] = mse["mse"]
    
    # Calculate mean absolute error
    mae = sum(abs(p - r) for p, r in zip(predictions, references)) / len(predictions)
    metric_results["mae"] = mae
    
    # Prepare and return results
    results = {
        "metrics": metric_results,
        "predictions": predictions,
        "references": references,
        "generated_texts": generated_texts
    }
    
    return results


def evaluate_translation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metrics: Dict[str, Any],
    generation_params: Dict[str, Any],
    task_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate model on translation task (EN-PL).
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Translation dataset
        metrics: Evaluation metrics
        generation_params: Generation parameters
        task_info: Task metadata
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating translation (EN-PL)")
    
    predictions = []
    references = []
    generated_texts = []
    
    # Process each example
    for example in tqdm(dataset, desc="Evaluating translation"):
        # Create prompt
        prompt = create_task_specific_prompt("translation", example)
        
        # Generate prediction using the standardized helper
        generated_text = process_model_prediction(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            generation_params=generation_params
        )
        
        # Skip samples with errors
        if generated_text.startswith("ERROR:"):
            logger.warning("Skipping example due to generation error.")
            continue
            
        generated_texts.append(generated_text)
        
        # Extract translation more robustly
        translation = ""
        if prompt in generated_text:
            # Try to extract everything after the prompt
            translation = generated_text.split(prompt, 1)[1].strip()
        else:
            # If prompt isn't found intact, look for common markers in the output
            for marker in ["Translation:", "Tłumaczenie:", ":"]:
                if marker in generated_text:
                    parts = generated_text.split(marker, 1)
                    if len(parts) > 1:
                        translation = parts[1].strip()
                        break
            
            # If no markers found, log this issue and use the full output as fallback
            if not translation:
                logger.warning("Could not extract translation cleanly. Using full output.")
                translation = generated_text
        
        predictions.append(translation)
        references.append(example["pl"])
    
    # Calculate metrics
    metric_results = {}

    # Validate translations are not empty or malformed
    valid_predictions = []
    valid_references = []
    invalid_count = 0

    for pred, ref in zip(predictions, references):
        # Check if prediction is valid (not empty and contains Polish characters)
        if len(pred.strip()) > 5 and re.search(r'[ąęóśłżźćń]', pred, re.IGNORECASE):
            valid_predictions.append(pred)
            valid_references.append(ref)
        else:
            invalid_count += 1

    if invalid_count > 0:
        logger.warning(f"Excluded {invalid_count} invalid translations from metrics calculation")

    # Use valid predictions/references for metrics
    if len(valid_predictions) > 0:
        # BLEU expects a list of references for each prediction
        refs_for_bleu = [[ref] for ref in valid_references]
        bleu = metrics["bleu"].compute(predictions=valid_predictions, references=refs_for_bleu)
        metric_results["bleu"] = bleu["bleu"]
        
        # Record the percentage of valid translations
        metric_results["valid_translation_percentage"] = len(valid_predictions) / len(predictions) * 100
    else:
        logger.error("No valid translations found for metric calculation")
        metric_results["bleu"] = 0.0
        metric_results["valid_translation_percentage"] = 0.0

    if "bleu" in metrics:
        # BLEU expects a list of references for each prediction
        refs_for_bleu = [[ref] for ref in references]
        bleu = metrics["bleu"].compute(predictions=predictions, references=refs_for_bleu)
        metric_results["bleu"] = bleu["bleu"]
    
    if "rouge" in metrics:
        rouge = metrics["rouge"].compute(predictions=predictions, references=references)
        for key, value in rouge.items():
            metric_results[key] = value
    
    # Prepare and return results
    results = {
        "metrics": metric_results,
        "predictions": predictions,
        "references": references,
        "generated_texts": generated_texts
    }
    
    return results


def evaluate_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metrics: Dict[str, Any],
    generation_params: Dict[str, Any],
    task_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate model on Polish text generation task.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Generation dataset
        metrics: Evaluation metrics
        generation_params: Generation parameters
        task_info: Task metadata
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating Polish text generation")
    
    generated_texts = []
    prompts = []
    
    # Process each example
    for example in tqdm(dataset, desc="Evaluating generation"):
        prompt = example["prompt"]
        prompts.append(prompt)
        
        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Track generation time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # Longer output for generation
                **generation_params
            )
        
        generation_time = time.time() - start_time
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated text (everything after the prompt)
        text = generated_text.replace(prompt, "").strip()
        
        generated_texts.append({
            "text": text,
            "generation_time": generation_time,
            "token_count": len(outputs[0]) - len(inputs["input_ids"][0])
        })
    
    # For generation, we primarily focus on qualitative metrics
    # Calculate some quantitative metrics
    metric_results = {
        "avg_generation_time": sum(g["generation_time"] for g in generated_texts) / len(generated_texts),
        "avg_token_count": sum(g["token_count"] for g in generated_texts) / len(generated_texts),
        "tokens_per_second": sum(g["token_count"] for g in generated_texts) / 
                            sum(g["generation_time"] for g in generated_texts)
    }
    
    # Prepare and return results
    results = {
        "metrics": metric_results,
        "prompts": prompts,
        "generated_texts": generated_texts
    }
    
    return results


def evaluate_classification(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    metrics: Dict[str, Any],
    generation_params: Dict[str, Any],
    task_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Default evaluation for classification tasks.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset: Classification dataset
        metrics: Evaluation metrics
        generation_params: Generation parameters
        task_info: Task metadata
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating generic classification task")
    
    predictions = []
    references = []
    generated_texts = []
    
    # Get label key from dataset
    label_key = "label"
    if "keys" in task_info and any("label" in k for k in task_info["keys"]):
        label_key = next(k for k in task_info["keys"] if "label" in k)
    
    # Process each example
    for example in tqdm(dataset, desc="Evaluating classification"):
        # Create generic prompt
        prompt = f"Task: Classification\nText: {str(example)}\nLabel:"
        
        # Generate prediction using the standardized helper
        generated_text = process_model_prediction(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=10,
            generation_params=generation_params
        )
        
        # Skip samples with errors
        if generated_text.startswith("ERROR:"):
            logger.warning("Skipping example due to generation error.")
            continue
            
        generated_texts.append(generated_text)
        
        # For generic classification, just use the first number found in the output
        prediction = None
        numbers = re.findall(r'\b\d+\b', generated_text.replace(prompt, ""))
        if numbers:
            try:
                prediction = int(numbers[0])
            except:
                pass
        
        # Default if no clear answer found
        if prediction is None:
            prediction = 0
        
        predictions.append(prediction)
        references.append(example[label_key])
    
    # Calculate metrics
    metric_results = {}
    
    if "accuracy" in metrics:
        accuracy = metrics["accuracy"].compute(predictions=predictions, references=references)
        metric_results["accuracy"] = accuracy["accuracy"]
    
    if "f1" in metrics:
        # Only use F1 for binary classification
        if len(set(references)) <= 2:
            f1 = metrics["f1"].compute(predictions=predictions, references=references, average="binary")
            metric_results["f1"] = f1["f1"]
    
    # Prepare and return results
    results = {
        "metrics": metric_results,
        "predictions": predictions,
        "references": references,
        "generated_texts": generated_texts
    }
    
    return results


def save_evaluation_results(
    results: Dict[str, Any],
    results_dir: str,
    filename_prefix: str = "evaluation_results"
) -> None:
    """
    Save evaluation results to disk.
    
    Args:
        results: Evaluation results
        results_dir: Directory to save results
        filename_prefix: Prefix for saved files
    """
    import os
    import json
    from datetime import datetime
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results to JSON
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Convert any non-serializable objects to strings
    def convert_for_json(obj):
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        return str(obj)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=convert_for_json)
    
    logger.info(f"Saved evaluation results to {filepath}")


if __name__ == "__main__":
    # This section would typically include test code
    print("Evaluation module updated to support Polish GLUE (KLEJ) benchmark")
