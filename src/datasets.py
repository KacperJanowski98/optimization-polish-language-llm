"""
Dataset preparation module for Polish language model evaluation using Polish GLUE.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any
from datasets import load_dataset, Dataset, DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Polish GLUE (KLEJ) benchmark datasets
POLISH_GLUE_DATASETS = {
    "cdsc-e": {
        "description": "Compositional Distributional Semantics Corpus - Entailment task",
        "task_type": "classification",
        "metric": "accuracy",
        "keys": ["sentence_A", "sentence_B", "entailment_judgment"]
    },
    "cdsc-r": {
        "description": "Compositional Distributional Semantics Corpus - Relatedness task",
        "task_type": "regression",
        "metric": "spearman_correlation",
        "keys": ["sentence_A", "sentence_B", "relatedness_score"]
    },
    "cbd": {
        "description": "Cyberbullying Detection dataset",
        "task_type": "classification",
        "metric": "f1",
        "keys": ["text", "label"]
    },
    "polemo2-in": {
        "description": "PolEmo2.0 - Sentiment analysis (in-domain)",
        "task_type": "classification",
        "metric": "accuracy",
        "keys": ["text", "label"]
    },
    "polemo2-out": {
        "description": "PolEmo2.0 - Sentiment analysis (out-of-domain)",
        "task_type": "classification",
        "metric": "accuracy",
        "keys": ["text", "label"]
    },
    "dyk": {
        "description": "Did You Know - Genuine question detection",
        "task_type": "classification",
        "metric": "accuracy",
        "keys": ["question", "label"]
    },
    "psc": {
        "description": "Polish Summaries Corpus",
        "task_type": "classification",
        "metric": "accuracy",
        "keys": ["extract_text", "summary_text", "label"]
    },
    "ar": {
        "description": "Allegro Reviews - Sentiment analysis",
        "task_type": "classification",
        "metric": "accuracy",
        "keys": ["text", "rating"]
    }
}

def prepare_polish_datasets(
    tasks: List[str] = ["klej"],
    klej_datasets: List[str] = None,
    max_samples: int = 100,
    data_dir: Optional[str] = None
) -> Dict[str, Union[Dataset, DatasetDict, Dict[str, Any]]]:
    """
    Prepare datasets for Polish language evaluation using Polish GLUE (KLEJ).
    
    Args:
        tasks (List[str]): List of tasks to prepare datasets for. Use "klej" for Polish GLUE.
        klej_datasets (List[str]): Specific KLEJ datasets to load (default: all)
        max_samples (int): Maximum number of samples per dataset
        data_dir (Optional[str]): Directory for caching or loading local datasets
        
    Returns:
        Dict[str, Dict]: Dictionary of task-specific datasets with metadata
    """
    datasets = {}
    
    # Set data directory to project default if not specified
    if data_dir is None:
        # Assuming this script is in src/ and data/ is at the same level
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
    
    # Process each requested task
    for task in tasks:
        if task.lower() == "klej" or task.lower() == "polish_glue":
            logger.info("Preparing Polish GLUE (KLEJ) datasets")
            klej_data = load_polish_glue(klej_datasets, max_samples, data_dir)
            datasets.update(klej_data)
        elif task == "translation":
            datasets["translation"] = prepare_translation_dataset(max_samples, data_dir)
        elif task == "generation":
            datasets["generation"] = prepare_generation_dataset(max_samples, data_dir)
        else:
            logger.warning(f"Unknown task: {task}, skipping")
    
    return datasets

def load_polish_glue(
    dataset_names: Optional[List[str]] = None,
    max_samples: int = 100,
    data_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load datasets from the Polish GLUE (KLEJ) benchmark.
    
    Args:
        dataset_names: List of specific KLEJ dataset names to load (if None, load all)
        max_samples: Maximum number of samples per dataset
        data_dir: Directory for caching datasets
        
    Returns:
        Dict[str, Dict]: Dictionary of KLEJ datasets with metadata
    """
    klej_datasets = {}
    
    # If no specific datasets are requested, load all
    if dataset_names is None:
        dataset_names = list(POLISH_GLUE_DATASETS.keys())
    
    # Load each requested dataset
    for name in dataset_names:
        if name not in POLISH_GLUE_DATASETS:
            logger.warning(f"Unknown KLEJ dataset: {name}, skipping")
            continue
        
        logger.info(f"Loading KLEJ dataset: {name}")
        
        try:
            # Load the dataset from Hugging Face
            dataset = load_dataset(
                "allegro/klej",
                name,
                split="train",
                cache_dir=data_dir
            )
            
            # Take a subset if needed
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            # Format dataset info
            info = POLISH_GLUE_DATASETS[name]
            klej_datasets[f"klej_{name}"] = {
                "dataset": dataset,
                "description": info["description"],
                "task_type": info["task_type"],
                "metric": info["metric"],
                "keys": info["keys"]
            }
            
            logger.info(f"Successfully loaded {name} with {len(dataset)} samples")
            
        except Exception as e:
            logger.error(f"Failed to load KLEJ dataset {name}: {str(e)}")
            
            # Try to create a fallback minimal dataset if loading fails
            if name == "cdsc-e":
                klej_datasets[f"klej_{name}"] = create_fallback_cdsc_e()
            elif name == "polemo2-in":
                klej_datasets[f"klej_{name}"] = create_fallback_polemo()
    
    return klej_datasets

def create_task_specific_prompt(task: str, example: Dict[str, Any]) -> str:
    """
    Create a task-specific prompt for each dataset/task.
    
    Args:
        task: The task name (e.g., "klej_cdsc-e")
        example: A data example from the dataset
        
    Returns:
        str: Formatted prompt for the model
    """
    # Extract task type from task name if it's a KLEJ task
    base_task = task.replace("klej_", "") if task.startswith("klej_") else task
    
    # Format prompts based on task
    if base_task == "cdsc-e":
        prompt = (f"Zdecyduj, czy drugie zdanie wynika z pierwszego. Odpowiedz ENTAILMENT, "
                 f"CONTRADICTION lub NEUTRAL.\n\n"
                 f"Zdanie 1: {example['sentence_A']}\n"
                 f"Zdanie 2: {example['sentence_B']}\n"
                 f"Odpowiedź:")
    
    elif base_task == "cdsc-r":
        prompt = (f"Oceń podobieństwo semantyczne poniższych zdań w skali od 0 do 5, "
                 f"gdzie 0 oznacza brak podobieństwa, a 5 oznacza bardzo wysokie podobieństwo.\n\n"
                 f"Zdanie 1: {example['sentence_A']}\n"
                 f"Zdanie 2: {example['sentence_B']}\n"
                 f"Ocena podobieństwa (0-5):")
    
    elif base_task == "cbd":
        prompt = (f"Poniższy tekst zawiera cyberprzemoc czy nie? Odpowiedz TAK lub NIE.\n\n"
                 f"Tekst: {example['text']}\n"
                 f"Odpowiedź:")
    
    elif base_task.startswith("polemo2"):
        prompt = (f"Określ wydźwięk emocjonalny poniższego tekstu. "
                 f"Wybierz jedną z opcji: POZYTYWNY, NEGATYWNY, NEUTRALNY lub MIESZANY.\n\n"
                 f"Tekst: {example['text']}\n"
                 f"Wydźwięk:")
    
    elif base_task == "dyk":
        prompt = (f"Czy poniższe pytanie jest szczere i zmierza do uzyskania informacji? Odpowiedz TAK lub NIE.\n\n"
                 f"Pytanie: {example['question']}\n"
                 f"Odpowiedź:")
    
    elif base_task == "psc":
        prompt = (f"Czy poniższy tekst jest poprawnym streszczeniem artykułu? Odpowiedz TAK lub NIE.\n\n"
                 f"Artykuł: {example['extract_text']}\n"
                 f"Streszczenie: {example['summary_text']}\n"
                 f"Odpowiedź:")
    
    elif base_task == "ar":
        prompt = (f"Oceń wydźwięk poniższej recenzji produktu w skali od 1 do 5, "
                 f"gdzie 1 oznacza bardzo negatywną ocenę, a 5 oznacza bardzo pozytywną ocenę.\n\n"
                 f"Recenzja: {example['text']}\n"
                 f"Ocena (1-5):")
    
    elif task == "translation":
        prompt = f"Przetłumacz poniższy tekst z angielskiego na polski:\n\n{example['en']}"
    
    elif task == "generation":
        prompt = example['prompt']
    
    else:
        # Generic fallback prompt
        prompt = f"Zadanie: {task}\nDane: {str(example)}\nOdpowiedź:"
    
    return prompt

def create_fallback_cdsc_e() -> Dict:
    """
    Create a minimal fallback dataset for CDSC-E if loading fails.
    
    Returns:
        Dict: Dataset info with minimal examples
    """
    logger.info("Creating fallback CDSC-E dataset")
    
    examples = [
        {
            "sentence_A": "Mężczyzna gra na gitarze.",
            "sentence_B": "Ktoś używa instrumentu muzycznego.",
            "entailment_judgment": 0  # ENTAILMENT
        },
        {
            "sentence_A": "Dzieci bawią się na placu zabaw.",
            "sentence_B": "Dzieci śpią w łóżkach.",
            "entailment_judgment": 1  # CONTRADICTION
        },
        {
            "sentence_A": "Kobieta czyta książkę.",
            "sentence_B": "Książka ma czerwoną okładkę.",
            "entailment_judgment": 2  # NEUTRAL
        }
    ]
    
    dataset = Dataset.from_dict({
        "sentence_A": [ex["sentence_A"] for ex in examples],
        "sentence_B": [ex["sentence_B"] for ex in examples],
        "entailment_judgment": [ex["entailment_judgment"] for ex in examples]
    })
    
    return {
        "dataset": dataset,
        "description": "Fallback CDSC-E dataset",
        "task_type": "classification",
        "metric": "accuracy",
        "keys": ["sentence_A", "sentence_B", "entailment_judgment"]
    }

def create_fallback_polemo() -> Dict:
    """
    Create a minimal fallback dataset for PolEmo if loading fails.
    
    Returns:
        Dict: Dataset info with minimal examples
    """
    logger.info("Creating fallback PolEmo dataset")
    
    examples = [
        {
            "text": "Ten film był wspaniały, bardzo mi się podobał!",
            "label": 0  # POSITIVE
        },
        {
            "text": "Nie polecam tego produktu, jakość jest fatalna.",
            "label": 1  # NEGATIVE
        },
        {
            "text": "Wczoraj kupiłem nowy telefon.",
            "label": 2  # NEUTRAL
        },
        {
            "text": "Produkt działa dobrze, ale jest dość drogi i ma pewne wady.",
            "label": 3  # MIXED
        }
    ]
    
    dataset = Dataset.from_dict({
        "text": [ex["text"] for ex in examples],
        "label": [ex["label"] for ex in examples]
    })
    
    return {
        "dataset": dataset,
        "description": "Fallback PolEmo dataset",
        "task_type": "classification",
        "metric": "accuracy",
        "keys": ["text", "label"]
    }

def prepare_translation_dataset(max_samples: int = 100, data_dir: str = None) -> Dict[str, Any]:
    """
    Prepare a dataset for English-Polish translation evaluation.
    
    Args:
        max_samples (int): Maximum number of samples
        data_dir (str): Directory for caching/saving data
        
    Returns:
        Dict: Translation dataset with metadata
    """
    logger.info("Loading translation dataset (English-Polish)")
    
    try:
        # Try to load OPUS Books dataset (English-Polish)
        dataset = load_dataset(
            "opus_books", 
            "en-pl",
            split="train",
            cache_dir=data_dir
        )
        
        # Format the dataset for our use
        dataset = dataset.map(lambda x: {
            "en": x["translation"]["en"],
            "pl": x["translation"]["pl"]
        })
        
        # Remove the original translation column
        dataset = dataset.remove_columns(["id", "translation"])
        
        # Take a subset of the data
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        logger.info(f"Loaded translation dataset with {len(dataset)} samples")
    
    except Exception as e:
        logger.error(f"Failed to load translation dataset: {str(e)}")
        
        # Fallback to creating a minimal dataset
        logger.info("Creating minimal fallback translation dataset")
        
        # Sample English-Polish translation pairs
        minimal_data = {
            "en": [
                "Hello, how are you?",
                "What is your name?",
                "I love learning languages.",
                "The weather is beautiful today.",
                "Where is the nearest restaurant?",
                "What time is it?",
                "I need to buy some groceries.",
                "Can you help me, please?",
                "The book is on the table.",
                "I will see you tomorrow."
            ],
            "pl": [
                "Cześć, jak się masz?",
                "Jak masz na imię?",
                "Uwielbiam uczyć się języków.",
                "Dzisiaj jest piękna pogoda.",
                "Gdzie jest najbliższa restauracja?",
                "Która jest godzina?",
                "Muszę kupić trochę artykułów spożywczych.",
                "Czy możesz mi pomóc, proszę?",
                "Książka jest na stole.",
                "Zobaczę cię jutro."
            ]
        }
        
        dataset = Dataset.from_dict(minimal_data)
    
    # Return with metadata similar to KLEJ datasets
    return {
        "dataset": dataset,
        "description": "English-Polish translation dataset",
        "task_type": "generation",
        "metric": "bleu",
        "keys": ["en", "pl"]
    }

def prepare_generation_dataset(max_samples: int = 100, data_dir: str = None) -> Dict[str, Any]:
    """
    Prepare a dataset for Polish text generation evaluation.
    
    Args:
        max_samples (int): Maximum number of samples
        data_dir (str): Directory for caching/saving data
        
    Returns:
        Dict: Polish text generation dataset with metadata
    """
    logger.info("Creating a Polish text generation prompts dataset")
    
    generation_prompts = [
        "Opisz zalety uczenia się języka polskiego.",
        "Wyjaśnij, dlaczego sztuczna inteligencja jest ważna we współczesnym świecie.",
        "Napisz krótkie opowiadanie o przygodzie w górach.",
        "Podaj przepis na tradycyjne polskie danie.",
        "Wyjaśnij, jak działa algorytm wyszukiwania binarnego.",
        "Opisz swoje wymarzone wakacje w Polsce.",
        "Napisz list motywacyjny do wymarzonej pracy.",
        "Przygotuj instrukcję obsługi prostego urządzenia.",
        "Napisz recenzję ostatnio przeczytanej książki.",
        "Opisz główne wyzwania związane z ochroną środowiska."
    ]
    
    # Expand with more examples if needed
    generation_prompts = generation_prompts * ((max_samples // len(generation_prompts)) + 1)
    generation_prompts = generation_prompts[:max_samples]
    
    # Convert to dataset
    dataset = Dataset.from_dict({
        "prompt": generation_prompts
    })
    
    logger.info(f"Created generation dataset with {len(dataset)} samples")
    
    # Return with metadata similar to KLEJ datasets
    return {
        "dataset": dataset,
        "description": "Polish text generation prompts",
        "task_type": "generation",
        "metric": "qualitative",
        "keys": ["prompt"]
    }

if __name__ == "__main__":
    # Test the dataset preparation
    print("Testing Polish GLUE dataset preparation...")
    
    # Load just a couple of KLEJ datasets
    datasets = prepare_polish_datasets(
        tasks=["klej"],
        klej_datasets=["cdsc-e", "polemo2-in"],
        max_samples=5
    )
    
    for task, dataset_info in datasets.items():
        print(f"\nTask: {task}")
        print(f"Description: {dataset_info['description']}")
        print(f"Task type: {dataset_info['task_type']}")
        print(f"Metric: {dataset_info['metric']}")
        print(f"Dataset size: {len(dataset_info['dataset'])}")
        
        # Show a sample with a generated prompt
        sample = dataset_info['dataset'][0]
        print("\nSample item:")
        for key in dataset_info['keys']:
            print(f"  {key}: {sample[key]}")
        
        # Generate a task-specific prompt
        prompt = create_task_specific_prompt(task, sample)
        print("\nGenerated prompt:")
        print(prompt)
