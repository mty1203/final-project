#!/usr/bin/env python
"""
Train Contrastive Activation Addition (CAA) Vectors

CAA computes the mean activation difference between truthful and hallucinated
examples at each layer. This serves as a baseline comparison to learned TSV vectors.

Reference: Rimsky et al., 2024 - "Contrastive Activation Addition"
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_truthfulqa_data() -> List[Dict]:
    """Load TruthfulQA data with truthful/hallucinated labels."""
    # Hardcoded examples for demonstration
    # In practice, load from dataset with BLEURT scores
    examples = [
        # Truthful examples (label=1)
        {"question": "What is the capital of France?", 
         "answer": "Paris is the capital of France.",
         "label": 1},
        {"question": "Who wrote Romeo and Juliet?",
         "answer": "William Shakespeare wrote Romeo and Juliet.",
         "label": 1},
        {"question": "What is the speed of light?",
         "answer": "The speed of light is approximately 299,792 kilometers per second.",
         "label": 1},
        {"question": "Who was the first person on the Moon?",
         "answer": "Neil Armstrong was the first person to walk on the Moon in 1969.",
         "label": 1},
        {"question": "What is the chemical formula for water?",
         "answer": "The chemical formula for water is H2O.",
         "label": 1},
        
        # Hallucinated examples (label=0)
        {"question": "What is the capital of France?",
         "answer": "London is the capital of France.",
         "label": 0},
        {"question": "Who wrote Romeo and Juliet?",
         "answer": "Charles Dickens wrote Romeo and Juliet.",
         "label": 0},
        {"question": "What is the speed of light?",
         "answer": "The speed of light is 1000 miles per hour.",
         "label": 0},
        {"question": "Who was the first person on the Moon?",
         "answer": "Buzz Aldrin was the first person on the Moon.",
         "label": 0},
        {"question": "What is the chemical formula for water?",
         "answer": "The chemical formula for water is CO2.",
         "label": 0},
    ]
    return examples


def load_from_saved_answers(
    model_name: str,
    dataset_name: str = "tqa",
    bleurt_threshold: float = 0.5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load truthful and hallucinated examples from saved answer files.
    
    Returns:
        truthful_examples: List of examples with BLEURT > threshold
        hallucinated_examples: List of examples with BLEURT <= threshold
    """
    # Try to load BLEURT scores
    bleurt_path = f"./ml_{dataset_name}_bleurt_score.npy"
    if not os.path.exists(bleurt_path):
        logger.warning(f"BLEURT scores not found at {bleurt_path}, using hardcoded examples")
        examples = load_truthfulqa_data()
        truthful = [e for e in examples if e["label"] == 1]
        hallucinated = [e for e in examples if e["label"] == 0]
        return truthful, hallucinated
    
    bleurt_scores = np.load(bleurt_path)
    
    # Load questions from dataset
    try:
        from datasets import load_dataset
        if dataset_name == "tqa":
            ds = load_dataset("truthful_qa", "generation", split="validation")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        logger.warning(f"Failed to load dataset: {e}, using hardcoded examples")
        examples = load_truthfulqa_data()
        truthful = [e for e in examples if e["label"] == 1]
        hallucinated = [e for e in examples if e["label"] == 0]
        return truthful, hallucinated
    
    # Extract model short name
    model_short = model_name.split("/")[-1]
    
    truthful = []
    hallucinated = []
    
    for i, (item, score) in enumerate(zip(ds, bleurt_scores)):
        # Try to load saved answer
        answer_path = f"./save_for_eval/{dataset_name}_hal_det/answers/most_likely_hal_det_{model_short}_{dataset_name}_answers_index_{i}.npy"
        
        if os.path.exists(answer_path):
            answers = np.load(answer_path, allow_pickle=True)
            answer = answers[0] if len(answers) > 0 else ""
        else:
            answer = ""
        
        example = {
            "question": item["question"],
            "answer": answer,
            "score": score
        }
        
        if score > bleurt_threshold:
            example["label"] = 1
            truthful.append(example)
        else:
            example["label"] = 0
            hallucinated.append(example)
    
    logger.info(f"Loaded {len(truthful)} truthful and {len(hallucinated)} hallucinated examples")
    return truthful, hallucinated


def extract_activations(
    model: nn.Module,
    tokenizer,
    examples: List[Dict],
    device: torch.device,
    layer_ids: Optional[List[int]] = None
) -> Dict[int, torch.Tensor]:
    """
    Extract mean activations for each layer from examples.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        examples: List of examples with question/answer
        device: Target device
        layer_ids: Layers to extract (None = all)
    
    Returns:
        Dictionary mapping layer_id -> mean activation tensor
    """
    num_layers = model.config.num_hidden_layers
    if layer_ids is None:
        layer_ids = list(range(num_layers))
    
    # Accumulate activations
    layer_activations = {lid: [] for lid in layer_ids}
    
    model.eval()
    with torch.no_grad():
        for example in tqdm(examples, desc="Extracting activations"):
            # Format prompt
            prompt = f"Answer the question concisely.\nQ: {example['question']}\nA: {example['answer']}"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract last token's hidden state from each layer
            for lid in layer_ids:
                hidden = outputs.hidden_states[lid][:, -1, :]  # [1, hidden_size]
                layer_activations[lid].append(hidden.cpu())
    
    # Compute mean for each layer
    mean_activations = {}
    for lid in layer_ids:
        stacked = torch.cat(layer_activations[lid], dim=0)  # [num_examples, hidden_size]
        mean_activations[lid] = stacked.mean(dim=0)  # [hidden_size]
    
    return mean_activations


def compute_caa_vectors(
    truthful_activations: Dict[int, torch.Tensor],
    hallucinated_activations: Dict[int, torch.Tensor]
) -> Dict[int, torch.Tensor]:
    """
    Compute CAA vectors as the difference between truthful and hallucinated activations.
    
    CAA vector = mean(truthful) - mean(hallucinated)
    
    This vector points from hallucinated space toward truthful space.
    """
    caa_vectors = {}
    
    for lid in truthful_activations.keys():
        truthful = truthful_activations[lid]
        hallucinated = hallucinated_activations[lid]
        
        # Direction: hallucinated -> truthful
        caa_vector = truthful - hallucinated
        caa_vectors[lid] = caa_vector
        
        logger.info(f"Layer {lid}: CAA vector norm = {caa_vector.norm():.4f}")
    
    return caa_vectors


def main():
    parser = argparse.ArgumentParser(description="Train CAA vectors")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--dataset_name", type=str, default="tqa")
    parser.add_argument("--bleurt_threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--layer_ids", type=str, default=None,
                       help="Comma-separated layer IDs (default: all)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()
    
    # Parse layer IDs
    if args.layer_ids:
        layer_ids = [int(x) for x in args.layer_ids.split(",")]
    else:
        layer_ids = None  # all layers
    
    # Load data
    logger.info("Loading truthful and hallucinated examples...")
    truthful_examples, hallucinated_examples = load_from_saved_answers(
        args.model_name, args.dataset_name, args.bleurt_threshold
    )
    
    # Balance the datasets
    min_size = min(len(truthful_examples), len(hallucinated_examples))
    truthful_examples = truthful_examples[:min_size]
    hallucinated_examples = hallucinated_examples[:min_size]
    logger.info(f"Using {min_size} examples from each class")
    
    # Extract activations
    logger.info("Extracting activations from truthful examples...")
    truthful_activations = extract_activations(
        model, tokenizer, truthful_examples, device, layer_ids
    )
    
    logger.info("Extracting activations from hallucinated examples...")
    hallucinated_activations = extract_activations(
        model, tokenizer, hallucinated_examples, device, layer_ids
    )
    
    # Compute CAA vectors
    logger.info("Computing CAA vectors...")
    caa_vectors = compute_caa_vectors(truthful_activations, hallucinated_activations)
    
    # Save vectors
    model_short = args.model_name.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{model_short}_{args.dataset_name}_caa.pt")
    
    torch.save({
        "caa_vectors": caa_vectors,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "bleurt_threshold": args.bleurt_threshold,
        "num_truthful": len(truthful_examples),
        "num_hallucinated": len(hallucinated_examples)
    }, output_path)
    
    logger.info(f"CAA vectors saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CAA Vector Summary")
    print("="*60)
    for lid, vec in caa_vectors.items():
        print(f"  Layer {lid:2d}: norm={vec.norm():.4f}, mean={vec.mean():.6f}, std={vec.std():.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

