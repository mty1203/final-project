#!/usr/bin/env python
"""
Train Hallucination Risk Probe

Trains a probe to predict hallucination risk from LLM hidden states.
Supports multiple probe architectures:
- Linear: Simple linear classifier
- MLP: Two-layer MLP with GELU activation
- Contrastive: Contrastive learning objective

Reference: Park et al., 2025 - "TSV: Steer LLM Latents for Hallucination Detection"
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from models.probe import LinearProbe, MLPProbe, ContrastiveProbe

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_training_data(
    model_name: str,
    dataset_name: str = "tqa",
    bleurt_threshold: float = 0.5,
    max_samples: int = 500
) -> Tuple[List[Dict], List[int]]:
    """
    Load training data with hallucination labels.
    
    Returns:
        examples: List of {question, answer} dicts
        labels: List of labels (1=truthful, 0=hallucinated)
    """
    # Try to load BLEURT scores
    bleurt_path = f"./ml_{dataset_name}_bleurt_score.npy"
    
    if not os.path.exists(bleurt_path):
        logger.warning(f"BLEURT scores not found, using hardcoded examples")
        # Fallback to hardcoded examples
        examples = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
            {"question": "What is the capital of Italy?", "answer": "Rome"},
            {"question": "What is the capital of Spain?", "answer": "Madrid"},
            {"question": "What is the capital of UK?", "answer": "London"},
            {"question": "What is the capital of France?", "answer": "London"},
            {"question": "What is the capital of Germany?", "answer": "Paris"},
            {"question": "What is the capital of Italy?", "answer": "Madrid"},
            {"question": "What is the capital of Spain?", "answer": "Berlin"},
            {"question": "What is the capital of UK?", "answer": "Rome"},
        ]
        labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        return examples, labels
    
    bleurt_scores = np.load(bleurt_path)
    
    # Load questions
    try:
        from datasets import load_dataset
        if dataset_name == "tqa":
            ds = load_dataset("truthful_qa", "generation", split="validation")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        logger.warning(f"Failed to load dataset: {e}")
        # Return minimal examples
        examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 2+2?", "answer": "5"},
        ]
        return examples, [1, 0]
    
    model_short = model_name.split("/")[-1]
    examples = []
    labels = []
    
    for i, (item, score) in enumerate(zip(ds, bleurt_scores)):
        if len(examples) >= max_samples:
            break
            
        # Try to load saved answer
        answer_path = f"./save_for_eval/{dataset_name}_hal_det/answers/most_likely_hal_det_{model_short}_{dataset_name}_answers_index_{i}.npy"
        
        if os.path.exists(answer_path):
            answers = np.load(answer_path, allow_pickle=True)
            answer = answers[0] if len(answers) > 0 else ""
        else:
            continue
        
        examples.append({
            "question": item["question"],
            "answer": answer
        })
        labels.append(1 if score > bleurt_threshold else 0)
    
    logger.info(f"Loaded {len(examples)} examples, {sum(labels)} truthful, {len(labels) - sum(labels)} hallucinated")
    return examples, labels


def extract_features(
    model: nn.Module,
    tokenizer,
    examples: List[Dict],
    device: torch.device,
    layer_id: int = -1,
    batch_size: int = 8
) -> torch.Tensor:
    """
    Extract hidden state features from examples.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        examples: List of {question, answer} dicts
        device: Target device
        layer_id: Which layer to extract (-1 = last)
        batch_size: Batch size for extraction
    
    Returns:
        features: [num_examples, hidden_size]
    """
    features = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(examples), batch_size), desc="Extracting features"):
            batch = examples[i:i+batch_size]
            
            # Format prompts
            prompts = [
                f"Answer the question concisely.\nQ: {ex['question']}\nA: {ex['answer']}"
                for ex in batch
            ]
            
            # Tokenize
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs, output_hidden_states=True)
            
            # Get hidden states from specified layer
            hidden = outputs.hidden_states[layer_id]  # [batch, seq, hidden]
            
            # Get last non-padding token for each example
            attention_mask = inputs.attention_mask
            seq_lengths = attention_mask.sum(dim=1) - 1
            
            batch_features = []
            for j, seq_len in enumerate(seq_lengths):
                feat = hidden[j, seq_len, :].cpu().float()
                batch_features.append(feat)
            
            features.extend(batch_features)
    
    return torch.stack(features)


def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    hidden_size: int,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 0.01
) -> LinearProbe:
    """Train a linear probe."""
    probe = LinearProbe(hidden_size).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    features = features.to(device)
    labels = labels.to(device)
    
    probe.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = probe(features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            acc = ((preds > 0.5) == labels).float().mean()
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.4f}")
    
    return probe


def train_mlp_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    hidden_size: int,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.001,
    weight_decay: float = 0.01
) -> MLPProbe:
    """Train an MLP probe."""
    probe = MLPProbe(hidden_size, intermediate_size=256).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    features = features.to(device)
    labels = labels.to(device)
    
    probe.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = probe(features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            acc = ((preds > 0.5) == labels).float().mean()
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.4f}")
    
    return probe


def train_contrastive_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    hidden_size: int,
    device: torch.device,
    epochs: int = 200,
    lr: float = 0.001,
    temperature: float = 0.1
) -> ContrastiveProbe:
    """Train a contrastive probe with InfoNCE loss."""
    probe = ContrastiveProbe(hidden_size, temperature=temperature).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)
    
    features = features.to(device)
    labels = labels.to(device)
    
    probe.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Get embeddings
        embeddings = probe.get_embeddings(features)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Create label matrix (same class = positive)
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        
        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Mask out self-similarity
        mask = torch.eye(len(labels), device=device)
        log_prob = log_prob * (1 - mask)
        label_matrix = label_matrix * (1 - mask)
        
        # Average over positives
        num_positives = label_matrix.sum(dim=1).clamp(min=1)
        contrastive_loss = -(log_prob * label_matrix).sum(dim=1) / num_positives
        contrastive_loss = contrastive_loss.mean()
        
        # Classification loss
        preds = probe(features)
        cls_loss = F.binary_cross_entropy(preds, labels)
        
        loss = contrastive_loss + cls_loss
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            acc = ((preds > 0.5) == labels).float().mean()
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.4f}")
    
    return probe


def main():
    parser = argparse.ArgumentParser(description="Train hallucination risk probe")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--dataset_name", type=str, default="tqa")
    parser.add_argument("--bleurt_threshold", type=float, default=0.5)
    parser.add_argument("--probe_type", type=str, default="linear",
                       choices=["linear", "mlp", "contrastive"])
    parser.add_argument("--layer_id", type=int, default=-1,
                       help="Layer to extract features from (-1 = last)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    model.eval()
    
    hidden_size = model.config.hidden_size
    
    # Load training data
    logger.info("Loading training data...")
    examples, labels = load_training_data(
        args.model_name,
        args.dataset_name,
        args.bleurt_threshold,
        args.max_samples
    )
    
    # Extract features
    logger.info(f"Extracting features from layer {args.layer_id}...")
    features = extract_features(
        model, tokenizer, examples, device, args.layer_id
    )
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Label distribution: {sum(labels)} truthful, {len(labels) - sum(labels)} hallucinated")
    
    # Free up memory
    del model
    torch.cuda.empty_cache()
    
    # Train probe
    logger.info(f"Training {args.probe_type} probe...")
    if args.probe_type == "linear":
        probe = train_linear_probe(features, labels_tensor, hidden_size, device, args.epochs, args.lr)
    elif args.probe_type == "mlp":
        probe = train_mlp_probe(features, labels_tensor, hidden_size, device, args.epochs, args.lr)
    elif args.probe_type == "contrastive":
        probe = train_contrastive_probe(features, labels_tensor, hidden_size, device, args.epochs, args.lr)
    
    # Evaluate
    probe.eval()
    with torch.no_grad():
        preds = probe(features.to(device))
        acc = ((preds > 0.5) == labels_tensor.to(device)).float().mean()
        
        # Compute AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, preds.cpu().numpy())
    
    logger.info(f"Final Accuracy: {acc:.4f}")
    logger.info(f"Final AUC: {auc:.4f}")
    
    # Save probe
    model_short = args.model_name.split("/")[-1]
    output_path = os.path.join(
        args.output_dir,
        f"{model_short}_{args.dataset_name}_{args.probe_type}_probe.pt"
    )
    
    torch.save({
        "state_dict": probe.state_dict(),
        "probe_type": args.probe_type,
        "hidden_size": hidden_size,
        "layer_id": args.layer_id,
        "model_name": args.model_name,
        "accuracy": acc.item(),
        "auc": auc
    }, output_path)
    
    logger.info(f"Probe saved to {output_path}")


if __name__ == "__main__":
    main()

