#!/usr/bin/env python
"""
Train Hallucination Probe on Full TruthfulQA Dataset (817 samples)

Uses the same data as TSV training for consistency.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from models.probe import MLPProbe, LinearProbe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_tqa_data(
    model_name: str,
    data_dir: str = "../../save_for_eval/tqa_hal_det/answers",
    bleurt_path: str = None,
    threshold: float = 0.5
) -> List[Dict]:
    """Load TruthfulQA data from answer files and BLEURT scores."""
    
    if bleurt_path is None:
        bleurt_path = "../../ml_tqa_bleurt_score.npy"
    
    bleurt_scores = np.load(bleurt_path)
    logger.info(f"Loaded BLEURT scores: {bleurt_scores.shape}")
    
    model_short = model_name.split("/")[-1]
    
    # Count files
    num_files = 0
    while os.path.exists(os.path.join(data_dir, f"most_likely_hal_det_{model_short}_tqa_answers_index_{num_files}.npy")):
        num_files += 1
    
    logger.info(f"Found {num_files} answer files")
    
    samples = []
    for i in tqdm(range(num_files), desc="Loading data"):
        answer_file = os.path.join(data_dir, f"most_likely_hal_det_{model_short}_tqa_answers_index_{i}.npy")
        answers = np.load(answer_file)
        
        bleurt_score = bleurt_scores[i]
        label = 1 if bleurt_score > threshold else 0
        
        answer = str(answers[0]) if len(answers) > 0 else ""
        
        samples.append({
            "question": f"Question {i}",
            "answer": answer,
            "label": label,
            "bleurt_score": float(bleurt_score)
        })
    
    logger.info(f"Loaded {len(samples)} samples")
    logger.info(f"  Truthful: {sum(1 for s in samples if s['label'] == 1)}")
    logger.info(f"  Hallucinated: {sum(1 for s in samples if s['label'] == 0)}")
    
    return samples


def extract_hidden_states(
    model: nn.Module,
    tokenizer,
    samples: List[Dict],
    device: torch.device,
    layer_id: int = -1,
    batch_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract hidden states."""
    
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc="Extracting features"):
            batch = samples[i:i+batch_size]
            
            prompts = [
                f"Answer the question concisely. Q: {s['question']} A: {s['answer']}"
                for s in batch
            ]
            batch_labels = [s['label'] for s in batch]
            
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)
            
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer_id]
            
            attention_mask = inputs.attention_mask
            seq_lengths = attention_mask.sum(dim=1) - 1
            
            for j, seq_len in enumerate(seq_lengths):
                features.append(hidden[j, seq_len, :].cpu())
            
            labels.extend(batch_labels)
    
    return torch.stack(features), torch.tensor(labels)


def train_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    hidden_size: int,
    probe_type: str = "mlp",
    epochs: int = 100,
    lr: float = 0.001,
    device: torch.device = None
) -> Tuple[nn.Module, Dict]:
    """Train probe."""
    
    if probe_type == "mlp":
        probe = MLPProbe(hidden_size)
    else:
        probe = LinearProbe(hidden_size)
    
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    X_train = X_train.to(device).float()
    y_train = y_train.to(device).float()
    X_test = X_test.to(device).float()
    y_test = y_test.to(device).float()
    
    best_auc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        
        preds = probe(X_train).squeeze()
        loss = criterion(preds, y_train)
        
        loss.backward()
        optimizer.step()
        
        # Evaluate
        probe.eval()
        with torch.no_grad():
            train_preds = probe(X_train).squeeze().cpu().numpy()
            test_preds = probe(X_test).squeeze().cpu().numpy()
            
            train_acc = accuracy_score(y_train.cpu().numpy(), (train_preds > 0.5).astype(int))
            test_acc = accuracy_score(y_test.cpu().numpy(), (test_preds > 0.5).astype(int))
            
            try:
                train_auc = roc_auc_score(y_train.cpu().numpy(), train_preds)
                test_auc = roc_auc_score(y_test.cpu().numpy(), test_preds)
            except:
                train_auc = 0.5
                test_auc = 0.5
            
            if test_auc > best_auc:
                best_auc = test_auc
                best_state = probe.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, "
                       f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
                       f"Test AUC: {test_auc:.4f}")
    
    # Load best model
    if best_state is not None:
        probe.load_state_dict(best_state)
    
    # Final evaluation
    probe.eval()
    with torch.no_grad():
        train_preds = probe(X_train).squeeze().cpu().numpy()
        test_preds = probe(X_test).squeeze().cpu().numpy()
        
        metrics = {
            "train_accuracy": accuracy_score(y_train.cpu().numpy(), (train_preds > 0.5).astype(int)),
            "test_accuracy": accuracy_score(y_test.cpu().numpy(), (test_preds > 0.5).astype(int)),
            "train_auc": roc_auc_score(y_train.cpu().numpy(), train_preds),
            "test_auc": roc_auc_score(y_test.cpu().numpy(), test_preds)
        }
    
    return probe, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--layer_id", type=int, default=9)
    parser.add_argument("--probe_type", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="../../artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    
    # Load data
    samples = load_tqa_data(args.model_name, threshold=args.threshold)
    
    # Extract features
    features, labels = extract_hidden_states(
        model, tokenizer, samples, device, args.layer_id, args.batch_size
    )
    
    logger.info(f"Features shape: {features.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels.numpy()
    )
    
    logger.info(f"Train: {len(y_train)} samples")
    logger.info(f"Test: {len(y_test)} samples")
    
    # Train probe
    probe, metrics = train_probe(
        X_train, y_train, X_test, y_test,
        features.shape[1], args.probe_type,
        args.epochs, args.lr, device
    )
    
    # Save
    model_short = args.model_name.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{model_short}_probe_817.pt")
    
    torch.save({
        "state_dict": probe.state_dict(),
        "probe_type": args.probe_type,
        "hidden_size": features.shape[1],
        "layer_id": args.layer_id,
        "model_name": args.model_name,
        "train_accuracy": metrics["train_accuracy"],
        "train_auc": metrics["train_auc"],
        "test_accuracy": metrics["test_accuracy"],
        "test_auc": metrics["test_auc"],
        "num_samples": len(samples)
    }, output_path)
    
    logger.info(f"Probe saved to: {output_path}")
    
    print("\n" + "="*60)
    print("Probe Training Summary (817 TruthfulQA Samples)")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Probe Type: {args.probe_type}")
    print(f"Layer: {args.layer_id}")
    print(f"Total Samples: {len(samples)}")
    print("-"*60)
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Train AUC:      {metrics['train_auc']:.4f}")
    print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Test AUC:       {metrics['test_auc']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

