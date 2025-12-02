#!/usr/bin/env python
"""
Train TSV using Logistic Regression on Full TruthfulQA Dataset (817 samples)

Uses the pre-generated answers and BLEURT scores from the original TSV pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_tqa_data(
    model_name: str,
    data_dir: str = "../../save_for_eval/tqa_hal_det/answers",
    bleurt_path: str = None,
    threshold: float = 0.5
) -> Tuple[List[Dict], np.ndarray]:
    """
    Load TruthfulQA data from pre-generated answer files and BLEURT scores.
    
    Returns:
        samples: List of {question, answer, label} dicts
        bleurt_scores: Raw BLEURT scores
    """
    # Try to load from HuggingFace, fallback to local JSON
    ds = None
    
    try:
        from datasets import load_dataset
        logger.info("Loading TruthfulQA dataset from HuggingFace...")
        # Clear cache and try fresh download
        ds = load_dataset("truthful_qa", "generation", split="validation", 
                         download_mode="force_redownload")
    except Exception as e:
        logger.warning(f"HuggingFace load failed: {e}")
    
    if ds is None:
        # Fallback: load questions from answer files directly
        logger.info("Loading questions from answer files...")
        model_short = model_name.split("/")[-1]
        
        # Count available files
        num_files = 0
        while os.path.exists(os.path.join(data_dir, f"most_likely_hal_det_{model_short}_tqa_answers_index_{num_files}.npy")):
            num_files += 1
        
        if num_files == 0:
            raise FileNotFoundError(f"No answer files found in {data_dir}")
        
        logger.info(f"Found {num_files} answer files")
        
        # Create dummy dataset with just indices
        ds = [{"question": f"Question {i}", "index": i} for i in range(num_files)]
    
    logger.info(f"Dataset size: {len(ds)}")
    
    # Load BLEURT scores
    if bleurt_path is None:
        bleurt_path = "../../ml_tqa_bleurt_score.npy"
    
    if not os.path.exists(bleurt_path):
        raise FileNotFoundError(f"BLEURT scores not found at {bleurt_path}")
    
    bleurt_scores = np.load(bleurt_path)
    logger.info(f"Loaded BLEURT scores: {bleurt_scores.shape}")
    
    # Load answers and create samples
    samples = []
    model_short = model_name.split("/")[-1]
    
    for i in tqdm(range(len(ds)), desc="Loading answers"):
        # Get question (from dataset or use placeholder)
        if isinstance(ds[i], dict) and "question" in ds[i]:
            question = ds[i]["question"]
        else:
            question = f"Question {i}"
        
        # Load answer file
        answer_file = os.path.join(
            data_dir, 
            f"most_likely_hal_det_{model_short}_tqa_answers_index_{i}.npy"
        )
        
        if not os.path.exists(answer_file):
            logger.warning(f"Answer file not found: {answer_file}")
            continue
        
        answers = np.load(answer_file)
        
        # Get BLEURT score and label
        if i < len(bleurt_scores):
            bleurt_score = bleurt_scores[i]
        else:
            logger.warning(f"No BLEURT score for index {i}")
            continue
            
        label = 1 if bleurt_score > threshold else 0  # 1 = truthful, 0 = hallucinated
        
        # Use first answer (most likely)
        answer = str(answers[0]) if len(answers) > 0 else ""
        
        samples.append({
            "question": question,
            "answer": answer,
            "label": label,
            "bleurt_score": float(bleurt_score)
        })
    
    logger.info(f"Loaded {len(samples)} samples")
    logger.info(f"  Truthful (score > {threshold}): {sum(1 for s in samples if s['label'] == 1)}")
    logger.info(f"  Hallucinated (score <= {threshold}): {sum(1 for s in samples if s['label'] == 0)}")
    
    return samples, bleurt_scores


def extract_hidden_states(
    model: nn.Module,
    tokenizer,
    samples: List[Dict],
    device: torch.device,
    layer_id: int = -1,
    batch_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract hidden states from samples.
    
    Returns:
        features: [num_samples, hidden_size]
        labels: [num_samples]
    """
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc="Extracting hidden states"):
            batch = samples[i:i+batch_size]
            
            prompts = [
                f"Answer the question concisely. Q: {s['question']} A: {s['answer']}"
                for s in batch
            ]
            batch_labels = [s['label'] for s in batch]
            
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
            
            # Get hidden state from target layer
            hidden = outputs.hidden_states[layer_id]  # [batch, seq_len, hidden_size]
            
            # Get last non-padding token for each example
            attention_mask = inputs.attention_mask
            seq_lengths = attention_mask.sum(dim=1) - 1
            
            for j, seq_len in enumerate(seq_lengths):
                last_hidden = hidden[j, seq_len, :].cpu().numpy()
                features.append(last_hidden)
            
            labels.extend(batch_labels)
    
    return np.array(features), np.array(labels)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 0.1
) -> Tuple[np.ndarray, Dict]:
    """Train logistic regression and return TSV vector."""
    
    logger.info(f"Training logistic regression with C={C}...")
    logger.info(f"Train: {len(y_train)} samples ({y_train.sum()} truthful, {len(y_train) - y_train.sum()} hallucinated)")
    logger.info(f"Test: {len(y_test)} samples ({y_test.sum()} truthful, {len(y_test) - y_test.sum()} hallucinated)")
    
    clf = LogisticRegression(
        C=C,
        max_iter=1000,
        solver='lbfgs',
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    
    # Metrics
    train_preds = clf.predict(X_train)
    train_probs = clf.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, train_preds)
    train_auc = roc_auc_score(y_train, train_probs)
    
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    
    metrics = {
        "train_accuracy": train_acc,
        "train_auc": train_auc,
        "test_accuracy": test_acc,
        "test_auc": test_auc
    }
    
    logger.info(f"Train - Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
    logger.info(f"Test  - Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")
    
    # TSV = weight vector (direction from hallucinated to truthful)
    tsv_vector = clf.coef_[0]
    
    return tsv_vector, metrics


def apply_lm_head_constraint(
    tsv_vector: np.ndarray,
    lm_head_weight: np.ndarray,
    max_logit_change: float = 3.0,
    alpha: float = 1.5
) -> np.ndarray:
    """Scale TSV to limit max logit change."""
    delta_logits = alpha * tsv_vector @ lm_head_weight.T
    max_change = np.abs(delta_logits).max()
    
    if max_change > max_logit_change:
        scale = max_logit_change / max_change
        tsv_vector = tsv_vector * scale
        logger.info(f"Scaled TSV by {scale:.4f} (max logit change: {max_logit_change})")
    
    return tsv_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--layer_id", type=int, default=9)
    parser.add_argument("--C", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5, help="BLEURT threshold for truthful/hallucinated")
    parser.add_argument("--max_logit_change", type=float, default=3.0)
    parser.add_argument("--expected_alpha", type=float, default=1.5)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="../../artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
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
    samples, bleurt_scores = load_tqa_data(
        args.model_name,
        threshold=args.threshold
    )
    
    # Extract hidden states
    features, labels = extract_hidden_states(
        model, tokenizer, samples, device, args.layer_id, args.batch_size
    )
    
    logger.info(f"Features shape: {features.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=args.test_size, 
        random_state=args.seed, 
        stratify=labels
    )
    
    # Train logistic regression
    tsv_vector, metrics = train_logistic_regression(
        X_train, y_train, X_test, y_test, args.C
    )
    
    # Apply lm_head constraint
    logger.info("Applying lm_head constraint...")
    lm_head_weight = model.lm_head.weight.detach().cpu().numpy().astype(np.float32)
    tsv_vector = apply_lm_head_constraint(
        tsv_vector.astype(np.float32),
        lm_head_weight,
        args.max_logit_change,
        args.expected_alpha
    )
    
    tsv_tensor = torch.from_numpy(tsv_vector).float()
    
    # Compute logit statistics
    delta_logits = args.expected_alpha * tsv_tensor @ torch.from_numpy(lm_head_weight).T
    max_change = delta_logits.abs().max().item()
    
    logger.info(f"\nFinal TSV statistics:")
    logger.info(f"  Norm: {tsv_tensor.norm():.4f}")
    logger.info(f"  Max logit change: {max_change:.4f}")
    
    # Top affected tokens
    top_inc = torch.topk(delta_logits, 5)
    top_dec = torch.topk(-delta_logits, 5)
    
    logger.info("\nTop tokens with INCREASED probability:")
    for idx, val in zip(top_inc.indices, top_inc.values):
        logger.info(f"  {tokenizer.decode([idx.item()])!r}: +{val.item():.3f}")
    
    logger.info("\nTop tokens with DECREASED probability:")
    for idx, val in zip(top_dec.indices, top_dec.values):
        logger.info(f"  {tokenizer.decode([idx.item()])!r}: -{val.item():.3f}")
    
    # Save TSV
    model_short = args.model_name.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{model_short}_logreg_tsv_817.pt")
    
    num_layers = model.config.num_hidden_layers
    tsv_vectors = [
        torch.zeros(model.config.hidden_size) if i != args.layer_id else tsv_tensor
        for i in range(num_layers)
    ]
    
    torch.save({
        "tsv_vectors": tsv_vectors,
        "tsv_single": tsv_tensor,
        "model_name": args.model_name,
        "layer_id": args.layer_id,
        "method": "logistic_regression_817",
        "train_accuracy": metrics["train_accuracy"],
        "train_auc": metrics["train_auc"],
        "test_accuracy": metrics["test_accuracy"],
        "test_auc": metrics["test_auc"],
        "C": args.C,
        "threshold": args.threshold,
        "max_logit_change": args.max_logit_change,
        "num_train_samples": len(y_train),
        "num_test_samples": len(y_test),
        "num_total_samples": len(samples)
    }, output_path)
    
    logger.info(f"\nTSV saved to: {output_path}")
    
    print("\n" + "="*70)
    print("TSV Training Summary (Logistic Regression - 817 TruthfulQA Samples)")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Layer: {args.layer_id}")
    print(f"Total Samples: {len(samples)}")
    print(f"BLEURT Threshold: {args.threshold}")
    print(f"Train/Test Split: {1-args.test_size:.0%}/{args.test_size:.0%}")
    print("-"*70)
    print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Train AUC:      {metrics['train_auc']:.4f}")
    print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Test AUC:       {metrics['test_auc']:.4f}")
    print("-"*70)
    print(f"TSV Norm:       {tsv_tensor.norm():.4f}")
    print(f"Max Logit Δ:    {max_change:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()

