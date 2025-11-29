#!/usr/bin/env python
"""
Train TSV with lm_head Constraint

This script trains TSV vectors with an additional constraint to prevent
the TSV direction from causing extreme logit changes when projected
through the lm_head.

Key differences from original TSV training:
1. Add logits regularization loss to prevent extreme token probability changes
2. Monitor the effect of TSV on vocab distribution during training
3. Optionally use contrastive loss in logits space

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
from torch.cuda.amp import GradScaler
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_last_non_padded_token_rep(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Get the last non-padded token's representation for each sequence."""
    if attention_mask.dim() == 3:
        attention_mask = attention_mask.squeeze(1)
    lengths = attention_mask.sum(dim=1).long()
    batch_size = hidden_states.size(0)
    last_token_reps = torch.stack([hidden_states[i, lengths[i]-1, :] for i in range(batch_size)])
    return last_token_reps


def collate_fn(prompts: List[torch.Tensor], labels: List[int], device: torch.device):
    """Collate prompts and labels into batched tensors."""
    max_seq_len = max(p.size(1) for p in prompts)
    batch_size = len(prompts)
    
    prompts_padded = torch.zeros(batch_size, max_seq_len, dtype=prompts[0].dtype, device=device)
    for i, prompt in enumerate(prompts):
        seq_len = prompt.size(1)
        prompts_padded[i, :seq_len] = prompt.squeeze(0)
    
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    return prompts_padded, labels_tensor


def compute_ot_loss_cos(
    embeddings: torch.Tensor,
    centroids: torch.Tensor,
    labels_oh: torch.Tensor,
    batch_size: int,
    cos_temp: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Optimal Transport loss with cosine similarity.
    
    Args:
        embeddings: [batch_size, hidden_size] - normalized embeddings
        centroids: [2, hidden_size] - cluster centroids
        labels_oh: [batch_size, 2] - one-hot labels
        batch_size: batch size
        cos_temp: temperature for cosine similarity
    
    Returns:
        loss: scalar loss
        similarities: [batch_size, 2] - similarity scores
    """
    # Normalize
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    centroids = F.normalize(centroids, p=2, dim=-1)
    
    # Compute similarities
    similarities = torch.matmul(embeddings, centroids.T) / cos_temp
    
    # Soft cross-entropy loss
    log_probs = F.log_softmax(similarities, dim=-1)
    loss = -(labels_oh * log_probs).sum(dim=-1).mean()
    
    return loss, similarities


def compute_logits_regularization(
    tsv_vector: torch.Tensor,
    lm_head_weight: torch.Tensor,
    alpha: float = 1.0,
    max_logit_change: float = 5.0
) -> torch.Tensor:
    """
    Compute regularization loss to prevent extreme logit changes.
    
    When we steer: steered_hidden = hidden + alpha * tsv
    The logit change is: delta_logits = alpha * tsv @ lm_head.T
    
    We want to penalize large delta_logits to prevent any single token
    from dominating after steering.
    
    Args:
        tsv_vector: [hidden_size] - the TSV direction
        lm_head_weight: [vocab_size, hidden_size] - lm_head weights
        alpha: expected steering strength
        max_logit_change: maximum allowed logit change
    
    Returns:
        loss: regularization loss
    """
    # Compute logit changes from TSV
    delta_logits = alpha * torch.matmul(tsv_vector, lm_head_weight.T)  # [vocab_size]
    
    # Penalize logit changes that exceed the threshold
    excess = F.relu(delta_logits.abs() - max_logit_change)
    loss = excess.mean()
    
    return loss


def compute_kl_regularization(
    hidden_states: torch.Tensor,
    tsv_vector: torch.Tensor,
    lm_head_weight: torch.Tensor,
    alpha: float = 1.0,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute KL divergence regularization between original and steered distributions.
    
    This ensures the TSV doesn't drastically change the output distribution.
    
    Args:
        hidden_states: [batch_size, hidden_size]
        tsv_vector: [hidden_size]
        lm_head_weight: [vocab_size, hidden_size]
        alpha: steering strength
        temperature: softmax temperature
    
    Returns:
        kl_loss: KL divergence loss
    """
    # Original logits
    original_logits = torch.matmul(hidden_states, lm_head_weight.T) / temperature
    original_probs = F.softmax(original_logits, dim=-1)
    
    # Steered logits
    steered_hidden = hidden_states + alpha * tsv_vector.unsqueeze(0)
    steered_logits = torch.matmul(steered_hidden, lm_head_weight.T) / temperature
    steered_log_probs = F.log_softmax(steered_logits, dim=-1)
    
    # KL divergence: KL(original || steered)
    kl_loss = F.kl_div(steered_log_probs, original_probs, reduction='batchmean')
    
    return kl_loss


def update_centroids_ema(
    centroids: torch.Tensor,
    embeddings: torch.Tensor,
    labels_oh: torch.Tensor,
    ema_decay: float = 0.99
) -> torch.Tensor:
    """Update centroids using exponential moving average."""
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    
    # Compute weighted mean for each class
    for c in range(2):
        weights = labels_oh[:, c].unsqueeze(-1)
        if weights.sum() > 0:
            weighted_mean = (embeddings * weights).sum(dim=0) / weights.sum()
            centroids[c] = ema_decay * centroids[c] + (1 - ema_decay) * weighted_mean
    
    return F.normalize(centroids, p=2, dim=-1)


class TSVTrainer:
    """Trainer for TSV vectors with lm_head constraint."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: torch.device,
        layer_id: int = 9,
        hidden_size: int = 2048,
        lr: float = 0.005,
        cos_temp: float = 0.1,
        ema_decay: float = 0.99,
        logits_reg_weight: float = 0.1,
        kl_reg_weight: float = 0.05,
        max_logit_change: float = 3.0,
        expected_alpha: float = 1.5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.cos_temp = cos_temp
        self.ema_decay = ema_decay
        self.logits_reg_weight = logits_reg_weight
        self.kl_reg_weight = kl_reg_weight
        self.max_logit_change = max_logit_change
        self.expected_alpha = expected_alpha
        
        # Initialize TSV vector
        self.tsv = nn.Parameter(torch.zeros(hidden_size, device=device), requires_grad=True)
        
        # Initialize centroids
        self.centroids = F.normalize(
            torch.randn(2, hidden_size, device=device, dtype=torch.float16),
            p=2, dim=-1
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW([self.tsv], lr=lr)
        self.scaler = GradScaler()
        
        # Get lm_head weight (frozen)
        self.lm_head_weight = model.lm_head.weight.detach()
        
        # Statistics
        self.stats = {
            "ot_losses": [],
            "logits_reg_losses": [],
            "kl_reg_losses": [],
            "total_losses": [],
            "max_logit_changes": []
        }
    
    def train_step(
        self,
        prompts: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_oh: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one training step."""
        self.optimizer.zero_grad()
        
        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.model(
                prompts,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Get hidden states from target layer
            hidden_states = outputs.hidden_states[self.layer_id]
            last_token_rep = get_last_non_padded_token_rep(hidden_states, attention_mask)
            
            # Apply TSV to hidden states
            steered_rep = last_token_rep + self.tsv.half()
            
            # Normalize for OT loss
            steered_rep_norm = F.normalize(steered_rep, p=2, dim=-1)
            
            # OT Loss (main objective)
            ot_loss, similarities = compute_ot_loss_cos(
                steered_rep_norm, self.centroids, labels_oh,
                prompts.size(0), self.cos_temp
            )
            
            # Logits regularization loss
            logits_reg_loss = compute_logits_regularization(
                self.tsv.float(),
                self.lm_head_weight.float(),
                self.expected_alpha,
                self.max_logit_change
            )
            
            # KL regularization loss
            kl_reg_loss = compute_kl_regularization(
                last_token_rep.float(),
                self.tsv.float(),
                self.lm_head_weight.float(),
                self.expected_alpha
            )
            
            # Total loss
            total_loss = (
                ot_loss +
                self.logits_reg_weight * logits_reg_loss +
                self.kl_reg_weight * kl_reg_loss
            )
        
        # Backward pass
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update centroids
        with torch.no_grad():
            self.centroids = update_centroids_ema(
                self.centroids, steered_rep_norm.detach(), labels_oh, self.ema_decay
            )
        
        # Compute max logit change for monitoring
        with torch.no_grad():
            delta_logits = self.expected_alpha * torch.matmul(
                self.tsv.float(), self.lm_head_weight.float().T
            )
            max_change = delta_logits.abs().max().item()
        
        return {
            "ot_loss": ot_loss.item(),
            "logits_reg_loss": logits_reg_loss.item(),
            "kl_reg_loss": kl_reg_loss.item(),
            "total_loss": total_loss.item(),
            "max_logit_change": max_change
        }
    
    def evaluate(
        self,
        prompts: List[torch.Tensor],
        labels: List[int],
        batch_size: int = 16
    ) -> float:
        """Evaluate TSV on test set, return AUROC."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                prompts_padded, labels_tensor = collate_fn(
                    batch_prompts, batch_labels, self.device
                )
                attention_mask = (prompts_padded != 0).float()
                
                outputs = self.model(
                    prompts_padded,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                hidden_states = outputs.hidden_states[self.layer_id]
                last_token_rep = get_last_non_padded_token_rep(hidden_states, attention_mask)
                
                # Apply TSV
                steered_rep = last_token_rep + self.tsv.half()
                steered_rep_norm = F.normalize(steered_rep, p=2, dim=-1)
                
                # Compute similarity to centroids
                similarities = torch.matmul(steered_rep_norm, self.centroids.T)
                probs = F.softmax(similarities / self.cos_temp, dim=-1)
                
                # Prediction: probability of being truthful (class 1)
                preds = probs[:, 1].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(batch_labels)
        
        # Check if both classes are present
        if len(set(all_labels)) < 2:
            # Return accuracy instead
            preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
            acc = sum(p == l for p, l in zip(preds_binary, all_labels)) / len(all_labels)
            return acc
        
        auroc = roc_auc_score(all_labels, all_preds)
        return auroc
    
    def get_tsv_vector(self) -> torch.Tensor:
        """Get the trained TSV vector."""
        return self.tsv.detach().cpu()
    
    def save(self, path: str, args):
        """Save TSV vector and training info."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Compute final logit statistics
        with torch.no_grad():
            delta_logits = self.expected_alpha * torch.matmul(
                self.tsv.float(), self.lm_head_weight.float().T
            )
            
            # Find top affected tokens
            top_k = 10
            top_increases = torch.topk(delta_logits, top_k)
            top_decreases = torch.topk(-delta_logits, top_k)
        
        payload = {
            "tsv_vectors": [
                torch.zeros(self.hidden_size) if i != args.str_layer else self.tsv.detach().cpu()
                for i in range(self.model.config.num_hidden_layers)
            ],
            "tsv_single": self.tsv.detach().cpu(),
            "centroids": self.centroids.detach().cpu(),
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "layer_id": args.str_layer,
            "logits_reg_weight": self.logits_reg_weight,
            "kl_reg_weight": self.kl_reg_weight,
            "max_logit_change": self.max_logit_change,
            "expected_alpha": self.expected_alpha,
            "stats": self.stats,
            "top_increased_tokens": {
                "indices": top_increases.indices.cpu().tolist(),
                "values": top_increases.values.cpu().tolist()
            },
            "top_decreased_tokens": {
                "indices": top_decreases.indices.cpu().tolist(),
                "values": top_decreases.values.cpu().tolist()
            }
        }
        
        torch.save(payload, path)
        logger.info(f"TSV saved to {path}")
        
        # Print top affected tokens
        logger.info("Top tokens with increased probability after steering:")
        for idx, val in zip(top_increases.indices[:5], top_increases.values[:5]):
            token = self.tokenizer.decode([idx.item()])
            logger.info(f"  {token!r}: +{val.item():.3f}")
        
        logger.info("Top tokens with decreased probability after steering:")
        for idx, val in zip(top_decreases.indices[:5], top_decreases.values[:5]):
            token = self.tokenizer.decode([idx.item()])
            logger.info(f"  {token!r}: -{val.item():.3f}")


def load_training_data(
    model_name: str,
    tokenizer,
    dataset_name: str = "tqa",
    bleurt_threshold: float = 0.5,
    device: torch.device = torch.device("cuda")
) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor], List[int]]:
    """
    Load training and test data.
    
    Returns:
        train_prompts, train_labels, test_prompts, test_labels
    """
    # Try to load BLEURT scores
    bleurt_path = f"./ml_{dataset_name}_bleurt_score.npy"
    
    if not os.path.exists(bleurt_path):
        logger.warning(f"BLEURT scores not found at {bleurt_path}")
        logger.info("Using hardcoded examples for training")
        
        # Hardcoded examples - more balanced and diverse
        examples = [
            # Truthful examples (label=1)
            ("What is the capital of France?", "Paris", 1),
            ("What is the capital of Germany?", "Berlin", 1),
            ("What is the capital of Italy?", "Rome", 1),
            ("What is the capital of Spain?", "Madrid", 1),
            ("What is the capital of UK?", "London", 1),
            ("Who wrote Romeo and Juliet?", "Shakespeare", 1),
            ("What is 2+2?", "4", 1),
            ("What is the speed of light?", "About 300,000 km/s", 1),
            ("Who was the first person on the Moon?", "Neil Armstrong", 1),
            ("What is the chemical formula for water?", "H2O", 1),
            ("How many planets are in our solar system?", "Eight", 1),
            ("Who painted the Mona Lisa?", "Leonardo da Vinci", 1),
            ("What is the largest ocean?", "Pacific Ocean", 1),
            ("What year did World War II end?", "1945", 1),
            ("What is the smallest country?", "Vatican City", 1),
            ("What is the tallest mountain?", "Mount Everest", 1),
            # Hallucinated examples (label=0)
            ("What is the capital of France?", "London", 0),
            ("What is the capital of Germany?", "Paris", 0),
            ("What is the capital of Italy?", "Madrid", 0),
            ("What is the capital of Spain?", "Berlin", 0),
            ("What is the capital of UK?", "Rome", 0),
            ("Who wrote Romeo and Juliet?", "Charles Dickens", 0),
            ("What is 2+2?", "5", 0),
            ("What is the speed of light?", "1000 mph", 0),
            ("Who was the first person on the Moon?", "Buzz Aldrin", 0),
            ("What is the chemical formula for water?", "CO2", 0),
            ("How many planets are in our solar system?", "Nine", 0),
            ("Who painted the Mona Lisa?", "Michelangelo", 0),
            ("What is the largest ocean?", "Atlantic Ocean", 0),
            ("What year did World War II end?", "1944", 0),
            ("What is the smallest country?", "Monaco", 0),
            ("What is the tallest mountain?", "K2", 0),
        ]
        
        prompts = []
        labels = []
        for q, a, label in examples:
            text = f"Answer the question concisely.\nQ: {q}\nA: {a}"
            tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)
            prompts.append(tokens)
            labels.append(label)
        
        # Split 80/20
        split_idx = int(len(prompts) * 0.8)
        return prompts[:split_idx], labels[:split_idx], prompts[split_idx:], labels[split_idx:]
    
    # Load from saved answers
    bleurt_scores = np.load(bleurt_path)
    
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "generation", split="validation")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    model_short = model_name.split("/")[-1]
    
    prompts = []
    labels = []
    
    for i, (item, score) in enumerate(zip(ds, bleurt_scores)):
        answer_path = f"./save_for_eval/{dataset_name}_hal_det/answers/most_likely_hal_det_{model_short}_{dataset_name}_answers_index_{i}.npy"
        
        if not os.path.exists(answer_path):
            continue
        
        answers = np.load(answer_path, allow_pickle=True)
        answer = answers[0] if len(answers) > 0 else ""
        
        text = f"Answer the question concisely.\nQ: {item['question']}\nA: {answer}"
        tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)
        prompts.append(tokens)
        labels.append(1 if score > bleurt_threshold else 0)
    
    # Shuffle and split
    indices = np.random.permutation(len(prompts))
    prompts = [prompts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    split_idx = int(len(prompts) * 0.8)
    
    logger.info(f"Loaded {len(prompts)} samples, {sum(labels)} truthful, {len(labels) - sum(labels)} hallucinated")
    
    return prompts[:split_idx], labels[:split_idx], prompts[split_idx:], labels[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Train TSV with lm_head constraint")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--dataset_name", type=str, default="tqa")
    parser.add_argument("--str_layer", type=int, default=9)
    parser.add_argument("--bleurt_threshold", type=float, default=0.5)
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--cos_temp", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    
    # Regularization weights
    parser.add_argument("--logits_reg_weight", type=float, default=0.1,
                       help="Weight for logits regularization loss")
    parser.add_argument("--kl_reg_weight", type=float, default=0.05,
                       help="Weight for KL divergence regularization")
    parser.add_argument("--max_logit_change", type=float, default=3.0,
                       help="Maximum allowed logit change per token")
    parser.add_argument("--expected_alpha", type=float, default=1.5,
                       help="Expected steering strength for regularization")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="../../artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
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
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    hidden_size = model.config.hidden_size
    
    # Load data
    logger.info("Loading training data...")
    train_prompts, train_labels, test_prompts, test_labels = load_training_data(
        args.model_name, tokenizer, args.dataset_name, args.bleurt_threshold, device
    )
    
    logger.info(f"Train: {len(train_prompts)} samples, Test: {len(test_prompts)} samples")
    
    # Create trainer
    trainer = TSVTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_id=args.str_layer,
        hidden_size=hidden_size,
        lr=args.lr,
        cos_temp=args.cos_temp,
        ema_decay=args.ema_decay,
        logits_reg_weight=args.logits_reg_weight,
        kl_reg_weight=args.kl_reg_weight,
        max_logit_change=args.max_logit_change,
        expected_alpha=args.expected_alpha
    )
    
    # Training loop
    logger.info("Starting training...")
    best_auroc = 0.0
    
    for epoch in range(args.epochs):
        # Shuffle training data
        indices = np.random.permutation(len(train_prompts))
        train_prompts_shuffled = [train_prompts[i] for i in indices]
        train_labels_shuffled = [train_labels[i] for i in indices]
        
        epoch_losses = []
        
        for i in tqdm(range(0, len(train_prompts_shuffled), args.batch_size), 
                      desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch_prompts = train_prompts_shuffled[i:i+args.batch_size]
            batch_labels = train_labels_shuffled[i:i+args.batch_size]
            
            prompts_padded, labels_tensor = collate_fn(batch_prompts, batch_labels, device)
            attention_mask = (prompts_padded != 0).float()
            
            # One-hot labels
            labels_oh = F.one_hot(labels_tensor, num_classes=2).float()
            
            # Training step
            step_losses = trainer.train_step(prompts_padded, attention_mask, labels_oh)
            epoch_losses.append(step_losses)
        
        # Aggregate epoch losses
        avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()}
        
        # Store stats
        for k, v in avg_losses.items():
            if k not in trainer.stats:
                trainer.stats[k] = []
            trainer.stats[k].append(v)
        
        # Evaluate
        auroc = trainer.evaluate(test_prompts, test_labels, args.batch_size)
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"OT Loss: {avg_losses['ot_loss']:.4f} | "
            f"Logits Reg: {avg_losses['logits_reg_loss']:.4f} | "
            f"KL Reg: {avg_losses['kl_reg_loss']:.4f} | "
            f"Max Δlogit: {avg_losses['max_logit_change']:.2f} | "
            f"AUROC: {auroc:.4f}"
        )
        
        if auroc > best_auroc:
            best_auroc = auroc
            logger.info(f"  New best AUROC: {best_auroc:.4f}")
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    model_short = args.model_name.split("/")[-1]
    output_path = os.path.join(args.output_dir, f"{model_short}_{args.dataset_name}_tsv_constrained.pt")
    trainer.save(output_path, args)
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best AUROC: {best_auroc:.4f}")
    logger.info(f"TSV saved to: {output_path}")
    
    # Print TSV statistics
    tsv = trainer.get_tsv_vector()
    logger.info(f"TSV vector norm: {tsv.norm():.4f}")


if __name__ == "__main__":
    main()

