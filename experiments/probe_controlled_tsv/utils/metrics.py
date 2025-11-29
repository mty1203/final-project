"""
Evaluation Metrics for Truthfulness Experiments

Implements:
- Truthfulness metrics (accuracy, BLEURT)
- Hallucination detection metrics (precision, recall, F1)
- Utility metrics (perplexity, fluency)
- Efficiency metrics (latency, steering rate)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time
import re


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    # Truthfulness
    accuracy: float = 0.0
    bleurt_mean: float = 0.0
    bleurt_std: float = 0.0
    
    # Hallucination
    hallucination_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    
    # Utility
    perplexity: float = 0.0
    avg_length: float = 0.0
    
    # Efficiency
    tokens_per_second: float = 0.0
    steering_rate: float = 0.0
    mean_risk: float = 0.0
    mean_alpha: float = 0.0
    
    # Raw data
    num_samples: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "bleurt_mean": self.bleurt_mean,
            "bleurt_std": self.bleurt_std,
            "hallucination_rate": self.hallucination_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "perplexity": self.perplexity,
            "avg_length": self.avg_length,
            "tokens_per_second": self.tokens_per_second,
            "steering_rate": self.steering_rate,
            "mean_risk": self.mean_risk,
            "mean_alpha": self.mean_alpha,
            "num_samples": self.num_samples
        }


def compute_exact_match(prediction: str, references: List[str]) -> bool:
    """Check if prediction exactly matches any reference."""
    pred_clean = prediction.strip().lower()
    for ref in references:
        if pred_clean == ref.strip().lower():
            return True
    return False


def compute_substring_match(prediction: str, references: List[str]) -> bool:
    """Check if any reference is a substring of prediction."""
    pred_clean = prediction.strip().lower()
    for ref in references:
        ref_clean = ref.strip().lower()
        if ref_clean in pred_clean:
            return True
    return False


def compute_keyword_match(prediction: str, references: List[str], threshold: float = 0.5) -> bool:
    """Check if prediction contains enough keywords from references."""
    pred_words = set(prediction.lower().split())
    
    for ref in references:
        ref_words = set(ref.lower().split())
        if not ref_words:
            continue
        overlap = len(pred_words & ref_words) / len(ref_words)
        if overlap >= threshold:
            return True
    
    return False


def heuristic_hallucination_check(
    prediction: str,
    references: List[str],
    method: str = "substring"
) -> bool:
    """
    Heuristically determine if a prediction is hallucinated.
    
    Args:
        prediction: Model's generated answer
        references: List of correct reference answers
        method: Matching method ("exact", "substring", "keyword")
    
    Returns:
        True if prediction appears to be hallucinated (doesn't match references)
    """
    if not prediction.strip():
        return True  # Empty is hallucinated
    
    if method == "exact":
        return not compute_exact_match(prediction, references)
    elif method == "substring":
        return not compute_substring_match(prediction, references)
    elif method == "keyword":
        return not compute_keyword_match(prediction, references)
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_bleurt_scores(
    predictions: List[str],
    references: List[List[str]],
    device: torch.device = torch.device("cuda")
) -> Tuple[List[float], float, float]:
    """
    Compute BLEURT scores for predictions against references.
    
    Args:
        predictions: List of generated answers
        references: List of reference answer lists (one list per prediction)
        device: Target device
    
    Returns:
        scores: List of BLEURT scores
        mean: Mean score
        std: Standard deviation
    """
    try:
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        
        model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
        tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
        model.eval()
        
        scores = []
        
        with torch.no_grad():
            for pred, refs in zip(predictions, references):
                # Compute score against each reference, take max
                ref_scores = []
                for ref in refs:
                    inputs = tokenizer([pred], [ref], padding='longest', return_tensors='pt')
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    score = model(**inputs).logits.item()
                    ref_scores.append(score)
                
                scores.append(max(ref_scores) if ref_scores else 0.0)
        
        return scores, np.mean(scores), np.std(scores)
    
    except ImportError:
        # BLEURT not available, return placeholder
        return [0.0] * len(predictions), 0.0, 0.0


def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_length: int = 256
) -> float:
    """
    Compute perplexity of generated texts.
    
    Lower perplexity = more fluent/natural text.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(device)
            
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            num_tokens = inputs.input_ids.size(1)
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = np.exp(avg_loss)
    
    return perplexity


def compute_generation_stats(
    generations: List[str]
) -> Dict[str, float]:
    """Compute statistics about generated texts."""
    lengths = [len(g.split()) for g in generations]
    
    return {
        "avg_length": np.mean(lengths),
        "min_length": np.min(lengths),
        "max_length": np.max(lengths),
        "std_length": np.std(lengths)
    }


def compute_risk_coverage_curve(
    risks: List[float],
    hallucinated: List[bool],
    num_thresholds: int = 20
) -> Dict[str, List[float]]:
    """
    Compute risk-coverage curve.
    
    Shows how hallucination rate changes as we intervene on more samples
    (lower risk threshold = more intervention).
    
    Args:
        risks: List of risk scores
        hallucinated: List of hallucination labels
        num_thresholds: Number of threshold points
    
    Returns:
        Dictionary with threshold, coverage, and hallucination_rate lists
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    coverages = []
    hal_rates = []
    
    for thresh in thresholds:
        # Coverage = fraction of samples with risk >= threshold
        coverage = np.mean([r >= thresh for r in risks])
        coverages.append(coverage)
        
        # Hallucination rate among covered samples
        covered_hal = [h for r, h in zip(risks, hallucinated) if r >= thresh]
        hal_rate = np.mean(covered_hal) if covered_hal else 0.0
        hal_rates.append(hal_rate)
    
    return {
        "thresholds": thresholds.tolist(),
        "coverages": coverages,
        "hallucination_rates": hal_rates
    }


class LatencyTracker:
    """Track generation latency."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.total_time = 0.0
        self.total_tokens = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self, num_tokens: int):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.total_time += elapsed
            self.total_tokens += num_tokens
            self.start_time = None
    
    def tokens_per_second(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.total_tokens / self.total_time


def evaluate_generations(
    predictions: List[str],
    references: List[List[str]],
    risk_traces: Optional[List[List[float]]] = None,
    steering_triggers: Optional[List[List[int]]] = None,
    compute_bleurt: bool = False,
    device: torch.device = torch.device("cuda")
) -> EvaluationResult:
    """
    Comprehensive evaluation of generated text.
    
    Args:
        predictions: List of generated answers
        references: List of reference answer lists
        risk_traces: Optional list of risk score traces per generation
        steering_triggers: Optional list of steering trigger step lists
        compute_bleurt: Whether to compute BLEURT scores
        device: Target device
    
    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult()
    result.num_samples = len(predictions)
    
    # Hallucination detection
    hallucinated = [
        heuristic_hallucination_check(pred, refs)
        for pred, refs in zip(predictions, references)
    ]
    result.hallucination_rate = np.mean(hallucinated)
    result.accuracy = 1.0 - result.hallucination_rate
    
    # BLEURT scores
    if compute_bleurt:
        _, result.bleurt_mean, result.bleurt_std = compute_bleurt_scores(
            predictions, references, device
        )
    
    # Generation stats
    stats = compute_generation_stats(predictions)
    result.avg_length = stats["avg_length"]
    
    # Steering stats
    if risk_traces:
        all_risks = [r for trace in risk_traces for r in trace]
        result.mean_risk = np.mean(all_risks) if all_risks else 0.0
    
    if steering_triggers and risk_traces:
        total_tokens = sum(len(trace) for trace in risk_traces)
        total_triggers = sum(len(triggers) for triggers in steering_triggers)
        result.steering_rate = total_triggers / max(1, total_tokens)
    
    return result

