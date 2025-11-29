"""
Steering Methods for Truthfulness Control

Implements various steering strategies:
- TSV-Fixed: Fixed alpha steering with TSV
- CAA: Contrastive Activation Addition
- Probe-TSV: Adaptive alpha based on probe risk
- Multi-Layer: Steering across multiple layers
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class SteeringMode(Enum):
    NONE = "none"
    TSV_FIXED = "tsv_fixed"
    CAA = "caa"
    PROBE_TSV = "probe_tsv"
    MULTI_LAYER = "multi_layer"


@dataclass
class SteeringConfig:
    """Configuration for steering behavior."""
    mode: SteeringMode = SteeringMode.NONE
    
    # TSV settings
    tsv_vector: Optional[torch.Tensor] = None
    layer_id: int = 9
    
    # Alpha control
    alpha_fixed: float = 1.0
    alpha_max: float = 2.0
    alpha_min: float = 0.0
    risk_threshold: float = 0.6
    
    # Mixing
    steer_mix: float = 0.7
    
    # Multi-layer settings
    layer_ids: Optional[List[int]] = None
    layer_weights: Optional[List[float]] = None


class AdaptiveAlphaScheduler:
    """
    Computes adaptive steering strength based on risk score.
    
    Implements several scheduling strategies:
    - linear: α = α_max * risk
    - threshold: α = α_max if risk > threshold else 0
    - sigmoid: α = α_max * sigmoid(k * (risk - threshold))
    - exponential: α = α_max * (exp(k * risk) - 1) / (exp(k) - 1)
    """
    
    def __init__(
        self,
        alpha_max: float = 2.0,
        alpha_min: float = 0.0,
        threshold: float = 0.6,
        schedule: str = "linear",
        k: float = 10.0  # steepness for sigmoid/exponential
    ):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.threshold = threshold
        self.schedule = schedule
        self.k = k
    
    def __call__(self, risk: float) -> float:
        """Compute alpha from risk score."""
        if self.schedule == "linear":
            # Linear scaling: α proportional to risk
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * risk
        
        elif self.schedule == "threshold":
            # Binary: full steering above threshold, none below
            alpha = self.alpha_max if risk >= self.threshold else self.alpha_min
        
        elif self.schedule == "sigmoid":
            # Smooth transition around threshold
            import math
            x = self.k * (risk - self.threshold)
            sigmoid = 1 / (1 + math.exp(-x))
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * sigmoid
        
        elif self.schedule == "exponential":
            # Exponential scaling for aggressive high-risk steering
            import math
            exp_risk = (math.exp(self.k * risk) - 1) / (math.exp(self.k) - 1)
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * exp_risk
        
        elif self.schedule == "quadratic":
            # Quadratic scaling
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * (risk ** 2)
        
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
        
        return max(self.alpha_min, min(self.alpha_max, alpha))


class LogitsSpaceSteering:
    """
    Steering in logits space (more stable than hidden state space).
    
    Instead of:
        steered_hidden = hidden + α * tsv_vector
        logits = hidden @ lm_head.T
    
    We do:
        logits_shift = tsv_vector @ lm_head.T
        steered_logits = logits + α * logits_shift
    
    This avoids numerical explosion when TSV aligns with certain lm_head rows.
    """
    
    def __init__(
        self,
        tsv_vector: torch.Tensor,
        lm_head_weight: torch.Tensor,
        steer_mix: float = 0.7
    ):
        self.tsv_vector = tsv_vector
        self.steer_mix = steer_mix
        
        # Pre-compute TSV's effect in logits space
        with torch.no_grad():
            self.tsv_logit_shift = torch.matmul(
                tsv_vector.unsqueeze(0),
                lm_head_weight.T
            )  # [1, vocab_size]
    
    def steer(
        self,
        logits: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Apply steering to logits.
        
        Args:
            logits: [batch_size, vocab_size]
            alpha: steering strength
        
        Returns:
            steered_logits: [batch_size, vocab_size]
        """
        if alpha <= 0:
            return logits
        
        steered = logits + alpha * self.tsv_logit_shift.to(logits.device)
        return (1 - self.steer_mix) * logits + self.steer_mix * steered


class MultiLayerSteering:
    """
    Steering across multiple layers with layer-specific weights.
    
    Each layer can have:
    - Different TSV vectors (or shared)
    - Different alpha values
    - Different risk thresholds
    """
    
    def __init__(
        self,
        tsv_vectors: List[torch.Tensor],
        layer_ids: List[int],
        layer_weights: Optional[List[float]] = None,
        alpha_scheduler: Optional[AdaptiveAlphaScheduler] = None
    ):
        self.tsv_vectors = tsv_vectors
        self.layer_ids = layer_ids
        self.layer_weights = layer_weights or [1.0] * len(layer_ids)
        self.alpha_scheduler = alpha_scheduler or AdaptiveAlphaScheduler()
        
        assert len(self.layer_weights) == len(layer_ids)
    
    def get_layer_alpha(self, layer_idx: int, risk: float) -> float:
        """Get steering alpha for a specific layer."""
        base_alpha = self.alpha_scheduler(risk)
        weight = self.layer_weights[layer_idx]
        return base_alpha * weight
    
    def get_tsv_for_layer(self, layer_idx: int) -> torch.Tensor:
        """Get TSV vector for a specific layer."""
        if layer_idx < len(self.tsv_vectors):
            return self.tsv_vectors[layer_idx]
        return self.tsv_vectors[-1]  # fallback to last


class CAAVectorBank:
    """
    Contrastive Activation Addition (CAA) vector bank.
    
    Stores mean activation differences between truthful and hallucinated examples
    for each layer. Used as baseline comparison to learned TSV.
    """
    
    def __init__(self):
        self.vectors: Dict[int, torch.Tensor] = {}
        self.metadata: Dict[str, any] = {}
    
    def add_vector(self, layer_id: int, vector: torch.Tensor):
        """Add a CAA vector for a layer."""
        self.vectors[layer_id] = vector
    
    def get_vector(self, layer_id: int) -> Optional[torch.Tensor]:
        """Get CAA vector for a layer."""
        return self.vectors.get(layer_id)
    
    def save(self, path: str):
        """Save vector bank to file."""
        torch.save({
            "vectors": self.vectors,
            "metadata": self.metadata
        }, path)
    
    @classmethod
    def load(cls, path: str, device: torch.device) -> "CAAVectorBank":
        """Load vector bank from file."""
        data = torch.load(path, map_location=device)
        bank = cls()
        bank.vectors = {k: v.to(device) for k, v in data["vectors"].items()}
        bank.metadata = data.get("metadata", {})
        return bank


class SteeringController:
    """
    Main controller that orchestrates steering during generation.
    
    Combines probe prediction with steering application.
    """
    
    def __init__(
        self,
        config: SteeringConfig,
        probe: Optional[nn.Module] = None,
        lm_head_weight: Optional[torch.Tensor] = None,
        device: torch.device = torch.device("cuda")
    ):
        self.config = config
        self.probe = probe
        self.device = device
        
        # Initialize steering method
        if config.tsv_vector is not None and lm_head_weight is not None:
            self.logits_steering = LogitsSpaceSteering(
                config.tsv_vector.to(device),
                lm_head_weight,
                config.steer_mix
            )
        else:
            self.logits_steering = None
        
        # Initialize alpha scheduler
        self.alpha_scheduler = AdaptiveAlphaScheduler(
            alpha_max=config.alpha_max,
            alpha_min=config.alpha_min,
            threshold=config.risk_threshold,
            schedule="linear"
        )
        
        # Statistics tracking
        self.stats = {
            "total_tokens": 0,
            "steered_tokens": 0,
            "risk_sum": 0.0,
            "alpha_sum": 0.0
        }
    
    def compute_risk(self, hidden_states: torch.Tensor) -> float:
        """Compute hallucination risk from hidden states."""
        if self.probe is None:
            return 0.5  # default risk if no probe
        
        with torch.no_grad():
            # Ensure correct dtype
            hidden = hidden_states.float() if hidden_states.dtype == torch.float16 else hidden_states
            risk = self.probe(hidden)
            return risk.item() if risk.numel() == 1 else risk.mean().item()
    
    def should_steer(self, risk: float) -> bool:
        """Determine if steering should be applied."""
        if self.config.mode == SteeringMode.NONE:
            return False
        if self.config.mode == SteeringMode.TSV_FIXED:
            return True  # always steer with fixed alpha
        return risk >= self.config.risk_threshold
    
    def get_alpha(self, risk: float) -> float:
        """Get steering strength based on risk and mode."""
        if self.config.mode == SteeringMode.TSV_FIXED:
            return self.config.alpha_fixed
        elif self.config.mode == SteeringMode.PROBE_TSV:
            return self.alpha_scheduler(risk)
        elif self.config.mode == SteeringMode.CAA:
            return self.config.alpha_fixed
        return 0.0
    
    def apply_steering(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply steering to logits based on hidden states.
        
        Returns:
            steered_logits: Modified logits
            info: Dictionary with steering information
        """
        info = {"risk": 0.0, "alpha": 0.0, "steered": False}
        
        if self.config.mode == SteeringMode.NONE:
            return logits, info
        
        # Compute risk
        risk = self.compute_risk(hidden_states)
        info["risk"] = risk
        
        # Update stats
        self.stats["total_tokens"] += 1
        self.stats["risk_sum"] += risk
        
        # Check if should steer
        if not self.should_steer(risk):
            return logits, info
        
        # Get alpha
        alpha = self.get_alpha(risk)
        info["alpha"] = alpha
        
        if alpha <= 0 or self.logits_steering is None:
            return logits, info
        
        # Apply steering
        steered_logits = self.logits_steering.steer(logits, alpha)
        info["steered"] = True
        
        # Update stats
        self.stats["steered_tokens"] += 1
        self.stats["alpha_sum"] += alpha
        
        return steered_logits, info
    
    def get_stats(self) -> Dict:
        """Get steering statistics."""
        total = max(1, self.stats["total_tokens"])
        steered = self.stats["steered_tokens"]
        return {
            "total_tokens": total,
            "steered_tokens": steered,
            "steering_rate": steered / total,
            "mean_risk": self.stats["risk_sum"] / total,
            "mean_alpha": self.stats["alpha_sum"] / max(1, steered)
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            "total_tokens": 0,
            "steered_tokens": 0,
            "risk_sum": 0.0,
            "alpha_sum": 0.0
        }

