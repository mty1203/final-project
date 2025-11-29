"""
Hallucination Risk Probe Models

Implements various probe architectures for predicting hallucination risk
from LLM hidden states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LinearProbe(nn.Module):
    """Simple linear probe for hallucination risk prediction."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, hidden_size] or [batch_size, seq_len, hidden_size]
        Returns:
            risk: [batch_size] or [batch_size, seq_len] - sigmoid probability
        """
        x = self.dropout(hidden_states)
        logits = self.linear(x).squeeze(-1)
        return torch.sigmoid(logits)


class MLPProbe(nn.Module):
    """MLP probe with one hidden layer for better representation."""
    
    def __init__(self, hidden_size: int, intermediate_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dropout(hidden_states)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x).squeeze(-1)
        return torch.sigmoid(logits)


class MultiLayerProbe(nn.Module):
    """
    Probe that aggregates information from multiple layers.
    Inspired by Tuned Lens approach.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        layer_ids: Optional[list] = None,
        aggregation: str = "attention",  # "mean", "attention", "last"
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layer_ids = layer_ids or list(range(num_layers))
        self.aggregation = aggregation
        
        if aggregation == "attention":
            # Learnable attention over layers
            self.layer_attention = nn.Linear(hidden_size, 1)
        
        # Final projection
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states_list: list) -> torch.Tensor:
        """
        Args:
            hidden_states_list: List of tensors [batch_size, seq_len, hidden_size]
                               one for each layer
        Returns:
            risk: [batch_size, seq_len] - sigmoid probability
        """
        # Select specified layers
        selected = [hidden_states_list[i] for i in self.layer_ids]
        
        if self.aggregation == "mean":
            # Simple mean aggregation
            aggregated = torch.stack(selected, dim=0).mean(dim=0)
        
        elif self.aggregation == "last":
            # Use last selected layer
            aggregated = selected[-1]
        
        elif self.aggregation == "attention":
            # Attention-weighted aggregation
            stacked = torch.stack(selected, dim=0)  # [num_layers, batch, seq, hidden]
            attn_scores = self.layer_attention(stacked).squeeze(-1)  # [num_layers, batch, seq]
            attn_weights = F.softmax(attn_scores, dim=0)  # [num_layers, batch, seq]
            aggregated = (stacked * attn_weights.unsqueeze(-1)).sum(dim=0)  # [batch, seq, hidden]
        
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        x = self.dropout(aggregated)
        logits = self.fc(x).squeeze(-1)
        return torch.sigmoid(logits)


class ContrastiveProbe(nn.Module):
    """
    Contrastive probe that learns to distinguish truthful vs hallucinated states
    using contrastive learning objective.
    """
    
    def __init__(self, hidden_size: int, projection_size: int = 128, temperature: float = 0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_size)
        )
        self.classifier = nn.Linear(projection_size, 1)
        self.temperature = temperature
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.projector(hidden_states)
        projected = F.normalize(projected, p=2, dim=-1)
        logits = self.classifier(projected).squeeze(-1)
        return torch.sigmoid(logits)
    
    def get_embeddings(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get normalized embeddings for contrastive loss computation."""
        projected = self.projector(hidden_states)
        return F.normalize(projected, p=2, dim=-1)


def load_probe(
    probe_path: str,
    hidden_size: int,
    device: torch.device,
    probe_type: str = "linear"
) -> nn.Module:
    """
    Load a pre-trained probe from checkpoint.
    
    Args:
        probe_path: Path to probe weights
        hidden_size: Model hidden size
        device: Target device
        probe_type: Type of probe ("linear", "mlp", "multi_layer", "contrastive")
    
    Returns:
        Loaded probe model
    """
    if probe_type == "linear":
        probe = LinearProbe(hidden_size)
    elif probe_type == "mlp":
        probe = MLPProbe(hidden_size)
    elif probe_type == "contrastive":
        probe = ContrastiveProbe(hidden_size)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
    
    state_dict = torch.load(probe_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(state_dict, dict):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "probe_state_dict" in state_dict:
            state_dict = state_dict["probe_state_dict"]
    
    probe.load_state_dict(state_dict)
    probe = probe.to(device)
    probe.eval()
    
    return probe

