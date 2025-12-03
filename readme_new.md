# Probe-Controlled TSV: Implementation Guide

This document explains the **Probe-Controlled TSV** extension built on top of the original TSV codebase. This implementation follows the proposal design: learn a truthfulness direction (TSV), train a probe to predict hallucination risk, and use adaptive steering during generation.

---

## What's New: Modifications from Original TSV

### 1. **Logistic Regression TSV** (vs Original OT-based TSV)

**Original TSV** (`tsv_main.py`):
- Uses **Optimal Transport (OT) loss** with Sinkhorn algorithm
- Maintains 2 centroids in hidden space
- Two-phase training: exemplars → augmented data

**New Implementation** (`train_tsv_logreg_full.py`):
- Uses **Logistic Regression** to learn separation direction
- Simpler: `TSV = LogisticRegression.coef_[0]`
- Directly optimizes classification accuracy
- Includes **lm_head constraint** to prevent logit explosion

**Why the change?**
- More interpretable (TSV is the classifier's weight vector)
- Faster training (no iterative Sinkhorn)
- Better generalization on our 817-sample dataset (Test AUC: 0.77 vs random)

### 2. **Hallucination Probe**

**Original TSV**: No probe component.

**New Implementation** (`train_probe_full.py`):
- MLP probe: `hidden_size (2048) → 256 → 1`
- Trained on same data as TSV (BLEURT-based labels)
- Outputs risk score ∈ [0, 1] for each hidden state
- Test AUC: ~0.80 (good risk prediction)

### 3. **Adaptive Steering** (vs Fixed-α)

**Original TSV**: Fixed steering strength (if used for generation).

**New Implementation** (`run_full_experiment.py`):
- **Baseline**: No steering (α = 0)
- **Fixed**: Constant α for all tokens
- **Adaptive**: α varies with probe risk:
  ```
  if risk >= threshold:
      α = α_max * (risk - threshold) / (1 - threshold)
  else:
      α = 0  # No intervention
  ```

**Result**: Adaptive only steers ~33% of tokens, achieving better accuracy (0.35 vs 0.25) than baseline.

### 4. **Memory & Compatibility Improvements**

Applied to both original and new code:
- 4-bit quantization support
- CPU offloading for large models
- Compatible with GPT-Neo (uses `hidden_states` list)
- English comments/logging (original had Chinese)

---

## Implementation Details

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Phase                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TruthfulQA (817 samples)                                  │
│       │                                                     │
│       ├──→ Generate answers (tsv_main.py)                  │
│       ├──→ Compute BLEURT scores (tsv_main.py)              │
│       │                                                     │
│       ▼                                                     │
│  Extract hidden states (Layer 9, last token)               │
│       │                                                     │
│       ├──→ Train TSV (Logistic Regression)                │
│       │   └──→ TSV = clf.coef_[0]                          │
│       │                                                     │
│       └──→ Train Probe (MLP)                                │
│           └──→ Probe: hidden → risk score                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Inference Phase                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  For each generation step t:                               │
│       │                                                     │
│       ├──→ Forward pass → hidden_t, logits_t               │
│       │                                                     │
│       ├──→ Probe: hidden_t → risk_t                        │
│       │                                                     │
│       ├──→ Compute α_t:                                    │
│       │   if risk_t >= threshold:                           │
│       │       α_t = α_max * (risk_t - threshold) / ...     │
│       │   else:                                             │
│       │       α_t = 0                                       │
│       │                                                     │
│       └──→ Apply steering:                                 │
│           logits_t = logits_t + α_t * (TSV @ lm_head.T)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. TSV Training (`train_tsv_logreg_full.py`)

**Input**: 817 (question, answer) pairs with BLEURT scores

**Process**:
1. Extract hidden states from layer 9:
   ```python
   prompt = f"Q: {question} A: {answer}"
   outputs = model(prompt, output_hidden_states=True)
   hidden = outputs.hidden_states[9][:, -1, :]  # Last token
   ```

2. Create labels:
   ```python
   label = 1 if bleurt_score > 0.5 else 0  # 1=truthful, 0=hallucinated
   ```

3. Train logistic regression:
   ```python
   clf = LogisticRegression(C=0.1, class_weight='balanced')
   clf.fit(X_train, y_train)
   tsv_vector = clf.coef_[0]  # [2048]
   ```

4. Apply lm_head constraint:
   ```python
   delta_logits = alpha * tsv_vector @ lm_head.weight.T
   if max(|delta_logits|) > 3.0:
       tsv_vector = tsv_vector * (3.0 / max(|delta_logits|))
   ```

**Output**: `artifacts/gpt-neo-1.3B_logreg_tsv_817.pt`

#### 2. Probe Training (`train_probe_full.py`)

**Architecture**:
```python
class MLPProbe(nn.Module):
    def __init__(self, hidden_size=2048):
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 1)
        self.activation = nn.GELU()
    
    def forward(self, hidden):
        x = self.activation(self.fc1(hidden))
        return torch.sigmoid(self.fc2(x))  # Risk score
```

**Training**: Binary cross-entropy loss on same features/labels as TSV

**Output**: `artifacts/gpt-neo-1.3B_probe_817.pt`

#### 3. Adaptive Steering (`run_full_experiment.py`)

**Algorithm**:
```python
for step in range(max_new_tokens):
    # Forward pass
    outputs = model(generated, output_hidden_states=True)
    logits = outputs.logits[:, -1, :]
    hidden = outputs.hidden_states[layer_id][:, -1, :]
    
    # Probe prediction
    risk = probe(hidden.float())
    
    # Adaptive α
    if risk >= threshold:
        alpha = alpha_max * (risk - threshold) / (1 - threshold)
    else:
        alpha = 0
    
    # Apply steering in logit space
    if alpha > 0:
        logit_shift = tsv_vector @ lm_head.weight.T
        logits = logits + alpha * logit_shift
    
    # Sample next token
    next_token = sample(logits)
```

**Why logit space?**
- More numerically stable than hidden space steering
- Directly controls token probabilities
- Easier to analyze and debug

---

## How to Reproduce Results

### Prerequisites

1. **Environment**: See `experiments/probe_controlled_tsv/ENVIRONMENT.md`
   - Python 3.11.5
   - PyTorch 2.8.0 (CUDA 12.8)
   - transformers 4.32.1
   - scikit-learn 1.3.0

2. **GPU**: 16GB VRAM (RTX 5070 Ti tested)
   - GPT-Neo-1.3B uses ~4-6GB in FP16

3. **Data**: TruthfulQA validation split (817 samples)

### Step-by-Step Reproduction

#### Option 1: One-Command Pipeline

```bash
cd experiments/probe_controlled_tsv/scripts/
./run_all.sh
```

This runs:
1. `01_generate_samples.sh` → Generate answers + BLEURT
2. `02_train_tsv.sh` → Train Logistic Regression TSV
3. `03_train_probe.sh` → Train MLP probe
4. `04_run_experiments.sh` → Run Baseline/Fixed/Adaptive

#### Option 2: Manual Steps

**Step 1: Generate Data** (if not exists)

```bash
cd experiments/probe_controlled_tsv/scripts/
./01_generate_samples.sh
```

This calls `tsv_main.py` to:
- Generate model answers for TruthfulQA questions
- Compute BLEURT scores as ground truth labels

**Outputs**:
- `save_for_eval/tqa_hal_det/answers/*.npy` (817 files)
- `ml_tqa_bleurt_score.npy` (817 scores)

**Step 2: Train TSV**

```bash
./02_train_tsv.sh
```

**Expected output**:
```
Train Accuracy: 0.9694
Train AUC:      0.9947
Test Accuracy:  0.7500
Test AUC:       0.7696
TSV Norm:       2.9669
Max Logit Δ:    1.2221
```

**Checkpoint**: `artifacts/gpt-neo-1.3B_logreg_tsv_817.pt`

**Step 3: Train Probe**

```bash
./03_train_probe.sh
```

**Expected output**:
```
Train Accuracy: 0.8361
Train AUC:      0.9203
Test Accuracy:  0.7317
Test AUC:       0.8047
```

**Checkpoint**: `artifacts/gpt-neo-1.3B_probe_817.pt`

**Step 4: Run Experiments**

```bash
./04_run_experiments.sh
```

**Expected results** (20 TruthfulQA samples):

| Method   | Accuracy | Hal Rate | BLEURT | Style Sim | Steer Rate |
|----------|----------|----------|--------|-----------|------------|
| Baseline | 0.25     | 0.75     | 0.32   | 0.82      | 0.0%       |
| Fixed    | 0.25     | 0.75     | 0.31   | 0.79      | 100.0%     |
| Adaptive | **0.35** | **0.65** | **0.39** | 0.81      | 33.0%      |

**Outputs**:
- `results/experiment_*/summary.json` (aggregated metrics)
- `results/experiment_*/*_generations.json` (per-question details)

### Verification

To verify your results match ours:

```bash
# Check TSV quality
python -c "
import torch
data = torch.load('artifacts/gpt-neo-1.3B_logreg_tsv_817.pt', weights_only=False)
print(f'TSV Test AUC: {data[\"test_auc\"]:.4f}')  # Should be ~0.77
"

# Check Probe quality
python -c "
import torch
data = torch.load('artifacts/gpt-neo-1.3B_probe_817.pt', weights_only=False)
print(f'Probe Test AUC: {data[\"test_auc\"]:.4f}')  # Should be ~0.80
"

# Check experiment results
cat experiments/probe_controlled_tsv/results/experiment_*/summary.json | grep -A 5 '"adaptive"'
```

### Troubleshooting

**Issue**: OOM (Out of Memory)
- **Solution**: Reduce `--batch_size` in training scripts (default: 16)
- Or use 4-bit quantization: add `--load_in_4bit` flag

**Issue**: TruthfulQA dataset loading fails
- **Solution**: Code has fallback to load from answer files directly
- Or manually download: `datasets load_dataset("truthful_qa", "generation")`

**Issue**: Results differ from expected
- **Check**: Random seed (default: 42) should be consistent
- **Check**: Model version (should be `EleutherAI/gpt-neo-1.3B`)
- **Check**: Layer ID (should be 9)

---

## File Structure

```
tsv-main/
├── tsv_main.py                    # Original TSV (OT-based) + data generation
├── README.md                      # This file (overview + modifications)
├── readme_new.md                  # This file (implementation guide)
│
├── experiments/probe_controlled_tsv/
│   ├── train_tsv_logreg_full.py   # Logistic Regression TSV training
│   ├── train_probe_full.py        # MLP probe training
│   ├── run_full_experiment.py     # Baseline/Fixed/Adaptive comparison
│   │
│   ├── models/
│   │   ├── probe.py               # Probe architectures
│   │   └── steering.py            # Steering controllers
│   │
│   ├── scripts/
│   │   ├── 01_generate_samples.sh
│   │   ├── 02_train_tsv.sh
│   │   ├── 03_train_probe.sh
│   │   ├── 04_run_experiments.sh
│   │   └── run_all.sh
│   │
│   ├── ENVIRONMENT.md              # Environment setup
│   ├── PRELIMINARY_RESULTS.md     # Experimental results
│   └── results/                   # Experiment outputs
│
└── artifacts/                     # Trained models (TSV, Probe)
```

---

## Relation to Original TSV

The original TSV code (`tsv_main.py`) implements:
- **Optimal Transport-based TSV training** (2-phase, centroid-based)
- **Data generation pipeline** (answers + BLEURT scores)

Our extensions:
- **Reuse** the data generation pipeline (answers + BLEURT)
- **Replace** OT-based TSV with Logistic Regression TSV
- **Add** probe training and adaptive steering
- **Keep** original TSV code intact (can still run OT-based training)

Both methods can coexist: use `tsv_main.py` for original TSV, or `experiments/probe_controlled_tsv/` for the new experiments.

---

## References

- **Original TSV**: Park et al., "Steer LLM Latents for Hallucination Detection" (ICML 2025)
- **Probe-Controlled TSV**: This implementation (see `PRELIMINARY_RESULTS.md` for details)
