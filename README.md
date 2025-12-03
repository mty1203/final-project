# TSV

Source code for ICML 2025 paper [Steer LLM Latents for Hallucination Detection](https://arxiv.org/abs/2503.01917) by Seongheon Park, Xuefeng Du, Min-Hsuan Yeh, Haobo Wang, and Yixuan Li

---

## Modifications from Original TSV

This repository contains the original TSV implementation plus extensions for **Probe-Controlled TSV** experiments. Key modifications:

### 1. **Memory Optimization**

**This Version**:
- **4-bit quantization support** via `bitsandbytes` for memory-efficient model loading
- **CPU offloading** using `device_map="auto"` and `max_memory` to handle models larger than GPU VRAM
- **Default model changed** from LLaMA3.1-8B to GPT-Neo-1.3B for 16GB GPU compatibility
- **KV cache disabled** during training (`model.config.use_cache = False`) to save VRAM

### 2. **Hidden State Extraction Improvements**

**This Version**:
- Uses `output.hidden_states[layer_number]` for compatibility with `CausalLMOutputWithPast` (e.g., GPT-Neo)
- Avoids `torch.stack` of all layers to reduce memory usage
- Extracts only the last non-padding token's hidden state per example

### 3. **TSV Saving & Reusability**

**This Version**:
- Added `--save_tsv_path` argument to save TSV vectors as `.pt` files
- Standardized checkpoint format with metadata (model name, layer, hyperparameters)
- Enables reuse of trained TSV in downstream experiments(combined with probe for steering)

### 4. **New Experimental Framework: Probe-Controlled TSV**

**Original TSV**: Focused on hallucination **detection** using Optimal Transport loss.

**This Version**:
- **Logistic Regression TSV**: Alternative TSV training method using sklearn's LogisticRegression
- **Hallucination Probe**: MLP classifier to predict hallucination risk from hidden states
- **Adaptive Steering**: Probe-controlled steering that only intervenes when risk is high

See `experiments/probe_controlled_tsv/` for the new experimental framework.

---

## Requirements

For the Probe-Controlled TSV experiments, see `experiments/probe_controlled_tsv/ENVIRONMENT.md`.

---

## Original TSV Pipeline

### LLM response generation

Generate responses for each question to construct an unlabeled QA dataset in the wild.

```bash
bash gen.sh
```

### GT generation

Generate [BLEURT](https://arxiv.org/abs/2004.04696) score for each QA pair

```bash
bash gt.sh
```

### Train TSV

Train TSV for hallucination detection using Optimal Transport loss.

```bash
bash train.sh
```

To save TSV vectors for reuse:

```bash
python tsv_main.py --model_name gpt-neo-1.3B --dataset_name tqa \
    --str_layer 9 --component res --lam 5 \
    --save_tsv_path artifacts/tsv_vectors.pt
```

---

## Probe-Controlled TSV Experiments

For the new Probe-Controlled TSV experiments, see:`report.pdf`

### Quick Run

```bash
cd experiments/probe_controlled_tsv/scripts/
./run_all.sh  # Run complete pipeline
```

---

## Implementation Details

### Original TSV Training (`tsv_main.py`)

The original TSV training uses **Optimal Transport (OT) loss**:

1. **Two-phase training**:
   - Phase 1: Train on exemplars to initialize centroids
   - Phase 2: Augment with selected data using Sinkhorn algorithm

2. **Centroid-based separation**:
   - Maintains 2 centroids (hallucinated vs truthful) in hidden space
   - Uses cosine similarity and OT to maximize separation

3. **TSV injection**:
   - TSV vectors are injected via forward hooks at specified layers
   - Controlled by `--component` (res/attn/mlp) and `--str_layer`

### Logistic Regression TSV (`experiments/probe_controlled_tsv/train_tsv_logreg_full.py`)

Alternative TSV training method:

1. **Feature extraction**: Extract hidden states from layer 9 for all 817 TruthfulQA samples
2. **Labeling**: BLEURT > 0.5 → truthful (1), ≤ 0.5 → hallucinated (0)
3. **Logistic regression**: Train sklearn `LogisticRegression` on hidden states
4. **TSV extraction**: `TSV = clf.coef_[0]` (the weight vector)
5. **lm_head constraint**: Scale TSV to limit max logit change to prevent explosion

### Probe Training (`experiments/probe_controlled_tsv/train_probe_full.py`)

1. **Architecture**: MLP (hidden_size → 256 → 1) with GELU activation
2. **Training**: Binary classification on same features/labels as TSV
3. **Output**: Risk score ∈ [0, 1] indicating hallucination probability

### Adaptive Steering (`experiments/probe_controlled_tsv/run_full_experiment.py`)

During generation:
1. Compute hidden state at layer 9
2. Probe predicts risk `r_t`
3. If `r_t >= threshold`: `α_t = α_max * (r_t - threshold) / (1 - threshold)`
4. Apply steering: `logits = logits + α_t * (TSV @ lm_head.T)`

---

## Reproducing Results

### Original TSV Results

Follow the original pipeline:

```bash
# 1. Generate answers
python tsv_main.py --gene 1 --model_name gpt-neo-1.3B --dataset_name tqa --most_likely 1

# 2. Generate BLEURT scores
python tsv_main.py --generate_gt 1 --model_name gpt-neo-1.3B --dataset_name tqa --most_likely 1

# 3. Train TSV
python tsv_main.py --model_name gpt-neo-1.3B --dataset_name tqa \
    --str_layer 9 --component res --lam 5 \
    --init_num_epochs 20 --aug_num_epochs 20
```

### Probe-Controlled TSV Results

**Prerequisites**:
- Python 3.11.5
- PyTorch 2.8.0 with CUDA 12.8
- See `experiments/probe_controlled_tsv/ENVIRONMENT.md` for full requirements

**Steps**:

```bash
# Navigate to scripts directory
cd experiments/probe_controlled_tsv/scripts/

# Option 1: Run everything (if you don't have data yet)
./run_all.sh

# Option 2: Skip data generation (if you already have answers + BLEURT)
./run_all.sh --skip-generate

# Option 3: Step-by-step
./01_generate_samples.sh  # Generate answers + BLEURT (uses tsv_main.py)
./02_train_tsv.sh         # Train Logistic Regression TSV
./03_train_probe.sh       # Train MLP probe
./04_run_experiments.sh   # Run Baseline/Fixed/Adaptive comparison
```

**Expected Outputs**:

- TSV: `artifacts/gpt-neo-1.3B_logreg_tsv_817.pt`
  - Train AUC: ~0.99, Test AUC: ~0.77
- Probe: `artifacts/gpt-neo-1.3B_probe_817.pt`
  - Train AUC: ~0.92, Test AUC: ~0.80
- Results: `results/experiment_*/summary.json`

**Expected Metrics** (20 TruthfulQA samples):

| Method   | Accuracy | Hal Rate | BLEURT | Style Sim | Steer Rate |
|----------|----------|----------|--------|-----------|------------|
| Baseline | 0.25     | 0.75     | 0.32   | 0.82      | 0.0%       |
| Fixed    | 0.25     | 0.75     | 0.31   | 0.79      | 100.0%     |
| Adaptive | **0.35** | **0.65** | **0.39** | 0.81      | 33.0%      |

**Note**: Results may vary slightly due to random seeds and model initialization.

---

## Citation

```
@inproceedings{
park2025steer,
title={Steer {LLM} Latents for Hallucination Detection},
author={Seongheon Park and Xuefeng Du and Min-Hsuan Yeh and Haobo Wang and Yixuan Li},
booktitle={Forty-second International Conference on Machine Learning},
year={2025}
}
```

---

## Acknowledgement

We gratefully acknowledge [HaloScope](https://arxiv.org/abs/2409.17504), [ITI](https://arxiv.org/abs/2306.03341), and [ICV](https://arxiv.org/abs/2311.06668) for their inspiring ideas and open-source contributions, which served as valuable foundations for this work.
