#!/bin/bash
# ============================================================
# Step 4: Run Comparative Experiments
# ============================================================
# This script runs three experimental conditions:
# 1. Baseline: No steering (α = 0)
# 2. Fixed: Fixed-α steering (α = 0.3, always on)
# 3. Adaptive: Probe-controlled steering (α varies with risk)
#
# Metrics:
# - Accuracy (substring match with references)
# - Hallucination Rate
# - BLEURT Score (semantic similarity)
# - Style Similarity (embedding cosine)
# - Steering Rate (% of tokens steered)
#
# Prerequisites:
# - Run 02_train_tsv.sh (TSV trained)
# - Run 03_train_probe.sh (Probe trained)
#
# Output:
# - results/experiment_817/summary.json
# - results/experiment_817/*_generations.json
# ============================================================

set -e

# Configuration
MODEL_NAME="EleutherAI/gpt-neo-1.3B"
TSV_PATH="../../artifacts/gpt-neo-1.3B_logreg_tsv_817.pt"
PROBE_PATH="../../artifacts/gpt-neo-1.3B_probe_817.pt"
LAYER_ID=9

# Steering parameters
ALPHA_FIXED=0.3          # Fixed mode: constant α
ALPHA_MAX=0.5            # Adaptive mode: max α
RISK_THRESHOLD=0.6       # Adaptive mode: trigger threshold

# Generation parameters
NUM_SAMPLES=20           # Number of test questions
MAX_NEW_TOKENS=50
SEED=42

# Navigate to script directory
cd "$(dirname "$0")/.."
SCRIPT_DIR=$(pwd)

echo "============================================================"
echo "Step 4: Run Comparative Experiments"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "TSV: $TSV_PATH"
echo "Probe: $PROBE_PATH"
echo ""
echo "Parameters:"
echo "  - Fixed α: $ALPHA_FIXED"
echo "  - Adaptive α_max: $ALPHA_MAX"
echo "  - Risk threshold: $RISK_THRESHOLD"
echo "  - Test samples: $NUM_SAMPLES"
echo ""

# Check prerequisites
if [ ! -f "$TSV_PATH" ]; then
    echo "ERROR: TSV not found at $TSV_PATH"
    echo "Please run 02_train_tsv.sh first"
    exit 1
fi

if [ ! -f "$PROBE_PATH" ]; then
    echo "ERROR: Probe not found at $PROBE_PATH"
    echo "Please run 03_train_probe.sh first"
    exit 1
fi

# Create output directory
OUTPUT_DIR="results/experiment_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run experiments
echo "Running experiments..."
python run_full_experiment.py \
    --model_name "$MODEL_NAME" \
    --tsv_path "$TSV_PATH" \
    --probe_path "$PROBE_PATH" \
    --layer_id "$LAYER_ID" \
    --alpha_fixed "$ALPHA_FIXED" \
    --alpha_max "$ALPHA_MAX" \
    --risk_threshold "$RISK_THRESHOLD" \
    --num_samples "$NUM_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output_dir "$OUTPUT_DIR" \
    --seed "$SEED"

echo ""
echo "============================================================"
echo "Step 4 Complete!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files:"
echo "  - $OUTPUT_DIR/summary.json"
echo "  - $OUTPUT_DIR/baseline_generations.json"
echo "  - $OUTPUT_DIR/fixed_generations.json"
echo "  - $OUTPUT_DIR/adaptive_generations.json"
echo ""
echo "View results:"
echo "  cat $OUTPUT_DIR/summary.json | python -m json.tool"

