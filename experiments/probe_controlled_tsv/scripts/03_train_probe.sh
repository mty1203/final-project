#!/bin/bash
# ============================================================
# Step 3: Train Probe (Hallucination Risk Detector)
# ============================================================
# This script trains an MLP probe to predict hallucination risk
# from hidden states.
#
# Method: MLP (hidden_size -> 256 -> 1)
# - Input: Hidden states from layer 9
# - Labels: Same as TSV (BLEURT-based)
# - Output: Risk score in [0, 1]
#
# Prerequisites:
# - Run 01_generate_samples.sh first
# - Answer files and BLEURT scores exist
#
# Output:
# - artifacts/gpt-neo-1.3B_probe_817.pt
# ============================================================

set -e

# Configuration
MODEL_NAME="EleutherAI/gpt-neo-1.3B"
LAYER_ID=9
PROBE_TYPE="mlp"         # "mlp" or "linear"
THRESHOLD=0.5            # BLEURT threshold for labels
EPOCHS=100
LR=0.001
TEST_SIZE=0.2
BATCH_SIZE=16
SEED=42

# Navigate to script directory
cd "$(dirname "$0")/.."
SCRIPT_DIR=$(pwd)

echo "============================================================"
echo "Step 3: Train Probe (MLP)"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Layer: $LAYER_ID"
echo "Probe Type: $PROBE_TYPE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo ""

# Check prerequisites
BLEURT_FILE="../../ml_tqa_bleurt_score.npy"
if [ ! -f "$BLEURT_FILE" ]; then
    echo "ERROR: BLEURT scores not found at $BLEURT_FILE"
    echo "Please run 01_generate_samples.sh first"
    exit 1
fi

# Create output directory
mkdir -p ../../artifacts

# Train Probe
echo "Training Probe..."
python train_probe_full.py \
    --model_name "$MODEL_NAME" \
    --layer_id "$LAYER_ID" \
    --probe_type "$PROBE_TYPE" \
    --threshold "$THRESHOLD" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --test_size "$TEST_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "../../artifacts" \
    --seed "$SEED"

echo ""
echo "============================================================"
echo "Step 3 Complete!"
echo "============================================================"
echo "Probe saved to: ../../artifacts/gpt-neo-1.3B_probe_817.pt"
echo ""
echo "Expected metrics:"
echo "  - Train AUC: ~0.92"
echo "  - Test AUC: ~0.80"
echo ""
echo "Next: Run 04_run_experiments.sh"

