#!/bin/bash
# ============================================================
# Step 2: Train TSV (Truthfulness Steering Vector)
# ============================================================
# This script trains the TSV using Logistic Regression on
# the 817 TruthfulQA samples with BLEURT-based labels.
#
# Method: Logistic Regression
# - Input: Hidden states from layer 9
# - Labels: BLEURT > 0.5 = truthful, BLEURT <= 0.5 = hallucinated
# - Output: Weight vector as TSV direction
#
# Includes lm_head constraint to prevent logit explosion.
#
# Prerequisites:
# - Run 01_generate_samples.sh first
# - Answer files and BLEURT scores exist
#
# Output:
# - artifacts/gpt-neo-1.3B_logreg_tsv_817.pt
# ============================================================

set -e

# Configuration
MODEL_NAME="EleutherAI/gpt-neo-1.3B"
LAYER_ID=9
C=0.1                    # Regularization strength
THRESHOLD=0.5            # BLEURT threshold for labels
MAX_LOGIT_CHANGE=3.0     # lm_head constraint
TEST_SIZE=0.2            # Train/test split
BATCH_SIZE=16
SEED=42

# Navigate to script directory
cd "$(dirname "$0")/.."
SCRIPT_DIR=$(pwd)

echo "============================================================"
echo "Step 2: Train TSV (Logistic Regression)"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Layer: $LAYER_ID"
echo "Regularization C: $C"
echo "BLEURT Threshold: $THRESHOLD"
echo "Max Logit Change: $MAX_LOGIT_CHANGE"
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

# Train TSV
echo "Training TSV..."
python train_tsv_logreg_full.py \
    --model_name "$MODEL_NAME" \
    --layer_id "$LAYER_ID" \
    --C "$C" \
    --threshold "$THRESHOLD" \
    --max_logit_change "$MAX_LOGIT_CHANGE" \
    --test_size "$TEST_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "../../artifacts" \
    --seed "$SEED"

echo ""
echo "============================================================"
echo "Step 2 Complete!"
echo "============================================================"
echo "TSV saved to: ../../artifacts/gpt-neo-1.3B_logreg_tsv_817.pt"
echo ""
echo "Expected metrics:"
echo "  - Train AUC: ~0.99"
echo "  - Test AUC: ~0.77"
echo ""
echo "Next: Run 03_train_probe.sh"

