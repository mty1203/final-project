#!/bin/bash
# ============================================================
# Step 1: Generate Samples (Answers + BLEURT Scores)
# ============================================================
# This script generates model answers for TruthfulQA questions
# and computes BLEURT scores as ground truth labels.
#
# Prerequisites:
# - Model downloaded (EleutherAI/gpt-neo-1.3B)
# - TruthfulQA dataset accessible
#
# Output:
# - save_for_eval/tqa_hal_det/answers/*.npy (817 answer files)
# - ml_tqa_bleurt_score.npy (BLEURT scores)
# ============================================================

set -e

# Configuration
MODEL_NAME="gpt-neo-1.3B"
DATASET="tqa"
BATCH_SIZE=32
NUM_EXEMPLARS=16

# Navigate to project root
cd "$(dirname "$0")/../../.."
PROJECT_ROOT=$(pwd)

echo "============================================================"
echo "Step 1: Generate Samples"
echo "============================================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Check if answers already exist
ANSWER_DIR="./save_for_eval/${DATASET}_hal_det/answers"
if [ -d "$ANSWER_DIR" ] && [ $(ls -1 "$ANSWER_DIR" | grep "$MODEL_NAME" | wc -l) -ge 817 ]; then
    echo "✓ Answer files already exist ($(ls -1 "$ANSWER_DIR" | grep "$MODEL_NAME" | wc -l) files)"
else
    echo "Generating answers..."
    python tsv_main.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET" \
        --generate_answer \
        --most_likely \
        --batch_size "$BATCH_SIZE" \
        --num_exemplars "$NUM_EXEMPLARS"
    echo "✓ Answers generated"
fi

# Check if BLEURT scores exist
BLEURT_FILE="./ml_${DATASET}_bleurt_score.npy"
if [ -f "$BLEURT_FILE" ]; then
    echo "✓ BLEURT scores already exist: $BLEURT_FILE"
else
    echo "Generating BLEURT scores..."
    python tsv_main.py \
        --model_name "$MODEL_NAME" \
        --dataset_name "$DATASET" \
        --generate_gt \
        --most_likely \
        --batch_size "$BATCH_SIZE"
    echo "✓ BLEURT scores generated"
fi

echo ""
echo "============================================================"
echo "Step 1 Complete!"
echo "============================================================"
echo "Generated files:"
echo "  - $ANSWER_DIR/*.npy"
echo "  - $BLEURT_FILE"
echo ""
echo "Next: Run 02_train_tsv.sh"

