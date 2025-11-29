#!/bin/bash
# Run all experiments for Probe-Controlled TSV

set -e

# Configuration
MODEL_NAME="EleutherAI/gpt-neo-1.3B"
TSV_PATH="../../artifacts/gpt-neo-1.3B_tqa_tsv.pt"
PROBE_PATH="../../artifacts/probe_weights.pt"
LAYER_ID=9
NUM_SAMPLES=15
SEED=42

# Create artifacts directory if needed
mkdir -p ../../artifacts

echo "=============================================="
echo "Probe-Controlled TSV Experiments"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "TSV Path: $TSV_PATH"
echo "Probe Path: $PROBE_PATH"
echo "Layer: $LAYER_ID"
echo "Samples: $NUM_SAMPLES"
echo ""

# Check if TSV exists
if [ ! -f "$TSV_PATH" ]; then
    echo "Warning: TSV file not found at $TSV_PATH"
    echo "Please run TSV training first:"
    echo "  python ../../tsv_main.py --model_name gpt-neo-1.3B --component res --str_layer 9"
fi

# Check if Probe exists
if [ ! -f "$PROBE_PATH" ]; then
    echo "Warning: Probe file not found at $PROBE_PATH"
    echo "Training probe..."
    python ../train_probe.py \
        --model_name $MODEL_NAME \
        --probe_type linear \
        --layer_id $LAYER_ID \
        --output_dir ../../artifacts
fi

# Run main comparison
echo ""
echo "=============================================="
echo "Running Main Comparison"
echo "=============================================="
python ../run_experiments.py \
    --model_name $MODEL_NAME \
    --tsv_path $TSV_PATH \
    --probe_path $PROBE_PATH \
    --layer_id $LAYER_ID \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    --main

echo ""
echo "=============================================="
echo "Experiments Complete!"
echo "=============================================="
echo "Results saved to: results/"

