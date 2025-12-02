#!/bin/bash
# ============================================================
# Run All Steps: Complete Pipeline
# ============================================================
# This script runs the complete pipeline:
# 1. Generate samples (if not exist)
# 2. Train TSV
# 3. Train Probe
# 4. Run comparative experiments
#
# Usage:
#   ./run_all.sh           # Run all steps
#   ./run_all.sh --skip-generate  # Skip step 1 if data exists
# ============================================================

set -e

# Parse arguments
SKIP_GENERATE=false
for arg in "$@"; do
    case $arg in
        --skip-generate)
            SKIP_GENERATE=true
            shift
            ;;
    esac
done

# Navigate to script directory
cd "$(dirname "$0")"
SCRIPT_DIR=$(pwd)

echo "============================================================"
echo "Probe-Controlled TSV: Complete Pipeline"
echo "============================================================"
echo ""

# Step 1: Generate Samples
if [ "$SKIP_GENERATE" = true ]; then
    echo "Skipping Step 1 (--skip-generate flag set)"
else
    echo ">>> Running Step 1: Generate Samples"
    bash 01_generate_samples.sh
fi
echo ""

# Step 2: Train TSV
echo ">>> Running Step 2: Train TSV"
bash 02_train_tsv.sh
echo ""

# Step 3: Train Probe
echo ">>> Running Step 3: Train Probe"
bash 03_train_probe.sh
echo ""

# Step 4: Run Experiments
echo ">>> Running Step 4: Run Experiments"
bash 04_run_experiments.sh
echo ""

echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo ""
echo "Summary of outputs:"
echo "  - TSV: ../../artifacts/gpt-neo-1.3B_logreg_tsv_817.pt"
echo "  - Probe: ../../artifacts/gpt-neo-1.3B_probe_817.pt"
echo "  - Results: results/experiment_*/"
echo ""
echo "Expected results:"
echo "  | Method   | Accuracy | Hal Rate |"
echo "  |----------|----------|----------|"
echo "  | Baseline | ~0.25    | ~0.75    |"
echo "  | Fixed    | ~0.25    | ~0.75    |"
echo "  | Adaptive | ~0.35    | ~0.65    |"
