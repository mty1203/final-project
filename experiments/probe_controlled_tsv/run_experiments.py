#!/usr/bin/env python
"""
Run All Experiments for Probe-Controlled TSV

This script runs a comprehensive comparison of steering methods:
1. Base (no steering)
2. TSV-Fixed (fixed alpha)
3. CAA (contrastive activation addition)
4. Probe-TSV (adaptive alpha)

It also performs ablation studies on:
- Different alpha values
- Different risk thresholds
- Different layers
- Different alpha scheduling strategies
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
import numpy as np
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_command(cmd: List[str], cwd: str = None) -> int:
    """Run a command and return exit code."""
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def run_evaluation(
    mode: str,
    model_name: str,
    tsv_path: str,
    probe_path: str,
    output_dir: str,
    layer_id: int = 9,
    alpha_fixed: float = 1.0,
    alpha_max: float = 2.0,
    risk_threshold: float = 0.6,
    num_samples: int = 15,
    seed: int = 42,
    extra_args: Dict = None
) -> str:
    """Run a single evaluation and return output path."""
    
    cmd = [
        "python", "evaluate.py",
        "--model_name", model_name,
        "--mode", mode,
        "--layer_id", str(layer_id),
        "--alpha_fixed", str(alpha_fixed),
        "--alpha_max", str(alpha_max),
        "--risk_threshold", str(risk_threshold),
        "--num_samples", str(num_samples),
        "--seed", str(seed),
        "--output_dir", output_dir
    ]
    
    if tsv_path:
        cmd.extend(["--tsv_path", tsv_path])
    if probe_path:
        cmd.extend(["--probe_path", probe_path])
    
    if extra_args:
        for k, v in extra_args.items():
            cmd.extend([f"--{k}", str(v)])
    
    script_dir = Path(__file__).parent
    run_command(cmd, cwd=str(script_dir))
    
    return os.path.join(output_dir, f"eval_{mode}.json")


def run_main_comparison(args):
    """Run main comparison of all steering methods."""
    logger.info("="*60)
    logger.info("Running Main Comparison")
    logger.info("="*60)
    
    results = {}
    output_dir = os.path.join(args.output_dir, "main_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Base (no steering)
    logger.info("\n[1/4] Running Base (no steering)...")
    run_evaluation(
        mode="none",
        model_name=args.model_name,
        tsv_path=None,
        probe_path=None,
        output_dir=output_dir,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    # 2. TSV-Fixed
    logger.info("\n[2/4] Running TSV-Fixed...")
    run_evaluation(
        mode="tsv_fixed",
        model_name=args.model_name,
        tsv_path=args.tsv_path,
        probe_path=None,
        output_dir=output_dir,
        layer_id=args.layer_id,
        alpha_fixed=1.0,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    # 3. CAA
    if args.caa_path:
        logger.info("\n[3/4] Running CAA...")
        run_evaluation(
            mode="caa",
            model_name=args.model_name,
            tsv_path=args.caa_path,
            probe_path=None,
            output_dir=output_dir,
            layer_id=args.layer_id,
            alpha_fixed=1.0,
            num_samples=args.num_samples,
            seed=args.seed
        )
    else:
        logger.info("\n[3/4] Skipping CAA (no caa_path provided)")
    
    # 4. Probe-TSV (Ours)
    logger.info("\n[4/4] Running Probe-TSV (Ours)...")
    run_evaluation(
        mode="probe_tsv",
        model_name=args.model_name,
        tsv_path=args.tsv_path,
        probe_path=args.probe_path,
        output_dir=output_dir,
        layer_id=args.layer_id,
        alpha_max=args.alpha_max,
        risk_threshold=args.risk_threshold,
        num_samples=args.num_samples,
        seed=args.seed
    )
    
    logger.info(f"\nMain comparison results saved to {output_dir}")


def run_alpha_ablation(args):
    """Run ablation study on alpha values."""
    logger.info("="*60)
    logger.info("Running Alpha Ablation Study")
    logger.info("="*60)
    
    output_dir = os.path.join(args.output_dir, "alpha_ablation")
    os.makedirs(output_dir, exist_ok=True)
    
    alpha_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    for alpha in alpha_values:
        logger.info(f"\nRunning with alpha={alpha}...")
        
        # TSV-Fixed with different alphas
        run_evaluation(
            mode="tsv_fixed",
            model_name=args.model_name,
            tsv_path=args.tsv_path,
            probe_path=None,
            output_dir=os.path.join(output_dir, f"tsv_fixed_alpha_{alpha}"),
            layer_id=args.layer_id,
            alpha_fixed=alpha,
            num_samples=args.num_samples,
            seed=args.seed
        )
        
        # Probe-TSV with different alpha_max
        run_evaluation(
            mode="probe_tsv",
            model_name=args.model_name,
            tsv_path=args.tsv_path,
            probe_path=args.probe_path,
            output_dir=os.path.join(output_dir, f"probe_tsv_alpha_max_{alpha}"),
            layer_id=args.layer_id,
            alpha_max=alpha,
            risk_threshold=args.risk_threshold,
            num_samples=args.num_samples,
            seed=args.seed
        )
    
    logger.info(f"\nAlpha ablation results saved to {output_dir}")


def run_threshold_ablation(args):
    """Run ablation study on risk thresholds."""
    logger.info("="*60)
    logger.info("Running Risk Threshold Ablation Study")
    logger.info("="*60)
    
    output_dir = os.path.join(args.output_dir, "threshold_ablation")
    os.makedirs(output_dir, exist_ok=True)
    
    thresholds = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for thresh in thresholds:
        logger.info(f"\nRunning with risk_threshold={thresh}...")
        
        run_evaluation(
            mode="probe_tsv",
            model_name=args.model_name,
            tsv_path=args.tsv_path,
            probe_path=args.probe_path,
            output_dir=os.path.join(output_dir, f"threshold_{thresh}"),
            layer_id=args.layer_id,
            alpha_max=args.alpha_max,
            risk_threshold=thresh,
            num_samples=args.num_samples,
            seed=args.seed
        )
    
    logger.info(f"\nThreshold ablation results saved to {output_dir}")


def run_layer_ablation(args):
    """Run ablation study on steering layers."""
    logger.info("="*60)
    logger.info("Running Layer Ablation Study")
    logger.info("="*60)
    
    output_dir = os.path.join(args.output_dir, "layer_ablation")
    os.makedirs(output_dir, exist_ok=True)
    
    # For GPT-Neo-1.3B, there are 24 layers (0-23)
    layers = [3, 6, 9, 12, 15, 18, 21]
    
    for layer in layers:
        logger.info(f"\nRunning with layer_id={layer}...")
        
        run_evaluation(
            mode="probe_tsv",
            model_name=args.model_name,
            tsv_path=args.tsv_path,
            probe_path=args.probe_path,
            output_dir=os.path.join(output_dir, f"layer_{layer}"),
            layer_id=layer,
            alpha_max=args.alpha_max,
            risk_threshold=args.risk_threshold,
            num_samples=args.num_samples,
            seed=args.seed
        )
    
    logger.info(f"\nLayer ablation results saved to {output_dir}")


def aggregate_results(output_dir: str):
    """Aggregate results from all experiments into a summary table."""
    logger.info("="*60)
    logger.info("Aggregating Results")
    logger.info("="*60)
    
    summary = []
    
    # Walk through all result files
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.startswith("eval_") and file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    
                    metrics = data.get("metrics", {})
                    config = data.get("config", {})
                    
                    summary.append({
                        "experiment": os.path.relpath(root, output_dir),
                        "mode": config.get("mode", "unknown"),
                        "accuracy": metrics.get("accuracy", 0),
                        "hallucination_rate": metrics.get("hallucination_rate", 0),
                        "steering_rate": metrics.get("steering_rate", 0),
                        "mean_risk": metrics.get("mean_risk", 0),
                        "tokens_per_second": metrics.get("tokens_per_second", 0)
                    })
                except Exception as e:
                    logger.warning(f"Failed to read {path}: {e}")
    
    # Sort by accuracy
    summary.sort(key=lambda x: x["accuracy"], reverse=True)
    
    # Print summary table
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print(f"{'Experiment':<40} {'Mode':<12} {'Acc':>8} {'Hal%':>8} {'Steer%':>8} {'Risk':>8} {'Tok/s':>8}")
    print("-"*100)
    
    for row in summary:
        print(f"{row['experiment']:<40} {row['mode']:<12} "
              f"{row['accuracy']:>8.4f} {row['hallucination_rate']:>8.4f} "
              f"{row['steering_rate']:>8.4f} {row['mean_risk']:>8.4f} "
              f"{row['tokens_per_second']:>8.2f}")
    
    print("="*100)
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--tsv_path", type=str, default="../../artifacts/gpt-neo-1.3B_tqa_tsv.pt")
    parser.add_argument("--caa_path", type=str, default=None)
    parser.add_argument("--probe_path", type=str, default="../../artifacts/probe_weights.pt")
    parser.add_argument("--layer_id", type=int, default=9)
    parser.add_argument("--alpha_max", type=float, default=2.0)
    parser.add_argument("--risk_threshold", type=float, default=0.6)
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    
    # Experiment selection
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--main", action="store_true", help="Run main comparison")
    parser.add_argument("--alpha_ablation", action="store_true", help="Run alpha ablation")
    parser.add_argument("--threshold_ablation", action="store_true", help="Run threshold ablation")
    parser.add_argument("--layer_ablation", action="store_true", help="Run layer ablation")
    parser.add_argument("--aggregate", action="store_true", help="Only aggregate existing results")
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Output directory: {args.output_dir}")
    
    # Save config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    if args.aggregate:
        aggregate_results(os.path.dirname(args.output_dir))
        return
    
    # Run selected experiments
    if args.all or args.main:
        run_main_comparison(args)
    
    if args.all or args.alpha_ablation:
        run_alpha_ablation(args)
    
    if args.all or args.threshold_ablation:
        run_threshold_ablation(args)
    
    if args.all or args.layer_ablation:
        run_layer_ablation(args)
    
    # Aggregate results
    aggregate_results(args.output_dir)
    
    logger.info("\nAll experiments completed!")


if __name__ == "__main__":
    main()

