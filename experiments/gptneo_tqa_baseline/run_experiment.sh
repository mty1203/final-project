#!/usr/bin/env bash

set -e

cd "$(dirname "$0")/../.."  # 切到仓库根目录 /home/mty/cs762/tsv-main

# ==========================
# 实验配置(可通过环境变量覆盖)
# ==========================

MODEL_NAME=${MODEL_NAME:-gpt-neo-1.3B}
DATASET=${DATASET:-tqa}

BATCH_SIZE=${BATCH_SIZE:-32}
NUM_EXEMPLARS=${NUM_EXEMPLARS:-16}
NUM_SELECTED=${NUM_SELECTED:-32}

COMPONENT=${COMPONENT:-res}
STR_LAYER=${STR_LAYER:-9}
LAM=${LAM:-5}
SAVE_TSV_PATH=${SAVE_TSV_PATH:-artifacts/${MODEL_NAME}_${DATASET}_tsv.pt}

LOG_DIR="experiments/gptneo_tqa_baseline/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$SAVE_TSV_PATH")"

echo "=== 实验配置 ==="
echo "MODEL_NAME      = ${MODEL_NAME}"
echo "DATASET         = ${DATASET}"
echo "BATCH_SIZE      = ${BATCH_SIZE}"
echo "NUM_EXEMPLARS   = ${NUM_EXEMPLARS}"
echo "NUM_SELECTED    = ${NUM_SELECTED}"
echo "COMPONENT       = ${COMPONENT}"
echo "STR_LAYER       = ${STR_LAYER}"
echo "LAM             = ${LAM}"
echo "SAVE_TSV_PATH   = ${SAVE_TSV_PATH}"
echo "日志目录        = ${LOG_DIR}"
echo "================"

# ==========================
# 步骤 1: 生成 most-likely 答案
# ==========================

echo "[Step 1] 生成 most-likely 答案 ..."
python tsv_main.py \
  --model_name "${MODEL_NAME}" \
  --dataset_name "${DATASET}" \
  --gene 1 \
  --most_likely 1 \
  2>&1 | tee "${LOG_DIR}/step1_gene.log"

# ==========================
# 步骤 2: 生成 BLEURT ground truth 分数
# ==========================

echo "[Step 2] 生成 BLEURT GT 分数 ..."
python tsv_main.py \
  --model_name "${MODEL_NAME}" \
  --dataset_name "${DATASET}" \
  --generate_gt 1 \
  --most_likely 1 \
  2>&1 | tee "${LOG_DIR}/step2_generate_gt.log"

# ==========================
# 步骤 3: TSV 训练 + 测试
# ==========================

echo "[Step 3] TSV 训练 + 测试 ..."

PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} \
python tsv_main.py \
  --model_name "${MODEL_NAME}" \
  --dataset_name "${DATASET}" \
  --component "${COMPONENT}" \
  --str_layer "${STR_LAYER}" \
  --batch_size "${BATCH_SIZE}" \
  --num_exemplars "${NUM_EXEMPLARS}" \
  --num_selected_data "${NUM_SELECTED}" \
  --lam "${LAM}" \
  --save_tsv_path "${SAVE_TSV_PATH}" \
  2>&1 | tee "${LOG_DIR}/step3_train.log"

echo "=== 实验完成 ==="
echo "日志已保存到: ${LOG_DIR}"
echo "TSV 向量保存在: ${SAVE_TSV_PATH}"


