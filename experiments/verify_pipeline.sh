#!/usr/bin/env bash
#
# 验证 TSV + Probe 实验流程的完整性
#

set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "  TSV + Probe 实验流程验证"
echo "========================================"
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    local file=$1
    local desc=$2
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo -e "${GREEN}✓${NC} $desc"
        echo "    路径: $file"
        echo "    大小: $size"
        return 0
    else
        echo -e "${RED}✗${NC} $desc"
        echo "    路径: $file (不存在)"
        return 1
    fi
}

check_dir() {
    local dir=$1
    local desc=$2
    local pattern=$3
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "$pattern" 2>/dev/null | wc -l)
        echo -e "${GREEN}✓${NC} $desc"
        echo "    路径: $dir"
        echo "    文件数: $count"
        return 0
    else
        echo -e "${RED}✗${NC} $desc"
        echo "    路径: $dir (不存在)"
        return 1
    fi
}

echo "步骤 1: 检查生成的答案文件"
echo "----------------------------------------"
check_dir "save_for_eval/tqa_hal_det/answers" \
    "Most-Likely 答案文件" "*.npy"
echo ""

echo "步骤 2: 检查 BLEURT 分数文件"
echo "----------------------------------------"
check_file "ml_tqa_bleurt_score.npy" \
    "BLEURT Ground Truth 分数"
if [ -f "ml_tqa_bleurt_score.npy" ]; then
    python3 -c "
import numpy as np
scores = np.load('ml_tqa_bleurt_score.npy')
print(f'    样本数: {len(scores)}')
print(f'    平均分: {scores.mean():.3f}')
print(f'    范围: [{scores.min():.3f}, {scores.max():.3f}]')
print(f'    幻觉比例 (< 0.5): {(scores < 0.5).mean()*100:.1f}%')
" 2>/dev/null || echo "    (无法解析文件)"
fi
echo ""

echo "步骤 3: 检查 TSV 向量文件"
echo "----------------------------------------"
check_file "artifacts/gpt-neo-1.3B_tqa_tsv.pt" \
    "TSV 向量"
if [ -f "artifacts/gpt-neo-1.3B_tqa_tsv.pt" ]; then
    python3 -c "
import torch
data = torch.load('artifacts/gpt-neo-1.3B_tqa_tsv.pt', map_location='cpu')
print(f'    模型: {data.get(\"model_name\", \"N/A\")}')
print(f'    数据集: {data.get(\"dataset_name\", \"N/A\")}')
print(f'    组件: {data.get(\"component\", \"N/A\")}')
print(f'    层索引: {data.get(\"str_layer\", \"N/A\")}')
if 'tsv_vectors' in data:
    vec = data['tsv_vectors'][data.get('str_layer', 9)]
    print(f'    TSV 维度: {vec.shape}')
    print(f'    TSV 范数: {vec.norm().item():.3f}')
" 2>/dev/null || echo "    (无法解析文件)"
fi
echo ""

echo "步骤 4: 检查 Probe 权重文件"
echo "----------------------------------------"
check_file "artifacts/probe_weights.pt" \
    "Hallucination Probe 权重"
if [ -f "artifacts/probe_weights.pt" ]; then
    python3 -c "
import torch
state = torch.load('artifacts/probe_weights.pt', map_location='cpu')
weight = state.get('linear.weight', None)
bias = state.get('linear.bias', None)
if weight is not None:
    print(f'    权重维度: {weight.shape}')
    print(f'    权重范数: {weight.norm().item():.3f}')
if bias is not None:
    print(f'    偏置: {bias.item():.3f}')
print(f'    总参数量: {sum(p.numel() for p in state.values())}')
" 2>/dev/null || echo "    (无法解析文件)"
fi
echo ""

echo "步骤 5: 检查生成日志"
echo "----------------------------------------"
if [ -d "experiments/tsv_probe_generation/logs" ]; then
    runs=$(find experiments/tsv_probe_generation/logs -name "generations.jsonl" | wc -l)
    echo -e "${GREEN}✓${NC} 引导生成日志"
    echo "    路径: experiments/tsv_probe_generation/logs"
    echo "    运行次数: $runs"
    
    latest=$(find experiments/tsv_probe_generation/logs -name "summary.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$latest" ]; then
        echo "    最新结果: $latest"
        python3 -c "
import json
with open('$latest') as f:
    data = json.load(f)
print(f'      样本数: {data.get(\"num_samples\", \"N/A\")}')
print(f'      平均风险: {data.get(\"mean_risk\", 0):.3f}')
print(f'      触发率: {data.get(\"steering_trigger_rate\", 0)*100:.1f}%')
print(f'      幻觉率: {data.get(\"hallucination_rate\", 0)*100:.1f}%')
" 2>/dev/null || echo "      (无法解析 JSON)"
    fi
else
    echo -e "${YELLOW}⚠${NC} 引导生成日志"
    echo "    路径: experiments/tsv_probe_generation/logs (不存在)"
    echo "    提示: 尚未运行步骤5 (steer_with_probe.py)"
fi
echo ""

echo "========================================"
echo "  验证完成"
echo "========================================"
echo ""

# 统计完成情况
total=5
completed=0

[ -d "save_for_eval/tqa_hal_det/answers" ] && ((completed++))
[ -f "ml_tqa_bleurt_score.npy" ] && ((completed++))
[ -f "artifacts/gpt-neo-1.3B_tqa_tsv.pt" ] && ((completed++))
[ -f "artifacts/probe_weights.pt" ] && ((completed++))
[ -d "experiments/tsv_probe_generation/logs" ] && ((completed++))

echo "流程完成度: $completed/$total"
echo ""

if [ $completed -eq $total ]; then
    echo -e "${GREEN}✓ 所有步骤已完成！可以进行对比实验。${NC}"
    echo ""
    echo "建议运行:"
    echo "  python experiments/tsv_probe_generation/steer_with_probe.py \\"
    echo "    --model_name EleutherAI/gpt-neo-1.3B \\"
    echo "    --tsv_path artifacts/gpt-neo-1.3B_tqa_tsv.pt \\"
    echo "    --probe_path artifacts/probe_weights.pt \\"
    echo "    --layer_id 9 \\"
    echo "    --num_samples 50"
elif [ $completed -ge 4 ]; then
    echo -e "${YELLOW}⚠ 基础数据已准备完成，可以进行引导生成实验。${NC}"
    echo ""
    echo "运行步骤 5:"
    echo "  python experiments/tsv_probe_generation/steer_with_probe.py --help"
else
    echo -e "${RED}✗ 请先完成基础步骤。${NC}"
    echo ""
    echo "运行完整流程:"
    echo "  bash experiments/gptneo_tqa_baseline/run_experiment.sh"
    echo "  python experiments/tsv_probe_generation/train_probe.py --help"
fi

echo ""

