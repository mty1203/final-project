# Probe-Controlled TSV: Adaptive Truthfulness Steering

## 研究目标

通过自适应 steering 强度控制，在减少 hallucination 的同时保持模型的 utility 和 fluency。

## 方法对比

| 方法 | 描述 | α 控制 |
|------|------|--------|
| **Base** | 无 steering，原始模型 | - |
| **TSV-Fixed** | 固定 α 的 TSV steering | α = const |
| **CAA** | Contrastive Activation Addition | α = const |
| **Probe-TSV (Ours)** | Probe 预测风险，自适应 α | α = f(risk) |
| **Multi-Layer** | 多层联合 steering | α_l = f(risk, layer) |

## 目录结构

```
probe_controlled_tsv/
├── configs/           # 实验配置
├── models/            # 模型定义 (Probe, TSV wrapper)
├── utils/             # 工具函数
├── scripts/           # 运行脚本
├── logs/              # 实验日志
├── results/           # 结果输出
├── train_probe.py     # Probe 训练
├── train_caa.py       # CAA 向量计算
├── evaluate.py        # 统一评估脚本
└── run_experiments.py # 主实验入口
```

## 评估指标

1. **Truthfulness**: TruthfulQA accuracy, BLEURT score
2. **Hallucination Rate**: 基于参考答案的匹配率
3. **Utility**: Perplexity, generation length
4. **Efficiency**: Steering trigger rate, latency (tokens/sec)

## 快速开始

```bash
# 1. 训练 Probe
python train_probe.py --model_name EleutherAI/gpt-neo-1.3B --output_dir artifacts/

# 2. 计算 CAA 向量
python train_caa.py --model_name EleutherAI/gpt-neo-1.3B --output_dir artifacts/

# 3. 运行评估
python evaluate.py --config configs/probe_tsv.yaml

# 4. 运行完整实验
python run_experiments.py --all
```

## 参考文献

- TSV: Park et al., 2025 - "Steer LLM Latents for Hallucination Detection"
- ACT: Wang et al., 2024 - "Adaptive Activation Steering"
- CAA: Rimsky et al., 2024 - "Contrastive Activation Addition"
- Tuned Lens: Belrose et al., 2023 - "Eliciting Latent Predictions"

