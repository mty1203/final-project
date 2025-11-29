# TSV + Probe 联合引导实验完整流程

## 📋 实验目标

通过 **TSV (Transformer Steering Vector)** 和 **Hallucination Probe** 的联合作用，在文本生成过程中动态检测并降低模型的幻觉输出。

---

## 🔄 完整实验流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     步骤 0: 环境准备                              │
│  ✓ 安装依赖: transformers, torch, datasets, bleurt, sklearn     │
│  ✓ 下载模型: EleutherAI/gpt-neo-1.3B (或其他 LLM)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              步骤 1: 生成 Most-Likely 答案                        │
│  脚本: tsv_main.py --gene 1 --most_likely 1                     │
│  输入: TruthfulQA 数据集                                          │
│  输出: save_for_eval/tqa_hal_det/answers/*.npy                  │
│  说明: 对每个问题生成模型的"最可能"答案 (greedy decoding)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         步骤 2: 生成 BLEURT Ground Truth 分数                     │
│  脚本: tsv_main.py --generate_gt 1 --most_likely 1              │
│  输入: 步骤1的答案 + TruthfulQA 的参考答案                         │
│  输出: ml_tqa_bleurt_score.npy                                  │
│  说明: 用 BLEURT 评估每个生成答案的质量，作为幻觉检测的监督信号     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   步骤 3: 训练 TSV 向量                           │
│  脚本: tsv_main.py --component res --str_layer 9                │
│  输入: 步骤1的答案 + 步骤2的BLEURT分数                            │
│  输出: artifacts/gpt-neo-1.3B_tqa_tsv.pt                        │
│  说明: 学习一个 steering 向量，能将隐藏状态推向"更真实"的方向      │
│  核心方法: OT loss + clustering + EMA 更新                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              步骤 4: 训练 Hallucination Probe                    │
│  脚本: experiments/tsv_probe_generation/train_probe.py          │
│  输入: 步骤1的答案 + 步骤2的BLEURT分数                            │
│  输出: artifacts/probe_weights.pt                               │
│  说明: 训练一个线性探针，输入隐藏状态 → 输出幻觉风险概率           │
│  训练数据: 提取 layer_id 层的隐藏状态作为特征                     │
│          BLEURT < threshold 标记为幻觉(label=1)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             步骤 5: TSV + Probe 联合引导生成                      │
│  脚本: experiments/tsv_probe_generation/steer_with_probe.py     │
│  输入: TSV向量 + Probe权重 + 测试问题                            │
│  输出: 生成文本 + 风险轨迹 + 评估指标                             │
│  流程: 每生成一个 token:                                          │
│    1. 提取 hidden state                                         │
│    2. Probe 判断风险 → risk_score                               │
│    3. 若 risk > threshold: 沿TSV方向调整 hidden state           │
│    4. 重新计算 logits 并混合                                     │
│    5. 采样下一个 token                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   步骤 6: 分析实验结果                            │
│  查看: experiments/tsv_probe_generation/logs/                   │
│  指标: - 平均风险 (mean_risk)                                    │
│       - 引导触发率 (steering_trigger_rate)                       │
│       - 幻觉率 (hallucination_rate)                             │
│       - BLEURT 分数 (文本质量)                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📝 详细步骤说明

### 步骤 1: 生成 Most-Likely 答案

**作用**: 让模型对 TruthfulQA 的每个问题生成一个"最可能"的答案（使用 greedy decoding），作为后续训练的基础数据。

**命令**:
```bash
cd /home/mty/cs762/tsv-main
python tsv_main.py \
  --model_name gpt-neo-1.3B \
  --dataset_name tqa \
  --gene 1 \
  --most_likely 1
```

**输出**:
- `save_for_eval/tqa_hal_det/answers/most_likely_hal_det_gpt-neo-1.3B_tqa_answers_index_*.npy`
- 每个文件对应一个问题的答案（numpy 数组格式）

**检查结果**:
```bash
ls save_for_eval/tqa_hal_det/answers/ | wc -l  # 应该有 817+ 个文件
```

---

### 步骤 2: 生成 BLEURT Ground Truth 分数

**作用**: 用 BLEURT 评估步骤1生成的每个答案与参考答案的相似度，作为"是否幻觉"的监督信号。

**命令**:
```bash
cd /home/mty/cs762/tsv-main
python tsv_main.py \
  --model_name gpt-neo-1.3B \
  --dataset_name tqa \
  --generate_gt 1 \
  --most_likely 1
```

**输出**:
- `ml_tqa_bleurt_score.npy` (根目录)
- 一个长度为 817 的数组，每个元素是对应答案的 BLEURT 分数
- 分数越高 → 答案越接近参考 → 越可能是真实的

**检查结果**:
```bash
python -c "import numpy as np; s=np.load('ml_tqa_bleurt_score.npy'); print(f'形状: {s.shape}, 平均分: {s.mean():.3f}, 范围: [{s.min():.3f}, {s.max():.3f}]')"
```

---

### 步骤 3: 训练 TSV 向量

**作用**: 学习一个"引导向量"，当加到模型的隐藏状态上时，能让模型输出更真实的答案。

**核心原理**:
1. 将样本分为"真实"和"幻觉"两类（根据 BLEURT 分数）
2. 在指定层（如第9层 residual stream）上提取隐藏状态
3. 使用 Optimal Transport Loss 让真实/幻觉样本在隐藏空间分离
4. TSV 向量 = 真实簇中心 - 幻觉簇中心

**命令**:
```bash
cd /home/mty/cs762/tsv-main
python tsv_main.py \
  --model_name gpt-neo-1.3B \
  --dataset_name tqa \
  --component res \
  --str_layer 9 \
  --batch_size 32 \
  --num_exemplars 16 \
  --num_selected_data 32 \
  --lam 5 \
  --save_tsv_path artifacts/gpt-neo-1.3B_tqa_tsv.pt
```

**参数说明**:
- `--component res`: 在 residual stream 上操作
- `--str_layer 9`: 在第9层（模型中间层）施加 steering
- `--num_exemplars 16`: 每类选16个典型样本作为 exemplar
- `--num_selected_data 32`: 每轮迭代选32个样本训练
- `--lam 5`: OT loss 的权重系数

**输出**:
- `artifacts/gpt-neo-1.3B_tqa_tsv.pt`
  ```python
  {
    "tsv_vectors": [tensor_layer0, ..., tensor_layer9, ...],  # 所有层的TSV
    "model_name": "gpt-neo-1.3B",
    "dataset_name": "tqa",
    "component": "res",
    "str_layer": 9,
    "lam": 5
  }
  ```
- `TSV_gpt-neo-1.3B_tqa/exemplar_num_16_num_selected_data_32/res/9/5/log.txt` (训练日志)

**检查结果**:
```bash
python -c "import torch; d=torch.load('artifacts/gpt-neo-1.3B_tqa_tsv.pt'); print(f'TSV形状: {d[\"tsv_vectors\"][9].shape}')"
# 应该输出: TSV形状: torch.Size([2560]) (gpt-neo-1.3B 的 hidden_size)
```

---

### 步骤 4: 训练 Hallucination Probe

**作用**: 训练一个轻量级分类器，能在生成过程中实时判断"当前 token 是否可能导致幻觉"。

**核心原理**:
1. 对每个答案，提取其在指定层的最后一个 token 的隐藏状态
2. 根据 BLEURT 分数打标签: `BLEURT < 0.5 → 幻觉(1), ≥ 0.5 → 真实(0)`
3. 训练一个线性分类器: `sigmoid(W·h + b)`

**命令**:
```bash
cd /home/mty/cs762/tsv-main
python experiments/tsv_probe_generation/train_probe.py \
  --model_name EleutherAI/gpt-neo-1.3B \
  --dataset tqa \
  --answers_dir save_for_eval/tqa_hal_det/answers \
  --answers_prefix "most_likely_hal_det_{model}_{dataset}_answers_index_{idx}.npy" \
  --bleurt_scores ml_tqa_bleurt_score.npy \
  --bleurt_threshold 0.5 \
  --layer_id 9 \
  --max_samples 500 \
  --epochs 3 \
  --lr 1e-3 \
  --output_path artifacts/probe_weights.pt
```

**参数说明**:
- `--layer_id 9`: 与 TSV 作用在同一层
- `--max_samples 500`: 用 500 个样本训练（可增加到 2000）
- `--bleurt_threshold 0.5`: BLEURT < 0.5 视为幻觉

**输出**:
- `artifacts/probe_weights.pt`
  ```python
  {
    "linear.weight": tensor([2560]),
    "linear.bias": tensor([1])
  }
  ```
- 训练日志显示准确率（如 68.8%）

**检查结果**:
```bash
python -c "import torch; s=torch.load('artifacts/probe_weights.pt'); print(f'Probe 参数: weight {s[\"linear.weight\"].shape}, bias {s[\"linear.bias\"].shape}')"
```

---

### 步骤 5: TSV + Probe 联合引导生成

**作用**: 在实际生成时，根据 Probe 实时判断的风险，动态注入 TSV 来降低幻觉。

**核心流程** (伪代码):
```python
for each token in generation:
    hidden = model.forward(...)  # 获取隐藏状态
    risk = probe(hidden)         # Probe 判断风险
    
    if risk > threshold:
        # 高风险! 沿 TSV 方向调整
        steered_hidden = hidden + alpha * risk * TSV
        steered_logits = lm_head(steered_hidden)
        # 混合原始 logits 和调整后的 logits
        final_logits = (1-mix)*logits + mix*steered_logits
    else:
        final_logits = logits
    
    next_token = sample(final_logits)
```

**命令**:
```bash
cd /home/mty/cs762/tsv-main
python experiments/tsv_probe_generation/steer_with_probe.py \
  --model_name EleutherAI/gpt-neo-1.3B \
  --tsv_path artifacts/gpt-neo-1.3B_tqa_tsv.pt \
  --probe_path artifacts/probe_weights.pt \
  --layer_id 9 \
  --risk_threshold 0.7 \
  --steer_alpha 1.0 \
  --steer_mix 0.7 \
  --num_samples 50 \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_p 0.9 \
  --output_dir experiments/tsv_probe_generation/logs/run_001
```

**参数说明**:
- `--risk_threshold 0.7`: Probe 输出 > 0.7 才触发 steering
- `--steer_alpha 1.0`: Steering 强度系数
- `--steer_mix 0.7`: 混合比例 (70% steered + 30% original)
- `--num_samples 50`: 测试 50 个问题

**输出**:
1. **生成日志**: `logs/run_001/generations.jsonl`
   ```json
   {
     "index": 0,
     "question": "What happens if you crack your knuckles?",
     "generated": "Cracking knuckles releases gas bubbles...",
     "risk_trace": [0.02, 0.86, 0.93, 0.45, ...],
     "steer_trigger_steps": [1, 2, 3],
     "hallucinated": false,
     "bleurt_score": 0.82
   }
   ```

2. **汇总指标**: `logs/run_001/summary.json`
   ```json
   {
     "mean_risk": 0.494,
     "steering_trigger_rate": 0.144,
     "hallucination_rate": 0.28,
     "bleurt_mean": 0.75
   }
   ```

**检查结果**:
```bash
cat experiments/tsv_probe_generation/logs/run_001/summary.json | jq '.'
```

---

### 步骤 6: 分析实验结果

**关键指标解读**:

1. **Mean Risk** (平均风险)
   - 所有生成 token 的平均 probe 风险分数
   - 越低越好 (说明生成内容整体风险低)

2. **Steering Trigger Rate** (引导触发率)
   - 有多少比例的 token 触发了 TSV 引导
   - 约 10-20% 比较合理 (太高说明模型本身就很容易幻觉)

3. **Hallucination Rate** (幻觉率)
   - 生成的答案中，有多少被判定为幻觉
   - 对比 baseline (不用 TSV) 看是否下降

4. **BLEURT Mean** (文本质量)
   - 生成答案与参考答案的平均相似度
   - 应该保持或提升 (说明 steering 没有破坏文本质量)

**对比实验建议**:
```bash
# Baseline: 不用 steering
python steer_with_probe.py ... --steer_alpha 0.0 --output_dir logs/baseline

# 低强度 steering
python steer_with_probe.py ... --steer_alpha 0.5 --output_dir logs/alpha_0.5

# 高强度 steering
python steer_with_probe.py ... --steer_alpha 1.5 --output_dir logs/alpha_1.5

# 不同阈值
python steer_with_probe.py ... --risk_threshold 0.5 --output_dir logs/threshold_0.5
python steer_with_probe.py ... --risk_threshold 0.9 --output_dir logs/threshold_0.9
```

---

## 🔍 一键运行完整流程

我们提供了一个脚本来自动化前3步 (baseline 实验):

```bash
cd /home/mty/cs762/tsv-main
bash experiments/gptneo_tqa_baseline/run_experiment.sh
```

然后手动运行步骤 4 和 5:

```bash
# 步骤 4: 训练 Probe
python experiments/tsv_probe_generation/train_probe.py \
  --model_name EleutherAI/gpt-neo-1.3B \
  --dataset tqa \
  --layer_id 9 \
  --max_samples 500 \
  --epochs 3 \
  --output_path artifacts/probe_weights.pt

# 步骤 5: 联合引导生成
python experiments/tsv_probe_generation/steer_with_probe.py \
  --model_name EleutherAI/gpt-neo-1.3B \
  --tsv_path artifacts/gpt-neo-1.3B_tqa_tsv.pt \
  --probe_path artifacts/probe_weights.pt \
  --layer_id 9 \
  --num_samples 50
```

---

## ✅ 当前状态检查

```bash
cd /home/mty/cs762/tsv-main

# 检查步骤 1 输出
ls save_for_eval/tqa_hal_det/answers/*.npy | wc -l
# 期望: > 800

# 检查步骤 2 输出
ls -lh ml_tqa_bleurt_score.npy
# 期望: 存在，约 6-7 KB

# 检查步骤 3 输出
ls -lh artifacts/gpt-neo-1.3B_tqa_tsv.pt
# 期望: 存在，约 200 KB

# 检查步骤 4 输出
ls -lh artifacts/probe_weights.pt
# 期望: 存在，约 10 KB
```

**根据你的当前状态**:
- ✅ 步骤 1: 已完成 (817 个答案文件存在)
- ✅ 步骤 2: 已完成 (ml_tqa_bleurt_score.npy 存在)
- ✅ 步骤 3: 已完成 (gpt-neo-1.3B_tqa_tsv.pt 存在)
- ✅ 步骤 4: 已完成 (probe_weights.pt 存在)
- ✅ 步骤 5: 已测试运行成功

**你可以直接进行完整的对比实验了！**

---

## 📊 实验建议

### 基础实验 (理解系统行为)
1. 运行 baseline (不用 steering): `--steer_alpha 0.0`
2. 运行标准 steering: `--steer_alpha 1.0 --risk_threshold 0.7`
3. 对比两者的幻觉率和文本质量

### 参数调优实验
- **调整 `steer_alpha`**: 0.0, 0.5, 1.0, 1.5, 2.0
- **调整 `risk_threshold`**: 0.5, 0.6, 0.7, 0.8, 0.9
- **调整 `steer_mix`**: 0.5, 0.7, 0.9

### 深入分析
- 查看 `risk_trace`: 哪些 token 触发了 steering？
- 对比生成文本: steering 如何改变了输出？
- 计算 AUROC: Probe 的判别能力如何？

---

## 🐛 常见问题

### Q1: OOM (显存不足)
**解决**: 
- 减小 `--batch_size` (默认32 → 16)
- 减小 `--num_selected_data` (默认32 → 16)
- 使用 4-bit 量化: `--load_in_4bit`

### Q2: 数据集加载失败
**现象**: `DatasetGenerationError`
**解决**: 脚本已经处理，会自动使用模拟数据

### Q3: Probe 准确率很低
**解决**:
- 增加训练样本: `--max_samples 2000`
- 增加训练轮数: `--epochs 10`
- 调整 BLEURT 阈值: `--bleurt_threshold 0.4`

---

## 📚 相关文件

- 主训练脚本: `tsv_main.py`
- TSV 层注入: `llm_layers.py`
- 训练工具: `train_utils.py`
- Baseline 实验: `experiments/gptneo_tqa_baseline/`
- 联合引导实验: `experiments/tsv_probe_generation/`

---

祝实验顺利！如有问题，请查看各脚本的 `--help` 或阅读 `README.md`。

