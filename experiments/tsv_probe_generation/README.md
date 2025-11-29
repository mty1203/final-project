## TSV + Probe 联合 Steering 实验框架

本实验旨在在文本生成(generation)过程中,利用 **TSV(Transformer Steering Vector)** 与一个轻量级 **Hallucination Probe** 联合,对模型输出进行在线评估与干预,以降低幻觉(hallucination)发生率。整个实验流程与代码均封装在 `experiments/tsv_probe_generation/` 目录内,可在 16GB 单卡环境中复现。

---

### 1. 背景与思路

1. **TSV**: 由 `tsv_main.py` 训练得到的 steering 向量,针对特定层(residual)可将隐藏状态沿“更真实”方向平移。
2. **Probe**: 一个对指定层隐藏状态进行二分类的线性探针,输出“当前 token 可能产生幻觉”的概率,可由 `HallucinationProbe`(Logistic Regression) 或其他判别器训练得到。
3. **在线 steering**: 在生成循环中,对每个 token:
   - 获取隐藏状态 `h`。
   - Probe 输出风险 `p = sigmoid(W h + b)`。
   - 若 `p` 超过阈值,就把 `h` 沿 TSV 方向调整,再重新投影到 logits,混合原 logits 与 steered logits。
4. **Metrics**: 既关注传统的语言流畅度(BLEU/BLEURT)等,也记录 probe 风险分布与 hallucination 率。

---

### 2. 目录结构

```
experiments/tsv_probe_generation/
├── README.md                # 实验说明 (本文件)
├── steer_with_probe.py      # 生成 + steering + 评估主脚本
├── train_probe.py           # 训练探针并保存为 probe_weights.pt
└── logs/                    # (运行后生成) 记录每次实验的指标
```

> 说明: TSV 向量、Probe 权重等实参默认认为存放在主工程输出路径中(如 `TSV_*` 目录或自定义 `*.pt` 文件);脚本通过命令行参数传入。

---

### 3. 依赖与输入

1. **模型与 Tokenizer**: 使用 `transformers` 中的 `AutoTokenizer`, `AutoModelForCausalLM`。建议采用与 TSV 训练一致的模型(如 GPT-Neo-1.3B)。
2. **TSV 参数文件**: 假定是一个 `torch.save` 的 `.pt` 文件,内容格式示例:
   ```python
   torch.save({
       "tsv_vectors": [tensor_layer0, tensor_layer1, ...],
       "str_layer": 9
   }, "artifacts/tsv_vectors.pt")
   ```
3. **Probe 权重**: 也是 `torch.save` 的 `.pt` 文件,与 `HallucinationProbe` (简单线性层) 结构对应。
4. **生成评测数据**: 默认演示用 TruthfulQA validation (`datasets.load_dataset("truthful_qa", "generation", split="validation")`)。
5. **可选 BLEURT/Ground Truth**: 若已经完成 `tsv_main.py` 中生成的 BLEURT 分数(np arrays),可通过参数传入,用于离线度量 hallucination rate。

---

### 4. 实验步骤

1. **准备 TSV 与 Probe**  
   - 使用 `tsv_main.py` 训练得到 TSV 向量(例如 residual 上的 `TSV_*` 目录)。
   - 训练一个简单的 probe: 从 TSV 训练阶段保存的隐藏状态 + 标签中拟合 logistic regression,保存为 `probe_weights.pt`。
2. **编辑配置文件** (可选)  
   - 创建 `artifacts/tsv_probe_config.json`,记录 `tsv_path`, `probe_path`, `layer_id`, `risk_threshold`, `steer_alpha`, `steer_mix` 等;也可直接在命令行传参。
3. **运行脚本**  
   - `steer_with_probe.py` 支持命令行参数(详见下节),输出:
     - 生成文本
     - 每个样本的 risk trace
     - 与 BLEURT/参考答案的对比指标
     - 聚合日志保存在 `experiments/tsv_probe_generation/logs/`
4. **记录指标**  
   - 记录 `平均 risk`, `高风险 token 比例`, `平均 BLEURT`, `Hallucination Rate`, `Steering 触发次数`, `最终 AUROC (可选)`。

---

### 5. 脚本使用

```bash
cd /home/mty/cs762/tsv-main
python experiments/tsv_probe_generation/steer_with_probe.py \
  --model_name EleutherAI/gpt-neo-1.3B \
  --dataset tqa \
  --tsv_path artifacts/tsv_vectors.pt \
  --probe_path artifacts/probe_weights.pt \
  --output_dir experiments/tsv_probe_generation/logs/run_001 \
  --num_samples 100 \
  --steer_alpha 0.8 \
  --steer_mix 0.7 \
  --risk_threshold 0.6 \
  --layer_id 9 \
  --max_new_tokens 64
```

主要参数说明:

| 参数 | 说明 |
| ---- | ---- |
| `--tsv_path` | TSV 向量的 `.pt` 文件路径 |
| `--probe_path` | Probe 权重 `.pt` 文件路径 |
| `--layer_id` | 对应 TSV/Probe 所作用的 Transformer 层索引 |
| `--steer_alpha` | 风险越大,越沿 TSV 方向平移隐藏状态的系数 |
| `--steer_mix` | Steered logits 与原 logits 的混合比例 |
| `--risk_threshold` | 超过此概率才触发 steering |
| `--num_samples` | 评测的样本数 |
| `--bleurt_scores` | (可选) 预先算好的 BLEURT 分数 `.npy`,用于对照分析 |

#### 5.1 训练 Probe

若还没有 `probe_weights.pt`, 可先运行 `train_probe.py`。该脚本依赖 baseline 实验生成的 most-likely 答案与 BLEURT 分数,自动抽取隐藏状态并训练线性探针:

```bash
cd /home/mty/cs762/tsv-main
python experiments/tsv_probe_generation/train_probe.py \
  --model_name EleutherAI/gpt-neo-1.3B \
  --dataset tqa \
  --answers_dir save_for_eval/tqa_hal_det/answers \
  --answers_prefix "most_likely_hal_det_{model}_{dataset}_answers_index_{idx}.npy" \
  --bleurt_scores ml_tqa_bleurt_score.npy \
  --layer_id 9 \
  --max_samples 2000 \
  --epochs 5 \
  --output_path artifacts/probe_weights.pt
```

完成后,`artifacts/probe_weights.pt` 即可在 `steer_with_probe.py --probe_path` 中直接使用。

---

### 6. Metrics 与记录

1. **Risk Metrics**
   - 平均 risk (`mean_probe_score`)
   - 超阈值比例 (`risk_gt_threshold`)
   - Steering 触发次数/百分比
2. **文本质量**
   - BLEURT/BERTScore(若传入 `--bleurt_scores`)
   - 长度、重复度、Perplexity(可扩展)
3. **Hallucination Proxy**
   - 若有 ground truth 标签(如 TSV 训练时 `gt_label`),可重新跑一次判别器,计算 AUROC/Accuracy。

日志文件中会以 JSONL+Markdown 的方式记录每条样本及聚合结果,便于后续分析。

---

### 7. 可扩展方向

- **多层 TSV 组合**: 根据 probe 的不同权重,对多个层同时施加 steering。
- **自适应阈值**: 根据上下文动态调整 `risk_threshold`。
- **更复杂的 probe**: 使用小型 MLP 或 Transformer 作为探针,联合训练。
- **多指标联合优化**: 在 `steer_with_probe.py` 中加入 reward model,实现 RLHF-style steering。

---

### 8. 结果记录建议

在 `experiments/tsv_probe_generation/logs/` 中新增 `summary.md` 或 `results.csv`,记录每次实验的配置与指标:

| run_id | model | layer | steer_alpha | threshold | mean_risk | risk>th (%) | BLEURT | hallucination_rate | notes |

方便横向比较不同 steering 策略的效果。

---

有了该框架,你可以在不改动主训练 pipeline 的情况下,快速验证“TSV + Probe”在生成阶段降低幻觉的效果。如需在 pipeline 中自动更新 TSV/Probe,可在训练完成后把最新的向量与探针权重拷贝到 `artifacts/` 并调用此脚本。祝实验顺利! 😊


