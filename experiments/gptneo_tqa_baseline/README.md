## 实验: TSV + GPT‑Neo‑1.3B 在 TruthfulQA (tqa) 上的结果复现

本实验将原始 TSV 方法迁移到较小的开源模型 **EleutherAI/gpt-neo-1.3B** 上,在 16GB 显存环境下可稳定运行,并记录清晰的实验步骤与指标。

---

### 1. 实验目标

- **模型**: `EleutherAI/gpt-neo-1.3B`
- **数据集**: TruthfulQA `tqa` (generation 任务)
- **任务**: 通过 TSV (Transformer Steering Vectors) 学习 hallucination 检测,用 **AUROC** 衡量检测性能。
- **核心问题**: 在资源受限(单卡 16GB)条件下,验证 TSV 思路在小模型上的有效性。

---

### 2. 目录结构

- `experiments/gptneo_tqa_baseline/`
  - `README.md` (本文件)
  - `run_experiment.sh` (一键跑完三阶段: 生成答案 → 生成 GT 分数 → TSV 训练)
  - `logs/`
    - `step1_gene.log` (生成答案日志)
    - `step2_generate_gt.log` (生成 BLEURT GT 日志)
    - `step3_train.log` (TSV 训练与测试日志,含 AUROC)

模型相关的缓存/输出仍沿用原工程的路径:

- 生成答案: `save_for_eval/tqa_hal_det/answers/*.npy`
- BLEURT 得分: `ml_tqa_bleurt_score.npy`
- TSV 训练日志: `TSV_{model_name}_{dataset_name}/.../log.txt`

---

### 3. 实验流程(三阶段)

#### 阶段 1: 用 GPT‑Neo‑1.3B 生成答案

- 脚本会调用:
  - `tsv_main.py` 中 `if args.gene:` 分支
- 生成文件模式:
  - `./save_for_eval/tqa_hal_det/answers/most_likely_hal_det_gpt-neo-1.3B_tqa_answers_index_{i}.npy`
- 命令 (由 `run_experiment.sh` 自动执行):

```bash
python tsv_main.py \
  --model_name gpt-neo-1.3B \
  --dataset_name tqa \
  --gene 1 \
  --most_likely 1
```

#### 阶段 2: 生成 BLEURT ground truth 分数

- 脚本会调用:
  - `tsv_main.py` 中 `elif args.generate_gt:` 分支
- 输出:
  - `ml_tqa_bleurt_score.npy` (most_likely 模式)

```bash
python tsv_main.py \
  --model_name gpt-neo-1.3B \
  --dataset_name tqa \
  --generate_gt 1 \
  --most_likely 1
```

#### 阶段 3: TSV 训练 + 评估

- 脚本会调用:
  - `tsv_main.py` 中 `else:` 分支
  - 首先构建 prompts / labels, 再调用 `train_model`
- 本实验采用:
  - `num_exemplars = 16`
  - `num_selected_data = 32`
  - `batch_size = 32` (可根据显存再调小到 16/8/4/1)
  - `component = res` (在 residual 输出上加 TSV)
  - `str_layer = 9`
  - `lam = 5`

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python tsv_main.py \
  --model_name gpt-neo-1.3B \
  --dataset_name tqa \
  --component res \
  --str_layer 9 \
  --batch_size 32 \
  --num_exemplars 16 \
  --num_selected_data 32 \
  --lam 5
```

---

### 4. 评价指标 (Metrics)

1. **主指标: Test AUROC (Area Under ROC Curve)**
   - 在 `tsv_main.py::train_model` 中通过:
     - `test_predictions, test_labels_combined = test_model(...)`
     - `roc_auc_score(test_labels_combined, test_predictions)`
   - 在日志中记为:
     - `Best test AUROC: {value}, at epoch: {epoch}`
   - 我们关心:
     - 最佳 AUROC 值
     - 对应 epoch (初始阶段 + 自监督阶段)

2. **训练损失 (Train Loss)**
   - 每个 epoch 记录:
     - `Epoch [e/E], Loss: x.xxx`
   - 观察损失是否平稳下降,与 AUROC 变化是否一致。

3. **选择样本数与类分布**
   - `num_exemplars`, `num_selected_data`
   - Sinkhorn 选择的伪标签分布(可从中间日志或额外打印中查看,如需要可后续扩展)。

---

### 5. 一键运行脚本

在仓库根目录 (`/home/mty/cs762/tsv-main`) 下:

```bash
chmod +x experiments/gptneo_tqa_baseline/run_experiment.sh
./experiments/gptneo_tqa_baseline/run_experiment.sh
```

该脚本会依次执行:

1. 生成 most-likely 答案 (`step1_gene.log`)
2. 生成 BLEURT GT 分数 (`step2_generate_gt.log`)
3. TSV 训练 + 测试 (`step3_train.log`, 同时在 TSV 日志目录中也有详细记录)

---

### 6. 可调节的超参数

在 `run_experiment.sh` 中可以通过环境变量覆盖默认配置:

- `MODEL_NAME` (默认 `gpt-neo-1.3B`)
- `DATASET` (默认 `tqa`)
- `BATCH_SIZE` (默认 `32`, 若显存仍紧张可调为 `16/8/4/1`)
- `NUM_EXEMPLARS` (默认 `16`)
- `NUM_SELECTED` (默认 `32`)
- `COMPONENT` (默认 `res`)
- `STR_LAYER` (默认 `9`)
- `LAM` (默认 `5`)

示例:

```bash
MODEL_NAME=gpt-neo-2.7B BATCH_SIZE=8 NUM_EXEMPLARS=32 NUM_SELECTED=64 \
./experiments/gptneo_tqa_baseline/run_experiment.sh
```

---

### 7. 实验记录建议

建议在本目录下额外维护一个简单的表格(例如 `results.md` 或 `results.csv`),记录每次实验设置与结果:

- `model_name, batch_size, num_exemplars, num_selected, component, str_layer, lam, best_auroc, best_epoch, notes`

方便后续比较不同超参数/模型规模下 TSV 的效果。


