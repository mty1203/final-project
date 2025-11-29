# TSV 训练方法对比：原始 vs 新方法

## 核心问题

### 原始 TSV 训练的问题

原始 TSV 训练（`tsv_main.py`）**只优化隐藏空间（Hidden Space）中的簇分离**，完全没有考虑当 TSV 向量通过 `lm_head` 投影到词汇空间（Vocab Space）后会发生什么。

**结果**：训练出的 TSV 向量在隐藏空间中能很好地区分"truthful"和"hallucinated"样本，但当应用 steering 时：

```python
steered_hidden = hidden + α * tsv_vector
steered_logits = steered_hidden @ lm_head.weight.T
```

某些 token 的 logit 会**爆炸性增长**（从 -16.8 暴涨到 +472.5），导致该 token 的概率变成 100%，其他所有 token 概率变成 0%。

**实际表现**：模型生成重复的 "NASL NASL NASL..."，完全无法正常生成文本。

---

## 对比表格

| 方面 | 原始 TSV 训练 | 新 TSV 训练（带约束） |
|------|--------------|---------------------|
| **训练目标** | 只优化隐藏空间的簇分离 | 优化簇分离 + 约束 logits 变化 |
| **损失函数** | 只有 OT Loss | OT Loss + Logits Reg + KL Reg |
| **是否考虑 lm_head** | ❌ 不考虑 | ✅ 显式约束 |
| **Max Δlogit** | 无限制（可达 472） | 限制在 ~2.5 以内 |
| **生成效果** | ❌ 重复 "NASL" | ✅ 正常生成 |

---

## 详细对比

### 1. 损失函数

#### 原始方法（`tsv_main.py`）

```python
# 只有 Optimal Transport Loss
ot_loss, similarities = compute_ot_loss_cos(
    last_token_rep, centroids, batch_labels_oh, batch_size, args
)
loss = ot_loss  # 只有这一个损失！
```

**问题**：
- 只关心 hidden space 中的分离
- 完全不考虑投影到 vocab space 后的效果
- TSV 向量可能学到与 `lm_head.weight[某个token]` 高度对齐的方向

#### 新方法（`train_tsv_constrained.py`）

```python
# 1. Optimal Transport Loss (主目标)
ot_loss, similarities = compute_ot_loss_cos(
    steered_rep_norm, centroids, labels_oh, batch_size, cos_temp
)

# 2. Logits Regularization (新加入)
logits_reg_loss = compute_logits_regularization(
    tsv_vector, lm_head_weight, expected_alpha, max_logit_change
)

# 3. KL Divergence Regularization (新加入)
kl_reg_loss = compute_kl_regularization(
    last_token_rep, tsv_vector, lm_head_weight, expected_alpha
)

# 总损失 = 主损失 + 两个约束项
total_loss = (
    ot_loss +
    logits_reg_weight * logits_reg_loss +  # 默认 0.2
    kl_reg_weight * kl_reg_loss             # 默认 0.1
)
```

**改进**：
- ✅ 保持隐藏空间的簇分离（OT Loss）
- ✅ 限制 logit 变化幅度（Logits Reg）
- ✅ 保持输出分布稳定（KL Reg）

---

### 2. Logits Regularization 详解

```python
def compute_logits_regularization(tsv_vector, lm_head_weight, alpha, max_logit_change):
    """
    计算：当应用 steering 时，每个 token 的 logit 会变化多少？
    
    delta_logits = α * tsv @ lm_head.T  # [vocab_size]
    
    如果某个 token 的 delta_logit > max_logit_change，就惩罚它。
    """
    delta_logits = alpha * torch.matmul(tsv_vector, lm_head_weight.T)
    
    # 只惩罚超过阈值的部分
    excess = F.relu(delta_logits.abs() - max_logit_change)
    loss = excess.mean()
    
    return loss
```

**作用**：
- 防止任何单个 token 的 logit 变化过大
- 确保 steering 后，所有 token 的概率分布仍然合理
- 默认 `max_logit_change = 2.0`，意味着单个 token 的 logit 变化不超过 2.0

**效果**：
- 原始 TSV：某个 token 的 logit 变化可达 +472
- 新 TSV：所有 token 的 logit 变化都在 ±2.5 以内

---

### 3. KL Divergence Regularization 详解

```python
def compute_kl_regularization(hidden_states, tsv_vector, lm_head_weight, alpha):
    """
    计算：steering 前后的输出概率分布的差异
    
    我们希望 steering 后，输出分布不要变化太大。
    """
    # 原始分布
    original_logits = hidden_states @ lm_head.T
    original_probs = softmax(original_logits)
    
    # Steering 后的分布
    steered_hidden = hidden_states + alpha * tsv_vector
    steered_logits = steered_hidden @ lm_head.T
    steered_probs = softmax(steered_logits)
    
    # KL 散度：衡量两个分布的差异
    kl_loss = KL(original_probs || steered_probs)
    
    return kl_loss
```

**作用**：
- 确保 steering 不会完全改变模型的输出行为
- 保持模型的"自然性"和"流畅性"
- 防止 steering 过度修正

---

## 为什么需要这些约束？

### 问题根源

**Hidden Space ≠ Vocab Space**

- **Hidden Space**：高维连续空间（如 2048 维），TSV 在这里分离簇
- **Vocab Space**：离散的词汇空间（如 50257 个 token），通过 `lm_head` 投影得到

**关键洞察**：在隐藏空间中"好"的方向，投影到词汇空间后可能变成"坏"的方向。

### 具体例子

假设 TSV 向量在隐藏空间中指向"truthful"方向，但：

1. **如果 TSV 与 `lm_head.weight[49151]`（"NASL" token）高度对齐**：
   - Steering 时：`delta_logits[49151] = α * tsv @ lm_head[49151]` 会非常大
   - 结果：NASL token 的概率变成 100%

2. **如果 TSV 与多个常见 token 的权重都对齐**：
   - 这些 token 的 logit 都会大幅增加
   - 结果：输出分布完全扭曲

### 解决方案

通过 **Logits Regularization** 和 **KL Regularization**，我们：

1. **限制 logit 变化**：确保没有单个 token 会"爆炸"
2. **保持分布稳定**：确保 steering 后的分布与原始分布相似
3. **平衡分离和稳定性**：在保持簇分离的同时，避免过度扭曲输出

---

## 训练过程对比

### 原始训练

```
Epoch 1: OT Loss = 0.65
Epoch 2: OT Loss = 0.60
...
Epoch 50: OT Loss = 0.55
```

**只关注**：hidden space 中的簇分离越来越好

### 新训练

```
Epoch 1: OT Loss = 0.65, Logits Reg = 0.05, KL Reg = 0.02, Max Δlogit = 1.5
Epoch 2: OT Loss = 0.60, Logits Reg = 0.03, KL Reg = 0.01, Max Δlogit = 1.8
...
Epoch 50: OT Loss = 0.55, Logits Reg = 0.00, KL Reg = 0.00, Max Δlogit = 2.5
```

**同时关注**：
- 簇分离（OT Loss 下降）
- Logit 变化（Logits Reg 被抑制）
- 分布稳定性（KL Reg 被抑制）
- 实际效果（Max Δlogit 在合理范围内）

---

## 实验结果对比

| 指标 | 原始 TSV | 新 TSV（带约束） |
|------|---------|----------------|
| **Max Δlogit** | 472.5 | 2.5 |
| **生成质量** | ❌ "NASL NASL..." | ✅ 正常文本 |
| **Accuracy** | 0.200 | 0.333 |
| **Hallucination Rate** | 0.800 | 0.667 |

---

## 总结

### 原始方法的局限

1. **只优化隐藏空间**：忽略了投影到词汇空间后的效果
2. **无约束**：TSV 可以学到任意方向，包括会导致 logit 爆炸的方向
3. **实际应用失败**：虽然训练指标好，但实际生成时完全失败

### 新方法的改进

1. **多目标优化**：同时优化簇分离和 logits 稳定性
2. **显式约束**：通过正则化项限制 TSV 的方向
3. **实际应用成功**：训练出的 TSV 可以正常用于 steering

### 核心思想

> **"在隐藏空间中分离簇" 和 "在词汇空间中保持稳定" 同样重要！**

我们不能只优化隐藏空间，还必须考虑当 TSV 通过 `lm_head` 投影后会发生什么。这就是为什么需要加入 `lm_head` 约束的原因。

---

## 代码位置

- **原始训练**：`tsv_main.py` (第 66-250 行)
- **新训练**：`experiments/probe_controlled_tsv/train_tsv_constrained.py`
- **关键函数**：
  - `compute_logits_regularization()` (第 111-142 行)
  - `compute_kl_regularization()` (第 145-179 行)

