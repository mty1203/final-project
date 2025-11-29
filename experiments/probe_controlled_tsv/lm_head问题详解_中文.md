# lm_head 作用与问题详解（中文版）

## 📚 目录

1. [lm_head 是什么？](#1-lm_head-是什么)
2. [lm_head 的作用](#2-lm_head-的作用)
3. [原始 TSV 训练的问题](#3-原始-tsv-训练的问题)
4. [为什么会出现问题？](#4-为什么会出现问题)
5. [解决方案](#5-解决方案)
6. [实际例子](#6-实际例子)

---

## 1. lm_head 是什么？

### 简单理解

`lm_head` 是语言模型的**最后一层**，负责把模型内部的"理解"（hidden state）转换成"下一个词是什么"的概率。

```
你输入："The capital of France is"
         ↓
模型理解：[0.1, -0.3, 0.5, ..., 0.2]  ← 这是 hidden state（2048个数字）
         ↓
lm_head 转换
         ↓
词汇概率：[P("Paris")=0.85, P("London")=0.08, ..., P("NASL")=0.0001]
         ↓
模型输出："Paris"
```

### 数学表示

```python
# lm_head 是一个矩阵
lm_head.weight.shape = [50257, 2048]
# 50257 = 词汇表大小（所有可能的 token）
# 2048 = 隐藏层维度

# 计算过程
hidden_state = [0.1, -0.3, 0.5, ..., 0.2]  # 2048维
logits = hidden_state @ lm_head.weight.T    # 得到 50257 个 logit
probs = softmax(logits)                     # 转换为概率
```

**关键点**：`lm_head` 的每一行对应一个 token，比如：
- 第 0 行 → "the" token
- 第 1 行 → "a" token
- 第 49151 行 → "NASL" token

---

## 2. lm_head 的作用

### 作用 1：空间转换

- **输入空间**：隐藏状态是**连续的高维向量**（2048维实数）
- **输出空间**：词汇概率是**离散的分布**（50257个 token 的概率）

就像把"连续的颜色"转换成"离散的颜色名称"。

### 作用 2：决定下一个词

对于每个可能的 token，`lm_head` 计算一个分数（logit），然后转换成概率。

**例子**：
```python
输入："The capital of France is"
hidden = [0.1, -0.3, 0.5, ...]  # 模型的理解

通过 lm_head：
  logit["Paris"] = 10.5   →  P("Paris") = 85%
  logit["London"] = 2.3  →  P("London") = 8%
  logit["Berlin"] = 1.8  →  P("Berlin") = 5%
  logit["NASL"] = -16.8  →  P("NASL") = 0.0001%

模型选择："Paris"（概率最高）
```

---

## 3. 原始 TSV 训练的问题

### 问题场景

原始 TSV 训练**只关心隐藏空间中的分离**，完全忽略了 `lm_head` 的存在。

```python
# 训练过程
tsv = learn_tsv()  # 在隐藏空间中学习分离方向
loss = separation_loss(tsv)  # 只优化分离

# 应用时
steered_hidden = hidden + α * tsv
steered_logits = steered_hidden @ lm_head.weight.T  # ⚠️ 问题在这里！
```

### 具体问题

**场景**：我们学到的 TSV 向量在隐藏空间中能很好地区分 truthful 和 hallucinated 样本。

**但是**，当我们计算它对 logits 的影响时：

```python
# 计算 TSV 对每个 token 的影响
delta_logits = α * tsv @ lm_head.weight.T

# 结果可能是：
delta_logits["Paris"] = +0.5      # 合理
delta_logits["London"] = -0.3     # 合理
delta_logits["NASL"] = +472.5     # ⚠️ 爆炸！
```

**为什么会爆炸？**

因为 TSV 向量与 `lm_head.weight[49151]`（"NASL" token 的权重）**高度对齐**！

想象两个箭头：
- TSV 向量：→
- NASL token 的权重：→（几乎同一个方向）

当它们"对齐"时，点积会非常大：
```python
对齐度 = tsv @ lm_head.weight["NASL"].T
如果对齐度 ≈ 236，那么：
delta_logits["NASL"] = 2.0 * 236 = 472  # 爆炸！
```

### 实际发生的情况

```python
# 原始 logits（正常）
original_logits = {
    "Paris": 10.5,
    "London": 2.3,
    "NASL": -16.8,  # 很低，几乎不可能被采样
}

# 应用 steering 后
steered_logits = {
    "Paris": 11.0,
    "London": 2.0,
    "NASL": 455.7,  # ⚠️ 爆炸！
}

# Softmax 后
probs = {
    "Paris": 0.0001%,    # 几乎为 0
    "London": 0.0001%,   # 几乎为 0
    "NASL": 99.97%,      # 几乎 100%！
}
```

**结果**：模型**只能生成 "NASL"**，其他所有词的概率都被压到接近 0！

---

## 4. 为什么会出现问题？

### 原因 1：空间不匹配

```
隐藏空间（2048维连续空间）
  ↓ 通过 lm_head 投影
词汇空间（50257维离散空间）
```

- **隐藏空间**：高维、连续、语义丰富
- **词汇空间**：离散、稀疏、每个维度对应一个词

**问题**：在隐藏空间中"好"的方向，投影到词汇空间后可能变成"坏"的方向。

### 原因 2：权重不均匀

`lm_head` 的权重是预训练时学到的，不同 token 的权重可能差异很大：

```python
# 常见词（如 "the"）的权重
lm_head.weight["the"]  # 可能接近 [0.1, 0.2, ..., 0.1]

# 罕见词（如 "NASL"）的权重
lm_head.weight["NASL"]  # 可能接近 [10.0, -5.0, ..., 8.0]  # 更大的值！
```

如果 TSV 向量恰好与某个罕见词的权重对齐，就会导致该词的 logit 爆炸。

### 原因 3：训练目标不完整

原始 TSV 训练的目标：

```python
loss = separation_loss()  # 只优化隐藏空间的分离
# ❌ 没有考虑投影到词汇空间后的效果
```

**缺失的部分**：
- 没有约束 logit 变化
- 没有约束输出分布
- 没有考虑实际应用时的效果

---

## 5. 解决方案

### 核心思想

**在训练时就考虑 `lm_head` 的影响**，而不是等到应用时才发现问题。

### 方法 1：Logits Regularization（Logits 正则化）

```python
def compute_logits_regularization(tsv, lm_head_weight, alpha, max_logit_change):
    """
    计算：如果应用 steering，每个 token 的 logit 会变化多少？
    如果变化超过阈值（如 2.0），就惩罚它。
    """
    delta_logits = alpha * tsv @ lm_head_weight.T
    
    # 只惩罚超过阈值的部分
    excess = relu(|delta_logits| - max_logit_change)
    loss = excess.mean()
    
    return loss
```

**效果**：
- 训练时，如果 TSV 会导致某个 token 的 logit 变化 > 2.0，就惩罚它
- 最终学到的 TSV 不会与任何 token 的权重过度对齐
- Max Δlogit 被限制在合理范围内（~2.5）

### 方法 2：KL Divergence Regularization（KL 散度正则化）

```python
def compute_kl_regularization(hidden, tsv, lm_head_weight, alpha):
    """
    计算：steering 前后的输出分布差异
    如果差异太大，就惩罚它。
    """
    # 原始分布
    original_probs = softmax(hidden @ lm_head_weight.T)
    
    # Steering 后的分布
    steered_probs = softmax((hidden + alpha * tsv) @ lm_head_weight.T)
    
    # KL 散度：衡量两个分布的差异
    kl_loss = KL(original_probs || steered_probs)
    
    return kl_loss
```

**效果**：
- 确保 steering 后的分布与原始分布相似
- 防止输出分布完全扭曲
- 保持模型的"自然性"

### 完整的损失函数

```python
# 原始训练（只有分离损失）
loss = separation_loss

# 新训练（分离损失 + 两个约束）
loss = (
    separation_loss +              # 主目标：分离簇
    0.2 * logits_reg_loss +        # 约束1：限制 logit 变化
    0.1 * kl_reg_loss              # 约束2：保持分布稳定
)
```

---

## 6. 实际例子

### 例子 1：原始 TSV（无约束）

```
问题："What is the capital of France?"

隐藏空间：
  Truthful 样本: ●●●
  Hallucinated 样本: ○○○
  TSV 方向: →（很好的分离方向）

投影到词汇空间：
  "Paris": logit = 10.5 + 0.5 = 11.0
  "London": logit = 2.3 - 0.3 = 2.0
  "NASL": logit = -16.8 + 472.5 = 455.7  ⚠️ 爆炸！
  
结果：
  P("NASL") = 99.97%
  P(其他所有词) ≈ 0%
  
生成：NASL NASL NASL NASL...  ❌
```

### 例子 2：新 TSV（有约束）

```
问题："What is the capital of France?"

隐藏空间：
  Truthful 样本: ●●●
  Hallucinated 样本: ○○○
  TSV 方向: →（分离方向，但被约束）

投影到词汇空间：
  "Paris": logit = 10.5 + 0.5 = 11.0
  "London": logit = 2.3 - 0.3 = 2.0
  "NASL": logit = -16.8 + 2.0 = -14.8  ✅ 合理
  
结果：
  P("Paris") = 85%
  P("London") = 8%
  P("NASL") = 0.0001%
  
生成：Paris  ✅
```

---

## 7. 关键要点总结

1. **lm_head 的作用**：将隐藏状态转换为词汇概率分布
2. **原始问题**：TSV 在隐藏空间中很好，但投影到词汇空间后导致 logit 爆炸
3. **根本原因**：TSV 可能与某个 token 的权重过度对齐
4. **解决方案**：在训练时加入 logits 和 KL 约束
5. **效果**：Max Δlogit 从 472 降到 2.5，生成恢复正常

---

## 8. 类比理解

想象你在一个**高维空间**中学习"如何区分好苹果和坏苹果"：

- **原始方法**：只关心在这个空间中如何分离，完全不考虑"如何把苹果拿出来"
- **问题**：当你真的去"拿苹果"时（投影到输出空间），可能拿到的是"炸弹"而不是"苹果"
- **新方法**：在训练时就考虑"拿苹果"的过程，确保拿到的总是"苹果"而不是"炸弹"

`lm_head` 就是那个"拿苹果"的机制！

---

## 9. 数学公式对比

### 原始方法

```
训练目标：
  min_tsv Separation_Loss(hidden + tsv)
  
应用时：
  steered_logits = (hidden + α * tsv) @ lm_head.T
  ⚠️ 没有约束，可能爆炸
```

### 新方法

```
训练目标：
  min_tsv [
    Separation_Loss(hidden + tsv) +
    λ1 * Logits_Reg(tsv, lm_head) +
    λ2 * KL_Reg(hidden, hidden + tsv, lm_head)
  ]
  
应用时：
  steered_logits = (hidden + α * tsv) @ lm_head.T
  ✅ 有约束，稳定
```

---

## 10. 可视化

运行 `visualize_lm_head_issue.py` 可以看到：
- 原始 logits 分布
- Steering 后的 logits 分布（无约束 vs 有约束）
- Logit 变化的对比

图表已保存到：`lm_head_issue_visualization.png`

