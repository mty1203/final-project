# lm_head 详解：作用、问题与解决方案

## 1. 什么是 lm_head？

### 基本概念

`lm_head`（Language Model Head）是 Transformer 语言模型的**最后一层**，负责将隐藏状态（hidden states）转换为词汇概率分布。

```
输入文本 → Tokenizer → Embeddings → Transformer Layers → Hidden States → lm_head → Logits → Softmax → Token Probabilities
                                                                              ↑
                                                                        这就是 lm_head！
```

### 数学表示

```python
# 假设模型配置
hidden_size = 2048      # 隐藏层维度
vocab_size = 50257      # 词汇表大小（GPT-Neo-1.3B）

# lm_head 是一个线性层
lm_head = nn.Linear(hidden_size, vocab_size)
# 权重形状: [50257, 2048]

# 前向传播
hidden_state = model(...)  # [batch_size, seq_len, 2048]
logits = hidden_state @ lm_head.weight.T  # [batch_size, seq_len, 50257]
probs = softmax(logits)   # [batch_size, seq_len, 50257]
```

### 具体例子

假设我们有一个隐藏状态向量 `h = [0.1, -0.3, 0.5, ..., 0.2]`（2048维）：

```python
# lm_head 的每一行对应一个 token
lm_head.weight[0]  # "the" token 的权重向量 [2048维]
lm_head.weight[1]  # "a" token 的权重向量
lm_head.weight[49151]  # "NASL" token 的权重向量
...

# 计算每个 token 的 logit
logit_the = h @ lm_head.weight[0].T    # 点积
logit_a = h @ lm_head.weight[1].T
logit_nasl = h @ lm_head.weight[49151].T
...

# 所有 logits 组成一个向量
logits = [logit_the, logit_a, ..., logit_nasl, ...]  # [50257维]

# Softmax 后得到概率
probs = softmax(logits)
# probs[0] = P("the" | context)
# probs[1] = P("a" | context)
# probs[49151] = P("NASL" | context)
```

---

## 2. lm_head 的作用

### 作用 1：将连续空间映射到离散空间

- **输入**：隐藏状态是**连续的高维向量**（2048维实数空间）
- **输出**：词汇概率是**离散的分布**（50257个 token 的概率）

```
Hidden Space (连续)     →    Vocab Space (离散)
[0.1, -0.3, 0.5, ...]  →    [P("the"), P("a"), ..., P("NASL"), ...]
```

### 作用 2：决定下一个 token 的概率

对于每个可能的 token，`lm_head` 计算一个 **logit**（未归一化的分数），然后通过 softmax 转换为概率。

```python
# 例子：生成下一个 token
context = "The capital of France is"
hidden = model.encode(context)  # [1, 2048]

# lm_head 计算所有 token 的 logits
logits = hidden @ lm_head.weight.T  # [1, 50257]

# 假设结果：
logits["Paris"] = 10.5
logits["London"] = 2.3
logits["Berlin"] = 1.8
logits["NASL"] = -16.8
...

# Softmax 后
probs["Paris"] = 0.85    # 85% 概率
probs["London"] = 0.08   # 8% 概率
probs["Berlin"] = 0.05   # 5% 概率
probs["NASL"] = 0.0001   # 几乎为 0
```

**模型会采样 "Paris"**，因为它的概率最高。

---

## 3. 原始 TSV 训练的问题

### 问题场景

原始 TSV 训练只优化**隐藏空间**中的簇分离，完全忽略了 `lm_head` 的存在。

```python
# 原始训练过程
# 1. 在隐藏空间中学习 TSV 向量
tsv = learn_tsv_in_hidden_space()  # [2048维]

# 2. 训练目标：分离 truthful 和 hallucinated 样本
loss = optimal_transport_loss(truthful_hidden, hallucinated_hidden, tsv)
# ✅ 这个损失只关心隐藏空间中的分离

# 3. 但是！当应用 steering 时：
steered_hidden = hidden + α * tsv
steered_logits = steered_hidden @ lm_head.weight.T  # ⚠️ 问题在这里！
```

### 问题的根源

**关键洞察**：在隐藏空间中"好"的方向，投影到词汇空间后可能变成"坏"的方向！

#### 具体例子

假设我们学到的 TSV 向量是：
```python
tsv = [0.1, 0.2, -0.3, ..., 0.15]  # [2048维]
```

这个向量在隐藏空间中能很好地区分 truthful 和 hallucinated 样本。

**但是**，当我们计算它对 logits 的影响时：

```python
# 计算 TSV 对每个 token 的 logit 影响
delta_logits = α * tsv @ lm_head.weight.T  # [50257维]

# 假设 α = 2.0
delta_logits = 2.0 * tsv @ lm_head.weight.T

# 结果可能是：
delta_logits["Paris"] = +0.5      # 合理
delta_logits["London"] = -0.3     # 合理
delta_logits["NASL"] = +472.5     # ⚠️ 爆炸！
delta_logits["the"] = +1.2        # 合理
...
```

**为什么会爆炸？**

因为 TSV 向量与 `lm_head.weight[49151]`（"NASL" token 的权重）**高度对齐**！

```python
# 点积很大意味着两个向量方向相似
similarity = tsv @ lm_head.weight[49151].T
# 如果 similarity ≈ 236，那么：
delta_logits["NASL"] = 2.0 * 236 = 472  # 爆炸！
```

### 实际发生的情况

```python
# 原始 logits（正常）
original_logits = {
    "Paris": 10.5,
    "London": 2.3,
    "Berlin": 1.8,
    "NASL": -16.8,  # 很低，几乎不可能被采样
    ...
}

# 应用 steering 后
steered_logits = original_logits + delta_logits
steered_logits = {
    "Paris": 10.5 + 0.5 = 11.0,
    "London": 2.3 - 0.3 = 2.0,
    "Berlin": 1.8 + 0.2 = 2.0,
    "NASL": -16.8 + 472.5 = 455.7,  # ⚠️ 爆炸！
    ...
}

# Softmax 后
probs = softmax(steered_logits)
probs = {
    "Paris": 0.0001,    # 几乎为 0
    "London": 0.0001,   # 几乎为 0
    "Berlin": 0.0001,   # 几乎为 0
    "NASL": 0.9997,     # 99.97%！其他所有 token 几乎为 0
    ...
}
```

**结果**：模型**只能生成 "NASL"**，其他所有 token 的概率都被压到接近 0！

---

## 4. 为什么会出现这个问题？

### 原因 1：隐藏空间和词汇空间的不匹配

```
Hidden Space (2048维连续空间)
  ↓ 通过 lm_head 投影
Vocab Space (50257维离散空间)
```

- **隐藏空间**：高维、连续、语义丰富
- **词汇空间**：离散、稀疏、每个维度对应一个 token

**问题**：在隐藏空间中"分离簇"的方向，不一定在词汇空间中"保持稳定"。

### 原因 2：lm_head 权重的不均匀性

`lm_head` 的权重是**预训练时学到的**，不同 token 的权重向量可能差异很大：

```python
# 常见 token（如 "the", "a"）的权重
lm_head.weight["the"]  # 可能接近 [0.1, 0.2, ..., 0.1]

# 罕见 token（如 "NASL"）的权重
lm_head.weight["NASL"]  # 可能接近 [10.0, -5.0, ..., 8.0]  # 更大的值！
```

如果 TSV 向量恰好与某个罕见 token 的权重对齐，就会导致该 token 的 logit 爆炸。

### 原因 3：训练目标不完整

原始 TSV 训练的目标函数：

```python
loss = optimal_transport_loss(hidden_space_separation)
# 只优化隐藏空间的分离
# ❌ 没有考虑投影到词汇空间后的效果
```

**缺失的部分**：
- 没有约束 logit 变化
- 没有约束输出分布
- 没有考虑实际应用时的效果

---

## 5. 解决方案：lm_head 约束

### 核心思想

**在训练时就考虑 `lm_head` 的影响**，而不是等到应用时才发现问题。

### 方法 1：Logits Regularization

```python
def compute_logits_regularization(tsv, lm_head_weight, alpha, max_logit_change):
    """
    计算：如果应用 steering，每个 token 的 logit 会变化多少？
    如果变化超过阈值，就惩罚它。
    """
    # 计算 logit 变化
    delta_logits = alpha * tsv @ lm_head_weight.T  # [50257]
    
    # 只惩罚超过阈值的部分
    excess = relu(|delta_logits| - max_logit_change)
    loss = excess.mean()
    
    return loss
```

**效果**：
- 训练时，如果 TSV 会导致某个 token 的 logit 变化 > 2.0，就惩罚它
- 最终学到的 TSV 不会与任何 token 的权重过度对齐
- Max Δlogit 被限制在合理范围内（~2.5）

### 方法 2：KL Divergence Regularization

```python
def compute_kl_regularization(hidden, tsv, lm_head_weight, alpha):
    """
    计算：steering 前后的输出分布差异
    如果差异太大，就惩罚它。
    """
    # 原始分布
    original_logits = hidden @ lm_head_weight.T
    original_probs = softmax(original_logits)
    
    # Steering 后的分布
    steered_hidden = hidden + alpha * tsv
    steered_logits = steered_hidden @ lm_head_weight.T
    steered_probs = softmax(steered_logits)
    
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
# 原始训练（只有 OT Loss）
loss = ot_loss

# 新训练（OT Loss + 两个约束）
loss = (
    ot_loss +                           # 主目标：分离簇
    0.2 * logits_reg_loss +             # 约束1：限制 logit 变化
    0.1 * kl_reg_loss                   # 约束2：保持分布稳定
)
```

---

## 6. 可视化对比

### 原始 TSV（无约束）

```
Hidden Space:
  Truthful: ●●●
  Hallucinated: ○○○
  TSV: → (很好的分离方向)

投影到 Vocab Space:
  "Paris": logit = 10.5 + 0.5 = 11.0
  "London": logit = 2.3 - 0.3 = 2.0
  "NASL": logit = -16.8 + 472.5 = 455.7  ⚠️ 爆炸！
  
结果：
  P("NASL") = 99.97%
  P(其他所有 token) ≈ 0%
  
生成：NASL NASL NASL NASL...
```

### 新 TSV（有约束）

```
Hidden Space:
  Truthful: ●●●
  Hallucinated: ○○○
  TSV: → (分离方向，但被约束)

投影到 Vocab Space:
  "Paris": logit = 10.5 + 0.5 = 11.0
  "London": logit = 2.3 - 0.3 = 2.0
  "NASL": logit = -16.8 + 2.0 = -14.8  ✅ 合理
  
结果：
  P("Paris") = 85%
  P("London") = 8%
  P("NASL") = 0.0001%
  
生成：Paris (正常！)
```

---

## 7. 数学公式总结

### 原始方法

```
训练目标：
  min_tsv OT_Loss(hidden + tsv, centroids)
  
应用时：
  steered_logits = (hidden + α * tsv) @ lm_head.T
  ⚠️ 没有约束，可能爆炸
```

### 新方法

```
训练目标：
  min_tsv [
    OT_Loss(hidden + tsv, centroids) +
    λ1 * Logits_Reg(tsv, lm_head) +
    λ2 * KL_Reg(hidden, hidden + tsv, lm_head)
  ]
  
应用时：
  steered_logits = (hidden + α * tsv) @ lm_head.T
  ✅ 有约束，稳定
```

---

## 8. 关键要点

1. **lm_head 的作用**：将隐藏状态转换为词汇概率分布
2. **原始问题**：TSV 在隐藏空间中很好，但投影到词汇空间后导致 logit 爆炸
3. **根本原因**：TSV 可能与某个 token 的权重过度对齐
4. **解决方案**：在训练时加入 logits 和 KL 约束
5. **效果**：Max Δlogit 从 472 降到 2.5，生成恢复正常

---

## 9. 类比理解

想象你在一个**高维空间**中学习"如何区分好苹果和坏苹果"：

- **原始方法**：只关心在这个空间中如何分离，完全不考虑"如何把苹果拿出来"
- **问题**：当你真的去"拿苹果"时（投影到输出空间），可能拿到的是"炸弹"而不是"苹果"
- **新方法**：在训练时就考虑"拿苹果"的过程，确保拿到的总是"苹果"而不是"炸弹"

`lm_head` 就是那个"拿苹果"的机制！

