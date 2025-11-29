# 设计验证：当前实现 vs 提案设计

## 提案设计（三步法）

### Step 1: Find the "Truth" Direction
> We first learn a single direction vector in the model's activation space, the Truthfulness Separator Vector (vTSV), that points from "hallucinated" states toward "truthful" ones.

### Step 2: Predict Hallucination Risk
> We train a tiny, efficient probe that monitors the LLM's internal states at each step. The probe outputs a real-time risk score (r̂t).

### Step 3: Steer Adaptively
> We use the risk score to control the steering strength (αt). If the risk is low, we do nothing (αt = 0). If the risk is high, we nudge the model's activations in the direction of vTSV.

---

## 当前实现分析

### ✅ Step 1: TSV 学习 - **符合设计**

**代码位置**: `train_tsv_constrained.py`

```python
# 学习 TSV 向量
self.tsv = nn.Parameter(torch.zeros(hidden_size, device=device), requires_grad=True)

# 训练目标：分离 truthful 和 hallucinated 样本
ot_loss, similarities = compute_ot_loss_cos(
    steered_rep_norm, self.centroids, labels_oh, batch_size, cos_temp
)
```

**符合程度**: ✅ **完全符合**
- 学习单一方向向量 vTSV
- 从 hallucinated 指向 truthful
- 使用 Optimal Transport Loss 进行分离

**改进点**:
- 加入了 lm_head 约束（这是对原始 TSV 的改进，不影响设计思路）

---

### ✅ Step 2: Probe 预测风险 - **符合设计**

**代码位置**: `models/probe.py`, `train_probe.py`

```python
class LinearProbe(nn.Module):
    def __init__(self, hidden_size: int):
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.linear(hidden_states).squeeze(-1)
        return torch.sigmoid(logits)  # 输出 [0, 1] 的风险分数
```

**在生成时的使用** (`models/steering.py`):

```python
def compute_risk(self, hidden_states: torch.Tensor) -> float:
    """Compute hallucination risk from hidden states."""
    hidden = hidden_states.float()
    risk = self.probe(hidden)  # 输出 r̂t
    return risk.item()
```

**符合程度**: ✅ **完全符合**
- 轻量级 probe（单层线性或 MLP）
- 实时监控每个 step 的 hidden states
- 输出风险分数 r̂t ∈ [0, 1]

---

### ⚠️ Step 3: 自适应 Steering - **部分符合，需要修正**

**代码位置**: `models/steering.py`

#### 当前实现

```python
def get_alpha(self, risk: float) -> float:
    if self.config.mode == SteeringMode.PROBE_TSV:
        return self.alpha_scheduler(risk)  # α = f(risk)
    return 0.0

def should_steer(self, risk: float) -> bool:
    return risk >= self.config.risk_threshold  # 只有 risk >= threshold 才 steer
```

**AdaptiveAlphaScheduler**:

```python
if self.schedule == "linear":
    alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * risk
```

#### 问题分析

| 设计要求 | 当前实现 | 符合？ |
|---------|---------|-------|
| 风险低时 αt = 0 | `should_steer()` 检查 `risk >= threshold` | ⚠️ 部分符合 |
| 风险高时 αt > 0 | `get_alpha()` 返回 `f(risk)` | ✅ 符合 |
| α 与 risk 成正比 | `alpha = alpha_min + (alpha_max - alpha_min) * risk` | ⚠️ 有问题 |

**问题**：当前实现有两个问题：

1. **threshold 逻辑不完整**：
   - 设计要求：`risk 低 → α = 0`
   - 当前：只有 `risk >= threshold` 才 steer，但 `alpha_min` 可能不是 0

2. **alpha 计算不符合"only when needed"**：
   - 设计要求：`α = 0 if risk < threshold else f(risk)`
   - 当前：`alpha = alpha_min + (alpha_max - alpha_min) * risk`（线性，总是 > 0）

---

## 修正建议

### 修正后的 α 计算逻辑

```python
def get_alpha(self, risk: float) -> float:
    """
    设计要求：
    - 如果 risk < threshold: α = 0（不 steer）
    - 如果 risk >= threshold: α = f(risk)（自适应 steer）
    """
    if self.config.mode == SteeringMode.PROBE_TSV:
        if risk < self.config.risk_threshold:
            return 0.0  # 风险低，不 steer
        else:
            # 风险高，α 与 risk 成正比
            # 将 risk 从 [threshold, 1] 映射到 [0, alpha_max]
            normalized_risk = (risk - self.config.risk_threshold) / (1.0 - self.config.risk_threshold)
            return self.config.alpha_max * normalized_risk
    return 0.0
```

### 修正后的流程

```
每个 token 生成时：
  1. 获取 hidden_states
  2. risk = probe(hidden_states)  # 预测风险
  3. if risk < threshold:
       α = 0  # 不 steer
     else:
       α = alpha_max * (risk - threshold) / (1 - threshold)  # 自适应 steer
  4. steered_logits = logits + α * tsv_logit_shift
```

---

## 当前实现 vs 设计对比表

| 设计要求 | 当前实现 | 状态 |
|---------|---------|------|
| **Step 1: TSV 学习** | | |
| 学习单一方向向量 | ✅ `self.tsv = nn.Parameter(...)` | ✅ |
| 从 hallucinated → truthful | ✅ OT Loss 分离 | ✅ |
| **Step 2: Probe 预测** | | |
| 轻量级 probe | ✅ LinearProbe / MLPProbe | ✅ |
| 实时监控 hidden states | ✅ `compute_risk(hidden_states)` | ✅ |
| 输出风险分数 r̂t | ✅ `torch.sigmoid(logits)` | ✅ |
| **Step 3: 自适应 Steering** | | |
| risk 低 → α = 0 | ⚠️ `alpha_min` 可能 > 0 | ⚠️ 需修正 |
| risk 高 → α > 0 | ✅ `alpha_scheduler(risk)` | ✅ |
| α 与 risk 成正比 | ⚠️ 线性但不从 0 开始 | ⚠️ 需修正 |
| "only when needed" | ⚠️ 依赖 threshold 检查 | ⚠️ 需明确 |

---

## 结论

**总体符合程度**: 85%

- ✅ **Step 1 (TSV 学习)**: 完全符合
- ✅ **Step 2 (Probe 预测)**: 完全符合
- ⚠️ **Step 3 (自适应 Steering)**: 基本符合，但 α 的计算逻辑需要调整

**需要修正的地方**:
1. 确保 `alpha_min = 0`
2. 修改 α 计算：`risk < threshold → α = 0`
3. 让 α 从 threshold 开始线性增长到 alpha_max

---

## 修正代码

需要修改 `models/steering.py` 中的 `AdaptiveAlphaScheduler` 或 `get_alpha` 方法。

