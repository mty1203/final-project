#!/usr/bin/env python
"""测试 TSV steering 是否真的在改变 logits"""

import torch
import numpy as np

# 加载 TSV
tsv_data = torch.load('artifacts/gpt-neo-1.3B_tqa_tsv.pt', map_location='cpu')
tsv_vec = tsv_data['tsv_vectors'][9]

print("=" * 60)
print("测试 TSV Steering 效果")
print("=" * 60)
print()

# 模拟隐藏状态
hidden_size = 2048
hidden_state = torch.randn(1, hidden_size)

print("1. 原始隐藏状态统计:")
print(f"   范数: {hidden_state.norm():.6f}")
print()

# 测试不同 alpha 值的效果
risk = 0.9
for alpha in [0.0, 0.5, 1.0, 2.0]:
    steered = hidden_state + alpha * risk * tsv_vec
    diff = steered - hidden_state
    
    print(f"2. Alpha={alpha}, Risk={risk}:")
    print(f"   Steered 范数: {steered.norm():.6f}")
    print(f"   偏移量范数: {diff.norm():.6f}")
    print(f"   相对偏移: {(diff.norm() / hidden_state.norm() * 100):.2f}%")
    
    # 模拟 logits 变化 (简化版)
    # 假设 lm_head 是一个随机矩阵
    vocab_size = 50257
    lm_head_weight = torch.randn(vocab_size, hidden_size) * 0.01
    
    logits_original = torch.matmul(hidden_state, lm_head_weight.T)
    logits_steered = torch.matmul(steered, lm_head_weight.T)
    
    logits_diff = (logits_steered - logits_original).abs()
    
    print(f"   Logits 平均变化: {logits_diff.mean():.6f}")
    print(f"   Logits 最大变化: {logits_diff.max():.6f}")
    
    # Top-5 token 是否改变
    top5_orig = torch.topk(logits_original, 5).indices
    top5_steer = torch.topk(logits_steered, 5).indices
    overlap = len(set(top5_orig[0].tolist()) & set(top5_steer[0].tolist()))
    
    print(f"   Top-5 重叠数: {overlap}/5")
    print()

print("=" * 60)
print("结论:")
print("  - Alpha=0.0 应该完全不改变 logits")
print("  - Alpha>0 应该改变 logits 和 top token")
print("  - 如果实验中 baseline 和 steering 完全相同,")
print("    可能是随机种子导致的采样一致性问题")
print("=" * 60)

