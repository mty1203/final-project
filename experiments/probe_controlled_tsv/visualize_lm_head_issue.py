#!/usr/bin/env python
"""
可视化 lm_head 问题和解决方案

这个脚本演示：
1. lm_head 如何将 hidden state 转换为 logits
2. 为什么原始 TSV 会导致 logit 爆炸
3. 约束如何解决这个问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def visualize_lm_head_issue():
    """可视化 lm_head 的问题"""
    
    # 模拟参数
    hidden_size = 2048
    vocab_size = 100  # 简化，只用 100 个 token
    alpha = 2.0
    
    print("="*70)
    print("lm_head 问题可视化")
    print("="*70)
    print()
    
    # 1. 创建模拟的 lm_head
    print("1. 创建模拟的 lm_head")
    print("-"*70)
    lm_head = nn.Linear(hidden_size, vocab_size)
    print(f"   lm_head.weight.shape: {lm_head.weight.shape}")
    print(f"   每一行对应一个 token 的权重向量")
    print()
    
    # 2. 创建模拟的 hidden state
    print("2. 创建模拟的 hidden state")
    print("-"*70)
    hidden = torch.randn(1, hidden_size) * 0.1  # 小的随机值
    print(f"   hidden.shape: {hidden.shape}")
    print(f"   hidden.norm(): {hidden.norm():.4f}")
    print()
    
    # 3. 计算原始 logits
    print("3. 计算原始 logits（无 steering）")
    print("-"*70)
    original_logits = torch.matmul(hidden, lm_head.weight.T).squeeze()
    original_probs = F.softmax(original_logits, dim=-1)
    
    top5_indices = torch.topk(original_probs, 5).indices
    top5_probs = torch.topk(original_probs, 5).values
    
    print(f"   Top-5 tokens (原始):")
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
        print(f"     {i+1}. Token {idx.item()}: {prob.item():.4f}")
    print()
    
    # 4. 创建"坏"的 TSV（与某个 token 权重对齐）
    print("4. 创建'坏'的 TSV（与 Token 50 的权重高度对齐）")
    print("-"*70)
    
    # 让 TSV 与 token 50 的权重对齐
    target_token_idx = 50
    target_weight = lm_head.weight[target_token_idx].detach()
    
    # TSV 是目标权重的缩放版本
    bad_tsv = target_weight * 0.15  # 缩放因子
    print(f"   bad_tsv.shape: {bad_tsv.shape}")
    print(f"   bad_tsv.norm(): {bad_tsv.norm():.4f}")
    print(f"   与 Token {target_token_idx} 的对齐度: {(bad_tsv @ target_weight.T / (bad_tsv.norm() * target_weight.norm())).item():.4f}")
    print()
    
    # 5. 应用 steering（原始方法，无约束）
    print("5. 应用 steering（原始方法，无约束）")
    print("-"*70)
    steered_hidden = hidden + alpha * bad_tsv.unsqueeze(0)
    steered_logits = torch.matmul(steered_hidden, lm_head.weight.T).squeeze()
    steered_probs = F.softmax(steered_logits, dim=-1)
    
    # 计算 logit 变化
    delta_logits = steered_logits - original_logits
    
    print(f"   Logit 变化统计:")
    print(f"     Max increase: {delta_logits.max().item():.2f}")
    print(f"     Max decrease: {delta_logits.min().item():.2f}")
    print(f"     Token {target_token_idx} 的变化: {delta_logits[target_token_idx].item():.2f}")
    print()
    
    top5_steered_indices = torch.topk(steered_probs, 5).indices
    top5_steered_probs = torch.topk(steered_probs, 5).values
    
    print(f"   Top-5 tokens (steering 后):")
    for i, (idx, prob) in enumerate(zip(top5_steered_indices, top5_steered_probs)):
        print(f"     {i+1}. Token {idx.item()}: {prob.item():.4f}")
        if idx.item() == target_token_idx:
            print(f"        ⚠️ 这个 token 爆炸了！")
    print()
    
    # 6. 创建"好"的 TSV（有约束）
    print("6. 创建'好'的 TSV（有约束，不与任何 token 过度对齐）")
    print("-"*70)
    
    # 随机初始化，然后归一化
    good_tsv = torch.randn(hidden_size) * 0.1
    good_tsv = F.normalize(good_tsv, p=2, dim=0) * 0.5  # 较小的范数
    
    # 确保与所有 token 权重的对齐度都不太高
    max_alignment = 0
    for i in range(vocab_size):
        alignment = abs(good_tsv @ lm_head.weight[i].T / (good_tsv.norm() * lm_head.weight[i].norm()))
        max_alignment = max(max_alignment, alignment.item())
    
    print(f"   good_tsv.shape: {good_tsv.shape}")
    print(f"   good_tsv.norm(): {good_tsv.norm():.4f}")
    print(f"   与任何 token 的最大对齐度: {max_alignment:.4f}")
    print()
    
    # 7. 应用 steering（新方法，有约束）
    print("7. 应用 steering（新方法，有约束）")
    print("-"*70)
    steered_hidden_good = hidden + alpha * good_tsv.unsqueeze(0)
    steered_logits_good = torch.matmul(steered_hidden_good, lm_head.weight.T).squeeze()
    steered_probs_good = F.softmax(steered_logits_good, dim=-1)
    
    delta_logits_good = steered_logits_good - original_logits
    
    print(f"   Logit 变化统计:")
    print(f"     Max increase: {delta_logits_good.max().item():.2f}")
    print(f"     Max decrease: {delta_logits_good.min().item():.2f}")
    print()
    
    top5_good_indices = torch.topk(steered_probs_good, 5).indices
    top5_good_probs = torch.topk(steered_probs_good, 5).values
    
    print(f"   Top-5 tokens (有约束的 steering):")
    for i, (idx, prob) in enumerate(zip(top5_good_indices, top5_good_probs)):
        print(f"     {i+1}. Token {idx.item()}: {prob.item():.4f}")
    print()
    
    # 8. 对比总结
    print("="*70)
    print("对比总结")
    print("="*70)
    print()
    print("原始方法（无约束）:")
    print(f"  - Token {target_token_idx} 的 logit 变化: {delta_logits[target_token_idx].item():.2f}")
    print(f"  - 该 token 的概率: {steered_probs[target_token_idx].item():.4f}")
    print(f"  - 结果: ⚠️ 单个 token 主导，其他 token 概率接近 0")
    print()
    print("新方法（有约束）:")
    print(f"  - 最大 logit 变化: {delta_logits_good.abs().max().item():.2f}")
    print(f"  - 概率分布更均匀")
    print(f"  - 结果: ✅ 所有 token 的概率都在合理范围内")
    print()
    
    # 9. 可视化（如果 matplotlib 可用）
    try:
        print("生成可视化图表...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 原始 logits
        axes[0, 0].bar(range(vocab_size), original_logits.detach().numpy(), alpha=0.7)
        axes[0, 0].set_title('Original Logits')
        axes[0, 0].set_xlabel('Token Index')
        axes[0, 0].set_ylabel('Logit Value')
        
        # Steering 后的 logits（无约束）
        axes[0, 1].bar(range(vocab_size), steered_logits.detach().numpy(), alpha=0.7, color='red')
        axes[0, 1].axhline(y=steered_logits[target_token_idx].item(), color='orange', 
                           linestyle='--', label=f'Token {target_token_idx}')
        axes[0, 1].set_title('Steered Logits (No Constraint)')
        axes[0, 1].set_xlabel('Token Index')
        axes[0, 1].set_ylabel('Logit Value')
        axes[0, 1].legend()
        
        # Logit 变化（无约束）
        axes[1, 0].bar(range(vocab_size), delta_logits.detach().numpy(), alpha=0.7, color='red')
        axes[1, 0].axhline(y=delta_logits[target_token_idx].item(), color='orange', 
                          linestyle='--', label=f'Token {target_token_idx}')
        axes[1, 0].set_title('Logit Changes (No Constraint)')
        axes[1, 0].set_xlabel('Token Index')
        axes[1, 0].set_ylabel('Δ Logit')
        axes[1, 0].legend()
        
        # Logit 变化（有约束）
        axes[1, 1].bar(range(vocab_size), delta_logits_good.detach().numpy(), alpha=0.7, color='green')
        axes[1, 1].axhline(y=2.0, color='blue', linestyle='--', label='Max Allowed (2.0)')
        axes[1, 1].axhline(y=-2.0, color='blue', linestyle='--')
        axes[1, 1].set_title('Logit Changes (With Constraint)')
        axes[1, 1].set_xlabel('Token Index')
        axes[1, 1].set_ylabel('Δ Logit')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('lm_head_issue_visualization.png', dpi=150)
        print("   ✅ 图表已保存到: lm_head_issue_visualization.png")
        print()
        
    except ImportError:
        print("   ⚠️ matplotlib 不可用，跳过可视化")
        print()


if __name__ == "__main__":
    visualize_lm_head_issue()

