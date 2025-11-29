# References and Related Work

## Direct References

### TSV (Truthfulness Separator Vector)

**Park et al., 2025** - "Steer LLM Latents for Hallucination Detection"
- Original TSV method that learns a direction vector in hidden space to separate truthful from hallucinated activations
- Uses Optimal Transport loss to learn cluster separation
- **Note**: The original paper does NOT include lm_head constraints

### Contrastive Activation Addition (CAA)

**Rimsky et al., 2024** - "Contrastive Activation Addition"
- Baseline method using mean activation differences between classes
- Simpler than learned TSV but serves as a useful comparison

### Adaptive Activation Steering

**Wang et al., 2024/25** - "Adaptive Activation Steering (ACT)"
- Uses adaptive intensity (α) for steering
- Replaces heuristic vector banks with learned vectors
- **Note**: ACT uses different vector selection methods, not TSV

---

## Our Contribution: lm_head Constrained TSV Training

### Novelty

The **lm_head constraint** in `train_tsv_constrained.py` is **our innovation** to solve the logit explosion problem observed in the original TSV training.

**Key insight**: The original TSV only optimizes hidden space separation, but when the TSV vector is projected through `lm_head` to vocab space, it can cause extreme logit changes (e.g., +472 for a single token).

**Our solution**: Add two regularization terms:
1. **Logits Regularization**: Limits per-token logit changes
2. **KL Divergence Regularization**: Maintains output distribution stability

### Why No Direct Literature?

This is a **practical fix** for a specific problem we encountered. While there's no direct literature on "TSV with lm_head constraints," the underlying ideas come from:

1. **Regularization in neural networks** (general ML theory)
2. **Distribution stability** (KL divergence is standard)
3. **Logit space constraints** (similar to temperature scaling, logit clipping)

---

## Related Work in Similar Areas

### 1. Logit Space Interventions

While not directly about TSV training, several papers work in logit space:

- **Contrastive Decoding** (Li et al., 2022): Uses logit differences between models
- **DExperts** (Liu et al., 2021): Trains expert models and combines logits
- **Logit Lens** (Nostalgebraist, 2020): Visualizes intermediate logits

**Connection**: These methods operate directly in logit space, which is more stable than hidden space steering.

### 2. Regularization in Transformer Training

- **DropHead** (Zhou et al., 2020): Regularizes attention heads
- **Token-Level Masking** (various): Regularizes token representations

**Connection**: Similar philosophy of adding constraints during training to prevent undesirable behaviors.

### 3. Activation Steering Methods

- **ACT** (Wang et al., 2024): Adaptive steering with vector banks
- **CAA** (Rimsky et al., 2024): Contrastive activation addition
- **Steering Vectors** (Turner et al., 2023): Learned directions for behavior control

**Connection**: All use hidden space steering, but none explicitly address the logit explosion problem.

---

## Theoretical Justification

### Why Logits Regularization?

**Problem**: When `steered_hidden = hidden + α * tsv`, the logit change is:
```
delta_logits = α * tsv @ lm_head.T
```

If `tsv` aligns with `lm_head[token_i]`, then `delta_logits[token_i]` can be extremely large.

**Solution**: Penalize `|delta_logits| > threshold` during training.

**Justification**: This is similar to:
- **L2 regularization** on weights (prevents extreme values)
- **Gradient clipping** (prevents extreme updates)
- **Logit clipping** in inference (prevents extreme outputs)

### Why KL Regularization?

**Problem**: Steering can completely change the output distribution.

**Solution**: Minimize KL divergence between original and steered distributions.

**Justification**: This is standard in:
- **Knowledge distillation** (Hinton et al., 2015): Uses KL to match distributions
- **Distribution matching** (various): Common regularization technique

---

## Citation Format

If you want to cite this work, you could say:

> "We extend the TSV method (Park et al., 2025) by adding lm_head constraints during training. Specifically, we add logits regularization to prevent extreme token probability changes and KL divergence regularization to maintain output distribution stability. This addresses the logit explosion problem observed when applying TSV steering in practice."

Or in a footnote:

> "The original TSV training (Park et al., 2025) only optimizes hidden space separation. We add constraints on the projection through lm_head to prevent logit explosion, which is a novel contribution."

---

## Future Work

Potential research directions:

1. **Theoretical analysis**: Why does logit explosion happen? Can we predict it?
2. **Better constraints**: Are there more principled ways to constrain TSV?
3. **Multi-layer constraints**: Should we constrain all layers simultaneously?
4. **Adaptive constraints**: Should constraint strength vary during training?

---

## Summary

| Aspect | Status |
|--------|--------|
| **Original TSV** | Park et al., 2025 |
| **lm_head constraints** | **Our contribution** (no direct literature) |
| **Theoretical basis** | Regularization theory, KL divergence |
| **Similar methods** | Logit space interventions, distribution matching |

**Bottom line**: The lm_head constraint is a practical innovation to solve a real problem. While not directly from literature, it's grounded in well-established regularization principles.

