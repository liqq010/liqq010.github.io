---
title: "Fixing Softmax and Cross-Entropy: Two Numerical Pitfalls"
date: 2026-10-25
draft: false
math: true
tags: ["softmax", "cross-entropy", "numerical stability"]
summary: "Softmax Collapse and Softmax Cancellation: two floating-point pitfalls in Softmax and Cross-Entropy, and mathematically equivalent fixes."
ShowToc: false
---

**Paper:** [A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642)

Standard Softmax and Cross-Entropy can suffer from numerical precision issues that affect training dynamics. This paper identifies two problems and proposes mathematically equivalent but numerically more stable formulations.

## 1. Softmax Collapse

**Problem.** For the target class $t$, the Cross-Entropy gradient is

$$
\frac{\partial L}{\partial z_t} = q_t - 1, \qquad q = \operatorname{softmax}(z)
$$

When the target logit dominates, $q_t$ may round to exactly 1 in floating-point arithmetic, producing a zero gradient even though the true gradient is nonzero. Meanwhile, non-target gradients remain positive, violating the zero-sum gradient property.

**Fix.** Rewrite Cross-Entropy using Softplus:

$$
\begin{aligned}
L &= -\log\frac{e^{z_t}}{\sum_j e^{z_j}} \\
  &= \log\Big(1 + \sum_{j\ne t} e^{z_j - z_t}\Big) \\
  &= \operatorname{softplus}(a)
\end{aligned}
$$

where

$$
a = \operatorname{logsumexp}_{j\ne t}(z_j - z_t)
$$

The target gradient becomes

$$
\boxed{\frac{\partial L}{\partial z_t} = -\sigma(a) = q_t - 1}
$$

This avoids subtracting 1 from a probability extremely close to 1.

Minimal example (FP32):

```python
import torch

z = torch.tensor([18.0, 0.0], dtype=torch.float32)
q = torch.softmax(z, dim=-1)

g_std = q[0] - 1
a = z[1] - z[0]
g_fixed = -torch.sigmoid(a)

print(g_std)    # 0.0
print(g_fixed)  # -1.52e-8
```

The two gradients are mathematically identical, but only the stable formulation preserves the small nonzero gradient.

## 2. Softmax Cancellation

**Problem.** In knowledge distillation, let

$$
p = \operatorname{softmax}(z_T), \qquad q = \operatorname{softmax}(z_S)
$$

The gradient with respect to student logits is

$$
\boxed{g_i = q_i - p_i}
$$

When student and teacher predictions are nearly identical, subtracting two close probabilities can cause catastrophic cancellation. Even identical logits may produce small nonzero gradients, leading to unnecessary student updates.

**Fix.** Compute the gradient from logit differences instead of subtracting probabilities.

$$
g_i = q_i - p_i = p_i\left(\frac{q_i}{p_i} - 1\right)
$$

Let $\delta_i = z_{S,i} - z_{T,i}$. From the Softmax definition,

$$
\frac{q_i}{p_i} = \frac{e^{\delta_i}}{\sum_j p_j e^{\delta_j}}
$$

Then center it:

$$
d_i = \delta_i - \sum_j p_j \delta_j
$$

Since Softmax is invariant to a constant shift,

$$
q_i = \frac{p_i e^{d_i}}{\sum_j p_j e^{d_j}}
$$

Let

$$
s_i = d_i - \operatorname{log1p}\Big(\sum_j p_j \operatorname{expm1}(d_j)\Big)
$$

Then

$$
q_i = p_i e^{s_i}
$$

Therefore,

$$
\boxed{g_i = q_i - p_i = p_i \operatorname{expm1}(s_i)}
$$

This is mathematically equivalent to the original gradient, but `expm1` avoids subtracting nearly identical probabilities.

To reduce floating-point violations of the zero-sum gradient property, apply:

$$
\boxed{g_i = u_i - q_i \sum_j u_j, \qquad u_i = p_i \operatorname{expm1}(s_i)}
$$

In exact arithmetic, $\sum_j u_j = 0$, so the correction does not change the theoretical gradient.

Minimal example (FP32):

```python
import torch
import torch.nn.functional as F

t = torch.tensor([2., 1., 0.], dtype=torch.float32)
s = t.clone().requires_grad_()

p = F.softmax(t, dim=-1)
loss = -(p * F.log_softmax(s, dim=-1)).sum()
g_std = torch.autograd.grad(loss, s)[0]

# Stable gradient
d = s.detach() - t
d = d - (p * d).sum()
r = d - torch.log1p((p * torch.expm1(d)).sum())  # s_i in the text
u = p * torch.expm1(r)
q = F.softmax(s.detach(), dim=-1)
g_fixed = u - q * u.sum()

print(g_std)    # May contain nonzero residuals
print(g_fixed)  # tensor([0., 0., 0.])
```

When teacher and student logits are identical, the stable formulation produces zero gradient without subtracting nearly identical Softmax probabilities.

## Takeaway

Both fixes preserve the original mathematical objective and gradients:

- **Softmax Collapse:** Replace $q_t - 1$ with $-\sigma(a)$ to avoid precision loss near 1.
- **Softmax Cancellation:** Replace $q - p$ with $p \operatorname{expm1}(s)$ to avoid catastrophic cancellation.

Mathematical equivalence does not guarantee numerical equivalence in floating-point arithmetic.
