---
title: "Rotation Representations Part 1: Rotation Matrices, RoPE, and 6D/9D"
date: 2026-10-24
draft: false
math: true
tags: ["rotation", "RoPE", "3D", "representation learning"]
summary: "Rotation matrices, complex numbers and RoPE, and the 6D and 9D rotation representations used in neural networks."
cover:
  image: "cover.png"
  alt: "Rotations in human body models, camera poses and RoPE"
  hidden: true
  relative: true
---

## Introduction

{{< inline-svg "rotations-everywhere.svg" >}}

Rotations appear everywhere: the joint rotations of parametric human body models, camera poses in SLAM and 3D reconstruction, and positional encoding in LLMs.

There are many ways to represent a 3D rotation: rotation matrices, axis-angle, rotation vectors, Euler angles, quaternions, and the 6D and 9D representations commonly used in neural networks.

The post is split into two parts. Part 1 focuses on what is most relevant to neural networks: rotation matrices, RoPE, and the 6D and 9D representations. Part 2 goes further into the other representations: axis-angle, rotation vectors, quaternions and Euler angles, and how they connect to each other.

## Rotation Matrices

Let $T$ be a rotation. It is a linear transformation.

Any 3D vector can be written as a combination of the three coordinate axes:

$$v = (x, y, z) = x\,e_1 + y\,e_2 + z\,e_3$$

$$e_1 = (1,0,0), \quad e_2 = (0,1,0), \quad e_3 = (0,0,1)$$

By linearity,

$$T(v) = x\,T(e_1) + y\,T(e_2) + z\,T(e_3)$$

So once we know where the three axes go, we know where every vector goes. Put the rotated axes into the columns of a matrix:

$$R = \begin{bmatrix} T(e_1) & T(e_2) & T(e_3) \end{bmatrix}$$

The rule for multiplying a matrix by a vector is exactly "use the vector's coordinates as weights and add up the columns":

$$R\,v = x\,(\text{column } 1) + y\,(\text{column } 2) + z\,(\text{column } 3) = T(v)$$

So $R\,v$ is the rotated vector.

<div class="callout">The $j$-th column of a rotation matrix is where the $j$-th axis lands after the rotation. $R\,v$ adds up the columns, weighted by the coordinates of $v$.</div>

### 2D rotation matrix

{{< inline-svg "rotation-2d.svg" >}}

Rotating counterclockwise by $\theta$ sends $e_1$ to $(\cos\theta, \sin\theta)$ and $e_2$ to $(-\sin\theta, \cos\theta)$. Putting them in the columns:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

Multiplying out $R(\theta)\,v$ for $v = (x, y)$:

$$R(\theta)\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta \end{bmatrix} = \cos\theta \begin{bmatrix} x \\ y \end{bmatrix} + \sin\theta \begin{bmatrix} -y \\ x \end{bmatrix}$$

$$v \text{ rotated by } \theta = \cos\theta \cdot v + \sin\theta \cdot (v \text{ rotated by } 90^\circ)$$

The result is a weighted sum of two vectors: a $\cos\theta$ share of the original $v$, plus a $\sin\theta$ share of $v$ rotated by 90°, where $(x, y)$ rotated by 90° is $(-y, x)$.

### Rotating the 2D plane with complex numbers

Define $i^2 = -1$, and draw the complex number $a + b\,i$ as the point $(a, b)$ in the plane:

{{< inline-svg "complex-plane.svg" >}}

Multiplying by $i$ = rotating counterclockwise by 90°. Multiply any point $a + b\,i$ by $i$:

$$(a + b\,i) \times i = a\,i + b\,(i \cdot i) = a\,i - b = -b + a\,i$$

So the coordinates change as

$$(a, b) \xrightarrow{\ \times\, i\ } (-b, a)$$

Rotating by any angle $\theta$. To rotate by $\theta$, multiply by the point on the unit circle at angle $\theta$:

$$\text{the point at angle } \theta \text{ on the unit circle} = (\cos\theta, \sin\theta) = \cos\theta + i\sin\theta$$

{{< inline-svg "unit-circle.svg" >}}

To see why, multiply any point $a + b\,i$ by it:

$$\begin{aligned} (a + b\,i)(\cos\theta + i\sin\theta) &= a\cos\theta + a\sin\theta\,i + b\cos\theta\,i + b\sin\theta\,(i \cdot i) \\ &= (a\cos\theta - b\sin\theta) + (a\sin\theta + b\cos\theta)\,i \end{aligned}$$

The coordinates change as

$$x' = a\cos\theta - b\sin\theta, \qquad y' = a\sin\theta + b\cos\theta$$

This is exactly the 2D rotation matrix:

$$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix}$$

**Conclusion: multiplying by $\cos\theta + i\sin\theta$ rotates a point counterclockwise by $\theta$.**

<div id="complex-rot"></div>
<script src="complex-rotation.js"></script>

<div class="callout">Rotation = a $\cos$ share of "stay" + a $\sin$ share of "rotate by 90°". In 2D, "rotate by 90°" is multiplying by $i$.</div>

Composing rotations = multiplying. Rotate a point $z$ by $\theta_1$, then by $\theta_2$:

$$\begin{aligned} &z \times (\cos\theta_1 + i\sin\theta_1) \times (\cos\theta_2 + i\sin\theta_2) \\ =\ &z \times \big(\cos(\theta_1 + \theta_2) + i\sin(\theta_1 + \theta_2)\big) \end{aligned}$$

The angles simply add. Expanding the product in the middle gives exactly the angle-addition formulas for $\cos$ and $\sin$.

Euler's formula.

$$e^{i\theta} = \cos\theta + i\sin\theta$$

It simply means "the point at angle $\theta$ on the unit circle".

"Adding the angles" is adding the exponents:

$$e^{i\theta_1} \cdot e^{i\theta_2} = e^{i(\theta_1 + \theta_2)}$$

### RoPE

RoPE (rotary position embedding) rotates the query and the key by angles proportional to their positions. The core idea: when two rotated vectors are dotted together, the result naturally contains their relative position.

$$(R(\alpha)\,q)^\top (R(\beta)\,k) = q^\top R(\alpha)^\top R(\beta)\,k$$

Since $R(\alpha)^\top = R(-\alpha)$ (transposing flips the sign of $\sin$), and $R(a)\,R(b) = R(a + b)$ (multiplying rotation matrices adds the angles),

$$R(\alpha)^\top R(\beta) = R(-\alpha)\,R(\beta) = R(\beta - \alpha)$$

$$(R(\alpha)\,q)^\top (R(\beta)\,k) = q^\top R(\beta - \alpha)\,k$$

The dot product of the two rotated vectors depends only on the angle difference $\beta - \alpha$, not on $\alpha$ and $\beta$ separately. With $\alpha = m\theta$ and $\beta = n\theta$ for a query at position $m$ and a key at position $n$, where $\theta$ is the angle per position, it depends only on $n - m$.

**Matrix form.** A pair $(q_0, q_1)$ is rotated by a 2D rotation matrix:

$$\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} q_0 \\ q_1 \end{bmatrix} = \begin{bmatrix} q_0 \\ q_1 \end{bmatrix} \cos\theta + \begin{bmatrix} -q_1 \\ q_0 \end{bmatrix} \sin\theta$$

For $q$ and $k$ of dimension $d_{model}$, split them into pairs and rotate each pair. Pair $j$ at position $m$ is rotated by $m\,\theta_j$, with $\theta_j = 1 / 10000^{2j/d_{model}}$: each pair has its own angle per position.

```python
class RoPE(nn.Module):
    def __init__(self, d_model, max_seq_len, theta):
        super().__init__()

        # 1 / theta ** (2*i / d_model)
        inv_freq = 1. / theta ** (torch.arange(0, d_model, 2) / d_model) # d_model / 2
        inv_freq = torch.repeat_interleave(inv_freq, 2, dim=-1) # d_model
        pos = torch.arange(max_seq_len) # L
        angles = torch.einsum('l,d->ld', pos, inv_freq) # l d_model
        self.register_buffer('sin', angles.sin(), persistent=False)
        self.register_buffer('cos', angles.cos(), persistent=False)

    def forward(self, x, position_ids):
        # B H T D
        sin = self.sin[position_ids][None, None]
        cos = self.cos[position_ids][None, None]
        rotate_half = torch.stack([-x[...,1::2], x[...,::2]], dim=-1).flatten(-2)
        out = cos * x + sin * rotate_half
        return out
```

**Complex form.** Treat each pair $(q_0, q_1)$ as the complex number $q_0 + i\,q_1$. Rotating it by $\theta$ is multiplying by $\cos\theta + i\sin\theta = e^{i\theta}$:

$$(q_0 + i\,q_1)(\cos\theta + i\sin\theta) = (q_0\cos\theta - q_1\sin\theta) + i\,(q_0\sin\theta + q_1\cos\theta)$$

the same result as the matrix form. So rotating all pairs is one element-wise complex multiplication.

```python
class RoPE_complex(nn.Module):
    def __init__(self, d_model, max_seq_len, theta):
        super().__init__()

        # 1 / theta ** (2*i / d_model)
        inv_freq = 1. / theta ** (torch.arange(0, d_model, 2) / d_model) # d_model / 2
        t = torch.arange(max_seq_len) # L
        freqs = torch.outer(t, inv_freq)
        # polar(r, θ)  =  r·cos(θ) + i·r·sin(θ)  =  r·e^{iθ}
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # e^{i ang}
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)

    def forward(self, x, position_ids):
        B, H, T, d = x.shape
        fc = self.freqs_cis[position_ids]
        # [a, b] → a + bi
        xc = torch.view_as_complex(x.float().reshape(B, H, T, d//2, 2))
        out = torch.view_as_real(xc * fc[None, None, :, :]).flatten(-2)
        return out.type_as(x)
```

### 3D rotation matrices

Rotations about the $z$, $x$ and $y$ axes:

$$R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

$$R_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}$$

$$R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}$$

When rotating about a coordinate axis, that axis stays where it is. Since each column is where an axis lands, the corresponding column is the axis itself: $(0, 0, 1)$ in $R_z$, $(1, 0, 0)$ in $R_x$, $(0, 1, 0)$ in $R_y$. The other two axes rotate in their plane, just like in 2D.

#### Arbitrary 3D rotations

A general 3D rotation is not about a coordinate axis. Take any three orthonormal vectors $u = (x_u, y_u, z_u)$, $v = (x_v, y_v, z_v)$, $w = (x_w, y_w, z_w)$ that form a right-handed set ($w = u \times v$):

$$u \cdot u = v \cdot v = w \cdot w = 1, \qquad u \cdot v = v \cdot w = w \cdot u = 0$$

Putting them in the rows gives a rotation:

$$R_{uvw} = \begin{bmatrix} x_u & y_u & z_u \\ x_v & y_v & z_v \\ x_w & y_w & z_w \end{bmatrix}$$

$R_{uvw}\,u = (u \cdot u,\ v \cdot u,\ w \cdot u) = (1, 0, 0)$, so $R_{uvw}$ rotates $u$ to the $x$ axis, $v$ to $y$, and $w$ to $z$. Its transpose has $u, v, w$ as columns and rotates the other way: it sends $x, y, z$ to $u, v, w$ respectively.

A rotation matrix must be orthogonal with determinant $+1$:

$$R^\top R = I, \qquad \det R = +1$$

Orthogonal: a rotation keeps lengths and angles, so the columns stay unit length and perpendicular to each other. Entry $(i, j)$ of $R^\top R$ is column $i$ dotted with column $j$, so $R^\top R = I$.

The determinant is the factor by which a transformation scales volume, and its sign tells whether the shape is flipped into its mirror image. A rotation keeps volume and does not flip, so $\det R = +1$.

### Predicting rotations: 6D and 9D

A rotation matrix changes continuously with the rotation, so it is a natural target for a network to predict. The problem is that the numbers a network outputs do not naturally satisfy $R^\top R = I$ and $\det R = +1$. We need a correction step that turns any output into a valid rotation matrix. The 6D and 9D representations are two ways to do this.

**6D.** Output two columns $a_1, a_2$, orthonormalize them with Gram–Schmidt, and get the third column from the cross product:

$$b_1 = \frac{a_1}{\|a_1\|}, \qquad b_2 = \frac{a_2 - (b_1 \cdot a_2)\,b_1}{\|a_2 - (b_1 \cdot a_2)\,b_1\|}, \qquad b_3 = b_1 \times b_2$$

$$R = \begin{bmatrix} b_1 & b_2 & b_3 \end{bmatrix}$$

{{< inline-svg "gram-schmidt.svg" >}}

**9D.** Output the whole $3 \times 3$ matrix $M$, take its SVD $M = U \Sigma V^\top$, and use the nearest rotation:

$$R = U V^\top$$

If $\det(U V^\top) = -1$, flip the sign of the last column of $U$ first, so that $\det R = +1$.

### The rotation axis

Every 3D rotation matrix has a direction that does not move (Euler's rotation theorem): vectors on the rotation axis $n$ are unchanged by the rotation, $R\,n = n$. In linear-algebra terms, $n$ is an eigenvector of $R$ with eigenvalue 1. This leads to the next representation: axis-angle and rotation vectors, covered in Part 2. Rodrigues' rotation formula, quaternions and Euler angles will also be covered there.

## Citation

If you find this post useful, please cite it as:

> Li, Quanquan. "Rotation Representations Part 1: Rotation Matrices, RoPE, and 6D/9D." *Quanquan Li's blog*, October 24, 2026. https://liqq010.github.io/posts/rotation-representations-part1/

Or in BibTeX:

```bibtex
@misc{li2026rotationpart1,
  author       = {Li, Quanquan},
  title        = {Rotation Representations Part 1: Rotation Matrices, RoPE, and 6D/9D},
  year         = {2026},
  month        = {October},
  day          = {24},
  howpublished = {\url{https://liqq010.github.io/posts/rotation-representations-part1/}},
  note         = {Blog post}
}
```
