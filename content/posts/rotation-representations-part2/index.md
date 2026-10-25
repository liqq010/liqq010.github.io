---
title: "Rotation Representations Part 2: Beyond the Matrix"
date: 2026-10-24
draft: true
math: true
tags: ["rotation", "quaternion", "3D", "representation learning"]
summary: "Axis-angle, rotation vectors, Rodrigues' formula, quaternions, Euler angles, swing–twist decomposition, and which rotation representations are continuous."
---

[Part 1]({{< relref "rotation-representations-part1" >}}) covered rotation matrices and the 6D and 9D representations used in neural networks. This part covers the other representations.

As we described in the rotation matrix part of Part 1, every 3D rotation matrix has a direction that does not move. So it is natural to represent a rotation by its rotation axis and rotation angle.

## Axis-Angle and Rotation Vectors

Any 3D rotation can be described as "rotate about an axis $n$ by an angle $\theta$".

**Axis-angle.** A rotation is represented by a pair $(n, \theta)$: $n$ is a unit vector (the rotation axis), and $\theta$ is the rotation angle. The direction of rotation follows the right-hand rule: with the thumb pointing along $n$, the fingers curl in the positive direction.

This uses four numbers with one constraint, $\|n\| = 1$. The representation is not unique: $(n, \theta)$ and $(-n, -\theta)$ are the same rotation, and when $\theta = 0$ the axis is undefined.

**Rotation vector.** Combine the axis and the angle into one vector:

$$r = \theta\, n$$

The direction of $r$ is the rotation axis, and its length is the rotation angle.

Why do this? Axis-angle uses four numbers, but a rotation has only three degrees of freedom: $n$ has three numbers, yet its length must always be 1, so the length carries no information. The rotation vector puts $\theta$ into that unused length. Now three numbers describe a rotation with no constraint to maintain: any 3D vector $r$ is a valid rotation, and nothing is lost, since $\theta = \|r\|$ and $n = r / \|r\|$.

<div class="callout">The length of $n$ carries no information, so store $\theta$ in it: three numbers, no constraint.</div>

Rotating by $\theta > \pi$ about $n$ is the same as rotating by $2\pi - \theta$ about $-n$, so $\theta \in [0, \pi]$ is enough. All rotation vectors therefore fill a solid ball of radius $\pi$. On its surface, opposite points $r$ and $-r$ are the same rotation: turning by $\pi$ about $n$ or about $-n$ ends in the same place.

<div class="callout">Rotation vectors cannot simply be added, because rotations do not commute.</div>

Axis-angle and rotation vectors only describe a rotation. To actually rotate a vector, or to get the rotation matrix, we need Rodrigues' rotation formula.

## Rodrigues' Rotation Formula

### The problem

Given the axis $n$ and the angle $\theta$, find the rotated vector $v'$ and the rotation matrix $R$ directly.

### The rotated vector

Split $v$ into a part along the axis and a part perpendicular to it:

$$v_\parallel = (n \cdot v)\,n, \qquad v_\perp = v - v_\parallel$$

{{< inline-svg "rodrigues-split.svg" >}}

$v_\parallel$ lies on the axis, so it does not move. $v_\perp$ lies in the plane perpendicular to $n$, and rotates in that plane just like in 2D:

$$v_\perp' = \cos\theta \cdot v_\perp + \sin\theta \cdot (v_\perp \text{ rotated by } 90^\circ \text{ about } n)$$

The only missing piece is the 3D version of "rotate by 90°". It is $n \times v$: the cross product is perpendicular to $n$ and to $v$, has length $\|v_\perp\|$ (since $\|n\| = 1$), and follows the right-hand rule. Since $n \times v_\parallel = 0$, $n \times v = n \times v_\perp$.

{{< inline-svg "rodrigues-plane.svg" >}}

Adding the two parts back:

$$v' = v_\parallel + \cos\theta \cdot v_\perp + \sin\theta \cdot (n \times v)$$

Substituting $v_\perp = v - (n \cdot v)\,n$:

$$v' = \cos\theta \cdot v + \sin\theta \cdot (n \times v) + (1 - \cos\theta)(n \cdot v)\,n$$

The third term is a compensation. $\cos\theta \cdot v$ scales all of $v$, including $v_\parallel$, which should not change. The term $(1 - \cos\theta)(n \cdot v)\,n$ adds back the missing $(1 - \cos\theta)$ share of $v_\parallel$.

<div class="callout">Rodrigues' formula = the 2D rotation formula $\cos\theta \cdot v + \sin\theta \cdot (v \text{ rotated by } 90^\circ)$, with the 3D "rotate by 90°" being $n \times v$.</div>

### The rotation matrix

Write out the cross product:

$$n \times v = \begin{bmatrix} n_2 v_3 - n_3 v_2 \\ n_3 v_1 - n_1 v_3 \\ n_1 v_2 - n_2 v_1 \end{bmatrix}$$

Each entry is a linear combination of $v_1, v_2, v_3$. Copying the coefficients into a matrix:

$$n \times v = K v, \qquad K = \begin{bmatrix} 0 & -n_3 & n_2 \\ n_3 & 0 & -n_1 \\ -n_2 & n_1 & 0 \end{bmatrix}$$

Applying $K$ twice rotates $v_\perp$ by 90° twice, which flips it:

$$K^2 v = n \times (n \times v) = -v_\perp$$

So $v_\perp = -K^2 v$ and $v_\parallel = v - v_\perp = (I + K^2)\,v$. Substituting into $v' = v_\parallel + \cos\theta \cdot v_\perp + \sin\theta \cdot K v$:

$$v' = (I + K^2)\,v - \cos\theta \cdot K^2 v + \sin\theta \cdot K v$$

$$R = I + \sin\theta \cdot K + (1 - \cos\theta)\,K^2$$

<div class="callout">On the plane perpendicular to $n$, $K^2 = -I$: rotating by 90° twice is a flip. This is the same fact as $J^2 = -I$ for the 2D 90° rotation matrix $J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$, and $i^2 = -1$ for complex numbers. It will show up again with quaternions.</div>

## Quaternions

### Warm-up: complex numbers in 2D

As we saw in [Part 1]({{< relref "rotation-representations-part1" >}}), multiplying by $\cos\theta + i\sin\theta$ is a rotation, because multiplying by $i$ is a 90° rotation, and expanding the product gives exactly $\cos\theta \cdot v + \sin\theta \cdot (v \text{ rotated by } 90^\circ)$.

As also shown in Part 1, composing rotations multiplies the numbers, and the angles add.

<div class="callout">Complex numbers turn "rotate by 90°" into a number. So a rotation is multiplying by a number, and composing rotations is multiplying the numbers.</div>

### Extending to 3D

We want a kind of number that rotates a 3D point when we multiply by it. Two imaginary units are not enough; it takes a real part plus $i$, $j$, $k$, four numbers in total:

$$q = w + x\,i + y\,j + z\,k, \qquad i^2 = j^2 = k^2 = ijk = -1$$

From these rules, the products of two units are

$$ij = k, \quad jk = i, \quad ki = j, \qquad ji = -k, \quad kj = -i, \quad ik = -j$$

Write a 3D point as a pure quaternion $v = x\,i + y\,j + z\,k$ (real part 0). For two pure quaternions, the product is

$$u\,v = -u \cdot v + u \times v$$

For example, $i\,i = -1 = -i \cdot i + i \times i$, and $i\,j = k = -i \cdot j + i \times j$. So for a unit axis $n$, and for $v_\perp$ in the plane perpendicular to $n$:

$$n^2 = -1, \qquad n\,v_\perp = n \times v_\perp$$

Multiplying by $n$ is a 90° rotation in that plane: $n$ plays the role of $i$. So try the same form as in 2D, and multiply only on the left by $q = \cos\theta + n\sin\theta$:

$$\begin{aligned} q\,v_\perp &= \cos\theta \cdot v_\perp + \sin\theta \cdot (n \times v_\perp) \\ q\,v_\parallel &= \cos\theta \cdot v_\parallel - \sin\theta\,(n \cdot v) \end{aligned}$$

The perpendicular part rotates correctly, but the part along the axis is turned into the real part.

To undo this, multiply on the right by the conjugate $q^* = \cos\theta - n\sin\theta$ as well, with $q\,q^* = 1$. $v_\parallel$ points along $n$, so it commutes with $q$. $v_\perp$ anticommutes with $n$ ($n\,v_\perp = -v_\perp\,n$), so $v_\perp\,q^* = q\,v_\perp$. Therefore

$$q\,v_\parallel\,q^* = v_\parallel\,q\,q^* = v_\parallel, \qquad q\,v_\perp\,q^* = q^2\,v_\perp$$

The part along the axis is turned back, but the perpendicular part is rotated twice, by $2\theta$ in total. So use half the angle:

$$v' = q\,v\,q^*, \qquad q = \cos\frac{\theta}{2} + n\sin\frac{\theta}{2}$$

<div class="callout">The half angle is not an arbitrary rule: the rotation is applied twice, once from each side.</div>

### Using quaternions in practice

- **Axis-angle → quaternion:** $q = \left(\cos\frac{\theta}{2},\ n\sin\frac{\theta}{2}\right)$.
- **Composition:** rotating by $q_1$ and then by $q_2$ is $q_2\,q_1$.
- **Rotating a vector:** $v' = q\,v\,q^*$, or convert $q$ to a rotation matrix first.
- **Normalization:** after many multiplications, rounding errors make $\|q\|$ drift away from 1, so normalize $q$ from time to time.

<div class="callout">A quaternion is another way to store axis-angle: it stores the cos and sin of the half angle. Then composition becomes multiplication with no trigonometric functions, and interpolation becomes moving along an arc on a sphere.</div>

### Sign ambiguity and slerp

$q$ and $-q$ are the same rotation: the two signs cancel in $q\,v\,q^*$. Requiring $w \ge 0$ removes the ambiguity, but introduces jumps: near $\theta = \pi$, $w \approx 0$, and a small change in the rotation can push $w$ below 0, so $q$ suddenly jumps to $-q$.

Before slerp, check the dot product $q_0 \cdot q_1$. If it is less than 0, negate one of them, so that the interpolation takes the shorter arc:

$$\text{slerp}(q_0, q_1, t) = \frac{\sin((1 - t)\,\Omega)}{\sin\Omega}\,q_0 + \frac{\sin(t\,\Omega)}{\sin\Omega}\,q_1, \qquad \cos\Omega = q_0 \cdot q_1$$

## Quaternions and Rodrigues

Expanding $q\,v\,q^*$ gives exactly Rodrigues' formula. Take one matrix entry as an example. With $q = (w, x, y, z) = \left(\cos\frac{\theta}{2},\ n\sin\frac{\theta}{2}\right)$, reading off the coefficient of $v_2$ in the first component of $q\,v\,q^*$ gives

$$R_{12} = 2(xy - wz) = 2\sin^2\frac{\theta}{2}\,n_1 n_2 - 2\sin\frac{\theta}{2}\cos\frac{\theta}{2}\,n_3$$

In Rodrigues' formula, $K_{12} = -n_3$ and $K^2 = n n^\top - I$, so $(K^2)_{12} = n_1 n_2$:

$$R_{12} = -\sin\theta\,n_3 + (1 - \cos\theta)\,n_1 n_2$$

The half angles multiply together, and the double-angle formulas bring them back to the full angle:

$$2\sin^2\frac{\theta}{2} = 1 - \cos\theta, \qquad 2\sin\frac{\theta}{2}\cos\frac{\theta}{2} = \sin\theta$$

Doing the same for every entry gives the rotation matrix of a unit quaternion $q = (w, x, y, z)$:

$$R = \begin{bmatrix} 1 - 2(y^2 + z^2) & 2(xy - wz) & 2(xz + wy) \\ 2(xy + wz) & 1 - 2(x^2 + z^2) & 2(yz - wx) \\ 2(xz - wy) & 2(yz + wx) & 1 - 2(x^2 + y^2) \end{bmatrix}$$

<div class="callout">Both say the same thing: the part along the axis stays, and the part in the perpendicular plane rotates as $\cos\theta \cdot v + \sin\theta \cdot (v \text{ rotated by } 90^\circ)$. The quaternion just stores the half angle, and in return composition is a multiplication.</div>

## Euler Angles

Euler angles rotate about coordinate axes three times in a row, for example yaw $\psi$ about $z$, pitch $\beta$ about $y$ and roll $\phi$ about $x$:

$$R = R_z(\psi)\,R_y(\beta)\,R_x(\phi)$$

This is intuitive: three numbers, each with a clear meaning.

**Gimbal lock.** When pitch $= 90^\circ$, $R_y(90^\circ)$ turns the $x$ axis onto the $z$ axis (pointing the other way), so yaw and roll now rotate about the same axis. Only $\psi - \phi$ matters, and one degree of freedom is lost: no small change in $\psi$ or $\phi$ can tilt about the missing axis.

<div id="gimbal-lock"></div>
<script src="gimbal-lock.js"></script>

## Swing–Twist Decomposition

Sometimes we care about how much a rotation turns about a given axis $a$, for example the twist of a forearm about the bone direction. Any rotation can be split into two parts:

$$q = q_{\text{swing}}\,q_{\text{twist}}$$

$q_{\text{twist}}$ is a rotation about $a$, and $q_{\text{swing}}$ is a rotation about an axis perpendicular to $a$. To compute it, keep only the part of $q$'s vector part along $a$, normalize to get the twist, and the rest is the swing:

$$q_{\text{twist}} = \frac{(w,\ (u \cdot a)\,a)}{\|(w,\ (u \cdot a)\,a)\|}, \qquad q_{\text{swing}} = q\,q_{\text{twist}}^*$$

where $q = (w, u)$. This is used, for example, to set separate joint limits for swing and twist. The twist is undefined when $w = 0$ and $u \perp a$, i.e. a 180° swing.

## Continuity

When a neural network predicts a rotation, it learns the map from rotation to representation. A network is a continuous function, so it struggles to fit a target that jumps: two nearly identical rotations should have nearby representations. Going the other way, from representation to rotation, is continuous for every representation; the question is only the direction from rotation to representation.

- **Euler angles:** discontinuous. Angles wrap around at $\pm 180^\circ$, and near gimbal lock a small change in the rotation can need a large change in the angles.
- **Axis-angle and rotation vectors:** discontinuous. On the surface of the radius-$\pi$ ball, $r$ and $-r$ are the same rotation, so a rotation passing through $\theta = \pi$ jumps from one side of the ball to the other.
- **Quaternions:** discontinuous. To get one output per rotation, we must pick between $q$ and $-q$, e.g. $w \ge 0$, and the choice jumps at $w = 0$.
- **Rotation matrices (9D) and 6D:** continuous. The matrix is the rotation itself, and the 6D representation is just its first two columns.

The figure below rotates about the $z$ axis from 0° to 360° and plots one component of each representation. The rotation changes smoothly, but the first three curves jump at 180°. Move over any chart, or drag the slider.

<div id="continuity-plot"></div>
<script src="continuity.js"></script>

This is not a matter of choosing the representation well: every representation with 4 or fewer numbers is discontinuous. This is why networks use the 6D and 9D representations from [Part 1]({{< relref "rotation-representations-part1" >}}).

## Summary

| Representation | Numbers | Definition | Drawbacks |
|---|---|---|---|
| Rotation matrix | 9 | $R^\top R = I$, $\det R = +1$ | 6 constraints; network outputs need a correction (SVD) |
| Euler angles | 3 | Three rotations about coordinate axes | Gimbal lock; axis order must be agreed on; discontinuous |
| Axis-angle | 4 | $(n, \theta)$, $\|n\| = 1$ | Not unique; axis undefined at $\theta = 0$; discontinuous |
| Rotation vector | 3 | $r = \theta\,n$ | Cannot be added; $r$ and $-r$ are the same at $\|r\| = \pi$; discontinuous |
| Quaternion | 4 | $\left(\cos\frac{\theta}{2},\ n\sin\frac{\theta}{2}\right)$, $\|q\| = 1$ | $q$ and $-q$ are the same rotation; needs normalization; discontinuous |
| 6D | 6 | First two columns of $R$ | Needs Gram–Schmidt |
| 9D | 9 | Full $3 \times 3$ matrix | Needs SVD |

## Citation

If you find this post useful, please cite it as:

> Li, Quanquan. "Rotation Representations Part 2: Beyond the Matrix." *Quanquan Li's blog*, October 24, 2026. https://liqq010.github.io/posts/rotation-representations-part2/

Or in BibTeX:

```bibtex
@misc{li2026rotationpart2,
  author       = {Li, Quanquan},
  title        = {Rotation Representations Part 2: Beyond the Matrix},
  year         = {2026},
  month        = {October},
  day          = {24},
  howpublished = {\url{https://liqq010.github.io/posts/rotation-representations-part2/}},
  note         = {Blog post}
}
```
