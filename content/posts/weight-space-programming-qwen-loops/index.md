---
title: "Directional Ablation, and How Astra Fixed Qwen3.5-0.8B's Reasoning Loops"
date: 2026-11-04
draft: false
math: true
tags: ["weight-space programming", "directional ablation", "interpretability", "LLM"]
summary: "How an LLM edited 0.016% of Qwen3.5-0.8B's weights to break its reasoning loops: directional ablation from the refusal-direction work, then pushed further with a strength α and 64 selected neurons per layer."
ShowToc: true
---

**Paper:** [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2026)

<figure>
{{< inline-svg "loop-signal-by-layer.svg" >}}
<figcaption>The looping signal grows layer by layer; Astra edited the two largest MLP writers, layers 15 and 19.</figcaption>
</figure>

The paper uses Astra to edit the weights of Qwen3.5-0.8B directly, with no training or gradient descent, to make it better at math, which raises MATH-500 from 7.9% to 48.1%. Actually, it is not magic or uninterpretable. Astra uses a method from directional ablation, introduced by Arditi et al. (2024) to remove refusal from chat models. This note is mainly about directional ablation, with Astra's edit as a worked example.

Part 1 explains directional ablation as introduced by Arditi et al. (2024). Part 2 follows how Astra applied the same idea to Qwen3.5-0.8B. Part 3 covers how Astra went beyond plain ablation to reach the final edit.

---

## 0. Setup: how an MLP writes into the residual stream

Qwen3.5-0.8B has hidden size 1024 and FFN intermediate size 3584. For the MLP in layer $\ell$:

- $a \in \mathbb{R}^{3584}$: the MLP's intermediate activation at one token position (the elementwise product of gate and up in SwiGLU). $a_j$ is the activation of neuron $j$.
- $W \in \mathbb{R}^{1024 \times 3584}$: the `down_proj` weight. Its $j$-th column $w_j \in \mathbb{R}^{1024}$ is the vector neuron $j$ writes into the residual stream.
- $m = W a = \sum_j a_j w_j \in \mathbb{R}^{1024}$: the MLP output added to the residual stream.

Think of the MLP as 3584 writers: writer $j$ adds its fixed vector $w_j$ to the residual stream, scaled by its activation $a_j$.

---

## Part 1. Directional ablation (Arditi et al., 2024)

### 1.1 Idea

Find a unit vector $u$ in residual space that represents some behavior, then modify the weights so that they can no longer write anything along $u$, keeping only the component of each weight column orthogonal to $u$.

### 1.2 The operation

Let $u \in \mathbb{R}^{1024}$ with $\|u\| = 1$. The component of the MLP output along $u$ is

$$u^\top m = u^\top W a = \sum_j \underbrace{(u^\top w_j)}_{c_j}\, a_j$$

where $c_j = u^\top w_j$ is a scalar: how much neuron $j$ writes along $u$ per unit of activation.

Each column decomposes into a part along $u$ and a part orthogonal to it:

$$w_j = \underbrace{(u^\top w_j)\,u}_{\text{along } u} + \underbrace{\big(w_j - (u^\top w_j)\,u\big)}_{\text{orthogonal to } u}$$

Directional ablation keeps only the orthogonal part of every column:

$$W' = (I - u u^\top)\,W = W - u\,(u^\top W)$$

Here $uu^\top$ is a $1024 \times 1024$ projection matrix and $u^\top W$ is the length-3584 row vector of all $c_j$. The projection acts on the **output side** (the 1024-dim residual space), not on the 3584-dim activation side. After the edit, $u^\top W' = u^\top W - (u^\top u)(u^\top W) = 0$, so for any activation $a$ the layer writes nothing along $u$, while the orthogonal part $(I - uu^\top)W$ is unchanged. This is a rank-1 edit. There is no strength parameter: the component along $u$ is removed exactly once.

### 1.3 Finding $u$: difference of means

Arditi et al. take $u$ to be the difference between the mean activations of harmful and harmless prompts, normalized.

**Why a difference.** The underlying assumption is the linear representation hypothesis: a concept corresponds to a fixed direction, and the stronger the concept, the larger the hidden state's component along it:

$$h_i = b + z_i\, v + \epsilon_i$$

- $h_i \in \mathbb{R}^d$: hidden state of prompt $i$ at some layer and position.
- $b$: the part shared by all prompts (chat template, overall activation mean).
- $v$: the concept direction we want.
- $z_i$: the concept's strength on prompt $i$ (large for harmful, small for harmless).
- $\epsilon_i$: content-specific variation (topic, wording) that varies unsystematically across prompts.

Taking group means and subtracting:

$$\mu_{\text{harm}} - \mu_{\text{safe}} = (\bar z_{\text{harm}} - \bar z_{\text{safe}})\, v + (\bar\epsilon_{\text{harm}} - \bar\epsilon_{\text{safe}})$$

Subtracting removes the shared part $b$, which is usually much larger than the $v$ component. Averaging makes the $\epsilon$ terms cancel; the more samples, the closer $\bar\epsilon_{\text{harm}} - \bar\epsilon_{\text{safe}}$ is to zero. What remains is approximately a multiple of $v$, and normalizing gives $u$.

### 1.4 Limitation: confounds

A difference of means captures **every** systematic difference between the two groups, not just the concept of interest. If harmful prompts are, say, systematically longer, a length direction $g$ also differs between groups:

$$h_i = b + z_i\, v + \ell_i\, g + \epsilon_i \quad\Rightarrow\quad \mu_{\text{harm}} - \mu_{\text{safe}} = (\bar z_{\text{harm}} - \bar z_{\text{safe}})\,v + (\bar \ell_{\text{harm}} - \bar \ell_{\text{safe}})\,g$$

The second term does not cancel, so $u$ is a mix of $v$ and $g$. Two consequences follow: ablating $u$ also damages the unrelated direction $g$ (collateral damage), and the part of $v$ orthogonal to $u$ is left in place (incomplete removal).

### 1.5 What "never writing along $u$" means

$u$ is an axis; the difference vector is only the tool used to pick it. Define a hidden state's coordinate on this axis as $p(h) = u^\top h$. By construction, all of the gap between the two group means lies on this axis: $u^\top(\mu_{\text{harm}} - \mu_{\text{safe}}) = \|\mu_{\text{harm}} - \mu_{\text{safe}}\|$, and the gap is zero along every direction orthogonal to $u$.

Arditi et al. apply the projection to **every** matrix that writes into the residual stream: the embedding, every attention output projection, and every MLP `down_proj`. The residual at any position is the sum of these writers:

$$h = e + \sum_\ell \big(m_\ell + o_\ell\big)$$

Since each writer satisfies $u^\top W' a = 0$ for any input, the sum also gives $u^\top h = 0$ for every prompt and every token. So it is not just that the two groups' **difference** becomes zero; each group's **coordinate** becomes zero. "Equal" would be a weaker condition: if the groups sat at 3 and 1 on the axis, moving both to 1 also makes the difference zero, but leaves content along $u$. Ablation instead empties the axis entirely. (Moving one group to the other's position would be activation steering, which is a different operation.)

Three things follow:

- The gap between the group means along $u$ is gone. At this linear level, the average "harmful" and "harmless" representations can no longer be told apart.
- Any downstream component that reads the concept is computing something like $r^\top h$. Whatever part of that readout depends on the $u$ component now reads 0, so the signal does not propagate.
- Everything orthogonal to $u$ (topic, syntax, and so on) is untouched, so the rest of the model's abilities are largely preserved.

### 1.6 Result on refusal

The model stops refusing harmful requests. Downstream components decide whether to refuse by reading the residual stream's component along $u$; with $u^\top h \equiv 0$, a harmful prompt looks like a harmless one on this axis and the model answers normally. This held across more than a dozen open chat models up to the 70B scale, with only small changes on standard capability benchmarks, because only one direction was removed. For this reason the paper calls $u$ the *refusal direction*: it behaves like a "the model intends to refuse" signal rather than the model's understanding that content is harmful.

---

## Part 2. Astra on Qwen3.5-0.8B: the same recipe

Astra cannot run the model directly. Each forward pass is a request to the harness that specifies what to return. It had 50 forward passes in this run, and every training-set problem it used came from the GSM8K and MATH **training** sets; it never saw MATH-500.

### 2.1 Diagnosis

The first forward pass generated answers on 96 practice problems (32 GSM8K, 64 MATH). Only 11 were correct and 85 hit the 32,768-token limit, with the truncated tails about 95% repeated 8-grams. Astra concluded that the main problem was looping, not inability to do the math.

### 2.2 Building four contrast pairs

Astra picked four problems from those 96 and, for each, wrote two continuations that share the same prompt and the same correct opening reasoning, then diverge. Both texts in every pair were written by Astra, not generated by Qwen.

The productive version is correct reasoning that keeps moving forward. It stops mid-thought inside `<think>` and does not give a boxed final answer. The repetitive version is a short, clean imitation of the loops Astra saw in Qwen's real outputs. For example:

> **Prompt:** Rafael works 10 hours on Monday and 8 on Tuesday, with 20 hours left to work. He earns <span>&#36;</span>20 per hour. Find his weekly pay.
>
> **Productive:** The first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours. His pay is 38\*20=760 dollars. Checking: 200+160+400=760, consistent with the total. The hours and hourly rate have the correct units.
>
> **Repetitive:** The first two days total 10+8=18 hours. Adding the remaining 20 gives 38 hours. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time. But is 20 the total? No, it is the remaining time.

Qwen's real loop on this problem kept asking whether the 20 hours were remaining or total; on the arithmetic-sequence problem it repeated "Wait, 88−85=3. Wait, 85−82=3." Astra's version of the latter was "Now 88-85=3 and 85-82=3." repeated.

The contrast is therefore *progressing vs. stuck*, not *correct vs. wrong*: the shared prefix is correct in both, and the repetitive version keeps re-asking an already-settled question.

| Problem | Source | Original model in forward pass 1 |
|---|---|---|
| Rafael's weekly pay | `gsm8k_train_1024` | Looped, truncated |
| Remainder of $6x^4-14x^3-\dots$ divided by $2x-6$ | `math_train_intermediate_algebra_738` | Looped, truncated |
| Terms of 88, 85, 82, … before −17 | `math_train_algebra_808` | Looped, truncated |
| Two necklace machines | `gsm8k_train_1766` | Correct (~7,400 tokens), not truncated |

Only three of the four problems actually looped; the logs do not say why the necklace problem was included. The practice files also contain reference solutions, which Astra had access to when writing the productive versions.

A caveat from Section 1.4 applies here: the two versions use different words (the repetitive Rafael text says "But is 20 the total?" three times), so $u$ may partly encode those specific phrases rather than only the looping state. Astra was aware of this and tried a content-matched variant: the difference between the hidden states at the first and the last occurrence of the same repeated sentence. Both sides then have exactly the same tokens, so the only remaining difference is the state "this is the $N$-th repetition". The final edit still used the productive-vs-repetitive direction described below.

There is a second limitation: with only 4 pairs, the averaging step can cancel very little of $\epsilon$. A direction estimated from so few samples is a hypothesis, which is why so much of the run (Sections 3.2 and 3.3) went into causal checks.

### 2.3 Teacher-forced forward pass: what is recorded

**Teacher forcing** means feeding a fixed, complete text through the model in one forward pass with no sampling. At each position the model predicts the next token given the true preceding text, and the harness returns per-token log-probabilities plus any requested internal states. This allows controlled comparisons between matched texts at low cost.

The 8 texts (4 pairs) were run through the original model in the 4th forward pass, returning two kinds of per-token data:

| Data | Where | Shape per text | Used for |
|---|---|---|---|
| Residual hidden state $h$ | 25 positions $L = 0, \dots, 24$ | $T_i \times 1024$ per position | Computing $u$ and choosing layers |
| MLP intermediate activation $a$ | Inside each of the 24 MLPs | $T_i \times 3584$ per layer | Layer attribution and choosing the 64 neurons |

Position $L = 0$ is the embedding output; position $L \ge 1$ is the residual **after** layer $L-1$ (both its attention/DeltaNet and MLP have been added). So layer 15's output is at position 16, and layer 19's at position 20.

Text lengths ranged from 98 to 146 tokens. The texts are padded to the longest and stacked, giving $H^{(L)} \in \mathbb{R}^{8 \times T_{\max} \times 1024}$ per position, with productive texts at even indices and repetitive texts at odd indices. Only the first $T_i$ positions of text $i$ are used.

### 2.4 From tokens to $u$

For each residual position $L$:

1. **Average the tail of each text.** Take the last 36 tokens of text $i$ and average over them:
   $$\bar h^{(L)}_i = \frac{1}{36}\sum_{t = T_i - 35}^{T_i} h^{(L)}_{i,t} \in \mathbb{R}^{1024}$$
   Only the tail is used because the prefix is identical within a pair; the difference lives in the continuation, and the end of the repetitive text is in the "already looping" state. This gives 8 vectors per position. No normalization yet.
2. **Subtract within each pair and average over pairs.**
   $$d^{(L)} = \frac{1}{4}\sum_{k=1}^{4}\left(\bar h^{(L)}_{\text{rep},k} - \bar h^{(L)}_{\text{prod},k}\right) \in \mathbb{R}^{1024}$$
3. **Normalize.**
   $$u^{(L)} = \frac{d^{(L)}}{\|d^{(L)}\|}$$

So each position gets its own unit vector $u^{(L)}$. Astra computed $d^{(L)}$ at every position to see where the contrast grows, but only two directions were used for edits: $u^{(16)}$ for layer 15 and $u^{(20)}$ for layer 19.

The shapes through the pipeline:

| Step | Operation | Shape |
|---|---|---|
| Input | 8 texts (4 pairs: productive + repetitive), length $T_i \in [98, 146]$ | token sequences |
| Forward | Teacher-forced pass on the original Qwen, residual at position $L$ | $[T_i, 1024]$ per text; padded stack $[8, T_{\max}, 1024]$ |
| Tail | Keep the last 36 tokens of each text | $[36, 1024]$ per text |
| Mean | Average over tokens | $[1024]$ per text, $[8, 1024]$ in total |
| Diff | Within-pair difference: repetitive − productive | $[4, 1024]$ |
| Average | Average over the 4 pairs | $d$: $[1024]$ |
| Normalize | $u = d / \lVert d\rVert$ | $u$: $[1024]$, $\lVert u\rVert = 1$ |

$T_i$ is the actual length of text $i$; $T_{\max} = 146$ is the longest. Padded positions are never used. The same pipeline runs once at $L = 16$ (for layer 15) and once at $L = 20$ (for layer 19).

### 2.5 Choosing layers 15 and 19: per-layer attribution

Layer $\ell$ adds to the residual stream:

$$h^{(\ell+1)} = h^{(\ell)} + \text{attn}_\ell + m_\ell$$

where $\text{attn}_\ell$ is the attention (or DeltaNet) output and $m_\ell = W_\ell a_\ell$ the MLP output. Taking "repetitive − productive" on both sides gives an exact split of each layer's increment to the contrast:

$$d^{(\ell+1)} - d^{(\ell)} = \Delta\text{attn}_\ell + \Delta m_\ell$$

With $u_\ell = d^{(\ell+1)}/\|d^{(\ell+1)}\|$, three scalars are computed per layer:

- **Total increment:** $(d^{(\ell+1)} - d^{(\ell)})^\top u_\ell$
- **MLP part:** $(W_\ell\, \Delta \bar a_\ell)^\top u_\ell$, where $\Delta \bar a_\ell \in \mathbb{R}^{3584}$ is the "repetitive − productive" difference of MLP activations, computed the same way as for hidden states (last-36-token mean, averaged over 4 pairs). Because $m = Wa$ is linear, this is exact.
- **Attention part:** total increment minus MLP part.

Values from the logs (second half of the network):

| Layer $\ell$ | $\lVert d^{(\ell+1)}\rVert$ | Total increment | MLP part | Attention part |
|---|---|---|---|---|
| 13 | 1.088 | 0.243 | 0.093 | 0.150 |
| 14 | 1.273 | 0.362 | 0.150 | 0.212 |
| **15** | 1.564 | 0.472 | **0.263** | 0.209 |
| 16 | 1.760 | 0.280 | 0.123 | 0.157 |
| 17 | 1.877 | 0.193 | 0.049 | 0.144 |
| 18 | 1.909 | 0.174 | 0.078 | 0.097 |
| **19** | 2.266 | 0.530 | **0.250** | 0.281 |
| 20 | 2.327 | 0.132 | 0.105 | 0.027 |
| 21 | 2.515 | 0.258 | 0.164 | 0.094 |
| 22 | 2.626 | 0.309 | 0.110 | 0.199 |

The contrast norm grows from about 0.16 at layer 0 to about 2.6, so the looping state is amplified layer by layer. The selection criterion is the **MLP part**, because the edit targets MLP `down_proj`; attention contributions cannot be changed by editing the MLP. Layers 15 (0.263) and 19 (0.250) have the two largest MLP parts across all 24 layers (third is layer 21 at 0.164), and also the two largest total increments. The single highest-scoring neuron in the same analysis was neuron 11 of layer 19.

### 2.6 First edit: standard directional ablation

Attribution only proposed candidates; experiments decided. Astra's sequence was:

1. Zero out a single neuron (column 11 of $W_{19}$): weak effect.
2. Directional ablation on layer 19 only: still weak.
3. Directional ablation on layers 15 and 19 together:
   $$W'_{15} = W_{15} - u^{(16)}\big(u^{(16)\top} W_{15}\big), \qquad W'_{19} = W_{19} - u^{(20)}\big(u^{(20)\top} W_{19}\big)$$
   with $W_{15}, W_{19} \in \mathbb{R}^{1024 \times 3584}$ and $u^{(16)}, u^{(20)} \in \mathbb{R}^{1024}$. On 128 new practice problems, accuracy went from **8.6% to 32.8%**.

Astra also probed wider sets (all MLPs in layers 12–22; layers 15, 19 and 22) before settling on 15 and 19, and later confirmed with single-layer isolation that both layers contribute.

Note how this differs from Arditi et al. (Section 1.5). They project **every** writer, so $u^\top h = 0$ everywhere. Astra edits only two `down_proj` matrices; the embedding, all attention outputs, and the other 22 MLPs can still write along $u$, so the residual stream's $u$ component is **not** zeroed. What the edit does is stop the two biggest MLP amplifiers from growing the looping signal (and, at α = 6 in Part 3, actively push against it). "Cutting the two main amplifiers" is a more accurate description than "removing the direction from the representation".

---

## Part 3. Beyond ablation: from α = 1 to the final edit

### 3.1 Adding a strength parameter α

Astra generalized the projection with a strength $\alpha$:

$$W' = W - \alpha\, u\,(u^\top W)$$

Arditi et al. have no such parameter; their operation corresponds to $\alpha = 1$. (Some later open-source "abliteration" tools added a similar coefficient, usually near 1.) Per column, the component along $u$ becomes $(1 - \alpha)\,c_j$:

| α | Component along $u$ after edit | Meaning |
|---|---|---|
| 0 | $c_j$ | No change |
| 1 | $0$ | Standard directional ablation |
| 2 | $-c_j$ | Reflection |
| 6 | $-5\,c_j$ | Reversed and amplified 5× |

For $\alpha > 1$ the edit no longer removes a direction; it pushes against it, which is closer to steering. Unlike activation steering, which adds a fixed vector to every input, this weight edit pushes in proportion to the neurons' activations.

Dose-response with all 3584 columns edited:

| α | Result |
|---|---|
| 1 | 128 new problems: 8.6% → 32.8% |
| 2 | Another 128 new problems × 2 samples: 8.2% → 43.0% |
| 3 | Exploratory: 8.2% → 48.0% |

Larger α gave larger gains, but also more answers that finished with a wrong result.

### 3.2 Mechanism checks with teacher forcing

Astra wrote six new contexts (geometry, probability, sequences, modular arithmetic, algebra, calculus), each with a productive and a repetitive continuation, plus controls that close `</think>` before the calculation is finished. Comparing per-token log-probabilities before and after the edit, the repetitive continuations became clearly less likely while the productive ones were almost unchanged, and the probability of closing thinking prematurely stayed negligible. This argues against the edit simply forcing early stopping.

Astra also found copying heads (layer 15 head 7, layer 19 head 5) that attend to the matching position in the previous repetition. Halving their outputs was less selective than the MLP edit, so this route was dropped.

### 3.3 Controls

- **Norm-matched random direction:** a random direction orthogonal to $u$, with the same Frobenius norm of the change ($\|W' - W\|_F$, matched within 0.03%). Result: +1.6pp ± 3.5, i.e. no effect. The direction $u$ matters, not the size of the change.
- **Single-layer isolation:** editing only layer 15 or only layer 19 each reduced the likelihood of repetitive text, confirming that both layers contribute.

### 3.4 Restricting the edit to 64 neurons per layer

Each neuron $j$ in layer $\ell$ (15 or 19) gets a score:

$$\text{score}_j = \underbrace{(u^\top w_j)}_{c_j} \times \underbrace{\Delta \bar a_j}_{\text{activation difference}}$$

- $w_j$: column $j$ of `down_proj`; $u$: the layer's direction ($u^{(16)}$ or $u^{(20)}$).
- $c_j$: how much neuron $j$ writes along $u$ per unit of activation.
- $\Delta \bar a_j$: neuron $j$'s activation difference "repetitive − productive", averaged over the last 24 tokens of each text, then over the 4 pairs (same forward-pass data as before).

This score is exactly neuron $j$'s share of the MLP part from Section 2.5:

$$u^\top W\, \Delta \bar a = \sum_{j=1}^{3584} (u^\top w_j)\, \Delta \bar a_j = \sum_{j=1}^{3584} \text{score}_j$$

A positive score means the neuron widens the gap along $u$: either $c_j > 0$ and it fires more during loops, or $c_j < 0$ and it fires less. The 3584 scores are sorted and the top 64 are kept, separately per layer. On the four scoring pairs these 64 account for about 50% (layer 15) and 56% (layer 19) of the total positive score.

The sign of $u$ matters here, even though it does not for the projection ($uu^\top$ is sign-invariant). With a flipped $u$, the top scores would pick neurons that suppress looping. Before scoring, Astra oriented $u$ so that "repetitive − productive" projects positively onto it, using a separate set of contrast texts.

The edit becomes

$$W' = W - \alpha\; u\,\big(s \odot (u^\top W)\big)$$

where $s \in \{0,1\}^{3584}$ is 1 only for the 64 selected neurons and $\odot$ is elementwise multiplication. At α = 4, editing only 64 columns performed about the same as editing all columns. Astra then raised α to 6; an exploratory test on 64 problems went from 7/64 to 35/64.

### 3.5 Final edit

$$W' = W - 6\; u\,\big(s \odot (u^\top W)\big), \quad \text{applied to layers 15 and 19}$$

- Unselected columns are unchanged.
- Each selected column becomes $w'_j = w_j - 6\,c_j\,u$: its component along $u$ goes from $c_j$ to $-5\,c_j$, and its orthogonal component is unchanged.
- 64 columns × 1024 rows × 2 layers = **131,072** changed weights, about **0.016%** of the model.
- Relative Frobenius change $\|W' - W\|_F / \|W\|_F$: **10.6%** (layer 15) and **8.1%** (layer 19). The count of changed weights is tiny, but the change is concentrated: the selected columns are rewritten substantially.

So the final edit is not strictly an ablation. It is a strong reversal applied to a small set of neurons, built on the directional-ablation framework.

### 3.6 Frozen confirmation and results

From this point no parameters were changed. On 1,152 practice problems never used before, accuracy went from **10.6% to 60.4%**, with 579 problems rescued and 5 made worse. Correct edited answers still had long reasoning (median about 9,200 tokens), another sign the edit did not just shorten thinking.

The authors' evaluation on held-out benchmarks (thinking on, up to 32,768 tokens):

| Qwen3.5-0.8B | MATH-500 | GSM8K | IFEval | Hit token limit |
|---|---|---|---|---|
| Base | 7.9% | 13.8% | 21.0% | 93.0% |
| Edited (50 forward passes) | 48.1% | 67.1% | 62.3% | 33.1% |

Two further observations from the paper:

- **Budget matters.** With only 10 or 25 forward passes, looping barely decreased and the gains were much smaller. The 50-pass run had room for the diagnosis, the attribution, and the controls above.
- **Removing an obstacle vs. adding an ability.** On Gemma 3 270M, which rarely loops, Astra could not meaningfully improve math scores. The Qwen gain came from removing a failure mode that was hiding ability the model already had; a weight edit like this cannot add ability that is not there.

---

## Sources

- Blog post: https://rikiyat.github.io/blog/weight-space-programming/
- Edited model: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- Full logs: https://github.com/RikiyaT/lm_understand-logs, run `exp8_math/runs/p50c/`
  - `readable/trajectory.md`: all of Astra's code and outputs
  - `work_notes/final_report.md`, `work_notes/experiment_history.md`: summaries and the final formula
  - `requests/fwd_4.json.claimed`: the 8 contrast texts
  - `edit/edit.npz`, `work_notes/frozen_weights_manifest.json`: the bit-exact edit and the 64 selected neuron indices per layer
- Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*.
