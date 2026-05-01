# LLFlow: Comprehensive Study Notes (Revised)

**Paper:** Low-Light Image Enhancement with Normalizing Flow (AAAI 2022, Oral)  
**Authors:** Yufei Wang, Renjie Wan, Wenhan Yang, Haoliang Li, Lap-Pui Chau, Alex C. Kot  
**Task:** Low-light image enhancement (converting dark images into normally exposed ones)

---

## Table of Contents

1. [What Problem Does This Paper Solve?](#1-what-problem-does-this-paper-solve)
2. [The Core Idea (Plain English)](#2-the-core-idea-plain-english)
3. [Background: What Is a Normalizing Flow?](#3-background-what-is-a-normalizing-flow)
4. [The Mathematics of LLFlow](#4-the-mathematics-of-llflow)
5. [Architecture](#5-architecture)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Key Takeaways and Mental Models](#8-key-takeaways-and-mental-models)

---

## 1. What Problem Does This Paper Solve?

### The task

Given a **low-light image** (dark, noisy, low visibility), produce a **normally exposed image** (bright, clean, natural colors).

### Why it is hard

Low-light enhancement is **ill-posed**: for a single dark image, there are many valid "correct" bright versions. They might differ in:

- Overall brightness level
- Color saturation
- How much shadow detail is revealed
- Local contrast choices

### What previous methods do wrong (according to this paper)

Most CNN-based methods learn a deterministic mapping:

$$
\hat{x} = F_\theta(x_l)
$$

trained with pixel-wise losses like:

$$
\mathcal{L} = \lvert \hat{x} - x_{ref} \rvert_1
$$

The paper argues this is fundamentally limited because:

1. **Regression to mean:** When multiple valid outputs exist, the network learns to average them, producing blurry or washed-out results.
2. **Weak error model:** An $L_1$ loss treats all pixel errors equally. It cannot tell the difference between "slightly different brightness" (acceptable) and "noisy artifacts" (unacceptable). The paper shows an example (Fig. 1) where a noisy image and a slightly-brighter image have the same $L_1$ distance to the reference, but humans clearly prefer the brighter one.
3. **No uncertainty modeling:** The network produces exactly one output and has no way to represent that it is uncertain about some regions.

### What this paper proposes instead

Rather than predicting one fixed output, **model the full conditional distribution** of valid normally exposed images given the low-light input:

$$
p(x_{ref} \mid x_l)
$$

Use a **normalizing flow** to make this distribution tractable and trainable with exact likelihood.

> **Key insight in one sentence:** LLFlow replaces "make every output pixel close to GT" with "make GT land in a simple latent distribution under an invertible, condition-aware transform."

---

## 2. The Core Idea (Plain English)

Think of it this way:

- A **standard CNN** says: "Given this dark image, here is THE one correct bright image."
- **LLFlow** says: "Given this dark image, here is the SPACE of plausible bright images, and here is how likely each one is."

To make this space tractable, LLFlow uses a trick:

1. Define an **invertible neural network** that can transform images into simple latent codes and back.
2. Train it so that **real, valid normally exposed images** get mapped to **simple, Gaussian-like latent codes**.
3. At test time, start from a simple latent code and run the network backwards to produce a normally exposed image.

The "conditional" part means the invertible network is always aware of what the low-light input looks like, so it generates outputs that are appropriate for that specific dark scene.

---

## 3. Background: What Is a Normalizing Flow?

This section explains normalizing flows from scratch. If you already know this, skip to Section 4.

### 3.1 The fundamental idea

A normalizing flow is a way to model complex probability distributions by transforming a simple distribution (like a Gaussian) through a series of invertible functions.

**Analogy:** Imagine you have a bag of perfectly round marbles (simple distribution). You pass them through a series of tubes that reshape them into complex, interesting shapes (complex distribution). The key constraint is that each tube is reversible — you can always push the shaped objects back through the tube to get round marbles again.

### 3.2 Why invertibility matters

If a transformation $f$ is invertible, then given ANY output, you can uniquely recover the input. This means:

- **Forward direction:** complex data $\rightarrow$ simple latent code (used for training)
- **Reverse direction:** simple latent code $\rightarrow$ complex data (used for generation)

Both directions use the same network weights. No information is lost.

### 3.3 The change of variables formula

This is the mathematical heart of all normalizing flows. It answers the question: "If I know the probability of $z$ and I know how $z$ maps to $x$, what is the probability of $x$?"

Suppose we have:
- A simple base distribution $p_Z(z)$ (e.g., a Gaussian)
- An invertible function $f$ such that $z = f(x)$

Then the probability density of $x$ is:

$$
p_X(x) = p_Z(f(x)) \cdot \left\lvert \det \frac{\partial f(x)}{\partial x} \right\rvert
$$

**What does each piece mean?**

| Term | Meaning |
|------|---------|
| $p_Z(f(x))$ | "How likely is the latent code that $x$ maps to?" |
| $\det \frac{\partial f(x)}{\partial x}$ | "How much does $f$ stretch or compress space at the point $x$?" |

**Why do we need the determinant?** When you stretch or compress space, probabilities must be adjusted. If a transformation squishes a region to half its size, the density in that region doubles. The Jacobian determinant captures exactly this volume change.

### 3.4 Log-likelihood form

In practice, we work with log-probabilities (they are numerically more stable and turn products into sums):

$$
\log p_X(x) = \log p_Z(f(x)) + \log \left\lvert \det \frac{\partial f(x)}{\partial x} \right\rvert
$$

### 3.5 Composing multiple layers

A normalizing flow is typically a sequence of simpler invertible functions:

$$
f = f_N \circ f_{N-1} \circ \cdots \circ f_1
$$

The log-determinant of a composition is the sum of individual log-determinants:

$$
\log \left\lvert \det \frac{\partial f}{\partial x} \right\rvert = \sum_{n=1}^{N} \log \left\lvert \det \frac{\partial f_n}{\partial h_{n-1}} \right\rvert
$$

where $h_0 = x$, $h_n = f_n(h_{n-1})$, and $z = h_N$.

> **Plain English:** Each layer contributes its own volume-change factor. We add them all up (in log-space) to get the total volume change of the entire transformation.

### 3.6 Training a normalizing flow

Training is straightforward: **maximize the log-likelihood** of the training data.

For each training sample $x$:
1. Compute $z = f(x)$ (forward pass through the invertible network)
2. Evaluate how likely $z$ is under the base distribution $p_Z(z)$
3. Add up the log-determinants from each layer
4. Form the log-likelihood: $\log p_X(x) = \log p_Z(z) + \sum_n \log \lvert \det J_n \rvert$
5. Minimize the negative of this (i.e., minimize NLL)

No adversarial training. No reconstruction loss needed. Just maximum likelihood.

### 3.7 The Glow-style flow step

LLFlow builds on the **Glow** architecture (Kingma & Dhariwal, 2018). Each "flow step" in Glow consists of three sub-operations:

1. **ActNorm** (Activation Normalization): A learned per-channel affine transform (scale + shift). Think of it like a smarter version of batch normalization that works at test time too.

2. **Invertible 1x1 Convolution:** A learned channel mixing operation. It acts like a rotation/permutation of channels. This ensures that information gets mixed between channels (otherwise the coupling layer below would always ignore the same channels).

3. **Affine Coupling Layer:** This is the workhorse. It splits the input channels into two halves:
   - First half $z_1$ passes through unchanged
   - Second half $z_2$ is transformed using parameters computed from $z_1$:

$$
z_2' = z_2 \cdot s(z_1) + t(z_1)
$$

where $s$ and $t$ are neural networks that predict scale and shift parameters.

**Why is this invertible?** Because $z_1$ passes through unchanged, you can always recompute $s(z_1)$ and $t(z_1)$, then reverse the operation on $z_2'$ to get $z_2$.

**Why is the determinant easy to compute?** The Jacobian of this operation is triangular, so its determinant is just the product of diagonal entries: $\prod s(z_1)$.

### 3.8 Multi-scale architecture

Glow (and LLFlow) use a "squeeze" operation that reshapes spatial dimensions into channel dimensions:

- A $H \times W \times C$ tensor becomes $\frac{H}{2} \times \frac{W}{2} \times 4C$

This is done at each "level" of the flow. It is a simple reshaping (no learned parameters, perfectly invertible) that allows the flow to operate at progressively lower spatial resolutions but with more channels.

---

## 4. The Mathematics of LLFlow

Now that you understand flows in general, here is how LLFlow specifically formulates its problem.

### 4.1 The conditional flow

LLFlow is not a standard (unconditional) flow. It is a **conditional** flow.

A standard flow models $p(x)$. LLFlow models $p(x \mid x_l)$, the distribution of normally exposed images **conditioned on** a low-light image $x_l$.

The transformation becomes:

$$
z = f_\theta(x_{ref} ; x_l)
$$

Read this as: "The invertible function $f_\theta$ maps the reference image $x_{ref}$ into latent code $z$, and it does so while being aware of the low-light image $x_l$."

The conditioning enters through the internal layers of the network (specifically through the affine coupling layers, which receive features extracted from $x_l$).

### 4.2 The full objective (paper Eq. 3 and 4)

Applying the change of variables formula in the conditional setting:

$$
p_{flow}(x_{ref} \mid x_l) = p_Z(z) \cdot \left\lvert \det \frac{\partial f_\theta(x_{ref}; x_l)}{\partial x_{ref}} \right\rvert
$$

Taking the negative log to get the training loss:

$$
\mathcal{L}(x_l, x_{ref}) = -\log p_Z(z) - \sum_{n=0}^{N-1} \log \left\lvert \det \frac{\partial f_n(h_n; g_n(x_l))}{\partial h_n} \right\rvert
$$

where:
- The network is split into $N$ invertible layers $f_0, f_1, \ldots, f_{N-1}$
- $h_0 = x_{ref}$ (the input to the first layer is the reference image)
- $h_{n+1} = f_n(h_n; g_n(x_l))$ (each layer transforms the data while conditioned on encoder features)
- $z = h_N$ (the final output is the latent code)
- $g_n(x_l)$ is the feature from the encoder at the appropriate scale for layer $n$

**What this means intuitively:**

The model is trained to make two things happen simultaneously:

1. **The latent code $z$ should look Gaussian** (first term penalizes unlikely latent codes)
2. **The transformation should be well-behaved** (second term accounts for how the layers reshape probability)

### 4.3 The paper's motivation: Why NLL is better than $L_1$

The paper makes an explicit connection between $L_1$ loss and likelihood. Minimizing $L_1$ is equivalent to maximum likelihood under a **Laplace distribution**:

$$
p(x \mid x_{ref}) = \frac{1}{2b} \exp\left( -\frac{\lvert x - x_{ref} \rvert}{b} \right)
$$

This is a fixed, simple distribution that treats all pixel errors the same way. The paper calls this "too simple" — it cannot distinguish between structured artifacts (which are perceptually bad) and slight brightness shifts (which are perceptually fine).

The normalizing flow learns a **much more complex conditional distribution** that better captures what "natural, well-exposed images" actually look like. It can assign low probability to artifact-heavy images that might have the same pixel-level error as a clean image.

### 4.4 The LLFlow-specific latent prior (paper Eq. 7, 8, 9)

This is the paper's most distinctive mathematical contribution.

**Standard flows** use $p_Z(z) = \mathcal{N}(0, I)$ as the base distribution.

**LLFlow** uses a Gaussian with a **non-zero, data-dependent mean**:

$$
p_Z(z) = \mathcal{N}(\mu, I)
$$

where $\mu$ is derived from the scene's color information.

Specifically, during training, the mean is randomly chosen between two sources:

$$
\mu = r(C(x_{ref}), g(x_l))
$$

where:
- $C(x_{ref})$ is the color map directly computed from the reference image
- $g(x_l)$ is the color map predicted by the encoder from the low-light image
- $r(a, b)$ is a random selection function:

$$
r(a, b) = \begin{cases} a & \text{with probability } p \\ b & \text{with probability } 1 - p \end{cases}
$$

The paper uses $p = 0.2$ (so 80% of the time, the encoder's prediction is used, and 20% of the time, the reference color map is used).

**Why do this?**

1. Using the encoder output $g(x_l)$ as the mean forces the encoder to learn a good color map estimate (because the flow's loss depends on how well $z$ matches this prior).
2. Occasionally using $C(x_{ref})$ gives the flow a "perfect" prior during training, which stabilizes learning.
3. At inference time, only $g(x_l)$ is available, so the model must have learned to work with it.

**Why not just use zero mean?**

A zero-mean prior says "the latent code should be centered at zero." The LLFlow prior says "the latent code should be centered around the scene's color structure." This encodes the insight that the chromatic content of the scene is relatively stable across illumination changes — it is the brightness that changes, not the underlying colors.

> **Plain English:** Instead of telling the model "map good images to generic noise," LLFlow tells it "map good images to codes that look like their underlying color structure." This gives the model a much more informative target.

### 4.5 The color map (paper Eq. 5)

The color map is defined as:

$$
C(x) = \frac{x}{\text{mean}_c(x)}
$$

where $\text{mean}_c(x)$ is the per-pixel mean across RGB channels.

**Example:** If a pixel has RGB values $(0.6, 0.3, 0.1)$, the mean is $\frac{0.6 + 0.3 + 0.1}{3} = 0.333$, so the color map value is $(\frac{0.6}{0.333}, \frac{0.3}{0.333}, \frac{0.1}{0.333}) = (1.8, 0.9, 0.3)$.

**What this captures:** The relative proportions of R, G, B at each pixel, independent of overall brightness. Whether the pixel is in bright light or shadow, this ratio stays roughly the same (this is the Retinex assumption — the color of a surface is intrinsic to the surface, not the illumination).

> **Plain English:** The color map answers "what color is this?" while ignoring "how bright is this?"

### 4.6 The noise map (paper Eq. 6)

$$
N(x) = \max\left( \lvert \nabla_x C(x) \rvert, \lvert \nabla_y C(x) \rvert \right)
$$

This computes spatial gradients of the color map and takes the maximum. Areas with high noise will have rapidly fluctuating color maps, producing high gradient values.

> **Plain English:** "Where is the color estimate unstable?" If the color map is changing rapidly pixel-to-pixel, that is likely noise rather than real scene content.

### 4.7 Putting the math all together

Here is the full training objective in one place, with all pieces labeled:

$$
\mathcal{L} = \underbrace{-\log \mathcal{N}(z \mid \mu, I)}_{\text{latent prior term}} - \underbrace{\sum_{n=0}^{N-1} \log \lvert \det J_n \rvert}_{\text{volume change term}}
$$

where:
- $z = f_\theta(x_{ref}; x_l)$ — run the reference image through the conditional flow
- $\mu = r(C(x_{ref}), g(x_l))$ — the scene-dependent prior mean
- $J_n$ — the Jacobian of layer $n$

Expanding the Gaussian log-probability:

$$
-\log \mathcal{N}(z \mid \mu, I) = \frac{1}{2} \lVert z - \mu \rVert^2 + \text{const}
$$

So the latent prior term is essentially saying: "the latent code $z$ should be close to $\mu$." If you squint, this looks like a very sophisticated $L_2$ loss — but it is operating in a learned, invertible latent space rather than in pixel space, which makes it much more powerful.

### 4.8 What does training actually minimize?

In the released implementation, the final scalar loss reported per sample is:

$$
\text{nll} = \frac{-\text{objective}}{\log(2) \cdot \text{pixels}}
$$

where:
- $\text{objective} = \text{logdet} + \log p_Z(z \mid \mu)$
- $\text{pixels}$ = total number of pixels (for normalization to bits-per-pixel)

This is the **negative log-likelihood in bits per dimension** (a standard way to report flow likelihoods).

---

## 5. Architecture

### 5.1 High-level overview

LLFlow consists of **two components** that work together:

```
                    Low-light image (x_l)
                           |
                           v
              +------------------------+
              |   Conditional Encoder  |  <-- Extracts features + color map
              +------------------------+
                    |              |
         color map g(x_l)    multi-scale features
                    |              |
                    v              v
              +----------------------------+
              |     Invertible Flow Net     |  <-- Maps images <-> latent codes
              +----------------------------+
                           |
             (Training: x_ref -> z)  (Inference: z -> enhanced image)
```

### 5.2 Component A: The Conditional Encoder

**Purpose:** Extract useful conditioning information from the low-light image.

**What it takes as input (12 channels total):**

| Input | Channels | Purpose |
|-------|----------|---------|
| $x_l$ (low-light image, log-transformed) | 3 | Raw scene information |
| $h(x_l)$ (histogram-equalized version) | 3 | Better global contrast |
| $C(x_l)$ (color map) | 3 | Illumination-invariant color |
| $N(x_l)$ (noise map) | 3 | Where the signal is unreliable |

**Architecture:** Based on RRDB (Residual-in-Residual Dense Blocks) from ESRGAN.

The RRDB blocks are a stack of dense blocks that each:
- Use 5 convolutional layers with dense connections (each layer receives all previous layer outputs)
- Apply residual scaling (multiply output by 0.2 before adding)
- Stack 3 such dense blocks per RRDB block

**What it produces:**
- **Multi-scale conditioning features** at different spatial resolutions (used to condition the flow at appropriate levels)
- **A refined color map** $g(x_l)$ (used as the latent prior mean)

> **Plain English:** The encoder's job is to look at the dark image and say: "Here is what I can figure out about the scene's colors and structure, at multiple scales of detail." This information then guides the flow network.

### 5.3 Component B: The Invertible Flow Network

**Purpose:** Learn the invertible mapping between image space and latent space, conditioned on the encoder outputs.

**Structure:** 3 levels, each consisting of:

1. **Squeeze layer:** Reshapes spatial dimensions into channels ($H \times W \times C \rightarrow \frac{H}{2} \times \frac{W}{2} \times 4C$)
2. **K flow steps** (K=12 in the standard model, K=4 in the small model)

**Each flow step contains (in forward order):**

1. **ActNorm:** Learned per-channel scale and bias (normalizes activations)
2. **Invertible 1x1 convolution:** Mixes channels (a learnable permutation)
3. **Conditional affine coupling:** The main transformation (described below)

**The conditional affine coupling in detail:**

This is where the low-light conditioning actually enters the flow. In LLFlow's implementation (`CondAffineSeparatedAndCond`), the coupling has two stages:

**Stage 1 — Feature conditioning:**
Given encoder features $ft$ from the appropriate scale:

$$
z' = (z + t_{feat}(ft)) \cdot s_{feat}(ft)
$$

where $s_{feat}$ and $t_{feat}$ are small neural networks that predict scale and shift from the encoder features.

**Stage 2 — Self conditioning (standard coupling):**
Split $z'$ into halves $z_1, z_2$:

$$
z_2' = (z_2 + t_{self}(z_1, ft)) \cdot s_{self}(z_1, ft)
$$

Both stages are invertible (divide instead of multiply, subtract instead of add) and have tractable Jacobian determinants (the determinant is just the product of the scale factors).

> **Plain English:** At every flow step, the network asks: "Given what the encoder tells me about this dark image, how should I transform the data?" The conditioning is not applied once at the beginning — it is applied at EVERY step, at EVERY scale.

### 5.4 Important architectural notes

**No latent split in the released configs:** Although the code supports splitting latent variables at intermediate levels (common in Glow), all released configurations disable this (`split.enable: false`). The full latent code has the same total dimensionality as the input image.

**Same-resolution operation:** Despite code variable names suggesting super-resolution heritage, LLFlow operates at `scale: 1`. Input and output have the same spatial resolution.

**The standard model vs. the small model:**

| Property | Small model | Standard model |
|----------|-------------|----------------|
| Encoder channels (nf) | 32 | 64 |
| RRDB blocks (nb) | 4 | 24 |
| Flow steps per level (K) | 4 | 12 |
| Flow levels (L) | 3 | 3 |

---

## 6. Training Pipeline

This section is especially important for future work. It describes exactly what happens during training, step by step.

### 6.1 Overview of one training iteration

```
1. Load a (low-light, normal-light) image pair
2. Preprocess: crop, flip, rotate, noise, log-transform, histogram-eq
3. Forward pass:
   a. Encoder processes low-light image → conditioning features + color map
   b. Flow forward pass: reference image → latent code z (accumulating log-determinants)
   c. Evaluate latent prior: how likely is z under N(mu, I)?
   d. Combine into NLL loss
4. Backward pass: compute gradients
5. Optimizer step: update all parameters
```

### 6.2 Data loading and preprocessing

**Dataset:** Paired low-light / normal-light images from LOL or LOL-v2.

**Augmentation pipeline (in order):**

1. **Random crop** to $160 \times 160$ patches
2. **Random horizontal flip** (50% probability)
3. **Random rotation** (0°, 90°, or 270°)
4. **Optional noise injection:** With some probability, add Gaussian noise to the low-light image (makes the model more robust)
5. **Log transform** of the low-light image:
   $$
   x_l \leftarrow \log(x_l + 10^{-3})
   $$
   This is important — it compresses the extremely dark values into a more manageable range. Without this, very dark pixels (near 0) would be numerically difficult to work with.
6. **Concatenate histogram-equalized version:** The final "LQ" tensor has 6 channels (3 from log-transformed low-light + 3 from histogram equalization)

**The GT image is NOT log-transformed.** It stays in standard [0, 1] RGB range.

### 6.3 The forward pass in detail

Here is what happens inside `optimize_parameters()`:

**Step 1: Encoder forward pass**

```
lr_enc = self.netG.rrdbPreprocessing(self.var_L)
```

This runs the low-light input through the conditional encoder. Inside the encoder:
- Exponentiates the first 3 channels back to get raw low-light intensities
- Computes color map (ratio of each channel to channel sum)
- Computes noise map (max of x/y gradients of color map)
- Concatenates everything and runs through RRDB blocks
- Produces multi-scale features and a refined color map

**Step 2: Dequantization noise**

The reference image gets tiny uniform noise added:

$$
x_{ref}' = x_{ref} + \frac{U(0, 1) - 0.5}{\text{quant}}
$$

where `quant` = 32 in the configs. This is a standard flow technique to avoid probability mass collapsing onto discrete values. The log-determinant is adjusted by $-\log(\text{quant}) \times \text{pixels}$.

**Step 3: Flow forward pass through all layers**

The reference image passes through all squeeze layers and flow steps. At each flow step:
- ActNorm normalizes (contributes to logdet)
- 1x1 convolution mixes channels (contributes to logdet)
- Affine coupling transforms data using encoder features (contributes to logdet)

After all layers, we have the latent code $z$ and the total accumulated $\text{logdet}$.

**Step 4: Evaluate the latent prior**

The random selector chooses the prior mean (80% encoder prediction, 20% reference color map):

```python
mean = lr_enc['color_map'] if random() > 0.2 else gt / (gt.sum(dim=1) + 1e-4)
```

Both are squeezed to match the latent spatial dimensions.

Then evaluate:

$$
\text{objective} = \text{logdet} + \log \mathcal{N}(z \mid \mu, I)
$$

**Step 5: Compute NLL**

$$
\text{nll} = \frac{-\text{objective}}{\log(2) \cdot \text{pixels}}
$$

This normalizes to bits per dimension.

### 6.4 The loss function

In the standard released configs:

$$
\mathcal{L}_{total} = \text{weight\_fl} \times \text{nll\_loss}
$$

with `weight_fl = 1` and `weight_l1 = 0`.

**There is NO perceptual loss, NO VGG loss, NO adversarial loss in the standard training.**

The model trains purely on exact negative log-likelihood. This is one of the cleanest aspects of the method — it demonstrates that a well-designed flow can achieve strong perceptual quality without needing any auxiliary losses.

The code does support an optional $L_1$ reconstruction loss (which runs the reverse flow to generate an image, then compares to GT), but this is disabled in all released configs.

### 6.5 Delayed encoder training

An important training strategy in the config:

```yaml
train_RRDB: false
train_RRDB_delay: 0.5
```

This means:
- The encoder (RRDB) starts with **frozen weights** (not trained)
- After 50% of total training iterations, the encoder unfreezes and begins training

**Why?** The flow needs to stabilize before the encoder's output (which conditions the flow) starts changing. If both change simultaneously from the start, training can be unstable. This staged approach is common in conditional flow methods.

In the optimizer, the encoder and flow have **separate parameter groups** with potentially different learning rates:
- Flow parameters: `lr_G = 5e-4`
- Encoder parameters: can have a different rate (`lr_RRDB`)

### 6.6 Learning rate schedule

Adam optimizer with:
- $\beta_1 = 0.9$, $\beta_2 = 0.99$
- Initial LR: $5 \times 10^{-4}$
- **MultiStepLR** schedule: LR is halved at relative milestones [0.5, 0.75, 0.9, 0.95] of total training

For LOL dataset: ~40,000-45,000 total iterations.

### 6.7 Mixed-precision training

The code uses `torch.cuda.amp.GradScaler` for mixed-precision (FP16) training:

```python
self.scaler.scale(total_loss).backward()
self.scaler.step(self.optimizer_G)
self.scaler.update()
```

This reduces memory usage and speeds up training on modern GPUs.

### 6.8 Validation during training

Every `val_freq` iterations (200 in standard configs):

1. Run inference on validation set (reverse flow with mean latent)
2. Adjust output brightness to match GT mean (following the KinD evaluation protocol)
3. Compute PSNR and SSIM
4. Save visual results
5. Save best model based on PSNR

**Important:** The brightness adjustment before evaluation means reported PSNR numbers include this post-processing. If you reproduce without it, numbers will differ.

### 6.9 Summary of what makes this training pipeline different

Compared to a standard CNN enhancement training loop:

| Aspect | Standard CNN | LLFlow |
|--------|--------------|--------|
| Loss computation | Forward CNN, compare output to GT | Forward flow on GT, evaluate latent likelihood |
| What flows through the net during training | Low-light image | Normal-light reference image |
| Loss type | Pixel reconstruction ($L_1$, $L_2$, perceptual) | Exact negative log-likelihood |
| Conditioning | N/A (input IS the conditioning) | Separate encoder branch processes low-light |
| Gradient flow | Through single forward pass | Through entire invertible chain |
| Memory | Standard | High (must store all intermediate activations for invertibility) |

> **Key conceptual difference:** In a standard CNN, the low-light image IS the network input. In LLFlow training, the REFERENCE image is fed through the flow, and the low-light image only enters as conditioning through the encoder.

---

## 7. Inference Pipeline

### 7.1 Overview

At test time, the flow runs in **reverse**:

```
1. Encoder processes low-light image → features + color map g(x_l)
2. Construct latent code z from the color map (or sample around it)
3. Run reverse flow: z → enhanced image
4. (Optional) brightness adjustment
```

### 7.2 Latent code selection

The paper offers two strategies:

**Deterministic (used in practice):** Set $z = g(x_l)$ directly (the encoder's color map prediction, squeezed to the right dimensions). This produces one fixed output.

**Stochastic (theoretical):** Sample $z \sim \mathcal{N}(g(x_l), \sigma^2 I)$ with temperature $\sigma$ to get diverse outputs. Higher $\sigma$ = more variation but potentially more artifacts.

The released configs use `heat: 0`, meaning the deterministic path.

### 7.3 The reverse flow in detail

Reverse operations happen in **opposite order** to forward:

For each layer (from last to first):
1. Undo affine coupling (divide by scale, subtract shift)
2. Undo 1x1 convolution (multiply by inverse)
3. Undo ActNorm (subtract bias, divide by scale)

Between levels: undo squeeze (reshape channels back to spatial dimensions).

The encoder features condition the reverse pass in the same way as the forward — the affine coupling layers receive the same encoder outputs.

### 7.4 Optional color map post-processing

If `encode_color_map` is enabled (disabled in released configs), the output gets an additional color correction:

$$
x_{out} = x_{out} \times \frac{\text{color\_map}_{predicted}}{\text{color\_map}_{output}}
$$

This forces the output's color ratios to match the encoder's prediction.

---

## 8. Key Takeaways and Mental Models

### 8.1 The one-sentence summary

> LLFlow uses an encoder to estimate stable scene/color cues from a dark image, then trains an invertible conditional model so that real normal-light images become likely under a simple latent distribution conditioned on those cues.

### 8.2 Five things to remember

1. **The training objective is pure NLL** — no perceptual loss, no adversarial loss, no pixel reconstruction in the standard recipe.

2. **The latent prior is color-map-centered** — not zero-mean Gaussian. This is the paper's main contribution beyond just "apply a flow to this task."

3. **Training pushes the REFERENCE through the flow** — this is counterintuitive. The low-light image only enters as conditioning. The reference is what gets density-modeled.

4. **Inference runs the flow BACKWARDS** — starting from a latent code (derived from the encoder) and producing an enhanced image.

5. **The encoder does heavy lifting** — it is not just a feature extractor. It produces the latent prior mean AND multi-scale conditioning features that guide every single flow step.

### 8.3 Good analogies for understanding

**The flow as a learned coordinate system:**
Think of the invertible network as defining a new coordinate system for images. In this coordinate system, all valid normally-exposed images for a given scene cluster neatly around a single point (the color-map mean). Bad images (noisy, artifact-heavy) are far from this center. Training teaches the network to define this coordinate system well.

**The encoder as a scene interpreter:**
The encoder reads the dark image and says "I think this scene has these colors and this structure." The flow then uses that interpretation to define what a "plausible enhancement" means for this specific scene.

**The random prior selector as curriculum:**
During training, giving the flow the "correct answer" (reference color map) 20% of the time is like giving a student the answer key occasionally — it prevents the flow from having to rely entirely on potentially noisy encoder predictions early in training.

### 8.4 Common misconceptions to avoid

| Misconception | Reality |
|---------------|---------|
| "LLFlow uses VGG/perceptual loss" | No. Pure NLL training in released configs. |
| "The latent space is zero-mean Gaussian" | No. It is Gaussian with color-map-dependent mean. |
| "LLFlow generates diverse outputs" | In theory yes, but the released implementation is effectively deterministic. |
| "The flow splits latents at different scales" | The code supports this, but all released configs disable it. |
| "The model explicitly learns a noise model" | It uses a noise map as input to the encoder, but does not learn a separate noise distribution. |
| "Training feeds the low-light through the flow" | No. The REFERENCE feeds through the flow. Low-light only conditions it. |

### 8.5 Quick reference for the paper's equations

| Paper Eq. | What it says |
|-----------|--------------|
| Eq. 1 | $L_1$ training = maximum likelihood under a Laplace distribution |
| Eq. 2 | The Laplace distribution that $L_1$ implicitly assumes |
| Eq. 3 | Change of variables: connects image density to latent density |
| Eq. 4 | The NLL loss decomposed layer by layer |
| Eq. 5 | Color map definition: $C(x) = x / \text{mean}_c(x)$ |
| Eq. 6 | Noise map definition: max of color map gradients |
| Eq. 7 | Full training objective with color-map prior |
| Eq. 8 | The Gaussian prior with color-map mean |
| Eq. 9 | The random selector between reference and predicted color maps |

### 8.6 For your future work

If you plan to build on this training pipeline, the most important things to understand deeply are:

1. **The forward-pass computation graph during training:** Reference → dequantize → squeeze+flow_steps (×3 levels) → latent → evaluate NLL. The gradients flow through the entire invertible chain back to both encoder and flow parameters.

2. **The encoder-flow coupling:** The encoder's output affects the flow in TWO ways: (a) through conditioning features at every coupling layer, and (b) through the prior mean. Changes to the encoder change the loss landscape for the flow and vice versa.

3. **The delayed training strategy:** Freezing the encoder early prevents instability. If you modify the architecture, you may need to tune when and how the encoder unfreezes.

4. **Memory constraints:** Flows are memory-hungry because invertibility requires storing intermediate activations. Mixed precision helps. Patch-based training (160×160) helps. But scaling to larger images or deeper flows requires careful memory management.
