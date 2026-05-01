# LLFlow Study Sheet

## Paper

**Title:** Low-Light Image Enhancement with Normalizing Flow  
**Short name:** LLFlow  
**Main task:** Convert a low-light image into a normally exposed image.

---

## 1. One-Sentence Summary

LLFlow models low-light enhancement as a **conditional density estimation problem** instead of a pure pixel regression problem.

The core idea is:

> LLFlow replaces "make every output pixel close to GT" with "make GT land in a simple latent distribution under an invertible, condition-aware transform."

That sentence is the best mental anchor for the whole paper.

---

## 2. Why This Paper Exists

Low-light enhancement is **ill-posed**.

For one low-light image, there is not always exactly one correct normally exposed output.

Different valid outputs can differ in:

- overall brightness
- local contrast
- color saturation
- how much shadow detail is revealed

Most standard CNN restoration methods learn a deterministic mapping like:

$$
\hat{x} = F_\theta(x_l)
$$

and train with a loss such as:

$$
\mathcal{L}_{L1} = \|\hat{x} - x_{ref}\|_1
$$

where:

- $x_l$ is the low-light image
- $x_{ref}$ is the normally exposed reference image

The paper argues that this is too restrictive because it encourages the network to predict a single average-looking answer. That can cause:

- over-smoothed results
- poor handling of ambiguity
- residual noise and artifacts
- weak color recovery

So the authors move from **deterministic reconstruction** to **probabilistic modeling**.

---

## 3. The Main Idea in Plain English

LLFlow says:

1. A low-light image does not determine one exact correct output.
2. So instead of predicting one output directly, learn the **conditional distribution** of valid normally exposed images.
3. Use a **normalizing flow** so this conditional distribution is tractable and trainable with exact likelihood.
4. Use an encoder to estimate an **illumination-invariant color map** that acts like a scene-dependent prior.

Another simple way to say it:

> The model learns what a plausible normally exposed image should look like, conditioned on the low-light input, rather than only learning how to match one paired target pixel by pixel.

---

## 4. Important Corrections to My Earlier Notes

These are the points that most needed correction.

### 4.1 LLFlow is **not** mainly a perceptual-loss or VGG-loss method

The main training objective is **negative log-likelihood (NLL)**.

The released configs in the repo use:

- `weight_fl: 1`
- `weight_l1: 0`

So the standard training is effectively **flow likelihood training**, not a hybrid `NLL + perceptual + reconstruction` setup.

There is an **optional L1 term** in the code for ablations or alternate training setups, but that is not the core method.

### 4.2 The latent prior is not just standard zero-mean Gaussian

The paper's main twist is that the latent variable is tied to a **color-map-based mean**, not just $\mathcal{N}(0, I)$.

Conceptually, the paper uses a latent prior centered around the illumination-invariant color information:

$$
z \sim \mathcal{N}(\mu(x_l), I)
$$

where $\mu(x_l)$ is related to the encoder output $g(x_l)$.

### 4.3 The repo is multi-scale, but the released configs do **not** use latent split

The flow architecture is still multi-level, but the provided configs set:

- `split.enable: false`

So do not describe the released implementation as if it is actively using split latents at different resolutions.

### 4.4 The noise handling is more limited than a full explicit noise generative model

The method does use a **noise map** derived from the color map gradients and it adds dequantization-like noise inside the flow pipeline, but it is not best described as a separate fully learned probabilistic noise model.

### 4.5 The released code is more deterministic at test time than the paper's theory suggests

The paper discusses sampling multiple valid outputs from a distribution around $g(x_l)$.

In the released configs and implementation, the practical path is mostly a **mean-based deterministic enhancement** rather than fully stochastic multi-sample generation.

---

## 5. Notation

I will use the following notation consistently.

- $x_l$: low-light input image
- $x_{ref}$: normally exposed reference image
- $x_h$: sometimes used in the paper for the high-light / normally exposed image
- $g(x_l)$: encoder output, especially the illumination-invariant color map
- $f_\theta(\cdot; x_l)$: conditional invertible mapping defined by the flow
- $z$: latent variable
- $C(x)$: color map derived from image $x$
- $N(x)$: noise map derived from the color map gradients

---

## 6. The Math: What the Paper Is Really Doing

This is the most important section if your goal is to understand the method deeply.

### 6.1 Why the paper criticizes pixel losses

The paper starts from the observation that minimizing an $L_1$ loss can be interpreted probabilistically.

If we predict $\hat{x} = F_\theta(x_l)$ and minimize:

$$
\|\hat{x} - x_{ref}\|_1
$$

that is closely related to assuming a Laplace-like conditional distribution around the target.

In the paper's framing, this is too weak a model of image quality because it does not strongly capture:

- natural image manifold structure
- structured artifacts
- perceptual realism
- one-to-many ambiguity

The key criticism is not that $L_1$ is mathematically invalid. It is that it is **too simple** for this problem.

### 6.2 Conditional flow formulation

Instead of directly outputting the enhanced image, LLFlow defines an invertible transformation that maps the reference image into a latent variable:

$$
z = f_\theta(x_{ref}; x_l)
$$

Here the mapping is **conditioned on the low-light input** or features extracted from it.

Because the transformation is invertible, we can also write the reverse direction:

$$
x_{ref} = f_\theta^{-1}(z; x_l)
$$

This matters because:

- training uses the forward direction: image to latent
- inference uses the reverse direction: latent to image

### 6.3 Change of variables

For a normalizing flow, the log-density of the image is:

$$
\log p(x_{ref} \mid x_l)
= \log p_Z(z \mid x_l)
+ \log \left| \det \frac{\partial f_\theta(x_{ref}; x_l)}{\partial x_{ref}} \right|
$$

with:

$$
z = f_\theta(x_{ref}; x_l)
$$

This is the mathematical backbone of the paper.

Interpretation:

- the first term says whether the latent code looks plausible under the latent prior
- the second term corrects for how the invertible mapping stretches or compresses volume

When the model is trained, it maximizes this conditional likelihood, or equivalently minimizes the negative log-likelihood:

$$
\mathcal{L}_{NLL} = -\log p(x_{ref} \mid x_l)
$$

### 6.4 Layer-wise view

If the invertible network is a sequence of invertible layers, the Jacobian term becomes a sum:

$$
\log p(x_{ref} \mid x_l)
= \log p_Z(z \mid x_l)
+ \sum_{n=1}^{N} \log \left| \det J_n \right|
$$

where $J_n$ is the Jacobian of the $n$-th invertible layer.

So the model is not just learning to reconstruct. It is learning a full density transformation.

### 6.5 The LLFlow-specific prior

This is the most distinctive mathematical idea in the paper.

A standard flow often assumes something like:

$$
z \sim \mathcal{N}(0, I)
$$

LLFlow does **not** want a generic zero-mean latent prior. Instead, it wants a prior whose mean reflects scene color structure.

The intuition is:

- global illumination changes
- underlying scene color should be more stable
- a color-map-like latent mean can guide the model toward more realistic color restoration

The paper introduces a prior centered around a color map. Conceptually:

$$
z \sim \mathcal{N}(\mu, I)
$$

where $\mu$ is chosen from either:

- the reference color map $C(x_{ref})$
- the predicted color map $g(x_l)$

using a random selector during training.

So the idea is:

$$
\mu = r(C(x_{ref}), g(x_l))
$$

where $r(\cdot)$ randomly chooses one of the two.

This helps bridge training and inference:

- during training, the model can benefit from reference-derived color structure
- during inference, only $g(x_l)$ is available

### 6.6 Why this is better than saying "just use a flow"

The paper is not merely saying:

> Use any invertible network and hope for the best.

It is saying:

> Use a conditional invertible network whose latent prior is guided by an illumination-invariant color representation.

That extra prior structure is one of the main reasons LLFlow is specifically designed for low-light enhancement rather than being just a generic conditional flow.

---

## 7. Color Map and Noise Map

These two ideas are easy to overlook, but they are central.

### 7.1 Color map

The paper defines a Retinex-inspired color map:

$$
C(x) = \frac{x}{\mathrm{mean}_c(x)}
$$

where $\mathrm{mean}_c(x)$ is the per-pixel mean over channels.

Interpretation:

- dividing by channel mean removes some illumination magnitude information
- what remains is closer to chromatic / reflectance-like information
- this is more stable across lighting changes than raw RGB intensities

Easy interpretation:

> The color map tries to keep "what color is this surface?" while reducing "how bright is it right now?"

### 7.2 Code-level implementation detail

In the paper, the color map is described using channel mean.  
In the code, it is implemented as a normalized channel ratio using the channel sum.

That difference is not conceptually important here. It is basically a chromatic normalization step.

### 7.3 Noise map

The paper defines a noise map from gradients of the color map:

$$
N(x) = \max\left( |\partial_x C(x)|, |\partial_y C(x)| \right)
$$

Interpretation:

- strong high-frequency disturbance in the color map often signals noise
- this map is used as an auxiliary cue to guide the encoder

Easy interpretation:

> The model builds a map of "where the color estimate looks unstable or noisy" and feeds that back into the conditioning branch.

---

## 8. Architecture Overview

LLFlow has **two major components**.

### 8.1 Component A: Conditional encoder

The encoder takes the low-light input and estimates:

- an illumination-invariant color map
- multi-scale conditioning features for the flow

In the paper and code, the encoder is RRDB-based.

Its inputs are built from several sources:

- low-light image $x_l$
- histogram-equalized image $h(x_l)$
- color map $C(x_l)$
- noise map $N(x_l)$

So the encoder is not a plain CNN on raw RGB only.

It is a feature extractor designed to give the flow branch both:

- local detail cues
- global illumination/color cues

### 8.2 Component B: Invertible network

The second component is the flow itself.

Its job is to learn an invertible transformation between:

- image space of normally exposed images
- latent space

conditioned on the encoder features.

The paper describes a 3-level invertible network with squeeze layers and repeated flow steps.

Each flow step is Glow-like in spirit:

- ActNorm
- invertible $1 \times 1$ convolution
- affine coupling transform

with conditioning injected from the encoder features.

### 8.3 The best conceptual picture

The encoder answers:

> What stable scene/color information can I infer from the dark image?

The flow answers:

> Given that information, what distribution of valid normally exposed images should this image belong to?

That is the architecture in one sentence.

---

## 9. Architecture in the Released Repo

This section is the bridge between paper and code.

### 9.1 Main files

- `code/models/modules/LLFlow_arch.py`: top-level LLFlow module
- `code/models/modules/ConditionEncoder.py`: conditional encoder
- `code/models/modules/FlowUpsamplerNet.py`: multi-level flow architecture
- `code/models/modules/FlowStep.py`: individual flow step
- `code/models/LLFlow_model.py`: training/inference wrapper

### 9.2 What the encoder produces in code

The encoder returns multi-scale features such as:

- `fea_up0`
- `fea_up1`
- `fea_up2`
- `fea_up4`
- `last_lr_fea`
- `color_map`

These are used to condition the flow at different resolutions.

### 9.3 What the flow looks like in code

The provided configs use:

- `L: 3` flow levels
- `K: 12` flow steps per level in the standard config
- `K: 4` in the small config
- `split.enable: false`

So the released implementation is:

- multi-level
- conditional
- invertible
- no active latent split in the provided configs

### 9.4 Small model vs. standard model

This matters for study.

The **small model** is not identical to the paper's full-capacity configuration.

Typical differences:

- fewer channels
- fewer RRDB blocks
- fewer flow steps

So when reading the repo, always separate:

- the **paper-level method idea**
- the **specific training config you are currently inspecting**

---

## 10. Training Pipeline

This is one of the most important sections for future work.

## 10.1 High-level training flow

For each training pair:

1. Load paired low/high images.
2. Apply low-light preprocessing and augmentation.
3. Feed low-light image into the encoder to get conditioning features and color-map guidance.
4. Feed the reference high-light image through the forward flow under that condition.
5. Compute exact negative log-likelihood.
6. Backpropagate and update parameters.

So training is not:

$$
x_l \to \hat{x} \to \text{compare with } x_{ref}
$$

It is more like:

$$
(x_{ref}, x_l) \to z \to \text{evaluate how plausible } z \text{ is under the conditional prior}
$$

That difference is fundamental.

### 10.2 Dataset setup

The repo uses paired datasets.

For LOL:

- train on `our485`
- validate/test on `eval15`

The dataloader returns:

- `LQ`: low-light tensor
- `GT`: normal-light tensor

### 10.3 Input preprocessing in the repo

This is one of the most useful concrete details.

The low-light image may go through:

- random crop
- random horizontal flip
- random rotation
- optional synthetic noise injection
- log transform of the low-light tensor
- optional concatenation with histogram-equalized input

Easy interpretation:

> The model is not fed raw low-light RGB only. It is fed a deliberately shaped conditioning input that emphasizes illumination structure and robustness.

### 10.4 Why the log transform matters

The repo often uses:

$$
x_l \leftarrow \log(\mathrm{clamp}(x_l + 10^{-3}))
$$

This compresses dynamic range and can make the low-light signal easier to model numerically.

Then inside the encoder, the code exponentiates the first 3 channels back when building the raw color map estimate.

That is a very implementation-specific detail worth remembering.

### 10.5 What exactly gets optimized

In standard configs, the important terms are:

- `weight_fl = 1`
- `weight_l1 = 0`

So the effective objective is:

$$
\mathcal{L} \approx \mathcal{L}_{NLL}
$$

with optional alternatives available in code.

If L1 is enabled, the code can also:

1. run reverse flow to generate an image
2. compare it to the reference
3. add an $L_1$ reconstruction term

But that is optional, not the standard released recipe.

### 10.6 How the forward pass works during training

Training uses the **forward flow direction**:

$$
z = f_\theta(x_{ref}; x_l)
$$

The model computes:

- the latent code $z$
- the log-determinant contributions from the invertible layers
- the log-probability under the latent prior

Then it forms NLL and backpropagates.

### 10.7 How the reverse pass works during inference

Inference uses the **inverse flow direction**:

$$
\hat{x} = f_\theta^{-1}(z; x_l)
$$

In theory, one can sample different $z$ values around the latent mean and get different valid outputs.

In practice, the released code path behaves much closer to:

- estimate encoder color map from the low-light image
- use that as the latent guidance
- run reverse flow once
- output a deterministic enhanced image

So the paper is **probabilistic in principle**, but the released implementation is **closer to deterministic in everyday use**.

### 10.8 Learning rate schedule

The configs use Adam and a multi-step learning rate decay schedule.

Typical paper setting on LOL:

- patch size: $160 \times 160$
- batch size: 16
- learning rate: $5 \times 10^{-4}$

The repo expresses schedule milestones as relative fractions of total iterations in config.

### 10.9 Validation behavior

The repo periodically:

- runs inference on validation data
- saves validation images
- computes PSNR and SSIM

Important nuance:

the code adjusts global brightness before PSNR evaluation, following the evaluation style used by KinD. So reported PSNR is not a naive raw output PSNR.

That matters if you later compare reproduction numbers.

### 10.10 Checkpointing and experiment structure

The training script creates an experiment directory with:

- logs
- checkpoints
- training state
- validation images
- optional TensorBoard logs

This is standard, but it is useful when you start tracing experiments.

---

## 11. What Makes LLFlow Different From a Standard CNN Enhancer

### Standard CNN view

Input low-light image, run a feedforward network, output one enhanced image.

Training says:

> Be close to this target image.

### LLFlow view

Input low-light image, extract condition features, map the reference image into latent space with an invertible model, and train the latent/image transformation to have high conditional likelihood.

Training says:

> Under the condition provided by the low-light image, the reference image should map to a simple latent code that looks probable under the correct scene-aware prior.

This is much closer to **distribution matching** than plain pixel regression.

---

## 12. What the Paper Claims This Buys You

The claimed benefits are:

- better handling of one-to-many ambiguity
- stronger constraint on natural image structure than plain pixel loss
- better brightness recovery
- less artifact amplification
- richer colors
- better theoretical ability to represent multiple plausible outputs

The color-map prior is especially important for the paper's claim about improved saturation and color consistency.

---

## 13. What to Remember About the Released Implementation

If you are studying the repo for future work, keep these code-grounded facts in mind.

### 13.1 Standard released configs are mostly NLL-only

Do not describe this repo as primarily trained with perceptual loss or VGG loss.

### 13.2 The model is same-resolution enhancement

Even though some internal names come from SR / Glow-style codebases, the config uses:

- `scale: 1`

So this is low-light enhancement at the same spatial resolution.

### 13.3 The flow is condition-heavy

The conditioning information is not optional decoration. It is central to the method.

The encoder features are used throughout the flow to guide affine coupling behavior.

### 13.4 The small config and full config differ materially

If you inspect `LOL_smallNet.yml` first, do not mistake it for the paper's full-capacity setting.

### 13.5 Inference is simpler than the full probabilistic story

The paper sells the ability to sample different valid outputs. The released repo, in typical use, is closer to a deterministic enhancement path centered on the encoder-derived color map.

---

## 14. A Good Mental Model for the Whole Paper

If you only remember one mental model, use this:

1. The low-light image is ambiguous.
2. So one fixed regression target is not a satisfying formulation.
3. LLFlow uses a conditional flow to model the space of valid normal-light outputs.
4. The encoder extracts stable color/scene cues that define what the latent space should look like.
5. The flow learns an invertible mapping so that good normal-light images become simple latent codes under that condition.

In one sentence:

> LLFlow learns not just how to brighten an image, but how a valid normally exposed image distribution should look when conditioned on the dark input.

---

## 15. Short Answer Version

If you need a fast explanation later, this is the one to use.

**What is LLFlow?**  
A conditional normalizing-flow model for low-light enhancement.

**What is the core idea?**  
Map a reference normal-light image into a simple latent distribution conditioned on the low-light input, instead of directly regressing pixels with $L_1$.

**What is the special trick?**  
Use an illumination-invariant color map as latent prior guidance.

**What are the two main parts?**  
An RRDB-based conditional encoder and a Glow-style invertible flow network.

**How is it trained?**  
Mostly by exact negative log-likelihood in the released configs.

**Why is it interesting?**  
Because low-light enhancement is one-to-many, and LLFlow is designed to model that uncertainty more naturally than a deterministic CNN.

---

## 16. Final Takeaways

- LLFlow is best understood as **conditional density modeling for low-light enhancement**.
- The paper's key mathematical move is **change-of-variables likelihood training**.
- The paper's key domain-specific move is the **illumination-invariant color-map prior**.
- The repo's key practical fact is that the released configs are **mostly NLL-only, no perceptual loss, no active latent split**.
- The training pipeline is centered on **paired data + low-light preprocessing + encoder conditioning + forward-flow NLL optimization**.

If I had to compress the whole method into one line:

> LLFlow uses an encoder to estimate stable scene/color cues from a dark image, then trains an invertible conditional model so real normally exposed images become likely under a simple latent distribution conditioned on those cues.