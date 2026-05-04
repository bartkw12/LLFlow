# LLFlow Paired Test Final Analysis

## Overview

This document summarizes the paired-image evaluation performed on LLFlow using a custom indoor bathroom image pair:

- `high` / GT: lights-on reference image
- `low` / LR: lights-off input image

Three paired test results were collected:

1. `LOL_smallNet` on the original high-resolution image pair
2. `LOL-pc` on a resized image pair
3. `LOL_smallNet` on the same resized image pair

The resized comparison is the fairest model-to-model evaluation because both models were tested on the same input resolution.

## Important Evaluation Note

The original `LOL_smallNet` result was computed on the full-resolution pair (`5712 x 4284`), but `LOL-pc` could not be evaluated at that size because it ran out of GPU memory. To make a fair comparison, both images were resized to `1280 x 960`, and both models were evaluated again on that same resized pair.

Because of this, the original full-resolution `LOL_smallNet` result should not be compared directly against the resized `LOL-pc` result.

## Metric Definitions

The paired test reports the following metrics:

- **PSNR**: Peak Signal-to-Noise Ratio. Higher is better. Measures pixel-level similarity to the ground-truth image.
- **SSIM**: Structural Similarity Index. Higher is better. Measures structural and contrast similarity to the ground-truth image.
- **LPIPS**: Learned Perceptual Image Patch Similarity. Lower is better. Measures perceptual distance between the result and the ground truth.
- **LRC PSNR**: A reconstruction-style PSNR used by this repo to compare the original low-light input and a reconstructed version derived from the model output. It is not the main quality metric for GT fidelity, but it gives a rough sense of how strongly the output differs from the input.

## Full Result Table

| Test Setup | Resolution | PSNR | SSIM | LPIPS | LRC PSNR | Notes |
|---|---:|---:|---:|---:|---:|---|
| `LOL_smallNet` | `5712 x 4284` | 13.14 | 0.72 | 0.63 | 6.79 | Original full-resolution pair; not directly comparable to `LOL-pc` |
| `LOL-pc` | `1280 x 960` | 12.78 | 0.60 | 0.63 | 6.64 | Resized pair |
| `LOL_smallNet` | `1280 x 960` | 12.80 | 0.60 | 0.63 | 6.67 | Resized pair |

## Fair Comparison: Resized Pair Only

The fairest comparison is between the two resized-pair results:

| Model | Resolution | PSNR | SSIM | LPIPS | LRC PSNR |
|---|---:|---:|---:|---:|---:|
| `LOL_smallNet` | `1280 x 960` | 12.80 | 0.60 | 0.63 | 6.67 |
| `LOL-pc` | `1280 x 960` | 12.78 | 0.60 | 0.63 | 6.64 |

## Interpretation

### 1. Fair model comparison

On the resized bathroom pair, `LOL_smallNet` and `LOL-pc` produced almost identical results.

- `LOL_smallNet` had a very small advantage in PSNR: `12.80` vs `12.78`
- SSIM was identical at `0.60`
- LPIPS was identical at `0.63`
- `LOL_smallNet` had a slightly higher LRC PSNR: `6.67` vs `6.64`

These differences are extremely small and should be treated as negligible for practical purposes. On this custom indoor pair, the larger `LOL-pc` model did not provide a meaningful improvement over the smaller `LOL_smallNet` model.

### 2. Why the earlier comparison was misleading

The earlier result made `LOL_smallNet` appear clearly better, but that comparison was not methodologically clean because:

1. `LOL_smallNet` was first evaluated on the original full-resolution pair
2. `LOL-pc` had to be evaluated on a smaller resized pair because of GPU memory limits

Since resolution affects image detail, smoothness, alignment behavior, and pixel statistics, the original-resolution `LOL_smallNet` result and the resized `LOL-pc` result cannot be used for a strict head-to-head model ranking.

### 3. What the results suggest

These paired results suggest that this custom indoor bathroom scene is challenging for LLFlow regardless of model size. Both models achieved modest fidelity:

- PSNR remained relatively low
- SSIM was moderate
- LPIPS remained relatively high

This indicates that neither model reproduced the lights-on reference especially closely. That does not mean the pipeline failed technically. A more likely explanation is that the custom paired setup differs from the LOL training distribution and may also include real-world factors such as:

1. slight scene or camera misalignment
2. exposure and white-balance differences
3. indoor lighting effects that differ from the benchmark data

## Practical Conclusion

For this custom indoor paired example:

- `LOL-pc` required resizing to fit into GPU memory
- once resolution was controlled, `LOL-pc` did not outperform `LOL_smallNet`
- both models behaved very similarly on the resized test pair

The larger model therefore did not justify its additional memory cost on this specific custom evaluation.

## Report-Ready Summary

On a custom paired indoor bathroom test, a fair comparison was performed by evaluating both `LOL_smallNet` and `LOL-pc` on the same resized image pair at `1280 x 960`. The two models produced nearly identical results: `LOL_smallNet` achieved PSNR `12.80`, SSIM `0.60`, LPIPS `0.63`, and LRC PSNR `6.67`, while `LOL-pc` achieved PSNR `12.78`, SSIM `0.60`, LPIPS `0.63`, and LRC PSNR `6.64`. This indicates that, for this indoor scene, the larger `LOL-pc` model did not provide a meaningful performance improvement over the smaller `LOL_smallNet` model. Overall, both models showed similar behavior, suggesting that this custom indoor pair is challenging for LLFlow regardless of model size.