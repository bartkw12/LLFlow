# LLFlow Quick Training Hyperparameters

This note explains the changes made in [code/confs/LOL_smallNet_quick.yml](code/confs/LOL_smallNet_quick.yml) relative to [code/confs/LOL_smallNet.yml](code/confs/LOL_smallNet.yml).

## Goal

The quick config is meant for a short training walkthrough:

- verify the training loop runs
- see logging, validation, and checkpoint saving happen
- keep GPU memory use and runtime modest
- avoid changing the model architecture more than necessary

It is not meant to produce a strong final model.

## Dataset Path

`D:\LOLdataset` is not a required drive location. It is only the example root path used in the original config. You can point `datasets.train.root` and `datasets.val.root` to any folder you want, as long as it has the structure expected by the LoL loader:

```text
YOUR_DATASET_ROOT/
  our485/
    low/
    high/
  eval15/
    low/
    high/
```

If you want to train on the actual LOL benchmark, you need the LOL dataset downloaded locally. If you only want a smoke test of the training process, you can also create a small subset with the same folder structure.

## Changes Made

| Setting | Original | Quick run | Effect | Why reduce or change it |
| --- | --- | --- | --- | --- |
| `name` | `train_rebuttal_smallNet_ch32_blocks1` | `train_quick_smallNet_300iter` | Controls the experiment output folder name under `experiments/` | Using a fresh name prevents accidental resume from an older run and keeps the quick test isolated |
| `use_tb_logger` | `true` | `false` | Enables TensorBoard logging | Disabling it removes one extra dependency and keeps the first short run simpler |
| `datasets.train.batch_size` | `16` | `4` | Number of samples processed per step | Lowering it cuts VRAM use and makes the run more likely to fit on smaller GPUs |
| `datasets.train.GT_size` | `160` | `128` | Random crop size used for training patches | Smaller patches reduce memory and compute per step |
| `train.niter` | `45000` | `300` | Total number of training iterations | This is the main change that makes the run short enough for a walkthrough |
| `train.val_freq` | `200` | `100` | How often validation runs | More frequent validation lets you see that stage of the pipeline during a short run |
| `logger.print_freq` | `100` | `10` | How often progress is printed | A short run needs denser logging so you can actually observe the training loop |
| `logger.save_checkpoint_freq` | `1000` | `100` | How often model checkpoints and training state are saved | A short run would otherwise finish before saving much; lowering this makes checkpointing visible |

## Settings Intentionally Left Alone

These settings were kept the same on purpose:

- `network_G` structure: keeps the same small model layout so the run still exercises the intended LLFlow architecture
- `lr_G`: the learning rate was not changed because the goal is only to observe the process, not to tune optimization
- `lr_scheme`, `beta1`, `beta2`, `weight_fl`, and related optimization settings: kept unchanged so behavior stays close to the original training recipe
- `resume_state: auto`: kept to preserve the repo's normal load and resume behavior

## Practical Notes

- This repository trains by total iterations, not by a direct epoch count.
- With the LOL dataset and `batch_size: 4`, `300` iterations is only a few epochs, not hundreds of epochs.
- If your GPU still runs out of memory, lower `batch_size` from `4` to `2` first.
- If that is still not enough, reduce `GT_size` from `128` to `96`.
- If you want the fastest possible process-only run, create a tiny LOL-style subset and point both dataset roots to that subset folder.

## Recommendation

For a first walkthrough on the 3080 Ti, the quick config is a reasonable starting point. If the goal is only to watch the end-to-end flow once, an even smaller run such as `niter: 100` with a tiny dataset subset is also valid.