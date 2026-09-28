# Data challenge: line detection in spectrograms with diffusion models

The goal of this challenge is to detect line-like structures (emissions) in
spectrogram patches dominated by a correlated background noise, without
training on labeled examples. A diffusion model trained on noise-only
patches is used to "purify" a patch; the residual between the patch and its
reconstruction highlights the structures the model does not consider normal.

Two trained diffusion models are provided. **The data challenge deals with 
the last stage of the pipeline**: turning the residual map into a decision 
(`horizontal`, `other`). The method, the notations, the test database and the task
are described in the note `diffusion_anomaly.pdf`.

---

## Project structure

```
data-challenge/
├── diffusion_anomaly.pdf    # Description of the method and of the task
├── models/
│   ├── unet32.pt                # Trained diffusion model (32 base channels, fast)
│   └── unet64.pt                # Trained diffusion model (64 base channels)
├── dataset/
│   ├── test/                    # 2863 test patches (48 x 48, .npy)
│   ├── annotations.json         # Labels of 502 test patches
│   └── view_annotations.py      # Display the annotated patches and their labels
├── src/
│   ├── network.py               # U-Net noise-prediction network
│   ├── diffusion.py             # Noise schedule and forward diffusion
│   ├── inference.py             # Model loading and DDIM reverse process
│   ├── dataset.py               # PyTorch datasets of patches
│   └── train.py                 # Training script (not needed for the challenge)
└── scripts/
    └── reconstruct_test_patch.py  # Example: purify one patch and plot the error
```

---

## Requirements

Python >= 3.10 with `numpy`, `torch` and `matplotlib`. A GPU is not required.

---

## Quick start

Purify a test patch and display the original, its reconstruction and the
absolute error. The parameters (model, patch, `T_START`, number of reverse
steps, seed...) are set in the "Parameters" section at the top of the script:

```bash
python scripts/reconstruct_test_patch.py
```

Display the annotated test patches with their labels, and print the number of
annotated patches for each combination of labels. The label filter and the
grid size are set at the top of the script:

```bash
python dataset/view_annotations.py
```

---

## Important remarks

- **Normalization.** The patches of `dataset/test/` are not normalized: each
  patch must be standardized to zero mean and unit variance before being fed
  to a model (see `load_patch` in the scripts).
