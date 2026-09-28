"""
Reconstruct a test patch with a trained diffusion model and display the error.

Steps:
1. Load a trained model (checkpoint with its hyperparameters)
2. Load a patch from dataset/test/ and normalize it as during training
3. Diffuse it to timestep t_start, then run the DDIM reverse process
4. Display the original patch, its reconstruction and the absolute error

Usage: set the parameters in the "Parameters" section below, then run
    python scripts/reconstruct_test_patch.py
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusion import diffuse
from inference import (
    load_diffusion_model,
    make_ddim_timesteps,
    reverse_diffusion_ddim,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def resolve(path: str) -> str:
    """
    Resolve a path relative to the project root (if not absolute).
    """
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_patch(path: str, mean: float | None, std: float | None) -> np.ndarray:
    """
    Load a (H, W) or (1, H, W) patch and normalize it as NoisePatchDataset.
    """
    patch = np.load(path).astype(np.float32).squeeze()
    if patch.ndim != 2:
        raise ValueError(f"Invalid patch shape {patch.shape} in file {path}")

    if mean is None:
        return (patch - patch.mean()) / (patch.std() + 1e-8)
    return (patch - mean) / (std + 1e-8)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
@torch.no_grad()
def main(
    model_path: str,
    image_path: str,
    t_start: int | None,
    ddim_steps: int,
    device: str,
    seed: int,
    save: str | None,
) -> None:
    """
    Run diffusion reconstruction on a single test patch and visualize errors.
    """
    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    model, schedule = load_diffusion_model(model_path, device)
    hparams = torch.load(model_path, map_location="cpu")["hparams"]
    T = schedule.timesteps
    print(f"[INFO] Loaded {model_path} (T={T}, "
          f"base_channels={hparams['base_channels']})")

    # -------------------------------------------------------------------------
    # Load and normalize image
    # -------------------------------------------------------------------------
    image = load_patch(image_path, hparams["data_mean"], hparams["data_std"])
    x0 = torch.from_numpy(image)[None, None].to(device)  # (1,1,H,W)
    print(f"[INFO] Image: {image_path} {tuple(image.shape)}")

    # -------------------------------------------------------------------------
    # Diffusion + DDIM reconstruction
    # -------------------------------------------------------------------------
    if t_start is None:
        t_start = T // 4
    ddim_steps = min(ddim_steps, t_start + 1)

    torch.manual_seed(seed)
    t = torch.tensor([t_start], device=device)
    xt, _ = diffuse(x0, t, schedule)

    timesteps = make_ddim_timesteps(ddim_steps, T, t_start)
    x0_hat, _ = reverse_diffusion_ddim(xt, timesteps, model, schedule)

    # -------------------------------------------------------------------------
    # Errors
    # -------------------------------------------------------------------------
    x0_np = x0.squeeze().cpu().numpy()
    x0_hat_np = x0_hat.squeeze().cpu().numpy()
    abs_error = np.abs(x0_np - x0_hat_np)
    print(f"[INFO] t_start={t_start}, DDIM steps={ddim_steps}, "
          f"MSE={np.mean(abs_error ** 2):.4f}, "
          f"max |error|={abs_error.max():.4f}")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vmin, vmax = x0_np.min(), x0_np.max()

    im = axes[0].imshow(x0_np, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original (normalized)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    im = axes[1].imshow(x0_hat_np, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Reconstruction (t_start={t_start})")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    im = axes[2].imshow(abs_error, origin="lower", cmap="magma")
    axes[2].set_title("Absolute error")
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(os.path.basename(image_path))
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    if save is not None:
        fig.savefig(save, dpi=150)
        print(f"[INFO] Saved figure: {save}")
    plt.show()


# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    MODEL = "models/unet64.pt"
    TEST_DIR = "dataset/test"
    #IMAGE = "patch_9099_13.npy"
    #IMAGE = "patch_9107_22.npy" # Horizontal line
    IMAGE = "patch_9150_13.npy" # Other
    
    # Code timestep up to which the patch is noised
    T_START = 600

    # Number of DDIM reverse steps
    DDIM_STEPS = 6

    # Random seed of the noise added to the patch 
    SEED = 0

    # Path where the figure is saved or None
    SAVE = None

    # Computation device: "cpu" or "cuda"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    test_dir = resolve(TEST_DIR)
    if IMAGE is None:
        files = sorted(f for f in os.listdir(test_dir) if f.endswith(".npy"))
        image_path = os.path.join(
            test_dir, np.random.default_rng(SEED).choice(files)
        )
    else:
        image_path = os.path.join(test_dir, IMAGE)

    main(
        model_path=resolve(MODEL),
        image_path=image_path,
        t_start=T_START,
        ddim_steps=DDIM_STEPS,
        device=DEVICE,
        seed=SEED,
        save=SAVE,
    )
