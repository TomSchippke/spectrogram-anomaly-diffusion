"""
Training script for a DDPM noise model.

Features
--------
- Train from a YAML configuration file
- Automatic create a run directory
- Log metrics to a JSON file
- Sanity check (overfit one batch)
- CPU / GPU agnostic
- Periodic checkpoints (with the hyperparameters needed to reload them)
- Optional EMA of the weights, gradient clipping, seeding

Usage
-----
python train.py --config configs/test.yaml

Optional overrides
------------------
python train.py --config configs/config.yaml --batch_size 8 --sanity
"""

import os
import copy
import argparse
import random
import shutil
import json
from typing import Dict, Any, List

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
from dataset import NoisePatchDataset
from network import UNetDiffusion
from diffusion import DiffusionSchedule, ddpm_loss

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_device(device_str: str | None = None) -> torch.device:
    """
    Select computation device.

    Parameters
    ----------
    device_str : str or None
        User-specified device string (e.g. "cpu", "cuda", "cuda:1").
        If None, automatically selects CUDA if available.

    Returns
    -------
    torch.device
    """
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_str)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    return device


def load_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy and PyTorch RNGs for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_dir(config_path: str, runs_root: str = "runs") -> str:
    """
    Create a unique run directory based on the configuration filename.

    Example
    -------
    configs/test.yaml, runs_root="runs" -> runs/test/run_000/
    configs/t_sweep/t_500.yaml, runs_root="runs/t_sweep"
        -> runs/t_sweep/t_500/run_000/

    The configuration file is copied into the run directory to ensure
    reproducibility.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    runs_root : str
        Root directory of the runs (config key system.checkpoint_dir).

    Returns
    -------
    str
        Path to the created run directory.
    """
    config_name = os.path.splitext(config_path)[0]
    base_dir = os.path.join(runs_root, os.path.basename(config_name))
    os.makedirs(base_dir, exist_ok=True)

    existing = [
        d for d in os.listdir(base_dir) if d.startswith("run_")
    ]
    indices = []
    for d in existing:
        try:
            indices.append(int(d.split("_")[1]))
        except Exception:
            pass

    next_idx = max(indices) + 1 if indices else 0
    run_dir = os.path.join(base_dir, f"run_{next_idx:03d}")

    os.makedirs(run_dir)
    os.makedirs(os.path.join(run_dir, "checkpoints"))

    shutil.copy(config_path, os.path.join(run_dir, "config.yaml"))

    print(f"[INFO] Created run directory: {run_dir}")
    return run_dir


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: str,
    name: str,
    hparams: Dict[str, Any],
    ema_model: torch.nn.Module | None = None,
) -> None:
    """
    Save a training checkpoint.

    The checkpoint includes:
    - Model state dictionary (the EMA weights if EMA is enabled; the raw
      weights are then stored under "raw_model_state_dict")
    - Optimizer state dictionary
    - Number of optimizer steps performed
    - Hyperparameters needed to rebuild the model and diffusion schedule

    Parameters
    ----------
    model : torch.nn.Module
        Neural network to save.
    optimizer : torch.optim.Optimizer
        Optimizer associated with the model.
    step : int
        Number of optimizer steps performed.
    checkpoint_dir : str
        Directory where checkpoints are stored.
    name : str
        Filename of the checkpoint.
    hparams : Dict[str, Any]
        Model and diffusion hyperparameters.
    ema_model : torch.nn.Module, optional
        EMA copy of the model.
    """
    path = os.path.join(checkpoint_dir, name)
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hparams": hparams,
    }
    if ema_model is not None:
        ckpt["raw_model_state_dict"] = ckpt["model_state_dict"]
        ckpt["model_state_dict"] = ema_model.state_dict()
    torch.save(ckpt, path)
    print(f"[INFO] Saved checkpoint: {path}")


def save_metrics(path: str, metrics: Dict[str, Any]) -> None:
    """
    Save training metrics to a JSON file.

    Parameters
    ----------
    path : str
        Path to the metrics JSON file.
    metrics : Dict[str, Any]
        Dictionary containing training metrics.
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def check_dataset_size(dataset: NoisePatchDataset, batch_size: int) -> None:
    """
    Ensure the dataloader (drop_last=True) yields at least one batch.
    """
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset has {len(dataset)} patches, fewer than batch_size="
            f"{batch_size} (drop_last=True would yield no batch)"
        )


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, decay: float
) -> None:
    """
    Update the exponential moving average of the model weights.
    """
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.lerp_(p, 1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.copy_(b)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(
    device: torch.device,
    patch_dir: str,
    batch_size: int,
    num_steps: int,
    lr: float,
    num_timesteps: int,
    save_every: int,
    run_dir: str,
    base_channels: int,
    sanity: bool = False,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    scale_betas: bool = False,
    ema_decay: float | None = None,
    grad_clip: float | None = None,
    seed: int | None = None,
    data_mean: float | None = None,
    data_std: float | None = None,
) -> None:
    """
    Train a DDPM model on noise-only patches.

    Logs:
    - mean training loss per epoch
    - loss of the last step
    - hyperparameters
    """
    if num_steps <= 0 or save_every <= 0:
        raise ValueError("num_steps and save_every must be positive")
    if seed is not None:
        set_seed(seed)

    print(f"[INFO] Using device: {device}")

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    metrics_path = os.path.join(run_dir, "metrics.json")

    # Dataset
    dataset = NoisePatchDataset(patch_dir, data_mean, data_std)
    check_dataset_size(dataset, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0 if device.type == "cpu" else 4,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = UNetDiffusion(
        in_channels=1,
        base_channels=base_channels,
    ).to(device)

    ema_model = None
    if ema_decay is not None:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    schedule = DiffusionSchedule(
        timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        scale_betas=scale_betas,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    hparams = {
        "base_channels": base_channels,
        "num_timesteps": num_timesteps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "scale_betas": scale_betas,
        # Normalization used for training (None = per-patch z-score);
        # test patches must be normalized the same way.
        "data_mean": data_mean,
        "data_std": data_std,
    }

    def optimize(x0: torch.Tensor) -> float:
        loss = ddpm_loss(model, x0, schedule)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema_model is not None:
            update_ema(ema_model, model, ema_decay)
        return loss.item()

    metrics = {
        "training": {
            "loss_per_epoch": [],
            "last_step_loss": None,
            "num_steps": num_steps,
        },
        "hyperparameters": {
            **hparams,
            "learning_rate": lr,
            "batch_size": batch_size,
            "ema_decay": ema_decay,
            "grad_clip": grad_clip,
            "seed": seed,
        },
        "system": {
            "device": str(device),
        },
    }

    model.train()
    step = 0

    # -------------------------------------------------------------------------
    # Sanity check
    # -------------------------------------------------------------------------
    if sanity:
        x0 = next(iter(dataloader)).to(device)
        print(f"[SANITY] Overfitting on one batch of size {x0.size(0)}")

        losses = []
        for step in range(num_steps):
            loss = optimize(x0)
            losses.append(loss)

            if step % 100 == 0:
                print(f"[SANITY Step {step:04d}] Loss = {loss:.6f}")

        metrics["training"]["loss_per_epoch"].append(
            float(sum(losses) / len(losses))
        )
        metrics["training"]["last_step_loss"] = losses[-1]

        save_checkpoint(
            model, optimizer, num_steps, checkpoint_dir, "sanity_final.pt",
            hparams, ema_model,
        )
        save_metrics(metrics_path, metrics)
        print("[SANITY] Completed.")
        return

    # -------------------------------------------------------------------------
    # Full training
    # -------------------------------------------------------------------------
    epoch = 0
    while step < num_steps:
        epoch_losses: List[float] = []

        for x0 in dataloader:
            if step >= num_steps:
                break

            x0 = x0.to(device)
            loss = optimize(x0)
            epoch_losses.append(loss)

            if step % 100 == 0:
                print(f"[Step {step:06d}] Loss = {loss:.6f}")

            # `step` now counts the optimizer steps performed
            step += 1

            if step % save_every == 0 and step < num_steps:
                save_checkpoint(
                    model,
                    optimizer,
                    step,
                    checkpoint_dir,
                    f"model_step_{step}.pt",
                    hparams,
                    ema_model,
                )

        mean_epoch_loss = float(sum(epoch_losses) / len(epoch_losses))
        metrics["training"]["loss_per_epoch"].append(mean_epoch_loss)
        print(f"[Epoch {epoch:03d}] Mean loss = {mean_epoch_loss:.6f}")
        epoch += 1

        save_metrics(metrics_path, metrics)

    metrics["training"]["last_step_loss"] = epoch_losses[-1]

    save_checkpoint(
        model, optimizer, step, checkpoint_dir, "model_final.pt",
        hparams, ema_model,
    )
    save_metrics(metrics_path, metrics)
    print("[INFO] Training completed.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Parse arguments, load configuration, create run, launch training.
    """
    parser = argparse.ArgumentParser(description="Train DDPM noise model")
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--device", type=str)
    parser.add_argument("--patch_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_timesteps", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--unet_capacity", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sanity", action="store_true")

    args = parser.parse_args()
    cfg = load_config(args.config)

    def pick(arg_value, section: str, key: str, default=None):
        """CLI value if given, else config value, else default."""
        if arg_value is not None:
            return arg_value
        value = cfg.get(section, {}).get(key)
        if value is None:
            if default is None:
                raise KeyError(f"Missing config entry {section}.{key}")
            return default
        return value

    # Parse and validate everything before creating the run directory
    patch_dir = pick(args.patch_dir, "dataset", "patch_dir")
    batch_size = int(pick(args.batch_size, "dataset", "batch_size"))
    num_steps = int(pick(args.num_steps, "training", "num_steps"))
    lr = float(pick(args.lr, "training", "lr"))
    save_every = int(pick(args.save_every, "training", "save_every"))
    num_timesteps = int(
        pick(args.num_timesteps, "diffusion", "num_timesteps")
    )
    base_channels = int(pick(args.unet_capacity, "unet", "capacity", 64))

    diffusion_cfg = cfg.get("diffusion", {})
    beta_start = float(diffusion_cfg.get("beta_start", 1e-4))
    beta_end = float(diffusion_cfg.get("beta_end", 2e-2))
    scale_betas = bool(diffusion_cfg.get("scale_betas", False))

    training_cfg = cfg.get("training", {})
    ema_decay = training_cfg.get("ema_decay")
    ema_decay = float(ema_decay) if ema_decay is not None else None
    grad_clip = training_cfg.get("grad_clip")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    seed = args.seed if args.seed is not None else training_cfg.get("seed")

    sanity = bool(args.sanity or training_cfg.get("sanity", False))
    device = get_device(
        args.device or cfg.get("system", {}).get("device")
    )
    runs_root = cfg.get("system", {}).get("checkpoint_dir", "runs")
    dataset_cfg = cfg.get("dataset", {})
    data_mean = dataset_cfg.get("mean")
    data_mean = float(data_mean) if data_mean is not None else None
    data_std = dataset_cfg.get("std")
    data_std = float(data_std) if data_std is not None else None

    check_dataset_size(
        NoisePatchDataset(patch_dir, data_mean, data_std), batch_size
    )

    run_dir = create_run_dir(args.config, runs_root)

    train(
        device=device,
        patch_dir=patch_dir,
        batch_size=batch_size,
        num_steps=num_steps,
        lr=lr,
        num_timesteps=num_timesteps,
        save_every=save_every,
        run_dir=run_dir,
        base_channels=base_channels,
        sanity=sanity,
        beta_start=beta_start,
        beta_end=beta_end,
        scale_betas=scale_betas,
        ema_decay=ema_decay,
        grad_clip=grad_clip,
        seed=seed,
        data_mean=data_mean,
        data_std=data_std,
    )


if __name__ == "__main__":
    main()

