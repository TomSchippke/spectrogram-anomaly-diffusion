from typing import List, Optional, Tuple

import torch
from torch import Tensor

from network import UNetDiffusion
from diffusion import DiffusionSchedule

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def load_diffusion_model(
    ckpt_path: str,
    device: str | torch.device = "cpu",
    **overrides,
) -> Tuple[UNetDiffusion, DiffusionSchedule]:
    """
    Load a trained diffusion model and the schedule it was trained with.

    The hyperparameters are read from the "hparams" entry saved by
    train.py. Checkpoints saved before this entry existed must provide
    them through `overrides` (e.g. base_channels=64, num_timesteps=500).

    Parameters
    ----------
    ckpt_path : str
        Path to the checkpoint file.
    device : str or torch.device
        Device on which to load the model.
    **overrides
        Hyperparameters overriding (or completing) those in the checkpoint:
        base_channels, num_timesteps, beta_start, beta_end, scale_betas.

    Returns
    -------
    model : UNetDiffusion
        Model in eval mode.
    schedule : DiffusionSchedule
        Diffusion schedule matching training.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    hparams = {**ckpt.get("hparams", {}), **overrides}

    missing = [
        k for k in ("base_channels", "num_timesteps") if k not in hparams
    ]
    if missing:
        raise KeyError(
            f"Checkpoint {ckpt_path} has no {missing}; pass them as "
            f"keyword arguments"
        )

    model = UNetDiffusion(in_channels=1, base_channels=hparams["base_channels"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    schedule = DiffusionSchedule(
        timesteps=hparams["num_timesteps"],
        beta_start=hparams.get("beta_start", 1e-4),
        beta_end=hparams.get("beta_end", 2e-2),
        scale_betas=hparams.get("scale_betas", False),
        device=torch.device(device),
    )
    return model, schedule


# -----------------------------------------------------------------------------
# DDIM timestep subsampling
# -----------------------------------------------------------------------------


def make_ddim_timesteps(
    num_ddim_steps: int,
    num_diffusion_steps: int,
    t_start: int = None
) -> List[int]:
    """
    Create a subsampled DDIM timestep schedule based on the training diffusion 
    process. It Allows starting from an arbitrary timestep 
    t_start <= num_diffusion_steps-1.

    Parameters
    ----------
    num_ddim_steps : int
        Number of reverse steps to perform (e.g. 25, 50, 100). The last
        step goes from timestep 0 to the clean image.
    num_diffusion_steps : int
        Total number of diffusion steps used during training.
    t_start : int, optional
        Starting timestep. If None, defaults to the final timestep 
        (num_diffusion_steps - 1).

    Returns
    -------
    timesteps : List[int]
        Descending list of timesteps (e.g. [t_start, ..., 0])
    """
    if t_start is None:
        t_start = num_diffusion_steps - 1
    if not 0 <= t_start < num_diffusion_steps:
        raise ValueError("t_start must be in [0, num_diffusion_steps-1]")
    if not 1 <= num_ddim_steps <= t_start + 1:
        raise ValueError("num_ddim_steps must be in [1, t_start+1]")

    if num_ddim_steps == 1:
        return [t_start]

    # Rounding (rather than truncating) float64 values spaced >= 1 apart
    # guarantees distinct timesteps.
    timesteps = torch.linspace(
        t_start,
        0,
        steps=num_ddim_steps,
        dtype=torch.float64,
    ).round().long()
    return timesteps.tolist()


# -----------------------------------------------------------------------------
# DDIM Reverse step
# -----------------------------------------------------------------------------

def _alpha_bar(schedule: DiffusionSchedule, t: int) -> Tensor:
    """
    alpha_bar at timestep t, with alpha_bar(-1) = 1 (clean image).
    """
    if t < 0:
        return torch.ones((), device=schedule.alpha_bar.device)
    return schedule.alpha_bar[t]


@torch.no_grad()
def ddim_reverse_step(
    x: Tensor,
    t: int,
    t_prev: int,
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    clip_x0: Optional[float] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform one deterministic DDIM reverse step: x_t -> x_{t_prev}.

    Parameters
    ----------
    x : Tensor
        Current noisy sample at timestep t. Shape: (B,C,H,W)
    t : int
        Current timestep.
    t_prev : int
        Target timestep (< t). Use -1 to go to the clean image.
    model : torch.nn.Module
        Noise prediction model
    schedule : DiffusionSchedule
        Precomputed alpha_bar coefficients.
    clip_x0 : float, optional
        If given, the predicted clean image is clamped to [-clip_x0, clip_x0]
        (the predicted noise is recomputed accordingly). Useful at large t,
        where dividing by sqrt(alpha_bar_t) amplifies prediction errors.

    Returns
    -------
    x_prev : Tensor
        Sample at timestep t_prev. Shape: (B,C,H,W)
    x0_hat : Tensor
        Predicted clean image. Shape: (B,C,H,W)
    score : Tensor
        Score estimate at timestep t. Shape: (B,C,H,W)
    """
    if not t_prev < t:
        raise ValueError(f"t_prev ({t_prev}) must be < t ({t})")

    t_tensor = torch.full(
        (x.size(0),), t, device=x.device, dtype=torch.long
    )
    eps = model(x, t_tensor)

    alpha_bar_t = _alpha_bar(schedule, t)
    alpha_bar_prev = _alpha_bar(schedule, t_prev)

    sqrt_ab_t = torch.sqrt(alpha_bar_t)
    sqrt_1mab_t = torch.sqrt(1.0 - alpha_bar_t)

    # Score of x_t
    score = -eps / sqrt_1mab_t

    # Predict x0 from x_t
    x0_hat = (x - sqrt_1mab_t * eps) / sqrt_ab_t
    if clip_x0 is not None:
        x0_hat = x0_hat.clamp(-clip_x0, clip_x0)
        eps = (x - sqrt_ab_t * x0_hat) / sqrt_1mab_t

    # Deterministic DDIM update
    x_prev = torch.sqrt(alpha_bar_prev) * x0_hat + \
        torch.sqrt(1.0 - alpha_bar_prev) * eps

    return x_prev, x0_hat, score


# -----------------------------------------------------------------------------
# Full reverse diffusion with subsampled timesteps
# -----------------------------------------------------------------------------

@torch.no_grad()
def reverse_diffusion_ddim(
    xt: Tensor,
    timesteps: List[int],
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    clip_x0: Optional[float] = None,
) -> Tuple[Tensor, List[Tensor]]:
    """
    Run full DDIM reverse diffusion on a subsampled timestep sequence.

    A final step from timesteps[-1] to the clean image (alpha_bar = 1) is
    always performed, so the output is the predicted x0, not a sample that
    still contains the noise of timestep 0.

    Parameters
    ----------
    xt : Tensor
        Noisy input at timestep timesteps[0]. Shape: (B,C,H,W)
    timesteps : List[int]
        Strictly descending list of timesteps.
    model : torch.nn.Module
        Noise prediction network.
    schedule : DiffusionSchedule
        Original training schedule.
    clip_x0 : float, optional
        Clamp the predicted clean image at each step (see ddim_reverse_step).

    Returns
    -------
    x0_hat : Tensor
        Reconstructed image. Shape: (B,C,H,W)
    scores : List[Tensor]
        Score tensor at each timestep of `timesteps` (same length).
        Each tensor has shape (B, C, H, W).
    """
    x = xt
    scores = []

    targets = list(timesteps[1:]) + [-1]
    for t, t_prev in zip(timesteps, targets):
        x, _, score = ddim_reverse_step(
            x, t, t_prev, model, schedule, clip_x0
        )
        scores.append(score)

    return x, scores
