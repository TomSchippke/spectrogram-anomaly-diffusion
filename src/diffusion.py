from typing import Tuple

import torch
import torch.nn as nn


class DiffusionSchedule:
    """
    Diffusion noise schedule for DDPM.

    This class precomputes all coefficients required for the forward diffusion
    process x_0 -> x_t using a fixed beta schedule.

    Parameters
    ----------
    timesteps : int
        Number of diffusion steps T.
    beta_start : float
        Initial noise variance.
    beta_end : float
        Final noise variance.
    scale_betas : bool
        If True, beta_start and beta_end are rescaled by 1000 / timesteps
        (as in Improved DDPM), so that the total amount of noise added is
        roughly independent of T. If False, the betas are used as given,
        which makes x_T far from pure noise for small T (e.g. T=200).
    device : torch.device
        Device on which tensors are allocated.
    """
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        scale_betas: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.scale_betas = scale_betas
        self.device = device

        if scale_betas:
            scale = 1000.0 / timesteps
            beta_start = beta_start * scale
            beta_end = min(beta_end * scale, 0.999)

        self.betas = torch.linspace(
            beta_start, beta_end, timesteps, device=device
        )

        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)


def diffuse(
    x0: torch.Tensor,
    t: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply forward diffusion to clean data x0.

    Parameters
    ----------
    x0 : torch.Tensor
        Clean input image of shape (B, C, H, W).
    t : torch.Tensor
        Timesteps of shape (B,), values in [0, T-1].
    schedule : DiffusionSchedule
        Precomputed diffusion coefficients.
    noise : torch.Tensor, optional
        Noise tensor of same shape as x0. If None, sampled from N(0, I).

    Returns
    -------
    xt : torch.Tensor
        Noisy image at timestep t.
    noise : torch.Tensor
        The noise used to generate xt.
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_ab = schedule.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
    sqrt_1mab = schedule.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

    xt = sqrt_ab * x0 + sqrt_1mab * noise
    return xt, noise
    

def ddpm_loss(
    model: nn.Module,
    x0: torch.Tensor,
    schedule: DiffusionSchedule
) -> torch.Tensor:
    """
    Compute the DDPM training loss.

    This loss trains the model to predict the noise ε added during
    the forward diffusion process.

    Parameters
    ----------
    model : nn.Module
        Diffusion model εθ(x_t, t).
    x0 : torch.Tensor
        Clean input images of shape (B, C, H, W).
    schedule : DiffusionSchedule
        Diffusion noise schedule.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss value.
    """
    B = x0.size(0)
    device = x0.device

    # sample random timesteps
    t = torch.randint(
        0, schedule.timesteps, (B,), device=device
    )

    # forward diffusion
    xt, noise = diffuse(x0, t, schedule)

    # predict noise
    noise_pred = model(xt, t)

    # MSE loss
    loss = torch.mean((noise - noise_pred) ** 2)
    return loss
