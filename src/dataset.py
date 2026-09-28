import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NoisePatchDataset(Dataset):
    """
    PyTorch Dataset for noise-only spectrogram patches stored as `.npy` files.

    This dataset is designed for unsupervised learning (e.g. diffusion models)
    where each sample corresponds to a single time-frequency noise patch.

    Notes
    -----
    - Each `.npy` file must contain a single patch.
    - Accepted shapes are (H, W) or (1, H, W).
    - Returned tensors always have shape (1, H, W).
    - By default each patch is normalized to zero mean and unit variance
      using its own statistics. If `mean` and `std` are given, these global
      statistics are used instead, which preserves the contrast of
      anomalous patches relative to normal ones.
    """

    def __init__(
        self,
        patch_dir: str,
        mean: float | None = None,
        std: float | None = None,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        patch_dir : str
            Path to the directory containing `.npy` patch files.
        mean, std : float, optional
            Global normalization statistics. Both must be given, or neither
            (per-patch normalization).
        """
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be given together")

        self.patch_dir = patch_dir
        self.mean = mean
        self.std = std

        self.files = sorted(
            f for f in os.listdir(patch_dir) if f.endswith(".npy")
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No .npy files found in directory: {patch_dir}"
            )

    def __len__(self) -> int:
        """
        Return the total number of patches in the dataset.

        Returns
        -------
        int
            Number of available patches.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and return a single noise patch.

        Parameters
        ----------
        idx : int
            Index of the patch to load.

        Returns
        -------
        torch.Tensor
            Normalized noise patch of shape (1, H, W).
        """
        file_path = os.path.join(self.patch_dir, self.files[idx])
        patch = np.load(file_path)

        if patch.ndim == 2:
            patch = patch[None, :, :]
        elif patch.ndim != 3 or patch.shape[0] != 1:
            raise ValueError(
                f"Invalid patch shape {patch.shape} in file {file_path}"
            )

        patch = patch.astype(np.float32)
        if self.mean is None:
            patch -= patch.mean()
            patch /= (patch.std() + 1e-8)
        else:
            patch -= self.mean
            patch /= (self.std + 1e-8)

        return torch.from_numpy(patch)


class PatchWithMaskDataset(NoisePatchDataset):
    """
    Dataset returning a noise patch together with its corresponding mask.

    The mask must be stored as a `.npy` file with the same filename
    as the corresponding patch.

    Returns
    -------
    tuple
        (patch, mask, name)
    """

    def __init__(
        self,
        patch_dir: str,
        masks_dir: str,
        mean: float | None = None,
        std: float | None = None,
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        patch_dir : str
            Directory containing patch `.npy` files.
        masks_dir : str
            Directory containing mask `.npy` files with matching filenames.
        mean, std : float, optional
            Global normalization statistics (see NoisePatchDataset).
        """
        super().__init__(patch_dir, mean, std)
        self.masks_dir = masks_dir

    def __getitem__(self, idx: int):
        """
        Load and return a patch, its corresponding mask, and filename.

        Parameters
        ----------
        idx : int
            Index of the sample to load.

        Returns
        -------
        tuple
            patch : torch.Tensor
                Normalized noise patch of shape (1, H, W).
            mask : torch.Tensor
                Mask tensor of shape (1, H, W).
            name : str
                Filename of the patch/mask pair.
        """
        patch = super().__getitem__(idx)
        name = self.files[idx]

        mask_path = os.path.join(self.masks_dir, name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Mask not found for patch {name}"
            )

        mask = np.load(mask_path)

        if mask.ndim == 2:
            mask = mask[None, :, :]
        elif mask.ndim != 3:
            raise ValueError(
                f"Invalid mask shape {mask.shape} in file {mask_path}"
            )

        if mask.shape != tuple(patch.shape):
            raise ValueError(
                f"Mask shape {mask.shape} does not match patch shape "
                f"{tuple(patch.shape)} for {name}"
            )

        mask = torch.from_numpy(mask.astype(np.float32))

        return patch, mask, name

