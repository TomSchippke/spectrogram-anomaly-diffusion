import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal embedding of diffusion timesteps.

    This module maps a scalar timestep t to a fixed-dimensional embedding
    using sine and cosine functions with exponentially increasing frequencies.
    This embedding is standard in diffusion models (DDPM).

    Parameters
    ----------
    dim : int
        Dimension of the output embedding (must be even).
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 4 or dim % 2 != 0:
            raise ValueError(
                f"Time embedding dimension must be even and >= 4, got {dim}"
            )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal timestep embeddings.

        Parameters
        ----------
        t : torch.Tensor
            Tensor of shape (B,) containing diffusion timesteps.

        Returns
        -------
        emb : torch.Tensor
            Tensor of shape (B, dim) containing timestep embeddings.
        """
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)

        frequencies = torch.exp(
            torch.arange(half_dim, device=t.device) * -scale
        )

        args = t[:, None] * frequencies[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        return emb


def _check_spatial_size(x: torch.Tensor, encoders_numbers: int) -> None:
    """
    Check that H and W survive `encoders_numbers` 2x poolings and
    upsamplings without size mismatch.
    """
    factor = 2 ** encoders_numbers
    H, W = x.shape[-2:]
    if H % factor != 0 or W % factor != 0:
        raise ValueError(
            f"Input spatial size ({H}, {W}) must be divisible by {factor} "
            f"(2 ** encoders_numbers)"
        )


# ----------------- Modules for U-Net (diffusion) ------------------------------
class ConvBlock(nn.Module):
    """
    Convolutional block with timestep conditioning.

    This block consists of two 3×3 convolutions, each followed by GroupNorm
    and SiLU activation. A timestep embedding is projected and added after
    the first normalization, enabling conditioning on the diffusion step.

    Parameters
    ----------
    in_c : int
        Number of input channels.
    out_c : int
        Number of output channels.
    time_dim : int
        Dimension of the timestep embedding.
    """

    def __init__(self, in_c: int, out_c: int, time_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)

        self.time_proj = nn.Linear(time_dim, out_c)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, C, H, W).
        t_emb : torch.Tensor
            Timestep embedding of shape (B, time_dim).

        Returns
        -------
        out : torch.Tensor
            Output feature map of shape (B, out_c, H, W).
        """
        h = self.conv1(x)
        h = self.norm1(h)

        # Inject timestep information
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h


class EncoderBlock(nn.Module):
    """
    Encoder block of the U-Net architecture.

    Each encoder block applies a convolutional block followed by spatial
    downsampling via max pooling.

    Parameters
    ----------
    in_c : int
        Number of input channels.
    out_c : int
        Number of output channels.
    time_dim : int
        Dimension of the timestep embedding.
    """
    def __init__(self, in_c: int, out_c: int, time_dim: int):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c, time_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        Forward pass of the encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        t_emb : torch.Tensor
            Timestep embedding of shape (B, time_dim).

        Returns
        -------
        skip : torch.Tensor
            Feature map for skip connection.
        down : torch.Tensor
            Downsampled feature map.
        """
        skip = self.conv(x, t_emb)
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """
    Decoder block of the U-Net architecture.

    The decoder upsamples the input feature map, optionally concatenates
    a skip connection from the encoder, and applies a convolutional block.

    Parameters
    ----------
    in_c : int
        Number of input channels.
    out_c : int
        Number of output channels.
    time_dim : int
        Dimension of the timestep embedding.
    use_skip : bool, optional
        Whether to use skip connections (default: True).
    """

    def __init__(self, 
        in_c: int, 
        out_c: int, 
        time_dim: int, 
        use_skip: bool = True
    ) -> None:
    
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv = ConvBlock(
            in_c + out_c if use_skip else in_c,
            out_c,
            time_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the decoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, C, H, W).
        skip : torch.Tensor
            Skip connection tensor from encoder (same spatial size as 
            upsampled x).
        t_emb : torch.Tensor
            Timestep embedding of shape (B, time_dim).

        Returns
        -------
        out : torch.Tensor
            Output feature map after upsampling and convolution.
        """
        x = self.up(x)
        if self.use_skip:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x, t_emb)

        
class UNetDiffusion(nn.Module):
    """
    U-Net architecture for diffusion / score-based models.

    This network predicts the noise component added to an input image
    at a given diffusion timestep t.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (default: 1).
    base_channels : int, optional
        Number of channels after the first encoder (default: 64).
    time_dim : int, optional
        Dimension of the timestep embedding (default: 256).
    encoders_numbers : int, optional
        Number of encoder/decoder stages (default: 3).
    use_skip : bool, optional
        Whether to use skip connections (default: True).
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        time_dim: int = 256,
        encoders_numbers: int = 3,
        use_skip: bool = True
    ):
        super().__init__()

        if base_channels % 8 != 0:
            raise ValueError(
                f"base_channels must be divisible by 8 (GroupNorm), "
                f"got {base_channels}"
            )
        self.encoders_numbers = encoders_numbers

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        # Encoder
        self.encoders = nn.ModuleList()
        channels = base_channels

        self.encoders.append(
            EncoderBlock(in_channels, channels, time_dim)
        )

        for _ in range(1, encoders_numbers):
            self.encoders.append(
                EncoderBlock(channels, channels * 2, time_dim)
            )
            channels *= 2

        # Bottleneck
        self.bottleneck = ConvBlock(channels, channels * 2, time_dim)
        channels *= 2

        # Decoder
        self.decoders = nn.ModuleList()
        for _ in range(encoders_numbers):
            self.decoders.append(
                DecoderBlock(channels, channels // 2, time_dim, use_skip)
            )
            channels //= 2

        # Output head: predicts noise
        self.out = nn.Conv2d(channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion U-Net.

        Parameters
        ----------
        x : torch.Tensor
            Noisy input image of shape (B, 1, H, W).
        t : torch.Tensor
            Diffusion timesteps of shape (B,).

        Returns
        -------
        eps_hat : torch.Tensor
            Predicted noise of shape (B, 1, H, W).
        """
        _check_spatial_size(x, self.encoders_numbers)

        t = t.float()
        t_emb = self.time_embedding(t)

        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x, t_emb)
            skips.insert(0, skip)

        x = self.bottleneck(x, t_emb)

        for decoder, skip in zip(self.decoders, skips):
            x = decoder(x, skip, t_emb)

        return self.out(x)

# ----------------- Modules for U-Net (auto-encoder) ---------------------------

class ConvBlockAE(nn.Module):
    """
    Convolutional block for U-Net autoencoder.

    Two 3×3 convolutions, each followed by GroupNorm and SiLU.
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, C, H, W).

        Returns
        -------
        out : torch.Tensor
            Output feature map of shape (B, out_c, H, W).
        """
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x
        
        
class EncoderBlockAE(nn.Module):
    """
    Encoder block: ConvBlock followed by downsampling
    
    Parameters
    ----------
    in_c : int
        Number of input channels.
    out_c : int
        Number of output channels.
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = ConvBlockAE(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        skip : torch.Tensor
            Feature map for skip connection.
        down : torch.Tensor
            Downsampled feature map.
        """
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DecoderBlockAE(nn.Module):
    """
    Decoder block: Upsample + ConvBlock
    
    Parameters
    ----------
    in_c : int
        Number of input channels.
    out_c : int
        Number of output channels.
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvBlockAE(in_c, out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, C, H, W).

        Returns
        -------
        out : torch.Tensor
            Output feature map after upsampling and convolution.
        """
        x = self.up(x)
        return self.conv(x)

class UNetAutoEncoder(nn.Module):
    """
    U-Net-shaped auto-encoder without skip connections (the input must go
    through the bottleneck).

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels (default: 1).
    base_channels : int, optional
        Number of channels after the first encoder (default: 64).
    encoders_numbers : int, optional
        Number of encoder/decoder stages (default: 3).
    """
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        encoders_numbers: int = 3
    ):
        super().__init__()

        if base_channels % 8 != 0:
            raise ValueError(
                f"base_channels must be divisible by 8 (GroupNorm), "
                f"got {base_channels}"
            )
        self.encoders_numbers = encoders_numbers

        self.encoders = nn.ModuleList()
        channels = base_channels

        self.encoders.append(
            EncoderBlockAE(in_channels, channels)
        )

        for _ in range(1, encoders_numbers):
            self.encoders.append(
                EncoderBlockAE(channels, channels * 2)
            )
            channels *= 2

        self.bottleneck = ConvBlockAE(channels, channels * 2)
        channels *= 2

        self.decoders = nn.ModuleList()
        for _ in range(encoders_numbers):
            self.decoders.append(
                DecoderBlockAE(channels, channels // 2)
            )
            channels //= 2
            
        self.out = nn.Conv2d(channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the auto-encoder.

        Parameters
        ----------
        x : torch.Tensor
            Original image of shape (B, 1, H, W).

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed image of shape (B, 1, H, W).
        """
        _check_spatial_size(x, self.encoders_numbers)

        for encoder in self.encoders:
            _, x = encoder(x)

        x = self.bottleneck(x)

        for decoder in self.decoders:
            x = decoder(x)

        return self.out(x)

