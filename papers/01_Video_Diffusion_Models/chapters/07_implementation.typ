// Chapter 7: From-Scratch Implementation Guide

= From-Scratch Implementation Guide

This chapter provides a complete PyTorch implementation of Video Diffusion Models.

== Implementation Overview

=== Required Dependencies

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from typing import Optional, Tuple, List
```

== Core Components

=== 1. Noise Schedule

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Implementation: Cosine Noise Schedule*

  ```python
  class CosineNoiseSchedule:
      """Cosine noise schedule as in Nichol & Dhariwal (2021)."""

      def __init__(self, s: float = 0.008):
          self.s = s

      def __call__(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
          """
          Args:
              t: Timesteps in [0, 1], shape (B,) or scalar

          Returns:
              alpha_t: Signal coefficient
              sigma_t: Noise coefficient
          """
          # Ensure t is tensor
          if not isinstance(t, torch.Tensor):
              t = torch.tensor(t)

          # Cosine schedule
          f_t = torch.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2
          f_0 = math.cos(self.s / (1 + self.s) * math.pi / 2) ** 2

          alpha_t_sq = f_t / f_0
          alpha_t = torch.sqrt(alpha_t_sq)
          sigma_t = torch.sqrt(1 - alpha_t_sq)

          return alpha_t, sigma_t

      def get_snr(self, t: torch.Tensor) -> torch.Tensor:
          """Get log signal-to-noise ratio."""
          alpha_t, sigma_t = self(t)
          snr = (alpha_t ** 2) / (sigma_t ** 2)
          log_snr = torch.log(snr)
          return torch.clamp(log_snr, -20, 20)
  ```
]

=== 2. Timestep Embedding

```python
class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps, shape (B,)

        Returns:
            Embedding, shape (B, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=t.device)
            / half_dim
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding


class ConditioningMLP(nn.Module):
    """MLP to process conditioning embeddings."""

    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else out_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
```

=== 3. Spatial Convolution (1×3×3)

```python
class SpatialConv3d(nn.Module):
    """3D convolution that only operates spatially (1×k×k kernel)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, C, T, H, W)

        Returns:
            Output tensor, shape (B, C', T, H, W)
        """
        return self.conv(x)
```

=== 4. Spatial Self-Attention

```python
class SpatialSelfAttention(nn.Module):
    """Self-attention over spatial dimensions (H×W)."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.GroupNorm(32, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, C, T, H, W)

        Returns:
            Output tensor, shape (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape

        # Normalize
        x_norm = self.norm(x)

        # Reshape: treat T as batch
        x_flat = rearrange(x_norm, 'b c t h w -> (b t) (h w) c')

        # QKV projection
        qkv = self.qkv(x_flat)
        q, k, v = rearrange(
            qkv, 'bt n (three h d) -> three bt h n d',
            three=3, h=self.num_heads
        )

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = rearrange(out, 'bt h n d -> bt n (h d)')
        out = self.proj(out)
        out = rearrange(out, '(b t) (h w) c -> b c t h w', b=B, t=T, h=H, w=W)

        # Residual
        return x + out
```

=== 5. Temporal Self-Attention with Relative Position

```python
class TemporalSelfAttention(nn.Module):
    """Self-attention over temporal dimension with relative position bias."""

    def __init__(self, dim: int, num_heads: int = 8, max_frames: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.GroupNorm(32, dim)

        # Relative position bias
        # Range: -(max_frames-1) to +(max_frames-1)
        self.rel_pos_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * max_frames - 1)
        )
        self.max_frames = max_frames

    def get_rel_pos_bias(self, T: int) -> torch.Tensor:
        """Get relative position bias matrix for T frames."""
        # Create relative position indices
        positions = torch.arange(T)
        rel_pos = positions[:, None] - positions[None, :]  # (T, T)
        rel_pos = rel_pos + self.max_frames - 1  # Shift to positive indices

        # Gather bias values
        bias = self.rel_pos_bias[:, rel_pos]  # (num_heads, T, T)
        return bias

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, C, T, H, W)
            mask: Optional attention mask for joint image-video training

        Returns:
            Output tensor, shape (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape

        # Normalize
        x_norm = self.norm(x)

        # Reshape: treat (H×W) as batch
        x_flat = rearrange(x_norm, 'b c t h w -> (b h w) t c')

        # QKV projection
        qkv = self.qkv(x_flat)
        q, k, v = rearrange(
            qkv, 'bhw t (three h d) -> three bhw h t d',
            three=3, h=self.num_heads
        )

        # Attention with relative position bias
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        rel_pos_bias = self.get_rel_pos_bias(T).to(x.device)
        attn = attn + rel_pos_bias.unsqueeze(0)

        # Apply mask if provided (for joint image-video training)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape back
        out = rearrange(out, 'bhw h t d -> bhw t (h d)')
        out = self.proj(out)
        out = rearrange(out, '(b h w) t c -> b c t h w', b=B, h=H, w=W)

        # Residual
        return x + out
```

=== 6. ResBlock with Conditioning

```python
class ResBlock(nn.Module):
    """Residual block with conditioning injection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = SpatialConv3d(in_channels, out_channels)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = SpatialConv3d(out_channels, out_channels)

        # Conditioning projection (scale and shift)
        self.emb_proj = nn.Linear(emb_dim, out_channels * 2)

        self.dropout = nn.Dropout(dropout)

        # Skip connection
        if in_channels != out_channels:
            self.skip = SpatialConv3d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, C, T, H, W)
            emb: Conditioning embedding, shape (B, emb_dim)

        Returns:
            Output tensor, shape (B, C', T, H, W)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Inject conditioning
        emb_out = self.emb_proj(F.silu(emb))
        scale, shift = emb_out.chunk(2, dim=-1)
        scale = scale[:, :, None, None, None]  # (B, C, 1, 1, 1)
        shift = shift[:, :, None, None, None]
        h = h * (1 + scale) + shift

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip(x) + h
```

=== 7. Complete U-Net Block

```python
class UNetBlock(nn.Module):
    """U-Net block with ResBlocks and attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_res_blocks: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()

        self.res_blocks = nn.ModuleList()
        self.spatial_attns = nn.ModuleList()
        self.temporal_attns = nn.ModuleList()

        for i in range(num_res_blocks):
            self.res_blocks.append(
                ResBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    emb_dim,
                    dropout,
                )
            )
            if use_attention:
                self.spatial_attns.append(
                    SpatialSelfAttention(out_channels, num_heads)
                )
                self.temporal_attns.append(
                    TemporalSelfAttention(out_channels, num_heads)
                )
            else:
                self.spatial_attns.append(nn.Identity())
                self.temporal_attns.append(nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for res, spatial_attn, temporal_attn in zip(
            self.res_blocks, self.spatial_attns, self.temporal_attns
        ):
            x = res(x, emb)
            x = spatial_attn(x)
            if isinstance(temporal_attn, TemporalSelfAttention):
                x = temporal_attn(x, temporal_mask)
            else:
                x = temporal_attn(x)
        return x
```

=== 8. Complete 3D U-Net

```python
class VideoUNet(nn.Module):
    """Complete 3D U-Net for Video Diffusion Models."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 256,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (8, 16, 32),
        num_heads: int = 8,
        dropout: float = 0.1,
        emb_dim: int = 1024,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels

        # Timestep embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(base_channels),
            ConditioningMLP(base_channels, emb_dim, num_layers=4),
        )

        # Initial convolution
        self.conv_in = SpatialConv3d(in_channels, base_channels)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        ch = base_channels
        current_res = 64  # Assuming 64x64 input

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = current_res in attention_resolutions

            self.down_blocks.append(
                UNetBlock(ch, out_ch, emb_dim, num_res_blocks, num_heads, dropout, use_attn)
            )

            ch = out_ch
            if i < len(channel_mults) - 1:
                self.downsamplers.append(
                    nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
                current_res //= 2
            else:
                self.downsamplers.append(nn.Identity())

        # Middle block
        self.mid_block = UNetBlock(ch, ch, emb_dim, num_res_blocks, num_heads, dropout, True)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            # Account for skip connection (double channels)
            in_ch = ch + out_ch  # Skip connection concatenation

            use_attn = current_res in attention_resolutions

            self.up_blocks.append(
                UNetBlock(in_ch, out_ch, emb_dim, num_res_blocks + 1, num_heads, dropout, use_attn)
            )

            ch = out_ch
            if i > 0:
                self.upsamplers.append(
                    nn.ConvTranspose3d(ch, ch, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
                )
                current_res *= 2
            else:
                self.upsamplers.append(nn.Identity())

        # Output
        self.norm_out = nn.GroupNorm(32, ch)
        self.conv_out = SpatialConv3d(ch, in_channels)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy video, shape (B, T, H, W, C)
            t: Timesteps, shape (B,)
            cond: Optional conditioning, shape (B, cond_dim)
            temporal_mask: Optional mask for joint training

        Returns:
            Predicted noise, shape (B, T, H, W, C)
        """
        # Convert from (B, T, H, W, C) to (B, C, T, H, W)
        x = rearrange(x, 'b t h w c -> b c t h w')

        # Timestep embedding
        emb = self.time_embed(t)
        if cond is not None:
            emb = emb + cond  # Add conditioning

        # Initial conv
        h = self.conv_in(x)

        # Downsampling with skip connections
        skips = []
        for down_block, downsampler in zip(self.down_blocks, self.downsamplers):
            h = down_block(h, emb, temporal_mask)
            skips.append(h)
            h = downsampler(h)

        # Middle
        h = self.mid_block(h, emb, temporal_mask)

        # Upsampling with skip connections
        for up_block, upsampler, skip in zip(
            self.up_blocks, self.upsamplers, reversed(skips)
        ):
            h = torch.cat([h, skip], dim=1)  # Skip connection
            h = up_block(h, emb, temporal_mask)
            h = upsampler(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # Convert back to (B, T, H, W, C)
        h = rearrange(h, 'b c t h w -> b t h w c')

        return h
```

== Complete Training Script

```python
class VideoDiffusion:
    """Video Diffusion Model wrapper."""

    def __init__(
        self,
        model: VideoUNet,
        noise_schedule: CosineNoiseSchedule,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.noise_schedule = noise_schedule
        self.device = device

    def train_step(
        self,
        videos: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        optimizer: torch.optim.Optimizer = None,
    ) -> float:
        """Single training step."""
        self.model.train()
        B = videos.shape[0]

        # Sample timesteps
        t = torch.rand(B, device=self.device)

        # Get noise schedule
        alpha_t, sigma_t = self.noise_schedule(t)

        # Sample noise
        epsilon = torch.randn_like(videos)

        # Create noisy videos
        alpha_t = alpha_t[:, None, None, None, None]
        sigma_t = sigma_t[:, None, None, None, None]
        z_t = alpha_t * videos + sigma_t * epsilon

        # Predict noise
        epsilon_pred = self.model(z_t, t, cond)

        # Loss
        loss = F.mse_loss(epsilon_pred, epsilon)

        # Backward
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        cond: Optional[torch.Tensor] = None,
        num_steps: int = 256,
        guidance_weight: float = 0.0,
    ) -> torch.Tensor:
        """Generate videos using ancestral sampling."""
        self.model.eval()

        # Start from noise
        z = torch.randn(shape, device=self.device)

        timesteps = torch.linspace(1, 0, num_steps + 1, device=self.device)

        for i in range(num_steps):
            t = timesteps[i].expand(shape[0])
            s = timesteps[i + 1]

            alpha_t, sigma_t = self.noise_schedule(t[0])
            alpha_s, sigma_s = self.noise_schedule(s)

            # Model prediction
            if guidance_weight > 0 and cond is not None:
                # Classifier-free guidance
                epsilon_cond = self.model(z, t, cond)
                epsilon_uncond = self.model(z, t, None)
                epsilon_pred = (1 + guidance_weight) * epsilon_cond - guidance_weight * epsilon_uncond
            else:
                epsilon_pred = self.model(z, t, cond)

            # Reconstruct x
            x_pred = (z - sigma_t * epsilon_pred) / alpha_t

            # Reverse step
            if s > 0:
                lambda_t = torch.log(alpha_t**2 / sigma_t**2)
                lambda_s = torch.log(alpha_s**2 / sigma_s**2)
                coef = torch.exp(lambda_t - lambda_s)

                mu = coef * (alpha_s / alpha_t) * z + (1 - coef) * alpha_s * x_pred
                sigma = torch.sqrt((1 - coef) * sigma_s**2)

                z = mu + sigma * torch.randn_like(z)
            else:
                z = x_pred

        return z
```

== Usage Example

```python
# Initialize model
model = VideoUNet(
    in_channels=3,
    base_channels=256,
    channel_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    attention_resolutions=(8, 16, 32),
    num_heads=8,
    dropout=0.1,
    emb_dim=1024,
)

noise_schedule = CosineNoiseSchedule()
diffusion = VideoDiffusion(model, noise_schedule, device='cuda')

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.99))

for epoch in range(num_epochs):
    for videos in dataloader:
        videos = videos.to('cuda')  # (B, T, H, W, C)
        loss = diffusion.train_step(videos, optimizer=optimizer)
        print(f"Loss: {loss:.4f}")

# Sampling
samples = diffusion.sample(
    shape=(4, 16, 64, 64, 3),
    num_steps=256,
)
```

== Key Implementation Notes

=== Memory Optimization
- Use gradient checkpointing for large models
- Mixed precision training (FP16/BF16)
- Accumulate gradients for effective larger batch sizes

=== Common Pitfalls
1. *Tensor shape confusion*: Consistently use (B, T, H, W, C) or (B, C, T, H, W)
2. *Noise schedule numerical issues*: Clamp log-SNR to [-20, 20]
3. *Attention memory*: Use flash attention if available
4. *EMA*: Don't forget exponential moving average for sampling

=== Performance Tips
- Pre-compute noise schedule values
- Use efficient attention implementations (xformers, flash-attn)
- Optimize data loading (prefetch, pin memory)
