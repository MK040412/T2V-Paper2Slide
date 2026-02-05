// Chapter 4: Training Methodology

= Training Methodology

This chapter covers the complete training procedure for Video Diffusion Models.

== Training Objective

=== ε-Prediction Loss

The primary training objective uses ε-prediction:

#block(fill: rgb("#e8f4e8"), inset: 1em, radius: 4pt)[
  *Definition 4.1 (ε-Prediction Loss)*

  $ cal(L)_epsilon = EE_(bold(x), epsilon, t) [ || epsilon_theta (bold(z)_t, bold(c), lambda_t) - epsilon ||_2^2 ] $

  where:
  - $bold(x) tilde p_"data"$ is a video from the dataset
  - $epsilon tilde cal(N)(bold(0), bold(I))$ is Gaussian noise
  - $t tilde "Uniform"[0, 1]$
  - $bold(z)_t = alpha_t bold(x) + sigma_t epsilon$
]

=== v-Prediction Loss (Alternative)

For some experiments (BAIR, Kinetics), v-prediction is used:

$ bold(v) = alpha_t epsilon - sigma_t bold(x) $

$ cal(L)_v = EE_(bold(x), epsilon, t) [ || bold(v)_theta (bold(z)_t, bold(c), lambda_t) - bold(v) ||_2^2 ] $

v-prediction is equivalent to predicting the "velocity" in the diffusion process:
- At $t = 0$: $bold(v) approx epsilon$ (predicting noise)
- At $t = 1$: $bold(v) approx -bold(x)$ (predicting negative signal)

== Noise Schedule

=== Cosine Schedule

The paper uses a *cosine noise schedule* as proposed by Nichol & Dhariwal (2021):

#block(fill: rgb("#fff3cd"), inset: 1em, radius: 4pt)[
  *Definition 4.2 (Cosine Schedule)*

  $ alpha_t^2 = cos^2 ((t + s) / (1 + s) dot pi / 2) $

  where $s = 0.008$ is a small offset to prevent $alpha_0 = 0$.

  For variance-preserving: $sigma_t^2 = 1 - alpha_t^2$
]

#figure(
  ```
  Cosine Schedule Visualization:

  α_t (signal)     σ_t (noise)
  1.0 ─┐            1.0 ─          ─┐
      │ ╲                          │ ╱
  0.5 ─  ╲          0.5 ─      ╱   │
      │   ╲──           │  ╱      │
  0.0 ─────┴──────   0.0 ─┴───────┴──
      0    0.5  1       0    0.5   1
          t                  t

  log-SNR λ_t = log(α_t²/σ_t²)
  Range: [20, -20] (approximately)
  ```,
  caption: [Cosine noise schedule visualization]
)

=== Log-SNR Range

The log signal-to-noise ratio is clipped to $[-20, 20]$:

$ lambda_t = "clip"(log(alpha_t^2 \/ sigma_t^2), -20, 20) $

This prevents numerical instabilities at extreme noise levels.

== Training Algorithm

=== Complete Training Loop

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Algorithm 4.1: Video Diffusion Training*

  ```python
  def train_step(model, optimizer, video_batch, cond_batch):
      """Single training step for video diffusion model."""

      B, T, H, W, C = video_batch.shape

      # 1. Sample random timesteps
      t = torch.rand(B, device=device)

      # 2. Compute noise schedule parameters
      alpha_t, sigma_t = cosine_schedule(t)
      lambda_t = torch.log(alpha_t**2 / sigma_t**2)

      # 3. Sample noise
      epsilon = torch.randn_like(video_batch)

      # 4. Create noisy videos
      z_t = alpha_t[:, None, None, None, None] * video_batch + \
            sigma_t[:, None, None, None, None] * epsilon

      # 5. Predict noise
      epsilon_pred = model(z_t, cond_batch, lambda_t)

      # 6. Compute loss
      loss = F.mse_loss(epsilon_pred, epsilon)

      # 7. Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      return loss.item()
  ```
]

=== Joint Image-Video Training

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Algorithm 4.2: Joint Training Data Loading*

  ```python
  def load_joint_batch(video_loader, image_loader, num_images_per_video=8):
      """Load batch with both video and independent images."""

      # Load video batch
      videos = next(video_loader)  # [B, T, H, W, C]
      conds = videos.conditions     # [B, D_cond]

      # Load additional images (from random videos)
      images = next(image_loader)   # [B, K, H, W, C]

      # Concatenate along temporal dimension
      # videos: frames 0..T-1 (full temporal attention)
      # images: frames T..T+K-1 (identity attention)
      combined = torch.cat([videos, images], dim=1)  # [B, T+K, H, W, C]

      # Create attention mask
      T = videos.shape[1]
      K = images.shape[1]
      total = T + K

      # Video frames attend to all video frames
      # Image frames attend only to themselves
      mask = torch.zeros(total, total)
      mask[:T, :T] = 1  # Video-video attention
      mask[T:, T:] = torch.eye(K)  # Image self-attention only

      return combined, conds, mask
  ```
]

== Optimizer and Hyperparameters

=== Adam Optimizer Configuration

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    [*Hyperparameter*], [*Value*], [*Notes*],
    [Optimizer], [Adam], [Standard Adam],
    [$beta_1$], [0.9], [Momentum coefficient],
    [$beta_2$], [0.99], [RMSprop coefficient],
    [Learning rate], [0.0002-0.0003], [Task dependent],
    [Weight decay], [0.0-0.01], [Task dependent],
    [Gradient clipping], [None], [Not used],
  ),
  caption: [Optimizer configuration]
)

=== Exponential Moving Average (EMA)

The paper uses EMA for model weights:

$ theta_"EMA" arrow.l mu dot theta_"EMA" + (1 - mu) dot theta $

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    [*Task*], [*EMA Decay ($mu$)*],
    [UCF101], [0.9999],
    [BAIR], [0.999],
    [Kinetics], [0.9999],
    [Text-to-Video], [0.9999],
  ),
  caption: [EMA decay rates by task]
)

== Classifier-Free Guidance Training

=== Training with Dropout

To enable classifier-free guidance at inference, the model is trained with conditioning dropout:

#block(fill: rgb("#e8f4e8"), inset: 1em, radius: 4pt)[
  *Algorithm 4.3: Classifier-Free Guidance Training*

  ```python
  def train_step_cfg(model, optimizer, video_batch, cond_batch, p_uncond=0.1):
      """Training with classifier-free guidance."""

      B = video_batch.shape[0]

      # Randomly drop conditioning
      uncond_mask = torch.rand(B) < p_uncond
      cond_batch[uncond_mask] = 0  # Zero out conditioning

      # Rest of training is the same
      # ...
  ```
]

During inference, classifier-free guidance is applied:

$ tilde(epsilon)_theta = (1 + w) epsilon_theta (bold(z)_t, bold(c)) - w epsilon_theta (bold(z)_t, bold(0)) $

== Training Infrastructure

=== Compute Requirements

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    [*Task*], [*Hardware*], [*Training Steps*], [*Batch Size*],
    [UCF101], [128 TPU-v4], [60,000], [128],
    [BAIR], [128 TPU-v4], [660,000], [128],
    [Kinetics], [256 TPU-v4], [220,000], [256],
    [Text-to-Video (large)], [128 TPU-v4], [700,000], [128],
  ),
  caption: [Training compute requirements]
)

=== Data Augmentation

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    [*Task*], [*Augmentation*],
    [UCF101], [None],
    [BAIR], [Left-right flips],
    [Kinetics], [None],
    [Text-to-Video], [None],
  ),
  caption: [Data augmentation by task]
)

== Training Stability Considerations

=== Numerical Precision

- Use *mixed precision* (FP16/BF16) for efficient training
- Keep critical operations in FP32:
  - Noise schedule computation
  - Loss computation
  - Optimizer state

=== Gradient Scaling

With mixed precision, gradient scaling is recommended:

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    loss = compute_loss(model, batch)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

=== Memory Management

Video training is memory-intensive. Strategies used:
- *Gradient checkpointing*: Trade compute for memory
- *Small batch sizes*: Compensate with more steps
- *Joint training*: More effective use of batch

== Loss Weighting Considerations

=== Uniform Weighting

The paper uses uniform weighting over timesteps:

$ cal(L) = EE_t [ cal(L)_t ] "where" t tilde "Uniform"[0, 1] $

=== Alternative Weightings

Other works have explored different weightings:
- *Min-SNR weighting*: Down-weight high-noise timesteps
- *Importance sampling*: Sample timesteps based on loss magnitude
- *Loss-aware weighting*: Adaptive weighting during training

These alternatives may improve training but were not explored in this paper.
