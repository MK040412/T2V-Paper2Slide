// Chapter 4: Training Methodology
#import "../template.typ": *
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.2"

#show: chapter-template.with(
  title: "Training Methodology",
  subtitle: "Noise Schedules, Joint Training, and Optimization",
  chapter-num: 4,
)

= Training Methodology <training>

== Training Overview

The VDM training process involves:
1. Adding noise to videos using a noise schedule
2. Training a neural network to predict the added noise
3. Using joint image-video training for improved quality
4. Applying classifier-free guidance for conditioning

#figure(
  diagram(
    spacing: (8pt, 10pt),
    node-stroke: 1pt,

    // Data
    node((0, 0), [Video $bold(x)$\ $T times H times W times 3$], shape: rect, fill: blue.lighten(85%), width: 2.5cm),

    // Noise sampling
    node((2, 0), [Sample\ $epsilon tilde cal(N)(0,I)$\ $t tilde cal(U)(0,1)$], shape: rect, fill: yellow.lighten(80%), width: 2.5cm),

    // Noisy video
    node((4, 0), [$bold(z)_t = alpha_t bold(x)$\ $+ sigma_t epsilon$], shape: rect, fill: orange.lighten(80%), width: 2.5cm),

    // Network
    node((6, 0), [3D U-Net\ $epsilon_theta(bold(z)_t, t)$], shape: rect, fill: green.lighten(80%), width: 2.5cm),

    // Loss
    node((8, 0), [MSE Loss\ $||epsilon_theta - epsilon||^2$], shape: rect, fill: red.lighten(80%), width: 2.5cm),

    // Arrows
    edge((0, 0), (2, 0), "->"),
    edge((2, 0), (4, 0), "->"),
    edge((4, 0), (6, 0), "->"),
    edge((6, 0), (8, 0), "->"),
  ),
  caption: [Training pipeline: add noise, predict noise, minimize MSE]
)

== Noise Schedule

#definition("Cosine Noise Schedule")[
  The VDM uses a *cosine schedule* for $alpha_t$:
  $ alpha_t^2 = cos^2 ((t + s) / (1 + s) dot pi / 2) $ <cosine-schedule>
  where $s = 0.008$ is a small offset to avoid singularities at $t=0$.

  For variance-preserving diffusion: $sigma_t = sqrt(1 - alpha_t^2)$
]

#figure(
  cetz.canvas(length: 0.9cm, {
    import cetz.draw: *

    // Axes
    line((0, 0), (8, 0), stroke: black, mark: (end: "stealth"))
    line((0, 0), (0, 4.5), stroke: black, mark: (end: "stealth"))
    content((8.3, 0), $t$)
    content((0, 4.8), text(size: 9pt)[Value])

    // Alpha squared curve (cosine)
    let pts_alpha = ()
    for i in range(41) {
      let t = i / 40
      let s = 0.008
      let alpha_sq = calc.pow(calc.cos((t + s) / (1 + s) * calc.pi / 2), 2)
      pts_alpha.push((t * 8, alpha_sq * 4))
    }
    line(..pts_alpha, stroke: blue + 2pt)

    // Sigma squared curve
    let pts_sigma = ()
    for i in range(41) {
      let t = i / 40
      let s = 0.008
      let alpha_sq = calc.pow(calc.cos((t + s) / (1 + s) * calc.pi / 2), 2)
      let sigma_sq = 1 - alpha_sq
      pts_sigma.push((t * 8, sigma_sq * 4))
    }
    line(..pts_sigma, stroke: red + 2pt)

    // Log SNR curve (scaled)
    let pts_snr = ()
    for i in range(1, 40) {
      let t = i / 40
      let s = 0.008
      let alpha_sq = calc.pow(calc.cos((t + s) / (1 + s) * calc.pi / 2), 2)
      let sigma_sq = 1 - alpha_sq
      let snr = calc.log(alpha_sq / sigma_sq) / 10 + 2  // Scaled for display
      pts_snr.push((t * 8, calc.max(0, calc.min(4, snr))))
    }
    line(..pts_snr, stroke: green + 2pt)

    // Labels
    content((1.5, 3.8), text(fill: blue, size: 8pt)[$alpha_t^2$])
    content((6.5, 3.8), text(fill: red, size: 8pt)[$sigma_t^2$])
    content((2, 1.5), text(fill: green, size: 8pt)[$lambda_t$ (scaled)])

    // Markers
    content((0, -0.3), $0$)
    content((8, -0.3), $1$)
    content((-0.3, 0), $0$)
    content((-0.3, 4), $1$)
  }),
  caption: [Cosine noise schedule: $alpha_t^2$ (blue), $sigma_t^2$ (red), log-SNR $lambda_t$ (green, scaled)]
)

#mathblock(title: "Why Cosine Schedule?")[
  The cosine schedule was introduced in "Improved DDPM" (Nichol & Dhariwal, 2021) because:
  1. Smoother than linear schedule at endpoints
  2. More uniform log-SNR distribution
  3. Better sample quality in practice

  The offset $s = 0.008$ prevents:
  - $alpha_0 = 1$ exactly (numerical issues)
  - Too rapid noise increase at $t=0$
]

#codeblock(title: "PyTorch: Cosine Schedule")[
  ```python
  class CosineSchedule:
      def __init__(self, s=0.008, clip_min=-20, clip_max=20):
          self.s = s
          self.clip_min = clip_min
          self.clip_max = clip_max

      def __call__(self, t):
          """
          Args:
              t: timesteps in [0, 1], shape (B,) or scalar
          Returns:
              alpha_t, sigma_t of same shape
          """
          # Cosine schedule
          f_t = torch.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2
          f_0 = math.cos(self.s / (1 + self.s) * math.pi / 2) ** 2

          alpha_sq = f_t / f_0
          sigma_sq = 1 - alpha_sq

          # Compute log SNR and clip
          log_snr = torch.log(alpha_sq / sigma_sq)
          log_snr = log_snr.clamp(self.clip_min, self.clip_max)

          # Recover alpha, sigma from clipped log SNR
          alpha_sq = torch.sigmoid(log_snr)
          sigma_sq = torch.sigmoid(-log_snr)

          return alpha_sq.sqrt(), sigma_sq.sqrt()
  ```
]

== Log-SNR Clipping

#warning(title: "Numerical Stability")[
  The paper clips log-SNR to $[-20, 20]$:
  $ lambda_t = "clip"(log(alpha_t^2 / sigma_t^2), -20, 20) $

  This corresponds to:
  - At $lambda = 20$: $alpha^2 / sigma^2 approx 5 times 10^8$ (nearly pure signal)
  - At $lambda = -20$: $alpha^2 / sigma^2 approx 2 times 10^{-9}$ (nearly pure noise)
]

== Training Objective

#definition("Simple ε-Loss")[
  $ cal(L)_"simple" = EE_(bold(x), epsilon, t) [||epsilon_theta (bold(z)_t, t) - epsilon||_2^2] $ <simple-loss>
  where:
  - $bold(x)$: clean video from dataset
  - $epsilon tilde cal(N)(0, bold(I))$: sampled noise
  - $t tilde cal(U)(0, 1)$: uniform random timestep
  - $bold(z)_t = alpha_t bold(x) + sigma_t epsilon$: noisy video
]

#theorem("Loss Weighting Equivalence")[
  The simple loss is equivalent to a weighted variational bound:
  $ cal(L)_"simple" = EE_t [omega(lambda_t) dot cal(L)_"vlb"(t)] $
  where $omega(lambda_t) = 1$ (uniform weighting across timesteps).
]

#figure(
  tablex(
    columns: (auto, auto, 1fr),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Loss Type*], [*Weight $omega(lambda_t)$*], [*Characteristics*],
    hlinex(),
    [Simple], [$1$], [Uniform; stable training],
    [VLB], [$e^(-lambda_t)$], [Theoretically optimal; unstable],
    [SNR-weighted], [$1 / (1 + e^(-lambda_t))$], [Balance quality/stability],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Different loss weighting schemes]
)

== Joint Image-Video Training

#innovation(title: "Key Discovery")[
  Training on images *and* videos jointly dramatically improves quality:

  #figure(
    tablex(
      columns: (auto, auto, auto, auto),
      align: center + horizon,
      auto-vlines: false,
      hlinex(stroke: 1.5pt),
      [*Training*], [*FVD↓*], [*FID↓*], [*IS↑*],
      hlinex(),
      [Video only], [205], [37], [--],
      [*Joint (8 images)*], [*61*], [*15*], [*--*],
      hlinex(stroke: 1.5pt),
    ),
    caption: [Effect of joint training on UCF101]
  )

  *3.4× FVD improvement* from adding just 8 images per video!
]

#figure(
  diagram(
    spacing: (10pt, 10pt),
    node-stroke: 1pt,

    // Video batch
    node((0, 0), [Video Batch\ $B_v times T times H times W times 3$], shape: rect, fill: blue.lighten(80%), width: 3cm),

    // Image batch
    node((0, 1.5), [Image Batch\ $B_i times 1 times H times W times 3$], shape: rect, fill: green.lighten(80%), width: 3cm),

    // Combined
    node((2.5, 0.75), [Combined Batch\ $(B_v + B_i) times T times H times W times 3$], shape: rect, fill: purple.lighten(80%), width: 3.5cm),

    // Attention mask
    node((5, 0.75), [Masked\ Attention], shape: rect, fill: orange.lighten(80%), width: 2.5cm),

    // Arrows
    edge((0, 0), (2.5, 0.75), "->"),
    edge((0, 1.5), (2.5, 0.75), "->"),
    edge((2.5, 0.75), (5, 0.75), "->"),
  ),
  caption: [Joint image-video training: images treated as single-frame videos]
)

#keypoint(title: "Attention Masking")[
  The attention mechanism is masked so that:
  - *Video frames* can attend to *all frames* in the same video
  - *Images* can only attend to *themselves* (no cross-frame attention)

  This prevents images from trying to learn temporal patterns.
]

#codeblock(title: "PyTorch: Attention Masking")[
  ```python
  def create_attention_mask(batch_is_video, T):
      """
      Create attention mask for joint image-video training.

      Args:
          batch_is_video: (B,) bool tensor, True if sample is video
          T: number of frames

      Returns:
          mask: (B, T, T) attention mask
      """
      B = batch_is_video.shape[0]
      mask = torch.zeros(B, T, T, dtype=torch.bool)

      for b in range(B):
          if batch_is_video[b]:
              # Video: full attention over all frames
              mask[b] = True
          else:
              # Image: only self-attention (diagonal)
              mask[b] = torch.eye(T, dtype=torch.bool)

      return mask
  ```
]

== Why Does Joint Training Help?

#remark("Hypotheses from the Paper")[
  1. *Data augmentation*: Images provide diverse spatial patterns
  2. *Regularization*: Prevents overfitting to video-specific artifacts
  3. *Better spatial features*: Images teach high-quality frame generation
  4. *Disentanglement*: Separates spatial and temporal learning
]

#figure(
  diagram(
    spacing: (8pt, 8pt),
    node-stroke: 1pt,

    // Video learning
    node((0, 0), [Video Data], shape: rect, fill: blue.lighten(80%), width: 2cm),
    node((2, 0), [Temporal\ Patterns], shape: rect, fill: blue.lighten(70%), width: 2cm),

    // Image learning
    node((0, 1.5), [Image Data], shape: rect, fill: green.lighten(80%), width: 2cm),
    node((2, 1.5), [Spatial\ Details], shape: rect, fill: green.lighten(70%), width: 2cm),

    // Shared model
    node((4, 0.75), [Shared\ 3D U-Net], shape: rect, fill: purple.lighten(80%), width: 2.5cm),

    // Output
    node((6, 0.75), [Better\ Videos], shape: rect, fill: orange.lighten(80%), width: 2cm),

    edge((0, 0), (2, 0), "->"),
    edge((0, 1.5), (2, 1.5), "->"),
    edge((2, 0), (4, 0.75), "->"),
    edge((2, 1.5), (4, 0.75), "->"),
    edge((4, 0.75), (6, 0.75), "->"),
  ),
  caption: [Joint training: images teach spatial quality, videos teach temporal consistency]
)

== Classifier-Free Guidance Training

#definition("CFG Training Protocol")[
  For text-to-video generation, randomly drop conditioning during training:

  ```python
  if random() < p_uncond:  # p_uncond = 0.1 typically
      conditioning = null_embedding
  else:
      conditioning = text_embedding
  ```

  This trains both conditional and unconditional models simultaneously.
]

#figure(
  tablex(
    columns: (auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Parameter*], [*Value*],
    hlinex(),
    [Unconditional dropout], [10%],
    [Guidance scale (inference)], [1.0 - 15.0],
    [Text encoder], [T5-XXL (frozen)],
    hlinex(stroke: 1.5pt),
  ),
  caption: [CFG training configuration]
)

== Training Configuration

#figure(
  tablex(
    columns: (auto, auto, auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Parameter*], [*UCF101*], [*BAIR*], [*Kinetics*], [*Text2Video*],
    hlinex(),
    [Base channels], [256], [128], [256], [256],
    [Channel mult], [(1,2,4,8)], [(1,2,3,4)], [(1,2,4,8)], [(1,2,4,8)],
    [Learning rate], [3e-4], [2e-4], [2e-4], [3e-4],
    [Batch size], [256], [256], [512], [256],
    [Training steps], [60K], [660K], [220K], [700K],
    [Hardware], [128 TPUv4], [128 TPUv4], [256 TPUv4], [128 TPUv4],
    [EMA decay], [0.9999], [0.9999], [0.9999], [0.9999],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Training hyperparameters for different tasks]
)

== Optimizer and Learning Rate

#definition("AdamW Optimizer")[
  The paper uses AdamW with the following settings:
  $ theta_(t+1) = theta_t - eta (m_t / (sqrt(v_t) + epsilon) + lambda theta_t) $
  where:
  - $eta$: learning rate (2e-4 to 3e-4)
  - $beta_1 = 0.9$, $beta_2 = 0.999$: momentum coefficients
  - $epsilon = 10^(-8)$: numerical stability
  - $lambda$: weight decay (not specified, likely 0.01)
]

#codeblock(title: "PyTorch: Training Loop")[
  ```python
  def train(model, dataloader, schedule, optimizer, ema, epochs):
      model.train()

      for epoch in range(epochs):
          for videos, texts in dataloader:
              # Sample timesteps
              B = videos.shape[0]
              t = torch.rand(B, device=videos.device)

              # Get noise schedule parameters
              alpha_t, sigma_t = schedule(t)
              alpha_t = alpha_t[:, None, None, None, None]
              sigma_t = sigma_t[:, None, None, None, None]

              # Sample noise and create noisy videos
              eps = torch.randn_like(videos)
              z_t = alpha_t * videos + sigma_t * eps

              # CFG: randomly drop conditioning
              if random.random() < 0.1:
                  texts = None

              # Forward pass
              eps_pred = model(z_t, t, texts)

              # Loss
              loss = F.mse_loss(eps_pred, eps)

              # Backward pass
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              # Update EMA
              ema.update(model)
  ```
]

== Exponential Moving Average (EMA)

#definition("EMA of Model Weights")[
  $ theta_"ema" = beta dot theta_"ema" + (1 - beta) dot theta $
  where $beta = 0.9999$ (very slow update).

  The EMA model is used for inference, providing smoother predictions.
]

#codeblock(title: "PyTorch: EMA Implementation")[
  ```python
  class EMA:
      def __init__(self, model, decay=0.9999):
          self.decay = decay
          self.shadow = {}
          for name, param in model.named_parameters():
              if param.requires_grad:
                  self.shadow[name] = param.data.clone()

      def update(self, model):
          for name, param in model.named_parameters():
              if param.requires_grad:
                  self.shadow[name] = (
                      self.decay * self.shadow[name] +
                      (1 - self.decay) * param.data
                  )

      def apply(self, model):
          for name, param in model.named_parameters():
              if param.requires_grad:
                  param.data.copy_(self.shadow[name])
  ```
]

== Data Preprocessing

#figure(
  tablex(
    columns: (auto, 1fr),
    align: (center, left),
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Step*], [*Details*],
    hlinex(),
    [1. Frame extraction], [Extract 16 consecutive frames at native FPS],
    [2. Spatial resize], [Resize to 64×64 (or target resolution)],
    [3. Normalization], [Scale pixels to $[-1, 1]$],
    [4. Random crop], [Random spatial crop during training],
    [5. Horizontal flip], [50% probability for augmentation],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Video preprocessing pipeline]
)

#codeblock(title: "PyTorch: Data Preprocessing")[
  ```python
  class VideoDataset(Dataset):
      def __init__(self, video_paths, num_frames=16, size=64):
          self.video_paths = video_paths
          self.num_frames = num_frames
          self.size = size

      def __getitem__(self, idx):
          # Load video
          video = load_video(self.video_paths[idx])

          # Sample consecutive frames
          start = random.randint(0, len(video) - self.num_frames)
          frames = video[start:start + self.num_frames]

          # Resize
          frames = F.interpolate(frames, size=(self.size, self.size))

          # Normalize to [-1, 1]
          frames = frames / 127.5 - 1.0

          # Random horizontal flip
          if random.random() > 0.5:
              frames = torch.flip(frames, dims=[-1])

          return frames  # (T, C, H, W)
  ```
]

== Memory Optimization

#warning(title: "GPU Memory Constraints")[
  Video diffusion models are memory-intensive:
  - 16 frames × 64×64 × 3 = 196K values per sample
  - With gradients and optimizer states: ~10× memory
  - Typical batch size: 8-16 per GPU
]

#figure(
  tablex(
    columns: (auto, 1fr),
    align: (center, left),
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Technique*], [*Memory Savings*],
    hlinex(),
    [Gradient checkpointing], [2-3× reduction, slight speed penalty],
    [Mixed precision (FP16)], [2× reduction for activations],
    [Gradient accumulation], [Effective larger batches without memory],
    [Model parallelism], [Distribute across GPUs],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Memory optimization techniques]
)

#pagebreak()

== Complete Training Script

#codeblock(title: "PyTorch: Full Training Script")[
  ```python
  import torch
  import torch.nn.functional as F
  from torch.utils.data import DataLoader

  def main():
      # Config
      device = 'cuda'
      batch_size = 8
      lr = 3e-4
      num_steps = 60000

      # Model
      model = UNet3D(
          in_channels=3,
          out_channels=3,
          base_channels=256,
          channel_mults=(1, 2, 4, 8),
      ).to(device)

      # Schedule and optimizer
      schedule = CosineSchedule()
      optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
      ema = EMA(model, decay=0.9999)

      # Data
      dataset = VideoDataset(video_paths, num_frames=16, size=64)
      dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

      # Training loop
      step = 0
      while step < num_steps:
          for videos in dataloader:
              videos = videos.to(device)
              B = videos.shape[0]

              # Sample t ~ U(0, 1)
              t = torch.rand(B, device=device)

              # Forward diffusion
              alpha, sigma = schedule(t)
              alpha = alpha.view(B, 1, 1, 1, 1)
              sigma = sigma.view(B, 1, 1, 1, 1)

              eps = torch.randn_like(videos)
              z_t = alpha * videos + sigma * eps

              # Predict noise
              eps_pred = model(z_t, t)

              # Loss
              loss = F.mse_loss(eps_pred, eps)

              # Update
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              ema.update(model)

              step += 1
              if step % 1000 == 0:
                  print(f'Step {step}, Loss: {loss.item():.4f}')

              if step >= num_steps:
                  break

      # Save
      ema.apply(model)
      torch.save(model.state_dict(), 'vdm_ema.pt')

  if __name__ == '__main__':
      main()
  ```
]

#pagebreak()

== Summary

#figure(
  tablex(
    columns: (auto, 1fr),
    align: (center, left),
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Aspect*], [*Key Points*],
    hlinex(),
    [Noise Schedule], [Cosine schedule with $s=0.008$, log-SNR clipped to $[-20, 20]$],
    [Loss], [Simple MSE: $EE[||epsilon_theta - epsilon||^2]$],
    [Joint Training], [8 images per video batch → 3.4× FVD improvement],
    [CFG Training], [10% unconditional dropout],
    [Optimizer], [AdamW with lr=3e-4],
    [EMA], [Decay=0.9999 for smoother inference],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Chapter 4 Summary]
)
