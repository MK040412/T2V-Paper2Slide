// Chapter 5: Sampling and Reconstruction Guidance
#import "../template.typ": *
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.2"

#show: chapter-template.with(
  title: "Sampling & Reconstruction Guidance",
  subtitle: "From Noise to Video",
  chapter-num: 5,
)

= Sampling Algorithms <sampling>

== Overview

Sampling in diffusion models involves reversing the forward noising process. Starting from pure noise $bold(z)_1 tilde cal(N)(0, bold(I))$, we iteratively denoise to obtain a clean video $bold(x)$.

#figure(
  diagram(
    spacing: (8pt, 8pt),
    node-stroke: 1pt,

    // Reverse process
    node((0, 0), $bold(z)_1$\ (Noise), shape: rect, fill: red.lighten(70%), width: 1.5cm),
    node((1.5, 0), $bold(z)_(0.75)$, shape: rect, fill: orange.lighten(70%), width: 1.5cm),
    node((3, 0), $bold(z)_(0.5)$, shape: rect, fill: yellow.lighten(70%), width: 1.5cm),
    node((4.5, 0), $bold(z)_(0.25)$, shape: rect, fill: green.lighten(70%), width: 1.5cm),
    node((6, 0), $bold(x)$\ (Video), shape: rect, fill: blue.lighten(70%), width: 1.5cm),

    edge((0, 0), (1.5, 0), "->", [$p_theta$], label-side: center),
    edge((1.5, 0), (3, 0), "->", [$p_theta$], label-side: center),
    edge((3, 0), (4.5, 0), "->", [$p_theta$], label-side: center),
    edge((4.5, 0), (6, 0), "->", [$p_theta$], label-side: center),
  ),
  caption: [Reverse diffusion: iteratively denoise from $bold(z)_1$ to $bold(x)$]
)

== Ancestral Sampling

#definition("Ancestral Sampling Step")[
  For timestep $t arrow s$ (where $s < t$), sample:
  $ bold(z)_s = tilde(mu)_(s|t)(bold(z)_t, hat(bold(x))_theta) + tilde(sigma)_(s|t) epsilon $ <ancestral>
  where:
  - $hat(bold(x))_theta = (bold(z)_t - sigma_t epsilon_theta(bold(z)_t, t)) / alpha_t$: predicted clean video
  - $tilde(mu)_(s|t)$: posterior mean
  - $tilde(sigma)_(s|t)$: posterior standard deviation
  - $epsilon tilde cal(N)(0, bold(I))$: fresh noise
]

#theorem("Posterior Mean and Variance")[
  The posterior parameters are:
  $ tilde(mu)_(s|t) = e^(lambda_t - lambda_s) (alpha_s / alpha_t) bold(z)_t + (1 - e^(lambda_t - lambda_s)) alpha_s hat(bold(x))_theta $
  $ tilde(sigma)_(s|t)^2 = (1 - e^(lambda_t - lambda_s)) sigma_s^2 $
]

#mathblock(title: "Derivation of Posterior Mean")[
  Starting from Bayes' rule:
  $ p(bold(z)_s | bold(z)_t, bold(x)) = (p(bold(z)_t | bold(z)_s) p(bold(z)_s | bold(x))) / (p(bold(z)_t | bold(x))) $

  Both distributions are Gaussian, so the posterior is also Gaussian:
  $ p(bold(z)_s | bold(z)_t, bold(x)) = cal(N)(bold(z)_s; tilde(mu)_(s|t), tilde(sigma)_(s|t)^2 bold(I)) $

  The mean can be derived by completing the square in the exponent.
]

== DDPM Sampling

#algorithm("DDPM Sampling")[
  *Input*: Trained model $epsilon_theta$, number of steps $N$

  *Algorithm*:
  1. Sample $bold(z)_1 tilde cal(N)(0, bold(I))$
  2. For $n = N, N-1, ..., 1$:
     - Set $t = n/N$, $s = (n-1)/N$
     - Get $alpha_t, sigma_t, alpha_s, sigma_s$ from schedule
     - Predict $epsilon_theta(bold(z)_t, t)$
     - Compute $hat(bold(x))_theta = (bold(z)_t - sigma_t epsilon_theta) / alpha_t$
     - Compute posterior mean $tilde(mu)_(s|t)$
     - Sample $bold(z)_s = tilde(mu)_(s|t) + tilde(sigma)_(s|t) epsilon$ (if $s > 0$)
  3. Return $bold(x) = bold(z)_0$
]

#codeblock(title: "PyTorch: DDPM Sampling")[
  ```python
  @torch.no_grad()
  def ddpm_sample(model, schedule, shape, num_steps=1000, device='cuda'):
      """
      DDPM sampling algorithm.

      Args:
          model: trained epsilon-prediction model
          schedule: noise schedule
          shape: output shape (B, C, T, H, W)
          num_steps: number of sampling steps
      """
      # Start from pure noise
      z_t = torch.randn(shape, device=device)

      # Timestep discretization
      timesteps = torch.linspace(1, 0, num_steps + 1, device=device)

      for i in range(num_steps):
          t = timesteps[i]
          s = timesteps[i + 1]

          # Get schedule parameters
          alpha_t, sigma_t = schedule(t)
          alpha_s, sigma_s = schedule(s)

          # Predict noise
          t_batch = t.expand(shape[0])
          eps_pred = model(z_t, t_batch)

          # Predict x_0
          x_pred = (z_t - sigma_t * eps_pred) / alpha_t

          # Clamp prediction
          x_pred = x_pred.clamp(-1, 1)

          # Posterior parameters
          log_snr_t = torch.log(alpha_t**2 / sigma_t**2)
          log_snr_s = torch.log(alpha_s**2 / sigma_s**2)
          c = torch.exp(log_snr_t - log_snr_s)

          mu = c * (alpha_s / alpha_t) * z_t + (1 - c) * alpha_s * x_pred
          sigma = ((1 - c) * sigma_s**2).sqrt()

          # Sample (except for last step)
          if s > 0:
              z_t = mu + sigma * torch.randn_like(z_t)
          else:
              z_t = mu

      return z_t
  ```
]

== DDIM Sampling (Deterministic)

#definition("DDIM Update")[
  DDIM (Denoising Diffusion Implicit Models) provides deterministic sampling:
  $ bold(z)_s = alpha_s hat(bold(x))_theta + sigma_s ((bold(z)_t - alpha_t hat(bold(x))_theta) / sigma_t) $ <ddim>

  No random noise is added, making sampling deterministic.
]

#figure(
  tablex(
    columns: (auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Property*], [*DDPM*], [*DDIM*],
    hlinex(),
    [Stochasticity], [Stochastic], [Deterministic],
    [Steps needed], [~1000], [~50-100],
    [Sample diversity], [High], [Fixed per seed],
    [Interpolation], [Not possible], [Possible in latent space],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Comparison of DDPM and DDIM sampling]
)

#codeblock(title: "PyTorch: DDIM Sampling")[
  ```python
  @torch.no_grad()
  def ddim_sample(model, schedule, shape, num_steps=50, device='cuda'):
      """
      DDIM sampling (deterministic).
      """
      z_t = torch.randn(shape, device=device)
      timesteps = torch.linspace(1, 0, num_steps + 1, device=device)

      for i in range(num_steps):
          t = timesteps[i]
          s = timesteps[i + 1]

          alpha_t, sigma_t = schedule(t)
          alpha_s, sigma_s = schedule(s)

          t_batch = t.expand(shape[0])
          eps_pred = model(z_t, t_batch)

          # Predict x_0
          x_pred = (z_t - sigma_t * eps_pred) / alpha_t
          x_pred = x_pred.clamp(-1, 1)

          # DDIM update (deterministic)
          # z_s = alpha_s * x_pred + sigma_s * (z_t - alpha_t * x_pred) / sigma_t
          z_t = alpha_s * x_pred + sigma_s * eps_pred

      return z_t
  ```
]

== Classifier-Free Guidance at Inference

#definition("CFG at Sampling Time")[
  For conditional generation, combine predictions:
  $ tilde(epsilon)_theta = (1 + w) epsilon_theta(bold(z)_t, bold(c)) - w epsilon_theta(bold(z)_t, emptyset) $ <cfg-sample>
  where $w$ is the guidance scale.
]

#figure(
  cetz.canvas(length: 0.9cm, {
    import cetz.draw: *

    // Axes
    line((0, 0), (7, 0), stroke: black, mark: (end: "stealth"))
    line((0, 0), (0, 4.5), stroke: black, mark: (end: "stealth"))
    content((7.5, 0), $w$)
    content((-0.5, 4.5), text(size: 9pt)[Metric])

    // Quality curve (increases then plateaus)
    let pts_quality = ()
    for i in range(35) {
      let w = i / 5
      let quality = 3.5 * (1 - calc.exp(-w * 0.8))
      pts_quality.push((w, quality))
    }
    line(..pts_quality, stroke: blue + 2pt)

    // Diversity curve (decreases)
    let pts_diversity = ()
    for i in range(35) {
      let w = i / 5
      let diversity = 4 * calc.exp(-w * 0.2)
      pts_diversity.push((w, diversity))
    }
    line(..pts_diversity, stroke: red + 2pt)

    // Labels
    content((5.5, 3.8), text(fill: blue, size: 8pt)[Quality])
    content((5.5, 1), text(fill: red, size: 8pt)[Diversity])

    // Typical range
    rect((1, 0), (3, 4.3), fill: green.transparentize(90%), stroke: none)
    content((2, 0.3), text(fill: green.darken(20%), size: 7pt)[Typical: 1-3])

    // Axis labels
    content((0, -0.3), $0$)
    content((2.5, -0.3), $5$)
    content((5, -0.3), $10$)
  }),
  caption: [Effect of guidance scale $w$: quality vs diversity trade-off]
)

#codeblock(title: "PyTorch: CFG Sampling")[
  ```python
  @torch.no_grad()
  def sample_with_cfg(model, schedule, shape, condition, guidance_scale=7.5,
                      num_steps=50, device='cuda'):
      """
      Sample with classifier-free guidance.
      """
      z_t = torch.randn(shape, device=device)
      timesteps = torch.linspace(1, 0, num_steps + 1, device=device)

      for i in range(num_steps):
          t = timesteps[i]
          s = timesteps[i + 1]

          alpha_t, sigma_t = schedule(t)
          alpha_s, sigma_s = schedule(s)

          t_batch = t.expand(shape[0])

          # Conditional and unconditional predictions
          eps_cond = model(z_t, t_batch, condition)
          eps_uncond = model(z_t, t_batch, None)

          # CFG combination
          eps_pred = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

          # Update
          x_pred = (z_t - sigma_t * eps_pred) / alpha_t
          x_pred = x_pred.clamp(-1, 1)
          z_t = alpha_s * x_pred + sigma_s * eps_pred

      return z_t
  ```
]

#pagebreak()

= Reconstruction Guidance <reconstruction>

== The Problem with Naive Conditioning

For conditional generation where we want $p(bold(x)^b | bold(x)^a)$ (e.g., generating future frames $bold(x)^b$ given past frames $bold(x)^a$), a naive approach would:

1. Run the reverse process normally
2. At each step, replace the known portion $bold(x)^a$ with noisy versions of the actual data

#warning(title: "Why Naive Replacement Fails")[
  The replacement method gives:
  $ EE[bold(x)^b | bold(z)_t] quad "instead of" quad EE[bold(x)^b | bold(z)_t, bold(x)^a] $

  The model doesn't know that $bold(x)^a$ is *fixed* — it still samples freely, leading to inconsistent boundaries and poor quality.
]

#figure(
  diagram(
    spacing: (10pt, 10pt),
    node-stroke: 1pt,

    // Naive approach
    node((0, 0), [Naive\ Replacement], shape: rect, fill: red.lighten(80%), width: 2.5cm),
    node((2.5, 0), [Model predicts\ $EE[bold(x)^b | bold(z)_t]$], shape: rect, fill: orange.lighten(80%), width: 3cm),
    node((5.5, 0), [Inconsistent\ boundaries], shape: rect, fill: red.lighten(70%), width: 2.5cm),

    edge((0, 0), (2.5, 0), "->"),
    edge((2.5, 0), (5.5, 0), "->"),

    // Reconstruction guidance
    node((0, 1.5), [Recon.\ Guidance], shape: rect, fill: green.lighten(80%), width: 2.5cm),
    node((2.5, 1.5), [Model predicts\ $EE[bold(x)^b | bold(z)_t, bold(x)^a]$], shape: rect, fill: green.lighten(70%), width: 3cm),
    node((5.5, 1.5), [Consistent\ generation], shape: rect, fill: blue.lighten(80%), width: 2.5cm),

    edge((0, 1.5), (2.5, 1.5), "->"),
    edge((2.5, 1.5), (5.5, 1.5), "->"),
  ),
  caption: [Naive replacement vs. reconstruction guidance]
)

== Reconstruction Guidance Derivation

#theorem("Reconstruction Guidance")[
  To sample from $p(bold(x)^b | bold(x)^a)$, modify the prediction:
  $ tilde(bold(x))_theta^b = hat(bold(x))_theta^b - (w_r alpha_t) / 2 nabla_(bold(z)_t^b) ||bold(x)^a - hat(bold(x))_theta^a||_2^2 $ <recon-guidance>
  where:
  - $hat(bold(x))_theta$: standard model prediction
  - $w_r$: reconstruction guidance weight
  - $alpha_t$: signal coefficient at timestep $t$
]

#mathblock(title: "Mathematical Derivation")[
  We want to sample from $p(bold(z)_s | bold(z)_t, bold(x)^a)$.

  Using Bayes' rule:
  $ p(bold(z)_s | bold(z)_t, bold(x)^a) prop p(bold(x)^a | bold(z)_s) p(bold(z)_s | bold(z)_t) $

  The gradient of $log p(bold(x)^a | bold(z)_s)$ w.r.t. $bold(z)_s$ gives us the guidance term.

  Approximating with the model's prediction:
  $ nabla log p(bold(x)^a | bold(z)_t) approx -1/(2 sigma^2) nabla_(bold(z)_t) ||bold(x)^a - hat(bold(x))_theta^a||^2 $
]

== Applications of Reconstruction Guidance

=== 1. Video Prediction (Future Frame Generation)

#figure(
  cetz.canvas(length: 0.8cm, {
    import cetz.draw: *

    // Known frames
    for i in range(4) {
      rect((i * 1.8, 0), (i * 1.8 + 1.5, 1.2), fill: blue.lighten(80%), stroke: blue)
      content((i * 1.8 + 0.75, 0.6), text(size: 9pt)[F#str(i+1)])
    }

    // Unknown frames
    for i in range(4, 8) {
      rect((i * 1.8, 0), (i * 1.8 + 1.5, 1.2), fill: green.lighten(80%), stroke: green)
      content((i * 1.8 + 0.75, 0.6), text(size: 9pt)[F#str(i+1)])
    }

    // Labels
    content((3, -0.5), text(size: 8pt)[Known ($bold(x)^a$)])
    content((10.2, -0.5), text(size: 8pt)[Generated ($bold(x)^b$)])
  }),
  caption: [Video prediction: given first 4 frames, generate next 4]
)

=== 2. Autoregressive Extension (Long Videos)

Generate arbitrarily long videos by:
1. Generate first 16 frames
2. Use last 8 frames as $bold(x)^a$
3. Generate next 8 frames as $bold(x)^b$ using reconstruction guidance
4. Repeat

#figure(
  diagram(
    spacing: (5pt, 10pt),
    node-stroke: 1pt,

    node((0, 0), [Frames 1-16\ (generated)], shape: rect, fill: blue.lighten(80%), width: 2.5cm),
    node((2, 0), [Frames 9-16\ (context)], shape: rect, fill: orange.lighten(80%), width: 2cm),
    node((4, 0), [Frames 17-24\ (generated)], shape: rect, fill: green.lighten(80%), width: 2.5cm),
    node((6, 0), [Frames 17-24\ (context)], shape: rect, fill: orange.lighten(80%), width: 2cm),
    node((8, 0), [...], shape: rect, stroke: none),

    edge((0, 0), (2, 0), "->"),
    edge((2, 0), (4, 0), "->", [+guidance]),
    edge((4, 0), (6, 0), "->"),
    edge((6, 0), (8, 0), "->", [+guidance]),
  ),
  caption: [Autoregressive extension with reconstruction guidance]
)

=== 3. Temporal Super-Resolution (Frame Interpolation)

Double the frame rate by conditioning on even frames:

#figure(
  cetz.canvas(length: 0.7cm, {
    import cetz.draw: *

    // Even frames (known)
    for i in range(0, 8, 2) {
      rect((i * 1.5, 0), (i * 1.5 + 1, 1), fill: blue.lighten(80%), stroke: blue)
      content((i * 1.5 + 0.5, 0.5), text(size: 8pt)[#str(i+1)])
    }

    // Odd frames (to generate)
    for i in range(1, 8, 2) {
      rect((i * 1.5, 0), (i * 1.5 + 1, 1), fill: green.lighten(80%), stroke: green)
      content((i * 1.5 + 0.5, 0.5), text(size: 8pt)[#str(i+1)])
    }

    // Labels
    content((3, -0.5), text(size: 7pt)[Blue: known, Green: interpolated])
  }),
  caption: [Temporal super-resolution: interpolate odd frames]
)

=== 4. Spatial Super-Resolution

Upsample low-resolution videos:

#figure(
  diagram(
    spacing: (15pt, 10pt),
    node-stroke: 1pt,

    node((0, 0), [Low-res\ 64×64], shape: rect, fill: blue.lighten(80%), width: 2cm),
    node((2, 0), [Recon.\ Guidance], shape: rect, fill: yellow.lighten(80%), width: 2.5cm),
    node((4, 0), [High-res\ 128×128], shape: rect, fill: green.lighten(80%), width: 2cm),

    edge((0, 0), (2, 0), "->"),
    edge((2, 0), (4, 0), "->"),
  ),
  caption: [Spatial super-resolution with reconstruction guidance]
)

== Reconstruction Guidance Algorithm

#algorithm("Sampling with Reconstruction Guidance")[
  *Input*: Model $epsilon_theta$, known frames $bold(x)^a$, guidance weight $w_r$

  *Algorithm*:
  1. Sample $bold(z)_1 tilde cal(N)(0, bold(I))$
  2. For $t = 1, 0.99, ..., 0$:
     - Compute $hat(bold(x))_theta = (bold(z)_t - sigma_t epsilon_theta(bold(z)_t)) / alpha_t$
     - Compute reconstruction loss: $L = ||bold(x)^a - hat(bold(x))_theta^a||^2$
     - Compute gradient: $g = nabla_(bold(z)_t^b) L$
     - Update prediction: $tilde(bold(x))_theta^b = hat(bold(x))_theta^b - (w_r alpha_t) / 2 dot g$
     - Sample $bold(z)_s$ using $tilde(bold(x))_theta$
  3. Return generated frames $bold(x)^b$
]

#codeblock(title: "PyTorch: Reconstruction Guidance")[
  ```python
  @torch.no_grad()
  def sample_with_reconstruction_guidance(
      model, schedule, x_known, known_mask,
      guidance_weight=0.3, num_steps=100, device='cuda'
  ):
      """
      Sample with reconstruction guidance for conditional generation.

      Args:
          model: trained model
          schedule: noise schedule
          x_known: known frames (B, C, T, H, W)
          known_mask: boolean mask, True for known frames (T,)
          guidance_weight: w_r in the paper
          num_steps: sampling steps
      """
      shape = x_known.shape
      B, C, T, H, W = shape

      # Start from noise
      z_t = torch.randn(shape, device=device)

      # Put known frames
      timesteps = torch.linspace(1, 0, num_steps + 1, device=device)

      for i in range(num_steps):
          t = timesteps[i]
          s = timesteps[i + 1]

          alpha_t, sigma_t = schedule(t)
          alpha_s, sigma_s = schedule(s)

          # Enable gradient for guidance computation
          z_t.requires_grad_(True)

          # Predict x_0
          t_batch = t.expand(B)
          eps_pred = model(z_t, t_batch)
          x_pred = (z_t - sigma_t * eps_pred) / alpha_t

          # Reconstruction loss on known frames
          x_known_pred = x_pred[:, :, known_mask]
          x_known_actual = x_known[:, :, known_mask]
          recon_loss = F.mse_loss(x_known_pred, x_known_actual)

          # Gradient for unknown frames
          grad = torch.autograd.grad(recon_loss, z_t)[0]

          z_t.requires_grad_(False)

          # Apply reconstruction guidance to unknown frames
          x_pred_guided = x_pred.clone()
          unknown_mask = ~known_mask
          x_pred_guided[:, :, unknown_mask] -= (
              guidance_weight * alpha_t / 2 * grad[:, :, unknown_mask]
          )

          # Clamp
          x_pred_guided = x_pred_guided.clamp(-1, 1)

          # Replace known frames with noisy version of actual
          x_pred_guided[:, :, known_mask] = x_known[:, :, known_mask]

          # DDIM update
          z_t = alpha_s * x_pred_guided + sigma_s * (
              (z_t - alpha_t * x_pred_guided) / sigma_t
          )

      return z_t
  ```
]

== Experimental Validation

#figure(
  tablex(
    columns: (auto, auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Method*], [*FVD↓*], [*FID↓*], [*IS↑*],
    hlinex(),
    [Replacement], [451], [26], [7.0],
    [*Recon. Guidance*], [*136*], [*14*], [*10.3*],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Video prediction on Kinetics-600: 3.3× FVD improvement]
)

#innovation(title: "Key Results")[
  Reconstruction guidance provides:
  - *3.3× better FVD* than naive replacement
  - *1.9× better FID* for frame quality
  - *1.5× better IS* for sample quality
  - Smooth temporal transitions at boundaries
]

== Guidance Weight Selection

#figure(
  cetz.canvas(length: 0.8cm, {
    import cetz.draw: *

    // Axes
    line((0, 0), (7, 0), stroke: black, mark: (end: "stealth"))
    line((0, 0), (0, 4), stroke: black, mark: (end: "stealth"))
    content((7.5, 0), $w_r$)
    content((-0.3, 4.3), [FVD])

    // FVD curve (U-shaped)
    let pts = ()
    for i in range(35) {
      let w = i / 5
      let fvd = 2 + 0.5 * calc.pow(w - 1.5, 2)
      pts.push((w, fvd))
    }
    line(..pts, stroke: blue + 2pt)

    // Optimal point
    circle((1.5, 2), radius: 0.15, fill: red, stroke: none)
    content((1.5, 1.5), text(size: 8pt)[Optimal])

    // Labels
    content((0, -0.3), $0$)
    content((2.5, -0.3), $0.5$)
    content((5, -0.3), $1.0$)
  }),
  caption: [FVD vs reconstruction guidance weight: optimal around $w_r = 0.3$]
)

#warning(title: "Guidance Weight Tuning")[
  - Too low ($w_r < 0.1$): Insufficient conditioning, boundary artifacts
  - Optimal ($w_r approx 0.3$): Best balance
  - Too high ($w_r > 0.5$): Over-conditioning, reduced diversity, artifacts

  The paper uses $w_r = 0.3$ for most experiments.
]

#pagebreak()

== Complete Video Extension Pipeline

#codeblock(title: "PyTorch: Autoregressive Video Extension")[
  ```python
  def generate_long_video(
      model, schedule, text_condition,
      total_frames=64, chunk_size=16, overlap=8,
      guidance_weight=0.3, cfg_scale=7.5
  ):
      """
      Generate long videos autoregressively with reconstruction guidance.
      """
      device = next(model.parameters()).device
      B = 1  # Batch size
      C, H, W = 3, 64, 64

      # Generate first chunk
      shape = (B, C, chunk_size, H, W)
      video = sample_with_cfg(
          model, schedule, shape, text_condition,
          guidance_scale=cfg_scale
      )

      # Autoregressively extend
      while video.shape[2] < total_frames:
          # Use last `overlap` frames as context
          x_known = video[:, :, -overlap:]

          # Generate next chunk
          shape = (B, C, chunk_size, H, W)
          known_mask = torch.zeros(chunk_size, dtype=torch.bool, device=device)
          known_mask[:overlap] = True

          # Pad known frames to full chunk size
          x_known_padded = torch.zeros(shape, device=device)
          x_known_padded[:, :, :overlap] = x_known

          # Sample with reconstruction guidance
          new_chunk = sample_with_reconstruction_guidance(
              model, schedule, x_known_padded, known_mask,
              guidance_weight=guidance_weight
          )

          # Append new frames (excluding overlap)
          video = torch.cat([
              video,
              new_chunk[:, :, overlap:]
          ], dim=2)

          print(f'Generated {video.shape[2]} frames')

      # Trim to exact length
      return video[:, :, :total_frames]
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
    [*Topic*], [*Key Points*],
    hlinex(),
    [DDPM Sampling], [Stochastic, ~1000 steps, high diversity],
    [DDIM Sampling], [Deterministic, ~50-100 steps, interpolatable],
    [CFG], [$(1+w)epsilon_(cond) - w epsilon_(uncond)$, controls quality/diversity],
    [Reconstruction Guidance], [Conditions on known frames via gradient guidance],
    [Guidance Formula], [$tilde(x)^b = hat(x)^b - (w_r alpha_t)/2 nabla ||x^a - hat(x)^a||^2$],
    [Applications], [Video prediction, extension, super-resolution],
    [Optimal $w_r$], [~0.3 for best FVD],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Chapter 5 Summary]
)
