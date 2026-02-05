// Chapter 5: Sampling and Reconstruction Guidance

= Sampling and Reconstruction Guidance

This chapter covers the sampling algorithms, with special focus on the novel *reconstruction guidance* method.

== Standard Sampling Methods

=== Ancestral Sampling (DDPM-style)

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Algorithm 5.1: Ancestral Sampling*

  ```python
  def ancestral_sample(model, shape, num_steps=256, gamma=0.1):
      """Generate video using ancestral sampling."""

      device = next(model.parameters()).device

      # Start from pure noise
      z = torch.randn(shape, device=device)

      # Time steps from 1 to 0
      timesteps = torch.linspace(1, 0, num_steps + 1)

      for i in range(num_steps):
          t = timesteps[i]
          s = timesteps[i + 1]

          # Get noise schedule parameters
          alpha_t, sigma_t = cosine_schedule(t)
          alpha_s, sigma_s = cosine_schedule(s)
          lambda_t = torch.log(alpha_t**2 / sigma_t**2)
          lambda_s = torch.log(alpha_s**2 / sigma_s**2)

          # Model prediction
          epsilon_pred = model(z, t)

          # Reconstruct x
          x_pred = (z - sigma_t * epsilon_pred) / alpha_t

          # Compute reverse process mean
          coef = torch.exp(lambda_t - lambda_s)
          mu = coef * (alpha_s / alpha_t) * z + (1 - coef) * alpha_s * x_pred

          # Compute variance (interpolation between bounds)
          sigma_sq_s_t = (1 - coef) * sigma_s**2        # Lower bound
          sigma_sq_t_s = (1 - coef) * sigma_t**2        # Upper bound
          sigma = (sigma_sq_s_t**(1-gamma)) * (sigma_sq_t_s**gamma)
          sigma = torch.sqrt(sigma)

          # Sample z_s
          if s > 0:  # Don't add noise at final step
              noise = torch.randn_like(z)
              z = mu + sigma * noise
          else:
              z = mu

      return x_pred
  ```
]

=== Predictor-Corrector Sampling

The paper finds predictor-corrector sampling works especially well with reconstruction guidance:

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Algorithm 5.2: Predictor-Corrector Sampling*

  ```python
  def predictor_corrector_sample(model, shape, num_steps=256, delta=0.1):
      """Generate video using predictor-corrector sampling."""

      device = next(model.parameters()).device
      z = torch.randn(shape, device=device)
      timesteps = torch.linspace(1, 0, num_steps + 1)

      for i in range(num_steps):
          t = timesteps[i]
          s = timesteps[i + 1]
          alpha_t, sigma_t = cosine_schedule(t)
          alpha_s, sigma_s = cosine_schedule(s)

          # === PREDICTOR STEP (ancestral) ===
          epsilon_pred = model(z, t)
          x_pred = (z - sigma_t * epsilon_pred) / alpha_t

          # Reverse mean
          lambda_t = torch.log(alpha_t**2 / sigma_t**2)
          lambda_s = torch.log(alpha_s**2 / sigma_s**2)
          coef = torch.exp(lambda_t - lambda_s)
          mu = coef * (alpha_s / alpha_t) * z + (1 - coef) * alpha_s * x_pred
          sigma = torch.sqrt((1 - coef) * sigma_s**2)

          if s > 0:
              z = mu + sigma * torch.randn_like(z)
          else:
              z = mu

          # === CORRECTOR STEP (Langevin) ===
          if s > 0:
              epsilon_pred = model(z, s)
              noise = torch.randn_like(z)
              z = z - 0.5 * delta * sigma_s * epsilon_pred + \
                  torch.sqrt(delta) * sigma_s * noise

      return x_pred
  ```
]

=== Langevin Correction Step Intuition

The Langevin correction step performs a small gradient step on the score:

$ bold(z)_s arrow.l bold(z)_s - (delta / 2) sigma_s epsilon_theta (bold(z)_s) + sqrt(delta) sigma_s epsilon' $

#figure(
  ```
  Langevin Dynamics Intuition:

  Without correction:
  ┌─────────────────────────────────────────┐
  │  z_s may not exactly match the true    │
  │  marginal distribution p(z_s)          │
  │                                         │
  │  Distribution: ●●●○○○●●   (off target) │
  └─────────────────────────────────────────┘

  With Langevin correction:
  ┌─────────────────────────────────────────┐
  │  Correction moves z_s toward high-     │
  │  probability regions of p(z_s)         │
  │                                         │
  │  Before: ●●●○○○●●                      │
  │  After:  ●●●●●●●●●● (on target)        │
  └─────────────────────────────────────────┘
  ```,
  caption: [Langevin correction improves sampling accuracy]
)

== Reconstruction Guidance

This is the *key novel contribution* of the paper for conditional sampling.

=== Problem: Generating Longer/Higher-Res Videos

The model is trained on fixed-size videos (e.g., 16×64×64). How to generate:
- Longer videos (more frames)?
- Higher resolution videos?
- Videos conditioned on initial frames (video prediction)?

=== Prior Approach: Replacement Method

Previous work (Song et al., 2021) proposed the *replacement method*:

#figure(
  ```
  Replacement Method for p(x^b | x^a):

  Goal: Sample x^b conditioned on known x^a
  (e.g., x^a = first 5 frames, x^b = next 11 frames)

  Procedure:
  1. Start with z_1 = [z_1^a, z_1^b] ~ N(0, I)
  2. For each sampling step t → s:
     a. Sample z_s^a from q(z_s^a | x^a)  [exact forward process]
     b. Sample z_s^b using standard reverse process with z_t = [z_t^a, z_t^b]
     c. Replace: z_s = [z_s^a, z_s^b]

  Problem: z_s^b is updated toward E[x^b | z_t]
           But we need: E[x^b | z_t, x^a]
           The conditioning on x^a is missing!
  ```,
  caption: [Replacement method and its limitation]
)

=== Reconstruction Guidance Derivation

#block(fill: rgb("#fff3cd"), inset: 1em, radius: 4pt)[
  *Theorem 5.1 (Reconstruction Guidance)*

  For conditional sampling $p(bold(x)^b | bold(x)^a)$:

  $ EE[bold(x)^b | bold(z)_t, bold(x)^a] = EE[bold(x)^b | bold(z)_t] + (sigma_t^2 / alpha_t) nabla_(bold(z)_t^b) log q(bold(x)^a | bold(z)_t) $

  The second term corrects for the conditioning on $bold(x)^a$.
]

*Derivation*:

Using Bayes' rule:
$ p(bold(x)^a | bold(z)_t) p(bold(z)_t) = p(bold(z)_t | bold(x)^a) p(bold(x)^a) $

Taking the score (gradient of log):
$ nabla_(bold(z)_t) log p(bold(z)_t | bold(x)^a) = nabla_(bold(z)_t) log p(bold(z)_t) + nabla_(bold(z)_t) log p(bold(x)^a | bold(z)_t) $

The conditional score decomposes into:
- Unconditional score: $nabla_(bold(z)_t) log p(bold(z)_t) approx -epsilon_theta (bold(z)_t) / sigma_t$
- Guidance term: $nabla_(bold(z)_t) log p(bold(x)^a | bold(z)_t)$

=== Gaussian Approximation

We approximate $q(bold(x)^a | bold(z)_t)$ as Gaussian:

$ q(bold(x)^a | bold(z)_t) approx cal(N)(hat(bold(x))_theta^a (bold(z)_t), (sigma_t^2 / alpha_t^2) bold(I)) $

This gives:
$ nabla_(bold(z)_t^b) log q(bold(x)^a | bold(z)_t) approx -(alpha_t^2 / (2 sigma_t^2)) nabla_(bold(z)_t^b) || bold(x)^a - hat(bold(x))_theta^a (bold(z)_t) ||_2^2 $

=== Final Reconstruction Guidance Formula

#block(fill: rgb("#e8f4e8"), inset: 1em, radius: 4pt)[
  *Definition 5.1 (Reconstruction-Guided Denoising)*

  $ tilde(bold(x))_theta^b (bold(z)_t) = hat(bold(x))_theta^b (bold(z)_t) - (w_r alpha_t) / 2 nabla_(bold(z)_t^b) || bold(x)^a - hat(bold(x))_theta^a (bold(z)_t) ||_2^2 $

  where $w_r > 0$ is the *reconstruction guidance weight*.
]

=== Algorithm Implementation

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Algorithm 5.3: Reconstruction-Guided Sampling*

  ```python
  def reconstruction_guided_sample(
      model,
      x_a,           # Conditioning frames [B, T_a, H, W, C]
      num_frames_b,  # Number of frames to generate
      num_steps=256,
      guidance_weight=1.0,
  ):
      """Sample x^b conditioned on x^a using reconstruction guidance."""

      B, T_a, H, W, C = x_a.shape
      T_b = num_frames_b
      device = x_a.device

      # Initialize noise for both parts
      z_a = torch.randn(B, T_a, H, W, C, device=device)
      z_b = torch.randn(B, T_b, H, W, C, device=device)

      timesteps = torch.linspace(1, 0, num_steps + 1)

      for i in range(num_steps):
          t = timesteps[i]
          s = timesteps[i + 1]

          alpha_t, sigma_t = cosine_schedule(t)
          alpha_s, sigma_s = cosine_schedule(s)

          # === REPLACEMENT for z^a ===
          # Sample from exact forward process q(z_s^a | x^a)
          epsilon_a = torch.randn_like(x_a)
          z_a = alpha_s * x_a + sigma_s * epsilon_a

          # === RECONSTRUCTION GUIDANCE for z^b ===
          z = torch.cat([z_a, z_b], dim=1)  # [B, T_a+T_b, H, W, C]
          z.requires_grad_(True)

          # Model prediction
          x_pred = model.predict_x(z, t)
          x_pred_a = x_pred[:, :T_a]  # Reconstruction of x^a
          x_pred_b = x_pred[:, T_a:]  # Prediction of x^b

          # Reconstruction loss
          recon_loss = ((x_a - x_pred_a) ** 2).sum()

          # Gradient of reconstruction loss w.r.t. z^b
          grad_z_b = torch.autograd.grad(recon_loss, z)[0][:, T_a:]

          # Apply guidance
          x_pred_b_guided = x_pred_b - (guidance_weight * alpha_t / 2) * grad_z_b

          # Standard reverse step for z^b
          epsilon_pred_b = (z_b - alpha_t * x_pred_b_guided) / sigma_t
          # ... continue with ancestral sampling for z^b

      return x_pred_b
  ```
]

=== Reconstruction Guidance Diagram

#figure(
  ```
  Reconstruction Guidance Flow:

  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │   Known: x^a (conditioning frames)                                 │
  │   Want:  x^b (frames to generate)                                  │
  │                                                                     │
  │   z_t = [z_t^a, z_t^b]                                             │
  │          ↓                                                          │
  │   ┌─────────────────┐                                              │
  │   │  Model x̂_θ(z_t) │                                              │
  │   └────────┬────────┘                                              │
  │            ↓                                                        │
  │   [x̂^a_θ(z_t), x̂^b_θ(z_t)]                                        │
  │        ↓              ↓                                             │
  │   ┌────────────┐  ┌─────────────────────────────────────────┐      │
  │   │ Recon Loss │  │ Standard prediction                      │      │
  │   │‖x^a - x̂^a‖²│  │                                         │      │
  │   └─────┬──────┘  └──────────────────────┬──────────────────┘      │
  │         │                                 │                         │
  │         │    ∂L/∂z_t^b                    │                         │
  │         └──────────┐                      │                         │
  │                    ↓                      ↓                         │
  │   ┌─────────────────────────────────────────────────────────┐      │
  │   │                                                         │      │
  │   │   x̃^b = x̂^b - (w_r · α_t / 2) · ∂‖x^a - x̂^a‖² / ∂z_t^b │      │
  │   │                                                         │      │
  │   └─────────────────────────────────────────────────────────┘      │
  │                    ↓                                                │
  │   Use x̃^b for reverse diffusion step                              │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
  ```,
  caption: [Reconstruction guidance computation flow]
)

== Applications of Reconstruction Guidance

=== 1. Autoregressive Video Extension

Extend 16-frame model to generate 64+ frame videos:

#figure(
  ```
  Autoregressive Extension (16 frames → 64 frames):

  Step 1: Generate first block
  ┌────────────────────────────────────────────────────────┐
  │   Sample x^(1) ~ p_θ(x)     [frames 1-16]             │
  └────────────────────────────────────────────────────────┘

  Step 2: Extend with conditioning
  ┌────────────────────────────────────────────────────────┐
  │   x^a = x^(1)[last 8 frames]   [frames 9-16]          │
  │   Sample x^(2) ~ p_θ(x^b | x^a) [frames 17-24]        │
  │   Using reconstruction guidance                        │
  └────────────────────────────────────────────────────────┘

  Step 3: Continue...
  ┌────────────────────────────────────────────────────────┐
  │   x^a = x^(2)[last 8 frames]   [frames 17-24]         │
  │   Sample x^(3) ~ p_θ(x^b | x^a) [frames 25-32]        │
  └────────────────────────────────────────────────────────┘

  ...repeat until desired length
  ```,
  caption: [Autoregressive video extension]
)

=== 2. Temporal Super-Resolution

Increase frame rate (e.g., 16 frames → 64 frames at same duration):

#figure(
  ```
  Temporal Super-Resolution:

  Low frame rate:  [f1, _, _, _, f2, _, _, _, f3, ...]
  High frame rate: [f1, f', f', f', f2, f', f', f', f3, ...]

  x^a = keyframes at positions {0, 4, 8, 12, ...}
  x^b = frames to interpolate at positions {1,2,3, 5,6,7, 9,10,11, ...}

  Sample x^b conditioned on x^a using reconstruction guidance
  ```,
  caption: [Temporal super-resolution via reconstruction guidance]
)

=== 3. Spatial Super-Resolution

Upsample 64×64 → 128×128:

#block(fill: rgb("#fff3cd"), inset: 1em, radius: 4pt)[
  *Spatial Super-Resolution Formula*

  $ tilde(bold(x))_theta (bold(z)_t) = hat(bold(x))_theta (bold(z)_t) - (w_r alpha_t) / 2 nabla_(bold(z)_t) || bold(x)^a - "downsample"(hat(bold(x))_theta (bold(z)_t)) ||_2^2 $

  where:
  - $bold(x)^a$ is the low-resolution ground truth
  - $hat(bold(x))_theta (bold(z)_t)$ is the high-resolution prediction
  - "downsample" uses differentiable bilinear interpolation
]

=== 4. Combined Temporal + Spatial SR

The paper demonstrates simultaneous:
- Temporal extension: 16 frames → 64 frames
- Spatial upsampling: 64×64 → 128×128
- Frame rate increase: frameskip 4 → frameskip 1

== Comparison: Replacement vs Reconstruction Guidance

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    [*Metric*], [*Replacement*], [*Reconstruction Guidance*],
    [FVD↓], [451.45], [*136.22*],
    [FID-avg↓], [25.95], [*13.77*],
    [IS-avg↑], [7.00], [*10.30*],
    [Temporal Coherence], [Poor], [*Good*],
  ),
  caption: [Comparison of conditional sampling methods (w=2.0)]
)

The reconstruction guidance method dramatically improves temporal coherence.

== Sampling Hyperparameters

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    [*Task*], [*Sampler*], [*Steps*], [*Recon. Guidance Weight*],
    [UCF101], [Ancestral], [256], [N/A],
    [BAIR], [Langevin], [256 + 256], [50],
    [Kinetics], [Langevin], [128 + 128], [9],
    [Text-to-Video], [Ancestral], [256], [Variable],
  ),
  caption: [Sampling configurations by task]
)
