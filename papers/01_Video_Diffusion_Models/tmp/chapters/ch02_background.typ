// Chapter 2: Diffusion Model Background
#import "../template.typ": *
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.2"

#show: chapter-template.with(
  title: "Diffusion Model Background",
  subtitle: "Mathematical Foundations",
  chapter-num: 2,
)

= Diffusion Model Fundamentals <background>

== Overview of Diffusion Models

Diffusion models are a class of generative models that learn to reverse a gradual noising process. The key insight is that while adding noise is simple, a neural network can learn to denoise step by step.

#figure(
  diagram(
    spacing: (8pt, 8pt),
    node-stroke: 1pt,

    // Forward process
    node((0, 0), $bold(x)$, shape: rect, fill: green.lighten(80%), width: 1.2cm),
    node((1.5, 0), $bold(z)_0.25$, shape: rect, fill: green.lighten(70%), width: 1.2cm),
    node((3, 0), $bold(z)_0.5$, shape: rect, fill: yellow.lighten(70%), width: 1.2cm),
    node((4.5, 0), $bold(z)_0.75$, shape: rect, fill: orange.lighten(70%), width: 1.2cm),
    node((6, 0), $bold(z)_1$, shape: rect, fill: red.lighten(70%), width: 1.2cm),

    // Forward edges
    edge((0, 0), (1.5, 0), "->", label: $q$, label-side: center),
    edge((1.5, 0), (3, 0), "->", label: $q$, label-side: center),
    edge((3, 0), (4.5, 0), "->", label: $q$, label-side: center),
    edge((4.5, 0), (6, 0), "->", label: $q$, label-side: center),

    // Reverse edges
    edge((6, 0), (4.5, 0), "<-", bend: 40deg, label: $p_theta$, label-side: center),
    edge((4.5, 0), (3, 0), "<-", bend: 40deg, label: $p_theta$, label-side: center),
    edge((3, 0), (1.5, 0), "<-", bend: 40deg, label: $p_theta$, label-side: center),
    edge((1.5, 0), (0, 0), "<-", bend: 40deg, label: $p_theta$, label-side: center),

    // Labels
    node((3, 0.8), text(size: 9pt)[Forward (noising) $q$], shape: rect, stroke: none),
    node((3, -0.8), text(size: 9pt)[Reverse (denoising) $p_theta$], shape: rect, stroke: none),
  ),
  caption: [Forward process adds noise; reverse process learns to denoise]
)

== Continuous-Time Formulation

The VDM paper uses a *continuous-time* diffusion formulation, which offers several advantages over discrete-time:

#keypoint(title: "Why Continuous Time?")[
  - More flexible noise scheduling
  - Easier theoretical analysis
  - Better sampling algorithms (ODE/SDE solvers)
  - Natural extension to variable number of steps
]

== The Forward Process

#definition("Forward Diffusion Process")[
  The forward process gradually corrupts data $bold(x)$ into noise:
  $ q(bold(z)_t | bold(x)) = cal(N)(bold(z)_t; alpha_t bold(x), sigma_t^2 bold(I)) $ <forward>
  where:
  - $bold(z)_t$: noisy latent at time $t in [0, 1]$
  - $alpha_t$: signal scaling coefficient (decreases with $t$)
  - $sigma_t$: noise standard deviation (increases with $t$)
  - $bold(I)$: identity matrix
]

#mathblock(title: "Reparametrization Trick")[
  We can sample $bold(z)_t$ directly without iterating:
  $ bold(z)_t = alpha_t bold(x) + sigma_t epsilon, quad epsilon tilde cal(N)(0, bold(I)) $ <reparam>

  This is crucial for efficient training!
]

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *

    // Axes
    line((0, 0), (8, 0), stroke: black, mark: (end: "stealth"))
    line((0, 0), (0, 4), stroke: black, mark: (end: "stealth"))
    content((8.3, 0), $t$)
    content((0, 4.3), text(size: 9pt)[Coefficient])

    // Alpha curve (decreasing)
    bezier((0, 3.5), (4, 2), (8, 0.2), stroke: blue + 2pt)
    content((1, 3.8), text(fill: blue, size: 9pt)[$alpha_t$])

    // Sigma curve (increasing)
    bezier((0, 0.2), (4, 1.5), (8, 3.5), stroke: red + 2pt)
    content((7, 3.8), text(fill: red, size: 9pt)[$sigma_t$])

    // Time markers
    content((0, -0.3), $0$)
    content((8, -0.3), $1$)

    // Annotations
    content((2, -0.8), text(size: 8pt)[Clean data])
    content((6, -0.8), text(size: 8pt)[Pure noise])
  }),
  caption: [Signal coefficient $alpha_t$ decreases while noise coefficient $sigma_t$ increases]
)

== Signal-to-Noise Ratio

#definition("Log Signal-to-Noise Ratio")[
  The log-SNR is defined as:
  $ lambda_t = log (alpha_t^2 / sigma_t^2) $ <logsnr>
  This quantity decreases monotonically from $+infinity$ (clean data) to $-infinity$ (pure noise).
]

#theorem("SNR Properties")[
  For a well-defined diffusion process:
  1. $lambda_t$ is strictly monotonically decreasing
  2. $lim_(t arrow 0) lambda_t = +infinity$ (pure signal)
  3. $lim_(t arrow 1) lambda_t = -infinity$ (pure noise)
  4. The process is uniquely defined by the SNR schedule
]

#warning(title: "Practical SNR Clipping")[
  In practice, the paper clips log-SNR to $[-20, 20]$:
  $ lambda_t in [-20, 20] $
  This prevents numerical instability at extreme values.
]

== Variance-Preserving vs. Variance-Exploding

The VDM uses a *variance-preserving* formulation:

#figure(
  tablex(
    columns: (auto, 1fr, 1fr),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Property*], [*Variance Preserving (VP)*], [*Variance Exploding (VE)*],
    hlinex(),
    [Constraint], [$alpha_t^2 + sigma_t^2 = 1$], [No constraint],
    [At $t=0$], [$alpha_0 = 1, sigma_0 = 0$], [$sigma_0 = 0$],
    [At $t=1$], [$alpha_1 approx 0, sigma_1 approx 1$], [$sigma_1 arrow infinity$],
    [Used in], [*VDM*, DDPM, DDIM], [Score SDE],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Comparison of variance-preserving and variance-exploding formulations]
)

#mathblock(title: "Variance-Preserving Constraint")[
  For VP diffusion:
  $ alpha_t^2 + sigma_t^2 = 1 $
  This ensures $"Var"(bold(z)_t) approx "Var"(bold(x))$ throughout the process (assuming unit variance data).
]

== Markov Property

#theorem("Transition Distribution")[
  The forward process satisfies the Markov property. For $s < t$:
  $ q(bold(z)_t | bold(z)_s) = cal(N)(bold(z)_t; (alpha_t / alpha_s) bold(z)_s, sigma_(t|s)^2 bold(I)) $ <transition>
  where:
  $ sigma_(t|s)^2 = (1 - e^(lambda_t - lambda_s)) sigma_t^2 $
]

#remark("Derivation Intuition")[
  This follows from the fact that:
  $ bold(z)_t = (alpha_t / alpha_s) bold(z)_s + sigma_(t|s) epsilon' $
  where $epsilon'$ is independent of $bold(z)_s$.
]

== The Reverse Process

#definition("Reverse (Generative) Process")[
  The reverse process learns to denoise:
  $ p_theta (bold(z)_s | bold(z)_t) = cal(N)(bold(z)_s; tilde(mu)_(s|t)(bold(z)_t, bold(x)_theta), tilde(sigma)_(s|t)^2 bold(I)) $
  where $bold(x)_theta$ is the network's prediction of the clean data.
]

#theorem("Posterior Mean")[
  The optimal posterior mean is:
  $ tilde(mu)_(s|t) = e^(lambda_t - lambda_s) (alpha_s / alpha_t) bold(z)_t + (1 - e^(lambda_t - lambda_s)) alpha_s bold(x)_theta $ <posterior-mean>
]

#figure(
  diagram(
    spacing: (15pt, 15pt),
    node-stroke: 1pt,

    // Network
    node((0, 0), $bold(z)_t$, shape: rect, fill: orange.lighten(80%), width: 1.5cm),
    node((2, 0), [Neural Net\ $theta$], shape: rect, fill: blue.lighten(80%), width: 2.5cm, height: 1.2cm),
    node((4, 0), $hat(bold(x))_theta$, shape: rect, fill: green.lighten(80%), width: 1.5cm),

    // Arrows
    edge((0, 0), (2, 0), "->"),
    edge((2, 0), (4, 0), "->"),

    // Timestep input
    node((2, 1), $t$, shape: circle, fill: gray.lighten(80%), radius: 0.4cm),
    edge((2, 1), (2, 0), "->"),
  ),
  caption: [Neural network predicts clean data from noisy input and timestep]
)

== Parameterization Choices

The network can predict different targets. The paper uses *ε-prediction*:

#figure(
  tablex(
    columns: (auto, auto, 1fr),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Parameterization*], [*Network Output*], [*Conversion to $bold(x)$*],
    hlinex(),
    [$bold(x)$-prediction], [$hat(bold(x))_theta (bold(z)_t, t)$], [Direct],
    [*ε-prediction*], [$epsilon_theta (bold(z)_t, t)$], [$hat(bold(x)) = (bold(z)_t - sigma_t epsilon_theta) / alpha_t$],
    [$bold(v)$-prediction], [$bold(v)_theta (bold(z)_t, t)$], [$hat(bold(x)) = alpha_t bold(z)_t - sigma_t bold(v)_theta$],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Different parameterization choices]
)

#keypoint(title: "Why ε-prediction?")[
  The paper uses *ε-prediction* because:
  1. Stable training across all noise levels
  2. Well-understood from DDPM literature
  3. Natural connection to score matching: $nabla_(bold(z)_t) log p(bold(z)_t) = -epsilon / sigma_t$
]

#mathblock(title: "Conversion Formula")[
  Given $epsilon_theta$ prediction, recover $hat(bold(x))_theta$:
  $ hat(bold(x))_theta = (bold(z)_t - sigma_t epsilon_theta (bold(z)_t, t)) / alpha_t $ <eps-to-x>
]

== Training Objective

#definition("ε-Prediction Training Loss")[
  $ cal(L) = EE_(bold(x), epsilon tilde cal(N)(0, I), t tilde cal(U)(0,1)) [ omega(lambda_t) ||epsilon_theta (bold(z)_t, t) - epsilon||_2^2 ] $ <loss>
  where $bold(z)_t = alpha_t bold(x) + sigma_t epsilon$ and $omega(lambda_t)$ is a weighting function.
]

#figure(
  diagram(
    spacing: (10pt, 12pt),
    node-stroke: 1pt,

    // Data flow
    node((0, 0), $bold(x)$, shape: rect, fill: green.lighten(80%), width: 1.2cm),
    node((1.5, 0), [Sample\ $epsilon, t$], shape: rect, fill: yellow.lighten(80%), width: 1.8cm),
    node((3, 0), [$bold(z)_t = alpha_t bold(x)$\ $+ sigma_t epsilon$], shape: rect, fill: orange.lighten(80%), width: 2cm),
    node((5, 0), [Network\ $epsilon_theta$], shape: rect, fill: blue.lighten(80%), width: 1.8cm),
    node((7, 0), [Loss\ $||epsilon_theta - epsilon||^2$], shape: rect, fill: red.lighten(80%), width: 2cm),

    edge((0, 0), (1.5, 0), "->"),
    edge((1.5, 0), (3, 0), "->"),
    edge((3, 0), (5, 0), "->"),
    edge((5, 0), (7, 0), "->"),
  ),
  caption: [Training pipeline: add noise, predict noise, compute MSE loss]
)

#theorem("Loss Weighting")[
  For the simple loss ($omega(lambda_t) = 1$), the objective is equivalent to:
  $ cal(L)_"simple" = EE_(bold(x), epsilon, t) [||epsilon_theta - epsilon||_2^2] $
  This uniformly weights all timesteps.
]

== Classifier-Free Guidance (CFG)

#definition("Classifier-Free Guidance")[
  For conditional generation $p(bold(x) | bold(c))$, CFG combines conditional and unconditional predictions:
  $ tilde(epsilon)_theta (bold(z)_t, bold(c)) = (1 + w) epsilon_theta (bold(z)_t, bold(c)) - w epsilon_theta (bold(z)_t, emptyset) $ <cfg>
  where:
  - $bold(c)$: conditioning information (e.g., text, class label)
  - $emptyset$: null/unconditional embedding
  - $w$: guidance weight (typically 1-15)
]

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *

    // Axes
    line((0, 0), (6, 0), stroke: black, mark: (end: "stealth"))
    line((0, 0), (0, 4), stroke: black, mark: (end: "stealth"))
    content((6.5, 0), $w$)
    content((-0.3, 4.3), text(size: 9pt)[Quality])

    // Quality curve
    bezier((0, 1), (3, 3.5), (6, 3.2), stroke: blue + 2pt)

    // Diversity curve
    bezier((0, 3), (3, 2), (6, 0.5), stroke: red + 2pt)

    // Labels
    content((5, 3.5), text(fill: blue, size: 8pt)[Quality])
    content((5, 0.8), text(fill: red, size: 8pt)[Diversity])

    // Optimal zone
    rect((2, 0), (4, 4), fill: green.transparentize(90%), stroke: none)
    content((3, 0.3), text(fill: green.darken(20%), size: 7pt)[Optimal])
  }),
  caption: [Trade-off between quality (higher $w$) and diversity (lower $w$)]
)

#warning(title: "Training with CFG")[
  During training, randomly drop conditioning with probability $p_"uncond"$ (typically 10-20%):
  ```python
  if random() < p_uncond:
      c = null_embedding  # Train unconditional
  else:
      c = text_embedding  # Train conditional
  ```
]

== Extension to Video

For video $bold(x) in RR^(T times H times W times C)$, all equations extend naturally:

#mathblock(title: "Video Forward Process")[
  $ q(bold(z)_t | bold(x)) = cal(N)(bold(z)_t; alpha_t bold(x), sigma_t^2 bold(I)) $
  where $bold(x), bold(z)_t in RR^(T times H times W times C)$

  The same noise is applied independently to each spatiotemporal location.
]

#keypoint(title: "Key Insight")[
  The mathematical framework is *identical* for images and videos. The difference lies entirely in the *network architecture* (Chapter 3) and how it processes the temporal dimension.
]

#pagebreak()

== Summary

#figure(
  tablex(
    columns: (auto, 1fr),
    align: (center, left),
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Concept*], [*Key Formula/Description*],
    hlinex(),
    [Forward], [$q(bold(z)_t | bold(x)) = cal(N)(alpha_t bold(x), sigma_t^2 bold(I))$],
    [Reparametrization], [$bold(z)_t = alpha_t bold(x) + sigma_t epsilon$],
    [Log-SNR], [$lambda_t = log(alpha_t^2 / sigma_t^2)$, monotonically decreasing],
    [Transition], [$q(bold(z)_t | bold(z)_s) = cal(N)((alpha_t/alpha_s) bold(z)_s, sigma_(t|s)^2 bold(I))$],
    [Training], [$cal(L) = EE[||epsilon_theta(bold(z)_t, t) - epsilon||^2]$],
    [CFG], [$tilde(epsilon) = (1+w) epsilon_theta(bold(z)_t, bold(c)) - w epsilon_theta(bold(z)_t, emptyset)$],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Chapter 2 Summary: Key mathematical concepts]
)

== Code Connection Preview

The mathematical concepts map directly to code:

#codeblock(title: "Python: Forward Process")[
  ```python
  def forward_process(x, t, schedule):
      """Add noise to clean data x at timestep t."""
      alpha_t, sigma_t = schedule(t)
      eps = torch.randn_like(x)
      z_t = alpha_t * x + sigma_t * eps
      return z_t, eps
  ```
]

#codeblock(title: "Python: Training Step")[
  ```python
  def train_step(model, x, schedule):
      """Single training iteration."""
      t = torch.rand(x.shape[0])  # Random timesteps
      z_t, eps = forward_process(x, t, schedule)
      eps_pred = model(z_t, t)
      loss = F.mse_loss(eps_pred, eps)
      return loss
  ```
]
