// Video Diffusion Models - Comprehensive Technical Analysis
// Complete standalone document with Typst packages

#import "@preview/ctheorems:1.1.3": *
#import "@preview/showybox:2.0.3": showybox

#show: thmrules

#let definition = thmbox("definition", "Definition", fill: rgb("#e8f4e8"))
#let theorem = thmbox("theorem", "Theorem", fill: rgb("#fff3cd"))
#let lemma = thmbox("lemma", "Lemma", fill: rgb("#e3f2fd"))

#set document(
  title: "Video Diffusion Models: Complete Technical Analysis",
  author: "T2V-Paper2Slide",
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 9pt)
      _Video Diffusion Models - Technical Analysis_
      #h(1fr)
      #counter(page).display()
    ]
  },
)

#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1.1")
#set math.equation(numbering: "(1)")
#set par(justify: true)

#show raw.where(block: true): it => {
  block(fill: luma(245), inset: 10pt, radius: 4pt, width: 100%, it)
}

// ============================================================================
// TITLE PAGE
// ============================================================================

#align(center)[
  #v(3cm)
  #text(size: 28pt, weight: "bold")[Video Diffusion Models]
  #v(0.5cm)
  #text(size: 18pt)[Complete Technical Analysis]
  #v(0.3cm)
  #text(size: 14pt)[for From-Scratch Implementation]
  #v(1cm)
  #text(size: 14pt, style: "italic")[
    Based on: Ho et al. "Video Diffusion Models" (arXiv:2204.03458)
  ]
  #v(0.3cm)
  #text(size: 12pt)[Google Research, April 2022]
  #v(2cm)

  #showybox(
    frame: (border-color: blue.darken(20%), title-color: blue.lighten(80%), body-color: blue.lighten(95%)),
    title-style: (color: black, weight: "bold"),
    title: "Abstract"
  )[
    This document provides a comprehensive technical analysis of the Video Diffusion Models paper,
    the first work to successfully apply diffusion models to video generation. We cover mathematical
    foundations, architectural details, training procedures, and sampling algorithms for from-scratch implementation.
  ]

  #v(1.5cm)
  #text(size: 11pt)[*Document Version*: 1.0 | *Last Updated*: 2024]
]

#pagebreak()

#outline(title: [Table of Contents], indent: 2em, depth: 2)

#pagebreak()

// ============================================================================
// CHAPTER 1: INTRODUCTION
// ============================================================================

= Introduction

== Motivation

Video generation is one of the most challenging tasks in generative modeling, requiring both *spatial coherence* (realistic frames) and *temporal coherence* (plausible motion).

#showybox(
  frame: (border-color: orange, body-color: orange.lighten(95%)),
  title: "Challenge: Memory Requirements"
)[
  A 16-frame video at 64×64 resolution:
  $ "Data" = 16 times 64 times 64 times 3 = 196,608 "values" $
  This is *16× more* than a single image!
]

== Key Contributions

#figure(
  table(
    columns: (auto, 1fr),
    inset: 10pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(230) } else { white },
    [*\#*], [*Contribution*],
    [1], [First video diffusion model with 3D U-Net],
    [2], [Space-time factorized attention (1000× speedup)],
    [3], [Joint image-video training (3.4× quality improvement)],
    [4], [Reconstruction guidance for video extension],
    [5], [State-of-the-art on UCF101, BAIR, Kinetics-600],
  ),
  caption: [Summary of paper contributions]
)

#pagebreak()

// ============================================================================
// CHAPTER 2: BACKGROUND
// ============================================================================

= Diffusion Model Background

== Forward Process

#definition("Forward Process")[
  The forward process gradually adds noise to data $bold(x)$:
  $ q(bold(z)_t | bold(x)) = cal(N)(bold(z)_t; alpha_t bold(x), sigma_t^2 bold(I)) $
  where $alpha_t, sigma_t$ define a noise schedule with decreasing log-SNR $lambda_t = log(alpha_t^2 / sigma_t^2)$.
]

#theorem("Markov Property")[
  $ q(bold(z)_t | bold(z)_s) = cal(N)(bold(z)_t; (alpha_t / alpha_s) bold(z)_s, sigma_(t|s)^2 bold(I)) $
  where $sigma_(t|s)^2 = (1 - e^(lambda_t - lambda_s)) sigma_t^2$ for $s < t$.
]

== Training Objective

#definition("ε-Prediction Loss")[
  $ cal(L) = EE_(bold(x), epsilon, t) [ || epsilon_theta (bold(z)_t) - epsilon ||_2^2 ] $
  where $bold(z)_t = alpha_t bold(x) + sigma_t epsilon$ and $epsilon tilde cal(N)(0, I)$.
]

== Classifier-Free Guidance

#definition("Guidance Formula")[
  $ tilde(epsilon)_theta (bold(z)_t, bold(c)) = (1 + w) epsilon_theta (bold(z)_t, bold(c)) - w epsilon_theta (bold(z)_t) $
  Higher $w$ → better quality, lower diversity.
]

#pagebreak()

// ============================================================================
// CHAPTER 3: ARCHITECTURE
// ============================================================================

= 3D U-Net Architecture

== Overview

#showybox(
  frame: (border-color: blue, body-color: blue.lighten(95%)),
  title: "Model Input/Output"
)[
  *Input*: Noisy video $bold(z)_t in RR^(T times H times W times C)$, conditioning $bold(c)$, timestep $t$ \
  *Output*: Predicted noise $epsilon_theta in RR^(T times H times W times C)$
]

== Space-Time Factorization

The key innovation is *factorizing* spatial and temporal processing:

```
Block Structure:
┌─────────────────────────────────────────────────────┐
│   Input: x ∈ ℝ^(T×H×W×C)                           │
│          ↓                                          │
│   Spatial Conv (1×3×3) - process each frame        │
│          ↓                                          │
│   Spatial Attention - attend over (H×W)            │
│          ↓                                          │
│   Temporal Attention - attend over T               │
│   + Relative position embeddings                   │
│          ↓                                          │
│   Output: y ∈ ℝ^(T×H×W×C')                         │
└─────────────────────────────────────────────────────┘
```

#showybox(
  frame: (border-color: green, body-color: green.lighten(95%)),
  title: "Computational Advantage"
)[
  *Full 3D*: $O(T^2 H^2 W^2)$ \
  *Factorized*: $O(T H^2 W^2 + H W T^2)$

  For $T=16, H=W=64$: *1000× reduction!*
]

== Architecture Details

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(230) } else { white },
    [*Stage*], [*Resolution*], [*Channels*], [*Shape*],
    [Input], [64×64], [3], [16×64×64×3],
    [Enc Block 1], [32×32], [256], [16×32×32×256],
    [Enc Block 2], [16×16], [512], [16×16×16×512],
    [Enc Block 3], [8×8], [1024], [16×8×8×1024],
    [Bottleneck], [4×4], [2048], [16×4×4×2048],
    [Dec + Skip], [...], [...], [Symmetric],
    [Output], [64×64], [3], [16×64×64×3],
  ),
  caption: [U-Net tensor dimensions]
)

#pagebreak()

// ============================================================================
// CHAPTER 4: TRAINING
// ============================================================================

= Training Methodology

== Cosine Noise Schedule

#definition("Cosine Schedule")[
  $ alpha_t^2 = cos^2 ((t + s) / (1 + s) dot pi / 2) $
  where $s = 0.008$ is a small offset. Log-SNR clipped to $[-20, 20]$.
]

== Joint Image-Video Training

#showybox(
  frame: (border-color: purple, body-color: purple.lighten(95%)),
  title: "Key Result"
)[
  Adding 8 independent images per video batch:
  - *FVD*: 205 → 61 (3.4× improvement)
  - *FID*: 37 → 15 (2.5× improvement)
]

The attention mask ensures video frames interact with each other, while images only attend to themselves.

== Hyperparameters

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 6pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(230) } else { white },
    [*Param*], [*UCF101*], [*BAIR*], [*Kinetics*], [*Text2Video*],
    [Channels], [256], [128], [256], [256],
    [Mult], [1,2,4,8], [1,2,3,4], [1,2,4,8], [1,2,4,8],
    [LR], [3e-4], [2e-4], [2e-4], [3e-4],
    [Steps], [60K], [660K], [220K], [700K],
    [Hardware], [128 TPU], [128 TPU], [256 TPU], [128 TPU],
  ),
  caption: [Training configurations]
)

#pagebreak()

// ============================================================================
// CHAPTER 5: SAMPLING
// ============================================================================

= Sampling & Reconstruction Guidance

== Standard Ancestral Sampling

Starting from $bold(z)_1 tilde cal(N)(0, I)$:
$ bold(z)_s = tilde(mu)_(s|t)(bold(z)_t, hat(bold(x))_theta) + tilde(sigma)_(s|t) epsilon $

== Reconstruction Guidance (Key Innovation)

#theorem("Reconstruction Guidance")[
  For conditional sampling $p(bold(x)^b | bold(x)^a)$:
  $ tilde(bold(x))_theta^b = hat(bold(x))_theta^b - (w_r alpha_t) / 2 nabla_(bold(z)_t^b) || bold(x)^a - hat(bold(x))_theta^a ||_2^2 $
]

#showybox(
  frame: (border-color: red, body-color: red.lighten(95%)),
  title: "Why is this needed?"
)[
  The naive replacement method gives $EE[bold(x)^b | bold(z)_t]$, but we need $EE[bold(x)^b | bold(z)_t, bold(x)^a]$.

  Reconstruction guidance adds the missing conditioning gradient!
]

== Applications

1. *Autoregressive Extension*: Generate 64+ frames from 16-frame model
2. *Temporal Super-Resolution*: Increase frame rate (interpolation)
3. *Spatial Super-Resolution*: 64×64 → 128×128

#pagebreak()

// ============================================================================
// CHAPTER 6: EXPERIMENTS
// ============================================================================

= Experiments

== UCF101 (Unconditional Generation)

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 10pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(230) } else if y == 5 { green.lighten(80%) } else { white },
    [*Method*], [*FID↓*], [*IS↑*],
    [MoCoGAN], [26998], [12.42],
    [TGAN-v2], [3431], [28.87],
    [DVD-GAN], [--], [32.97],
    [VideoGPT], [--], [24.69],
    [*Video Diffusion*], [*295*], [*57*],
  ),
  caption: [UCF101: 11× FID improvement, IS approaching real data (60.2)]
)

== Video Prediction Results

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(230) } else { white },
    [*Dataset*], [*Prev. SOTA*], [*VDM*],
    [BAIR (FVD↓)], [86.9], [*66.9*],
    [Kinetics (FVD↓)], [25.4], [*16.2*],
    [Kinetics (IS↑)], [12.5], [*15.6*],
  ),
  caption: [Video prediction benchmarks]
)

== Reconstruction Guidance vs Replacement

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    stroke: 0.5pt,
    fill: (x, y) => if y == 0 { luma(230) } else if y == 2 { green.lighten(80%) } else { white },
    [*Method*], [*FVD↓*], [*FID↓*], [*IS↑*],
    [Replacement], [451], [26], [7.0],
    [*Recon. Guidance*], [*136*], [*14*], [*10.3*],
  ),
  caption: [3.3× FVD improvement with reconstruction guidance]
)

#pagebreak()

// ============================================================================
// CHAPTER 7: IMPLEMENTATION
// ============================================================================

= Implementation Guide

== Core Components

=== 1. Noise Schedule

```python
class CosineSchedule:
    def __init__(self, s=0.008):
        self.s = s

    def __call__(self, t):
        f = torch.cos((t + self.s) / (1 + self.s) * pi / 2) ** 2
        f0 = math.cos(self.s / (1 + self.s) * pi / 2) ** 2
        alpha = torch.sqrt(f / f0)
        sigma = torch.sqrt(1 - f / f0)
        return alpha, sigma
```

=== 2. Spatial Attention

```python
class SpatialAttn(nn.Module):
    def forward(self, x):  # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        out = self.attention(x)  # Attend over H*W
        return rearrange(out, '(b t) (h w) c -> b c t h w',
                        b=B, t=T, h=H, w=W)
```

=== 3. Temporal Attention

```python
class TemporalAttn(nn.Module):
    def forward(self, x):  # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b h w) t c')
        out = self.attention(x) + self.rel_pos_bias  # Over T
        return rearrange(out, '(b h w) t c -> b c t h w',
                        b=B, h=H, w=W)
```

=== 4. Training Step

```python
def train_step(model, videos, optimizer):
    t = torch.rand(B)
    alpha, sigma = schedule(t)
    eps = torch.randn_like(videos)
    z_t = alpha * videos + sigma * eps

    eps_pred = model(z_t, t)
    loss = F.mse_loss(eps_pred, eps)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
```

#pagebreak()

= References

+ Ho et al. "Video Diffusion Models." arXiv:2204.03458, 2022.
+ Ho et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
+ Nichol & Dhariwal. "Improved DDPM." ICML 2021.
+ Song et al. "Score-Based Generative Modeling through SDEs." ICLR 2021.
+ Ho & Salimans. "Classifier-Free Diffusion Guidance." NeurIPS Workshop 2021.
