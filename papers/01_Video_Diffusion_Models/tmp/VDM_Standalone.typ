// Video Diffusion Models - Complete Technical Analysis (Standalone)
// All content in single file for reliable compilation

#import "@preview/ctheorems:1.1.3": *
#import "@preview/showybox:2.0.3": showybox
#import "@preview/tablex:0.0.9": tablex, rowspanx, colspanx, hlinex, vlinex

#show: thmrules.with(qed-symbol: $square$)

#let definition = thmbox("definition", "Definition", fill: rgb("#e8f4e8"))
#let theorem = thmbox("theorem", "Theorem", fill: rgb("#fff3cd"))
#let lemma = thmbox("lemma", "Lemma", fill: rgb("#e3f2fd"))
#let remark = thmbox("remark", "Remark", fill: rgb("#fff8e1"))
#let algorithm = thmbox("algorithm", "Algorithm", fill: rgb("#fce4ec"))

#let keypoint(title: "Key Point", body) = showybox(
  frame: (border-color: blue.darken(20%), title-color: blue.lighten(80%), body-color: blue.lighten(95%)),
  title-style: (color: black, weight: "bold"),
  title: title,
  body
)

#let warning(title: "Important", body) = showybox(
  frame: (border-color: orange.darken(20%), title-color: orange.lighten(80%), body-color: orange.lighten(95%)),
  title-style: (color: black, weight: "bold"),
  title: title,
  body
)

#let innovation(title: "Innovation", body) = showybox(
  frame: (border-color: green.darken(20%), title-color: green.lighten(80%), body-color: green.lighten(95%)),
  title-style: (color: black, weight: "bold"),
  title: title,
  body
)

#let mathblock(title: "Formula", body) = showybox(
  frame: (border-color: purple.darken(20%), title-color: purple.lighten(80%), body-color: purple.lighten(95%)),
  title-style: (color: black, weight: "bold"),
  title: title,
  body
)

#let codeblock(title: "Code", body) = showybox(
  frame: (border-color: gray.darken(20%), title-color: gray.lighten(80%), body-color: gray.lighten(95%)),
  title-style: (color: black, weight: "bold"),
  title: title,
  body
)

#set document(
  title: "Video Diffusion Models: Complete Technical Analysis",
  author: "T2V-Paper2Slide",
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2.5cm),
  header: context {
    if counter(page).get().first() > 2 [
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
  #v(2cm)
  #text(size: 32pt, weight: "bold", fill: blue.darken(20%))[Video Diffusion Models]
  #v(0.5cm)
  #text(size: 20pt)[Complete Technical Analysis]
  #v(0.3cm)
  #text(size: 14pt, style: "italic")[For From-Scratch Implementation]
  #v(1.5cm)

  #showybox(
    frame: (border-color: blue.darken(20%), title-color: blue.lighten(85%), body-color: blue.lighten(95%)),
    title-style: (color: black, weight: "bold"),
    title: "Paper Reference"
  )[
    *Ho, Salimans, Gritsenko, Chan, Norouzi, Fleet* \
    "Video Diffusion Models" - arXiv:2204.03458, April 2022 \
    Google Research, Brain Team
  ]
  #v(2cm)
  #text(size: 11pt)[*T2V-Paper2Slide* | Document Version 2.0]
]

#pagebreak()
#outline(title: [Table of Contents], indent: 2em, depth: 2)
#pagebreak()

// ============================================================================
// CHAPTER 1: INTRODUCTION
// ============================================================================

= Introduction

== Paper Overview

#keypoint(title: "Paper Information")[
  *Title*: Video Diffusion Models \
  *Authors*: Ho, Salimans, Gritsenko, Chan, Norouzi, Fleet \
  *Affiliation*: Google Research \
  *ArXiv*: 2204.03458 (April 2022) \
  *Significance*: First video diffusion model
]

== Key Contributions

#innovation(title: "Five Major Contributions")[
  1. *3D U-Net Architecture*: Extended image diffusion to video
  2. *Space-Time Factorized Attention*: 1000× speedup
  3. *Joint Image-Video Training*: 3.4× FVD improvement
  4. *Reconstruction Guidance*: Novel conditional generation
  5. *State-of-the-Art Results*: Best on UCF101, BAIR, Kinetics
]

== Challenge

#warning(title: "Memory Requirements")[
  A 16-frame video at 64×64:
  $ "Data" = 16 times 64 times 64 times 3 = 196,608 "values" $
  *16× more* than a single image!
]

#pagebreak()

// ============================================================================
// CHAPTER 2: BACKGROUND
// ============================================================================

= Diffusion Model Background

== Forward Process

#definition("Forward Diffusion")[
  $ q(bold(z)_t | bold(x)) = cal(N)(bold(z)_t; alpha_t bold(x), sigma_t^2 bold(I)) $
  where $alpha_t$ decreases and $sigma_t$ increases with $t$.
]

#mathblock(title: "Reparametrization")[
  $ bold(z)_t = alpha_t bold(x) + sigma_t epsilon, quad epsilon tilde cal(N)(0, bold(I)) $
]

== Log-SNR

#definition("Log Signal-to-Noise Ratio")[
  $ lambda_t = log(alpha_t^2 / sigma_t^2) $
  Decreases monotonically from $+infinity$ to $-infinity$.
]

== Training Objective

#definition("ε-Prediction Loss")[
  $ cal(L) = EE_(bold(x), epsilon, t) [||epsilon_theta(bold(z)_t, t) - epsilon||_2^2] $
]

== Classifier-Free Guidance

#definition("CFG Formula")[
  $ tilde(epsilon)_theta = (1 + w) epsilon_theta(bold(z)_t, bold(c)) - w epsilon_theta(bold(z)_t, emptyset) $
]

#pagebreak()

// ============================================================================
// CHAPTER 3: ARCHITECTURE
// ============================================================================

= 3D U-Net Architecture

== Overview

#keypoint(title: "Model I/O")[
  *Input*: Noisy video $bold(z)_t in RR^(T times H times W times C)$, timestep $t$ \
  *Output*: Predicted noise $epsilon_theta in RR^(T times H times W times C)$
]

== Space-Time Factorization

#innovation(title: "Computational Breakthrough")[
  *Full 3D attention*: $O(T^2 H^2 W^2)$ \
  *Factorized*: $O(T H^2 W^2 + H W T^2)$

  For $T=16, H=W=64$: *1000× speedup!*
]

== Block Structure

Each block contains:
1. Spatial Convolution $(1 times 3 times 3)$
2. Spatial Attention over $(H times W)$
3. Temporal Attention over $T$
4. Residual Connection

== Spatial Attention

#codeblock(title: "PyTorch: Spatial Attention")[
```python
class SpatialAttn(nn.Module):
    def forward(self, x):  # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        out = self.attention(x)  # Attend over H*W
        return rearrange(out, '(b t) (h w) c -> b c t h w',
                        b=B, t=T, h=H, w=W)
```
]

== Temporal Attention

#codeblock(title: "PyTorch: Temporal Attention")[
```python
class TemporalAttn(nn.Module):
    def forward(self, x):  # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b h w) t c')
        out = self.attention(x) + self.rel_pos_bias
        return rearrange(out, '(b h w) t c -> b c t h w',
                        b=B, h=H, w=W)
```
]

== Architecture Configuration

#figure(
  tablex(
    columns: (auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Parameter*], [*Value*],
    hlinex(),
    [Base Channels], [256],
    [Channel Mults], [(1, 2, 4, 8)],
    [ResBlocks/Stage], [2],
    [Attention Heads], [8],
    [Frames], [16],
    [Resolution], [64×64],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Default configuration]
)

#pagebreak()

// ============================================================================
// CHAPTER 4: TRAINING
// ============================================================================

= Training Methodology

== Cosine Schedule

#definition("Cosine Schedule")[
  $ alpha_t^2 = cos^2((t + s)/(1 + s) dot pi/2) $
  where $s = 0.008$. Log-SNR clipped to $[-20, 20]$.
]

== Joint Image-Video Training

#innovation(title: "Key Result")[
  Adding 8 images per video batch:
  - *FVD*: 205 → 61 (3.4× improvement)
  - *FID*: 37 → 15 (2.5× improvement)
]

== Training Configuration

#figure(
  tablex(
    columns: (auto, auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Parameter*], [*UCF101*], [*BAIR*], [*Kinetics*],
    hlinex(),
    [Learning Rate], [3e-4], [2e-4], [2e-4],
    [Batch Size], [256], [256], [512],
    [Steps], [60K], [660K], [220K],
    [Hardware], [128 TPU], [128 TPU], [256 TPU],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Training hyperparameters]
)

== Training Loop

#codeblock(title: "PyTorch: Training Step")[
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
]

#pagebreak()

// ============================================================================
// CHAPTER 5: SAMPLING
// ============================================================================

= Sampling & Reconstruction Guidance

== Ancestral Sampling

#theorem("Posterior Mean")[
  $ tilde(mu)_(s|t) = e^(lambda_t - lambda_s)(alpha_s/alpha_t)bold(z)_t + (1 - e^(lambda_t - lambda_s))alpha_s hat(bold(x))_theta $
]

== DDIM Sampling

#definition("DDIM Update")[
  $ bold(z)_s = alpha_s hat(bold(x))_theta + sigma_s ((bold(z)_t - alpha_t hat(bold(x))_theta)/sigma_t) $
  Deterministic, 50-100 steps sufficient.
]

== Reconstruction Guidance

#theorem("Reconstruction Guidance")[
  For $p(bold(x)^b | bold(x)^a)$:
  $ tilde(bold(x))_theta^b = hat(bold(x))_theta^b - (w_r alpha_t)/2 nabla_(bold(z)_t^b)||bold(x)^a - hat(bold(x))_theta^a||_2^2 $
]

#warning(title: "Why Needed?")[
  Naive replacement gives $EE[bold(x)^b | bold(z)_t]$, not $EE[bold(x)^b | bold(z)_t, bold(x)^a]$.
  Reconstruction guidance adds the missing conditioning gradient!
]

== Applications

1. *Video Prediction*: Generate future from past frames
2. *Autoregressive Extension*: Generate 64+ frame videos
3. *Temporal Super-Resolution*: Double frame rate
4. *Spatial Super-Resolution*: 64→128 pixels

#pagebreak()

// ============================================================================
// CHAPTER 6: EXPERIMENTS
// ============================================================================

= Experiments

== UCF101 Results

#figure(
  tablex(
    columns: (auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Method*], [*FID↓*], [*IS↑*],
    hlinex(),
    [MoCoGAN], [26998], [12.42],
    [TGAN-v2], [3431], [28.87],
    [DVD-GAN], [--], [32.97],
    [DIGAN], [577], [30.24],
    [*VDM*], [*295*], [*57.00*],
    hlinex(),
    [Real Data], [--], [60.20],
    hlinex(stroke: 1.5pt),
  ),
  caption: [UCF101: VDM IS approaches real data]
)

== Video Prediction

#figure(
  tablex(
    columns: (auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Dataset*], [*Prev. SOTA*], [*VDM*],
    hlinex(),
    [BAIR (FVD↓)], [86.9], [*66.9*],
    [Kinetics (FVD↓)], [25.4], [*16.2*],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Video prediction benchmarks]
)

== Reconstruction Guidance Ablation

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
  caption: [3.3× FVD improvement with reconstruction guidance]
)

#pagebreak()

// ============================================================================
// CHAPTER 7: IMPLEMENTATION
// ============================================================================

= Implementation Guide

== Architecture Mapping

#figure(
  tablex(
    columns: (auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Paper Concept*], [*Python Class*],
    hlinex(),
    [3D U-Net], [`UNet3D`],
    [Spatial Attention], [`SpatialAttention`],
    [Temporal Attention], [`TemporalAttention`],
    [Cosine Schedule], [`CosineSchedule`],
    [ε-Loss], [`train_step()`],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Paper to code mapping]
)

== Noise Schedule

#codeblock(title: "PyTorch: Cosine Schedule")[
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
]

== DDIM Sampling

#codeblock(title: "PyTorch: DDIM Sampler")[
```python
@torch.no_grad()
def ddim_sample(model, schedule, shape, num_steps=50):
    z_t = torch.randn(shape)
    timesteps = torch.linspace(1, 0, num_steps + 1)

    for i in range(num_steps):
        t, s = timesteps[i], timesteps[i + 1]
        alpha_t, sigma_t = schedule(t)
        alpha_s, sigma_s = schedule(s)

        eps_pred = model(z_t, t.expand(shape[0]))
        x_pred = (z_t - sigma_t * eps_pred) / alpha_t
        z_t = alpha_s * x_pred + sigma_s * eps_pred

    return z_t
```
]

== Implementation Checklist

#keypoint(title: "Components to Implement")[
  1. `CosineSchedule` - Noise schedule with log-SNR clipping
  2. `SpatialAttention` - Self-attention over H×W
  3. `TemporalAttention` - Self-attention over T with relative pos
  4. `ResBlock3D` - Convolution + attention + residual
  5. `UNet3D` - Encoder-decoder with skip connections
  6. Training loop with EMA
  7. DDIM sampler with CFG
  8. Reconstruction guidance
]

#pagebreak()

// ============================================================================
// REFERENCES
// ============================================================================

= References

+ Ho et al. "Video Diffusion Models." arXiv:2204.03458, 2022.
+ Ho et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
+ Nichol & Dhariwal. "Improved DDPM." ICML 2021.
+ Song et al. "Denoising Diffusion Implicit Models." ICLR 2021.
+ Ho & Salimans. "Classifier-Free Diffusion Guidance." NeurIPS Workshop 2021.
