// Chapter 1: Introduction to Video Diffusion Models
#import "../template.typ": *
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.2"

#show: chapter-template.with(
  title: "Introduction",
  subtitle: "The Dawn of Video Diffusion",
  chapter-num: 1,
)

= Introduction <intro>

== Paper Overview

#keypoint(title: "Paper Information")[
  *Title*: Video Diffusion Models \
  *Authors*: Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet \
  *Affiliation*: Google Research, Brain Team \
  *ArXiv*: 2204.03458 (April 2022) \
  *Significance*: First successful application of diffusion models to video generation
]

== The Challenge of Video Generation

Video generation is fundamentally more challenging than image generation due to the need for both *spatial coherence* (realistic individual frames) and *temporal coherence* (physically plausible motion across frames).

#figure(
  diagram(
    spacing: (15pt, 10pt),
    node-stroke: 1pt,
    edge-stroke: 1pt,
    node((0, 0), [Image Generation], shape: rect, fill: blue.lighten(80%), width: 3cm, height: 1cm),
    node((0, 1), [$H times W times C$], shape: rect, fill: blue.lighten(90%), width: 3cm),
    node((2, 0), [Video Generation], shape: rect, fill: green.lighten(80%), width: 3cm, height: 1cm),
    node((2, 1), [$T times H times W times C$], shape: rect, fill: green.lighten(90%), width: 3cm),
    edge((0, 0), (0, 1), "->"),
    edge((2, 0), (2, 1), "->"),
    edge((0, 0), (2, 0), "->", [+Temporal], label-pos: 0.5, label-side: center),
  ),
  caption: [Video generation extends image generation with temporal dimension]
)

#warning(title: "Memory Challenge")[
  A 16-frame video at 64×64 resolution requires:
  $ "Data Size" = underbrace(16, "frames") times underbrace(64 times 64, "spatial") times underbrace(3, "RGB") = 196,608 "values" $

  This is *16× more data* than a single image, making naive approaches computationally prohibitive.
]

== Why Diffusion Models for Video?

Before this paper, video generation was dominated by:
- *GANs*: Mode collapse, training instability, limited temporal coherence
- *Autoregressive models*: Slow generation, error accumulation
- *VAEs*: Blurry outputs, limited quality

#figure(
  tablex(
    columns: (auto, auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Method*], [*Quality*], [*Diversity*], [*Training Stability*],
    hlinex(),
    [GANs], [High], [Low], [Unstable],
    [VAEs], [Medium], [High], [Stable],
    [Autoregressive], [Medium], [High], [Stable],
    [*Diffusion*], [*High*], [*High*], [*Stable*],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Comparison of generative model paradigms]
)

== Key Contributions of the Paper

#innovation(title: "Five Major Contributions")[
  1. *3D U-Net Architecture*: Extended image diffusion U-Net to handle video by adding temporal dimensions

  2. *Space-Time Factorized Attention*: Achieved 1000× speedup by separating spatial and temporal attention

  3. *Joint Image-Video Training*: Achieved 3.4× FVD improvement by co-training with images

  4. *Reconstruction Guidance*: Novel technique for conditional video generation and extension

  5. *State-of-the-Art Results*: Best performance on UCF101, BAIR, and Kinetics-600
]

== Paper Structure Mapping

#figure(
  diagram(
    spacing: (8pt, 12pt),
    node-stroke: 1pt,

    // Paper sections
    node((0, 0), [Section 1\ Introduction], shape: rect, fill: yellow.lighten(80%), width: 2.5cm),
    node((1.5, 0), [Section 2\ Background], shape: rect, fill: orange.lighten(80%), width: 2.5cm),
    node((3, 0), [Section 3\ Architecture], shape: rect, fill: red.lighten(80%), width: 2.5cm),
    node((4.5, 0), [Section 4\ Experiments], shape: rect, fill: purple.lighten(80%), width: 2.5cm),

    // Our chapters
    node((0, 1.5), [Ch.1-2\ Foundations], shape: rect, fill: blue.lighten(80%), width: 2.5cm),
    node((1.5, 1.5), [Ch.3\ 3D U-Net], shape: rect, fill: blue.lighten(80%), width: 2.5cm),
    node((3, 1.5), [Ch.4-5\ Train+Sample], shape: rect, fill: blue.lighten(80%), width: 2.5cm),
    node((4.5, 1.5), [Ch.6-7\ Results+Code], shape: rect, fill: blue.lighten(80%), width: 2.5cm),

    edge((0, 0), (0, 1.5), "->"),
    edge((1.5, 0), (0, 1.5), "->"),
    edge((1.5, 0), (1.5, 1.5), "->"),
    edge((3, 0), (1.5, 1.5), "->"),
    edge((3, 0), (3, 1.5), "->"),
    edge((4.5, 0), (3, 1.5), "->"),
    edge((4.5, 0), (4.5, 1.5), "->"),
  ),
  caption: [Mapping from paper sections to our detailed chapter analysis]
)

== Target Applications

The VDM model addresses three main video generation tasks:

=== 1. Unconditional Video Generation
Generate videos from pure noise without any conditioning:
$ bold(x) tilde p_theta (bold(x)) $

=== 2. Video Prediction (Frame Conditioning)
Generate future frames given initial frames:
$ bold(x)_(t+1:T) tilde p_theta (bold(x)_(t+1:T) | bold(x)_(1:t)) $

=== 3. Text-to-Video Generation
Generate videos from text descriptions:
$ bold(x) tilde p_theta (bold(x) | "text") $

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *

    // Unconditional
    rect((0, 0), (2, 1.5), fill: blue.lighten(90%), stroke: blue)
    content((1, 0.75), text(size: 8pt)[Noise $arrow.r$ Video])
    content((1, -0.3), text(size: 7pt)[Unconditional])

    // Prediction
    rect((3, 0), (5, 1.5), fill: green.lighten(90%), stroke: green)
    content((4, 0.75), text(size: 8pt)[Frames $arrow.r$ Future])
    content((4, -0.3), text(size: 7pt)[Prediction])

    // Text-to-Video
    rect((6, 0), (8, 1.5), fill: purple.lighten(90%), stroke: purple)
    content((7, 0.75), text(size: 8pt)[Text $arrow.r$ Video])
    content((7, -0.3), text(size: 7pt)[Text-to-Video])
  }),
  caption: [Three main generation tasks addressed by VDM]
)

== Datasets Used

#figure(
  tablex(
    columns: (auto, auto, auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Dataset*], [*Type*], [*Videos*], [*Resolution*], [*Use Case*],
    hlinex(),
    [UCF101], [Actions], [13,320], [320×240], [Unconditional],
    [BAIR Robot], [Robotics], [43,264], [64×64], [Prediction],
    [Kinetics-600], [Actions], [480K], [Variable], [Prediction],
    [~10M T-I pairs], [Text-Image], [10M], [Variable], [Text-to-Video],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Datasets used for training and evaluation]
)

== Evaluation Metrics

#definition("Fréchet Video Distance (FVD)")[
  FVD measures the distance between distributions of real and generated videos in a learned feature space (I3D network):
  $ "FVD" = ||mu_r - mu_g||^2 + "Tr"(Sigma_r + Sigma_g - 2(Sigma_r Sigma_g)^(1/2)) $
  *Lower is better.* A perfect model achieves FVD = 0.
]

#definition("Inception Score (IS)")[
  IS measures both quality and diversity of generated samples:
  $ "IS" = exp(EE_bold(x)[D_"KL"(p(y|bold(x)) || p(y))]) $
  *Higher is better.* Real UCF101 data achieves IS ≈ 60.
]

#definition("Fréchet Inception Distance (FID)")[
  FID measures the quality of individual frames:
  $ "FID" = ||mu_r - mu_g||^2 + "Tr"(Sigma_r + Sigma_g - 2(Sigma_r Sigma_g)^(1/2)) $
  Computed on ImageNet-pretrained Inception features. *Lower is better.*
]

== Historical Context

#figure(
  diagram(
    spacing: (5pt, 15pt),
    node-stroke: 1pt,

    // Timeline nodes
    node((0, 0), [2020\ DDPM], shape: rect, fill: gray.lighten(80%), width: 2cm),
    node((1.5, 0), [2021\ Improved\ DDPM], shape: rect, fill: gray.lighten(70%), width: 2cm),
    node((3, 0), [2021\ CFG], shape: rect, fill: gray.lighten(60%), width: 2cm),
    node((4.5, 0), [*2022*\ *VDM*], shape: rect, fill: blue.lighten(70%), width: 2cm),

    edge((0, 0), (1.5, 0), "->"),
    edge((1.5, 0), (3, 0), "->"),
    edge((3, 0), (4.5, 0), "->"),
  ),
  caption: [Timeline of diffusion model development leading to VDM]
)

== Reading Guide

#keypoint(title: "How to Use This Document")[
  - *Chapter 2*: Mathematical background on diffusion models
  - *Chapter 3*: Detailed 3D U-Net architecture with diagrams
  - *Chapter 4*: Training procedures and optimization
  - *Chapter 5*: Sampling algorithms and reconstruction guidance
  - *Chapter 6*: Experimental results and ablations
  - *Chapter 7*: Complete PyTorch implementation guide

  Each chapter builds on previous ones. For implementation, follow chapters sequentially.
]

#pagebreak()

== Summary

#figure(
  tablex(
    columns: (auto, 1fr),
    align: (center, left),
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Aspect*], [*Description*],
    hlinex(),
    [Problem], [Video generation requires spatial + temporal coherence],
    [Solution], [3D U-Net diffusion model with factorized attention],
    [Key Innovation], [Space-time factorization, reconstruction guidance],
    [Results], [SOTA on UCF101 (IS: 57), BAIR (FVD: 66.9), Kinetics (FVD: 16.2)],
    [Impact], [Foundation for all subsequent video diffusion models],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Chapter 1 Summary]
)
