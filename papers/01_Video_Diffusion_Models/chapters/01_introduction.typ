// Chapter 1: Introduction

= Introduction

== Motivation and Context

Video generation is one of the most challenging tasks in generative modeling. Unlike image generation, video requires modeling both:
- *Spatial coherence*: Each frame must be a realistic image
- *Temporal coherence*: Consecutive frames must form a plausible motion sequence

Before this work, video generation was primarily tackled using:
- *GANs* (Generative Adversarial Networks): MoCoGAN, DVD-GAN, TGAN
- *VAEs* (Variational Autoencoders): VideoGPT, VQ-VAE
- *Autoregressive models*: Video Transformer

However, diffusion models had shown remarkable success in image generation, achieving state-of-the-art results in:
- Unconditional image generation
- Text-to-image generation (GLIDE, DALL-E 2)
- Image super-resolution
- Audio generation (WaveGrad, DiffWave)

This paper asks: *Can we extend diffusion models to video generation?*

== Key Challenges Addressed

=== Challenge 1: Memory Constraints
#block(fill: luma(245), inset: 1em, radius: 4pt)[
Video data is inherently high-dimensional:
- A 16-frame video at 64×64 resolution with 3 color channels requires:
  $ "Memory" = 16 times 64 times 64 times 3 = 196,608 "values per sample" $
- Compare to a single image: $64 times 64 times 3 = 12,288$ values
- *16× more memory* just for this small resolution!
]

*Solution*: Space-time factorized architecture that processes spatial and temporal dimensions separately.

=== Challenge 2: Temporal Coherence
Generating temporally coherent video is much harder than generating individual frames because:
- Motion must be physically plausible
- Objects must maintain consistent appearance across frames
- Scene dynamics must follow natural patterns

*Solution*:
- Temporal attention blocks that attend over the time dimension
- Relative position embeddings for temporal ordering
- Joint image-video training to leverage large image datasets

=== Challenge 3: Long Video Generation
Training is limited to short clips (e.g., 16 frames) due to memory constraints. How to generate longer videos?

*Solution*: Reconstruction-guided sampling - a novel conditional sampling technique that enables:
- Autoregressive temporal extension (generate more frames)
- Temporal super-resolution (increase frame rate)
- Spatial super-resolution (increase resolution)

== Paper Contributions

#figure(
  table(
    columns: (auto, 1fr),
    inset: 10pt,
    align: left,
    [*Contribution*], [*Description*],
    [1. First Video Diffusion Model], [Successfully applied diffusion models to video generation for the first time],
    [2. 3D U-Net Architecture], [Space-time factorized architecture suitable for video data],
    [3. Joint Training], [Simultaneous training on video and image data for better quality],
    [4. Reconstruction Guidance], [New conditional sampling method for extending videos],
    [5. State-of-the-Art Results], [Best scores on UCF101, BAIR, Kinetics-600 benchmarks],
  ),
  caption: [Summary of paper contributions]
)

== Problem Formulation

=== Video as Data
A video $bold(x)$ is represented as a 4D tensor:
$ bold(x) in RR^(T times H times W times C) $
where:
- $T$ = number of frames (temporal dimension)
- $H$ = height (spatial dimension)
- $W$ = width (spatial dimension)
- $C$ = channels (typically 3 for RGB)

=== Generation Task
The goal is to learn a generative model $p_theta (bold(x))$ that can:
1. *Unconditional generation*: Sample new videos from $p_theta (bold(x))$
2. *Conditional generation*: Sample videos given conditioning $bold(c)$ from $p_theta (bold(x) | bold(c))$

Conditioning $bold(c)$ can be:
- Text descriptions (text-to-video)
- Initial frames (video prediction)
- Low-resolution videos (super-resolution)
- Class labels (class-conditional generation)

== Why Diffusion Models?

Diffusion models offer several advantages for video generation:

=== Advantage 1: Training Stability
Unlike GANs, diffusion models:
- Have a well-defined training objective (denoising)
- Don't suffer from mode collapse
- Don't require careful balancing of generator/discriminator

=== Advantage 2: Sample Quality
Diffusion models have shown:
- Higher fidelity samples than GANs on many benchmarks
- Better coverage of the data distribution
- More detailed and realistic outputs

=== Advantage 3: Flexible Conditioning
The diffusion framework naturally supports:
- Classifier-free guidance for improved conditional generation
- Multiple conditioning modalities
- Conditional sampling without retraining

=== Advantage 4: Principled Likelihood
Diffusion models:
- Optimize a variational lower bound on log-likelihood
- Have connections to score matching and SDEs
- Provide a principled probabilistic framework

== Document Overview

This document is organized as follows:

#table(
  columns: (auto, 1fr),
  inset: 8pt,
  [*Chapter*], [*Content*],
  [2. Background], [Mathematical foundations of diffusion models],
  [3. Architecture], [Detailed 3D U-Net architecture with diagrams],
  [4. Training], [Training objectives, joint training, noise schedules],
  [5. Sampling], [Sampling algorithms and reconstruction guidance],
  [6. Experiments], [Benchmarks, metrics, and results analysis],
  [7. Implementation], [Complete PyTorch implementation from scratch],
)
