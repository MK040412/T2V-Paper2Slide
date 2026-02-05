// Chapter 6: Experiments and Results

= Experiments and Results

This chapter presents the experimental evaluation of Video Diffusion Models.

== Evaluation Metrics

=== Fréchet Video Distance (FVD)

#block(fill: rgb("#e8f4e8"), inset: 1em, radius: 4pt)[
  *Definition 6.1 (FVD)*

  FVD measures the distance between the distribution of generated and real videos:

  $ "FVD" = || mu_r - mu_g ||^2 + "Tr"(Sigma_r + Sigma_g - 2(Sigma_r Sigma_g)^(1\/2)) $

  where $(mu_r, Sigma_r)$ and $(mu_g, Sigma_g)$ are mean and covariance of I3D features for real and generated videos.
]

FVD captures both:
- *Spatial quality*: How realistic are individual frames?
- *Temporal quality*: How coherent is the motion?

Lower FVD = Better quality.

=== Fréchet Inception Distance (FID)

FID measures image quality using Inception-v3 features:

$ "FID" = || mu_r - mu_g ||^2 + "Tr"(Sigma_r + Sigma_g - 2(Sigma_r Sigma_g)^(1\/2)) $

For videos, the paper reports:
- *FID-avg*: Average FID across all frames
- *FID-first*: FID of the first frame only

=== Inception Score (IS)

IS measures both quality and diversity:

$ "IS" = exp(EE_bold(x) [D_"KL"(p(y|bold(x)) || p(y))]) $

where $p(y|bold(x))$ is the predicted class distribution.

Higher IS = Better quality and diversity.

== Benchmark Datasets

=== UCF101

#figure(
  table(
    columns: (auto, 1fr),
    inset: 8pt,
    [*Property*], [*Value*],
    [Domain], [Human actions],
    [Classes], [101 action categories],
    [Videos], [13,320 total],
    [Resolution], [Varies (resized to 64×64)],
    [Frames], [16 frames at 25 fps],
    [Task], [Unconditional generation],
  ),
  caption: [UCF101 dataset properties]
)

=== BAIR Robot Pushing

#figure(
  table(
    columns: (auto, 1fr),
    inset: 8pt,
    [*Property*], [*Value*],
    [Domain], [Robot manipulation],
    [Videos], [~44,000 training, 256 evaluation],
    [Resolution], [64×64],
    [Frames], [16 frames (1 conditioning + 15 generated)],
    [Task], [Video prediction (conditioned on 1st frame)],
  ),
  caption: [BAIR Robot Pushing dataset properties]
)

=== Kinetics-600

#figure(
  table(
    columns: (auto, 1fr),
    inset: 8pt,
    [*Property*], [*Value*],
    [Domain], [Human actions (diverse)],
    [Classes], [600 action categories],
    [Videos], [~400,000 training, 50,000 evaluation],
    [Resolution], [64×64],
    [Frames], [16 frames (5 conditioning + 11 generated)],
    [Task], [Video prediction],
  ),
  caption: [Kinetics-600 dataset properties]
)

=== Internal Text-Video Dataset

#figure(
  table(
    columns: (auto, 1fr),
    inset: 8pt,
    [*Property*], [*Value*],
    [Size], [10 million captioned videos],
    [Captions], [BERT-large embeddings],
    [Resolution], [64×64 and 128×128],
    [Frames], [16 frames (frameskip 1 and 4)],
    [Task], [Text-conditioned video generation],
  ),
  caption: [Internal text-video dataset properties]
)

== Results: Unconditional Video Generation (UCF101)

=== Comparison with Prior Methods

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    [*Method*], [*Resolution*], [*FID↓*], [*IS↑*],
    [MoCoGAN], [16×64×64], [26998 ± 33], [12.42],
    [TGAN-F], [16×64×64], [8942 ± 4], [13.62],
    [TGAN-ODE], [16×64×64], [26512 ± 27], [15.2],
    [TGAN-F], [16×128×128], [7817 ± 10], [22.91],
    [VideoGPT], [16×128×128], [--], [24.69],
    [TGAN-v2], [16×64×64], [3431 ± 19], [26.60],
    [TGAN-v2], [16×128×128], [3497 ± 26], [28.87],
    [DVD-GAN], [16×128×128], [--], [32.97],
    [*Video Diffusion (Ours)*], [16×64×64], [*295 ± 3*], [*57 ± 0.62*],
    [Real data], [16×64×64], [--], [60.2],
  ),
  caption: [UCF101 unconditional generation results]
)

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Key Findings - UCF101*

  - *11× improvement* in FID over previous SOTA (TGAN-v2)
  - *2× improvement* in IS, approaching real data (57 vs 60.2)
  - Achieved at lower resolution (64×64) vs competitors (128×128)
]

== Results: Video Prediction (BAIR)

=== Comparison with Prior Methods

#figure(
  table(
    columns: (auto, auto),
    inset: 8pt,
    [*Method*], [*FVD↓*],
    [DVD-GAN], [109.8],
    [VideoGPT], [103.3],
    [TrIVD-GAN-FP], [103.3],
    [Transframer], [100],
    [CCVS], [99],
    [VideoTransformer], [94],
    [FitVid], [93.6],
    [NUWA], [86.9],
    [*Video Diffusion (ancestral, 512)*], [68.19],
    [*Video Diffusion (Langevin, 256)*], [*66.92*],
  ),
  caption: [BAIR Robot Pushing video prediction results]
)

Note: Video Diffusion uses an *unconditionally trained* model with reconstruction guidance, while most competitors are trained specifically for video prediction.

== Results: Video Prediction (Kinetics-600)

=== Comparison with Prior Methods

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    [*Method*], [*FVD↓*], [*IS↑*],
    [Video Transformer], [170 ± 5], [--],
    [DVD-GAN-FP], [69.1 ± 0.78], [--],
    [Video VQ-VAE], [64.3 ± 2.04], [--],
    [CCVS], [55 ± 1], [--],
    [TrIVD-GAN-FP], [25.74 ± 0.66], [12.54],
    [Transframer], [25.4], [--],
    [*Video Diffusion (ancestral, 256)*], [18.6], [15.39],
    [*Video Diffusion (Langevin, 128)*], [*16.2 ± 0.34*], [*15.64*],
  ),
  caption: [Kinetics-600 video prediction results]
)

== Results: Text-Conditioned Video Generation

=== Effect of Joint Training

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 6pt,
    [*Image Frames*], [*FVD↓*], [*FID-avg↓*], [*IS-avg↑*], [*FID-first↓*], [*IS-first↑*],
    [0], [205.42], [37.40], [7.58], [40.87], [8.74],
    [4], [70.74], [18.42], [8.53], [22.19], [9.91],
    [8], [60.72], [15.44], [8.82], [18.98], [10.12],
  ),
  caption: [Effect of joint image-video training (small model)]
)

#block(fill: rgb("#fff3cd"), inset: 1em, radius: 4pt)[
  *Key Finding*

  Adding 8 independent image frames per video:
  - *3.4× improvement* in FVD (205 → 61)
  - *2.4× improvement* in FID (37 → 15)
  - *16% improvement* in IS (7.6 → 8.8)
]

=== Effect of Classifier-Free Guidance

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 6pt,
    [*Guidance $w$*], [*FVD↓*], [*FID-avg↓*], [*IS-avg↑*], [*FID-first↓*], [*IS-first↑*],
    [1.0], [43.70], [12.39], [10.07], [16.19], [11.22],
    [2.0], [48.79], [10.47], [12.10], [13.75], [13.46],
    [5.0], [160.21], [13.52], [13.46], [16.95], [14.75],
  ),
  caption: [Effect of classifier-free guidance (frameskip 1)]
)

Trade-off:
- Higher guidance → Better IS (quality), worse FVD/FID (diversity)
- $w = 2$ provides good balance

=== Autoregressive Extension Results

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 6pt,
    [*Guidance $w$*], [*Method*], [*FVD↓*], [*FID-avg↓*], [*IS-avg↑*], [*FID-first↓*],
    [2.0], [Replacement], [451.45], [25.95], [7.00], [16.33],
    [2.0], [Recon. Guidance], [*136.22*], [*13.77*], [*10.30*], [16.34],
    [5.0], [Replacement], [456.24], [26.05], [7.04], [16.30],
    [5.0], [Recon. Guidance], [*133.92*], [*13.59*], [*10.31*], [16.28],
  ),
  caption: [64×64×64 video generation via autoregressive extension]
)

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Key Finding*

  Reconstruction guidance vs. replacement method:
  - *3.3× improvement* in FVD (451 → 136)
  - *1.9× improvement* in FID (26 → 14)
  - *1.5× improvement* in IS (7 → 10)
  - Dramatically better temporal coherence
]

== Qualitative Analysis

=== Sample Quality Observations

From visual inspection (see paper Figures 2-5):

1. *Temporal Coherence*: Videos show smooth, plausible motion
2. *Object Consistency*: Objects maintain appearance across frames
3. *Text Alignment*: Generated videos match text prompts
4. *Resolution Quality*: Clear, detailed frames

=== Failure Cases

The paper acknowledges limitations:
- Complex scene dynamics still challenging
- Some temporal flickering in high-motion scenes
- Text-video alignment not perfect for complex prompts

== Ablation Studies Summary

#figure(
  table(
    columns: (auto, 1fr),
    inset: 8pt,
    [*Ablation*], [*Finding*],
    [Joint training], [Essential for good quality (3.4× FVD improvement)],
    [Classifier-free guidance], [Improves quality at cost of diversity],
    [Reconstruction guidance], [Critical for video extension (3.3× FVD improvement)],
    [Langevin correction], [Improves conditional sampling quality],
    [v-prediction vs ε], [v-prediction better for video prediction tasks],
  ),
  caption: [Summary of ablation study findings]
)

== Computational Cost Analysis

=== Training Cost

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    [*Task*], [*Hardware*], [*Steps*], [*Est. GPU-hours*],
    [UCF101], [128 TPU-v4], [60K], [~2,000],
    [BAIR], [128 TPU-v4], [660K], [~22,000],
    [Kinetics], [256 TPU-v4], [220K], [~15,000],
    [Text-to-Video], [128 TPU-v4], [700K], [~23,000],
  ),
  caption: [Estimated training computational cost]
)

=== Inference Cost

- *Ancestral sampling*: 256 forward passes
- *Langevin sampling*: 256 + 256 = 512 forward passes
- *Reconstruction guidance*: Additional backward pass per step

Per video generation at 16×64×64:
- ~2-5 seconds on modern GPU (A100)
- Memory: ~8-16 GB
