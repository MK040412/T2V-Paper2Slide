// Chapter 6: Experiments and Results
#import "../template.typ": *
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.2"

#show: chapter-template.with(
  title: "Experiments and Results",
  subtitle: "Benchmarks, Ablations, and Analysis",
  chapter-num: 6,
)

= Experimental Results <experiments>

== Evaluation Setup

#keypoint(title: "Evaluation Metrics")[
  *FVD (Fréchet Video Distance)*: Measures overall video quality using I3D features. Lower is better.

  *FID (Fréchet Inception Distance)*: Measures individual frame quality. Lower is better.

  *IS (Inception Score)*: Measures quality and diversity. Higher is better.
]

== Benchmark Datasets

#figure(
  tablex(
    columns: (auto, auto, auto, auto, 1fr),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Dataset*], [*Size*], [*Resolution*], [*Classes*], [*Task*],
    hlinex(),
    [UCF101], [13K], [320×240], [101], [Unconditional video generation],
    [BAIR Robot], [43K], [64×64], [-], [Video prediction],
    [Kinetics-600], [480K], [Variable], [600], [Video prediction],
    [~10M T-I], [10M], [Variable], [-], [Text-to-video],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Evaluation benchmarks]
)

== UCF101: Unconditional Generation

#innovation(title: "State-of-the-Art Results")[
  VDM achieves significant improvements over previous methods on UCF101:

  #figure(
    tablex(
      columns: (auto, auto, auto),
      align: center + horizon,
      auto-vlines: false,
      hlinex(stroke: 1.5pt),
      [*Method*], [*FID↓*], [*IS↑*],
      hlinex(),
      [MoCoGAN (2018)], [26998], [12.42],
      [TGAN-v2 (2020)], [3431], [28.87],
      [DVD-GAN (2019)], [--], [32.97],
      [VideoGPT (2021)], [--], [24.69],
      [MoCoGAN-HD (2021)], [838], [33.95],
      [DIGAN (2022)], [577], [30.24],
      hlinex(stroke: 0.5pt + gray),
      [*VDM (Ours)*], [*295*], [*57.00*],
      hlinex(),
      [Real Data], [--], [60.20],
      hlinex(stroke: 1.5pt),
    ),
    caption: [UCF101 unconditional generation (16 frames, 64×64)]
  )
]

#figure(
  cetz.canvas(length: 0.5cm, {
    import cetz.draw: *

    // IS Bar chart
    let methods = ("MoCoGAN", "TGAN-v2", "DVD-GAN", "VideoGPT", "DIGAN", "VDM", "Real")
    let is_scores = (12.42, 28.87, 32.97, 24.69, 30.24, 57.00, 60.20)
    let colors = (gray, gray, gray, gray, gray, blue, green)

    for (i, (method, score, color)) in methods.zip(is_scores).zip(colors).enumerate() {
      let height = score / 60 * 5
      rect((i * 2, 0), (i * 2 + 1.5, height), fill: color.lighten(60%), stroke: color)
      content((i * 2 + 0.75, height + 0.4), text(size: 7pt)[#calc.round(score, digits: 1)])
      content((i * 2 + 0.75, -0.5), text(size: 6pt, angle: -45deg)[#method])
    }

    // Y-axis
    line((0, 0), (0, 6), stroke: black)
    content((-0.5, 3), text(size: 8pt, angle: 90deg)[IS ↑])
    for val in (0, 20, 40, 60) {
      let y = val / 60 * 5
      line((-0.2, y), (0, y), stroke: black)
      content((-0.6, y), text(size: 6pt)[#val])
    }
  }),
  caption: [Inception Score comparison on UCF101 (higher is better)]
)

#remark("Analysis")[
  VDM's IS of 57.00 is remarkably close to real data (60.20), representing a *73% improvement* over the previous best (DIGAN: 30.24). The model has essentially closed the gap to real data quality.
]

== BAIR Robot Pushing: Video Prediction

#figure(
  tablex(
    columns: (auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Method*], [*FVD↓*],
    hlinex(),
    [SV2P (2018)], [203.4],
    [SAVP (2019)], [116.4],
    [DVD-GAN-FP (2019)], [109.8],
    [VideoGPT (2021)], [103.3],
    [CCVS (2021)], [99.0],
    [FitVid (2021)], [93.6],
    [Transframer (2022)], [86.9],
    hlinex(stroke: 0.5pt + gray),
    [*VDM (Ours)*], [*66.9*],
    hlinex(stroke: 1.5pt),
  ),
  caption: [BAIR video prediction: 1 context frame → 15 future frames]
)

#keypoint(title: "Task Description")[
  Given the first frame of a robot arm pushing objects, predict the next 15 frames.

  *Challenge*: Multiple plausible futures (object can move in different directions).

  *VDM Achievement*: 23% improvement over previous state-of-the-art (Transframer).
]

== Kinetics-600: Large-Scale Video Prediction

#figure(
  tablex(
    columns: (auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Method*], [*FVD↓*], [*IS↑*],
    hlinex(),
    [DVD-GAN-FP], [69.2], [--],
    [Transframer], [25.4], [12.5],
    hlinex(stroke: 0.5pt + gray),
    [*VDM (Ours)*], [*16.2*], [*15.6*],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Kinetics-600 video prediction: 5 context frames → 11 future frames]
)

#remark("Significance")[
  Kinetics-600 is a diverse dataset with 600 action classes. VDM's strong performance demonstrates generalization beyond simple domains.
]

== Text-to-Video Generation

#figure(
  tablex(
    columns: (auto, auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Metric*], [*CogVideo*], [*VDM (T2V)*], [*Improvement*],
    hlinex(),
    [FVD↓], [1294], [--], [--],
    [CLIP Score↑], [0.2631], [0.2712], [+3.1%],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Text-to-video comparison on MSR-VTT]
)

#warning(title: "Text-to-Video Limitations")[
  The VDM text-to-video model is trained on only ~10M text-image pairs (not text-video).

  Later models like Make-A-Video and Imagen Video use much larger datasets and achieve better results. VDM's text-to-video was primarily a proof of concept.
]

#pagebreak()

= Ablation Studies <ablations>

== Effect of Joint Image-Video Training

#innovation(title: "Key Finding: Joint Training")[
  Adding images during training dramatically improves video quality:

  #figure(
    tablex(
      columns: (auto, auto, auto, auto),
      align: center + horizon,
      auto-vlines: false,
      hlinex(stroke: 1.5pt),
      [*Images/Batch*], [*FVD↓*], [*FID↓*], [*Improvement*],
      hlinex(),
      [0 (video only)], [205], [37], [baseline],
      [4], [104], [20], [2.0× FVD],
      [8], [61], [15], [3.4× FVD],
      [16], [72], [15], [2.8× FVD],
      hlinex(stroke: 1.5pt),
    ),
    caption: [UCF101: Effect of number of images per video batch]
  )
]

#figure(
  cetz.canvas(length: 0.8cm, {
    import cetz.draw: *

    // Axes
    line((0, 0), (7, 0), stroke: black, mark: (end: "stealth"))
    line((0, 0), (0, 5), stroke: black, mark: (end: "stealth"))
    content((7.5, 0), text(size: 9pt)[Images/batch])
    content((-0.5, 5.3), text(size: 9pt)[FVD])

    // Data points and line
    let data = ((0, 205), (4, 104), (8, 61), (16, 72))
    let scaled = data.map(p => (p.at(0) / 16 * 6, (250 - p.at(1)) / 250 * 4.5))

    line(..scaled, stroke: blue + 2pt)
    for (x, y) in scaled {
      circle((x, y), radius: 0.15, fill: blue, stroke: none)
    }

    // Labels
    for ((raw_x, raw_y), (x, y)) in data.zip(scaled) {
      content((x, y + 0.4), text(size: 7pt)[#raw_y])
    }

    // X-axis labels
    for x in (0, 4, 8, 16) {
      content((x / 16 * 6, -0.3), text(size: 7pt)[#x])
    }

    // Optimal marker
    content((3, 1), text(fill: green.darken(20%), size: 8pt)[Optimal])
  }),
  caption: [FVD vs number of images: optimal at 8 images per batch]
)

#remark("Interpretation")[
  - *Too few images*: Insufficient spatial supervision
  - *Optimal (8)*: Best balance of spatial and temporal learning
  - *Too many images*: May overwhelm temporal learning
]

== Reconstruction Guidance vs Replacement

#figure(
  tablex(
    columns: (auto, auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Method*], [*FVD↓*], [*FID↓*], [*IS↑*],
    hlinex(),
    [Naive Replacement], [451], [26], [7.0],
    [Reconstruction Guidance], [136], [14], [10.3],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Kinetics-600 video prediction: reconstruction guidance achieves 3.3× better FVD]
)

#figure(
  diagram(
    spacing: (10pt, 10pt),
    node-stroke: 1pt,

    // Replacement results
    node((0, 0), [Replacement], shape: rect, fill: red.lighten(80%), width: 2.5cm),
    node((2, 0), [FVD: 451], shape: rect, fill: red.lighten(70%), width: 2cm),
    node((4, 0), [Boundary\ artifacts], shape: rect, fill: red.lighten(60%), width: 2.5cm),

    // Guidance results
    node((0, 1.5), [Recon. Guidance], shape: rect, fill: green.lighten(80%), width: 2.5cm),
    node((2, 1.5), [FVD: 136], shape: rect, fill: green.lighten(70%), width: 2cm),
    node((4, 1.5), [Smooth\ transitions], shape: rect, fill: green.lighten(60%), width: 2.5cm),

    edge((0, 0), (2, 0), "->"),
    edge((2, 0), (4, 0), "->"),
    edge((0, 1.5), (2, 1.5), "->"),
    edge((2, 1.5), (4, 1.5), "->"),
  ),
  caption: [Replacement vs reconstruction guidance]
)

== Effect of Guidance Weight

#figure(
  tablex(
    columns: (auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Weight $w_r$*], [*FVD↓*], [*Quality*],
    hlinex(),
    [0.0], [~300], [Poor conditioning],
    [0.1], [180], [Mild conditioning],
    [0.3], [136], [Optimal],
    [0.5], [165], [Over-conditioned],
    [1.0], [250], [Too aggressive],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Effect of reconstruction guidance weight on Kinetics-600]
)

== Autoregressive Extension Quality

#figure(
  tablex(
    columns: (auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Frames*], [*FVD↓*], [*Degradation*],
    hlinex(),
    [16 (base)], [16.2], [--],
    [32], [18.5], [+14%],
    [64], [24.3], [+50%],
    [128], [35.2], [+117%],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Quality degradation with autoregressive extension on Kinetics-600]
)

#warning(title: "Quality Degradation")[
  Each autoregressive step accumulates errors. For very long videos (128+ frames), quality degrades noticeably.

  *Solutions in later work*:
  - Hierarchical generation (coarse-to-fine)
  - Longer base models
  - Better conditioning mechanisms
]

== Temporal Super-Resolution

#figure(
  tablex(
    columns: (auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Upsampling*], [*FVD↓*], [*Smoothness*],
    hlinex(),
    [2×], [12.4], [High],
    [4×], [18.7], [Medium],
    [8×], [28.9], [Lower],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Temporal super-resolution results]
)

== Computational Analysis

#figure(
  tablex(
    columns: (auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Attention Type*], [*Memory*], [*Speed*],
    hlinex(),
    [Full 3D], [OOM], [--],
    [Factorized (VDM)], [~8GB], [~2.5s/step],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Memory and speed for 16×64×64 videos on single GPU]
)

#keypoint(title: "Computational Advantage")[
  The factorized attention makes video diffusion *tractable*:
  - Full 3D attention would require >100GB for a single batch
  - Factorized attention fits in ~8GB
  - This is the key enabler for the entire model
]

#pagebreak()

= Comparison with Subsequent Work <comparison>

== VDM vs Later Models

#figure(
  tablex(
    columns: (auto, auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Model*], [*Year*], [*Resolution*], [*Key Improvement*],
    hlinex(),
    [VDM], [2022], [64×64], [First diffusion video model],
    [Make-A-Video], [2022], [768×768], [Pseudo-3D, no T-V pairs],
    [Imagen Video], [2022], [1280×768], [7-model cascade],
    [Video LDM], [2023], [512×512], [Latent space, frozen T2I],
    [SVD], [2023], [1024×576], [Data curation, open weights],
    [Sora], [2024], [1920×1080], [DiT, world simulation],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Evolution of video diffusion models]
)

#remark("Historical Impact")[
  VDM established the foundation that all subsequent video diffusion models built upon:
  1. Demonstrated diffusion can work for video
  2. Introduced space-time factorization
  3. Proposed reconstruction guidance for conditioning
  4. Showed joint image-video training helps

  These techniques appear (with modifications) in virtually all later work.
]

== Quality Progression

#figure(
  cetz.canvas(length: 0.6cm, {
    import cetz.draw: *

    // Timeline
    line((0, 2), (12, 2), stroke: black + 1pt, mark: (end: "stealth"))

    // Models
    let models = (
      (0, "VDM\n64×64", blue),
      (3, "Make-A-Video\n768×768", green),
      (5, "Video LDM\n512×512", orange),
      (8, "SVD\n1024×576", purple),
      (11, "Sora\n1080p", red),
    )

    for (x, label, color) in models {
      circle((x, 2), radius: 0.2, fill: color.lighten(50%), stroke: color)
      content((x, 0.8), text(size: 7pt, fill: color.darken(20%))[#label])
    }

    // Years
    content((0, 2.6), text(size: 6pt)[Apr 2022])
    content((3, 2.6), text(size: 6pt)[Sep 2022])
    content((5, 2.6), text(size: 6pt)[Apr 2023])
    content((8, 2.6), text(size: 6pt)[Nov 2023])
    content((11, 2.6), text(size: 6pt)[Feb 2024])
  }),
  caption: [Timeline of video diffusion model development]
)

#pagebreak()

== Summary

#figure(
  tablex(
    columns: (auto, 1fr),
    align: (center, left),
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Benchmark*], [*VDM Achievement*],
    hlinex(),
    [UCF101 (IS)], [57.0 (vs real 60.2) - 73% improvement over prior SOTA],
    [BAIR (FVD)], [66.9 - 23% improvement over Transframer],
    [Kinetics (FVD)], [16.2 - 36% improvement over Transframer],
    [Joint Training], [3.4× FVD improvement with 8 images/batch],
    [Recon. Guidance], [3.3× FVD improvement over replacement],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Chapter 6 Summary: Key experimental results]
)

#innovation(title: "Overall Impact")[
  Video Diffusion Models (VDM) demonstrated that:

  1. *Diffusion models can generate coherent videos* with both spatial and temporal quality

  2. *Space-time factorization* makes computation tractable (1000× speedup)

  3. *Joint image-video training* dramatically improves quality (3.4×)

  4. *Reconstruction guidance* enables flexible conditioning (3.3× better than naive)

  These insights directly influenced all subsequent video generation research.
]
