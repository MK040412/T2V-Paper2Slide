// Video Diffusion Models - Complete Technical Analysis
// Main compilation file

#import "template.typ": *
#import "@preview/ctheorems:1.1.3": *
#import "@preview/showybox:2.0.3": showybox
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.2"
#import "@preview/tablex:0.0.9": tablex, rowspanx, colspanx, hlinex, vlinex

#show: thmrules.with(qed-symbol: $square$)

#let definition = thmbox("definition", "Definition", fill: rgb("#e8f4e8"), base_level: 1)
#let theorem = thmbox("theorem", "Theorem", fill: rgb("#fff3cd"), base_level: 1)
#let lemma = thmbox("lemma", "Lemma", fill: rgb("#e3f2fd"), base_level: 1)
#let corollary = thmbox("corollary", "Corollary", fill: rgb("#f3e5f5"), base_level: 1)
#let remark = thmbox("remark", "Remark", fill: rgb("#fff8e1"), base_level: 1)
#let example = thmbox("example", "Example", fill: rgb("#e0f7fa"), base_level: 1)
#let algorithm = thmbox("algorithm", "Algorithm", fill: rgb("#fce4ec"), base_level: 1)

// Document setup
#set document(
  title: "Video Diffusion Models: Complete Technical Analysis",
  author: "T2V-Paper2Slide",
)

#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
  header: context {
    if counter(page).get().first() > 2 [
      #set text(size: 9pt)
      _Video Diffusion Models - Complete Technical Analysis_
      #h(1fr)
      #counter(page).display()
    ]
  },
  footer: context {
    align(center)[
      #set text(size: 8pt, fill: gray)
      T2V-Paper2Slide | Comprehensive Implementation Guide
    ]
  }
)

#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1.1")
#set math.equation(numbering: "(1)")
#set par(justify: true, leading: 0.65em)

// Code styling
#show raw.where(block: true): it => {
  block(
    fill: luma(245),
    inset: (x: 12pt, y: 10pt),
    radius: 4pt,
    width: 100%,
    stroke: (left: 3pt + blue.darken(20%)),
    it
  )
}

// ============================================================================
// TITLE PAGE
// ============================================================================

#align(center)[
  #v(2cm)

  #text(size: 32pt, weight: "bold", fill: blue.darken(20%))[
    Video Diffusion Models
  ]

  #v(0.5cm)

  #text(size: 20pt)[
    Complete Technical Analysis
  ]

  #v(0.3cm)

  #text(size: 14pt, style: "italic")[
    For From-Scratch Implementation in PyTorch
  ]

  #v(1.5cm)

  #showybox(
    frame: (border-color: blue.darken(20%), title-color: blue.lighten(85%), body-color: blue.lighten(95%)),
    title-style: (color: black, weight: "bold"),
    title: "Paper Reference"
  )[
    *Ho, Salimans, Gritsenko, Chan, Norouzi, Fleet* \
    "Video Diffusion Models" \
    arXiv:2204.03458, April 2022 \
    Google Research, Brain Team
  ]

  #v(1cm)

  #showybox(
    frame: (border-color: green.darken(20%), title-color: green.lighten(85%), body-color: green.lighten(95%)),
    title-style: (color: black, weight: "bold"),
    title: "Document Overview"
  )[
    This comprehensive guide covers:
    - Mathematical foundations of video diffusion
    - 3D U-Net architecture with factorized attention
    - Training methodology and noise schedules
    - Sampling algorithms and reconstruction guidance
    - Complete PyTorch implementation

    *Target*: Researchers and engineers implementing video diffusion from scratch.
  ]

  #v(1.5cm)

  #line(length: 60%, stroke: 1pt + gray)

  #v(0.5cm)

  #text(size: 11pt)[
    *T2V-Paper2Slide Project* \
    Document Version 2.0 | #datetime.today().display()
  ]
]

#pagebreak()

// ============================================================================
// TABLE OF CONTENTS
// ============================================================================

#outline(
  title: [Table of Contents],
  indent: 2em,
  depth: 3,
)

#pagebreak()

// ============================================================================
// CHAPTERS
// ============================================================================

#include "chapters/ch01_introduction.typ"
#pagebreak()

#include "chapters/ch02_background.typ"
#pagebreak()

#include "chapters/ch03_architecture.typ"
#pagebreak()

#include "chapters/ch04_training.typ"
#pagebreak()

#include "chapters/ch05_sampling.typ"
#pagebreak()

#include "chapters/ch06_experiments.typ"
#pagebreak()

#include "chapters/ch07_implementation.typ"

// ============================================================================
// REFERENCES
// ============================================================================

#pagebreak()

= References

#set text(size: 10pt)

+ Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., & Fleet, D. J. (2022). *Video Diffusion Models*. arXiv:2204.03458.

+ Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.

+ Nichol, A. Q., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML 2021.

+ Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR 2021.

+ Ho, J., & Salimans, T. (2021). *Classifier-Free Diffusion Guidance*. NeurIPS Workshop 2021.

+ Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021.

+ Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.

+ Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.

+ Soomro, K., Zamir, A. R., & Shah, M. (2012). *UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild*. arXiv:1212.0402.

+ Carreira, J., & Zisserman, A. (2017). *Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset*. CVPR 2017.
