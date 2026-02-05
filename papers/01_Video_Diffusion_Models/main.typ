// Video Diffusion Models - Comprehensive Technical Analysis
// For from-scratch implementation

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

// Title Page
#align(center)[
  #v(3cm)
  #text(size: 28pt, weight: "bold")[Video Diffusion Models]
  #v(0.5cm)
  #text(size: 18pt)[Complete Technical Analysis for From-Scratch Implementation]
  #v(1cm)
  #text(size: 14pt, style: "italic")[
    Based on: Ho et al. "Video Diffusion Models" (arXiv:2204.03458)
  ]
  #v(0.5cm)
  #text(size: 12pt)[Google Research, April 2022]
  #v(2cm)
  #block(width: 80%, fill: luma(245), inset: 1.5em, radius: 8pt)[
    #text(size: 11pt)[
      *Abstract*: This document provides a comprehensive technical analysis of the Video Diffusion Models paper,
      the first work to successfully apply diffusion models to video generation. We cover all mathematical
      foundations, architectural details, training procedures, and sampling algorithms with sufficient depth
      for implementing the model from scratch in PyTorch.
    ]
  ]
  #v(2cm)
  #text(size: 11pt)[
    *Document Version*: 1.0 \
    *Last Updated*: 2024
  ]
]

#pagebreak()

// Table of Contents
#outline(
  title: [Table of Contents],
  indent: 2em,
  depth: 3,
)

#pagebreak()

// Include chapters
#include "chapters/01_introduction.typ"
#pagebreak()
#include "chapters/02_background.typ"
#pagebreak()
#include "chapters/03_architecture.typ"
#pagebreak()
#include "chapters/04_training.typ"
#pagebreak()
#include "chapters/05_sampling.typ"
#pagebreak()
#include "chapters/06_experiments.typ"
#pagebreak()
#include "chapters/07_implementation.typ"
