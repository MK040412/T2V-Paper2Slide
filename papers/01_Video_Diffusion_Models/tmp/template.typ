// Video Diffusion Models - Detailed Technical Analysis Template
// Using comprehensive Typst packages for visualization

// Package imports
#import "@preview/ctheorems:1.1.3": *
#import "@preview/showybox:2.0.3": showybox
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.2"
#import "@preview/tablex:0.0.9": tablex, rowspanx, colspanx, hlinex, vlinex

// Theorem environment setup
#show: thmrules.with(qed-symbol: $square$)

#let definition = thmbox("definition", "Definition", fill: rgb("#e8f4e8"), base_level: 1)
#let theorem = thmbox("theorem", "Theorem", fill: rgb("#fff3cd"), base_level: 1)
#let lemma = thmbox("lemma", "Lemma", fill: rgb("#e3f2fd"), base_level: 1)
#let corollary = thmbox("corollary", "Corollary", fill: rgb("#f3e5f5"), base_level: 1)
#let remark = thmbox("remark", "Remark", fill: rgb("#fff8e1"), base_level: 1)
#let example = thmbox("example", "Example", fill: rgb("#e0f7fa"), base_level: 1)
#let algorithm = thmbox("algorithm", "Algorithm", fill: rgb("#fce4ec"), base_level: 1)

// Custom styling functions
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

#let codeblock(title: "Code", body) = showybox(
  frame: (border-color: gray.darken(20%), title-color: gray.lighten(80%), body-color: gray.lighten(95%)),
  title-style: (color: black, weight: "bold"),
  title: title,
  body
)

#let mathblock(title: "Mathematical Formulation", body) = showybox(
  frame: (border-color: purple.darken(20%), title-color: purple.lighten(80%), body-color: purple.lighten(95%)),
  title-style: (color: black, weight: "bold"),
  title: title,
  body
)

// Architecture diagram helpers using CeTZ
#let draw-layer-box(x, y, w, h, label, fill-color) = {
  import cetz.draw: *
  rect((x, y), (x + w, y + h), fill: fill-color, stroke: black)
  content((x + w/2, y + h/2), text(size: 8pt)[#label])
}

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

// Inline code styling
#show raw.where(block: false): it => {
  box(
    fill: luma(240),
    inset: (x: 3pt, y: 0pt),
    radius: 2pt,
    it
  )
}

// Export template functions
#let chapter-template(
  title: "",
  subtitle: "",
  chapter-num: 1,
  body
) = {
  set document(
    title: "VDM Chapter " + str(chapter-num) + ": " + title,
    author: "T2V-Paper2Slide",
  )

  set page(
    paper: "a4",
    margin: (x: 2cm, y: 2.5cm),
    header: context {
      if counter(page).get().first() > 1 [
        #set text(size: 9pt)
        _Chapter #chapter-num: #title_
        #h(1fr)
        #counter(page).display()
      ]
    },
    footer: context {
      align(center)[
        #set text(size: 8pt, fill: gray)
        Video Diffusion Models - Technical Analysis | Page #counter(page).display()
      ]
    }
  )

  set text(font: "New Computer Modern", size: 11pt)
  set heading(numbering: (..nums) => {
    let nums = nums.pos()
    if nums.len() == 1 {
      str(chapter-num) + "."
    } else {
      str(chapter-num) + "." + nums.slice(1).map(str).join(".")
    }
  })
  set math.equation(numbering: "(1)")
  set par(justify: true, leading: 0.65em)

  // Title block
  align(center)[
    #v(1cm)
    #text(size: 12pt, fill: blue.darken(20%))[*Chapter #chapter-num*]
    #v(0.3cm)
    #text(size: 24pt, weight: "bold")[#title]
    #v(0.2cm)
    #text(size: 14pt, style: "italic")[#subtitle]
    #v(0.5cm)
    #line(length: 60%, stroke: 1pt + gray)
    #v(1cm)
  ]

  body
}
