# T2V-Paper2Slide: Text-to-Video Diffusion Models - From Paper to Implementation

A comprehensive repository for understanding and implementing Text-to-Video (T2V) diffusion models from scratch.

## Overview

This repository provides detailed technical documentation for implementing state-of-the-art Text-to-Video diffusion models. Each paper is analyzed with:

- Mathematical foundations
- Architecture diagrams
- Implementation details
- Code examples (from scratch)

## Repository Structure

```
T2V-Paper2Slide/
├── papers/                           # Paper analyses (Typst)
│   ├── 00_T2V_Paper_Overview.typ     # Comprehensive paper list
│   └── 01_Video_Diffusion_Models/    # VDM detailed analysis
│       ├── paper.pdf                 # Original paper
│       ├── README.md                 # Quick overview
│       ├── main.typ                  # Main Typst document
│       └── chapters/                 # Detailed chapters
│           ├── 01_introduction.typ
│           ├── 02_background.typ
│           ├── 03_architecture.typ   # 3D U-Net with diagrams
│           ├── 04_training.typ
│           ├── 05_sampling.typ       # Reconstruction guidance
│           ├── 06_experiments.typ
│           └── 07_implementation.typ # PyTorch code
├── wiki/                             # GitHub Wiki pages
│   ├── Home.md
│   ├── 01-Video-Diffusion-Models.md
│   └── Paper-List.md
├── references/                       # Reference materials
├── implementations/                  # From-scratch implementations
└── README.md
```

## Wiki

📚 **[View the Wiki](wiki/Home.md)** for comprehensive documentation and guides.

## Papers Covered (Chronological)

### 2022 - Foundation Year
| Paper | Affiliation | Key Innovation | Priority |
|-------|-------------|----------------|----------|
| Video Diffusion Models (VDM) | Google | First video diffusion | ★★★★★ |
| CogVideo | Tsinghua | Large-scale transformer T2V | ★★★☆☆ |
| Make-A-Video | Meta | Pseudo-3D conv, no text-video data | ★★★★★ |
| Imagen Video | Google | Cascaded diffusion (7 models) | ★★★★☆ |
| LVDM | HKUST/Tencent | 3D latent space, 1000+ frames | ★★★★★ |
| MagicVideo | ByteDance | Efficient latent generation | ★★★☆☆ |

### 2023 - Rapid Advancement
| Paper | Affiliation | Key Innovation | Priority |
|-------|-------------|----------------|----------|
| Video LDM (Align Your Latents) | Stability AI | Temporal layers on frozen T2I | ★★★★★ |
| Text2Video-Zero | Picsart | Zero-shot, no training | ★★★★☆ |
| AnimateDiff | CUHK/Shanghai AI | Plug-and-play motion module | ★★★★★ |
| ModelScope T2V | Alibaba | Open-source reference | ★★★★☆ |
| LaVie | Shanghai AI Lab | Cascaded LDM + Vimeo25M | ★★★★☆ |
| VideoCrafter1 | Tencent | Open weights | ★★★☆☆ |
| Stable Video Diffusion | Stability AI | Best open-source | ★★★★★ |

### 2024 - Scaling Era
| Paper | Affiliation | Key Innovation | Priority |
|-------|-------------|----------------|----------|
| VideoCrafter2 | Tencent | Data-efficient training | ★★★★☆ |
| Sora | OpenAI | DiT for video, 1-min generation | ★★★★★ |
| Open-Sora | Community | Open Sora reproduction | ★★★★☆ |

## Implementation Roadmap

Recommended order for implementing from scratch:

1. **Video Diffusion Models** - Understand the fundamentals
2. **LVDM** - Learn latent space efficiency
3. **Make-A-Video** - Master pseudo-3D convolutions
4. **AnimateDiff** - Implement plug-and-play architecture
5. **Video LDM** - Design temporal layers
6. **Stable Video Diffusion** - Build state-of-the-art system
7. **Sora (via DiT)** - Explore transformer-based future

## Key Components

### Core Building Blocks
- 3D VAE (Spatiotemporal compression)
- Temporal Self-Attention
- Pseudo-3D Convolutions
- Cascaded Generation
- Diffusion Transformer (DiT)

## Resources

### Official Papers
- [Video Diffusion Models](https://arxiv.org/abs/2204.03458)
- [Make-A-Video](https://arxiv.org/abs/2209.14792)
- [Imagen Video](https://arxiv.org/abs/2210.02303)
- [LVDM](https://arxiv.org/abs/2211.13221)
- [AnimateDiff](https://arxiv.org/abs/2307.04725)
- [Stable Video Diffusion](https://arxiv.org/abs/2311.15127)
- [Sora Technical Report](https://openai.com/index/video-generation-models-as-world-simulators/)

### Surveys
- [A Survey on Video Diffusion Models (ACM Computing Surveys)](https://arxiv.org/abs/2405.03150)
- [Awesome Video Diffusion Models (GitHub)](https://github.com/ChenHsing/Awesome-Video-Diffusion-Models)

## License

This repository is for educational purposes.
