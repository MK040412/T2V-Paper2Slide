# T2V-Paper2Slide Wiki

Welcome to the Text-to-Video Diffusion Models wiki! This repository provides comprehensive technical documentation for understanding and implementing state-of-the-art T2V diffusion models from scratch.

## Quick Navigation

### Paper Analyses
| Paper | Year | Status | Link |
|-------|------|--------|------|
| [Video Diffusion Models](./01-Video-Diffusion-Models) | 2022 | ✅ Complete | [Full Analysis](../papers/01_Video_Diffusion_Models/) |
| CogVideo | 2022 | 📝 Planned | - |
| Make-A-Video | 2022 | 📝 Planned | - |
| Imagen Video | 2022 | 📝 Planned | - |
| AnimateDiff | 2023 | 📝 Planned | - |
| Stable Video Diffusion | 2023 | 📝 Planned | - |
| Sora | 2024 | 📝 Planned | - |

### Implementation Guides
- [Diffusion Basics](./Diffusion-Basics) - Mathematical foundations
- [3D U-Net Architecture](./3D-UNet-Architecture) - Space-time factorized networks
- [Training Guide](./Training-Guide) - Complete training procedures
- [Sampling Methods](./Sampling-Methods) - DDPM, DDIM, and reconstruction guidance

### Resources
- [Paper List](./Paper-List) - Complete chronological paper list
- [Datasets](./Datasets) - Common video generation datasets
- [Metrics](./Metrics) - FVD, FID, IS evaluation
- [FAQ](./FAQ) - Frequently asked questions

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+
- 16GB+ GPU memory (for training)

### Repository Structure
```
T2V-Paper2Slide/
├── papers/                    # Detailed paper analyses
│   ├── 00_T2V_Paper_Overview.typ  # Overview of all papers
│   └── 01_Video_Diffusion_Models/ # VDM complete analysis
│       ├── paper.pdf         # Original paper
│       ├── main.typ          # Main Typst document
│       └── chapters/         # Detailed chapters
├── implementations/          # From-scratch implementations
├── references/               # Additional materials
└── wiki/                     # This wiki
```

## Implementation Roadmap

Recommended learning order:

1. **Video Diffusion Models** (VDM) - Foundation paper
   - Learn basic video diffusion formulation
   - Understand 3D U-Net architecture
   - Master reconstruction guidance

2. **Latent Video Diffusion** (LVDM) - Efficiency
   - Learn latent space compression
   - Understand 3D VAE

3. **Make-A-Video** - Architecture innovations
   - Pseudo-3D convolutions
   - Text-to-video without paired data

4. **AnimateDiff** - Modularity
   - Plug-and-play motion modules
   - Works with any T2I model

5. **Stable Video Diffusion** - SOTA open source
   - Training strategies
   - Data curation

6. **Sora / DiT** - Future direction
   - Diffusion Transformers
   - Scalable architecture

## Contributing

Feel free to contribute:
- Paper analyses
- Implementation improvements
- Bug fixes
- Documentation

## Contact

For questions and discussions, please open an issue on the repository.
