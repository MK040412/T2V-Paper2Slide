# Video Diffusion Models (VDM)

**The First Comprehensive Video Diffusion Model**

## Paper Information

| Item | Details |
|------|---------|
| **Title** | Video Diffusion Models |
| **Authors** | Jonathan Ho*, Tim Salimans*, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet |
| **Affiliation** | Google Research |
| **ArXiv** | [2204.03458](https://arxiv.org/abs/2204.03458) (April 2022) |
| **Project Page** | [video-diffusion.github.io](https://video-diffusion.github.io/) |
| **Citations** | ~1500+ |

## Abstract

Generating temporally coherent high fidelity video is an important milestone in generative modeling research. This paper proposes a diffusion model for video generation that shows very promising initial results. The model is a natural extension of the standard image diffusion architecture, enabling joint training from image and video data.

## Key Contributions

1. **First Video Diffusion Model** - Extended standard image diffusion to video domain with 3D U-Net
2. **Space-Time Factorized Architecture** - Efficient spatio-temporal modeling
3. **Joint Image-Video Training** - Reduces gradient variance, speeds up optimization
4. **Reconstruction-Guided Sampling** - Novel conditional sampling for temporal/spatial extension
5. **State-of-the-Art Results** - On UCF101, BAIR Robot Pushing, Kinetics-600

## Directory Structure

```
01_Video_Diffusion_Models/
├── paper.pdf                    # Original paper
├── README.md                    # This file
├── main.typ                     # Main Typst document
├── chapters/
│   ├── 01_introduction.typ      # Introduction & Motivation
│   ├── 02_background.typ        # Diffusion Model Background
│   ├── 03_architecture.typ      # 3D U-Net Architecture
│   ├── 04_training.typ          # Training Methodology
│   ├── 05_sampling.typ          # Sampling & Reconstruction Guidance
│   ├── 06_experiments.typ       # Experiments & Results
│   └── 07_implementation.typ    # From-Scratch Implementation Guide
└── assets/
    └── diagrams/                # Architecture diagrams
```

## Quick Links

- [Main Document](main.typ) - Complete technical analysis
- [Architecture Details](chapters/03_architecture.typ) - 3D U-Net with diagrams
- [Implementation Guide](chapters/07_implementation.typ) - PyTorch code from scratch

## Model Specifications

| Configuration | UCF101 | BAIR | Kinetics-600 | Text-to-Video |
|--------------|--------|------|--------------|---------------|
| Resolution | 16×64×64 | 16×64×64 | 16×64×64 | 16×64×64 |
| Base Channels | 256 | 128 | 256 | 256 |
| Channel Mult. | 1,2,4,8 | 1,2,3,4 | 1,2,4,8 | 1,2,4,8 |
| Attention Res. | 8,16,32 | 8,16,32 | 8,16,32 | 8,16,32 |
| Training Steps | 60K | 660K | 220K | 700K |
| Hardware | 128 TPU-v4 | 128 TPU-v4 | 256 TPU-v4 | 128 TPU-v4 |

## Results Summary

### UCF101 (Unconditional)
| Method | FID↓ | IS↑ |
|--------|------|-----|
| Previous SOTA (TGAN-v2) | 3431 | 28.87 |
| **Video Diffusion (Ours)** | **295** | **57** |

### BAIR Robot Pushing (Video Prediction)
| Method | FVD↓ |
|--------|------|
| Previous SOTA (NUWA) | 86.9 |
| **Video Diffusion (Ours)** | **66.92** |

### Kinetics-600 (Video Prediction)
| Method | FVD↓ | IS↑ |
|--------|------|-----|
| Previous SOTA (TrIVD-GAN-FP) | 25.74 | 12.54 |
| **Video Diffusion (Ours)** | **16.2** | **15.64** |

## Citation

```bibtex
@article{ho2022video,
  title={Video Diffusion Models},
  author={Ho, Jonathan and Salimans, Tim and Gritsenko, Alexey and Chan, William and Norouzi, Mohammad and Fleet, David J},
  journal={arXiv preprint arXiv:2204.03458},
  year={2022}
}
```
