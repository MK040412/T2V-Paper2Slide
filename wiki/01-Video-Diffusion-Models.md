# Video Diffusion Models (VDM)

**The First Comprehensive Video Diffusion Model**

> Ho, Salimans, Gritsenko, Chan, Norouzi, Fleet. "Video Diffusion Models." arXiv:2204.03458, April 2022.

## Overview

Video Diffusion Models is the foundational paper that successfully extended diffusion models from images to videos. It introduced several key innovations that became standard in subsequent work.

## Key Contributions

### 1. Space-Time Factorized 3D U-Net
- Converts 2D image diffusion architecture to 3D
- Factorizes attention into spatial and temporal components
- Reduces computational complexity from O(T²H²W²) to O(TH²W² + HWT²)

### 2. Joint Image-Video Training
- Train on both video and image data simultaneously
- Reduces gradient variance
- Improves sample quality significantly

### 3. Reconstruction Guidance
- Novel conditional sampling method
- Enables video extension and super-resolution
- Outperforms replacement method by 3.3× in FVD

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    3D U-Net Architecture                     │
│                                                              │
│   Input: z_t (noisy video) + timestep t + conditioning c    │
│                          ↓                                   │
│   ┌────────────────────────────────────────────────────┐    │
│   │                    ENCODER                          │    │
│   │  ResBlock → SpatialAttn → TemporalAttn → Downsample│    │
│   │  (repeated for each resolution level)               │    │
│   └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│   ┌────────────────────────────────────────────────────┐    │
│   │                    MIDDLE                           │    │
│   │  ResBlock → SpatialAttn → TemporalAttn → ResBlock  │    │
│   └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│   ┌────────────────────────────────────────────────────┐    │
│   │                    DECODER                          │    │
│   │  Concat(skip) → ResBlock → SpatialAttn → TempAttn  │    │
│   │  → Upsample (repeated for each resolution level)   │    │
│   └────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│   Output: ε_θ (predicted noise) or x̂ (predicted clean video)│
└─────────────────────────────────────────────────────────────┘
```

## Results Summary

### UCF101 (Unconditional)
| Metric | Previous SOTA | VDM | Improvement |
|--------|--------------|-----|-------------|
| FID ↓ | 3431 | **295** | 11.6× |
| IS ↑ | 28.87 | **57** | 2× |

### BAIR Robot Pushing (Video Prediction)
| Metric | Previous SOTA | VDM | Improvement |
|--------|--------------|-----|-------------|
| FVD ↓ | 86.9 | **66.92** | 1.3× |

### Kinetics-600 (Video Prediction)
| Metric | Previous SOTA | VDM | Improvement |
|--------|--------------|-----|-------------|
| FVD ↓ | 25.4 | **16.2** | 1.6× |
| IS ↑ | 12.54 | **15.64** | 1.2× |

## Full Documentation

For complete technical details, see:
- [Full Analysis](../papers/01_Video_Diffusion_Models/) - Complete Typst document
- [Architecture Details](../papers/01_Video_Diffusion_Models/chapters/03_architecture.typ)
- [Training Guide](../papers/01_Video_Diffusion_Models/chapters/04_training.typ)
- [Implementation Code](../papers/01_Video_Diffusion_Models/chapters/07_implementation.typ)

## Quick Links

- [arXiv Paper](https://arxiv.org/abs/2204.03458)
- [Project Page](https://video-diffusion.github.io/)

## Citation

```bibtex
@article{ho2022video,
  title={Video Diffusion Models},
  author={Ho, Jonathan and Salimans, Tim and Gritsenko, Alexey and Chan, William and Norouzi, Mohammad and Fleet, David J},
  journal={arXiv preprint arXiv:2204.03458},
  year={2022}
}
```
