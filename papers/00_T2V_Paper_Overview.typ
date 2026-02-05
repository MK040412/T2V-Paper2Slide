// Text-to-Video Diffusion Models: Comprehensive Paper Overview
// Organized by Year and Importance (Citation Count & Impact)

#set document(title: "Text-to-Video Diffusion Models: Paper Overview", author: "T2V-Paper2Slide")
#set page(paper: "a4", margin: 2cm)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")

#align(center)[
  #text(size: 24pt, weight: "bold")[Text-to-Video Diffusion Models]
  #v(0.5em)
  #text(size: 16pt)[Comprehensive Paper Overview for Implementation]
  #v(0.5em)
  #text(size: 12pt, style: "italic")[Ordered by Year and Research Impact]
]

#v(2em)

#outline(title: "Contents", indent: 2em)

#pagebreak()

= Introduction

This document provides a comprehensive overview of influential Text-to-Video (T2V) diffusion model papers, organized chronologically from 2022 to 2024. Each paper is analyzed with focus on:

- *Core Architecture*: Model design and key components
- *Technical Innovation*: Novel contributions to the field
- *Training Strategy*: Data, compute, and methodology
- *Implementation Complexity*: Difficulty level for from-scratch implementation

The papers are ranked by citation count and research impact within each year.

#v(1em)

= 2022: Foundation Year

== Video Diffusion Models (VDM)
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, David J. Fleet \
  *Affiliation*: Google Research \
  *ArXiv*: 2204.03458 (April 2022) \
  *Citations*: ~1500+ \
  *Impact*: #text(fill: red, weight: "bold")[FOUNDATIONAL]
]

=== Key Contributions
1. *First comprehensive video diffusion model* - Extended image diffusion to video domain
2. *Space-time factorized U-Net* - Efficient architecture for spatio-temporal modeling
3. *Joint image-video training* - Strategy to prevent concept forgetting
4. *Gradient-based conditional sampling* - For text-conditioned generation

=== Architecture Overview
```
Input: x ∈ ℝ^(B × T × H × W × C)  // Batch × Time × Height × Width × Channels

U-Net Architecture:
├── Spatial Blocks (2D Conv + Self-Attention)
├── Temporal Blocks (1D Conv along time axis)
└── Cross-Attention for text conditioning

Key Innovation: Factorized attention
- Spatial attention: Attend over (H × W) for each frame
- Temporal attention: Attend over T for each spatial position
```

=== Training Details
- *Dataset*: Internal video dataset
- *Resolution*: 64×64 → 256×256 (with super-resolution)
- *Frame count*: 16 frames
- *Conditioning*: CLIP text encoder

=== Implementation Priority: ★★★★★ (Essential Foundation)

#v(1em)

== CogVideo
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, Jie Tang \
  *Affiliation*: Tsinghua University, Beijing Academy of AI \
  *ArXiv*: 2205.15868 (May 2022) \
  *Citations*: ~800+ \
  *Impact*: #text(fill: orange, weight: "bold")[HIGH - First Large-scale T2V Transformer]
]

=== Key Contributions
1. *Large-scale transformer-based T2V* - Extended CogView2 (T2I) to video
2. *Multi-frame-rate hierarchical training* - Improves text-video alignment
3. *Dual-channel attention* - Handles text and video tokens separately
4. *Inherit pre-trained T2I knowledge* - Efficient transfer learning

=== Architecture Overview
```
CogVideo Architecture (9B parameters):
├── Frozen CogView2 (T2I backbone)
├── Trainable temporal attention layers
├── VQ-VAE for video tokenization
└── Hierarchical generation:
    ├── Stage 1: Generate keyframes
    └── Stage 2: Frame interpolation (recursive)
```

=== Technical Details
- *Model size*: 9.4B parameters
- *Training data*: 5.4M text-video pairs (internal)
- *Resolution*: 480×480
- *Frames*: Variable (hierarchical)

=== Implementation Priority: ★★★☆☆ (Transformer-based alternative)

#v(1em)

== Make-A-Video
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, et al. \
  *Affiliation*: Meta AI \
  *ArXiv*: 2209.14792 (September 2022) \
  *Venue*: ICLR 2023 \
  *Citations*: ~1200+ \
  *Impact*: #text(fill: red, weight: "bold")[GROUNDBREAKING]
]

=== Key Contributions
1. *No text-video paired data required* - Learn motion from unlabeled video
2. *Pseudo-3D convolutions* - Factorized spatio-temporal layers
3. *Frame interpolation network* - Smooth high frame-rate generation
4. *Super-resolution cascade* - Multi-stage quality enhancement

=== Architecture Overview
```
Make-A-Video Pipeline:
┌─────────────────────────────────────────────────────┐
│ Stage 1: Text-to-Image (Pre-trained DALL-E style)   │
│   - P(x | y): Image prior network                   │
│   - Decoder: Image generation                       │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Stage 2: Spatiotemporal Layers (Learned from video) │
│   - Conv2D → Pseudo-3D Conv (Conv2D + Conv1D)       │
│   - Spatial Attn → Spatial + Temporal Attn          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Stage 3: Frame Interpolation Network                │
│   - Increase temporal resolution                    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Stage 4: Super Resolution (Spatial + Temporal)      │
│   - SR_l: Low-res → Mid-res                         │
│   - SR_h: Mid-res → High-res                        │
└─────────────────────────────────────────────────────┘
```

=== Pseudo-3D Convolution Details
```python
# Conceptual implementation
class Pseudo3DConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        self.spatial_conv = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.temporal_conv = nn.Conv1d(out_ch, out_ch, kernel_size=3)

    def forward(self, x):  # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # Spatial convolution
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.spatial_conv(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=T)
        # Temporal convolution
        x = rearrange(x, 'b c t h w -> (b h w) c t')
        x = self.temporal_conv(x)
        x = rearrange(x, '(b h w) c t -> b c t h w', h=H, w=W)
        return x
```

=== Implementation Priority: ★★★★★ (Key architectural innovation)

#v(1em)

== Imagen Video
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, et al. \
  *Affiliation*: Google Research \
  *ArXiv*: 2210.02303 (October 2022) \
  *Citations*: ~1000+ \
  *Impact*: #text(fill: red, weight: "bold")[STATE-OF-THE-ART 2022]
]

=== Key Contributions
1. *Cascaded video diffusion models* - 7 sub-models for progressive refinement
2. *v-parameterization* - Improved training stability and sample quality
3. *Conditioning augmentation* - Better generalization
4. *Progressive distillation* - Faster sampling

=== Architecture Overview (7 Model Cascade)
```
Imagen Video Cascade:
┌──────────────────────────────────────────────────────────┐
│ 1. Base Model: 16×40×24 (T×H×W), 5.6B params            │
│    - Text conditioning via T5-XXL encoder               │
│    - Video U-Net with temporal attention                │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 2. Temporal SR: 16→64 frames (×4)                       │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 3-4. Spatial SR: 40×24 → 80×48 → 320×192                │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 5. Temporal SR: 64→128 frames (×2)                      │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ 6-7. Spatial SR: 320×192 → 640×384 → 1280×768           │
└──────────────────────────────────────────────────────────┘

Final Output: 128 frames @ 1280×768, 24fps (~5.3 seconds)
Total Parameters: ~11.6B across all models
```

=== v-Parameterization
```
Standard: ε-prediction → ε_θ(x_t, t)
v-prediction: v_θ(x_t, t) where v = α_t · ε - σ_t · x

Benefits:
- More stable training at low noise levels
- Better for high-resolution generation
- Improved color consistency
```

=== Implementation Priority: ★★★★☆ (Complex but influential)

#v(1em)

== LVDM (Latent Video Diffusion Models)
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, Qifeng Chen \
  *Affiliation*: HKUST, Tencent AI Lab \
  *ArXiv*: 2211.13221 (November 2022) \
  *Citations*: ~500+ \
  *Impact*: #text(fill: orange, weight: "bold")[HIGH - Latent Space Pioneer]
]

=== Key Contributions
1. *3D latent space diffusion* - Efficient video generation
2. *Hierarchical generation* - 1000+ frame videos possible
3. *Conditional latent perturbation* - Reduces quality degradation
4. *Lightweight architecture* - Practical for research use

=== Architecture Overview
```
LVDM Pipeline:
┌─────────────────────────────────────────────────────────┐
│ Video Encoder (3D VAE)                                  │
│   Input: x ∈ ℝ^(B×T×H×W×3)                              │
│   Output: z ∈ ℝ^(B×T'×H'×W'×C), compression ~8×8×1     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Latent Diffusion Model                                  │
│   - 3D U-Net in latent space                            │
│   - Spatial + Temporal attention                        │
│   - Text conditioning via CLIP                          │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ Video Decoder (3D VAE Decoder)                          │
│   Reconstructs video from denoised latent               │
└─────────────────────────────────────────────────────────┘

Hierarchical Extension:
├── Keyframe generation (sparse temporal)
└── Interpolation (fill intermediate frames)
```

=== Implementation Priority: ★★★★★ (Practical and efficient)

#v(1em)

== MagicVideo
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Daquan Zhou, Weimin Wang, Hanshu Yan, et al. \
  *Affiliation*: ByteDance \
  *ArXiv*: 2211.11018 (November 2022) \
  *Citations*: ~400+ \
  *Impact*: #text(fill: blue, weight: "bold")[NOTABLE]
]

=== Key Contributions
1. *Efficient latent space generation* - Using pretrained VAE
2. *Adaptor layers* - Bridge T2I and video generation
3. *Directed self-attention* - Temporal consistency via causal attention

=== Implementation Priority: ★★★☆☆ (Good efficiency techniques)

#pagebreak()

= 2023: Rapid Advancement Year

== Align Your Latents (Video LDM)
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, et al. \
  *Affiliation*: Stability AI, NVIDIA, LMU Munich \
  *Venue*: CVPR 2023 \
  *ArXiv*: 2304.08818 \
  *Citations*: ~700+ \
  *Impact*: #text(fill: red, weight: "bold")[INFLUENTIAL - Foundation for SVD]
]

=== Key Contributions
1. *Temporal layers on frozen T2I* - Efficient video fine-tuning
2. *Temporal attention and convolution* - After each spatial block
3. *Long video via autoregressive* - Context-aware generation
4. *Driving video synthesis* - Demonstrated practical application

=== Architecture Overview
```
Video LDM Architecture:
┌────────────────────────────────────────────────────────────┐
│ Pre-trained Stable Diffusion (FROZEN)                      │
│   ├── Spatial Conv blocks                                  │
│   ├── Spatial Self-Attention                               │
│   └── Cross-Attention (text)                               │
└────────────────────────────────────────────────────────────┘
                          +
┌────────────────────────────────────────────────────────────┐
│ Temporal Layers (TRAINABLE)                                │
│   ├── Temporal Conv (after each spatial conv)              │
│   │     Conv2D output → reshape → Conv1D → reshape         │
│   └── Temporal Self-Attention (after each spatial attn)    │
│         - Attention over time dimension                    │
│         - Preserves spatial structure                      │
└────────────────────────────────────────────────────────────┘
```

=== Implementation Priority: ★★★★★ (Direct path to SVD)

#v(1em)

== Text2Video-Zero
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, et al. \
  *Affiliation*: Picsart AI Research \
  *Venue*: ICCV 2023 (Oral) \
  *ArXiv*: 2303.13439 \
  *Citations*: ~600+ \
  *Impact*: #text(fill: orange, weight: "bold")[HIGH - Zero-shot Innovation]
]

=== Key Contributions
1. *Zero-shot video generation* - No video training required
2. *Cross-frame attention* - Temporal consistency mechanism
3. *Motion dynamics injection* - Via latent warping
4. *Applicable to any T2I model* - Universal approach

=== Architecture Overview
```
Text2Video-Zero (Training-Free):
┌──────────────────────────────────────────────────────────┐
│ Modified Stable Diffusion Inference:                     │
│                                                          │
│ 1. Latent Initialization:                                │
│    - First frame: Random noise z_0                       │
│    - Frame k: z_k = warp(z_0, motion_field) + noise      │
│                                                          │
│ 2. Cross-Frame Attention (replace self-attention):       │
│    Q_k = W_q · z_k        (current frame)                │
│    K = W_k · z_0          (first frame - anchor)         │
│    V = W_v · z_0          (first frame - anchor)         │
│    Attn = softmax(Q_k · K^T / √d) · V                   │
│                                                          │
│ 3. Background Smoothing:                                 │
│    - Apply temporal smoothing to background regions      │
└──────────────────────────────────────────────────────────┘
```

=== Implementation Priority: ★★★★☆ (Easy to implement, good baseline)

#v(1em)

== AnimateDiff
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, et al. \
  *Affiliation*: CUHK, Shanghai AI Lab \
  *ArXiv*: 2307.04725 (July 2023) \
  *Citations*: ~800+ \
  *Impact*: #text(fill: red, weight: "bold")[HIGHLY INFLUENTIAL - Community Favorite]
]

=== Key Contributions
1. *Plug-and-play motion module* - Works with any personalized T2I
2. *Domain Adapter* - Bridge different model domains
3. *MotionLoRA* - Efficient motion customization
4. *Training on WebVid-10M* - Accessible dataset

=== Architecture Overview
```
AnimateDiff Architecture:
┌────────────────────────────────────────────────────────────┐
│ Base T2I Model (e.g., Stable Diffusion, any checkpoint)    │
│   └── U-Net with spatial attention                         │
└────────────────────────────────────────────────────────────┘
                          +
┌────────────────────────────────────────────────────────────┐
│ Motion Module (Plug-in)                                    │
│   ├── Inserted after each transformer block                │
│   ├── Temporal Self-Attention:                             │
│   │     Input: (B×H×W, T, C) - reshape from (B, T, H, W, C)│
│   │     Attention over T dimension                         │
│   │     Positional embedding: Sinusoidal (temporal)        │
│   └── Projection layers for dimension matching             │
└────────────────────────────────────────────────────────────┘
                          +
┌────────────────────────────────────────────────────────────┐
│ Domain Adapter (Optional)                                  │
│   - Aligns feature distributions                           │
│   - Small trainable layers                                 │
└────────────────────────────────────────────────────────────┘

Motion Module Details:
- Parameters: ~400M (motion module only)
- Training: WebVid-10M dataset
- Output: 16 frames @ 512×512
```

=== Implementation Priority: ★★★★★ (Excellent for experimentation)

#v(1em)

== ModelScope Text-to-Video
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, et al. \
  *Affiliation*: Alibaba Group \
  *ArXiv*: 2308.06571 (August 2023) \
  *Citations*: ~300+ \
  *Impact*: #text(fill: orange, weight: "bold")[HIGH - Open Source Leader]
]

=== Key Contributions
1. *Open-source T2V model* - Publicly available weights
2. *Spatio-temporal blocks* - Consistent frame generation
3. *Flexible frame count* - Adaptable during inference
4. *Foundation for ZeroScope* - Inspired community improvements

=== Architecture Overview
```
ModelScope T2V (1.7B parameters):
├── VQGAN Encoder/Decoder
├── CLIP Text Encoder
└── Denoising U-Net:
    ├── Spatial Conv2D + GroupNorm + SiLU
    ├── Spatial Self-Attention
    ├── Temporal Conv1D (0.5B params for temporal)
    └── Temporal Self-Attention
```

=== Implementation Priority: ★★★★☆ (Good open-source reference)

#v(1em)

== LaVie
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Yaohui Wang, Xinyuan Chen, Xin Ma, et al. (20 authors) \
  *Affiliation*: Shanghai AI Lab, NTU, CUHK \
  *ArXiv*: 2309.15103 (September 2023) \
  *Venue*: IJCV 2024 \
  *Citations*: ~400+ \
  *Impact*: #text(fill: orange, weight: "bold")[HIGH - Comprehensive Pipeline]
]

=== Key Contributions
1. *Cascaded latent diffusion* - Base → Interpolation → SR
2. *Rotary positional encoding (RoPE)* - For temporal attention
3. *Joint image-video fine-tuning* - Quality preservation
4. *Vimeo25M dataset* - 25M high-quality text-video pairs

=== Architecture Overview
```
LaVie Cascade:
┌──────────────────────────────────────────────────────────┐
│ Base T2V Model (400M params)                             │
│   - 16 frames @ 320×512                                  │
│   - Temporal self-attention with RoPE                    │
│   - Joint image-video training                           │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ Temporal Interpolation Model                             │
│   - 16 → 61 frames                                       │
│   - Condition on sparse keyframes                        │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ Video Super-Resolution                                   │
│   - 320×512 → 1280×2048                                  │
│   - Spatial upscaling with temporal consistency          │
└──────────────────────────────────────────────────────────┘
```

=== Implementation Priority: ★★★★☆ (Good cascade reference)

#v(1em)

== VideoCrafter1
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Haoxin Chen, Menghan Xia, Yingqing He, et al. \
  *Affiliation*: Tencent AI Lab \
  *ArXiv*: 2310.19512 (October 2023) \
  *Citations*: ~300+ \
  *Impact*: #text(fill: blue, weight: "bold")[NOTABLE - Open Source]
]

=== Key Contributions
1. *Open diffusion models* - Weights publicly available
2. *High-quality video generation* - 1024×576 resolution
3. *Text and image conditioning* - Flexible input

=== Implementation Priority: ★★★☆☆ (Good reference)

#v(1em)

== Stable Video Diffusion (SVD)
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Andreas Blattmann, Tim Dockhorn, Sumith Kulal, et al. \
  *Affiliation*: Stability AI \
  *ArXiv*: 2311.15127 (November 2023) \
  *Citations*: ~600+ \
  *Impact*: #text(fill: red, weight: "bold")[STATE-OF-THE-ART Open Source]
]

=== Key Contributions
1. *Systematic training strategy* - Three-stage pipeline
2. *Data curation methodology* - Quality over quantity
3. *Image-to-video focus* - Practical application
4. *Open weights* - Community accessible

=== Three-Stage Training
```
Stage 1: Text-to-Image Pretraining
├── Use Stable Diffusion 2.1 as base
└── Strong image generation foundation

Stage 2: Video Pretraining (Large Video Dataset)
├── Add temporal layers
├── Train on Large Video Dataset (LVD)
│   - 580M video clips (before filtering)
│   - Systematic filtering: cut detection, motion,
│     text, aesthetics, CLIP score
│   - Final: High-quality subset
└── Learn general video dynamics

Stage 3: High-Quality Video Finetuning
├── Curated high-quality video dataset
├── Higher resolution training
└── Improved temporal coherence
```

=== Architecture Details
```
SVD Architecture (1.52B parameters):
├── Base: Stable Diffusion 2.1 spatial layers
├── Temporal Convolution (after each spatial conv):
│   - 1D conv along time axis
│   - Residual connection
├── Temporal Attention (after each spatial attention):
│   - Self-attention over time
│   - 656M parameters for temporal processing
└── Conditioning:
    ├── Image: First frame encoding
    ├── FPS: Fourier encoding
    └── Motion bucket: Motion intensity control
```

=== Model Variants
- *SVD*: 14 frames @ 576×1024
- *SVD-XT*: 25 frames @ 576×1024

=== Implementation Priority: ★★★★★ (Best open-source option)

#pagebreak()

= 2024: Scaling Era

== VideoCrafter2
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Haoxin Chen, Yong Zhang, Xiaodong Cun, et al. \
  *Affiliation*: Tencent AI Lab \
  *Venue*: CVPR 2024 \
  *ArXiv*: 2401.09047 (January 2024) \
  *Citations*: ~200+ \
  *Impact*: #text(fill: orange, weight: "bold")[HIGH - Data Efficiency]
]

=== Key Contributions
1. *Low-quality video + high-quality image* - Novel training scheme
2. *Disentangled spatial-temporal learning* - Better generalization
3. *Improved visual quality* - State-of-the-art on benchmarks

=== Training Strategy
```
VideoCrafter2 Training Scheme:
┌──────────────────────────────────────────────────────────┐
│ Problem: WebVid-10M has low visual quality               │
│ Solution: Disentangle appearance from motion             │
└──────────────────────────────────────────────────────────┘

Stage 1: Spatial Learning (High-quality images)
├── Train spatial layers on high-quality images
├── Freeze temporal layers
└── Learn good appearance/texture

Stage 2: Temporal Learning (Low-quality videos)
├── Freeze spatial layers
├── Train temporal layers on WebVid-10M
└── Learn motion dynamics only

Result: High visual quality + Good motion
```

=== Implementation Priority: ★★★★☆ (Practical training insights)

#v(1em)

== Sora
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: OpenAI Team \
  *Affiliation*: OpenAI \
  *Release*: February 2024 (Technical Report) \
  *Citations*: N/A (no formal paper) \
  *Impact*: #text(fill: red, weight: "bold")[PARADIGM SHIFT]
]

=== Key Contributions
1. *Diffusion Transformer (DiT) for video* - Scalable architecture
2. *Native aspect ratio training* - Flexible output sizes
3. *Spacetime patches* - Unified video-image representation
4. *Emergent world simulation* - Physics understanding

=== Architecture Overview
```
Sora Architecture (Estimated):
┌──────────────────────────────────────────────────────────┐
│ Video Compression Autoencoder                            │
│   - Compresses both spatial AND temporal dimensions      │
│   - Significant reduction enabling minute-long videos    │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ Spacetime Patchification                                 │
│   - Extract 3D patches from compressed video             │
│   - Patches as transformer tokens                        │
│   - Works for any resolution/duration/aspect ratio       │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ Diffusion Transformer (DiT)                              │
│   - Pure transformer (no U-Net convolutions)             │
│   - 3D positional encodings (spatiotemporal)             │
│   - Text conditioning via large language model           │
│   - Scales with compute (verified scaling laws)          │
└──────────────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────────────┐
│ Video Decoder                                            │
│   - Decompress latent to pixel space                     │
│   - Output: Up to 1080p, up to 1 minute                  │
└──────────────────────────────────────────────────────────┘
```

=== Key Technical Details
```
Spacetime Patches:
- Video v ∈ ℝ^(T × H × W × 3)
- Compress to z ∈ ℝ^(T' × H' × W' × C)  (spatiotemporal AE)
- Patchify: patches ∈ ℝ^(N × D) where N = (T'/p_t) × (H'/p_h) × (W'/p_w)
- Each patch = transformer token

Variable Duration/Resolution:
- No fixed positional embeddings
- Generate pos_embed from (t, h, w) coordinates
- Train on native aspect ratios (no cropping)

Conditioning:
- Detailed captions (DALL-E 3 recaptioning)
- Language model text encoder
```

=== Implementation Priority: ★★★★★ (Future direction, implement DiT first)

#v(1em)

== Open-Sora (Community)
#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *Authors*: Various (Colossal-AI, HPC-AI Tech) \
  *ArXiv*: 2412.20404, 2503.09642 \
  *Impact*: #text(fill: orange, weight: "bold")[HIGH - Open Reproduction]
]

=== Key Contributions
1. *Open-source Sora reproduction* - Democratizing video generation
2. *Efficient training* - \$200k for commercial-level quality
3. *Full pipeline release* - Data, code, weights

=== Implementation Priority: ★★★★☆ (Practical Sora-like implementation)

#pagebreak()

= Summary: Implementation Roadmap

== Recommended Implementation Order

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  align: (left, center, center, left),
  [*Paper*], [*Year*], [*Priority*], [*Reason*],
  [Video Diffusion Models], [2022], [1], [Foundation - understand basics],
  [LVDM], [2022], [2], [Latent space efficiency],
  [Make-A-Video], [2022], [3], [Pseudo-3D convolutions],
  [AnimateDiff], [2023], [4], [Plug-and-play architecture],
  [Video LDM], [2023], [5], [Temporal layer design],
  [Stable Video Diffusion], [2023], [6], [State-of-the-art open source],
  [Sora (via DiT)], [2024], [7], [Future: transformer-based],
)

== Key Components to Implement

1. *3D VAE* - Spatiotemporal compression
2. *Temporal Attention* - Frame-to-frame consistency
3. *Pseudo-3D Convolutions* - Efficient factorization
4. *Cascaded Generation* - Multi-stage refinement
5. *Diffusion Transformer* - Scalable architecture

#pagebreak()

= References

== 2022 Papers
1. Ho, J., et al. "Video Diffusion Models." arXiv:2204.03458
2. Hong, W., et al. "CogVideo: Large-scale Pretraining for Text-to-Video Generation." arXiv:2205.15868
3. Singer, U., et al. "Make-A-Video: Text-to-Video Generation without Text-Video Data." ICLR 2023, arXiv:2209.14792
4. Ho, J., et al. "Imagen Video: High Definition Video Generation with Diffusion Models." arXiv:2210.02303
5. He, Y., et al. "Latent Video Diffusion Models for High-Fidelity Long Video Generation." arXiv:2211.13221
6. Zhou, D., et al. "MagicVideo: Efficient Video Generation With Latent Diffusion Models." arXiv:2211.11018

== 2023 Papers
7. Blattmann, A., et al. "Align Your Latents: High-Resolution Video Synthesis with Latent Diffusion Models." CVPR 2023
8. Khachatryan, L., et al. "Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators." ICCV 2023
9. Guo, Y., et al. "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models." arXiv:2307.04725
10. Wang, J., et al. "ModelScope Text-to-Video Technical Report." arXiv:2308.06571
11. Wang, Y., et al. "LAVIE: High-Quality Video Generation with Cascaded Latent Diffusion Models." arXiv:2309.15103
12. Chen, H., et al. "VideoCrafter1: Open Diffusion Models for High-Quality Video Generation." arXiv:2310.19512
13. Blattmann, A., et al. "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets." arXiv:2311.15127

== 2024 Papers
14. Chen, H., et al. "VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models." CVPR 2024
15. OpenAI. "Video Generation Models as World Simulators." Technical Report, February 2024
16. Open-Sora Team. "Open-Sora: Democratizing Efficient Video Production for All." arXiv:2412.20404

== Survey Papers
17. Xing, Z., et al. "A Survey on Video Diffusion Models." ACM Computing Surveys, 2023
18. Chen, H., et al. "Survey of Video Diffusion Models." arXiv:2405.03150

