// Chapter 3: 3D U-Net Architecture

= 3D U-Net Architecture

This chapter provides a detailed analysis of the video diffusion model architecture.

== Architecture Overview

The core architecture is a *3D U-Net* that processes video data in a *space-time factorized* manner.

=== High-Level Structure

#block(fill: luma(245), inset: 1em, radius: 4pt)[
  *3D U-Net Input/Output*

  - *Input*: Noisy video $bold(z)_t in RR^(T times H times W times C)$, conditioning $bold(c)$, log-SNR $lambda_t$
  - *Output*: Denoised video prediction $hat(bold(x)) in RR^(T times H times W times C)$
]

#figure(
  ```
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        3D U-Net Architecture                           │
  │                                                                         │
  │   Input: (z_t, c, λ_t)                                                 │
  │          ↓                                                              │
  │   ┌─────────────────────────────────────────────────────────────────┐  │
  │   │                     ENCODER (Downsampling)                       │  │
  │   │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │  │
  │   │  │ Block 1 │───►│ Block 2 │───►│ Block 3 │───►│ Block K │      │  │
  │   │  │ N², M₁  │    │(N/2)²,M₂│    │(N/4)²,M₃│    │(N/2ᵏ)²,Mₖ│     │  │
  │   │  └────┬────┘    └────┬────┘    └────┬────┘    └────┬────┘      │  │
  │   │       │skip          │skip          │skip          │            │  │
  │   └───────┼──────────────┼──────────────┼──────────────┼────────────┘  │
  │           │              │              │              │                │
  │           │              │              │              ↓                │
  │   ┌───────┼──────────────┼──────────────┼──────────────────────────┐   │
  │   │       │              │              │          Middle Block     │   │
  │   └───────┼──────────────┼──────────────┼──────────────────────────┘   │
  │           │              │              │              │                │
  │   ┌───────┼──────────────┼──────────────┼──────────────┼────────────┐  │
  │   │       ↓              ↓              ↓              ↓            │  │
  │   │                     DECODER (Upsampling)                        │  │
  │   │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │  │
  │   │  │Block K' │───►│Block 3' │───►│Block 2' │───►│Block 1' │      │  │
  │   │  │  +skip  │    │  +skip  │    │  +skip  │    │  +skip  │      │  │
  │   │  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │  │
  │   └─────────────────────────────────────────────────────────────────┘  │
  │          ↓                                                              │
  │   Output: x̂ ∈ ℝ^(T×H×W×C)                                             │
  └─────────────────────────────────────────────────────────────────────────┘
  ```,
  caption: [High-level 3D U-Net architecture diagram]
)

=== Tensor Dimensions Through the Network

For a 16×64×64 video with base channels $M_1 = 256$ and multipliers $(1, 2, 4, 8)$:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    [*Stage*], [*Spatial*], [*Channels*], [*Tensor Shape*],
    [Input], [64×64], [3], [16×64×64×3],
    [After conv_in], [64×64], [256], [16×64×64×256],
    [Block 1 (down)], [32×32], [256], [16×32×32×256],
    [Block 2 (down)], [16×16], [512], [16×16×16×512],
    [Block 3 (down)], [8×8], [1024], [16×8×8×1024],
    [Block 4 (down)], [4×4], [2048], [16×4×4×2048],
    [Middle], [4×4], [2048], [16×4×4×2048],
    [Block 4' (up)], [4×4], [4096→2048], [16×4×4×2048],
    [Block 3' (up)], [8×8], [2048→1024], [16×8×8×1024],
    [Block 2' (up)], [16×16], [1024→512], [16×16×16×512],
    [Block 1' (up)], [32×32], [512→256], [16×32×32×256],
    [Output], [64×64], [3], [16×64×64×3],
  ),
  caption: [Tensor dimensions through the U-Net for 16×64×64 input]
)

== Space-Time Factorized Blocks

The key innovation is *factorizing* spatial and temporal processing.

=== Block Structure

Each U-Net block consists of:

#figure(
  ```
  ┌─────────────────────────────────────────────────────────────────┐
  │                    Space-Time Factorized Block                  │
  │                                                                 │
  │   Input: x ∈ ℝ^(T×H×W×C)                                       │
  │          ↓                                                      │
  │   ┌─────────────────────────────────────────────────────────┐  │
  │   │           Spatial Convolution (1×3×3)                    │  │
  │   │   • Process each frame independently                     │  │
  │   │   • 2D convolution with kernel (1, 3, 3)                │  │
  │   └─────────────────────────────────────────────────────────┘  │
  │          ↓                                                      │
  │   ┌─────────────────────────────────────────────────────────┐  │
  │   │           Spatial Self-Attention                         │  │
  │   │   • Attend over (H×W) positions                          │  │
  │   │   • Time axis treated as batch                           │  │
  │   └─────────────────────────────────────────────────────────┘  │
  │          ↓                                                      │
  │   ┌─────────────────────────────────────────────────────────┐  │
  │   │           Temporal Self-Attention                        │  │
  │   │   • Attend over T frames                                 │  │
  │   │   • Spatial axes treated as batch                        │  │
  │   │   • Relative position embeddings                         │  │
  │   └─────────────────────────────────────────────────────────┘  │
  │          ↓                                                      │
  │   Output: y ∈ ℝ^(T×H×W×C')                                     │
  └─────────────────────────────────────────────────────────────────┘
  ```,
  caption: [Space-time factorized block structure]
)

=== Why Factorization?

#block(fill: rgb("#e8f4e8"), inset: 1em, radius: 4pt)[
  *Computational Comparison*

  For video with $T$ frames and $H times W$ spatial resolution:

  *Full 3D Attention:*
  $ O((T dot H dot W)^2) = O(T^2 H^2 W^2) $

  *Factorized Attention:*
  $ O(T dot (H dot W)^2) + O(H dot W dot T^2) = O(T H^2 W^2 + H W T^2) $

  For $T = 16, H = W = 64$: Full = $10^12$ vs Factorized = $10^9$ → *1000× reduction*
]

== Spatial Processing

=== Spatial Convolution (1×3×3)

Each 2D convolution is converted to a *space-only 3D convolution*:

$ "Conv2D"(3 times 3) arrow.r "Conv3D"(1 times 3 times 3) $

#figure(
  ```
  Spatial Convolution Operation:

  Input: x ∈ ℝ^(T×H×W×C_in)

  Reshape to treat time as batch:
      x' = reshape(x, (T, H, W, C_in))  # No change needed

  Apply 2D conv to each frame:
      for t in range(T):
          y[t] = Conv2D(x[t], kernel_size=3)

  Equivalent to Conv3D with kernel (1, 3, 3):
      y = Conv3D(x, kernel_size=(1, 3, 3), padding=(0, 1, 1))

  Output: y ∈ ℝ^(T×H×W×C_out)
  ```,
  caption: [Spatial convolution as space-only 3D conv]
)

=== Spatial Self-Attention

Spatial attention attends over the $(H times W)$ spatial positions:

#block(fill: rgb("#fff3cd"), inset: 1em, radius: 4pt)[
  *Spatial Attention Mechanism*

  $ "SpatialAttn"(bold(x)) = "softmax"((bold(Q) bold(K)^top) / sqrt(d)) bold(V) $

  where for input $bold(x) in RR^(T times H times W times C)$:
  - Reshape: $bold(x)' in RR^((T dot H dot W) times C)$ treating $T$ as batch
  - Actually: $bold(x)' in RR^(T times (H dot W) times C)$
  - $bold(Q), bold(K), bold(V) in RR^(T times (H W) times d)$
  - Attention computed over $(H times W)$ dimension
]

#figure(
  ```
  Spatial Self-Attention Detail:

  Input: x ∈ ℝ^(T×H×W×C)

  1. Reshape: x' ∈ ℝ^(T × (H·W) × C)
     └── Each frame becomes a sequence of H·W tokens

  2. Project to Q, K, V:
     Q = x' · W_Q    ∈ ℝ^(T × (H·W) × d)
     K = x' · W_K    ∈ ℝ^(T × (H·W) × d)
     V = x' · W_V    ∈ ℝ^(T × (H·W) × d)

  3. Compute attention (batched over T):
     for each frame t:
         A[t] = softmax(Q[t] · K[t]ᵀ / √d)  ∈ ℝ^((H·W) × (H·W))
         out[t] = A[t] · V[t]               ∈ ℝ^((H·W) × d)

  4. Project and reshape:
     out' = out · W_O                        ∈ ℝ^(T × (H·W) × C)
     y = reshape(out', (T, H, W, C))

  Output: y ∈ ℝ^(T×H×W×C)
  ```,
  caption: [Spatial self-attention step-by-step]
)

== Temporal Processing

=== Temporal Self-Attention

After each spatial attention block, a temporal attention block is inserted:

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Temporal Attention Mechanism*

  $ "TemporalAttn"(bold(x)) = "softmax"((bold(Q) bold(K)^top + bold(R)) / sqrt(d)) bold(V) $

  where:
  - Input $bold(x) in RR^(T times H times W times C)$
  - Reshape: $bold(x)' in RR^((H dot W) times T times C)$ treating spatial as batch
  - $bold(Q), bold(K), bold(V) in RR^((H W) times T times d)$
  - $bold(R)$ = relative position bias matrix
]

#figure(
  ```
  Temporal Self-Attention Detail:

  Input: x ∈ ℝ^(T×H×W×C)

  1. Reshape: x' ∈ ℝ^((H·W) × T × C)
     └── Each spatial position becomes a sequence of T tokens

  2. Project to Q, K, V:
     Q = x' · W_Q    ∈ ℝ^((H·W) × T × d)
     K = x' · W_K    ∈ ℝ^((H·W) × T × d)
     V = x' · W_V    ∈ ℝ^((H·W) × T × d)

  3. Compute attention with relative position bias:
     for each spatial position (h,w):
         A[h,w] = softmax((Q[h,w] · K[h,w]ᵀ + R) / √d)  ∈ ℝ^(T × T)
         out[h,w] = A[h,w] · V[h,w]                     ∈ ℝ^(T × d)

  4. Project and reshape:
     out' = out · W_O                        ∈ ℝ^((H·W) × T × C)
     y = reshape(out', (T, H, W, C))

  Output: y ∈ ℝ^(T×H×W×C)
  ```,
  caption: [Temporal self-attention step-by-step]
)

=== Relative Position Embeddings

The paper uses *relative position embeddings* for temporal attention:

$ bold(R)_(i,j) = bold(r)_(i-j) $

where $bold(r)_k$ is a learned embedding for relative distance $k in {-(T-1), ..., 0, ..., (T-1)}$.

This allows the model to:
- Distinguish frame ordering without absolute time
- Generalize to different video lengths
- Understand temporal relationships (before/after)

#figure(
  ```
  Relative Position Bias Matrix R (T=4 example):

       t=0   t=1   t=2   t=3
  t=0 [ r_0   r_-1  r_-2  r_-3 ]
  t=1 [ r_1   r_0   r_-1  r_-2 ]
  t=2 [ r_2   r_1   r_0   r_-1 ]
  t=3 [ r_3   r_2   r_1   r_0  ]

  Where:
  - r_0: same frame (diagonal)
  - r_k (k>0): future frame
  - r_k (k<0): past frame
  ```,
  caption: [Relative position bias matrix structure]
)

== Conditioning Mechanism

=== Time and Conditioning Embedding

The conditioning information (time $lambda_t$ and signal $bold(c)$) is injected via embeddings:

#figure(
  ```
  Conditioning Embedding Pipeline:

  ┌──────────────────────────────────────────────────────────────┐
  │                    Timestep Embedding                        │
  │                                                              │
  │   λ_t (scalar) ──► Sinusoidal Encoding ──► MLP ──► emb_t    │
  │                    (Fourier features)     (layers)          │
  └──────────────────────────────────────────────────────────────┘
                              ↓
  ┌──────────────────────────────────────────────────────────────┐
  │                   Condition Embedding                        │
  │                                                              │
  │   c (text/class) ──► Encoder ──► Pooling ──► MLP ──► emb_c  │
  │                      (BERT)     (attention)                  │
  └──────────────────────────────────────────────────────────────┘
                              ↓
  ┌──────────────────────────────────────────────────────────────┐
  │                    Combined Embedding                        │
  │                                                              │
  │   emb = emb_t + emb_c                                       │
  │                                                              │
  │   (Added into each residual block)                          │
  └──────────────────────────────────────────────────────────────┘
  ```,
  caption: [Conditioning embedding pipeline]
)

=== Embedding Injection in ResBlocks

#figure(
  ```
  ResBlock with Conditioning:

  Input: x ∈ ℝ^(T×H×W×C), emb ∈ ℝ^d

  ┌─────────────────────────────────────────────────┐
  │   h = GroupNorm(x)                              │
  │   h = SiLU(h)                                   │
  │   h = Conv3D(h, 1×3×3)                          │
  │                                                 │
  │   # Inject conditioning                         │
  │   scale, shift = chunk(Linear(emb), 2)         │
  │   h = h * (1 + scale) + shift                  │
  │                                                 │
  │   h = GroupNorm(h)                              │
  │   h = SiLU(h)                                   │
  │   h = Dropout(h)                                │
  │   h = Conv3D(h, 1×3×3)                          │
  │                                                 │
  │   # Residual connection                         │
  │   if C_in ≠ C_out:                              │
  │       x = Conv3D(x, 1×1×1)                      │
  │   out = x + h                                   │
  └─────────────────────────────────────────────────┘

  Output: out ∈ ℝ^(T×H×W×C')
  ```,
  caption: [ResBlock with conditioning injection]
)

== Complete Block Diagram

=== Detailed Architecture Flow

#figure(
  ```
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                    Complete 3D U-Net Architecture                         ║
  ╠═══════════════════════════════════════════════════════════════════════════╣
  ║                                                                           ║
  ║  INPUTS                                                                   ║
  ║  ───────                                                                  ║
  ║  z_t: Noisy video         [B, T, H, W, 3]                                ║
  ║  c: Conditioning          [B, D_cond] (e.g., BERT embedding)             ║
  ║  λ_t: Log-SNR             [B, 1]                                         ║
  ║                                                                           ║
  ║  EMBEDDING                                                                ║
  ║  ─────────                                                                ║
  ║  emb_t = MLP(sinusoidal(λ_t))           [B, D_emb]                       ║
  ║  emb_c = MLP(attention_pool(c))         [B, D_emb]                       ║
  ║  emb = emb_t + emb_c                    [B, D_emb]                       ║
  ║                                                                           ║
  ║  ENCODER                                                                  ║
  ║  ───────                                                                  ║
  ║  h = Conv3D(z_t, 1×3×3)                 [B, T, H, W, M₁]                 ║
  ║                                                                           ║
  ║  # Level 1                                                                ║
  ║  for _ in range(num_res_blocks):                                          ║
  ║      h = ResBlock(h, emb)                                                 ║
  ║      h = SpatialAttn(h)                                                   ║
  ║      h = TemporalAttn(h)                                                  ║
  ║  skip₁ = h                              [B, T, H, W, M₁]                 ║
  ║  h = Downsample(h)                      [B, T, H/2, W/2, M₁]             ║
  ║                                                                           ║
  ║  # Level 2                                                                ║
  ║  h = Conv3D(h, 1×1×1, out=M₂)           [B, T, H/2, W/2, M₂]             ║
  ║  for _ in range(num_res_blocks):                                          ║
  ║      h = ResBlock(h, emb)                                                 ║
  ║      h = SpatialAttn(h)                                                   ║
  ║      h = TemporalAttn(h)                                                  ║
  ║  skip₂ = h                              [B, T, H/2, W/2, M₂]             ║
  ║  h = Downsample(h)                      [B, T, H/4, W/4, M₂]             ║
  ║                                                                           ║
  ║  # ... Continue for all K levels                                          ║
  ║                                                                           ║
  ║  MIDDLE                                                                   ║
  ║  ──────                                                                   ║
  ║  h = ResBlock(h, emb)                                                     ║
  ║  h = SpatialAttn(h)                                                       ║
  ║  h = TemporalAttn(h)                                                      ║
  ║  h = ResBlock(h, emb)                                                     ║
  ║                                                                           ║
  ║  DECODER                                                                  ║
  ║  ───────                                                                  ║
  ║  # Level K (innermost)                                                    ║
  ║  h = concat(h, skip_K)                  [B, T, H/2ᵏ, W/2ᵏ, 2·Mₖ]        ║
  ║  for _ in range(num_res_blocks + 1):                                      ║
  ║      h = ResBlock(h, emb)                                                 ║
  ║      h = SpatialAttn(h)                                                   ║
  ║      h = TemporalAttn(h)                                                  ║
  ║  h = Upsample(h)                                                          ║
  ║                                                                           ║
  ║  # ... Continue for all levels                                            ║
  ║                                                                           ║
  ║  OUTPUT                                                                   ║
  ║  ──────                                                                   ║
  ║  h = GroupNorm(h)                                                         ║
  ║  h = SiLU(h)                                                              ║
  ║  x̂ = Conv3D(h, 1×3×3, out=3)            [B, T, H, W, 3]                  ║
  ║                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
  ```,
  caption: [Complete architecture with tensor shapes]
)

== Joint Image-Video Training

=== Masking for Image Mode

A unique advantage of this architecture is easy adaptation to image-only training:

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Image Mode Masking*

  To train on independent images instead of video:
  1. Remove attention operation inside temporal attention blocks
  2. Fix attention matrix to identity (each frame attends only to itself)
  3. Effectively treats input as $T$ independent images
]

#figure(
  ```
  Joint Training Data Loading:

  ┌─────────────────────────────────────────────────────────────┐
  │   Input: Video batch + Image batch                          │
  │                                                             │
  │   Video: [v₁, v₂, ..., v_T]     T consecutive frames       │
  │   Images: [i₁, i₂, ..., i_K]    K independent frames       │
  │                                                             │
  │   Combined: [v₁, ..., v_T, i₁, ..., i_K]                   │
  │             └─────────────────┘  └────────────────┘         │
  │              Temporal attention   Identity attention        │
  │              (full connectivity)  (no cross-frame)          │
  │                                                             │
  │   Attention Mask:                                           │
  │       Frame:  1  2  3 ... T  T+1 T+2 ... T+K               │
  │           1 [ 1  1  1 ... 1   0   0  ...  0 ]              │
  │           2 [ 1  1  1 ... 1   0   0  ...  0 ]              │
  │          ..                                                 │
  │           T [ 1  1  1 ... 1   0   0  ...  0 ]              │
  │         T+1 [ 0  0  0 ... 0   1   0  ...  0 ]              │
  │         T+2 [ 0  0  0 ... 0   0   1  ...  0 ]              │
  │          ..                                                 │
  │         T+K [ 0  0  0 ... 0   0   0  ...  1 ]              │
  └─────────────────────────────────────────────────────────────┘
  ```,
  caption: [Attention masking for joint image-video training]
)

=== Benefits of Joint Training

From experiments (Table 4 in the paper):

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    [*Image Frames*], [*FVD↓*], [*FID-avg↓*], [*IS-avg↑*],
    [0], [205.42], [37.40], [7.58],
    [4], [70.74], [18.42], [8.53],
    [8], [60.72], [15.44], [8.82],
  ),
  caption: [Effect of joint training on sample quality]
)

Key insight: Adding independent image frames:
- Reduces gradient variance
- Acts as memory optimization (more independent samples per batch)
- Significantly improves both video and image quality metrics

== Architecture Hyperparameters

=== Model Configurations by Task

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 6pt,
    [*Parameter*], [*UCF101*], [*BAIR*], [*Kinetics*], [*Text-to-Video*],
    [Base channels], [256], [128], [256], [256],
    [Channel mult.], [1,2,4,8], [1,2,3,4], [1,2,4,8], [1,2,4,8],
    [Blocks/res], [2], [3], [2], [2],
    [Attn. resolutions], [8,16,32], [8,16,32], [8,16,32], [8,16,32],
    [Head dim], [64], [64], [64], [64],
    [Cond. emb. dim], [1024], [1024], [1024], [1024],
    [MLP layers], [4], [2], [2], [4],
    [Dropout], [0.1], [0.1], [0.1], [0.0],
  ),
  caption: [Architecture hyperparameters by task]
)
