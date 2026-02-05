// Chapter 3: 3D U-Net Architecture
#import "../template.typ": *
#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge
#import "@preview/cetz:0.3.2"

#show: chapter-template.with(
  title: "3D U-Net Architecture",
  subtitle: "Space-Time Factorized Design",
  chapter-num: 3,
)

= 3D U-Net Architecture <architecture>

== Overview

The VDM uses a 3D U-Net architecture that processes video tensors of shape $(T, H, W, C)$. The key innovation is *space-time factorization* which makes computation tractable.

#keypoint(title: "Model Specification")[
  *Input*: Noisy video $bold(z)_t in RR^(T times H times W times C)$ \
  *Additional Inputs*: Timestep $t$, optional conditioning $bold(c)$ \
  *Output*: Predicted noise $epsilon_theta in RR^(T times H times W times C)$ \
  *Architecture*: Encoder-Decoder with skip connections
]

== U-Net Structure

#figure(
  diagram(
    spacing: (4pt, 8pt),
    node-stroke: 1pt,

    // Encoder path
    node((0, 0), [Input\ 64×64×3], shape: rect, fill: blue.lighten(85%), width: 2cm, height: 0.8cm),
    node((0, 1.2), [Enc1\ 32×32×256], shape: rect, fill: blue.lighten(75%), width: 2cm, height: 0.8cm),
    node((0, 2.4), [Enc2\ 16×16×512], shape: rect, fill: blue.lighten(65%), width: 2cm, height: 0.8cm),
    node((0, 3.6), [Enc3\ 8×8×1024], shape: rect, fill: blue.lighten(55%), width: 2cm, height: 0.8cm),

    // Bottleneck
    node((1.5, 4.2), [Bottleneck\ 4×4×2048], shape: rect, fill: purple.lighten(70%), width: 2.5cm, height: 0.8cm),

    // Decoder path
    node((3, 3.6), [Dec3\ 8×8×1024], shape: rect, fill: green.lighten(55%), width: 2cm, height: 0.8cm),
    node((3, 2.4), [Dec2\ 16×16×512], shape: rect, fill: green.lighten(65%), width: 2cm, height: 0.8cm),
    node((3, 1.2), [Dec1\ 32×32×256], shape: rect, fill: green.lighten(75%), width: 2cm, height: 0.8cm),
    node((3, 0), [Output\ 64×64×3], shape: rect, fill: green.lighten(85%), width: 2cm, height: 0.8cm),

    // Encoder arrows
    edge((0, 0), (0, 1.2), "->", [↓2×], label-side: left),
    edge((0, 1.2), (0, 2.4), "->", [↓2×], label-side: left),
    edge((0, 2.4), (0, 3.6), "->", [↓2×], label-side: left),
    edge((0, 3.6), (1.5, 4.2), "->"),

    // Decoder arrows
    edge((1.5, 4.2), (3, 3.6), "->"),
    edge((3, 3.6), (3, 2.4), "->", [↑2×], label-side: right),
    edge((3, 2.4), (3, 1.2), "->", [↑2×], label-side: right),
    edge((3, 1.2), (3, 0), "->", [↑2×], label-side: right),

    // Skip connections
    edge((0, 3.6), (3, 3.6), "-->", stroke: red, bend: -20deg),
    edge((0, 2.4), (3, 2.4), "-->", stroke: red, bend: -15deg),
    edge((0, 1.2), (3, 1.2), "-->", stroke: red, bend: -10deg),
  ),
  caption: [U-Net encoder-decoder structure with skip connections (red dashed). Spatial resolution shown; all have T=16 frames.]
)

== The Key Innovation: Space-Time Factorization

#innovation(title: "Computational Breakthrough")[
  *Problem*: Full 3D attention over $(T, H, W)$ has complexity $O(T^2 H^2 W^2)$ \
  *Solution*: Factorize into spatial attention $O(H^2 W^2)$ + temporal attention $O(T^2)$

  For $T=16, H=W=64$:
  - Full 3D: $16^2 times 64^2 times 64^2 = 4.3 times 10^9$
  - Factorized: $64^2 times 64^2 + 16^2 = 1.7 times 10^7$
  - *Speedup: ~250×* (paper claims up to 1000× in practice)
]

== Block Structure

Each U-Net block contains the following sequence:

#figure(
  diagram(
    spacing: (8pt, 10pt),
    node-stroke: 1pt,

    // Block components
    node((0, 0), [Spatial Conv\ (1×3×3)], shape: rect, fill: blue.lighten(80%), width: 2.5cm),
    node((0, 1), [GroupNorm + SiLU], shape: rect, fill: gray.lighten(80%), width: 2.5cm),
    node((0, 2), [Spatial Attention\ over (H×W)], shape: rect, fill: orange.lighten(80%), width: 2.5cm),
    node((0, 3), [Temporal Attention\ over T], shape: rect, fill: green.lighten(80%), width: 2.5cm),
    node((0, 4), [Residual Connection], shape: rect, fill: purple.lighten(80%), width: 2.5cm),

    // Timestep embedding
    node((2, 1.5), [Time\ Embed], shape: circle, fill: yellow.lighten(70%), radius: 0.5cm),

    // Arrows
    edge((0, 0), (0, 1), "->"),
    edge((0, 1), (0, 2), "->"),
    edge((0, 2), (0, 3), "->"),
    edge((0, 3), (0, 4), "->"),
    edge((2, 1.5), (0, 1), "->"),
  ),
  caption: [Single U-Net block with factorized attention]
)

== Spatial Convolution

#definition("Spatial Convolution")[
  Operates on each frame independently using kernel size $(1, 3, 3)$:
  $ "Conv3D"(bold(x)) : RR^(T times H times W times C_"in") arrow RR^(T times H times W times C_"out") $

  The temporal dimension ($T$) is preserved; only spatial dimensions are processed.
]

#mathblock(title: "Mathematical Operation")[
  For each frame $t$ and spatial position $(h, w)$:
  $ y_(t,h,w) = sum_(i,j in [-1,0,1]) sum_c W_(i,j,c) dot x_(t,h+i,w+j,c) + b $
]

#codeblock(title: "PyTorch Implementation")[
  ```python
  class SpatialConv3D(nn.Module):
      def __init__(self, in_ch, out_ch):
          super().__init__()
          # Kernel: (temporal=1, height=3, width=3)
          self.conv = nn.Conv3d(in_ch, out_ch,
                                kernel_size=(1, 3, 3),
                                padding=(0, 1, 1))

      def forward(self, x):
          # x: (B, C, T, H, W)
          return self.conv(x)
  ```
]

== Spatial Attention

#definition("Spatial Self-Attention")[
  For each frame independently, compute attention over all spatial positions:
  $ "Attention"(Q, K, V) = "softmax"(Q K^T / sqrt(d_k)) V $
  where $Q, K, V in RR^((H times W) times d)$ for a single frame.
]

#figure(
  diagram(
    spacing: (6pt, 10pt),
    node-stroke: 1pt,

    // Input reshape
    node((0, 0), [$bold(x)$\ (B,C,T,H,W)], shape: rect, fill: blue.lighten(85%), width: 2.2cm),
    node((1.8, 0), [Reshape\ (B·T, H·W, C)], shape: rect, fill: yellow.lighten(80%), width: 2.5cm),

    // QKV
    node((3.6, -0.5), [$Q = x W_Q$], shape: rect, fill: orange.lighten(80%), width: 2cm),
    node((3.6, 0), [$K = x W_K$], shape: rect, fill: orange.lighten(80%), width: 2cm),
    node((3.6, 0.5), [$V = x W_V$], shape: rect, fill: orange.lighten(80%), width: 2cm),

    // Attention
    node((5.4, 0), [Attention\ (H·W, H·W)], shape: rect, fill: green.lighten(80%), width: 2.2cm),

    // Output
    node((7.2, 0), [Reshape\ (B,C,T,H,W)], shape: rect, fill: purple.lighten(80%), width: 2.2cm),

    // Arrows
    edge((0, 0), (1.8, 0), "->"),
    edge((1.8, 0), (3.6, -0.5), "->"),
    edge((1.8, 0), (3.6, 0), "->"),
    edge((1.8, 0), (3.6, 0.5), "->"),
    edge((3.6, -0.5), (5.4, 0), "->"),
    edge((3.6, 0), (5.4, 0), "->"),
    edge((3.6, 0.5), (5.4, 0), "->"),
    edge((5.4, 0), (7.2, 0), "->"),
  ),
  caption: [Spatial attention: reshape to combine batch and time, attend over H×W]
)

#codeblock(title: "PyTorch: Spatial Attention")[
  ```python
  class SpatialAttention(nn.Module):
      def __init__(self, dim, heads=8):
          super().__init__()
          self.heads = heads
          self.scale = (dim // heads) ** -0.5
          self.qkv = nn.Linear(dim, dim * 3)
          self.proj = nn.Linear(dim, dim)

      def forward(self, x):
          # x: (B, C, T, H, W)
          B, C, T, H, W = x.shape

          # Reshape: treat each frame independently
          x = rearrange(x, 'b c t h w -> (b t) (h w) c')

          # Standard multi-head attention
          qkv = self.qkv(x).chunk(3, dim=-1)
          q, k, v = map(lambda t: rearrange(t,
              'b n (h d) -> b h n d', h=self.heads), qkv)

          attn = (q @ k.transpose(-2, -1)) * self.scale
          attn = attn.softmax(dim=-1)
          out = attn @ v

          out = rearrange(out, 'b h n d -> b n (h d)')
          out = self.proj(out)

          # Reshape back
          out = rearrange(out, '(b t) (h w) c -> b c t h w',
                         b=B, t=T, h=H, w=W)
          return out
  ```
]

== Temporal Attention

#definition("Temporal Self-Attention")[
  For each spatial position independently, compute attention over all frames:
  $ "Attention"(Q, K, V) = "softmax"((Q K^T + R) / sqrt(d_k)) V $
  where $Q, K, V in RR^(T times d)$ for a single spatial position, and $R$ is a relative position bias.
]

#innovation(title: "Relative Position Embedding")[
  The paper uses *relative position embeddings* to encode temporal relationships:
  $ R_(i,j) = r_(|i - j|) $
  where $r$ is a learned embedding based on the *distance* between frames, not absolute position.

  This allows the model to generalize to different video lengths!
]

#figure(
  cetz.canvas(length: 0.8cm, {
    import cetz.draw: *

    // Frame representations
    for i in range(5) {
      rect((i * 1.5, 0), (i * 1.5 + 1, 1), fill: blue.lighten(80%), stroke: black)
      content((i * 1.5 + 0.5, 0.5), text(size: 8pt)[F#str(i+1)])
    }

    // Attention arrows (frame 3 attending to all)
    for i in range(5) {
      if i != 2 {
        let color = if calc.abs(i - 2) == 1 { green } else { orange }
        line((3.5, 1.1), (i * 1.5 + 0.5, 1.1), stroke: color + 1.5pt, mark: (end: "stealth"))
      }
    }

    // Labels
    content((3.5, 1.6), text(size: 7pt)[Current frame])
    content((0.5, -0.4), text(size: 6pt)[dist=2])
    content((2, -0.4), text(size: 6pt)[dist=1])
    content((5, -0.4), text(size: 6pt)[dist=1])
    content((6.5, -0.4), text(size: 6pt)[dist=2])
  }),
  caption: [Relative position: attention weight depends on frame distance, not absolute position]
)

#codeblock(title: "PyTorch: Temporal Attention")[
  ```python
  class TemporalAttention(nn.Module):
      def __init__(self, dim, heads=8, max_len=64):
          super().__init__()
          self.heads = heads
          self.scale = (dim // heads) ** -0.5
          self.qkv = nn.Linear(dim, dim * 3)
          self.proj = nn.Linear(dim, dim)

          # Relative position bias: (2*max_len-1, heads)
          self.rel_pos_bias = nn.Parameter(
              torch.zeros(2 * max_len - 1, heads))

      def get_rel_pos_bias(self, T):
          # Create relative position indices
          coords = torch.arange(T)
          relative_coords = coords[:, None] - coords[None, :]
          relative_coords += T - 1  # Shift to positive
          return self.rel_pos_bias[relative_coords]  # (T, T, heads)

      def forward(self, x):
          # x: (B, C, T, H, W)
          B, C, T, H, W = x.shape

          # Reshape: treat each spatial position independently
          x = rearrange(x, 'b c t h w -> (b h w) t c')

          # Standard multi-head attention
          qkv = self.qkv(x).chunk(3, dim=-1)
          q, k, v = map(lambda t: rearrange(t,
              'b n (h d) -> b h n d', h=self.heads), qkv)

          attn = (q @ k.transpose(-2, -1)) * self.scale

          # Add relative position bias
          rel_bias = self.get_rel_pos_bias(T)
          rel_bias = rearrange(rel_bias, 'i j h -> 1 h i j')
          attn = attn + rel_bias

          attn = attn.softmax(dim=-1)
          out = attn @ v

          out = rearrange(out, 'b h n d -> b n (h d)')
          out = self.proj(out)

          # Reshape back
          out = rearrange(out, '(b h w) t c -> b c t h w',
                         b=B, h=H, w=W)
          return out
  ```
]

== Time Embedding

#definition("Sinusoidal Time Embedding")[
  The diffusion timestep $t$ is embedded using sinusoidal positional encoding:
  $ "PE"(t, 2i) = sin(t / 10000^(2i/d)) $
  $ "PE"(t, 2i+1) = cos(t / 10000^(2i/d)) $
  Then projected through MLPs and added to hidden states.
]

#figure(
  diagram(
    spacing: (10pt, 10pt),
    node-stroke: 1pt,

    node((0, 0), [$t in [0,1]$], shape: rect, fill: yellow.lighten(80%), width: 1.8cm),
    node((1.5, 0), [Sinusoidal\ Embed], shape: rect, fill: orange.lighten(80%), width: 2cm),
    node((3, 0), [MLP], shape: rect, fill: blue.lighten(80%), width: 1.5cm),
    node((4.5, 0), [$e_t in RR^d$], shape: rect, fill: green.lighten(80%), width: 1.8cm),

    edge((0, 0), (1.5, 0), "->"),
    edge((1.5, 0), (3, 0), "->"),
    edge((3, 0), (4.5, 0), "->"),
  ),
  caption: [Time embedding pipeline]
)

#codeblock(title: "PyTorch: Time Embedding")[
  ```python
  class TimeEmbedding(nn.Module):
      def __init__(self, dim, max_period=10000):
          super().__init__()
          self.dim = dim
          self.max_period = max_period
          self.mlp = nn.Sequential(
              nn.Linear(dim, dim * 4),
              nn.SiLU(),
              nn.Linear(dim * 4, dim),
          )

      def forward(self, t):
          # t: (B,) in [0, 1]
          half = self.dim // 2
          freqs = torch.exp(
              -math.log(self.max_period) *
              torch.arange(half, device=t.device) / half
          )
          args = t[:, None] * freqs[None]
          emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
          return self.mlp(emb)
  ```
]

== Complete Block Architecture

#figure(
  diagram(
    spacing: (5pt, 8pt),
    node-stroke: 1pt,

    // Main path
    node((0, 0), [Input $bold(x)$], shape: rect, fill: blue.lighten(90%), width: 2cm),
    node((0, 1), [SpatialConv], shape: rect, fill: blue.lighten(80%), width: 2cm),
    node((0, 2), [GroupNorm], shape: rect, fill: gray.lighten(80%), width: 2cm),
    node((0, 3), [SiLU], shape: rect, fill: gray.lighten(80%), width: 2cm),
    node((0, 4), [SpatialConv], shape: rect, fill: blue.lighten(80%), width: 2cm),
    node((0, 5), [SpatialAttn], shape: rect, fill: orange.lighten(80%), width: 2cm),
    node((0, 6), [TemporalAttn], shape: rect, fill: green.lighten(80%), width: 2cm),
    node((0, 7), [Output], shape: rect, fill: purple.lighten(80%), width: 2cm),

    // Time embedding
    node((2, 3), [TimeEmbed], shape: rect, fill: yellow.lighten(80%), width: 2cm),

    // Skip connection
    node((2, 3.5), [], shape: circle, fill: red.lighten(80%), radius: 0.2cm),

    // Arrows
    edge((0, 0), (0, 1), "->"),
    edge((0, 1), (0, 2), "->"),
    edge((0, 2), (0, 3), "->"),
    edge((0, 3), (0, 4), "->"),
    edge((0, 4), (0, 5), "->"),
    edge((0, 5), (0, 6), "->"),
    edge((0, 6), (0, 7), "->"),
    edge((2, 3), (0, 3), "->", [+]),

    // Residual
    edge((0, 0), (2, 3.5), "->", bend: 30deg, stroke: red),
    edge((2, 3.5), (0, 7), "->", bend: 30deg, stroke: red, [+]),
  ),
  caption: [Complete ResBlock with factorized attention. Red path shows residual connection.]
)

== Detailed Tensor Shapes

#figure(
  tablex(
    columns: (auto, auto, auto, auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Layer*], [*Input Shape*], [*Output Shape*], [*Parameters*], [*FLOPs*],
    hlinex(),
    [Input], [-], [(B,3,16,64,64)], [-], [-],
    [Enc Block 1], [(B,3,T,64,64)], [(B,256,T,32,32)], [~2M], [~5G],
    [Enc Block 2], [(B,256,T,32,32)], [(B,512,T,16,16)], [~8M], [~10G],
    [Enc Block 3], [(B,512,T,16,16)], [(B,1024,T,8,8)], [~32M], [~20G],
    [Bottleneck], [(B,1024,T,8,8)], [(B,2048,T,4,4)], [~130M], [~40G],
    [Dec Block 3], [(B,2048,T,4,4)], [(B,1024,T,8,8)], [~130M], [~40G],
    [Dec Block 2], [(B,1024,T,8,8)], [(B,512,T,16,16)], [~32M], [~20G],
    [Dec Block 1], [(B,512,T,16,16)], [(B,256,T,32,32)], [~8M], [~10G],
    [Output], [(B,256,T,32,32)], [(B,3,T,64,64)], [~2K], [~1M],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Tensor shapes through the network (T=16 frames, 64×64 spatial)]
)

== Downsampling and Upsampling

#definition("Spatial Downsampling")[
  Uses strided convolution with kernel $(1, 4, 4)$ and stride $(1, 2, 2)$:
  $ H' = H/2, quad W' = W/2, quad T' = T $
]

#definition("Spatial Upsampling")[
  Uses transposed convolution or nearest-neighbor + convolution:
  $ H' = 2H, quad W' = 2W, quad T' = T $
]

#warning(title: "No Temporal Downsampling")[
  The VDM model does *not* downsample in the temporal dimension. All 16 frames are processed at full temporal resolution throughout the network.

  This is different from later models like SVD which use temporal downsampling.
]

== Architecture Configuration

#figure(
  tablex(
    columns: (auto, auto),
    align: center + horizon,
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Hyperparameter*], [*Value*],
    hlinex(),
    [Base Channels], [256],
    [Channel Multipliers], [(1, 2, 4, 8)],
    [ResBlocks per Stage], [2],
    [Attention Heads], [8],
    [Attention Head Dim], [64],
    [Dropout], [0.0],
    [Temporal Length], [16 frames],
    [Spatial Resolution], [64×64],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Default architecture configuration]
)

#pagebreak()

== Full Model Code

#codeblock(title: "PyTorch: Complete UNet3D")[
  ```python
  class UNet3D(nn.Module):
      def __init__(
          self,
          in_channels=3,
          out_channels=3,
          base_channels=256,
          channel_mults=(1, 2, 4, 8),
          num_res_blocks=2,
          attention_resolutions=(16, 8, 4),
          num_heads=8,
      ):
          super().__init__()

          # Time embedding
          self.time_embed = TimeEmbedding(base_channels)

          # Input convolution
          self.input_conv = nn.Conv3d(in_channels, base_channels,
                                      kernel_size=(1, 3, 3),
                                      padding=(0, 1, 1))

          # Encoder
          self.encoder = nn.ModuleList()
          channels = [base_channels]
          ch = base_channels

          for i, mult in enumerate(channel_mults):
              out_ch = base_channels * mult
              for _ in range(num_res_blocks):
                  self.encoder.append(
                      ResBlock3D(ch, out_ch, num_heads)
                  )
                  ch = out_ch
                  channels.append(ch)
              if i < len(channel_mults) - 1:
                  self.encoder.append(Downsample3D(ch))
                  channels.append(ch)

          # Bottleneck
          self.bottleneck = ResBlock3D(ch, ch, num_heads)

          # Decoder
          self.decoder = nn.ModuleList()
          for i, mult in reversed(list(enumerate(channel_mults))):
              out_ch = base_channels * mult
              for j in range(num_res_blocks + 1):
                  skip_ch = channels.pop()
                  self.decoder.append(
                      ResBlock3D(ch + skip_ch, out_ch, num_heads)
                  )
                  ch = out_ch
              if i > 0:
                  self.decoder.append(Upsample3D(ch))

          # Output
          self.output_conv = nn.Sequential(
              nn.GroupNorm(32, ch),
              nn.SiLU(),
              nn.Conv3d(ch, out_channels, kernel_size=(1, 3, 3),
                       padding=(0, 1, 1)),
          )

      def forward(self, x, t):
          # x: (B, C, T, H, W), t: (B,)
          t_emb = self.time_embed(t)
          x = self.input_conv(x)

          # Encoder with skip connections
          skips = [x]
          for layer in self.encoder:
              x = layer(x, t_emb) if hasattr(layer, 't_emb') else layer(x)
              skips.append(x)

          # Bottleneck
          x = self.bottleneck(x, t_emb)

          # Decoder
          for layer in self.decoder:
              if isinstance(layer, ResBlock3D):
                  x = torch.cat([x, skips.pop()], dim=1)
              x = layer(x, t_emb) if hasattr(layer, 't_emb') else layer(x)

          return self.output_conv(x)
  ```
]

#pagebreak()

== Summary

#figure(
  tablex(
    columns: (auto, 1fr),
    align: (center, left),
    auto-vlines: false,
    hlinex(stroke: 1.5pt),
    [*Component*], [*Description*],
    hlinex(),
    [Architecture], [3D U-Net with encoder-decoder + skip connections],
    [Key Innovation], [Space-time factorized attention (1000× speedup)],
    [Spatial Conv], [$(1,3,3)$ kernel, processes each frame independently],
    [Spatial Attention], [Self-attention over $H times W$ per frame],
    [Temporal Attention], [Self-attention over $T$ frames with relative position bias],
    [Time Embedding], [Sinusoidal + MLP, added to hidden states],
    [Downsampling], [Spatial only (stride 2), temporal preserved],
    hlinex(stroke: 1.5pt),
  ),
  caption: [Chapter 3 Summary]
)
