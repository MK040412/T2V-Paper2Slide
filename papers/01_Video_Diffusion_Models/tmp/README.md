# Video Diffusion Models - Detailed Technical Analysis

This directory contains comprehensive technical documentation for the Video Diffusion Models paper (Ho et al., 2022).

## Contents

### Main Document
- **VDM_Standalone.typ** - Complete standalone Typst source (all chapters in one file)
- **VDM_Complete_v2.pdf** - Compiled PDF (in parent directory)

### Template
- **template.typ** - Shared template with custom styling, theorem boxes, and helper functions

### Chapters
Each chapter provides detailed coverage of a specific aspect:

| File | Chapter | Description |
|------|---------|-------------|
| `ch01_introduction.typ` | Introduction | Paper overview, contributions, applications |
| `ch02_background.typ` | Background | Diffusion fundamentals, forward/reverse processes |
| `ch03_architecture.typ` | Architecture | 3D U-Net, factorized attention, tensor shapes |
| `ch04_training.typ` | Training | Noise schedules, joint training, optimization |
| `ch05_sampling.typ` | Sampling | DDPM, DDIM, CFG, reconstruction guidance |
| `ch06_experiments.typ` | Experiments | Benchmarks, ablations, comparisons |
| `ch07_implementation.typ` | Implementation | Complete PyTorch code with explanations |

## Packages Used

- `@preview/ctheorems:1.1.3` - Theorem/Definition boxes
- `@preview/showybox:2.0.3` - Highlighted info boxes
- `@preview/fletcher:0.5.3` - Diagrams and flowcharts
- `@preview/cetz:0.3.2` - Custom drawings and graphs
- `@preview/tablex:0.0.9` - Enhanced tables

## Compilation

To compile the complete document:

```bash
# Using Python typst package
pip install typst
python -c "import typst; typst.compile('VDM_Standalone.typ', output='output.pdf')"
```

Or with the Typst CLI:
```bash
typst compile VDM_Standalone.typ output.pdf
```

## Key Topics Covered

1. **Mathematical Foundations**
   - Forward/reverse diffusion processes
   - Variance-preserving formulation
   - Log-SNR and noise schedules
   - ε-prediction parameterization

2. **Architecture Details**
   - 3D U-Net structure
   - Space-time factorized attention (1000× speedup)
   - Spatial and temporal attention implementations
   - Relative position embeddings

3. **Training Methodology**
   - Cosine noise schedule
   - Joint image-video training (3.4× improvement)
   - Classifier-free guidance
   - EMA and optimization

4. **Sampling Algorithms**
   - DDPM (stochastic)
   - DDIM (deterministic)
   - Reconstruction guidance for conditional generation

5. **Implementation Guide**
   - Complete PyTorch code
   - Paper-to-code mapping
   - Implementation checklist
