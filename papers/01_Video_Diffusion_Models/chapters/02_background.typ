// Chapter 2: Diffusion Model Background

= Diffusion Model Background

This chapter provides the mathematical foundations needed to understand Video Diffusion Models.

== Continuous-Time Diffusion Framework

=== Forward Process (Noising)

A diffusion model defines a *forward process* $q(bold(z)|bold(x))$ that gradually adds noise to data $bold(x) tilde p(bold(x))$. In continuous time $t in [0, 1]$:

#block(fill: rgb("#e8f4e8"), inset: 1em, radius: 4pt)[
  *Definition 2.1 (Forward Process)*

  The forward process is a Gaussian process with:
  $ q(bold(z)_t | bold(x)) = cal(N)(bold(z)_t; alpha_t bold(x), sigma_t^2 bold(I)) $

  where $alpha_t$ and $sigma_t$ define a *noise schedule*.
]

The forward process is *Markovian*, meaning:
$ q(bold(z)_t | bold(z)_s) = cal(N)(bold(z)_t; (alpha_t \/ alpha_s) bold(z)_s, sigma_(t|s)^2 bold(I)) $

where $0 <= s < t <= 1$ and:
$ sigma_(t|s)^2 = (1 - e^(lambda_t - lambda_s)) sigma_t^2 $

=== Log Signal-to-Noise Ratio

The *log signal-to-noise ratio (SNR)* is defined as:
$ lambda_t = log[alpha_t^2 \/ sigma_t^2] $

Key property: $lambda_t$ *decreases monotonically* with $t$, meaning:
- At $t = 0$: High SNR → mostly signal (original data)
- At $t = 1$: Low SNR → mostly noise

At the final time $t = 1$, we have approximately:
$ q(bold(z)_1) approx cal(N)(bold(0), bold(I)) $

=== Variance Preserving vs. Variance Exploding

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 10pt,
    [*Schedule Type*], [*Constraint*], [*Properties*],
    [Variance Preserving (VP)], [$alpha_t^2 + sigma_t^2 = 1$], [
      - Total variance constant
      - Used in DDPM, this paper
      - $alpha_t = sqrt(1 - sigma_t^2)$
    ],
    [Variance Exploding (VE)], [$alpha_t = 1$], [
      - Signal unchanged
      - Noise variance increases
      - Used in NCSN, SMLD
    ],
  ),
  caption: [Comparison of noise schedule types]
)

This paper uses the *variance preserving* formulation.

== Reverse Process (Denoising)

=== Reverse Conditional Distribution

The forward process can be reversed. Given $bold(x)$ and $bold(z)_t$:
$ q(bold(z)_s | bold(z)_t, bold(x)) = cal(N)(bold(z)_s; tilde(mu)_(s|t)(bold(z)_t, bold(x)), tilde(sigma)_(s|t)^2 bold(I)) $

where (noting $s < t$):

#block(fill: rgb("#fff3cd"), inset: 1em, radius: 4pt)[
  *Reverse Process Parameters*
  $ tilde(mu)_(s|t)(bold(z)_t, bold(x)) = e^(lambda_t - lambda_s) (alpha_s \/ alpha_t) bold(z)_t + (1 - e^(lambda_t - lambda_s)) alpha_s bold(x) $
  $ tilde(sigma)_(s|t)^2 = (1 - e^(lambda_t - lambda_s)) sigma_s^2 $
]

=== Intuition Behind Reverse Process

#figure(
  ```
  Forward Process (Adding Noise):
  ┌────────┐    q(z_t|x)    ┌────────┐    q(z_1|z_t)   ┌────────┐
  │   x    │ ─────────────► │  z_t   │ ─────────────► │  z_1   │
  │ (data) │    add noise   │(noisy) │    add noise   │ (noise)│
  └────────┘                └────────┘                └────────┘

  Reverse Process (Removing Noise):
  ┌────────┐  p_θ(z_t|z_1)  ┌────────┐  p_θ(x|z_t)    ┌────────┐
  │  z_1   │ ◄───────────── │  z_t   │ ◄───────────── │   x̂   │
  │(noise) │   denoise      │(noisy) │   denoise      │(recon) │
  └────────┘                └────────┘                └────────┘
  ```,
  caption: [Forward and reverse diffusion processes]
)

== Training Objective

=== Denoising Objective

The key insight is that learning to *reverse* the forward process reduces to learning to *denoise*:

#block(fill: rgb("#e8f4e8"), inset: 1em, radius: 4pt)[
  *Theorem 2.1 (Denoising ↔ Score Matching)*

  Training a denoiser $hat(bold(x))_theta (bold(z)_t)$ to predict $bold(x)$ from noisy $bold(z)_t$ is equivalent to learning the score function:
  $ epsilon_theta (bold(z)_t) approx -sigma_t nabla_(bold(z)_t) log p(bold(z)_t) $
]

=== Training Loss

The model is trained using a weighted mean squared error loss:

$ cal(L) = EE_(epsilon, t) [w(lambda_t) || hat(bold(x))_theta (bold(z)_t) - bold(x) ||_2^2 ] $

where:
- $t tilde "Uniform"[0, 1]$
- $epsilon tilde cal(N)(bold(0), bold(I))$
- $bold(z)_t = alpha_t bold(x) + sigma_t epsilon$
- $w(lambda_t)$ is a weighting function

=== Parameterization Options

There are three common ways to parameterize the denoising model:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 10pt,
    [*Parameterization*], [*Model Output*], [*Reconstruction*],
    [$bold(x)$-prediction], [$hat(bold(x))_theta (bold(z)_t)$], [Direct prediction of clean data],
    [$epsilon$-prediction], [$epsilon_theta (bold(z)_t)$], [$hat(bold(x)) = (bold(z)_t - sigma_t epsilon_theta) \/ alpha_t$],
    [$bold(v)$-prediction], [$bold(v)_theta (bold(z)_t)$], [$hat(bold(x)) = alpha_t bold(z)_t - sigma_t bold(v)_theta$],
  ),
  caption: [Different parameterizations of the denoising model]
)

=== ε-Prediction (Used in this paper)

The $epsilon$-prediction parameterization defines:
$ hat(bold(x))_theta (bold(z)_t) = (bold(z)_t - sigma_t epsilon_theta (bold(z)_t)) / alpha_t $

The training objective becomes:
$ cal(L)_epsilon = EE_(epsilon, t) [|| epsilon_theta (bold(z)_t) - epsilon ||_2^2 ] $

*Intuition*: The model learns to predict the noise $epsilon$ that was added to create $bold(z)_t$.

=== v-Prediction (Alternative)

For certain models, the paper uses *v-prediction*:
$ bold(v) = alpha_t epsilon - sigma_t bold(x) $

Benefits of v-prediction:
- More stable training at low noise levels
- Better for high-resolution generation
- Improved color consistency

== Sampling Algorithms

=== Ancestral Sampling

Starting from $bold(z)_1 tilde cal(N)(bold(0), bold(I))$, the ancestral sampler follows:

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Algorithm 2.1: Ancestral Sampling*

  ```
  Input: Trained model ε_θ, number of steps N
  Output: Generated sample x̂

  1. z_1 ~ N(0, I)
  2. for t = 1, 1-1/N, 1-2/N, ..., 1/N do:
     a. s = t - 1/N
     b. x̂ = (z_t - σ_t · ε_θ(z_t)) / α_t
     c. μ̃ = e^(λ_t - λ_s) · (α_s/α_t) · z_t + (1 - e^(λ_t - λ_s)) · α_s · x̂
     d. σ̃² = (σ̃²_{s|t})^(1-γ) · (σ²_{t|s})^γ
     e. z_s = μ̃ + σ̃ · ε,  where ε ~ N(0, I)
  3. return x̂_θ(z_{1/N})
  ```
]

Parameter $gamma$ controls stochasticity:
- $gamma = 0$: Deterministic (DDIM-like)
- $gamma = 1$: Full stochasticity (DDPM)

=== Predictor-Corrector Sampling

This paper finds that predictor-corrector sampling works especially well with reconstruction guidance:

#block(fill: rgb("#d4edda"), inset: 1em, radius: 4pt)[
  *Algorithm 2.2: Predictor-Corrector Sampling*

  ```
  Input: Model ε_θ, steps N, Langevin step size δ = 0.1
  Output: Generated sample x̂

  1. z_1 ~ N(0, I)
  2. for t = 1, 1-1/N, ..., 1/N do:
     a. // Predictor step (ancestral)
        s = t - 1/N
        z_s = μ̃_{s|t}(z_t, x̂_θ(z_t)) + σ̃_{s|t} · ε

     b. // Corrector step (Langevin)
        z_s ← z_s - (δ/2) · σ_s · ε_θ(z_s) + √(δ) · σ_s · ε'

  3. return x̂_θ(z_{1/N})
  ```
]

The Langevin correction step helps the marginal distribution of $bold(z)_s$ match the true marginal.

== Conditional Generation

=== Conditioning Mechanism

For conditional generation $p_theta (bold(x) | bold(c))$, the conditioning signal $bold(c)$ is provided to the model:
$ hat(bold(x))_theta (bold(z)_t, bold(c)) $

Common conditioning types:
- Text embeddings (BERT, CLIP)
- Class labels (one-hot or embedding)
- Images (for super-resolution or editing)

=== Classifier-Free Guidance

#block(fill: rgb("#fff3cd"), inset: 1em, radius: 4pt)[
  *Definition 2.2 (Classifier-Free Guidance)*

  Classifier-free guidance adjusts the model prediction:
  $ tilde(epsilon)_theta (bold(z)_t, bold(c)) = (1 + w) epsilon_theta (bold(z)_t, bold(c)) - w epsilon_theta (bold(z)_t) $

  where $w > 0$ is the *guidance weight*.
]

Implementation details:
- Train jointly with conditional and unconditional objectives
- For embedding-based conditioning: set $bold(c) = bold(0)$ for unconditional
- Drop conditioning with probability $p_"uncond"$ during training (typically 10-20%)

Effect of guidance weight $w$:
- $w = 0$: No guidance (standard conditional sampling)
- $w > 0$: Emphasizes conditioning → higher quality, lower diversity
- Typical values: $w in [1, 5]$

=== Mathematical Interpretation

Classifier-free guidance can be understood as implicit classifier guidance:
$ tilde(epsilon)_theta (bold(z)_t, bold(c)) = epsilon_theta (bold(z)_t) - w sigma_t nabla_(bold(z)_t) log p(bold(c) | bold(z)_t) $

The second term guides samples toward regions where an *implicit* classifier $p(bold(c) | bold(z)_t)$ has high likelihood.

== Variational Bound Connection

The diffusion training objective is connected to a variational lower bound on the data log-likelihood:

$ log p(bold(x)) >= EE_q [ log p_theta (bold(x) | bold(z)_0) ] - D_"KL"(q(bold(z)_1 | bold(x)) || p(bold(z)_1)) - sum_t D_"KL"(q(bold(z)_s | bold(z)_t, bold(x)) || p_theta (bold(z)_s | bold(z)_t)) $

The denoising objective corresponds to a weighted version of this bound.
