# GravitationalWaveforms.jl

A gravitational-wave waveform toolkit: the phenomenological frequency-domain approximants
(`IMRPhenomD`, `IMRPhenomXAS`, `IMRPhenomXHM`, `IMRPhenomXP`, `IMRPhenomXPHM`), the SEOBNRv5
effective-one-body family (`SEOBNRv5IMRAlignedSpinF2`, `SEOBNRv5HM`, `SEOBNRv5PHM`) and
post-Newtonian models (`TaylorF2`, `TaylorT3`), differentiable and Float32/Float64-generic. Part of the same ecosystem as
[GravitationalWaveData.jl](https://github.com/JuliaGravitationalWaves/GravitationalWaveData.jl) (the `gwpy`-equivalent data toolkit) and
`GravitationalWaveBayes.jl` (GPU-native, Enzyme-differentiable Bayesian parameter estimation) — dependency
order `GravitationalWaveData <- GravitationalWaveforms <- GravitationalWaveBayes`.

## Where to start

- **New to the package?** Start with [Getting Started](generated/01_getting_started.md) —
  generate your first frequency-domain waveform and plot it.
- **Read the [Conventions](conventions.md) page early** — units (`dL` in Gpc, not Mpc!), the
  `m1 >= m2` labeling convention, the `t <= 0` time-domain convention, and the
  `h ~ Ampl*exp(i*Phi)` decomposition are used, unremarked, in every tutorial below.
- **Want the full function/type list?** See the [API Reference](api.md).
- **Coming from `ripple` (JAX), `lalsimulation` (C) or `pyseobnr` (Python)?** The Phenom and
  PN models are direct, cross-validated ports of the corresponding `lalsimulation` generators; the
  SEOBNRv5 models are ports of pySEOBNR 0.3.7, validated against it. The tutorials cite the
  matching `ripple`/LALSuite function or example where one exists.

## Models covered

| Model | Domain | Physics | Tutorial |
|---|---|---|---|
| `IMRPhenomD` | frequency | aligned-spin, `(2,2)` only | [02](generated/02_frequency_domain_models.md) |
| `IMRPhenomXAS` | frequency | aligned-spin, `(2,2)` only (successor to `IMRPhenomD`) | [02](generated/02_frequency_domain_models.md) |
| `IMRPhenomXHM` | frequency | aligned-spin, higher harmonics `(2,1),(3,3),(3,2),(4,4)` | [03](generated/03_higher_order_modes.md) |
| `IMRPhenomXP` | frequency | precessing, `(2,2)` only (dominant-mode twist-up) | [04](generated/04_precession.md) |
| `IMRPhenomXPHM` | frequency | precessing + higher harmonics (full model) | [05](generated/05_imrphenomxphm.md) |
| `TaylorT3` | time | non-spinning, closed-form 3.5PN inspiral | [06](generated/06_time_domain_taylort3.md) |
| `ODEToy` | time (step count) | 0PN toy chirp, ODE-integration architecture scaffold | [07](generated/07_odetoy_eob.md) |
| `SEOBNRv5IMRAlignedSpinF2` | frequency | effective-one-body, aligned spin, `(2,2)` inspiral-merger-ringdown | [10](generated/10_seobnrv5.md) |
| `SEOBNRv5HM` | frequency (time-domain modes via `seobnrv5hm_td_modes`) | effective-one-body, aligned spin, `(2,2),(2,1),(3,3),(3,2),(4,4),(4,3)` | [10](generated/10_seobnrv5.md) |
| `SEOBNRv5PHM` | frequency (time domain via `seobnrv5phm_td`) | effective-one-body, precessing + the same six modes | [10](generated/10_seobnrv5.md) |

Also exported, without a tutorial of their own: `TaylorF2` (frequency-domain PN inspiral),
`SEOBNRv5AlignedSpinF2` (SEOBNRv5 `(2,2)` inspiral only), `SineGaussian` (a burst, not a
compact binary) and the ODE ladder the SEOBNRv5 models were built up through
(`AlignedSpinPNHam`, `AlignedSpinPNHamF2`, `PrecessingSpinPNHam`, `SEOBNRv5AlignedSpin`). See the
[API Reference](api.md).

## Installation

```julia
using Pkg
Pkg.add("GravitationalWaveforms")
```
