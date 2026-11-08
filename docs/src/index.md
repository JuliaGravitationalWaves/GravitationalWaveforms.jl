# GravitationalWaveforms.jl

A gravitational-wave waveform toolkit: frequency-domain approximants
(`IMRPhenomD`, `IMRPhenomXAS`, `IMRPhenomXHM`, `IMRPhenomXP`, `IMRPhenomXPHM`) and time-domain
post-Newtonian/ODE-integrated models (`TaylorT3`, `ODEToy`), all Float32/Float64-generic. Part of the same ecosystem as
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
- **Coming from `ripple` (JAX) or `lalsimulation` (C)?** Every model here is a direct,
  cross-validated port of the corresponding `lalsimulation` generator — the tutorials cite the
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

## Installation

```julia
using Pkg
Pkg.add("GravitationalWaveforms")
```
