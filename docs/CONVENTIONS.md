# Conventions

Numerical and physical conventions shared across every model in GravitationalWaveforms.jl. Read this once;
every tutorial assumes it. Each point below is documented in more depth on the relevant
docstring — this page just collects them in one place, gwpy/GravitationalWaveData.jl-`docs/CONVENTIONS.md`-style.

## The `h ~ Ampl * exp(i*Phi)` convention

Every frequency-domain model (`IMRPhenomD`, `IMRPhenomXAS`, and the aligned-spin `(2,2)` mode
inside `IMRPhenomXHM`) exposes `Phi(model, f, ...)` and `Ampl(model, f, ...)` separately rather
than a single complex strain. The **restricted, face-on, `(2,2)`-mode-only** polarizations are
recovered as

```julia
hp = -Ampl(model, f, params...) .* cos.(Phi(model, f, params...))
hc = -Ampl(model, f, params...) .* sin.(Phi(model, f, params...))
```

at zero inclination. The time-domain models (`TaylorT3`, `ODEToy`) follow the identical
convention with `t`/`n` in place of `f`. Models that produce more than one mode
(`IMRPhenomXHM`, `IMRPhenomXP`, `IMRPhenomXPHM`) instead expose `h_lm`/`hphc`/`twist_up22`
returning `(hp, hc)` (or a single complex per-mode strain) directly, already summed/rotated over
inclination — see each function's own docstring.

## Units

- Masses (`mc`, `m1`, `m2`, total mass `M`) are always in **solar masses**.
- Luminosity distance `dL` is always in **Gpc, not Mpc** — passing an Mpc value silently gives an
  answer 1000x too large. This is because the amplitude prefactor's `GMsun_over_c2_Gpc` constant
  is pre-expressed in Gpc; see `IMRPhenomD.Ampl`'s docstring for the derivation.
- Aligned spins `chi1`, `chi2` are dimensionless, `[-1, 1]`. Precessing spin components
  (`chi1x, chi1y, chi1z`, ...) are dimensionless Cartesian components with `|chi| <= 1`.
- Frequencies `f` are in Hz; times `t` are in seconds.

## Mass/spin labeling: `m1 >= m2`

Every two-body model assumes **`m1` is the more massive body** and `chi1` is the (aligned or
Cartesian) spin **on `m1`**. Passing `m1 < m2` is undefined behavior for the precessing/HM models
(`validate_phenomxp_precession_params` throws `ArgumentError` if you get this backwards) — swap
bodies and spins together before calling if your own convention differs.

## Time-domain models: `t <= 0`, coalescence at `t = 0`

`TaylorT3` and `ODEToy` take a time argument `t` (or, for `ODEToy`, an integer RK4 step count
`n`, see below) following **LALSuite's own `SimInspiralChooseTDWaveform` output convention**:
`t <= 0`, with `t = 0` nominally "coalescence" (for `TaylorT3` specifically, not the formula's own
literal phasing singularity — see `TaylorT3`'s module docstring for why that distinction matters
and how `t0_shift` corrects for it). Both models also follow the GW-phase-is-twice-orbital-phase
convention (`Phi = 2*phi_orbital`), matching the frequency-domain models' own `Phi` meaning.

`ODEToy`'s domain variable is an integer step count `n`, not a continuous time, because a
fixed-step RK4 integrator only produces states at multiples of its step size `dt`; the
corresponding time is `t = n*dt`.

## SEOBNRv5 models

- `SEOBNRv5IMRAlignedSpinF2` and `SEOBNRv5AlignedSpinF2` follow the `Phi`/`Ampl` interface above
  (`mc`, `eta`); `SEOBNRv5HM` and `SEOBNRv5PHM` follow `IMRPhenomXHM`'s `h_lm`/`hphc` interface
  (component masses).
- `f_min` is the (2,2) starting frequency of the EOB integration; the waveform is zero below it,
  and a mode `(l, m)` starts at `m/2` times it.
- Time origin: `SEOBNRv5IMRAlignedSpinF2` and `SEOBNRv5HM` put `t = 0` at the merger (the
  `(2,2)` peak attachment time), so a likelihood's `t_c` is the merger time. `SEOBNRv5PHM` uses
  pySEOBNR's `t = 0` (the frame-invariant amplitude peak).
- `SEOBNRv5PHM` takes spin 3-vectors in the frame where the orbital angular momentum is along `z`
  at `f_ref` (default `f_min`), and needs a uniform grid `f = k * df` with `1/df` longer than the
  waveform. It swaps the bodies (and adds `pi` to `phiRef`) when `m1 < m2`.
- The time-domain functions `seobnrv5hm_td_modes` and `seobnrv5phm_td` work in geometric units
  (time in units of the total mass `M`, strain in units of `M/dL`).
- `phiRef` follows pySEOBNR/LAL (the spherical harmonics' azimuth is `pi/2 - phiRef`), which differs
  from the Phenom models' reference phase, so polarizations from the two families at the same
  `phiRef` are not directly comparable; compare `h_lm` mode by mode.

## Float32/Float64 genericity

Every model is generic over the element type of its numeric inputs via the `@Tlit` macro
(`TypeGenericity.jl`): pass `Float32` parameters throughout (masses, spins, frequencies) and get
`Float32` output, with no separate code path. This matters for GPU/`KernelAbstractions.jl` use
from `GravitationalWaveBayes.jl`, and is exercised directly in every model's test suite.

## No parameter-domain validation on the hot path

`Phi`, `Ampl`, `hphc`, `twist_up22`, and `h_lm` never check that their inputs are physically
sensible (`0 < eta <= 0.25`, `|chi| <= 1`, `dL > 0`, ...) — they are the hot path, called once per
walker per sampler step, and a host-only bounds check with string interpolation is neither
GPU-kernel-compatible nor useful to redo on every single evaluation. Each model instead ships a
matching `validate_*_params` function (`validate_phenomd_params`, `validate_phenomxas_params`,
`validate_phenomxp_precession_params`, `validate_taylort3_params`, `validate_odetoy_params`) that
throws `ArgumentError` on an out-of-domain value — call it once, explicitly, wherever parameters
first enter your code (a sampler's prior, a script's input parsing), not inside a hot loop.

## Scalar-or-vector domain argument

Every generator accepts either a scalar domain value (`f`, `t`, or `n`) — returning a scalar,
non-allocating — or an `AbstractVector` of them — returning a same-length `Vector`. There is no
separate "batch" API; broadcasting semantics are handled internally as a fast, allocating loop
(not `broadcast`/`.`-fusion — see `IMRPhenomD.Phi`'s docstring for why: allocating loops instead
of broadcast play more predictably with Enzyme's activity analysis in `GravitationalWaveBayes.jl`'s
differentiated call graphs).

## Overridable physical constants

`GMsun_over_c3` and `GMsun_over_c2_Gpc` are keyword arguments on every `Phi`/`Ampl`/`hphc`/
`twist_up22`/`h_lm` method, defaulting to this package's own constants but overridable — useful
for bit-exact cross-validation against a specific `lalsimulation` build's own constant values
(the comparison these numbers were validated against).

## Cross-validation against LALSuite and pySEOBNR

Every Phenom and post-Newtonian model in this package was ported from and numerically
cross-validated against `lalsimulation` (LIGO/Virgo/KAGRA's reference C implementation) during
development; the SEOBNRv5 models were ported from and validated against pySEOBNR 0.3.7. Measured agreement was overwhelmingly machine precision for the aligned-spin
models, and sub-percent to a few percent for some full precessing multi-mode assembly
configurations. The test suite holds reference values for each model.
