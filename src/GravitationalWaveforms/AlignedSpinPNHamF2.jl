export AlignedSpinPNHamF2, validate_alignedspinpnhamf2_params

"""
    AlignedSpinPNHamF2

The **closed-form, frequency-domain SPA (stationary-phase-approximation) sibling of
[`AlignedSpinPNHam`](@ref)** -- exactly the same relationship [`TaylorF2`](@ref) already has to
[`TaylorT3`](@ref) in this package (a time-domain PN model and its frequency-domain SPA
counterpart, same underlying PN content, two different closed-form parametrizations). Built for
`GravitationalWaveBayes.jl`'s relative-binning likelihood, which needs an `O(1)`-per-frequency-query,
`coeffs`-once/`scalar`-per-frequency waveform (`AlignedSpinPNHam.Phi`/`.Ampl` only accept
**time** and re-integrate an ODE from scratch on every call, which is unusable for relative
binning as-is).

# What this is, precisely
Non-spin content: identical to this package's own [`TaylorF2`](@ref)/[`TaylorT3`](@ref) 3PN
energy / 3.5PN flux (Buonanno, Iyer, Ochsner, Pan & Sathyaprakash, Phys. Rev. D 80, 084043 (2009),
arXiv:0907.0700), reused verbatim (see "Derivation and validation" below for the exact-agreement
proof). Spin content: **the same aligned-spin spin-orbit (1.5-3.5PN) and spin-spin (2-3PN, `S1.S2`
only) SPA phasing coefficients [`TaylorF2`](@ref) already carries** (`_tf2_3PNSO`, `_tf2_5PNSO`,
`_tf2_6PNSO`, `_tf2_7PNSO`, `_tf2_4PNS1S2`, `_tf2_4PNS1S2O`, `_tf2_6PNS1S2O` -- all defined in
`TaylorF2.jl`, cited there to `LALSimInspiralPNCoefficients.c`/ripple's `TaylorF2.py`), reused
directly rather than re-derived, **with the quadrupole-monopole/self-spin (`chi1^2`/`chi2^2`)
terms and all tidal terms dropped** -- because [`AlignedSpinPNHam`](@ref)'s own conservative
Hamiltonian `H_SS = (1/(3 eta r^3))*(X1^2 chi1)*(X2^2 chi2)` (see that file's docstring) is
*purely* a bilinear `S1.S2` coupling; it has no `S1^2`/`S2^2` (quadrupole/self-spin) structure at
all, and this model is scoped to match. No tidal parameters exist here (`AlignedSpinPNHam` is a
BBH-only model).

# Derivation and validation -- what was actually attempted, and what was found

**This is not a blind copy of `TaylorF2`.** The SPA
phasing is derived from `AlignedSpinPNHam`'s *own* validated `H_SO`/`H_SS`/flux content via the standard
energy-balance construction, rather than reusing `TaylorF2` wholesale -- so that construction was
carried out in full, and the result of that exercise is what determined this file's final design.

**Method** (a checkable power-series computation at BigFloat/400-bit precision): given
```
dE/dv = -eta*v*EB(v),           EB(v) = 1 + e2 v^2 + e3 v^3 + e4 v^4 + e6 v^6
F(v)  = (32 eta^2/5) v^10 FB(v), FB(v) = 1 + f2 v^2 + ... + f7 v^7  (+ f6l*log(v)*v^6)
```
with `e2=dETa1` (1PN, `TaylorT3.jl`), `e3=SOcoef` (leading spin-orbit, from `AlignedSpinPNHam`'s
own `H_SO`, matching its docstring's claim that `H_SO`'s `v^5` circular-orbit-energy coefficient
equals `-(1/5)*E3S1O(X1)*X1^2*chi1 - (1/5)*E3S1O(X2)*X2^2*chi2` exactly), `e4=dETa2 - (2/eta)*
SScoef` (2PN, non-spin `TaylorT3.jl` term plus `AlignedSpinPNHam`'s own leading spin-spin `H_SS`
energy correction), `e6=dETa3` (3PN, `TaylorT3.jl`; **no `e5`/`e7`** -- `AlignedSpinPNHam`'s
conservative Hamiltonian carries spin only at leading PN order by explicit, documented design, see
its "Explicit scope limit" section), and `f2..f7` = `AlignedSpinPNHam`'s own `_asph_flux_coeffs`
verbatim (full non-spin 3.5PN + spin-orbit to 3.5PN + spin-spin to 3PN) -- construct
`t(v) = -int (dE/dv)/F(v) dv`, `phi(v) = -int v^3 (dE/dv)/F(v) dv` (term-by-term power-series
integration, tracking the `log(v)` terms that arise both from `F(v)`'s own `3PN`-tail `log(v)`
term and from the well-known "spontaneous" `log(v)` that appears at the `2.5PN` order of `phi(v)`
whenever its integrand's power series passes through a `v^-1` term -- a standard, textbook feature
of this exact construction, not specific to this file), then
`Psi(v) = 2 v^3 t(v) - 2 phi(v) - pi/4`, re-expanded as a `v`-power series.

**Non-spin validation (exact, strong)**: with `chi1=chi2=0`, this construction was checked against
`TaylorF2.jl`'s own `raw0,raw2,raw3,raw4,raw5,rawl5,raw6,rawl6,raw7` at four different `eta`
(`0.25, 0.2479, 0.16, 0.05`) and matched to `~1e-15`-`1e-16` *relative* (BigFloat-precision-limited
agreement, i.e. exact) at **every single PN order including both `log(v)` terms** -- this
confirms the power-series machinery itself (division, integration, log-tracking) is correct, since
it exactly reproduces this package's own already-LALSuite-validated non-spin SPA phasing from raw
energy/flux inputs alone.

**Spin result -- a genuine, surfaced discrepancy, not silently resolved**: repeating the *same*
validated construction with `chi1,chi2 != 0` does **not** reproduce `TaylorF2.jl`'s `_tf2_3PNSO`-
based leading (1.5PN) spin-orbit phasing term, nor its 2PN `S1.S2` term, nor the higher orders --
not by a small correction, but by an `O(1)`-relative amount (e.g. `eta=0.2479, X1=0.546, X2=0.454,
chi1=0.6, chi2=-0.4`: the construction's own 1.5PN spin-orbit phasing contribution is `~0.258`
versus `TaylorF2`'s `_tf2_3PNSO`-based `~0.460`, `reldiff ~ -0.44`; internally self-consistent --
confirmed via an independent check that the same power-series "conversion factor" relating a raw
`E(v)/F(v)` bracket coefficient to its final SPA phasing contribution, extracted from the exact
non-spin match, reproduces this same spin-sector number when applied to
`SOcoef - FSO3(spin part)` directly). **This means the discrepancy is not a bug in the
construction above** (which is proven correct by the non-spin cross-check and is internally
self-consistent) -- it means a *naive* orbital energy-balance built from only `H_SO`'s standard
leading circular-orbit energy contribution and the standard leading spin-orbit flux term is
missing some additional physical ingredient that the published, LALSuite-cited `TaylorF2`
spin-orbit/spin-spin phasing coefficients do carry -- plausibly a genuine subtlety in how spin
enters the relationship between orbital-phase accumulation and energy balance beyond pure
`dE/dt=-F` (the literature's directly-published spin phasing formulas are typically derived by a
more complete route than the bare two-line energy-balance recipe used here, e.g. Bohe, Faye,
Marsat & Porter arXiv:1303.7412's own derivation). **This root cause is not resolved here** --
it remains an open question. **A
promising lead**: `PrecessingSpinPNHam.jl` independently resolved an analogous energy-balance
re-expansion puzzle for the `domega/dt` (`TaylorT4wdot`) series, finding the exact identity
`TaylorT4wdot_3PNSOCoeff(x) = PNFlux_3PNSOCoeff(x) - 2.5*PNEnergy_3PNSOCoeff(x)` at leading spin
order -- a *different* target quantity than the SPA phasing series here, but the same family of
"which precise linear combination of `PNEnergy`/`PNFlux` coefficients reproduces a given published
PN series" question; whoever revisits this discrepancy should check whether an analogous corrected
combination (rather than the bare `e_3 - f_3` used here) closes the gap.

**Resulting engineering decision**: given (1) the non-spin sector is proven to exactly reproduce
`TaylorF2.jl`'s own content via this construction, (2) the spin sector's from-scratch
reconstruction is unvalidated against any independent reference and demonstrably diverges from
the one reference that exists, and (3) shipping a novel, unvalidated spin-phasing formula would be
strictly worse than reusing already-validated content -- this file reuses `TaylorF2.jl`'s own
spin-orbit/spin-spin phasing coefficients directly (restricted to the `S1.S2`-only spin-spin scope
`AlignedSpinPNHam`'s own `H_SS` actually has, per above).

# Direct validation against real `lalsimulation` -- the decisive check

The engineering decision above is not just "reuse what's already in this package" -- it was
independently confirmed against **real LALSuite** (not just `TaylorF2.jl`'s own port), using `lalsimulation` 7.7.1,
`lalsimulation.SimInspiralTaylorF2AlignedPhasing(m1_SI, m2_SI, chi1z, chi2z, lal.CreateDict())`
(returns a `PNPhasingSeries` with `.v[0..8]`/`.vlogv[0..8]`, the exact same `pfaN,pfa2,...,pfa7`/
`pfl5,pfl6` this file computes, same index convention). Result, for the baseline system
(`mc=23.821232072402484, eta=0.24793388429752067`) at both `chi1=chi2=0` and `chi1=0.5,chi2=-0.3`
(see `test/alignedspinpnhamf2_tests.jl`'s `"Phasing coefficients regression vs real lalsimulation"`
testset for the exact baked-in values and full methodology):

- **`pfaN, pfa2, pfa3, pfa5, pfl5, pfl6, pfa7` all match LAL's real output to `~1e-14`-`1e-15`
  relative (i.e. bit-for-bit) at both spin configurations** -- confirming this model's entire
  non-spin sector AND its entire spin-orbit sector (1.5PN through 3.5PN) are byte-for-byte
  identical to what real `lalsimulation` computes, not merely internally self-consistent.
- **`pfa4`/`pfa6` (the 2PN/3PN slots) differ from LAL's *full* TaylorF2 output by *exactly* the
  quadrupole-monopole/self-spin term this model deliberately omits** (`_tf2_4PNQM2SO`,
  `_tf2_4PNSelf2SO`, `_tf2_4PNQM2S`, `_tf2_4PNSelf2S`, and their 3PN analogues, evaluated at
  `qm1=qm2=1` for a BBH) -- adding that exact term back recovers LAL's full value to
  `~1e-13`-`1e-15` relative. This is the sharpest possible confirmation that the one intentional
  scope restriction (`S1.S2`-only spin-spin, matching `AlignedSpinPNHam`'s own `H_SS`) is realized
  *exactly* as intended, with no other discrepancy hiding alongside it.

This is a materially stronger validation than the from-scratch derivation attempt above: it
confirms, against real `lalsimulation` directly, that this file's actual shipped content is
correct up to one precisely-quantified, deliberate omission -- while the from-scratch derivation
section is retained because it is genuine, required derivation work whose (negative, but
informative) result is exactly the kind of finding this project's culture wants surfaced, not
deleted once a cleaner validation route was found.

# Cross-check against `AlignedSpinPNHam`'s own time-domain output
See `test/alignedspinpnhamf2_tests.jl`'s `"Rough cross-check against AlignedSpinPNHam's own
time-domain output"` testset: instantaneous GW frequency `f(t)` from `AlignedSpinPNHam`'s
ODE-integrated trajectory (finite difference of its own `Phi(t)`) is matched, via the standard SPA
"group delay" identity `dPsi/df = 2*pi*t(v)`, against this model's own group delay at the same
frequency, at several `t` well inside `AlignedSpinPNHam`'s own confirmed-stable range
(`t <= ~0.05s`/`r >= ~9.7` for the ecosystem's baseline system, see that file's docstring). This
comparison is deliberately loose (factor-of-a-few, not tight): `AlignedSpinPNHam`'s conservative
sector is only 1PN (see its own "Explicit scope limit" section), a substantially cruder
approximant of the conservative dynamics than the 3PN energy-balance content this SPA model
carries, so an `O(1)`-level disagreement in detailed timing is *expected*, not a red flag -- the
sharp, `lalsimulation`-validated content above is what actually establishes this model's
correctness; this section only confirms the two siblings are qualitatively consistent (same
chirp direction, same order of magnitude).

# API
`Phi(model, f, mc, eta, chi1, chi2; ...)` / `Ampl(model, f, mc, eta, chi1, chi2, dL; ...)` --
identical calling convention to [`TaylorF2`](@ref) (no `lambda1`/`lambda2`, this model is BBH-only)
and identical parameter order to [`AlignedSpinPNHam`](@ref)'s own `Phi`/`Ampl` (`mc, eta, chi1,
chi2[, dL]`), so the two models are drop-in siblings for the same physical configuration. Same
`coeffs`-once/`scalar`-per-frequency, allocation-free, `@Tlit`-generic, array-in/array-out
contract as every other frequency-domain model in this package.
"""
struct AlignedSpinPNHamF2 end

"""
    validate_alignedspinpnhamf2_params(eta, chi1, chi2; dL = nothing)

Host-side sanity check: `0 < eta <= 0.25`, `-1 <= chi1,chi2 <= 1`, and (if `dL` given) `dL > 0`.
Throws `ArgumentError` on violation, `nothing` otherwise. Identical domain to
[`validate_alignedspinpnham_params`](@ref) (this model's time-domain sibling) -- not called by
`Phi`/`Ampl` themselves, matching every other model's documented rationale (hot-path; one-time
domain checks belong at the sampler/prior level).
"""
function validate_alignedspinpnhamf2_params(eta, chi1, chi2; dL = nothing)
    0.0 < eta <= 0.25 || throw(ArgumentError("Invalid eta = $eta: must be in (0, 0.25]"))
    -1.0 <= chi1 <= 1.0 || throw(ArgumentError("Invalid chi1 = $chi1: spin must be in [-1, 1]"))
    -1.0 <= chi2 <= 1.0 || throw(ArgumentError("Invalid chi2 = $chi2: spin must be in [-1, 1]"))
    if dL !== nothing
        dL > 0.0 || throw(ArgumentError("Invalid dL = $dL: luminosity distance must be positive (and must be in Gpc, see Ampl's docstring)"))
    end
    return nothing
end

# Reuses TaylorF2.jl's Schwarzschild-ISCO default cutoff (`_TF2_FISCO_MF`) -- same physical
# reasoning applies unchanged (SPA breaks down beyond ISCO regardless of spin content).

"""
    _asphf2_phi_coeffs(model::AlignedSpinPNHamF2, mc, eta, chi1, chi2; kwargs...)

Frequency-independent part of the phase -- see [`_tf2_phi_coeffs`](@ref) (this model's structural
template) for the general `coeffs`-once rationale. Non-spin terms verbatim `TaylorF2.jl`; spin
terms verbatim `TaylorF2.jl`'s `S1.S2`-only spin-orbit/spin-spin pieces (no quadrupole/self-spin,
no tidal) -- see [`AlignedSpinPNHamF2`](@ref)'s docstring for why.
"""
@inline function _asphf2_phi_coeffs(model::AlignedSpinPNHamF2,
    mc,
    eta,
    chi1,
    chi2;
    fCut = _TF2_FISCO_MF,
    t0 = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
    phiRef = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
    GMsun_over_c3 = GMsun_over_c3,
)
    T = promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2), typeof(t0), typeof(phiRef))
    fCut = T(fCut)
    t0 = T(t0)
    phiRef = T(phiRef)
    GMsun_over_c3 = T(GMsun_over_c3)

@Tlit T begin
    M = mc / eta^(3.0 / 5.0)
    Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
    mm1 = 0.5 * (1.0 + Seta)
    mm2 = 0.5 * (1.0 - Seta)

    S1S2 = chi1 * chi2

    raw0 = 1.0
    raw2 = 5.0 * (74.3 / 8.4 + 11.0 * eta) / 9.0
    raw3 = -16.0 * pi + _tf2_3PNSO(mm1) * chi1 + _tf2_3PNSO(mm2) * chi2
    raw4 = 5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta) / 72.0 +
           _tf2_4PNS1S2(eta) * S1S2 + _tf2_4PNS1S2O(eta) * S1S2
    raw5 = 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta) * pi + _tf2_5PNSO(mm1) * chi1 + _tf2_5PNSO(mm2) * chi2
    rawl5 = 5.0 / 3.0 * (772.9 / 8.4 - 13.0 * eta) * pi +
            3.0 * (_tf2_5PNSO(mm1) * chi1 + _tf2_5PNSO(mm2) * chi2)
    rawl6 = -684.8 / 2.1
    raw6 = 11583.231236531 / 4.694215680 - 640.0 / 3.0 * pi * pi -
           684.8 / 2.1 * MathConstants.eulergamma +
           eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * pi * pi) +
           eta * eta * 76.055 / 1.728 - eta * eta * eta * 127.825 / 1.296 +
           rawl6 * log(4.0) +
           _tf2_6PNSO(mm1) * chi1 + _tf2_6PNSO(mm2) * chi2 + _tf2_6PNS1S2O(eta) * S1S2
    raw7 = pi * (770.96675 / 2.54016 + 378.515 / 1.512 * eta - 740.45 / 7.56 * eta * eta) +
           _tf2_7PNSO(mm1) * chi1 + _tf2_7PNSO(mm2) * chi2

    pfaN_prefactor = 3.0 / (128.0 * eta)
    pfaN = raw0 * pfaN_prefactor
    pfa2 = raw2 * pfaN_prefactor
    pfa3 = raw3 * pfaN_prefactor
    pfa4 = raw4 * pfaN_prefactor
    pfa5 = raw5 * pfaN_prefactor
    pfl5 = rawl5 * pfaN_prefactor
    pfl6 = rawl6 * pfaN_prefactor
    pfa6 = raw6 * pfaN_prefactor
    pfa7 = raw7 * pfaN_prefactor

    return (
        T = T, M = M, GMsun_over_c3 = GMsun_over_c3, fCut = fCut, t0 = t0, phiRef = phiRef,
        pfaN = pfaN, pfa2 = pfa2, pfa3 = pfa3, pfa4 = pfa4, pfa5 = pfa5, pfl5 = pfl5,
        pfl6 = pfl6, pfa6 = pfa6, pfa7 = pfa7,
    )
end # @Tlit T
end

"""
    _asphf2_phi_scalar(coeffs, f)

Non-allocating phase evaluation at a single frequency -- identical structure to
[`_tf2_phi_scalar`](@ref) (no tidal terms).
"""
@inline function _asphf2_phi_scalar(coeffs, f)
    T = coeffs.T
@Tlit T begin
    f_safe = max(f, eps(T))
    fgrid = coeffs.M * coeffs.GMsun_over_c3 * f_safe

    v = cbrt(pi * fgrid)
    logv = log(v)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v
    v7 = v6 * v

    phasing = coeffs.pfa7 * v7 +
              (coeffs.pfa6 + coeffs.pfl6 * logv) * v6 +
              (coeffs.pfa5 + coeffs.pfl5 * logv) * v5 +
              coeffs.pfa4 * v4 + coeffs.pfa3 * v3 + coeffs.pfa2 * v2 + coeffs.pfaN
    phasing = phasing / v5

    phis = phasing - coeffs.t0 * fgrid - coeffs.phiRef
    return ifelse(fgrid < coeffs.fCut, phis, zero(T))
end # @Tlit T
end

"""
    Phi(model::AlignedSpinPNHamF2, f, mc, eta, chi1, chi2; fCut=<ISCO>, t0=0, phiRef=0,
        GMsun_over_c3=GMsun_over_c3)

Frequency-domain SPA phase -- see [`AlignedSpinPNHamF2`](@ref)'s docstring for the physics content
and derivation/validation. Same `t0`/`phiRef` external-alignment, array-in/array-out (scalar `f`
-> scalar result, non-allocating), no-internal-assertion contract as [`Phi(::TaylorF2, ...)`](@ref).
"""
function Phi(model::AlignedSpinPNHamF2,
    f,
    mc,
    eta,
    chi1,
    chi2;
    fCut = _TF2_FISCO_MF,
    t0 = zero(eltype(f)),
    phiRef = zero(eltype(f)),
    GMsun_over_c3 = GMsun_over_c3,
)
    coeffs = _asphf2_phi_coeffs(model, mc, eta, chi1, chi2; fCut = fCut, t0 = t0, phiRef = phiRef,
        GMsun_over_c3 = GMsun_over_c3)
    if f isa AbstractArray
        out = similar(f, coeffs.T)
        @inbounds for i in eachindex(f)
            out[i] = _asphf2_phi_scalar(coeffs, f[i])
        end
        return out
    else
        return _asphf2_phi_scalar(coeffs, f)
    end
end

"""
    _asphf2_ampl_coeffs(model::AlignedSpinPNHamF2, mc, eta, chi1, chi2, dL; kwargs...)

Frequency-independent part of the amplitude -- identical formula to [`_tf2_ampl_coeffs`](@ref)
(the restricted/Newtonian-order SPA amplitude carries no spin dependence in this package's
convention, see that function's docstring for the prefactor derivation).
"""
@inline function _asphf2_ampl_coeffs(model::AlignedSpinPNHamF2,
    mc,
    eta,
    chi1,
    chi2,
    dL;
    fCut = _TF2_FISCO_MF,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    T = promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2), typeof(dL))
    fCut = T(fCut)
    GMsun_over_c3 = T(GMsun_over_c3)
    GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc)

@Tlit T begin
    M = mc / eta^(3.0 / 5.0)
    ampPref = 4.0 * M * M / dL * GMsun_over_c2_Gpc * GMsun_over_c3 * sqrt(5.0 * pi * eta / 384.0)
    return (T = T, M = M, GMsun_over_c3 = GMsun_over_c3, fCut = fCut, ampPref = ampPref)
end
end

"""
    _asphf2_ampl_scalar(coeffs, f)

Non-allocating restricted-amplitude evaluation at a single frequency (`amp ~ f^(-7/6)`) --
identical to [`_tf2_ampl_scalar`](@ref).
"""
@inline function _asphf2_ampl_scalar(coeffs, f)
    T = coeffs.T
@Tlit T begin
    f_safe = max(f, eps(T))
    fgrid = coeffs.M * coeffs.GMsun_over_c3 * f_safe
    v = cbrt(pi * fgrid)
    amp = coeffs.ampPref * v^(-3.5)
    return ifelse(fgrid < coeffs.fCut, amp, zero(T))
end
end

"""
    Ampl(model::AlignedSpinPNHamF2, f, mc, eta, chi1, chi2, dL; fCut=<ISCO>,
        GMsun_over_c3=GMsun_over_c3, GMsun_over_c2_Gpc=GMsun_over_c2_Gpc)

Restricted (Newtonian) frequency-domain amplitude. `dL` in Gpc, same convention as
[`Ampl(::TaylorF2, ...)`](@ref)/[`Ampl(::AlignedSpinPNHam, ...)`](@ref). Same array-in/array-out,
non-allocating-scalar contract as every other model's `Ampl` in this package.
"""
function Ampl(model::AlignedSpinPNHamF2,
    f,
    mc,
    eta,
    chi1,
    chi2,
    dL;
    fCut = _TF2_FISCO_MF,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    coeffs = _asphf2_ampl_coeffs(model, mc, eta, chi1, chi2, dL;
        fCut = fCut, GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc)
    if f isa AbstractArray
        out = similar(f, coeffs.T)
        @inbounds for i in eachindex(f)
            out[i] = _asphf2_ampl_scalar(coeffs, f[i])
        end
        return out
    else
        return _asphf2_ampl_scalar(coeffs, f)
    end
end
