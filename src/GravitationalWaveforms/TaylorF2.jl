export TaylorF2, validate_taylorf2_params

"""
    TaylorF2

Frequency-domain, closed-form post-Newtonian inspiral model (stationary-phase approximation):
3.5PN point-mass phasing plus spin-orbit (3.5PN)/spin-spin (3PN) terms, restricted (Newtonian,
`f^(-7/6)`) amplitude, and optional 5PN-7PN tidal phasing (`lambda1`/`lambda2`, default `0`) --
this is a single approximant, not a separate "tidal" model, matching how LALSuite's own
`TaylorF2` and ripple's `TaylorF2.py` both fold tidal terms into the same function rather than a
separate one.

No merger/ringdown region exists in this model at all -- the PN series is evaluated up to the
Schwarzschild-ISCO frequency (`Mf = 1/(6^1.5 * pi)`, LALSuite's own default cutoff when no
explicit `fEnd`/tidal-contact frequency is given) and is exactly zero beyond that, matching
`Phi`/`Ampl`'s external-`t0`/`phiRef`-alignment, `ifelse`-branchless, GPU/Enzyme-safe convention
used throughout this package (see [`Phi`](@ref)/[`Ampl`](@ref) for [`IMRPhenomD`](@ref)).

Ported from `LALSimInspiralTaylorF2.c`'s `XLALSimInspiralTaylorF2AlignedPhasing`/
`XLALSimInspiralPNPhasing_F2` (default `phaseO = -1`, i.e. every implemented PN order; default
`amplitudeO = -1`, i.e. no SPA amplitude corrections beyond restricted/Newtonian) via ripple's
`TaylorF2.py`, which is itself a direct transcription of the same LAL source -- same porting
precedent as every other model in this package (validate against real LALSuite output, treat ripple as a convenience cross-reference, not the ground truth).
"""
struct TaylorF2 end

"""
    validate_taylorf2_params(eta, chi1, chi2; lambda1 = 0.0, lambda2 = 0.0, dL = nothing)

Host-side sanity check: `0 < eta <= 0.25`, `-1 <= chi1,chi2 <= 1`, `lambda1,lambda2 >= 0`
(dimensionless tidal deformabilities are non-negative by definition; `0` means "black hole, no
tidal deformation"), and (if `dL` given) `dL > 0`. See [`validate_phenomd_params`](@ref)'s
docstring for why this is a separate, opt-in call rather than something `Phi`/`Ampl` run
themselves (GPU-kernel/Enzyme hot-path compatibility).
"""
function validate_taylorf2_params(eta, chi1, chi2; lambda1 = 0.0, lambda2 = 0.0, dL = nothing)
    0.0 < eta <= 0.25 || throw(ArgumentError("Invalid eta = $eta: must be in (0, 0.25]"))
    -1.0 <= chi1 <= 1.0 || throw(ArgumentError("Invalid chi1 = $chi1: spin must be in [-1, 1]"))
    -1.0 <= chi2 <= 1.0 || throw(ArgumentError("Invalid chi2 = $chi2: spin must be in [-1, 1]"))
    lambda1 >= 0.0 || throw(ArgumentError("Invalid lambda1 = $lambda1: tidal deformability must be >= 0"))
    lambda2 >= 0.0 || throw(ArgumentError("Invalid lambda2 = $lambda2: tidal deformability must be >= 0"))
    if dL !== nothing
        dL > 0.0 || throw(ArgumentError("Invalid dL = $dL: luminosity distance must be positive (and must be in Gpc, see Ampl's docstring)"))
    end
    return nothing
end

# ============================================================================================
# PN coefficient helper functions -- ported one-to-one from ripple's TaylorF2.py
# (`get_3PNSOCoeff`, ..., `get_15PNTidalCoeff`), itself a transcription of
# `LALSimInspiralPNCoefficients.c`. `mByM` is a MASS FRACTION (`m1/M` or `m2/M`, i.e. `mm1`/`mm2`
# in this package's `Seta`-derived convention), not a full mass. Naming keeps ripple's
# "which PN order this term enters the phasing sum at, in v-power units" convention (`3PN` here
# means the `v^3` term, i.e. true 1.5PN spin-orbit -- matches this package's own `phi0..phi8`-style
# naming elsewhere, e.g. IMRPhenomD.jl's `TF2coeffsStructure`).
# ============================================================================================

@inline function _tf2_3PNSO(mByM)
    T = typeof(mByM)
@Tlit T begin
    return mByM * (25.0 + 38.0 / 3.0 * mByM)
end
end

@inline function _tf2_5PNSO(mByM)
    T = typeof(mByM)
@Tlit T begin
    return -mByM * (1391.5 / 8.4 - mByM * (1.0 - mByM) * 10.0 / 3.0 +
                     mByM * (1276.0 / 8.1 + mByM * (1.0 - mByM) * 170.0 / 9.0))
end
end

@inline function _tf2_6PNSO(mByM)
    T = typeof(mByM)
@Tlit T begin
    return pi * mByM * (1490.0 / 3.0 + mByM * 260.0)
end
end

@inline function _tf2_7PNSO(mByM)
    T = typeof(mByM)
@Tlit T begin
    eta = mByM * (1.0 - mByM)
    return mByM * (-17097.8035 / 4.8384 + eta * 28764.25 / 6.72 + eta * eta * 47.35 / 1.44 +
                    mByM * (-7189.233785 / 1.524096 + eta * 458.555 / 3.024 - eta * eta * 534.5 / 7.2))
end
end

@inline function _tf2_4PNS1S2(eta)
    T = typeof(eta)
    @Tlit T (247.0 / 4.8 * eta)
end
@inline function _tf2_4PNS1S2O(eta)
    T = typeof(eta)
    @Tlit T (-721.0 / 4.8 * eta)
end
@inline function _tf2_4PNQM2SO(mByM)
    T = typeof(mByM)
    @Tlit T (-720.0 / 9.6 * mByM * mByM)
end
@inline function _tf2_4PNSelf2SO(mByM)
    T = typeof(mByM)
    @Tlit T (1.0 / 9.6 * mByM * mByM)
end
@inline function _tf2_4PNQM2S(mByM)
    T = typeof(mByM)
    @Tlit T (240.0 / 9.6 * mByM * mByM)
end
@inline function _tf2_4PNSelf2S(mByM)
    T = typeof(mByM)
    @Tlit T (-7.0 / 9.6 * mByM * mByM)
end

@inline function _tf2_6PNS1S2O(eta)
    T = typeof(eta)
    @Tlit T ((326.75 / 1.12 + 557.5 / 1.8 * eta) * eta)
end

@inline function _tf2_6PNSelf2S(mByM)
    T = typeof(mByM)
@Tlit T begin
    return (-4108.25 / 6.72 - 108.5 / 1.2 * mByM + 125.5 / 3.6 * mByM * mByM) * mByM * mByM
end
end

@inline function _tf2_6PNQM2S(mByM)
    T = typeof(mByM)
@Tlit T begin
    return (4703.5 / 8.4 + 2935.0 / 6.0 * mByM - 120.0 * mByM * mByM) * mByM * mByM
end
end

@inline function _tf2_10PNTidal(mByM)
    T = typeof(mByM)
    @Tlit T ((-288.0 + 264.0 * mByM) * mByM * mByM * mByM * mByM)
end

@inline function _tf2_12PNTidal(mByM)
    T = typeof(mByM)
@Tlit T begin
    return (-15895.0 / 28.0 + 4595.0 / 28.0 * mByM + 5715.0 / 14.0 * mByM * mByM -
            325.0 / 7.0 * mByM * mByM * mByM) * mByM * mByM * mByM * mByM
end
end

@inline function _tf2_13PNTidal(mByM)
    T = typeof(mByM)
    @Tlit T (mByM * mByM * mByM * mByM * 24.0 * (12.0 - 11.0 * mByM) * pi)
end

@inline function _tf2_14PNTidal(mByM)
    T = typeof(mByM)
@Tlit T begin
    mByM3 = mByM * mByM * mByM
    mByM4 = mByM3 * mByM
    return -mByM4 * 5.0 * (193986935.0 / 571536.0 - 14415613.0 / 381024.0 * mByM -
                            57859.0 / 378.0 * mByM * mByM - 209495.0 / 1512.0 * mByM3 +
                            965.0 / 54.0 * mByM4 - 4.0 * mByM4 * mByM)
end
end

# Dimensionless Schwarzschild-ISCO frequency (`Mf`), LALSuite's own default TaylorF2 cutoff
# (`XLALSimInspiralTaylorF2`'s `f_max = fISCO` when no explicit `fEnd` is given and tidal terms
# are off, `LALSimInspiralTaylorF2.c` line ~530/559) -- `vISCO = 1/sqrt(6)`, `fISCO = vISCO^3/pi`.
const _TF2_FISCO_MF = (1.0 / sqrt(6.0))^3 / pi

"""
    _tf2_phi_coeffs(model::TaylorF2, mc, eta, chi1, chi2; kwargs...)

Frequency-independent part of the TaylorF2 phase, factored out of [`Phi`](@ref) for the same
reason as every other model's `_xxx_phi_coeffs` in this package (see
[`_phi_coeffs`](@ref)'s docstring): computed once per waveform, not once per frequency bin.
"""
@inline function _tf2_phi_coeffs(model::TaylorF2,
    mc,
    eta,
    chi1,
    chi2;
    lambda1 = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
    lambda2 = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
    fCut = _TF2_FISCO_MF,
    t0 = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
    phiRef = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
    GMsun_over_c3 = GMsun_over_c3,
)
    T = promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2), typeof(lambda1),
        typeof(lambda2), typeof(t0), typeof(phiRef))
    lambda1 = T(lambda1)
    lambda2 = T(lambda2)
    fCut = T(fCut)
    t0 = T(t0)
    phiRef = T(phiRef)
    GMsun_over_c3 = T(GMsun_over_c3)

@Tlit T begin
    M = mc / eta^(3.0 / 5.0)
    Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
    mm1 = 0.5 * (1.0 + Seta)
    mm2 = 0.5 * (1.0 - Seta)

    S1z2 = chi1 * chi1
    S2z2 = chi2 * chi2
    S1S2 = chi1 * chi2

    qm1, _ = _tidal_quadparam_octparam(lambda1)
    qm2, _ = _tidal_quadparam_octparam(lambda2)

    raw0 = 1.0
    raw2 = 5.0 * (74.3 / 8.4 + 11.0 * eta) / 9.0
    raw3 = -16.0 * pi + _tf2_3PNSO(mm1) * chi1 + _tf2_3PNSO(mm2) * chi2
    raw4 = 5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta * eta) / 72.0 +
           _tf2_4PNS1S2(eta) * S1S2 + _tf2_4PNS1S2O(eta) * S1S2 +
           (_tf2_4PNQM2SO(mm1) * qm1 + _tf2_4PNSelf2SO(mm1)) * S1z2 +
           (_tf2_4PNQM2SO(mm2) * qm2 + _tf2_4PNSelf2SO(mm2)) * S2z2 +
           (_tf2_4PNQM2S(mm1) * qm1 + _tf2_4PNSelf2S(mm1)) * S1z2 +
           (_tf2_4PNQM2S(mm2) * qm2 + _tf2_4PNSelf2S(mm2)) * S2z2
    raw5 = 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta) * pi + _tf2_5PNSO(mm1) * chi1 + _tf2_5PNSO(mm2) * chi2
    rawl5 = 5.0 / 3.0 * (772.9 / 8.4 - 13.0 * eta) * pi +
            3.0 * (_tf2_5PNSO(mm1) * chi1 + _tf2_5PNSO(mm2) * chi2)
    rawl6 = -684.8 / 2.1
    raw6 = 11583.231236531 / 4.694215680 - 640.0 / 3.0 * pi * pi -
           684.8 / 2.1 * MathConstants.eulergamma +
           eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * pi * pi) +
           eta * eta * 76.055 / 1.728 - eta * eta * eta * 127.825 / 1.296 +
           rawl6 * log(4.0) +
           _tf2_6PNSO(mm1) * chi1 + _tf2_6PNSO(mm2) * chi2 + _tf2_6PNS1S2O(eta) * S1S2 +
           (_tf2_6PNQM2S(mm1) * qm1 + _tf2_6PNSelf2S(mm1)) * S1z2 +
           (_tf2_6PNQM2S(mm2) * qm2 + _tf2_6PNSelf2S(mm2)) * S2z2
    raw7 = pi * (770.96675 / 2.54016 + 378.515 / 1.512 * eta - 740.45 / 7.56 * eta * eta) +
           _tf2_7PNSO(mm1) * chi1 + _tf2_7PNSO(mm2) * chi2

    rawt10 = lambda1 * _tf2_10PNTidal(mm1) + lambda2 * _tf2_10PNTidal(mm2)
    rawt12 = lambda1 * _tf2_12PNTidal(mm1) + lambda2 * _tf2_12PNTidal(mm2)
    rawt13 = lambda1 * _tf2_13PNTidal(mm1) + lambda2 * _tf2_13PNTidal(mm2)
    rawt14 = lambda1 * _tf2_14PNTidal(mm1) + lambda2 * _tf2_14PNTidal(mm2)
    # 7.5PN (v^15) tidal term deliberately omitted -- zero by the same LALSuite/ripple default.

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
    pft10 = rawt10 * pfaN_prefactor
    pft12 = rawt12 * pfaN_prefactor
    pft13 = rawt13 * pfaN_prefactor
    pft14 = rawt14 * pfaN_prefactor

    return (
        T = T, M = M, GMsun_over_c3 = GMsun_over_c3, fCut = fCut, t0 = t0, phiRef = phiRef,
        pfaN = pfaN, pfa2 = pfa2, pfa3 = pfa3, pfa4 = pfa4, pfa5 = pfa5, pfl5 = pfl5,
        pfl6 = pfl6, pfa6 = pfa6, pfa7 = pfa7,
        pft10 = pft10, pft12 = pft12, pft13 = pft13, pft14 = pft14,
    )
end # @Tlit T
end

"""
    _tf2_phi_scalar(coeffs, f)

Non-allocating TaylorF2 phase evaluation at a single frequency, using the frequency-independent
coefficients from [`_tf2_phi_coeffs`](@ref). See [`_phi_scalar`](@ref) (IMRPhenomD) for why this
split exists (GPU-kernel/Enzyme compatibility).
"""
@inline function _tf2_phi_scalar(coeffs, f)
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
    v10 = v5 * v5
    v12 = v10 * v2
    v13 = v12 * v
    v14 = v13 * v

    phasing = coeffs.pfa7 * v7 +
              (coeffs.pfa6 + coeffs.pfl6 * logv) * v6 +
              (coeffs.pfa5 + coeffs.pfl5 * logv) * v5 +
              coeffs.pfa4 * v4 + coeffs.pfa3 * v3 + coeffs.pfa2 * v2 + coeffs.pfaN +
              coeffs.pft14 * v14 + coeffs.pft13 * v13 + coeffs.pft12 * v12 + coeffs.pft10 * v10
    phasing = phasing / v5

    phis = phasing - coeffs.t0 * fgrid - coeffs.phiRef
    return ifelse(fgrid < coeffs.fCut, phis, zero(T))
end # @Tlit T
end

"""
    Phi(model::TaylorF2, f, mc, eta, chi1, chi2; lambda1=0, lambda2=0, fCut=<ISCO>, t0=0,
        phiRef=0, GMsun_over_c3=GMsun_over_c3)

TaylorF2 frequency-domain phase. Same `t0`/`phiRef` external-alignment, array-in/array-out
(scalar `f` -> scalar result, non-allocating), no-internal-assertion contract as
[`Phi(::IMRPhenomD, ...)`](@ref) -- see that function's docstring for the general rationale.
`lambda1`/`lambda2` (default `0`, i.e. plain BBH) add the 5PN-7PN tidal phasing terms.
"""
function Phi(model::TaylorF2,
    f,
    mc,
    eta,
    chi1,
    chi2;
    lambda1 = zero(eltype(f)),
    lambda2 = zero(eltype(f)),
    fCut = _TF2_FISCO_MF,
    t0 = zero(eltype(f)),
    phiRef = zero(eltype(f)),
    GMsun_over_c3 = GMsun_over_c3,
)
    coeffs = _tf2_phi_coeffs(model, mc, eta, chi1, chi2;
        lambda1 = lambda1, lambda2 = lambda2, fCut = fCut, t0 = t0, phiRef = phiRef,
        GMsun_over_c3 = GMsun_over_c3)
    if f isa AbstractArray
        out = similar(f, coeffs.T)
        @inbounds for i in eachindex(f)
            out[i] = _tf2_phi_scalar(coeffs, f[i])
        end
        return out
    else
        return _tf2_phi_scalar(coeffs, f)
    end
end

"""
    _tf2_ampl_coeffs(model::TaylorF2, mc, eta, chi1, chi2, dL; kwargs...)

Frequency-independent part of the TaylorF2 amplitude, factored out of [`Ampl`](@ref) for the
same reason as [`_ampl_coeffs`](@ref) (IMRPhenomD). The restricted-PN prefactor
`ampPref = 4*M^2/dL * GMsun_over_c2_Gpc * GMsun_over_c3 * sqrt(5*pi*eta/384)` is this package's
own re-derivation, in its own `dL`-in-Gpc unit convention, of `LALSimInspiralTaylorF2.c`'s
`amp0 = -4*m1*m2/r * LAL_MRSUN_SI * LAL_MTSUN_SI * sqrt(LAL_PI/12)` combined with the restricted
(leading-PN, `amplitudeO = -1`) `sqrt(-dEnergy/flux)*v` factor (`dEnergy = -eta*v`,
`flux = (32*eta^2/5)*v^10`, `m1*m2 = M^2*eta`) -- same magnitude as LALSuite/ripple's formula, but
with the sign dropped: LALSuite/ripple's own `amp0` is negative, an artifact of how their `h0`
happens to be assembled (`amp*cos(...) - amp*sin(...)*i`, immediately paired with a phase), not a
physically meaningful sign, and this package's own `Ampl`/`Phi` convention (`h = Ampl*cis(Phi)`,
see `Ampl(::IMRPhenomD, ...)`) needs `Ampl >= 0` since `Phi` alone carries the full phase --
confirmed by comparing against real LALSuite `TaylorF2` output : the negative-sign version reproduced `|LAL|` exactly
but with `rel_amp_diff == 2.0` identically (i.e. `gw_amp == -lal_amp` bit-for-bit), which is what
caught this.
"""
@inline function _tf2_ampl_coeffs(model::TaylorF2,
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
    _tf2_ampl_scalar(coeffs, f)

Non-allocating TaylorF2 restricted-amplitude evaluation at a single frequency (`amp ~
f^(-7/6)`), using the frequency-independent coefficients from [`_tf2_ampl_coeffs`](@ref).
"""
@inline function _tf2_ampl_scalar(coeffs, f)
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
    Ampl(model::TaylorF2, f, mc, eta, chi1, chi2, dL; fCut=<ISCO>, GMsun_over_c3=GMsun_over_c3,
        GMsun_over_c2_Gpc=GMsun_over_c2_Gpc)

TaylorF2 restricted (Newtonian, `amplitudeO = -1` in LALSuite's terms) frequency-domain
amplitude. `dL` in Gpc, same convention as [`Ampl(::IMRPhenomD, ...)`](@ref). Same array-in/
array-out, non-allocating-scalar contract as every other model's `Ampl` in this package.
"""
function Ampl(model::TaylorF2,
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
    coeffs = _tf2_ampl_coeffs(model, mc, eta, chi1, chi2, dL;
        fCut = fCut, GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc)
    if f isa AbstractArray
        out = similar(f, coeffs.T)
        @inbounds for i in eachindex(f)
            out[i] = _tf2_ampl_scalar(coeffs, f[i])
        end
        return out
    else
        return _tf2_ampl_scalar(coeffs, f)
    end
end
