"""
# IMRPhenomXPHM -- precessing, multi-harmonic

Port of LALSuite's `IMRPhenomXPHM` (`LALSimIMRPhenomXPHM.c`): the combination of
[`IMRPhenomXHM`](@ref)'s aligned-spin 5-mode content `(2,2),(2,1),(3,3),(3,2),(4,4)` with
[`IMRPhenomXP`](@ref)'s NNLO precession twisting-up, applied to EVERY mode rather than only the
dominant `(2,2)` one.

Structurally this file is thin: essentially all of the physics already lives in
`IMRPhenomXPrecession.jl` (the Euler angles `alpha`/`epsilon`/`cos(beta)`, the polarization-frame
`zeta` rotation) and `IMRPhenomXHM.jl` (the aligned-spin mode content, its alignment chain, and the
spin-weighted spherical harmonics). What is genuinely new here is (a) the Wigner-d
`d^l_{m,±mprime}(beta)` tables for the four non-dominant `(l,mprime)` pairs, (b) the per-mode
twist-and-accumulate function generalizing `_xp_twist_up22` off its hardcoded `mprime = 2`, and
(c) the top-level [`hphc`](@ref) orchestrator.

## Scope (deliberately identical to `IMRPhenomXPrecession.jl`'s)
- **NNLO Euler angles only** (`IMRPhenomXPrecVersion` 101-104), Convention 1. No MSA angles
  (`22x`), no PNR-tuned angles (`3xx`), no multibanding, no antisymmetric-waveform correction.
- **Exactly the 5 modes `IMRPhenomXHM` implements.** LALSuite's `IMRPhenomXPHMTwistUp` also carries
  an `(l,mprime) = (4,3)` block, but its own source comment marks it "only used when twisting
  PhenomHM" (a different, non-X model); `IMRPhenomXHM`'s default `ModeArray` has no `(4,3)`, so it
  is genuinely out of scope rather than an omission.
- **Testing-GR amplitude deviations `damp21`/`damp33` are not implemented.**
  `IMRPhenomXPHMTwistUp` multiplies the `(2,1)` and `(3,3)` transfer sums by `(1 + damp21)` /
  `(1 + damp33)`, read from `XLALSimInspiralWaveformParamsLookupNonGRAmp21`/`...Amp33` (lines
  1835-1836, 2075-2076, 2110-2111). Both default to `0`, so the factors are exactly `1` for any
  GR waveform -- the same class of non-GR testing knob this package skips everywhere else.
- `polarizationSymmetry` is hardcoded `+1` (LALSuite only sets `-1` on the antisymmetric-waveform
  branch, which is out of scope) -- same deliberate limit `_xp_twist_up22` already documents.
- **The PRECESSING final spin is NOT implemented** (inherited scope gap from
  `IMRPhenomXPrecession.jl`, not specific to this file). LALSuite's precessing X models replace the
  aligned-spin final spin with `XLALSimIMRPhenomXPrecessingFinalSpin2017(eta, chi1L, chi2L, chi_p)`
  and recompute `pWF->fRING`/`pWF->fDAMP` from it (`LALSimIMRPhenomX_precession.c` lines 1484,
  1517-1518); every co-precessing mode is then generated with that shifted ringdown frequency.
  This package feeds `IMRPhenomXAS`/`IMRPhenomXHM`'s own ALIGNED-spin `fRING`/`fDAMP` instead. The
  effect is nil below ~100 Hz (the inspiral does not depend on `fRING`) and grows through
  merger/ringdown, reaching **~7% in amplitude** for the configurations compared. It is the dominant disagreement with stock
  LALSuite; forcing LALSuite onto the aligned-spin final spin (`PhenomXPFinalSpinMod = 6`) brings
  every mode into machine-precision agreement across the full band. Fixing it would touch `IMRPhenomXAS.jl`/`IMRPhenomXHM*.jl` and change
  `IMRPhenomXP`'s output too.

## Three implementation details worth stating explicitly (each confirmed against the C source)

1. **Per-mode PN velocity.** Every mode's Euler angles are the SAME functions
   (`_xp_pn_euler_alpha_nnlo`, `_xp_pn_euler_epsilon_nnlo`, `_xp_cos_beta`), just evaluated at a
   RESCALED velocity `v_mprime = cbrt(pi * Mf * 2 / mprime)` instead of the dominant mode's
   `v = cbrt(pi * Mf)` (`Get_alpha_beta_epsilon`, `LALSimIMRPhenomXPHM.c` line 3682:
   `omega = LAL_PI * Mf * 2./mprime`, then `IMRPhenomXWignerdCoefficients(..., omega_cbrt, ...)`).
   So `mprime = 2` modes (`(2,2)` and `(3,2)`) share the dominant-mode `v` exactly.

2. **The `alpha`/`epsilon` offsets are the SAME single pair for all five modes.** A naive reading
   of `Get_alpha_epsilon_offset` (lines 3624-3665) suggests four per-`mprime` offset pairs, but for
   `IMRPhenomXPrecVersion` 101-104 all four are assigned the identical value:
   `LALSimIMRPhenomX_precession.c` lines 1191-1201 call `Get_alphaepsilon_atfref` once with a
   hardcoded `mprime = 2` and copy the result into `alpha_offset`, `alpha_offset_1`,
   `alpha_offset_3` and `alpha_offset_4` alike. `_xp_pn_euler_alpha_nnlo`/`_xp_pn_euler_epsilon_nnlo`
   already subtract `setup.alpha_offset`/`setup.epsilon_offset` internally, so calling them with the
   rescaled `v_mprime` is the complete story -- there is no per-mode setup work at all.

3. **`zeta_polarization` is applied ONCE, to the summed total.** `IMRPhenomXPHMTwistUp` only
   accumulates into a running `*hp`/`*hc`; the `2*zeta` rotation happens after the whole
   `ell`/`emmprime` loop finishes (`LALSimIMRPhenomXPHM.c` lines 1596-1610). `_xp_apply_zeta_rotation`
   is reused completely unchanged.

## Which `(2,2)` mode this uses (a deliberate choice)

The `(2,2)` input mode here is `IMRPhenomXHM`'s own `(2,2)` branch (`_xas_ampl_scalar` +
`_xhm22_phase_full`, i.e. the `timeshift`/`phiref22` alignment chain from `_xhm_system_setup`), NOT
`twist_up22`'s `Ampl(IMRPhenomXAS(),...)`/`Phi(IMRPhenomXAS(),...)` pair. This matters and is not
interchangeable: LALSuite's `IMRPhenomXPHM_hplushcross` calls `IMRPhenomXHMGenerateFDOneMode` for
*every* mode including `(2,2)` (`LALSimIMRPhenomXPHM.c` line 971), so all five modes share one
alignment convention. `Phi(::IMRPhenomXAS, ...)` uses this package's own decoupled `t0`/`phiRef`
external-alignment convention instead, which differs from `_xhm22_phase_full` by a linear-in-`Mf`
term plus a constant. For the single-mode `IMRPhenomXP` that difference is an unobservable overall
time/phase gauge choice (which is why `twist_up22` can and does use the XAS path); for a five-mode
sum it would be a genuine *relative* misalignment between `(2,2)` and the higher modes, i.e. a real
bug. Using the XHM `(2,2)` also inherits `IMRPhenomXHM`'s already-LALSuite-validated absolute
normalization and phase for free.

## Conventions
`m1 >= m2` throughout (this package's mass ordering, same as every other model here);
`chi1*`/`chi2*` are dimensionless spin components in the L-aligned source frame at `f_ref`;
`dL` in Gpc; `f` in Hz. As with `IMRPhenomXP` and `IMRPhenomXHM`, `IMRPhenomXPHM` does NOT
participate in the `Ampl`/`Phi` two-function contract -- see [`IMRPhenomXPHM`](@ref).
"""

# ================================================================================================
# Per-mode twisting-up. Ported from `IMRPhenomXPHMTwistUp` (`LALSimIMRPhenomXPHM.c` lines
# 1811-2250), NNLO-angle branch only, restricted to the five `(l,mprime)` pairs this package's
# `IMRPhenomXHM` provides.
# ================================================================================================

"""
    _xphm_twist_up_mode(l, mprime, v_mprime, setup, hlmprime)

Scalar, non-allocating twisting-up of ONE aligned-spin co-precessing-frame mode `hlmprime`
(`= h_{l,-mprime}(f)`, exactly what `_xhm_hlm_nonmixing`/`_xhm_hlm_32` return for a positive
`mprime`, matching `IMRPhenomXPHMTwistUp`'s own documented input convention: "the loop is with
mprime positive, but the mode has to be the negative one for positive frequencies") into its
`(hp, hc)` contribution to the inertial-frame polarizations. `v_mprime = cbrt(pi * Mf * 2 / mprime)`
is the per-mode rescaled PN velocity (see this file's module docstring, point 1); `setup` is a
[`_xp_precession_setup`](@ref) result.

`(l, mprime)` must be one of `(2,2),(2,1),(3,3),(3,2),(4,4)`; anything else throws `ArgumentError`.
Dispatch is `if`/`elseif` over `Integer` literals, matching `_xhm_swsh_fac`/`_xhm_fring_fdamp`'s
established convention in this package -- `(l, mprime)` are compile-time-known literals at every
call site, never AD targets.

The caller must SUM these contributions over all five modes and apply
[`_xp_apply_zeta_rotation`](@ref) once to the total, never per mode (module docstring, point 3).

# Structure of each branch (all five are the same shape)
With `cBetah = cos(beta/2) = sqrt(|1+cos_beta|/2)`, `sBetah = sin(beta/2) = sqrt(|1-cos_beta|/2)`
(`IMRPhenomXWignerdCoefficients_cosbeta`), the Wigner-d tables `d^l_{m,mprime}` and
`d^l_{m,-mprime}` (Appendix A of arXiv:2004.06503; the `-mprime` table is always generated from the
`+mprime` one by the source's own stated `d^l_{-m,-mprime} = ±(-1)^m d^l_{m,mprime}` symmetry), and
the transfer functions

    A_emm(m)     = exp(-i*m*alpha) * d^l_{m,-mprime} * Y_{l,m}(thetaJN, 0)
    A_emmstar(m) = exp(+i*m*alpha) * d^l_{m,+mprime} * conj(Y_{l,m}(thetaJN, 0))

the sums are, with `minus1l = (-1)^l` (the SAME quantity `_xhm_add_mode!` computes, from the same
`h_{l,-m} = (-1)^l conj(h_{l,m})` identity),

    hp_sum = sum_m (A_emm(m) + minus1l * A_emmstar(m))
    hc_sum = i * sum_m (A_emm(m) - minus1l * A_emmstar(m))

which reproduces the C source's four distinct-looking per-mode sign patterns exactly (`l = 2`:
`+`/`-`, lines 2043/2044 and 2076/2077; `l = 3`: `-`/`+`, lines 2108/2109 and 2141/2142; `l = 4`:
`+`/`-`, lines 2175/2176). The result is `cis(-mprime*epsilon) * hlmprime / 2` times `(hp_sum,
hc_sum)` (line 2216, generalizing `_xp_twist_up22`'s hardcoded `-2.0*epsilon`).

`Y_{l,m}(thetaJN, phi=0)` is real for every `(l,m)` (`Y_lm = fac_lm(theta) * exp(i*m*phi)` and
`phi = 0` identically here, `LALSimIMRPhenomX_precession.c` lines 1280-1302), so `conj` is a no-op
and this function calls `_xhm_swsh_fac` (the real part) directly rather than `_xhm_swsh` -- exactly
the same reasoning `_xp_twist_up22` already documents for its own `l = 2` harmonics.

Every sum is manually unrolled over `m = -l..l` into named scalars (no arrays, tuples, `Ref`s or
loops over indexable containers) so the function stays allocation-free and usable inside an
Enzyme/KernelAbstractions kernel, matching `_xp_twist_up22`'s own established discipline.
"""
@inline function _xphm_twist_up_mode(l::Integer, mprime::Integer, v_mprime, setup, hlmprime)
    T = setup.T
    hlm = Complex{T}(hlmprime)

    # (2,2) is handled before anything else is computed: `mprime = 2` means `v_mprime == v`, so
    # `_xp_twist_up22` (already LALSuite-validated for `IMRPhenomXP`) is exactly this branch and
    # delegating to it avoids duplicating its body here.
    if l == 2 && mprime == 2
        return _xp_twist_up22(v_mprime, setup, hlm)
    end

@Tlit T begin
    cos_beta, alpha, epsilon = _xp_euler_angles(v_mprime, setup)

    cBetah = sqrt(abs(1.0 + cos_beta) / 2.0)
    sBetah = sqrt(abs(1.0 - cos_beta) / 2.0)

    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah * cBetah2
    cBetah4 = cBetah * cBetah3
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah * sBetah2
    sBetah4 = sBetah * sBetah3

    theta = setup.thetaJN

    # exp(i*k*alpha) for the integer k range each l needs (lines 2032-2038 / 2085-2092 /
    # 2153-2162). alpha is real, so exp(-i*k*alpha) = conj(exp(i*k*alpha)).
    cis_a1  = cis(alpha)
    cis_a2  = cis_a1 * cis_a1
    cis_ma1 = conj(cis_a1)
    cis_ma2 = conj(cis_a2)

    if l == 2 && mprime == 1
        # d^2_{m,1}, m = -2..2 (line 2058) and d^2_{m,-1} via d^2_{-m,-1} = -(-1)^m d^2_{m,1}
        # (line 2060).
        d21_m2 = 2.0 * cBetah * sBetah3
        d21_m1 = 3.0 * cBetah2 * sBetah2 - sBetah4
        d21_0  = sqrt(6.0) * (cBetah3 * sBetah - cBetah * sBetah3)
        d21_p1 = cBetah2 * (cBetah2 - 3.0 * sBetah2)
        d21_p2 = -2.0 * cBetah3 * sBetah

        d2m1_m2 = -d21_p2
        d2m1_m1 =  d21_p1
        d2m1_0  = -d21_0
        d2m1_p1 =  d21_m1
        d2m1_p2 = -d21_m2

        Y_m2 = _xhm_swsh_fac(2, -2, theta)
        Y_m1 = _xhm_swsh_fac(2, -1, theta)
        Y_0  = _xhm_swsh_fac(2,  0, theta)
        Y_p1 = _xhm_swsh_fac(2,  1, theta)
        Y_p2 = _xhm_swsh_fac(2,  2, theta)

        Aemm_m2 = cis_a2  * d2m1_m2 * Y_m2
        Aemm_m1 = cis_a1  * d2m1_m1 * Y_m1
        Aemm_0  =           d2m1_0  * Y_0
        Aemm_p1 = cis_ma1 * d2m1_p1 * Y_p1
        Aemm_p2 = cis_ma2 * d2m1_p2 * Y_p2

        Astar_m2 = cis_ma2 * d21_m2 * Y_m2
        Astar_m1 = cis_ma1 * d21_m1 * Y_m1
        Astar_0  =           d21_0  * Y_0
        Astar_p1 = cis_a1  * d21_p1 * Y_p1
        Astar_p2 = cis_a2  * d21_p2 * Y_p2

        # minus1l = (-1)^2 = +1
        hp_sum = (Aemm_m2 + Astar_m2) + (Aemm_m1 + Astar_m1) + (Aemm_0 + Astar_0) +
                 (Aemm_p1 + Astar_p1) + (Aemm_p2 + Astar_p2)
        hc_sum = im * ((Aemm_m2 - Astar_m2) + (Aemm_m1 - Astar_m1) + (Aemm_0 - Astar_0) +
                       (Aemm_p1 - Astar_p1) + (Aemm_p2 - Astar_p2))

        eps_phase = cis(-1.0 * epsilon) * hlm / 2.0
        return (eps_phase * hp_sum, eps_phase * hc_sum)

    elseif l == 3 && mprime == 3
        cBetah5 = cBetah * cBetah4
        cBetah6 = cBetah * cBetah5
        sBetah5 = sBetah * sBetah4
        sBetah6 = sBetah * sBetah5
        cis_a3  = cis_a1 * cis_a2
        cis_ma3 = conj(cis_a3)

        # d^3_{m,3}, m = -3..3 (line 2097) and d^3_{m,-3} (line 2099).
        d33_m3 = sBetah6
        d33_m2 = sqrt(6.0) * cBetah * sBetah5
        d33_m1 = sqrt(15.0) * cBetah2 * sBetah4
        d33_0  = 2.0 * sqrt(5.0) * cBetah3 * sBetah3
        d33_p1 = sqrt(15.0) * cBetah4 * sBetah2
        d33_p2 = sqrt(6.0) * cBetah5 * sBetah
        d33_p3 = cBetah6

        d3m3_m3 =  d33_p3
        d3m3_m2 = -d33_p2
        d3m3_m1 =  d33_p1
        d3m3_0  = -d33_0
        d3m3_p1 =  d33_m1
        d3m3_p2 = -d33_m2
        d3m3_p3 =  d33_m3

        Y_m3 = _xhm_swsh_fac(3, -3, theta)
        Y_m2 = _xhm_swsh_fac(3, -2, theta)
        Y_m1 = _xhm_swsh_fac(3, -1, theta)
        Y_0  = _xhm_swsh_fac(3,  0, theta)
        Y_p1 = _xhm_swsh_fac(3,  1, theta)
        Y_p2 = _xhm_swsh_fac(3,  2, theta)
        Y_p3 = _xhm_swsh_fac(3,  3, theta)

        Aemm_m3 = cis_a3  * d3m3_m3 * Y_m3
        Aemm_m2 = cis_a2  * d3m3_m2 * Y_m2
        Aemm_m1 = cis_a1  * d3m3_m1 * Y_m1
        Aemm_0  =           d3m3_0  * Y_0
        Aemm_p1 = cis_ma1 * d3m3_p1 * Y_p1
        Aemm_p2 = cis_ma2 * d3m3_p2 * Y_p2
        Aemm_p3 = cis_ma3 * d3m3_p3 * Y_p3

        Astar_m3 = cis_ma3 * d33_m3 * Y_m3
        Astar_m2 = cis_ma2 * d33_m2 * Y_m2
        Astar_m1 = cis_ma1 * d33_m1 * Y_m1
        Astar_0  =           d33_0  * Y_0
        Astar_p1 = cis_a1  * d33_p1 * Y_p1
        Astar_p2 = cis_a2  * d33_p2 * Y_p2
        Astar_p3 = cis_a3  * d33_p3 * Y_p3

        # minus1l = (-1)^3 = -1
        hp_sum = (Aemm_m3 - Astar_m3) + (Aemm_m2 - Astar_m2) + (Aemm_m1 - Astar_m1) +
                 (Aemm_0 - Astar_0) + (Aemm_p1 - Astar_p1) + (Aemm_p2 - Astar_p2) +
                 (Aemm_p3 - Astar_p3)
        hc_sum = im * ((Aemm_m3 + Astar_m3) + (Aemm_m2 + Astar_m2) + (Aemm_m1 + Astar_m1) +
                       (Aemm_0 + Astar_0) + (Aemm_p1 + Astar_p1) + (Aemm_p2 + Astar_p2) +
                       (Aemm_p3 + Astar_p3))

        eps_phase = cis(-3.0 * epsilon) * hlm / 2.0
        return (eps_phase * hp_sum, eps_phase * hc_sum)

    elseif l == 3 && mprime == 2
        cBetah5 = cBetah * cBetah4
        cBetah6 = cBetah * cBetah5
        sBetah5 = sBetah * sBetah4
        cis_a3  = cis_a1 * cis_a2
        cis_ma3 = conj(cis_a3)

        # d^3_{m,2}, m = -3..3 (line 2131) and d^3_{m,-2} (line 2133).
        d32_m3 = sqrt(6.0) * cBetah * sBetah5
        d32_m2 = sBetah4 * (5.0 * cBetah2 - sBetah2)
        d32_m1 = sqrt(10.0) * sBetah3 * (2.0 * cBetah3 - cBetah * sBetah2)
        d32_0  = sqrt(30.0) * cBetah2 * (cBetah2 - sBetah2) * sBetah2
        d32_p1 = sqrt(10.0) * cBetah3 * (cBetah2 * sBetah - 2.0 * sBetah3)
        d32_p2 = cBetah4 * (cBetah2 - 5.0 * sBetah2)
        d32_p3 = -sqrt(6.0) * cBetah5 * sBetah

        d3m2_m3 = -d32_p3
        d3m2_m2 =  d32_p2
        d3m2_m1 = -d32_p1
        d3m2_0  =  d32_0
        d3m2_p1 = -d32_m1
        d3m2_p2 =  d32_m2
        d3m2_p3 = -d32_m3

        Y_m3 = _xhm_swsh_fac(3, -3, theta)
        Y_m2 = _xhm_swsh_fac(3, -2, theta)
        Y_m1 = _xhm_swsh_fac(3, -1, theta)
        Y_0  = _xhm_swsh_fac(3,  0, theta)
        Y_p1 = _xhm_swsh_fac(3,  1, theta)
        Y_p2 = _xhm_swsh_fac(3,  2, theta)
        Y_p3 = _xhm_swsh_fac(3,  3, theta)

        Aemm_m3 = cis_a3  * d3m2_m3 * Y_m3
        Aemm_m2 = cis_a2  * d3m2_m2 * Y_m2
        Aemm_m1 = cis_a1  * d3m2_m1 * Y_m1
        Aemm_0  =           d3m2_0  * Y_0
        Aemm_p1 = cis_ma1 * d3m2_p1 * Y_p1
        Aemm_p2 = cis_ma2 * d3m2_p2 * Y_p2
        Aemm_p3 = cis_ma3 * d3m2_p3 * Y_p3

        Astar_m3 = cis_ma3 * d32_m3 * Y_m3
        Astar_m2 = cis_ma2 * d32_m2 * Y_m2
        Astar_m1 = cis_ma1 * d32_m1 * Y_m1
        Astar_0  =           d32_0  * Y_0
        Astar_p1 = cis_a1  * d32_p1 * Y_p1
        Astar_p2 = cis_a2  * d32_p2 * Y_p2
        Astar_p3 = cis_a3  * d32_p3 * Y_p3

        # minus1l = (-1)^3 = -1
        hp_sum = (Aemm_m3 - Astar_m3) + (Aemm_m2 - Astar_m2) + (Aemm_m1 - Astar_m1) +
                 (Aemm_0 - Astar_0) + (Aemm_p1 - Astar_p1) + (Aemm_p2 - Astar_p2) +
                 (Aemm_p3 - Astar_p3)
        hc_sum = im * ((Aemm_m3 + Astar_m3) + (Aemm_m2 + Astar_m2) + (Aemm_m1 + Astar_m1) +
                       (Aemm_0 + Astar_0) + (Aemm_p1 + Astar_p1) + (Aemm_p2 + Astar_p2) +
                       (Aemm_p3 + Astar_p3))

        eps_phase = cis(-2.0 * epsilon) * hlm / 2.0
        return (eps_phase * hp_sum, eps_phase * hc_sum)

    elseif l == 4 && mprime == 4
        cBetah5 = cBetah * cBetah4
        cBetah6 = cBetah * cBetah5
        cBetah7 = cBetah * cBetah6
        cBetah8 = cBetah * cBetah7
        sBetah5 = sBetah * sBetah4
        sBetah6 = sBetah * sBetah5
        sBetah7 = sBetah * sBetah6
        sBetah8 = sBetah * sBetah7
        cis_a3  = cis_a1 * cis_a2
        cis_a4  = cis_a1 * cis_a3
        cis_ma3 = conj(cis_a3)
        cis_ma4 = conj(cis_a4)

        # d^4_{m,4}, m = -4..4 (line 2165) and d^4_{m,-4} (line 2167).
        d44_m4 = sBetah8
        d44_m3 = 2.0 * sqrt(2.0) * cBetah * sBetah7
        d44_m2 = 2.0 * sqrt(7.0) * cBetah2 * sBetah6
        d44_m1 = 2.0 * sqrt(14.0) * cBetah3 * sBetah5
        d44_0  = sqrt(70.0) * cBetah4 * sBetah4
        d44_p1 = 2.0 * sqrt(14.0) * cBetah5 * sBetah3
        d44_p2 = 2.0 * sqrt(7.0) * cBetah6 * sBetah2
        d44_p3 = 2.0 * sqrt(2.0) * cBetah7 * sBetah
        d44_p4 = cBetah8

        d4m4_m4 =  d44_p4
        d4m4_m3 = -d44_p3
        d4m4_m2 =  d44_p2
        d4m4_m1 = -d44_p1
        d4m4_0  =  d44_0
        d4m4_p1 = -d44_m1
        d4m4_p2 =  d44_m2
        d4m4_p3 = -d44_m3
        d4m4_p4 =  d44_m4

        Y_m4 = _xhm_swsh_fac(4, -4, theta)
        Y_m3 = _xhm_swsh_fac(4, -3, theta)
        Y_m2 = _xhm_swsh_fac(4, -2, theta)
        Y_m1 = _xhm_swsh_fac(4, -1, theta)
        Y_0  = _xhm_swsh_fac(4,  0, theta)
        Y_p1 = _xhm_swsh_fac(4,  1, theta)
        Y_p2 = _xhm_swsh_fac(4,  2, theta)
        Y_p3 = _xhm_swsh_fac(4,  3, theta)
        Y_p4 = _xhm_swsh_fac(4,  4, theta)

        Aemm_m4 = cis_a4  * d4m4_m4 * Y_m4
        Aemm_m3 = cis_a3  * d4m4_m3 * Y_m3
        Aemm_m2 = cis_a2  * d4m4_m2 * Y_m2
        Aemm_m1 = cis_a1  * d4m4_m1 * Y_m1
        Aemm_0  =           d4m4_0  * Y_0
        Aemm_p1 = cis_ma1 * d4m4_p1 * Y_p1
        Aemm_p2 = cis_ma2 * d4m4_p2 * Y_p2
        Aemm_p3 = cis_ma3 * d4m4_p3 * Y_p3
        Aemm_p4 = cis_ma4 * d4m4_p4 * Y_p4

        Astar_m4 = cis_ma4 * d44_m4 * Y_m4
        Astar_m3 = cis_ma3 * d44_m3 * Y_m3
        Astar_m2 = cis_ma2 * d44_m2 * Y_m2
        Astar_m1 = cis_ma1 * d44_m1 * Y_m1
        Astar_0  =           d44_0  * Y_0
        Astar_p1 = cis_a1  * d44_p1 * Y_p1
        Astar_p2 = cis_a2  * d44_p2 * Y_p2
        Astar_p3 = cis_a3  * d44_p3 * Y_p3
        Astar_p4 = cis_a4  * d44_p4 * Y_p4

        # minus1l = (-1)^4 = +1
        hp_sum = (Aemm_m4 + Astar_m4) + (Aemm_m3 + Astar_m3) + (Aemm_m2 + Astar_m2) +
                 (Aemm_m1 + Astar_m1) + (Aemm_0 + Astar_0) + (Aemm_p1 + Astar_p1) +
                 (Aemm_p2 + Astar_p2) + (Aemm_p3 + Astar_p3) + (Aemm_p4 + Astar_p4)
        hc_sum = im * ((Aemm_m4 - Astar_m4) + (Aemm_m3 - Astar_m3) + (Aemm_m2 - Astar_m2) +
                       (Aemm_m1 - Astar_m1) + (Aemm_0 - Astar_0) + (Aemm_p1 - Astar_p1) +
                       (Aemm_p2 - Astar_p2) + (Aemm_p3 - Astar_p3) + (Aemm_p4 - Astar_p4))

        eps_phase = cis(-4.0 * epsilon) * hlm / 2.0
        return (eps_phase * hp_sum, eps_phase * hc_sum)

    else
        throw(ArgumentError("_xphm_twist_up_mode: unsupported (l,mprime) = ($l,$mprime); only (2,2),(2,1),(3,3),(3,2),(4,4) are implemented"))
    end
end # @Tlit T
end

# ================================================================================================
# Public API
# ================================================================================================

"""
    IMRPhenomXPHM

Model-dispatch marker for `IMRPhenomXPHM` (precessing + higher-order-mode: twisting-up of all
five `(2,2),(2,1),(3,3),(3,2),(4,4)` aligned-spin `IMRPhenomXHM` modes -- see this file's module
docstring for the full scope). `IMRPhenomXPHM{V}`: `V` is LALSuite's `IMRPhenomXPrecVersion`, 223 (MSA
angles, `IMRPhenomXPrecessionMSA.jl`; LAL's default and GWTC-2.1's; `IMRPhenomXPHM()`) or 102 (NNLO angles,
`IMRPhenomXPHM(102)`). Matches this package's `IMRPhenomD`/`IMRPhenomXAS`/`IMRPhenomXP`/
`IMRPhenomXHM` dispatch-marker pattern (`struct ... end`), but deliberately does **not** participate
in the `Ampl(model,...)`/`Phi(model,...)` two-function contract, for BOTH of the reasons
`IMRPhenomXP` and `IMRPhenomXHM` individually cite: a precessing waveform's `h+`/`hx` are not
related by a fixed inclination-dependent factor, and a genuine sum over independently-evolving
modes has no single frequency-dependent phase. Use [`hphc`](@ref), which returns `(hp, hc)`
directly.
"""
struct IMRPhenomXPHM{V} end
# `V` = LALSuite's `IMRPhenomXPrecVersion`: 223 (MSA angles, LAL's default and GWTC-2.1's) or 102 (NNLO).
IMRPhenomXPHM() = IMRPhenomXPHM{223}()
IMRPhenomXPHM(version::Integer) = (version in (102, 223) ||
    throw(ArgumentError("IMRPhenomXPHM: precession version 102 (NNLO) or 223 (MSA), got $version"));
    IMRPhenomXPHM{Int(version)}())

"""
    _xphm_hphc_scalar(fi, setup, sys, ms21, ms33, ms44, m32)

Non-allocating `(hp, hc)` at a single frequency `fi` (Hz), given the precession setup
([`_xp_precession_setup`](@ref)), the aligned-spin system setup ([`_xhm_system_setup`](@ref)) and
the four non-dominant per-mode coefficient bundles. The `zeta_polarization` rotation is NOT applied
here -- the caller applies it once, to this function's output (which is already the full five-mode
sum), matching LALSuite's own placement (module docstring, point 3).

Factored out as a plain top-level function (not a closure) so the scalar and array branches of
[`hphc`](@ref) share it without capturing `setup`/`sys`/`ms21`/... -- the same rationale
`_xhm_hphc_scalar` documents for its own existence.

# Per-mode PN velocities
`Mf = M * GMsun_over_c3 * fi`, then `v_mprime = cbrt(pi * Mf * 2 / mprime)`, i.e.
`v2 = cbrt(pi*Mf)` for `(2,2)` AND `(3,2)` (both have `mprime = 2`), `v1 = cbrt(2*pi*Mf)` for
`(2,1)`, `v3 = cbrt(2*pi*Mf/3)` for `(3,3)`, `v4 = cbrt(pi*Mf/2)` for `(4,4)`.
"""
@inline function _xphm_hphc_scalar(fi, setup, sys, ms21, ms33, ms44, m32)
    T = sys.T
    fiT = T(fi)
    Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * fiT

@Tlit T begin
    piMf = T(pi) * Mf
    v2 = cbrt(piMf)                 # mprime = 2  -> (2,2) and (3,2)
    v1 = cbrt(2.0 * piMf)           # mprime = 1  -> (2,1)
    v3 = cbrt(2.0 * piMf / 3.0)     # mprime = 3  -> (3,3)
    v4 = cbrt(piMf / 2.0)           # mprime = 4  -> (4,4)
end

    # (2,2): IMRPhenomXHM's own (2,2) branch, NOT IMRPhenomXAS's Ampl/Phi pair -- see this file's
    # module docstring ("Which (2,2) mode this uses") for why the two are not interchangeable here.
    amp22 = _xas_ampl_scalar(sys.amplcoeffs22, fiT)
    phi22 = _xhm22_phase_full(sys.coeffs22, sys.timeshift, sys.phiref22, Mf)
    h22 = T(_XP_HAS_MODE_NORMALIZATION) * Complex{T}(amp22 * cos(phi22), amp22 * sin(phi22))
    h21 = _xhm_hlm_nonmixing(2, fiT, sys, ms21)
    h33 = _xhm_hlm_nonmixing(3, fiT, sys, ms33)
    h44 = _xhm_hlm_nonmixing(4, fiT, sys, ms44)
    h32 = _xhm_hlm_32(fiT, sys, m32)

    hp22, hc22 = _xphm_twist_up_mode(2, 2, v2, setup, h22)
    hp21, hc21 = _xphm_twist_up_mode(2, 1, v1, setup, h21)
    hp33, hc33 = _xphm_twist_up_mode(3, 3, v3, setup, h33)
    hp32, hc32 = _xphm_twist_up_mode(3, 2, v2, setup, h32)
    hp44, hc44 = _xphm_twist_up_mode(4, 4, v4, setup, h44)

    hp = hp22 + hp21 + hp33 + hp32 + hp44
    hc = hc22 + hc21 + hc33 + hc32 + hc44
    return hp, hc
end

"""
    hphc(model::IMRPhenomXPHM, f, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z,
         f_ref, inclination, phiRef, dL; GMsun_over_c3, GMsun_over_c2_Gpc)

End-to-end `IMRPhenomXPHM` waveform: `(hp, hc)` from twisting up all five aligned-spin
`(2,2),(2,1),(3,3),(3,2),(4,4)` modes with the NNLO precession Euler angles, summing them, and
applying the `2*zeta_polarization` rotation once to the total. `f` may be a scalar (returns a
scalar `(hp, hc)` tuple) or an `AbstractVector` (returns same-length `Vector{Complex{T}}`s) --
matching `twist_up22`/`hphc(::IMRPhenomXHM, ...)`'s own convention.

`m1`, `m2` are component masses in solar masses (`m1 >= m2`, this package's mass-ordering
convention); `chi1*`/`chi2*` are dimensionless spin components in the L-aligned (source) frame at
`f_ref`; `inclination` is the angle between the orbital angular momentum `L` and the line of sight
at `f_ref` (the same `IMRPhenomXP` convention -- it feeds `thetaJN`, it is not the observed
inclination of a precessing system); `phiRef` is the reference orbital phase at `f_ref`; `dL` is
the luminosity distance in Gpc.

The argument order deliberately mirrors [`twist_up22`](@ref)'s (`... f_ref, inclination, phiRef,
dL`), not `hphc(::IMRPhenomXHM, ...)`'s (`... dL, inclination, phiRef; f_ref`), since this model's
parameter set is `IMRPhenomXP`'s exactly -- six spin components plus a reference frequency that is
a genuine positional physical input here rather than an alignment keyword.

# `phiRef` enters ONLY through the precession-frame geometry
`phiRef` sets the source-frame line-of-sight vector `N_Sf` (hence `thetaJN` and `kappa`, hence
`alpha0`) and the `X_Sf` vector (hence `zeta_polarization`) -- see [`_xp_precession_setup`](@ref).
It is deliberately NOT passed to `_xhm_system_setup`: LALSuite's Convention 1 overwrites
`pWF->phi0 = 0` before generating any co-precessing mode, so the aligned-spin modes are built at
zero reference phase. See the extended comment in this function's body for why passing it in both
places is a genuine bug (an `mprime`-dependent, i.e. non-global, phase error) rather than a gauge
choice, and `test_phenomxphm.jl`'s "co-precessing modes are generated at phi0 = 0" testset for the
regression guard.

Unlike `twist_up22`, this function exposes no separate `phiRefOrbital` override -- with five modes
sharing one alignment chain there is no single-mode gauge freedom left to make one meaningful, and
the co-precessing reference phase is fixed at 0 by the convention above in any case.

**Note for `IMRPhenomXP`**: `twist_up22` defaults `phiRefOrbital = phiRef` and threads it into
`Phi(::IMRPhenomXAS, ...)`. Convention 1's `pWF->phi0 = 0` applies there too, so that default is
likely also wrong for `IMRPhenomXP` -- but for a SINGLE mode the resulting `exp(2i*phiRef)` is a
global constant phase, which changes neither `|hp|`/`|hc|` nor `dPhi/df`, so it is invisible to
that model's own validation and harmless to its own users. Left alone here deliberately (changing
`IMRPhenomXP`'s public behaviour would be a separate change).

No parameter-domain assertions run here (hot path); see `validate_phenomxp_precession_params` for
an opt-in host-side check.
"""
function hphc(model::IMRPhenomXPHM{V},
    f,
    m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z,
    f_ref, inclination, phiRef, dL;
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
) where {V}
    setup = _xp_precession_setup(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z,
        f_ref, inclination, phiRef; GMsun_over_c3 = GMsun_over_c3, prec_version = Val(V))

    # The aligned-spin mode content uses only the z spin components, exactly as LALSuite's own
    # `XLALSimIMRPhenomXPHM` does (`IMRPhenomXSetWaveformVariables(pWF, m1_SI, m2_SI, chi1z,
    # chi2z, ...)`, `LALSimIMRPhenomXPHM.c` line 684); the in-plane components enter only through
    # the precession setup above.
    #
    # *** THE CO-PRECESSING MODES ARE GENERATED AT phiRef = 0, NOT AT THE USER'S phiRef ***
    # This is not an approximation and not an oversight -- it is LALSuite's Convention-1 behaviour,
    # `LALSimIMRPhenomX_precession.c` lines 828-838: after computing the precession-frame geometry,
    # `IMRPhenomXGetAndSetPrecessionVariables` OVERWRITES `pWF->phi0 = 0` for `convention == 1`
    # (convention 0 sets it to `-phiJ_Sf` instead). Every subsequent
    # `IMRPhenomXHMGenerateFDOneMode` call therefore sees `phi0 = 0`. The user's `phiRef` enters
    # this model exclusively through the precession-frame geometry: the line-of-sight vector
    # `N_Sf` (hence `thetaJN` and `kappa`, hence `alpha0`) and the `X_Sf` vector (hence
    # `zeta_polarization`) -- all of which `_xp_precession_setup` above already computes from the
    # real `phiRef`.
    #
    # Passing the user's `phiRef` here instead is a REAL BUG, not a gauge choice: it multiplies
    # each co-precessing mode by an extra `exp(-i*mprime*phiRef)` while `alpha0` supplies a
    # compensating-looking but MPRIME-DEPENDENT phase, so the five modes end up mutually
    # misaligned. Measured against LALSuite without this (`phiRef = 1.2`): ~6-12% amplitude error and a ~2.4 rad (= 2*phiRef) phase error, versus the
    # machine-precision-level agreement obtained with `zero(phiRef)`.
    mc, eta = _xhm_mc_eta(m1, m2)
    # LALSuite's default `PhenomXPFinalSpinMod = 0` (see `_xas_precessing_final_spin`'s docstring
    # and `twist_up22`'s matching comment, `IMRPhenomXP`'s own precessing-final-spin wiring): every
    # co-precessing mode's ringdown/damping frequency uses the precessing, not aligned-spin, final
    # spin.
    final_spin = setup.final_spin   # version-dependent: see `_xp_precession_setup`
    sys = _xhm_system_setup(mc, eta, chi1z, chi2z, dL, f_ref, zero(phiRef);
        GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
        final_spin_override = final_spin)

    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)

    T = sys.T
    zeta = T(setup.zeta_polarization)

    if f isa AbstractArray
        hp = similar(f, Complex{T})
        hc = similar(f, Complex{T})
        @inbounds for i in eachindex(f)
            hp_i, hc_i = _xphm_hphc_scalar(f[i], setup, sys, ms21, ms33, ms44, m32)
            hp[i], hc[i] = _xp_apply_zeta_rotation(hp_i, hc_i, zeta)
        end
        return hp, hc
    else
        hp_s, hc_s = _xphm_hphc_scalar(f, setup, sys, ms21, ms33, ms44, m32)
        return _xp_apply_zeta_rotation(hp_s, hc_s, zeta)
    end
end
