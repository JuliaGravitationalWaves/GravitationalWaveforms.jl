export validate_phenomxp_precession_params

"""
    IMRPhenomXPrecession

Precession-frame setup AND twisting-up layer for the IMRPhenomXP "dominant-mode precession"
approximation (twisting up only IMRPhenomXAS's (2,2)-derived aligned-spin mode; the full
multi-harmonic IMRPhenomXHM/XPHM twisting-up is explicitly out of scope, since `IMRPhenomXHM` doesn't
exist in this package yet). This file builds the source-frame -> precession-frame parameter
conversion and the NNLO closed-form PN Euler angles (`alpha(f)`, `epsilon(f)`, `cos_beta(f)`), and
then (below, `_xp_twist_up22`/`twist_up22`) implements the twisting-up sum over spherical
harmonics that turns `(alpha, beta, epsilon)` plus the aligned-spin (2,2) mode into `h+`/`hx`.

Ported from `LALSimIMRPhenomX_precession.c` in LALSuite (local checkout used during porting:
`lalsimulation/lib/LALSimIMRPhenomX_precession.c`), specifically:
  - `IMRPhenomXGetAndSetPrecessionVariables` (source-frame parameter setup: `SL`/`Sperp`/`chi_p`
    around lines 256-286; `J0`/`thetaJ_Sf`/`phiJ_Sf` around lines 764-826; `thetaJN` around lines
    937-963; the frequency-independent `alpha1..alpha5`/`epsilon1..epsilon5` NNLO coefficients and
    `kappa`/`alpha0`/`epsilon0`/`alpha_offset`/`epsilon_offset` around lines 864-877, 1061-1201).
  - `XLALSimIMRPhenomXLPNAnsatz` (line ~1625) and the case-102 "3PN orbital angular momentum"
    coefficient set (lines 659-682) for the orbital angular momentum `L(v)` these formulas need.
  - `IMRPhenomX_PN_Euler_alpha_NNLO` / `IMRPhenomX_PN_Euler_epsilon_NNLO` (lines ~1893-1938) and
    the `cos_beta` NNLO formula in `IMRPhenomXPTwistUp22`'s `case 101-104` branch (lines
    1977-1998).
  - `IMRPhenomX_rotate_z` / `IMRPhenomX_rotate_y` (lines ~3793-3824) for the `kappa`/`alpha0`
    rotation.
  - `IMRPhenomXPTwistUp22` lines 2026-2079 (Wigner-d coefficients, the `m=-2..2`
    transfer-function sum, and the final `hp`/`hc` assembly) and
    `IMRPhenomXWignerdCoefficients_cosbeta` (lines 2100-2111), for `_xp_twist_up22`/`twist_up22`
    below. The spin-weighted spherical harmonics these need are ported from the `l==2` branch of
    `XLALSpinWeightedSphericalHarmonic` in `lal/lib/utilities/SphericalHarmonics.c` (a different
    LALSuite package, `lal` not `lalsimulation`; lines 67-88), evaluated only at `phi = 0` (see
    `_xp_twist_up22`'s docstring).

Citations follow this codebase's existing style (see `IMRPhenomXAS`'s module docstring): the NNLO PN
Euler angle expansion itself is Appendix G of Pratten et al 2020, arXiv:2004.06503 (see also
P Schmidt 2014, http://orca.cf.ac.uk/64062/ and https://dcc.ligo.org/LIGO-T1500602); the effective
precessing-spin parameter `chi_p` is Schmidt, Ohme & Hannam, PRD 91, 024043 (2015).

# Explicit scope limits (do not extend without re-reading the LALSuite source)
- **NNLO angles only** (`IMRPhenomXPrecVersion` 101-104 family in LALSuite). The MSA angle
  prescription (`case 220-224`), the numerical-ODE/SpinTaylor angle prescription
  (`precessing_tag == 3`), and anything PNR-tuned (`IMRPhenomXPNRUseTunedAngles`) are NOT
  implemented.
- Within the NNLO family, this hardcodes LALSuite's **`case 102`** orbital-angular-momentum ansatz
  (3PN, spin corrections through 3PN) rather than exposing all four of 101/102/103/104 -- `102` is
  grouped in LALSuite's own switch alongside the MSA-family default versions (220/221/224/etc.),
  making it the natural single choice for a dominant-mode-only port that isn't exposing a version
  selector yet.
- **Convention 1** (`IMRPhenomXPConvention == 1`) -- LALSuite's own default (see
  `DEFINE_INSERT_FUNC(PhenomXPConvention, INT4, "Convention", 1)` in
  `LALSimInspiralWaveformParams.c`) -- is used throughout for `phiJ_Sf`/`thetaJN`/`alpha0`/
  `epsilon0`. Convention 1 is also the algebraically simplest of the four LALSuite supports:
  `thetaJN` is a direct dot product (`acos(J0.N / |J0|)`, lines 958-963) rather than a
  rotate-then-`acos`-of-`Nz` construction, and `alpha0 = pi - kappa` uniformly (lines 906 and 925
  agree), so no aligned-spin-limit special case is needed for `alpha0` either. Conventions 0/5/6/7
  are NOT implemented.
- Antisymmetric-waveform bookkeeping (`IMRPhenomXAntisymmetricWaveform`) is skipped entirely. This
  also means `_xp_twist_up22` hardcodes `polarizationSymmetry = +1` (LALSuite only sets it to
  `-1` on that antisymmetric-waveform branch, lines 2062/2070-2071) rather than threading it
  through as a parameter.
- **The `zeta_polarization` polarization-frame angle (lines
  967-1042) is NOT safely skippable.**
  It rotates LAL's own "X,Y,N" waveframe (Schmidt et al 2017, arXiv:1703.01076) into the "P,Q,N"
  frame (Arun et al 2008, arXiv:0810.5336) `IMRPhenomXPTwistUp22`'s output is naturally expressed
  in, and `IMRPhenomXPGenerateFD` applies it unconditionally whenever nonzero (LALSimIMRPhenomX.c
  lines 2468-2492) -- there is no configuration-independent reason it vanishes. It IS now
  implemented: `_xp_precession_setup` computes it (frequency-independent, Convention 1 only,
  same restriction as everything else here), and `twist_up22` applies the `2*zeta_polarization`
  rotation as a separate post-processing step, mirroring LALSuite's own two-step structure rather
  than folding it into `_xp_twist_up22` (which stays a narrowly-scoped, faithful port of just
  `IMRPhenomXPTwistUp22` itself).
- `IMRPhenomXPNRUseTunedAngles` (the PNR-tuned-angle branch at the very top of
  `IMRPhenomXPTwistUp22`, lines 1969-1974) and `IMRPhenomXPTwistUp22_NumericalAngles` (used only
  for the numerical-ODE/SpinTaylor angle prescription) are NOT implemented -- `_xp_twist_up22`
  only ever consumes the NNLO angles computed above.
- **Mass ordering**: like this package's existing `IMRPhenomD`/`IMRPhenomXAS` (whose `Seta`/`chia`
  formulas already assume body 1 is the more massive one), this file assumes `m1 >= m2` on input.
  LALSuite's public wrapper (`XLALSimIMRPhenomXPCalculateModelParametersFromSourceFrame`) performs
  an automatic mass/spin swap (`XLALIMRPhenomXPCheckMassesAndSpins`) before calling into the code
  ported here; that swap is NOT replicated -- it is the caller's responsibility, exactly as it
  already is for `IMRPhenomD`/`IMRPhenomXAS`. See `validate_phenomxp_precession_params` for an opt-in
  host-side check.

# A LALSuite internal inconsistency found while porting (see also this port's commit message)
`LALSimIMRPhenomX_precession.c` defines the NNLO alpha/epsilon coefficients (`alpha1..alpha5`,
`epsilon1..epsilon5`) *twice*: once as the actually-used internal fields (lines 1065-1108, set
inside `IMRPhenomXGetAndSetPrecessionVariables` and consumed by `IMRPhenomXPTwistUp22`), and once
more in two standalone *public* wrapper functions (`XLALSimIMRPhenomXPNEuleralphaNNLO` /
`XLALSimIMRPhenomXPNEulerepsilonNNLO`, lines 1774-1888) that are not on the path used to generate
an actual `IMRPhenomXP` waveform. The two versions disagree in the sign of every term linear (or
odd-power) in `delta`, e.g. internal `alpha1 = -35/192 + 5*delta/(64*m1)` vs wrapper
`alpha1 = -35/192 - 5*delta/(64*m1)`, even though both scopes define `delta = sqrt(1-4*eta)`
(positive) and `m1 = 0.5*(1+delta)` identically. This port transcribes the **internal** formulas
(the ones LALSuite's actual waveform generation uses), not the public wrapper -- grabbing the
wrapper instead (it is the more prominently documented of the two, with a full Doxygen block) is
an easy way to silently introduce a wrong-sign bug in a precessing configuration while every
aligned-spin test (where `delta`-odd terms are the only thing so contaminated, and `chi_p`/`Sperp`
are separately forced to exactly zero either way) still passes.
"""

# ================================================================================================
# Orbital angular momentum PN ansatz. Ported from `XLALSimIMRPhenomXLPNAnsatz` (line ~1625) and
# the case-102 coefficient set (lines 659-682) of LALSimIMRPhenomX_precession.c.
# ================================================================================================

@inline function _xp_L_pn_ansatz(v, LNorm, L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L)
    T = promote_type(typeof(v), typeof(LNorm), typeof(L2))
    @Tlit T begin
        x = v * v
        x2 = x * x
        x3 = x * x2
        x4 = x * x3
        sqx = sqrt(x)
        return LNorm * (L0 + L1 * sqx + L2 * x + L3 * (x * sqx) + L4 * x2 + L5 * (x2 * sqx) +
                         L6 * x3 + L7 * (x3 * sqx) + L8 * x4 + L8L * x4 * log(x))
    end
end

# 3PN orbital angular momentum coefficients (LALSuite `IMRPhenomXPrecVersion` case 102), as a
# function of (eta, chi1L, chi2L, delta). Returns the 11-tuple `(L0,...,L8L)` that
# `_xp_L_pn_ansatz` consumes; `L1 = L7 = L8 = L8L = 0` for this version. Ported from lines 668-682.
@inline function _xp_L3PN_coeffs(eta, chi1L, chi2L, delta)
    T = promote_type(typeof(eta), typeof(chi1L), typeof(chi2L), typeof(delta))
    @Tlit T begin
        L0 = 1.0
        L1 = 0.0
        L2 = 3.0 / 2.0 + eta / 6.0
        L3 = (5.0 * (chi1L * (-2.0 - 2.0 * delta + eta) + chi2L * (-2.0 + 2.0 * delta + eta))) / 6.0
        L4 = (81.0 + (-57.0 + eta) * eta) / 24.0
        L5 = (-7.0 * (chi1L * (72.0 + delta * (72.0 - 31.0 * eta) + eta * (-121.0 + 2.0 * eta)) +
                       chi2L * (72.0 + eta * (-121.0 + 2.0 * eta) + delta * (-72.0 + 31.0 * eta)))) / 144.0
        L6 = (10935.0 + eta * (-62001.0 + eta * (1674.0 + 7.0 * eta) + 2214.0 * pi^2)) / 1296.0
        L7 = 0.0
        L8 = 0.0
        L8L = 0.0
        return (L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L)
    end
end

# ================================================================================================
# NNLO PN Euler angle coefficients (frequency-independent). Ported from lines 1061-1108 of
# LALSimIMRPhenomX_precession.c -- see this file's module docstring for why the *internal* (not
# the public-wrapper) formulas are used.
# ================================================================================================

@inline function _xp_alpha_epsilon_coeffs(eta, chi1L, chi2L, delta, m1, chip)
    T = promote_type(typeof(eta), typeof(chi1L), typeof(chi2L), typeof(delta), typeof(m1), typeof(chip))
    @Tlit T begin
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta

        delta2 = delta * delta
        delta3 = delta2 * delta

        m1_2 = m1 * m1
        m1_3 = m1 * m1_2
        m1_4 = m1 * m1_3
        m1_5 = m1 * m1_4
        m1_6 = m1_3 * m1_3
        m1_8 = m1_4 * m1_4

        m2 = 1.0 - m1
        q = m1 / m2

        chi_eff = m1 * chi1L + m2 * chi2L
        chiL = (1.0 + q) * chi_eff / q
        chiL2 = chiL * chiL
        chip2 = chip * chip

        alpha1 = -35.0 / 192.0 + (5.0 * delta) / (64.0 * m1)
        alpha2 = ((15.0 * chiL * delta * m1) / 128.0 - (35.0 * chiL * m1_2) / 128.0) / eta
        alpha3 = -5515.0 / 3072.0 + eta * (-515.0 / 384.0 - (15.0 * delta2) / (256.0 * m1_2) +
                  (175.0 * delta) / (256.0 * m1)) + (4555.0 * delta) / (7168.0 * m1) +
                 ((15.0 * chip2 * delta * m1_3) / 128.0 - (35.0 * chip2 * m1_4) / 128.0) / eta2
        alpha4L = (5.0 * chiL * delta2) / 16.0 - (5.0 * chiL * delta * m1) / 3.0 +
                  (2545.0 * chiL * m1_2) / 1152.0 +
                  ((-2035.0 * chiL * delta * m1) / 21504.0 + (2995.0 * chiL * m1_2) / 9216.0) / eta +
                  ((5.0 * chiL * chip2 * delta * m1_5) / 128.0 - (35.0 * chiL * chip2 * m1_6) / 384.0) / eta3 -
                  (35.0 * pi) / 48.0 + (5.0 * delta * pi) / (16.0 * m1)
        alpha5 = (5.0 * (-190512.0 * delta3 * eta6 +
                  2268.0 * delta2 * eta3 * m1 * (eta2 * (323.0 + 784.0 * eta) + 336.0 * (25.0 * chiL2 + chip2) * m1_4) +
                  7.0 * m1_3 * (8024297.0 * eta4 + 857412.0 * eta5 + 3080448.0 * eta6 +
                        143640.0 * chip2 * eta2 * m1_4 - 127008.0 * chip2 * (-4.0 * chiL2 + chip2) * m1_8 +
                        6048.0 * eta3 * ((2632.0 * chiL2 + 115.0 * chip2) * m1_4 - 672.0 * chiL * m1_2 * pi)) +
                  3.0 * delta * m1_2 * (-5579177.0 * eta4 + 80136.0 * eta5 - 3845520.0 * eta6 +
                        146664.0 * chip2 * eta2 * m1_4 + 127008.0 * chip2 * (-4.0 * chiL2 + chip2) * m1_8 -
                        42336.0 * eta3 * ((726.0 * chiL2 + 29.0 * chip2) * m1_4 - 96.0 * chiL * m1_2 * pi)))) /
                 (6.5028096e7 * eta4 * m1_3)

        epsilon1 = -35.0 / 192.0 + (5.0 * delta) / (64.0 * m1)
        epsilon2 = ((15.0 * chiL * delta * m1) / 128.0 - (35.0 * chiL * m1_2) / 128.0) / eta
        epsilon3 = -5515.0 / 3072.0 + eta * (-515.0 / 384.0 - (15.0 * delta2) / (256.0 * m1_2) +
                    (175.0 * delta) / (256.0 * m1)) + (4555.0 * delta) / (7168.0 * m1)
        epsilon4L = (5.0 * chiL * delta2) / 16.0 - (5.0 * chiL * delta * m1) / 3.0 +
                    (2545.0 * chiL * m1_2) / 1152.0 +
                    ((-2035.0 * chiL * delta * m1) / 21504.0 + (2995.0 * chiL * m1_2) / 9216.0) / eta -
                    (35.0 * pi) / 48.0 + (5.0 * delta * pi) / (16.0 * m1)
        epsilon5 = (5.0 * (-190512.0 * delta3 * eta3 +
                    2268.0 * delta2 * m1 * (eta2 * (323.0 + 784.0 * eta) + 8400.0 * chiL2 * m1_4) -
                    3.0 * delta * m1_2 * (eta * (5579177.0 + 504.0 * eta * (-159.0 + 7630.0 * eta)) +
                          254016.0 * chiL * m1_2 * (121.0 * chiL * m1_2 - 16.0 * pi)) +
                    7.0 * m1_3 * (eta * (8024297.0 + 36.0 * eta * (23817.0 + 85568.0 * eta)) +
                          338688.0 * chiL * m1_2 * (47.0 * chiL * m1_2 - 12.0 * pi)))) /
                   (6.5028096e7 * eta * m1_3)

        return (alpha1, alpha2, alpha3, alpha4L, alpha5, epsilon1, epsilon2, epsilon3, epsilon4L, epsilon5)
    end
end

# ================================================================================================
# Small vector-rotation helpers used only for the `kappa`/`alpha0` construction. Ported from
# `IMRPhenomX_rotate_z`/`IMRPhenomX_rotate_y` (lines ~3793-3824).
# ================================================================================================

@inline function _xp_rotate_z(angle, x, y, z)
    T = promote_type(typeof(angle), typeof(x), typeof(y), typeof(z))
    @Tlit T begin
        ca = cos(angle)
        sa = sin(angle)
        return (x * ca - y * sa, x * sa + y * ca, z)
    end
end

@inline function _xp_rotate_y(angle, x, y, z)
    T = promote_type(typeof(angle), typeof(x), typeof(y), typeof(z))
    @Tlit T begin
        ca = cos(angle)
        sa = sin(angle)
        return (x * ca + z * sa, y, -x * sa + z * ca)
    end
end

"""
    validate_phenomxp_precession_params(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z; f_ref = nothing)

Host-side sanity check for `_xp_precession_setup`'s parameter domain: `m1 >= m2 > 0` (this file's
mass-ordering convention, see its module docstring), `|chi1| <= 1`, `|chi2| <= 1`, and (if `f_ref`
is given) `f_ref > 0`. Throws `ArgumentError` on violation, returns `nothing` otherwise. Not called
from the setup function itself (matching `validate_phenomxas_params`'s rationale).
"""
function validate_phenomxp_precession_params(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z; f_ref = nothing)
    m2 > 0.0 || throw(ArgumentError("Invalid m2 = $m2: must be positive"))
    m1 >= m2 || throw(ArgumentError("Invalid m1 = $m1 < m2 = $m2: this file assumes m1 >= m2 (body 1 is the more massive one); swap bodies/spins before calling"))
    chi1_norm2 = chi1x^2 + chi1y^2 + chi1z^2
    chi2_norm2 = chi2x^2 + chi2y^2 + chi2z^2
    chi1_norm2 <= 1.0 || throw(ArgumentError("Invalid chi1 = ($chi1x, $chi1y, $chi1z): |chi1| must be <= 1"))
    chi2_norm2 <= 1.0 || throw(ArgumentError("Invalid chi2 = ($chi2x, $chi2y, $chi2z): |chi2| must be <= 1"))
    if f_ref !== nothing
        f_ref > 0.0 || throw(ArgumentError("Invalid f_ref = $f_ref: reference frequency must be positive"))
    end
    return nothing
end

"""
    _xp_precession_setup(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination, phiRef;
                          GMsun_over_c3 = GMsun_over_c3)

Compute the frequency-independent precession-frame setup quantities from source-frame parameters,
at reference frequency `f_ref` (Hz). `m1`, `m2` are component masses in solar masses (this file's
convention requires `m1 >= m2`, see the module docstring); `chi1*`/`chi2*` are dimensionless spin
components in the L-aligned (source) frame at `f_ref`.

Beyond the minimal `(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref)` set, this also takes
`inclination` (angle between the orbital angular momentum `L` and the line of sight at `f_ref`) and
`phiRef` (reference orbital phase): LALSuite's `thetaJN` genuinely needs both, since it is the
angle between `J0` and the line-of-sight direction `N`, and `N`'s components in the source frame
depend on both (`Nx_Sf = sin(incl)*cos(pi/2-phiRef)`, `Ny_Sf = sin(incl)*sin(pi/2-phiRef)`,
`Nz_Sf = cos(incl)`; LALSimIMRPhenomX_precession.c lines 865-867) -- it is not re-derivable from
just `J0` and the source-frame spins.

Returns a plain-bits `NamedTuple` (no heap allocation) with fields:
- `T`: the working float type.
- `eta`, `delta`, `m1n`, `m2n`: symmetric mass ratio, `sqrt(1-4eta)` (mass difference, `m1n-m2n`),
  and the total-mass-normalized component masses (`m1n+m2n == 1`).
- `SL`, `Sperp`, `chi_p`: effective aligned/in-plane spin (lines 260-284).
- `J0x`, `J0y`, `J0z`, `J0`, `thetaJ`, `phiJ`: the total angular momentum `J0 = L*Lhat + S1 + S2`
  at `f_ref` in the source frame, its magnitude, its polar angle from `Lhat` (`thetaJ`, *not*
  otherwise used downstream in this file but retained since it is a natural byproduct and is
  a quantity of interest), and its azimuthal angle `phiJ` (lines
  764-826).
- `thetaJN`: angle between `J0` and the line of sight `N` (convention-1 form, lines 958-963) --
  this is the `ytheta` the follow-on twisting-up task will need for the spherical harmonics
  (`yphi = 0`, LALSimIMRPhenomX_precession.c line ~1280).
- `kappa`, `alpha0`, `epsilon0`: the convention-1 rotation angle and Euler-angle integration
  constants (lines 864-877, 906/925, 1170).
- `Lcoeffs`: the 10-tuple of case-102 orbital-angular-momentum PN coefficients
  `(L0,...,L8L)`, for `_xp_cos_beta` to re-evaluate `L(v)` at any frequency.
- `alpha1,alpha2,alpha3,alpha4L,alpha5`, `epsilon1,...,epsilon5`: the frequency-independent NNLO
  Euler angle coefficients (lines 1065-1108).
- `alpha_offset`, `epsilon_offset`: the constant offsets (evaluated once, at `f_ref`) that make
  `alpha(f_ref) == alpha0` and `epsilon(f_ref) == epsilon0` (lines 1191-1201, `Get_alphaepsilon_atfref`).
"""
@inline function _xp_precession_setup_core(
    m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination, phiRef, ::Val{LV};
    GMsun_over_c3 = GMsun_over_c3,
) where {LV}
    T = promote_type(
        typeof(m1), typeof(m2), typeof(chi1x), typeof(chi1y), typeof(chi1z),
        typeof(chi2x), typeof(chi2y), typeof(chi2z), typeof(f_ref),
        typeof(inclination), typeof(phiRef),
    )
    # No @assert here (hot-ish path, called once per waveform / per Enzyme evaluation): use
    # `validate_phenomxp_precession_params` explicitly for host-side sanity checks.
    GMsun_over_c3 = T(GMsun_over_c3)

@Tlit T begin
    Mtot = m1 + m2
    m1n = m1 / Mtot
    m2n = m2 / Mtot
    eta = m1n * m2n
    # m1n - m2n, not sqrt(1 - 4 eta): exact (no cancellation near q = 1) and differentiable at q = 1, where the
    # sqrt form's derivative is Inf * 0 = NaN; >= 0 by the m1 >= m2 precondition
    delta = m1n - m2n

    chi1L = chi1z
    chi2L = chi2z

    m1_2 = m1n * m1n
    m2_2 = m2n * m2n

    # ---- Effective aligned/in-plane spin (Schmidt, Ohme & Hannam, PRD 91, 024043, 2015) ----
    SL = chi1L * m1_2 + chi2L * m2_2

    A1 = 2.0 + (3.0 * m2n) / (2.0 * m1n)
    A2 = 2.0 + (3.0 * m1n) / (2.0 * m2n)
    S1_perp = m1_2 * sqrt(chi1x * chi1x + chi1y * chi1y)
    S2_perp = m2_2 * sqrt(chi2x * chi2x + chi2y * chi2y)
    ASp1 = A1 * S1_perp
    ASp2 = A2 * S2_perp
    num = max(ASp1, ASp2)
    den = A1 * m1_2   # valid since m1n >= m2n is required (this file's mass-ordering convention)
    chi_p = num / den
    Sperp = chi_p * m1_2

    # ---- Reference-frequency orbital angular momentum L(v_ref) and J0 = L*Lhat + S1 + S2 ----
    # L(v) coefficients for L(v_ref) in J0: LAL's 102 set, or the 222/223 set (`IMRPhenomXPrecessionMSA.jl`)
    Lcoeffs = LV == 223 ? _xp_L3PN_coeffs_223(eta, chi1L, chi2L, delta) : _xp_L3PN_coeffs(eta, chi1L, chi2L, delta)
    Mf_ref = Mtot * GMsun_over_c3 * f_ref
    omega_ref = pi * Mf_ref
    v_ref = cbrt(omega_ref)
    LRef = _xp_L_pn_ansatz(v_ref, eta / v_ref, Lcoeffs...)

    J0x = m1_2 * chi1x + m2_2 * chi2x
    J0y = m1_2 * chi1y + m2_2 * chi2y
    J0z = m1_2 * chi1z + m2_2 * chi2z + LRef
    J0 = sqrt(J0x * J0x + J0y * J0y + J0z * J0z)
    J0safe = max(J0, eps(T))

    # `acos(J0z/J0safe)` hits `acos`'s OWN derivative singularity
    # (`d(acos(x))/dx = -1/sqrt(1-x^2)`, infinite at `x=+-1`) whenever the in-plane spin is exactly
    # zero -- the aligned-spin case makes `J0z/J0safe` exactly `+-1` (all of `J0` is along `z`).
    # Confirmed (not assumed) that this SEPARATE singularity, left unfixed, poisons the Enzyme
    # reverse-mode gradient of the ENTIRE `_xp_precession_setup` function -- including `phiJ` and
    # every other field, regardless of whether they even read `thetaJ` -- not just `thetaJ`'s own
    # read-out; the `phiJ`-only fix below is necessary but was NOT sufficient on its own (verified:
    # `phiJ`'s gradient stayed NaN in the full function until this fix was added too).
    #
    # First attempt, reverted: an epsilon-clamp on `acos`'s argument (mirroring `phiJ`'s own "nudge
    # off the exact singular point" fix below) DID fix the gradient, but at the cost of perturbing
    # `thetaJ`'s own VALUE at exact alignment by `~sqrt(2*eps)` -- for any `eps` large enough to keep
    # the derivative merely large-but-finite rather than re-triggering the same pole in floating
    # point, that perturbation (`~4.5e-5` rad at `eps=1e-9`) is far coarser than
    # `test_twisting_up.jl`'s own exact (`rtol=1e-10`) aligned-spin-limit closed-form comparison --
    # confirmed by actually breaking that test (60/63 assertions), not merely suspected. Unlike
    # `phiJ`'s fix (whose substituted point happens to give the SAME `atan(0,eps)==atan(0,0)==0.0`
    # forward value at the pole, so no such tradeoff exists there), `acos`'s pole has no such lucky
    # coincidence: `acos(1-eps) != acos(1)` for any `eps>0`.
    #
    # Actual fix: the standard numerical-analysis identity `acos(z/r) == atan2(sqrt(r^2-z^2), z)`
    # for the polar angle of a vector -- `atan2`'s OWN only singularity is at its origin (both
    # arguments exactly zero simultaneously), which here means `J0perp=J0z=0`, i.e. `J0=0` entirely
    # (the pre-existing, separately-guarded `J0<1e-10` case below), NOT merely `J0perp=0` (in-plane
    # spin zero) -- so this formula has NO pole at exactly the configuration that was breaking
    # `phiJ`. Confirmed empirically, not just algebraically: exact forward values at both the
    # aligned (`0.0`) and anti-aligned (`pi`) limits, a finite (`0.0`) gradient at the pole, and a
    # match to finite differences away from it -- all with zero perturbation to the forward value
    # anywhere, so no exact-reproduction test can be affected by this fix.
    thetaJ = ifelse(J0 < 1.0e-10, zero(T), atan(sqrt(J0x * J0x + J0y * J0y), J0z))
    # Plain `atan(J0y, J0x)` has a
    # perfectly well-defined forward value at the exact aligned-spin origin (`J0x=J0y=0`,
    # `atan(0,0)==0.0`), but its Enzyme reverse-mode derivative is `0/0` there
    # (`d(atan2)/dy = x/(x^2+y^2)`, `d(atan2)/dx = -y/(x^2+y^2)`) -- a live-reachable failure,
    # since `GravitationalWaveBayes.jl`'s own NUTS walker initialization starts every `IMRPhenomXP`/`IMRPhenomXPHM`
    # walker at exactly zero in-plane spin. Fix (the direction that entry already recommended,
    # the same "nudge off the exact singular point rather than special-case the value" technique
    # `GravitationalWaveBayes.jl`'s own `BatchedNUTS._clamp_unit` uses for the `asin`/`acos` poles): whenever the
    # in-plane radius is below a tiny threshold, substitute a fixed, arbitrary-but-nonzero point
    # `(1e-6, 0)` for `(J0x, J0y)` -- both branches of this `ifelse` are always finite
    # (no division), so this cannot NaN-contaminate a reverse-mode cotangent the way an unguarded
    # `ifelse` around a divide-by-the-active-values would (see
    # `SEOBNRv5AlignedSpinPAFixed.jl` padding-region fix for the general form of that trap). Away
    # from the origin (`J0perp >= 1e-20`) this is the identity -- `phiJ`'s value is completely
    # unchanged for every genuinely-precessing configuration. The `1e-6` floor (not `eps(T)`,
    # unlike `J0safe` two lines above): the true derivative at this coordinate singularity is
    # mathematically unbounded, not merely large, so any floor only trades "NaN" for "some large
    # but finite number" -- `eps(T)~2.2e-16` would make that number `~4.5e15`, uncomfortably close
    # to overflowing further Float64 arithmetic downstream, whereas `1e-6` keeps it a comfortable
    # `~1e6`, still clearly flagging "near-singular" without courting a second overflow failure.
    # The threshold is on the squared radius and must stay far below any real spin: at 1e-12 (J0perp < 1e-6, in-plane
    # spins up to about 4e-6) it replaced phiJ's value, a 3.5% waveform difference from LAL.
    J0perp2 = J0x * J0x + J0y * J0y   # squared radius -- NOT `sqrt(J0perp2)` (see below)
    J0x_safe = ifelse(J0perp2 < 1.0e-40, 1.0e-6, J0x)
    J0y_safe = ifelse(J0perp2 < 1.0e-40, zero(T), J0y)
    phiJ = atan(J0y_safe, J0x_safe)

    # ---- Line of sight N in the source frame (lines 865-867) and thetaJN (convention 1) ----
    Nx = sin(inclination) * cos(pi / 2.0 - phiRef)
    Ny = sin(inclination) * sin(pi / 2.0 - phiRef)
    Nz = cos(inclination)
    J0dotN = J0x * Nx + J0y * Ny + J0z * Nz
    # NOTE: `acos`'s derivative pole at `+-1` (same class as `thetaJ`'s own fix above) is also
    # LATENT here whenever inclination is exactly `0`/`pi` (face-on/face-off) -- tried a
    # `thetaJ`-style tiny-margin clamp here too, but reverted it: unlike `thetaJ`'s fix (confirmed
    # to actually fix a live, reachable NaN gradient, see above), this one is speculative -- no
    # confirmed reachable failure -- and it broke `test_precession_frame.jl`'s own exact
    # (`atol=1e-10`) "thetaJN reduces to inclination in the aligned-spin limit" test, since the
    # margin needed to matter for the derivative (`~1e-9`) is far coarser than that test's own
    # tolerance. Left as the original exact-boundary clamp; revisit only if a real gradient failure
    # at this specific configuration is ever actually observed (unlike `thetaJ`, which was).
    thetaJN = acos(clamp(J0dotN / J0safe, -1.0, 1.0))

    # ---- kappa / alpha0 / epsilon0 (convention 1, lines 864-877, 906, 925, 1170) ----
    tx, ty, tz = _xp_rotate_z(-phiJ, Nx, Ny, Nz)
    tx, ty, tz = _xp_rotate_y(-thetaJ, tx, ty, tz)
    kappa = atan(ty, tx)
    alpha0 = pi - kappa
    epsilon0 = phiJ - pi

    # ---- NNLO Euler angle coefficients and their f_ref offsets (lines 1065-1108, 1191-1201) ----
    alpha1, alpha2, alpha3, alpha4L, alpha5,
        epsilon1, epsilon2, epsilon3, epsilon4L, epsilon5 =
        _xp_alpha_epsilon_coeffs(eta, chi1L, chi2L, delta, m1n, chi_p)

    omega_ref_cbrt = v_ref
    omega_ref_cbrt2 = v_ref * v_ref
    logomega_ref = log(omega_ref)

    alpha_offset = (alpha1 / omega_ref + alpha2 / omega_ref_cbrt2 + alpha3 / omega_ref_cbrt +
                     alpha4L * logomega_ref + alpha5 * omega_ref_cbrt) - alpha0
    epsilon_offset = (epsilon1 / omega_ref + epsilon2 / omega_ref_cbrt2 + epsilon3 / omega_ref_cbrt +
                       epsilon4L * logomega_ref + epsilon5 * omega_ref_cbrt) - epsilon0

    # ---- zeta_polarization (Convention 1, lines 967-1042): rotates LAL's own "X,Y,N" waveframe
    # (Schmidt et al 2017, arXiv:1703.01076) into the "P,Q,N" frame (Arun et al 2008,
    # arXiv:0810.5336) IMRPhenomXPTwistUp22's hp/hc are naturally expressed in. Frequency-
    # independent (computed once here, from f_ref-anchored geometry only), applied by
    # `twist_up22` as a post-hoc rotation of hp/hc by `2*zeta_polarization`, exactly mirroring
    # `IMRPhenomXPGenerateFD`'s own separate post-processing step (LALSimIMRPhenomX.c lines
    # 2468-2492) rather than being folded into `_xp_twist_up22` itself (which stays a faithful,
    # narrowly-scoped port of just `IMRPhenomXPTwistUp22`).
    Xx_Sf = -cos(inclination) * sin(phiRef)
    Xy_Sf = -cos(inclination) * cos(phiRef)
    Xz_Sf = sin(inclination)
    txz1, tyz1, tzz1 = _xp_rotate_z(-phiJ, Xx_Sf, Xy_Sf, Xz_Sf)
    txz2, tyz2, tzz2 = _xp_rotate_y(-thetaJ, txz1, tyz1, tzz1)
    txz3, tyz3, tzz3 = _xp_rotate_z(-kappa, txz2, tyz2, tzz2)
    # Convention 1: N already lies in the J-frame's (pos x)-z half-plane by construction (lines
    # 955-961), so its components reduce to this direct closed form rather than needing its own
    # 3-rotation -- P = (Nz_Jf, 0, -Nx_Jf), Q = (0, 1, 0) (lines 1018-1032).
    Nz_Jf = cos(thetaJN)
    Nx_Jf = sin(thetaJN)
    XdotP = txz3 * Nz_Jf - tzz3 * Nx_Jf
    XdotQ = tyz3
    zeta_polarization = atan(XdotQ, XdotP)

    return (
        T = T, eta = eta, delta = delta, m1n = m1n, m2n = m2n,
        SL = SL, Sperp = Sperp, chi_p = chi_p,
        J0x = J0x, J0y = J0y, J0z = J0z, J0 = J0, thetaJ = thetaJ, phiJ = phiJ, thetaJN = thetaJN,
        kappa = kappa, alpha0 = alpha0, epsilon0 = epsilon0,
        Lcoeffs = Lcoeffs,
        alpha1 = alpha1, alpha2 = alpha2, alpha3 = alpha3, alpha4L = alpha4L, alpha5 = alpha5,
        epsilon1 = epsilon1, epsilon2 = epsilon2, epsilon3 = epsilon3, epsilon4L = epsilon4L, epsilon5 = epsilon5,
        alpha_offset = alpha_offset, epsilon_offset = epsilon_offset,
        zeta_polarization = zeta_polarization,
    )
end # @Tlit T
end

"""
    _xp_cos_beta(v, setup)

Non-allocating evaluation of `cos(beta(v))`, the NNLO precession opening angle between `L(v)` and
`J0`, at PN velocity parameter `v = (pi*M*f)^(1/3)` (geometric units). Ported from the
`case 101-104` branch of `IMRPhenomXPTwistUp22` (lines 1990-1996):
`s = Sperp / (L + SL)`, `cos_beta = copysign(1, L+SL) / sqrt(1+s^2)`, where `L = L(v)` (*not*
`L(v_ref)`) comes from re-evaluating the same case-102 PN ansatz used for `J0` at the current `v`.

In the aligned-spin limit (`Sperp == 0`), this reduces exactly to `copysign(1, L+SL)` (`beta == 0`,
no precession), independent of `v` -- the sanity check this port's tests rely on most directly.
"""
@inline function _xp_cos_beta(v, setup)
    T = setup.T
@Tlit T begin
    L = _xp_L_pn_ansatz(v, setup.eta / v, setup.Lcoeffs...)
    LplusSL = L + setup.SL
    s = setup.Sperp / LplusSL
    return copysign(1.0, LplusSL) / sqrt(1.0 + s * s)
end
end

"""
    _xp_pn_euler_alpha_nnlo(v, setup)

Non-allocating NNLO PN Euler angle `alpha(v)`, `v = (pi*M*f)^(1/3)`. Ported from
`IMRPhenomX_PN_Euler_alpha_NNLO` (lines 1893-1914): with `omega = v^3` (so `omega_cbrt = v`,
`omega_cbrt2 = v^2`), `alpha = alpha1/omega + alpha2/omega_cbrt2 + alpha3/omega_cbrt +
alpha4L*log(omega) + alpha5*omega_cbrt - alpha_offset`, using the frequency-independent
coefficients from `_xp_precession_setup`.
"""
@inline function _xp_pn_euler_alpha_nnlo(v, setup)
    T = setup.T
@Tlit T begin
    omega = v * v * v
    omega_cbrt = v
    omega_cbrt2 = v * v
    logomega = log(omega)
    return setup.alpha1 / omega + setup.alpha2 / omega_cbrt2 + setup.alpha3 / omega_cbrt +
           setup.alpha4L * logomega + setup.alpha5 * omega_cbrt - setup.alpha_offset
end
end

"""
    _xp_pn_euler_epsilon_nnlo(v, setup)

Non-allocating NNLO PN Euler angle `epsilon(v)`, the `epsilon`-angle analogue of
[`_xp_pn_euler_alpha_nnlo`](@ref). Ported from `IMRPhenomX_PN_Euler_epsilon_NNLO` (lines
1916-1938).
"""
@inline function _xp_pn_euler_epsilon_nnlo(v, setup)
    T = setup.T
@Tlit T begin
    omega = v * v * v
    omega_cbrt = v
    omega_cbrt2 = v * v
    logomega = log(omega)
    return setup.epsilon1 / omega + setup.epsilon2 / omega_cbrt2 + setup.epsilon3 / omega_cbrt +
           setup.epsilon4L * logomega + setup.epsilon5 * omega_cbrt - setup.epsilon_offset
end
end

# ================================================================================================
# Twisting-up: combine the NNLO Euler angles above with IMRPhenomXAS's aligned-spin (2,2) mode to
# produce the inertial-frame h+/hx (dominant-mode precession only -- see module docstring for the
# full scope statement). Ported from `IMRPhenomXPTwistUp22` (lines 1941-2098), specifically the
# `case 101-104` NNLO angle branch (lines 1990-1996, already implemented above by
# `_xp_cos_beta`/`_xp_pn_euler_alpha_nnlo`/`_xp_pn_euler_epsilon_nnlo`) followed by the Wigner-d
# construction, spin-weighted spherical harmonics, transfer-function sum, and final hp/hc assembly
# (lines 2026-2079), plus `IMRPhenomXWignerdCoefficients_cosbeta` (lines 2100-2111) for the
# beta-half-angle formula and the `l==2` branch of `XLALSpinWeightedSphericalHarmonic`
# (`lal/lib/utilities/SphericalHarmonics.c`, lines 67-88) for the spherical harmonics.
# ================================================================================================

"""
    IMRPhenomXP

Model-dispatch marker for the IMRPhenomXP dominant-mode-precession approximation (see this
module's docstring for the exact scope: NNLO angles, `(2,2)`-only twisting-up of IMRPhenomXAS, no
multi-harmonic XHM/XPHM). Matches this package's existing `IMRPhenomD`/`IMRPhenomXAS` dispatch-marker
pattern (`struct IMRPhenomD end`, `struct IMRPhenomXAS end`), but deliberately does **not** participate
in the `Ampl(model, ...)`/`Phi(model, ...)` two-function contract those models use: a precessing
waveform's `h+`/`hx` are not related by a fixed inclination-dependent factor the way an
aligned-spin model's are (the whole point of twisting-up is that precession mixes amplitude and
phase between the two polarizations in a frequency-dependent way), so forcing `IMRPhenomXP` through
that contract would misrepresent the physics. Use [`twist_up22`](@ref) instead, which returns
`(hp, hc)` directly.
"""
struct IMRPhenomXP end

"""
    _xp_twist_up22(v, setup, hAS)

Genuinely scalar, non-allocating twisting-up of the aligned-spin `(2,2)` mode strain `hAS`
(`IMRPhenomXAS`'s `Ampl(...) * exp(im*Phi(...))` at PN velocity `v = (pi*M*f)^(1/3)`, computed by the
caller -- see [`twist_up22`](@ref)) into the inertial-frame polarizations `(hp, hc)`, using the
Euler angles `alpha(v)`/`epsilon(v)`/`cos(beta(v))` from `setup` (a `_xp_precession_setup` result)
via `_xp_pn_euler_alpha_nnlo`/`_xp_pn_euler_epsilon_nnlo`/`_xp_cos_beta`. Ported from
`IMRPhenomXPTwistUp22` lines 2026-2079 (Wigner-d coefficients, the `m=-2..2` transfer-function
sum, and the final `hp`/`hc` assembly); no `acos` is needed anywhere, matching that source.

Written as straight-line scalar arithmetic (no arrays, `Ref`s, closures, or loops over a
tuple/array) so it stays allocation-free and can later be called from inside an Enzyme/
KernelAbstractions kernel (wiring that up is GravitationalWaveBayes.jl's job, not this function's) -- the
`m=-2:2` sum from the LALSuite source is manually unrolled into five named terms rather than
looped over an indexable container.

# Wigner-d half-angle coefficients (`IMRPhenomXWignerdCoefficients_cosbeta`, lines 2100-2111)
`cos(beta/2) = sqrt(|1+cos_beta|/2)`, `sin(beta/2) = sqrt(|1-cos_beta|/2)`, computed directly from
`cos_beta` (no `acos` of `cos_beta` itself, matching the source).

# Spin-weighted spherical harmonics (`XLALSpinWeightedSphericalHarmonic`, `s=-2,l=2,m=-2..2`,
`SphericalHarmonics.c` lines 67-88)
Evaluated at `theta = setup.thetaJN`, `phi = 0` (`IMRPhenomXPTwistUp22` always calls with
`yphi = 0.0`; `thetaJN` is the only angle that matters here -- see the module docstring's
Convention-1 discussion). Since `phi = 0` identically, the general complex phase factor
`exp(i*m*phi)` (`SphericalHarmonics.c` line 419) is exactly `1` for every `m`, so each `Y_{2,m}`
below is purely real and `conj(Y_{2,m}) == Y_{2,m}` -- the `A22emmstar` terms below omit the
`conj` call for exactly this reason (not an oversight).

# `polarizationSymmetry`
Hardcoded to `+1.0` here (LALSuite line 2062, `pPrec->PolarizationSymmetry`). LALSuite only sets
this to `-1` on the antisymmetric-waveform branch (`IMRPhenomXAntisymmetricWaveform`), which this
port explicitly skips (see module docstring's scope-limits section) -- hardcoding `+1` here is a
deliberate scope limit, not an oversight.

# Aligned-spin limit (the primary correctness check for this function, see `test_twisting_up.jl`)
When `Sperp == 0` exactly, `cos_beta == copysign(1, L+SL)` exactly (per `_xp_cos_beta`'s own
docstring), so `(cBetah, sBetah)` take the exact values `(1, 0)` (`beta == 0`, `L+SL > 0`) or
`(0, 1)` (`beta == pi`, `L+SL < 0`). In the `beta == 0` case, every Wigner-d coefficient except
`d2m2[m=-2] = cBetah^4 = 1` and `d22[m=2] = cBetah^4 = 1` vanishes, collapsing the five-term sum
to `hp_sum = exp(2i*alpha)*(Y_{2,-2}+Y_{2,2})`, `hc_sum = i*exp(2i*alpha)*(Y_{2,-2}-Y_{2,2})`; the
`beta == pi` case collapses analogously to `hp_sum = exp(-2i*alpha)*(Y_{2,2}+Y_{2,-2})`,
`hc_sum = i*exp(-2i*alpha)*(Y_{2,2}-Y_{2,-2})` (only `d2m2[m=2] = sBetah^4 = 1` and
`d22[m=-2] = sBetah^4 = 1` survive). Both closed forms are hand-derived from this function's own
sum (not from any reference we can run here) and checked numerically in the test suite.
"""
@inline function _xp_twist_up22(v, setup, hAS)
    T = setup.T
    hAST = Complex{T}(hAS)
@Tlit T begin
    cos_beta, alpha, epsilon = _xp_euler_angles(v, setup)

    cBetah = sqrt(abs(1.0 + cos_beta) / 2.0)
    sBetah = sqrt(abs(1.0 - cos_beta) / 2.0)

    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah * cBetah2
    cBetah4 = cBetah * cBetah3
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah * sBetah2
    sBetah4 = sBetah * sBetah3

    sqrt6 = sqrt(6.0)

    # Wigner-d coefficients d^2_{m,2}(beta) [d22_*] and d^2_{m,-2}(beta) [d2m2_*], m = -2..2.
    # Ported from lines 2044-2048 (Appendix A of arXiv:2004.06503); d2m2 reuses d22 via the
    # symmetry d^2_{-2,m} = (-1)^m d^2_{2,-m}, exactly as LALSuite's own d2m2[] initializer does.
    d22_m2 = sBetah4
    d22_m1 = 2.0 * cBetah * sBetah3
    d22_0  = sqrt6 * sBetah2 * cBetah2
    d22_p1 = 2.0 * cBetah3 * sBetah
    d22_p2 = cBetah4

    d2m2_m2 = d22_p2
    d2m2_m1 = -d22_p1
    d2m2_0  = d22_0
    d2m2_p1 = -d22_m1
    d2m2_p2 = d22_m2

    # Spin-weighted spherical harmonics Y_{2,m}(thetaJN, phi=0), s=-2, m=-2..2 (see docstring).
    theta = setup.thetaJN
    ct = cos(theta)
    st = sin(theta)
    Y2m2 = sqrt(5.0 / (64.0 * pi)) * (1.0 - ct) * (1.0 - ct)
    Y2m1 = sqrt(5.0 / (16.0 * pi)) * st * (1.0 - ct)
    Y20  = sqrt(15.0 / (32.0 * pi)) * st * st
    Y2p1 = sqrt(5.0 / (16.0 * pi)) * st * (1.0 + ct)
    Y2p2 = sqrt(5.0 / (64.0 * pi)) * (1.0 + ct) * (1.0 + ct)

    # exp(i*m*alpha), m = -2..2. alpha is real, so exp(-i*m*alpha) = conj(exp(i*m*alpha)).
    cis_alpha   = cis(alpha)
    cis_2alpha  = cis_alpha * cis_alpha
    cis_malpha  = conj(cis_alpha)
    cis_m2alpha = conj(cis_2alpha)

    # Transfer functions (Eq 3.5-3.6 of arXiv:2004.06503, lines 2064-2071), unrolled over
    # m = -2:2 rather than looped/tuple-indexed to keep this function scalar and allocation-free:
    #   A2m2emm(m)    = exp(-i*m*alpha) * d2m2[m] * Y2mA[m]
    #   A22emmstar(m) = exp(+i*m*alpha) * d22[m]  * conj(Y2mA[m])   (conj is a no-op: Y2mA real)
    A2m2emm_m2 = cis_2alpha  * d2m2_m2 * Y2m2
    A2m2emm_m1 = cis_alpha   * d2m2_m1 * Y2m1
    A2m2emm_0  =                d2m2_0  * Y20
    A2m2emm_p1 = cis_malpha  * d2m2_p1 * Y2p1
    A2m2emm_p2 = cis_m2alpha * d2m2_p2 * Y2p2

    A22emmstar_m2 = cis_m2alpha * d22_m2 * Y2m2
    A22emmstar_m1 = cis_malpha  * d22_m1 * Y2m1
    A22emmstar_0  =                d22_0  * Y20
    A22emmstar_p1 = cis_alpha   * d22_p1 * Y2p1
    A22emmstar_p2 = cis_2alpha  * d22_p2 * Y2p2

    polarizationSymmetry = 1.0  # see docstring: hardcoded, deliberate scope limit

    hp_sum = (A2m2emm_m2 + polarizationSymmetry * A22emmstar_m2) +
             (A2m2emm_m1 + polarizationSymmetry * A22emmstar_m1) +
             (A2m2emm_0  + polarizationSymmetry * A22emmstar_0 ) +
             (A2m2emm_p1 + polarizationSymmetry * A22emmstar_p1) +
             (A2m2emm_p2 + polarizationSymmetry * A22emmstar_p2)

    hc_sum = im * ((A2m2emm_m2 - polarizationSymmetry * A22emmstar_m2) +
                   (A2m2emm_m1 - polarizationSymmetry * A22emmstar_m1) +
                   (A2m2emm_0  - polarizationSymmetry * A22emmstar_0 ) +
                   (A2m2emm_p1 - polarizationSymmetry * A22emmstar_p1) +
                   (A2m2emm_p2 - polarizationSymmetry * A22emmstar_p2))

    # Note that gamma = -epsilon (LALSuite line 2074).
    eps_phase_hP = cis(-2.0 * epsilon) * hAST / 2.0

    hp = eps_phase_hP * hp_sum
    hc = eps_phase_hP * hc_sum

    return (hp, hc)
end
end

"""
    _XP_HAS_MODE_NORMALIZATION

Convention-conversion constant between GravitationalWaveBayes' `IMRPhenomXAS` `Ampl`/`Phi` output (normalized so
that the FACE-ON, `theta_jn = 0`, strain equals `Ampl*exp(im*Phi)` directly -- validated against
LALSuite's own face-on `hplus`) and the raw `(2,2)` mode coefficient
`IMRPhenomXPTwistUp22`'s `hAS` argument expects (the standard GW mode-decomposition convention,
`h+ - i*hx = sum_lm h_lm * (-2)Y_lm(iota, phi_c)`).

At `iota = 0`, `(-2)Y_{2,2}(0,0) = sqrt(5/(4*pi))` (a standard, textbook spin-weighted-spherical-
harmonic value) and `(-2)Y_{2,-2}(0,0) = 0`, so a raw mode coefficient `h_lm` produces a face-on
`hplus = h_lm * sqrt(5/(4*pi))`. This port's own twisting-up sum (`_xp_twist_up22`, beta=0 limit,
`theta=0`) independently confirms the same relation with its own extra factor of `1/2` from
`eps_phase_hP`: `Y2m2(0)=0`, `Y22(0) = sqrt(5/(64*pi))*4 = sqrt(5/(4*pi))`, so `|hp| = |hAS_mode| *
sqrt(5/(4*pi)) / 2` exactly.

**This constant is REQUIRED, not cosmetic**: without it, feeding `IMRPhenomXAS`'s `Ampl*exp(im*Phi)`
directly as `hAS` (i.e. implicitly assuming the two conventions coincide) reproduces a face-on
`|hp|` that is low by exactly this factor (confirmed numerically: `0.31539...` instead of `1.0`
for unit input) -- a real bug caught by comparing against the independently-known
`pfac^2+cfac^2` aligned-spin power (NOT caught by an earlier, weaker self-consistency test that
derived its own expected value from the same un-normalized formula rather than from an externally
anchored convention).
"""
const _XP_HAS_MODE_NORMALIZATION = 2.0 / sqrt(5.0 / (4.0 * pi))

"""
    _xp_apply_zeta_rotation(hp, hc, zeta_polarization)

Scalar, non-allocating polarization-frame rotation by `2*zeta_polarization`, applied to
`_xp_twist_up22`'s `(hp, hc)` output. Ported from `IMRPhenomXPGenerateFD`'s own post-processing
step (LALSimIMRPhenomX.c lines 2468-2492): `hp_new = cos(2*zeta)*hp + sin(2*zeta)*hc`, `hc_new =
cos(2*zeta)*hc - sin(2*zeta)*hp`. See `_xp_precession_setup`'s docstring (`zeta_polarization`
field) for what this angle is and why it's needed -- it is NOT generically negligible; an earlier
version of this module incorrectly assumed it was safely skippable for dominant-mode-only
twisting-up.

# Aligned-spin (`hc == -i*hp`) invariant -- proven algebraically, confirmed against real LALSuite
For a purely aligned-spin (non-precessing, `Sperp=0`, `cos_beta=1` exactly) signal, `hc == -i*hp`
identically BEFORE this rotation (see `test_twisting_up.jl`'s "Aligned-spin limit" testset). This
rotation ALWAYS preserves that exact ratio, for any `zeta_polarization` value -- substituting
`hc = -i*hp` gives `hp_new = hp*(cos(2z) - i*sin(2z)) = hp*exp(-2i*z)` and `hc_new = -i*hp*exp
(-2i*z) = -i*hp_new`, i.e. `(1, -i)` is an eigenvector of this rotation with eigenvalue
`exp(-2i*zeta)` (the standard fact that circular-polarization states are eigenstates of a
polarization-frame rotation). So an aligned-spin waveform is NOT "mixed" by this rotation in any
observable sense -- it only picks up an extra overall constant phase `exp(-2i*zeta)`, which does
NOT affect `|hp|`/`|hc|` or the gauge-invariant `dPhi/df` used elsewhere in this codebase's
LALSuite validation. Confirmed empirically too, not just algebraically: a diagnostic against
real LALSuite `IMRPhenomXP` output (`inclination=0`, aligned spins, `m1=30,m2=20,chi1=0.3,
chi2=-0.2,phiRef=0.7`) found `|LAL hplus|/|GravitationalWaveBayes Ampl*exp(iPhi)| = 1` to machine precision and
`LAL_hplus/h == LAL_hcross/(-i*h)` exactly at every frequency checked -- i.e. real LALSuite
genuinely applies this same extra phase at face-on/aligned-spin, this is not a port artifact.
"""
@inline function _xp_apply_zeta_rotation(hp, hc, zeta_polarization)
    T = real(promote_type(typeof(hp), typeof(hc), typeof(zeta_polarization)))
@Tlit T begin
    cosPolFac = cos(2.0 * zeta_polarization)
    sinPolFac = sin(2.0 * zeta_polarization)
    hp_new = cosPolFac * hp + sinPolFac * hc
    hc_new = cosPolFac * hc - sinPolFac * hp
    return (hp_new, hc_new)
end
end

"""
    twist_up22(model::IMRPhenomXP, f, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z,
               f_ref, inclination, phiRef, dL, phiRefOrbital = phiRef; kwargs...)

End-to-end IMRPhenomXP dominant-mode-precession waveform: builds the precession-frame setup
([`_xp_precession_setup`](@ref)), evaluates `IMRPhenomXAS`'s aligned-spin `(2,2)` mode internally to
get `hAS = Ampl(...) * exp(im*Phi(...))`, and twists it up via [`_xp_twist_up22`](@ref) to return
`(hp, hc)`. `f` may be a scalar (returns a scalar `(hp, hc)` tuple, non-allocating beyond `Ampl`/
`Phi`'s own scalar path) or an `AbstractVector` (returns same-length `Vector{Complex{T}}`s for
`hp`/`hc`) -- matching `Ampl`/`Phi`'s own scalar-vs-array convention. See [`IMRPhenomXP`](@ref)'s
docstring for why this returns `(hp, hc)` directly rather than participating in the `Ampl`/`Phi`
two-function contract.

`m1`, `m2` are component masses in solar masses (`m1 >= m2`, this package's/file's mass-ordering
convention -- see `_xp_precession_setup`'s docstring); `chi1*`/`chi2*` are dimensionless spin
components in the L-aligned (source) frame at `f_ref`; `inclination` is the angle between the
orbital angular momentum `L` and the line of sight at `f_ref`; `dL` is the luminosity distance in
Gpc (same convention as `Ampl(::IMRPhenomXAS, ...)`).

# `phiRef` vs `phiRefOrbital`
`phiRef` feeds `_xp_precession_setup`'s line-of-sight-vector construction (`Nx`/`Ny`, hence
`thetaJN`) -- see that function's docstring for why it needs `phiRef` explicitly. `phiRefOrbital`
is passed through to `Phi(::IMRPhenomXAS, ...)`'s own `phiRef` keyword, which is a simple external
constant-phase offset on the aligned-spin `(2,2)` phase (see that function's docstring: "no
internal time-of-coalescence alignment is performed"). In LALSuite's own construction these
coincide (a single `phiRef` value feeds both roles, see `LALSimIMRPhenomX_precession.c` lines
865-866 vs the phase alignment in `LALSimIMRPhenomX.c`), so `phiRefOrbital` defaults to `phiRef`
here for physical consistency by default; it is exposed separately because this package's
`Phi(::IMRPhenomXAS, ...)` already has its own independent, decoupled `phiRef`/`t0`
external-alignment convention (unlike LALSuite's tightly-coupled internal phase construction), and
overriding it independently is occasionally useful (e.g. re-aligning `hAS`'s phase without moving
the precession-frame geometry).

# Keyword arguments
- `fcutPar = 0.3`: forwarded to both `Ampl(::IMRPhenomXAS, ...)` and `Phi(::IMRPhenomXAS, ...)`.
- `GMsun_over_c3`, `GMsun_over_c2_Gpc`: same convention as `Ampl`/`Phi(::IMRPhenomXAS, ...)`.

No parameter-domain assertions run here -- this is the hot path; see
`validate_phenomxp_precession_params`/`validate_phenomxas_params` for opt-in host-side checks.
"""
function twist_up22(model::IMRPhenomXP,
    f,
    m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z,
    f_ref, inclination, phiRef, dL, phiRefOrbital = phiRef;
    fcutPar = 0.3,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    setup = _xp_precession_setup(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z,
        f_ref, inclination, phiRef; GMsun_over_c3 = GMsun_over_c3)
    T = setup.T
    m1T = T(m1)
    m2T = T(m2)
    Mtot = m1T + m2T
@Tlit T begin
    mc = setup.eta^(0.6) * Mtot   # eta^(3/5) * Mtot, the standard chirp-mass identity
end
    chi1zT = T(chi1z)
    chi2zT = T(chi2z)
    dLT = T(dL)
    phiRefOrbitalT = T(phiRefOrbital)
    GMsun_over_c3T = T(GMsun_over_c3)
    piT = T(pi)

    # LALSuite's default `PhenomXPFinalSpinMod = 0`: the precessing final spin (not the aligned-spin
    # one) sets the ringdown/damping frequency for the co-precessing `(2,2)` mode -- see
    # `_xas_precessing_final_spin`'s docstring. This is the single largest source of disagreement
    # with stock LALSuite for `IMRPhenomXP` (see the docstring of `IMRPhenomXPHM`).
    final_spin = T(setup.final_spin)
    ampl = Ampl(IMRPhenomXAS(), f, mc, setup.eta, chi1zT, chi2zT, dLT;
        fcutPar = fcutPar, GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
        final_spin_override = final_spin)
    phi = Phi(IMRPhenomXAS(), f, mc, setup.eta, chi1zT, chi2zT;
        fcutPar = fcutPar, phiRef = phiRefOrbitalT, GMsun_over_c3 = GMsun_over_c3,
        final_spin_override = final_spin)

    if f isa AbstractArray
        hp = similar(f, Complex{T})
        hc = similar(f, Complex{T})
        @inbounds for i in eachindex(f)
            Mf = Mtot * GMsun_over_c3T * T(f[i])
            v = cbrt(piT * Mf)
            hAS = ampl[i] * cis(phi[i]) * T(_XP_HAS_MODE_NORMALIZATION)
            hp_i, hc_i = _xp_twist_up22(v, setup, hAS)
            hp[i], hc[i] = _xp_apply_zeta_rotation(hp_i, hc_i, setup.zeta_polarization)
        end
        return hp, hc
    else
        Mf = Mtot * GMsun_over_c3T * T(f)
        v = cbrt(piT * Mf)
        hAS = ampl * cis(phi) * T(_XP_HAS_MODE_NORMALIZATION)
        hp_s, hc_s = _xp_twist_up22(v, setup, hAS)
        return _xp_apply_zeta_rotation(hp_s, hc_s, setup.zeta_polarization)
    end
end
