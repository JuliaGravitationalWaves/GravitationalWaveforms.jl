"""
    IMRPhenomXHM assembly

Ties together `IMRPhenomXHM_qnm.jl`, `IMRPhenomXHM_inspiral.jl`,
`IMRPhenomXHM_intermediate.jl` and `IMRPhenomXHM_ringdown.jl` into real, complete
`h_lm(f)` per mode for `(l,m) = (2,2),(2,1),(3,3),(3,2),(4,4)`, then sums them (spin-weighted
spherical harmonics, aligned-spin/non-precessing only -- no twisting-up here) into `h+(f)`/`hx(f)`. Ported from
`LALSimIMRPhenomXHM.c` (`IMRPhenomXHMGenerateFDOneMode`, `IMRPhenomXHM_MultiMode`,
`IMRPhenomXHMFDAddMode`), `LALSimIMRPhenomXHM_internals.c` (`IMRPhenomXHM_Get{Amplitude,Phase}
Coefficients`, `GetSpheroidalCoefficients`, `SpheroidalToSpherical`), a small but load-bearing
slice of `LALSimIMRPhenomX_internals.c` (`IMRPhenomX_Phase_22_ConnectionCoefficients`,
`IMRPhenomX_TimeShift_22`, `IMRPhenomX_Phase_22`, `IMRPhenomX_dPhase_22`) that none of Tasks
4a-4d needed but this one genuinely does (see "The (2,2) mode's OWN absolute-phase alignment"
below), and `LALSimIMRPhenomXUtilities.c` (`XLALSimIMRPhenomXLinb`, `XLALSimIMRPhenomXPsi4ToStrain`,
`XLALSimIMRPhenomXRingdownAmplitude22AnsatzAnalytical`) / `SphericalHarmonics.c`
(`XLALSpinWeightedSphericalHarmonic`, s=-2, l=2,3,4).

# `Ampl`/`Phi` vs a direct `h_lm(f)`/`(hp,hc)` contract -- the design decision
`IMRPhenomXHM` is fundamentally a SUM over 5 complex mode functions with independent amplitude
AND phase shapes -- there is no single frequency-dependent `(Ampl, Phi)` pair that describes
`h+`/`hx` the way there is for a dominant-mode-only model (`IMRPhenomD`/`IMRPhenomXAS`), or even
for `IMRPhenomXP`'s twisted-up dominant mode (which still has a well-defined single complex
`h+`/`hx` per frequency, just not expressible as `Ampl*exp(iPhi)` for a scalar `Phi`). Following
`IMRPhenomXP`'s own precedent in this package (`IMRPhenomXPrecession.jl`'s `IMRPhenomXP` marker
struct + `twist_up22` returning `(hp, hc)` directly, explicitly declining the `Ampl`/`Phi`
contract -- see that struct's docstring for the reasoning, which applies here even more strongly
since there isn't even a single dominant mode to hang a scalar phase off of): `IMRPhenomXHM`
(marker struct, this file) does NOT participate in `Ampl`/`Phi`. Two public entry points instead:
[`h_lm`](@ref) (single mode, complex, for per-mode validation/inspection -- the natural analogue
of what `SimIMRPhenomXHMGenerateFDOneMode` returns) and [`hphc`](@ref) (the actual deliverable,
`(hp, hc)` summed over all 5 modes). This is a deliberate, considered choice, not a shortcut: a
`Phi(::IMRPhenomXHM, ...)` that returned, say, "the (2,2) mode's phase" would silently misrepresent
the physics for any caller doing PE with real sub-dominant-mode content (exactly the GW190412/
GW190814 use case this whole phase exists for).

# Working convention: every per-mode function here returns `h_{l,-m}(f)`, NOT `h_{l,+m}(f)`
Confirmed directly from `XLALSimIMRPhenomXHMGenerateFDOneMode` (`LALSimIMRPhenomXHM.c` lines
287-331): the internal generator (`IMRPhenomXHMGenerateFDOneMode`, always called with a POSITIVE
`emm`, matching the `(l,m)` convention throughout this package) produces `h_{l,-m}`
directly; the wrapper only applies `(-1)^l * conj(...)` when the CALLER explicitly wants `+m`. This
package's own `h_lm(model, l, m, ...)` (below) mirrors that exactly: pass positive `m`, get
`h_{l,-m}` back. [`hphc`](@ref) reconstructs both `+m` and `-m` content from this single value via
the standard aligned-spin symmetry `h_{l,+m} = (-1)^l * conj(h_{l,-m})`, exactly matching
`IMRPhenomXHMFDAddMode`'s own `sym=1` branch (`LALSimIMRPhenomXHM.c` lines 2306-2327) -- see that
function's docstring analogue, [`_xhm_add_mode!`](@ref)`, for the actual formula transcribed.

# The (2,2) mode's OWN absolute-phase alignment -- a real finding, not present anywhere else in
# this package, and load-bearing for correct RELATIVE inter-mode phasing
`IMRPhenomXAS.jl`'s own `Phi(::IMRPhenomXAS, ...)` deliberately performs NO internal
time-of-coalescence alignment (its own docstring: "no internal time-of-coalescence alignment is
performed" -- `t0`/`phiRef` are simple external linear/constant offsets the caller controls). That
was always fine on its own (earlier validation used gauge-invariant `dPhi/df`, which is
insensitive to any constant+linear-in-f offset) -- but `IMRPhenomXHM_GetPhaseCoefficients`'s
`deltaphiLM` construction (`_internals.c` lines 2334-2369, the frequency-INDEPENDENT phase offset
every higher mode needs to be correctly aligned relative to the `(2,2)` mode) explicitly depends on
the *(2,2) mode's own* `linb`/`tshift`/`phifRef`-style internal alignment
(`IMRPhenomX_TimeShift_22`, `IMRPhenomX_Phase_22_ConnectionCoefficients`, and the reference-phase
constant `phiref22`) -- none of which `IMRPhenomXAS.jl` exposes or computes. Skipping this (e.g. by
naively using `Phi(::IMRPhenomXAS,...)`'s own `t0=phiRef=0` convention for the `(2,2)` mode inside
the multi-mode sum) would silently give every higher mode the WRONG relative phase against the
`(2,2)` mode -- invisible in a per-mode `dPhi/df` check (the validation metric of the per-region files, which
only sees each mode in isolation and is blind to a mode-dependent constant offset) but real and
visible the moment multiple modes are summed into `h+`/`hx`.
**Ported here** (not present in any other file in this package): [`_xhm22_phase_raw`](@ref)/
[`_xhm22_dphase_raw`](@ref) (reuse `IMRPhenomXAS.jl`'s already-validated `_xas_phi_ins_val`/
`_xas_phi_int_val`+`alpha1`/`_xas_phi_mrd_val`+`beta1` pieces directly -- these ARE
`IMRPhenomX_Phase_22`/`IMRPhenomX_dPhase_22`, just not previously exposed as standalone reusable
functions), [`_xhm_linb`](@ref)/[`_xhm_psi4_to_strain`](@ref) (`XLALSimIMRPhenomXLinb`/
`XLALSimIMRPhenomXPsi4ToStrain`, `LALSimIMRPhenomXUtilities.c` lines 379-415, plain closed-form
fits), [`_xhm_timeshift22`](@ref) (`IMRPhenomX_TimeShift_22`), [`_xhm_phiref22`](@ref)
(`phiref22`'s own formula, `_internals.c` line 2364/2611 -- identical in both the mixing-on and
mixing-off code paths). Confirmed algebraically consistent with this package's own
`_XP_HAS_MODE_NORMALIZATION` constant (`IMRPhenomXPrecession.jl`) -- see [`_xhm_amp0`](@ref)'s
docstring for the cross-check, a real, independent internal-consistency confirmation that this
port's normalization chain is correct, not just self-consistent.

# `(3,2)` mode-mixing: fully assembled here (not left as raw fits, unlike `IMRPhenomXHM_ringdown.jl`)
`IMRPhenomXHM_ringdown.jl`'s module docstring explicitly identified the two missing pieces that
blocked a standalone `(3,2)` ansatz there: the `(2,2)`-mode `IMRPhenomX_TimeShift_22`/
`IMRPhenomX_Phase_22_ConnectionCoefficients` machinery (now ported above, this file) and the
`fAmpMatchIM`/mode-mixing-aware boundary bootstrapping (`GetSpheroidalCoefficients`,
`SpheroidalToSpherical`, ported below as [`_xhm_spheroidal32_setup`](@ref)/[`_xhm_s2s`](@ref)).
Every genuine judgment call made while porting this ([`_xhm_rd_amp32_ansatz`](@ref)'s
`RDAux`-polynomial-vs-Lorentzian-vs-falloff three-way split, the intermediate region's
finite-difference-based boundary extraction at `fcutRD` matching `_internals.c`'s own `fstep=1e-7`
FD scheme lines 2183-2230 exactly, the 6-term `(c0,c1,c2,c3,c4,cL)` intermediate phase ansatz
`(3,2)` alone needs) is documented at its own function's docstring below, not summarized here.

# `IMRPhenomXHM_setup_mode_array`'s default modes -- exactly this package's own 5-mode scope
`(2,2),(2,1),(3,3),(3,2),(4,4)` (each with its `-m` partner reconstructed via symmetry) is
LALSuite's own DEFAULT active-mode set (`LALSimIMRPhenomXHM.c` lines 140-151) when no custom
`ModeArray` is supplied -- this port
hardcodes that same 5-mode set (no `ModeArray`/custom-mode-selection machinery); a caller wanting a
different mode subset is out of scope here.

# Not ported: `RingdownAmpVeto`/negative-amplitude clamp/PNR-coprecessing deviations
As in the other XHM files, code confirmed dead under the default release is not ported:
`IMRPhenomXHM_Ringdown_Amplitude_Veto` (dead, see `IMRPhenomXHM_ringdown.jl`), the `if(Amp<0 && ReleaseVersion!=122019)
Amp=FALSE_ZERO` clamp (`_internals.c` lines 2740/2769 -- a defensive numerical-noise floor, not
exercised by any validated configuration; `_xhm_mode_amplitude`/`_xhm_s2s`
below do not replicate it), and every `PNR_DEV_PARAMETER`/`IMRPhenomXPNRUseTunedCoprec*` deviation
(always zero under this package's scope, as in `IMRPhenomXHM_ringdown.jl`).
"""

# ================================================================================================
# (2,2) mode's own LALSuite-matching internal phase-alignment machinery (see module docstring's
# "The (2,2) mode's OWN absolute-phase alignment" section for why this is needed at all).
# ================================================================================================

"""
    _xhm_f_of_Mf(coeffs22, Mf)

Inverts `IMRPhenomXAS.jl`'s own `fgrid = M*GMsun_over_c3*f` scaling: given a dimensionless `Mf`,
returns the physical frequency `f` (Hz) that `_xas_phi_scalar`/`_xas_ampl_scalar` would need to
reproduce it, so this file's `Mf`-native functions can reuse those already-validated scalar
evaluators directly instead of re-deriving their branch structure.
"""
@inline _xhm_f_of_Mf(coeffs22, Mf) = Mf / (coeffs22.M * coeffs22.GMsun_over_c3)

"""
    _xhm22_phase_raw(coeffs22, Mf)

`IMRPhenomX_Phase_22(Mf)` (`LALSimIMRPhenomX_internals.c` lines 2654-2676): the RAW `(2,2)`-mode
phase (inspiral/intermediate/ringdown ansatz plus C1-continuity corrections, but NOT yet divided by
`eta` and with no `t0`/`phiRef`/`timeshift`/`phifRef` alignment applied at all) at dimensionless
frequency `Mf`. Obtained by calling `_xas_phi_scalar` with `t0=phiRef=0` (`coeffs22`'s own default,
see [`_xhm_system_setup`](@ref)) at the physical frequency that maps to this `Mf`, then multiplying
back by `eta` to undo `_xas_phi_scalar`'s own `/coeffs.eta` division -- i.e. this reuses
`IMRPhenomXAS.jl`'s already-LALSuite-validated inspiral/intermediate/ringdown-with-C1-continuity
construction directly rather than re-deriving it, avoiding drift risk (same rationale
`IMRPhenomXHM_inspiral.jl`'s own docstring gives for reusing `_xas_phi_coeffs.phi_ins`).
"""
@inline _xhm22_phase_raw(coeffs22, Mf) = coeffs22.eta * _xas_phi_scalar(coeffs22, _xhm_f_of_Mf(coeffs22, Mf))

"""
    _xhm22_dphase_raw(coeffs22, Mf)

`IMRPhenomX_dPhase_22(Mf)` (`LALSimIMRPhenomX_internals.c` lines 2683-2705, not shown in full here
but structurally identical to `IMRPhenomX_Phase_22`'s own three-region `ifelse`): the RAW
`(2,2)`-mode phase DERIVATIVE (`d(eta*Phase)/dMf`, i.e. not yet divided by `eta`), reusing
`IMRPhenomXAS.jl`'s own `_xas_phi_ins_deriv`/`_xas_phi_int_deriv`+`alpha1`/`_xas_phi_mrd_deriv`+
`beta1` pieces -- the exact same derivative building blocks `_xas_phi_coeffs` itself uses internally
to derive `alpha0`/`alpha1`/`beta0`/`beta1` (see that function's own `dphi_ins_f1`/`dphi_int_f1_raw`
etc. local variables), just re-exposed at a general `Mf` rather than only at the two join points.
Algebraically re-derived from already-validated pieces, not independently re-checked against a
fresh finite-difference here (this file's own unit tests cross-check it against a direct FD of
[`_xhm22_phase_raw`](@ref) instead, matching this package's established convention of pairing
analytic derivatives with an FD consistency test).
"""
@inline function _xhm22_dphase_raw(coeffs22, Mf)
    pi_ = coeffs22.phi_ins
    ifelse(
        Mf < coeffs22.f1,
        _xas_phi_ins_deriv(Mf, pi_.phi0, pi_.phi1, pi_.phi2, pi_.phi3, pi_.phi4, pi_.phi5, pi_.phi5L,
            pi_.phi6, pi_.phi6L, pi_.phi7, pi_.phi8, pi_.phi8L, pi_.sigma1, pi_.sigma2, pi_.sigma3,
            pi_.sigma4, pi_.phiN),
        ifelse(
            Mf < coeffs22.f2,
            _xas_phi_int_deriv(Mf, coeffs22.phi_int.b0, coeffs22.phi_int.b1, coeffs22.phi_int.b2,
                coeffs22.phi_int.b3, coeffs22.phi_int.b4, coeffs22.phi_mrd.cL, coeffs22.fdamp,
                coeffs22.fRD) + coeffs22.phi_int.alpha1,
            _xas_phi_mrd_deriv(Mf, coeffs22.phi_mrd.c0, coeffs22.phi_mrd.c1, coeffs22.phi_mrd.c2,
                coeffs22.phi_mrd.c4, coeffs22.phi_mrd.cL, coeffs22.fdamp, coeffs22.fRD) +
            coeffs22.phi_mrd.beta1,
        ),
    )
end

"""
    _xhm_linb(eta, S, dchi, delta)

`XLALSimIMRPhenomXLinb` (`LALSimIMRPhenomXUtilities.c` lines 396-415): a plain closed-form
parameter-space fit (no version switching, only one branch exists in the C source) for the
`(2,2)`-mode's own internal time-alignment constant. `S` here is `pWF->STotR` (`chiPNHat`... no,
`STotR`, confirmed by the C call site `XLALSimIMRPhenomXLinb(pWF->eta, pWF->STotR, pWF->dchi,
pWF->delta)`), `delta = sqrt(1-4eta)`.
"""
@inline function _xhm_linb(eta, S, dchi, delta)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi), typeof(delta))
    @Tlit T begin
        eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta; eta5 = eta3 * eta2; eta6 = eta4 * eta2
        S2 = S * S; S3 = S2 * S; S4 = S3 * S
        noSpin = 3155.1635543201924 + 1257.9949740608242 * eta - 32243.28428870599 * eta2 +
                 347213.65466875216 * eta3 - 1.9223851649491738e6 * eta4 + 5.3035911346921865e6 * eta5 -
                 5.789128656876938e6 * eta6
        eqSpin = (-24.181508118588667 + 115.49264174560281 * eta - 380.19778216022763 * eta2) * S +
                 (24.72585609641552 - 328.3762360751952 * eta + 725.6024119989094 * eta2) * S2 +
                 (23.404604124552 - 646.3410199799737 * eta + 1941.8836639529036 * eta2) * S3 +
                 (-12.814828278938885 - 325.92980012408367 * eta + 1320.102640190539 * eta2) * S4
        uneqSpin = -148.17317525117338 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

"""
    _xhm_psi4_to_strain(eta, S, dchi)

`XLALSimIMRPhenomXPsi4ToStrain` (`LALSimIMRPhenomXUtilities.c` lines 379-393): fit for the
time-difference between psi4's peak and strain's peak, used only inside [`_xhm_timeshift22`](@ref).
`S` is `pWF->STotR` at this call site too.
"""
@inline function _xhm_psi4_to_strain(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta
        S2 = S * S; S3 = S2 * S; S4 = S3 * S
        noSpin = 13.39320482758057 - 175.42481512989315 * eta + 2097.425116152503 * eta2 -
                 9862.84178637907 * eta3 + 16026.897939722587 * eta4
        eqSpin = (4.7895602776763 - 163.04871764530466 * eta + 609.5575850476959 * eta2) * S +
                 (1.3934428041390161 - 97.51812681228478 * eta + 376.9200932531847 * eta2) * S2 +
                 (15.649521097877374 + 137.33317057388916 * eta - 755.9566456906406 * eta2) * S3 +
                 (13.097315867845788 + 149.30405703643288 * eta - 764.5242164872267 * eta2) * S4
        uneqSpin = 105.37711654943146 * dchi * sqrt(1.0 - 4.0 * eta) * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

"""
    _xhm_timeshift22(coeffs22, eta, StotR, dchi, delta)

`IMRPhenomX_TimeShift_22` (`LALSimIMRPhenomX_internals.c` lines 2624-2645): `linb - dphi22Ref -
2*pi*(500+psi4tostrain)`, where `dphi22Ref = (1/eta)*dPhase_22(fRD22 - fdamp22)`. The "500" is
LALSuite's own hybrid-waveform time-alignment convention constant (psi4 peaks 500M before the end
of their NR-hybrid calibration waveforms); ported verbatim, not re-derived.
"""
@inline function _xhm_timeshift22(coeffs22, eta, StotR, dchi, delta)
    T = coeffs22.T
    frefFit = coeffs22.fRD - coeffs22.fdamp
    dphi22Ref = _xhm22_dphase_raw(coeffs22, frefFit) / eta
    linb = _xhm_linb(eta, StotR, dchi, delta)
    psi4tostrain = _xhm_psi4_to_strain(eta, StotR, dchi)
    @Tlit T begin
        return linb - dphi22Ref - 2.0 * pi * (500.0 + psi4tostrain)
    end
end

"""
    _xhm_phiref22(coeffs22, timeshift, MfRef, phi0)

`pWFHM->phiref22` (identical formula in both the `MixingOn==0` branch, `_internals.c` line 2364,
and the `MixingOn==1` branch, line 2611): `-(  (1/eta)*Phase_22(MfRef) + timeshift*MfRef ) + 2*phi0
+ pi/4`, with `phaseshift` (a `MixingOn==1`-only, antisymmetric-waveform-adjacent quantity this
port's scope never needs -- see `IMRPhenomXPrecession.jl`'s own `zeta_polarization`/antisymmetric
scope note for the analogous precedent) hardcoded to `0`. `phi0` is `phiRef` (the user's reference
orbital phase), `MfRef` is the dimensionless reference frequency.
"""
@inline function _xhm_phiref22(coeffs22, timeshift, MfRef, phi0)
    T = coeffs22.T
    raw = _xhm22_phase_raw(coeffs22, MfRef) / coeffs22.eta
    @Tlit T begin
        return -(raw + timeshift * MfRef) + 2.0 * phi0 + pi / 4.0
    end
end

"""
    _xhm22_phase_full(coeffs22, timeshift, phiref22, Mf)

`(1/eta)*IMRPhenomX_Phase_22(Mf) + timeshift*Mf + phiref22` -- the FULL LALSuite-matching `(2,2)`
mode phase (matches `IMRPhenomX_Phase_22`'s use at `LALSimIMRPhenomXHM_internals.c` line 2657
(`SpheroidalToSpherical`) and the `deltaphiLM` formula, both of which add `pWFHM->phaseshift` too --
hardcoded `0` here, see [`_xhm_phiref22`](@ref)'s docstring). NOT the same as
`Phi(::IMRPhenomXAS,...)`'s own `t0=phiRef=0` convention -- see module docstring.
"""
@inline _xhm22_phase_full(coeffs22, timeshift, phiref22, Mf) =
    _xhm22_phase_raw(coeffs22, Mf) / coeffs22.eta + timeshift * Mf + phiref22

# ================================================================================================
# System-wide setup: everything derived once per waveform call, shared by every mode (the (2,2)
# mode's own coefficients/alignment, final spin/radiated energy, spin combinations).
# ================================================================================================

"""
    _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phiRef; GMsun_over_c3, GMsun_over_c2_Gpc)

Everything every per-mode function in this file needs that does NOT depend on `(l,m)`: the `(2,2)`
mode's own phase coefficients (`_xas_phi_coeffs`, `t0=phiRef=0`) and amplitude coefficients
(`_xas_ampl_coeffs`, real `dL`), spin combinations (`Seta`/`StotR`/`chiPNHat`/`dchi`, the SAME
combinations `IMRPhenomXHM_inspiral.jl`/`_intermediate.jl`/`_ringdown.jl` already use), final spin
`a`/radiated energy `Erad` (`_xas_a_Erad`, needed for [`_xhm_fring_fdamp`](@ref) and this file's own
`(3,2)` mixing-coefficient fits), and this file's own `timeshift`/`phiref22`/`MfRef`/`phi0`
alignment chain (see module docstring).

`final_spin_override = NaN`: when finite, overrides the ringdown/damping-frequency final spin `a`
used everywhere in this file (see [`_xas_precessing_final_spin`](@ref)'s docstring) -- forwarded to
`coeffs22`/`amplcoeffs22`'s own `final_spin_override` too, so the `(2,2)` mode's ringdown is
consistent with every other mode's. Used by `IMRPhenomXPHM`; leave at the default `NaN` for plain
`IMRPhenomXHM` (recovers the pre-existing behavior exactly).
"""
@inline function _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phiRef;
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
    final_spin_override = NaN,
)
    T = promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2), typeof(dL), typeof(f_ref), typeof(phiRef))
    coeffs22 = _xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2;
        GMsun_over_c3 = GMsun_over_c3, final_spin_override = final_spin_override)
    amplcoeffs22 = _xas_ampl_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2, dL;
        GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
        final_spin_override = final_spin_override)

    @Tlit T begin
        Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
        mm1 = 0.5 * (1.0 + Seta)
        mm2 = 0.5 * (1.0 - Seta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
        StotR = (mm1 * mm1 * chi1 + mm2 * mm2 * chi2) / (mm1 * mm1 + mm2 * mm2)
        dchi = chi1 - chi2
    end

    timeshift = _xhm_timeshift22(coeffs22, eta, StotR, dchi, Seta)
    MfRef = T(f_ref) * coeffs22.M * coeffs22.GMsun_over_c3
    phiref22 = _xhm_phiref22(coeffs22, timeshift, MfRef, T(phiRef))
    a, Erad = _xas_a_Erad(eta, Seta, dchi, StotR)
    # Every OTHER mode's ringdown (`_xhm_fring_fdamp`, the `(3,2)`-mixing QNM fits below) reads
    # `sys.a` too -- unlike `_xas_phi_coeffs`/`_xas_ampl_coeffs` there's no separate `fISCO`/`Erad`
    # consumer of the aligned value inside THIS file, so it's safe to override `a` itself here
    # rather than keeping two copies (see `_xas_precessing_final_spin`'s docstring for why fISCO
    # specifically must stay on the aligned value -- that happens inside `coeffs22`/`amplcoeffs22`
    # above, which received `final_spin_override` directly and keep their own internal `a` for it).
    a = ifelse(isnan(T(final_spin_override)), a, T(final_spin_override))

    return (T = T, coeffs22 = coeffs22, amplcoeffs22 = amplcoeffs22, eta = eta, chi1 = chi1, chi2 = chi2,
        Seta = Seta, StotR = StotR, S = S, dchi = dchi, timeshift = timeshift, phiref22 = phiref22,
        MfRef = MfRef, phi0 = T(phiRef), a = a, Erad = Erad, dL = T(dL))
end

"""
    _xhm_amp0(sys)

The bare mass/distance prefactor every higher mode's raw `h_lm` amplitude is multiplied by
(`pWF->amp0` in `LALSimIMRPhenomX_internals.c` line 609, `M^2 * GMsun_over_c2 * GMsun_over_c3 /
distance`) -- NOT the same as `IMRPhenomXAS.jl`'s own `Overallamp` (which additionally includes the
`2*sqrt(5/(64*pi))` `(-2)Y_{2,2}`-related normalization baked in for its own "hplus at face-on"
convention). Computed here as `sys.amplcoeffs22.Overallamp * _XP_HAS_MODE_NORMALIZATION` rather than
re-deriving the bare mass/distance formula independently.

**Cross-check (a real, independent internal-consistency confirmation, not just algebra)**: comparing
`LALSimIMRPhenomX.c`'s own `Amp0 = pWF->amp0 * pWF->ampNorm` (used verbatim to build the raw
`(2,2)`-mode-only `htilde22` inside `XLALSimIMRPhenomXASGenerateFD`, `LALSimIMRPhenomX.c` line 782,
which `IMRPhenomXHMGenerateFDOneMode` reuses AS-IS for `h_{2,-2}` when `ell==2`, see
`LALSimIMRPhenomXHM.c` lines 306-320) against `IMRPhenomXAS.jl`'s own `Overallamp*amp0` product
(`_xas_ampl_coeffs`'s return) shows the two prefactors differ by EXACTLY `2*sqrt(5/(64*pi))` --
i.e. `_xas_ampl_coeffs`'s `Overallamp = 2*sqrt(5/(64*pi)) * pWF->amp0` (the `ampNorm`/`sqrt(2*eta/3)`
factor is identical in both and cancels). And `2*sqrt(5/(64*pi)) = 1/_XP_HAS_MODE_NORMALIZATION`
exactly (`_XP_HAS_MODE_NORMALIZATION = 2/sqrt(5/(4*pi))`, and `sqrt(5/(4*pi)) = 4*sqrt(5/(64*pi))`
-- confirmed numerically too: `2*sqrt(5/(64*pi)) ~ 0.31539`, `1/0.31539 ~ 3.1706 =
_XP_HAS_MODE_NORMALIZATION`). So `pWF->amp0*ampNorm = Overallamp/(1/_XP_HAS_MODE_NORMALIZATION) =
Overallamp * _XP_HAS_MODE_NORMALIZATION` -- exactly this function's formula, derived here via TWO
independent routes (direct bare-constant arithmetic, and reuse of the already-LALSuite-validated
`_XP_HAS_MODE_NORMALIZATION` constant from `IMRPhenomXPrecession.jl`'s own, separately-validated,
`hAS` normalization fix) landing on the identical answer.
"""
@inline _xhm_amp0(sys) = sys.amplcoeffs22.Overallamp * sys.T(_XP_HAS_MODE_NORMALIZATION)

# ================================================================================================
# Non-mixing modes (2,1),(3,3),(4,4): per-mode setup + piecewise amplitude/phase assembly.
# Ported from `IMRPhenomXHM_Amplitude_noModeMixing`/`_Phase_noModeMixing`
# (`LALSimIMRPhenomXHM_internals.c` lines 2707-2826) and the C1INSP/CINSP/C1RD/CRD gluing
# (`IMRPhenomXHM_GetPhaseCoefficients`, lines 2323-2331, `MixingOn==0` branch) plus `deltaphiLM`
# (lines 2367-2369 and 2400-2406 for the `(2,1)`-specific `LAL_PI` sign correction).
# ================================================================================================

"""
    _xhm_pn21_ampsign(ff, eta, chi1, chi2)

`IMRPhenomXHM_PN21AmpSign` (`_internals.c` lines 1937-1949): sign of a truncated PN expansion of the
`(2,1)` mode's real amplitude at `ff` (always called with `ff=0.008`, a fixed low frequency). Its
SIGN (not magnitude) feeds a `+pi` correction to `(2,1)`'s `deltaphiLM` -- LALSuite constructs the
`(2,1)` amplitude ansatz to always be positive by construction, so this fit exists purely to track
where the TRUE PN amplitude's sign flips across parameter space and compensate with a phase shift.
"""
@inline function _xhm_pn21_ampsign(ff, eta, chi1, chi2)
    T = promote_type(typeof(ff), typeof(eta), typeof(chi1), typeof(chi2))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        return (-16.0 * delta * eta * ff * pi^1.5) / (3.0 * sqrt(5.0)) +
               (4.0 * (2.0^(1.0 / 3.0)) * (chi1 - chi2 + delta * (chi1 + chi2)) * eta * ff^(4.0 / 3.0) * pi^(11.0 / 6.0)) / sqrt(5.0) +
               (2.0 * (2.0^(2.0 / 3.0)) * eta * (306.0 * delta - 360.0 * delta * eta) * ff^(5.0 / 3.0) * pi^(13.0 / 6.0)) / (189.0 * sqrt(5.0))
    end
end

"""
    _xhm_deltaphilm(l, m, eta, chi1, chi2, coeffs22, timeshift, phiref22, phi_lm, phiL_lm, C1INSP, CINSP)

`pPhase->deltaphiLM` (`_internals.c` lines 2334-2369, plus the `(2,1)`-specific `+pi` correction at
lines 2400-2406): the frequency-INDEPENDENT phase offset that aligns mode `(l,m)`'s phase against
the `(2,2)` mode's own (fully LALSuite-aligned, see module docstring) phase, per Eq. (4.13) of
arXiv:2001.10914. `falign = m_over_2 * base_arg`, `base_arg = (eta > etaEMR ? 0.6 : 1.0)*fMECO22`
(this is `two_over_m*falign`, mode-INDEPENDENT once `m_over_2` is factored out -- confirmed by
direct algebraic simplification of the C source's own `falign`/`two_over_m*falign` pair, not
assumed). `phi_lm`/`phiL_lm`/`C1INSP`/`CINSP` are mode `(l,m)`'s own already-computed inspiral phase
coefficients and inspiral/intermediate glue constants (from [`_xhm_mode_setup`](@ref)/
[`_xhm_mode32_setup`](@ref) -- this function is shared by both non-mixing and mixing modes, matching
the C source's own `deltaphiLM` computation running unconditionally after the `MixingOn` switch).
"""
@inline function _xhm_deltaphilm(l::Integer, m::Integer, eta, chi1, chi2, coeffs22, timeshift, phiref22,
    phi_lm, phiL_lm, C1INSP, CINSP,
)
    T = coeffs22.T
    etaEMR = T(0.05)
    @Tlit T begin
        m_over_2 = T(m) * 0.5
        Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
        mm1 = 0.5 * (1.0 + Seta)
        mm2 = 0.5 * (1.0 - Seta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
        dchi = chi1 - chi2
    end
    fMECO22 = _xas_fMECO(eta, S, dchi, Seta)
    base_arg = eta > etaEMR ? T(0.6) * fMECO22 : fMECO22
    falign = m_over_2 * base_arg

    raw_phase_at_base = _xhm22_phase_raw(coeffs22, base_arg) / coeffs22.eta
    bracket = raw_phase_at_base + phiref22
    insp_val = _xhm_insp_phase_ansatz_int(falign, phi_lm, phiL_lm) + C1INSP * falign + CINSP
    dphiLM = m_over_2 * bracket + timeshift * falign - T(3.0) * T(pi) / T(4.0) * (one(T) - m_over_2) - insp_val

    if l == 2 && m == 1 && _xhm_pn21_ampsign(T(0.008), eta, chi1, chi2) > 0
        dphiLM = dphiLM + T(pi)
    end
    return dphiLM
end

"""
    _xhm_mode_setup(l, m, sys)

Everything needed to evaluate mode `(l,m)`'s (`(2,1)`,`(3,3)` or `(4,4)`) amplitude and phase at any
frequency: chains `_xhm_fring_fdamp`, the inspiral coefficients, the
intermediate coefficients (via the already-existing `_xhm_inter_amp_coefficients_from_inspiral`
convenience wrapper), the ringdown coefficients, this file's own C1INSP/CINSP/C1RD/CRD glue
(reusing `IMRPhenomXHM_intermediate.jl`'s own [`_xhm_inter_phase_glue`](@ref) directly -- its
signature already matches exactly what this needs) and `deltaphiLM`. `fAmpMatchIN` doubles as
`fPhaseMatchIN` (`= fMECOlm`, confirmed identical under the default release by both `IMRPhenomXHM_inspiral.jl`'s
module docstring and direct reading of `IMRPhenomXHM_Amplitude_fcutInsp`'s `q<20` branch vs
`IMRPhenomXHM_Intermediate_CollocPtsFreqs`'s unconditional `fPhaseMatchIN=fMECOlm`); likewise
`fBoundIM` doubles as `fAmpMatchIM`/`fPhaseMatchIM` (`= fring - fdamp` for every non-mixing mode).
"""
@inline function _xhm_mode_setup(l::Integer, m::Integer, sys)
    T = sys.T
    fring, fdamp = _xhm_fring_fdamp(l, m, sys.a, sys.Erad)

    inspcoeffs = _xhm_insp_amp_coefficients(l, m, sys.eta, sys.chi1, sys.chi2)
    phi_lm, phiL_lm = _xhm_insp_phase_coefficients(l, m, sys.eta, sys.chi1, sys.chi2)

    fAmpMatchIN = inspcoeffs.fcutInsp
    fBoundIM = fring - fdamp

    # LAL's `Ampzero` (IMRPhenomXHM_SetHMWaveformVariables): the odd modes (2,1), (3,3) vanish for equal masses
    # (eta = 0.25 after rounding: |delta| < ~1e-8) and equal spins, and LAL skips them. The amplitude fits are 0/0 there
    # (NaN), so they are evaluated at safe delta, chidiff (AD: finite discarded partials) and the mode is zeroed through
    # its amplitude scales (every ansatz is linear in them: PNglobalfactor / PNdominant, intamp, c0 / c3), which every
    # evaluator (incl. GravitationalWaveBayes' fast and device paths) then honours. Continuous: the amplitudes are O(delta) before.
    az = ((l == 2 && m == 1) || (l == 3 && m == 3)) && sys.eta == T(0.25) && sys.chi1 == sys.chi2
    delta = ifelse(az, T(1.0e-3), sys.Seta); chid = ifelse(az, T(1.0e-3), sys.dchi / 2)
    rdamp = _xhm_rd_amp_coefficients(l, m, sys.eta, delta, sys.StotR, sys.S, chid, fring, fdamp)
    V4 = _xhm_rd_amp_ansatz(fBoundIM, rdamp)
    d4 = _xhm_rd_amp_dansatz(fBoundIM, rdamp)
    intamp, fmid, invfhalf = _xhm_inter_amp_coefficients_from_inspiral(
        l, m, sys.eta, delta, sys.StotR, sys.S, chid, inspcoeffs, fBoundIM, V4, d4)
    z = ifelse(az, zero(T), one(T))
    inspcoeffs = merge(inspcoeffs, (PNglobalfactor = z * inspcoeffs.PNglobalfactor, PNdominant = z * inspcoeffs.PNdominant))
    intamp = map(c -> z * c, intamp)
    rdamp = merge(rdamp, (c0 = z * rdamp.c0, c3 = z * rdamp.c3))

    rdphase = _xhm_rd_phase_coefficients(l, m, sys.eta, sys.StotR, sys.chi1, sys.chi2, fring)
    intphase = _xhm_inter_phase_coefficients(l, m, sys.eta, sys.StotR, sys.chi1, sys.chi2, fAmpMatchIN, fring, fdamp,
                                             sys.coeffs22)

    insp_phi_val = _xhm_insp_phase_ansatz_int(fAmpMatchIN, phi_lm, phiL_lm)
    insp_phi_deriv = _xhm_insp_phase_ansatz(fAmpMatchIN, phi_lm, phiL_lm)
    dphi0RD = _xhm_rd_phase_dphidf(fBoundIM, rdphase, fring, fdamp)
    phi0RD = _xhm_rd_phase_ansatzint(fBoundIM, rdphase, fring, fdamp)

    C1INSP, CINSP, C1RD, CRD = _xhm_inter_phase_glue(
        intphase, fring, fdamp, fAmpMatchIN, insp_phi_deriv, insp_phi_val, fBoundIM, dphi0RD, phi0RD)

    deltaphiLM = _xhm_deltaphilm(l, m, sys.eta, sys.chi1, sys.chi2, sys.coeffs22, sys.timeshift,
        sys.phiref22, phi_lm, phiL_lm, C1INSP, CINSP)

    return (l = l, m = m, fAmpMatchIN = fAmpMatchIN, fBoundIM = fBoundIM, fring = fring, fdamp = fdamp,
        inspcoeffs = inspcoeffs, phi_lm = phi_lm, phiL_lm = phiL_lm, intamp = intamp, fmid = fmid,
        invfhalf = invfhalf, rdamp = rdamp, intphase = intphase, rdphase = rdphase,
        C1INSP = C1INSP, CINSP = CINSP, C1RD = C1RD, CRD = CRD, deltaphiLM = deltaphiLM)
end

"""
    _xhm_mode_amplitude(Mf, ms)

Piecewise inspiral/intermediate/ringdown amplitude ansatz for a non-mixing mode, using the
coefficient bundle from [`_xhm_mode_setup`](@ref). `ifelse`-based (not ternary), matching
`IMRPhenomXAS.jl`'s own top-level `_xas_ampl_scalar`/`_xas_phi_scalar` convention for the
per-frequency hot-path region selector (see those functions' docstrings for the Enzyme-activity
rationale) -- this is the analogous top-level per-frequency function in this file.
"""
@inline function _xhm_mode_amplitude(Mf, ms)
    ifelse(
        Mf < ms.fAmpMatchIN,
        _xhm_insp_amp_ansatz(Mf, ms.inspcoeffs),
        ifelse(
            Mf < ms.fBoundIM,
            _xhm_inter_amp_ansatz(Mf, ms.intamp, ms.fmid, ms.invfhalf),
            _xhm_rd_amp_ansatz(Mf, ms.rdamp),
        ),
    )
end

"""
    _xhm_mode_phase(Mf, ms)

Piecewise inspiral/intermediate/ringdown phase ansatz (`deltaphiLM`-corrected) for a non-mixing
mode. See [`_xhm_mode_amplitude`](@ref)'s docstring for the `ifelse` convention note.
"""
@inline function _xhm_mode_phase(Mf, ms)
    core = ifelse(
        Mf < ms.fAmpMatchIN,
        _xhm_insp_phase_ansatz_int(Mf, ms.phi_lm, ms.phiL_lm) + ms.C1INSP * Mf + ms.CINSP,
        ifelse(
            Mf < ms.fBoundIM,
            _xhm_inter_phase_ansatzint(Mf, ms.intphase, ms.fring, ms.fdamp),
            _xhm_rd_phase_ansatzint(Mf, ms.rdphase, ms.fring, ms.fdamp) + ms.C1RD * Mf + ms.CRD,
        ),
    )
    return core + ms.deltaphiLM
end

"""
    _xhm_hlm_nonmixing(l, f, sys, ms)

`h_{l,-m}(f)` for a non-mixing mode: `(-1)^l * Amp0 * amp(Mf) * exp(i*phase(Mf))`, matching
`IMRPhenomXHMGenerateFDOneMode`'s own `Amp0 = -Amp0` for odd `ell` (`LALSimIMRPhenomXHM.c` lines
601-604) and `Amp0*amp*cexp(I*phi)` reconstruction (line 699).
"""
@inline function _xhm_hlm_nonmixing(l::Integer, f, sys, ms)
    T = sys.T
    Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * f
    amp = _xhm_mode_amplitude(Mf, ms)
    phi = _xhm_mode_phase(Mf, ms)
    sgn = isodd(l) ? -one(T) : one(T)
    return sgn * _xhm_amp0(sys) * Complex{T}(amp * cos(phi), amp * sin(phi))
end

# ================================================================================================
# Mode-mixing: (3,2) mode. Ported from `GetSpheroidalCoefficients`/`SpheroidalToSpherical`
# (`LALSimIMRPhenomXHM_internals.c` lines 2465-2687), `IMRPhenomXHM_RD_Amp_Coefficients`'s
# `RDAmpVersion==1` branch + `IMRPhenomXHM_RDAux_Amp_Coefficients` (`_ringdown.c` lines 727-892),
# `IMRPhenomXHM_Intermediate_CollocPtsFreqs`'s `modeTag==32` branch and
# `IMRPhenomXHM_GetPhaseCoefficients`'s `MixingOn==1` branch (`_internals.c` lines 2176-2331), and
# the `(3,2)`-only 6-coefficient `IMRPhenomXHM_Inter_Phase_Ansatz`/`AnsatzInt` (`_intermediate.c`
# lines 3258-3312).
#
# No closures anywhere in this section (unlike a natural first-draft `SpheroidalToSpherical`
# implemented as a closure over its own coefficient bundle): `_xhm_s2s` takes its ingredient bundle
# (`_xhm_mode32_core`'s return value) as a plain argument instead, matching this package's
# established avoidance of closures in anything that might end up on an Enzyme/GPU-kernel hot path
# (see `IMRPhenomXPrecession.jl`'s `_xp_twist_up22` docstring for the precedent: "no arrays, Refs,
# closures, or loops over a tuple/array"). Flagged explicitly since `(3,2)`'s own finite-difference
# boundary extraction (below) is real numerical machinery that would need re-examining before this
# mode is ever put on that hot path for real -- a known, explicitly-scoped limitation of this port,
# not silently glossed over.
# ================================================================================================

"""
    _xhm_qnmfit_re_l3m2lp2(a) / _xhm_qnmfit_im_l3m2lp2(a) / _xhm_qnmfit_re_l3m2lp3(a) / _xhm_qnmfit_im_l3m2lp3(a)

Spherical-spheroidal mixing-coefficient QNM fits (`evaluate_QNMfit_{re,im}_l3m2lp{2,3}`,
`LALSimIMRPhenomXHM_qnm.c` lines 206-311), functions of the `(2,2)`-mode's own final dimensionless
spin `a` only. The `l2m2lp{2,3}` fits (mixing coefficients `[0]`/`[1]`, needed only for a
hypothetical `(2,2)`-mode-mixing correction real `IMRPhenomXHM` never actually applies -- `(2,2)` is
the dominant mode and is used as-is, unmixed) are NOT ported: confirmed unused by
`SpheroidalToSpherical`, which only ever reads `mixingCoeffs[2]`/`[3]` (`_internals.c` line 2666).
"""
@inline function _xhm_qnmfit_re_l3m2lp2(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a; x3 = x2 * a; x4 = x2 * x2; x5 = x3 * x2
        return (a * (0.47513455283841244 - 0.9016636384605536 * a + 0.3844811236426182 * x2 +
                     0.0855565148647794 * x3 - 0.03620067426672167 * x4 - 0.006557249133752502 * x5)) /
               (-6.76894063440646 + 15.170831931186493 * a - 9.406169787571082 * x2 + 1.0 * x4)
    end
end

@inline function _xhm_qnmfit_im_l3m2lp2(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a; x3 = x2 * a; x4 = x2 * x2; x5 = x3 * x2; x6 = x3 * x3
        return (a * (-2.8704762147145533 + 4.436434016918535 * a - 1.0115343326360486 * x2 -
                     0.08965314412106505 * x3 - 0.4236810894599512 * x4 - 0.041787576033810676 * x5)) /
               (-171.80908957903395 + 272.362882450877 * a - 76.68544453077854 * x2 - 25.14197656531123 * x4 + 1.0 * x6)
    end
end

@inline function _xhm_qnmfit_re_l3m2lp3(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a; x3 = x2 * a; x4 = x2 * x2; x5 = x3 * x2; x6 = x3 * x3
        return (1.0 - 2.107852425643677 * a + 1.1906393634562715 * x2 + 0.02244848864087732 * x3 -
                0.09593447799423722 * x4 - 0.0021343381708933025 * x5 - 0.005319515989331159 * x6) /
               (1.0 - 2.1078515887706324 * a + 1.2043484690080966 * x2 - 0.08910191596778137 * x4 - 0.005471749827809503 * x6)
    end
end

@inline function _xhm_qnmfit_im_l3m2lp3(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a; x3 = x2 * a; x4 = x2 * x2; x5 = x3 * x2; x6 = x3 * x3
        return (a * (12.45701482868677 - 29.398484595717147 * a + 18.26221675782779 * x2 +
                     1.9308599142669403 * x3 - 3.159763242921214 * x4 - 0.0910871567367674 * x5)) /
               (345.52914639836257 - 815.4349339779621 * a + 538.3888932415709 * x2 - 69.3840921447381 * x4 + 1.0 * x6)
    end
end

"""
    _xhm_spheroidal_phase32_raw(sys, fring32, fdamp32)

The 4-point linear solve for `(alpha0_S, alphaL_S, alpha2_S, alpha4_S)` (`GetSpheroidalCoefficients`,
`122019`-branch, `_internals.c` lines 2492-2537 -- LIVE under the actual default release, see
`IMRPhenomXHM_ringdown.jl`'s module docstring for the confirmation that `IMRPhenomXHMRingdownPhaseVersion`
stays `122019`), using the `_xhm_rd_phase32_p1`-`p4` fits at frequencies `[fring22,
fring32-1.5*fdamp32, fring32-0.5*fdamp32, fring32+0.5*fdamp32]` (`IMRPhenomXHM_Ringdown_CollocPtsFreqs`'s
`122019` branch). NOT yet the final `alpha0_S`/`phi0_S` (see [`_xhm_spheroidal32_setup`](@ref) for
the additional `(2,2)`-mode-anchored time/phase-shift correction every `(3,2)` ringdown phase needs).
"""
@inline function _xhm_spheroidal_phase32_raw(sys, fring32, fdamp32)
    T = sys.T
    dchi = sys.chi1 - sys.chi2
    p1 = _xhm_rd_phase32_p1(sys.eta, sys.StotR, dchi, sys.chi1, sys.chi2)
    p2 = _xhm_rd_phase32_p2(sys.eta, sys.StotR, dchi)
    p3 = _xhm_rd_phase32_p3(sys.eta, sys.StotR, dchi)
    p4 = _xhm_rd_phase32_p4(sys.eta, sys.StotR, dchi)

    f0 = sys.coeffs22.fRD
    f1 = fring32 - T(1.5) * fdamp32
    f2 = fring32 - T(0.5) * fdamp32
    f3 = fring32 + T(0.5) * fdamp32
    row(f) = begin      # no `T` in the body: a closure over `T = sys.T` (a DataType value) breaks Enzyme on Julia 1.13.1
        L = fdamp32 / (fdamp32 * fdamp32 + (f - fring32) * (f - fring32))
        (one(L), L, one(L) / (f * f), one(L) / (f * f * f * f))
    end
    A = (row(f0), row(f1), row(f2), row(f3))
    b = (T(p1), T(p2), T(p3), T(p4))
    c = _xas_linsolve(A, b)
    return (alpha0_S = c[1], alphaL_S = c[2], alpha2_S = c[3], alpha4_S = c[4])
end

"""
    _xhm_spheroidal32_dphidf(f, sph, fring32, fdamp32) / _xhm_spheroidal32_ansatzint(f, sph, phi0_S, fring32, fdamp32)

The `(3,2)`-only spheroidal ringdown phase-derivative/phase ansatz (`IMRPhenomXHM_RD_Phase_Ansatz`/
`_AnsatzInt`'s `MixingOn==1`, `IMRPhenomXHMRingdownPhaseVersion==122019` branch, `_ringdown.c` lines
1404-1406/1449-1451): `dphi/df = alpha0_S + alpha2_S/f^2 + alpha4_S/f^4 + alphaL_S*fdamp/(fdamp^2+
(f-fring)^2)`, `phi = phi0_S + alpha0_S*f - alpha2_S/f - alpha4_S/(3f^3) + alphaL_S*atan((f-fring)/fdamp)`.
Note this uses `fring32`/`fdamp32` (the `(3,2)` mode's own QNM values) in the Lorentzian/atan term,
NOT `(2,2)`'s.
"""
@inline function _xhm_spheroidal32_dphidf(f, sph, fring32, fdamp32)
    T = promote_type(typeof(f), typeof(fring32), typeof(fdamp32))
    @Tlit T begin
        return sph.alpha0_S + sph.alpha2_S / (f * f) + sph.alpha4_S / (f * f * f * f) +
               sph.alphaL_S * fdamp32 / (fdamp32 * fdamp32 + (f - fring32) * (f - fring32))
    end
end

@inline function _xhm_spheroidal32_ansatzint(f, sph, phi0_S, fring32, fdamp32)
    T = promote_type(typeof(f), typeof(fring32), typeof(fdamp32))
    @Tlit T begin
        return phi0_S + sph.alpha0_S * f - sph.alpha2_S / f - sph.alpha4_S / (3.0 * f * f * f) +
               sph.alphaL_S * atan((f - fring32) / fdamp32)
    end
end

"""
    _xhm_spheroidal32_setup(sys, fring32, fdamp32)

The full `GetSpheroidalCoefficients` port: [`_xhm_spheroidal_phase32_raw`](@ref)'s raw 4-coefficient
solve, THEN the `alpha0_S` time-shift correction anchored to the `(2,2)` mode's own ringdown phase
derivative at `frefRD = fRING22 + fDAMP22` (`_internals.c` lines 2558-2573, `dphi22ref =
(1/eta)*dPhase_22(frefRD) + timeshift`, `tshift = _xhm_rd_phase32_timeshift` fit, `alpha0_S +=
dphi22ref + tshift - dphidf_raw(frefRD)`), THEN the `phi0_S` phase-shift correction anchored at
`frefRD2 = fRING22` (lines 2598-2624, `phi22ref = phase_full(frefRD2)`, `phishift =
_xhm_rd_phase32_phaseshift` fit, `phi0_S = phi22ref - ansatzint(frefRD2; phi0_S=0) + phishift`).
"""
@inline function _xhm_spheroidal32_setup(sys, fring32, fdamp32)
    T = sys.T
    sph_raw = _xhm_spheroidal_phase32_raw(sys, fring32, fdamp32)

    frefRD = sys.coeffs22.fRD + sys.coeffs22.fdamp
    dphi22ref = _xhm22_dphase_raw(sys.coeffs22, frefRD) / sys.eta + sys.timeshift
    tshift = _xhm_rd_phase32_timeshift(sys.eta, sys.StotR, sys.dchi)
    dphi32ref_raw = _xhm_spheroidal32_dphidf(frefRD, sph_raw, fring32, fdamp32)
    alpha0_S = sph_raw.alpha0_S + dphi22ref + tshift - dphi32ref_raw
    sph = (alpha0_S = alpha0_S, alphaL_S = sph_raw.alphaL_S, alpha2_S = sph_raw.alpha2_S, alpha4_S = sph_raw.alpha4_S)

    frefRD2 = sys.coeffs22.fRD
    phi22ref = _xhm22_phase_full(sys.coeffs22, sys.timeshift, sys.phiref22, frefRD2)
    phishift = _xhm_rd_phase32_phaseshift(sys.eta, sys.StotR, sys.chi1, sys.chi2)
    ansatzint_zero_phi0 = _xhm_spheroidal32_ansatzint(frefRD2, sph, zero(T), fring32, fdamp32)
    phi0_S = phi22ref - ansatzint_zero_phi0 + phishift

    return (sph = sph, phi0_S = phi0_S)
end

"""
    _xhm_rd_amp32_ansatz(f, c)

The `(3,2)` ringdown amplitude ansatz's THREE-way split (`IMRPhenomXHM_RD_Amp_Ansatz`'s
`RDAmpVersion` case 1/2 branch, `_ringdown.c` lines 920-937): the `RDAux` polynomial for `f <
fRDAux`, the exponential falloff for `f >= falloff`, else the Lorentzian
([`_xhm_rd_amp_lorentzian`](@ref), reused directly from `IMRPhenomXHM_ringdown.jl`). Uses plain
ternary `?:` for the inner branch selection, matching `IMRPhenomXHM_ringdown.jl`'s own already-merged
`_xhm_rd_amp_ansatz` convention for this exact kind of nested region split (only this file's
TOP-level per-frequency functions use `ifelse`, see [`_xhm_mode_amplitude`](@ref)'s docstring).
"""
@inline function _xhm_rd_amp32_ansatz(f, c)
    T = promote_type(typeof(f), typeof(c.c0))
    f < c.fRDAux ? (c.rdaux[1] + f * (c.rdaux[2] + f * (c.rdaux[3] + f * c.rdaux[4]))) :
    f >= c.falloff ? c.c3 * exp(-c.c4 * (f - c.falloff)) :
    _xhm_rd_amp_lorentzian(f, c.c0, c.c1, c.c2, c.fring, c.fdamp)
end

"""
    _xhm_rd_amp32_coefficients(sys, fring32, fdamp32, fAmpMatchIM)

`(3,2)`'s ringdown amplitude coefficients: the Lorentzian `(c0,c1,c2)` directly from `IMRPhenomXHM_ringdown.jl`'s
`_xhm_rd_amp32_alambda`/`_lambda`/`_sigma` fits (`RDAmpVersion==1`, `_ringdown.c` lines 739-742, no
linear solve -- confirmed live for `(3,2)` specifically, per `IMRPhenomXHM_ringdown.jl`'s own module
docstring), the exponential-falloff continuity constants `(c3,c4)` at `falloff = fring32+2*fdamp32`
(seeded from the Lorentzian's own value/derivative there, matching every other mode), and the
`RDAux` 4-coefficient polynomial (`IMRPhenomXHM_RDAux_Amp_Coefficients`, `_ringdown.c` lines
818-892): a 4x4 linear solve matching `rdaux1`/`rdaux2` (fits from the ringdown file) at
`[fAmpMatchIM, 0.5*(fAmpMatchIM+fRDAux)]` plus the Lorentzian's OWN value+derivative at `fRDAux =
fring32-fdamp32` (NOT `SpheroidalToSpherical` -- this is fully self-contained within the `(3,2)`
ringdown ansatz itself, confirmed by direct reading; no circular dependency on the mode-mixing
rotation).
"""
@inline function _xhm_rd_amp32_coefficients(sys, fring32, fdamp32, fAmpMatchIM)
    T = sys.T
    chidiff = sys.dchi / 2
    c0 = abs(_xhm_rd_amp32_alambda(sys.eta, sys.Seta, sys.StotR, chidiff))
    c1 = _xhm_rd_amp32_lambda(sys.eta, sys.Seta, sys.StotR, chidiff)
    c2 = _xhm_rd_amp32_sigma(sys.eta, sys.Seta, sys.StotR, chidiff)
    falloff = fring32 + T(2.0) * fdamp32
    c3 = _xhm_rd_amp_lorentzian(falloff, c0, c1, c2, fring32, fdamp32)
    c4 = -_xhm_rd_amp_dlorentzian(falloff, c0, c1, c2, fring32, fdamp32) / c3

    fRDAux = fring32 - fdamp32
    rdaux1 = abs(_xhm_rd_amp32_rdaux1(sys.eta, sys.Seta, sys.StotR, chidiff))
    rdaux2 = abs(_xhm_rd_amp32_rdaux2(sys.eta, sys.Seta, sys.S, chidiff))
    val_at_fRDAux = _xhm_rd_amp_lorentzian(fRDAux, c0, c1, c2, fring32, fdamp32)
    deriv_at_fRDAux = _xhm_rd_amp_dlorentzian(fRDAux, c0, c1, c2, fring32, fdamp32)
    fmid = T(0.5) * (fAmpMatchIM + fRDAux)

    Arow_val(f) = (one(T), f, f * f, f * f * f)
    Arow_deriv(f) = (zero(T), one(T), T(2.0) * f, T(3.0) * f * f)
    A = (Arow_val(fAmpMatchIM), Arow_val(fmid), Arow_val(fRDAux), Arow_deriv(fRDAux))
    bb = (T(rdaux1), T(rdaux2), val_at_fRDAux, deriv_at_fRDAux)
    rdaux = _xas_linsolve(A, bb)

    return (c0 = c0, c1 = c1, c2 = c2, c3 = c3, c4 = c4, fring = fring32, fdamp = fdamp32,
        falloff = falloff, fRDAux = fRDAux, rdaux = rdaux)
end

"""
    _xhm_mode32_core(sys)

Everything `(3,2)`'s [`_xhm_s2s`](@ref) (spheroidal-to-spherical rotation) needs, computed once:
`(3,2)`'s own `fring`/`fdamp` (`IMRPhenomXHM_qnm.jl`), the complex mixing coefficients
(`wf->mixingCoeffs[2]/[3] = -1 * (re+i*im)_{l3m2lp{2,3}}(afinal)`, `_internals.c` lines 64-71 --
the sign flip is part of the port, not a typo), the `(2,2)`-mode's OWN amplitude coefficients
(`_xas_ampl_coeffs`, needed for `SpheroidalToSpherical`'s `amp22`/`ampNorm` terms -- `dL` is passed
as `one(T)` since every quantity `_xhm_s2s` actually reads from this bundle, `amp_rd.gamma{1,2,3}`
and `amp0`(`=ampNorm`), is `dL`-independent; only `Overallamp` scales with `dL` and that field is
never read here), the spheroidal ringdown phase setup ([`_xhm_spheroidal32_setup`](@ref)), the
mode-mixing-specific `fAmpMatchIM = fRING22 - 0.5*fDAMP22` (`IMRPhenomXHM_Amplitude_fcutRD`'s
`MixingOn==1` branch -- a DIFFERENT formula than non-mixing modes' `fring-fdamp`, confirmed by direct
reading), and the ringdown amplitude coefficients.
"""
@inline function _xhm_mode32_core(sys)
    T = sys.T
    fring32, fdamp32 = _xhm_fring_fdamp(3, 2, sys.a, sys.Erad)
    mc2 = -Complex{T}(_xhm_qnmfit_re_l3m2lp2(sys.a), _xhm_qnmfit_im_l3m2lp2(sys.a))
    mc3 = -Complex{T}(_xhm_qnmfit_re_l3m2lp3(sys.a), _xhm_qnmfit_im_l3m2lp3(sys.a))
    amp22coeffs = _xas_ampl_coeffs(IMRPhenomXAS(), one(T), sys.eta, sys.chi1, sys.chi2, one(T))
    sph32 = _xhm_spheroidal32_setup(sys, fring32, fdamp32)
    fAmpMatchIM = sys.coeffs22.fRD - T(0.5) * sys.coeffs22.fdamp
    rdamp32 = _xhm_rd_amp32_coefficients(sys, fring32, fdamp32, fAmpMatchIM)
    return (fring32 = fring32, fdamp32 = fdamp32, mc2 = mc2, mc3 = mc3, amp22coeffs = amp22coeffs,
        sph32 = sph32, fAmpMatchIM = fAmpMatchIM, rdamp32 = rdamp32)
end

"""
    _xhm_s2s(Mf, sys, core)

`SpheroidalToSpherical` (`_internals.c` lines 2652-2669): `conj(mixingCoeffs[2])*wf22R +
conj(mixingCoeffs[3])*amplm*exp(i*philm)`, where `wf22R` is the `(2,2)`-mode's OWN ringdown ansatz
(`XLALSimIMRPhenomXRingdownAmplitude22AnsatzAnalytical` = this package's own
[`_xas_amp_rd_val`](@ref), times `exp(i*phase_full_22)`), rescaled by `ampNorm22*Mf^(-7/6)`
(unconditional here since `(3,2)`'s `IMRPhenomXHMRingdownAmpVersion` is always `1`, i.e. always
nonzero, under the default release -- matches the C source's own `if(RingdownAmpVersion!=0)` guard
exactly since that condition is always true in this port's scope), and `amplm`/`philm` are `(3,2)`'s
own spheroidal ringdown amplitude/phase ([`_xhm_rd_amp32_ansatz`](@ref)/
[`_xhm_spheroidal32_ansatzint`](@ref)). Returns the mode-mixed complex value DIRECTLY (not split
into amplitude/phase) -- deliberately, since `exp(i*carg(z)) = z/|z|` makes any subsequent
`|z|*exp(i*(carg(z)+linear-correction))` reconstruction exactly equal to `z*exp(i*linear-correction)`
without ever needing to unwrap `carg`'s `(-pi,pi]` branch (see [`_xhm_hlm_32`](@ref)'s docstring for
where this matters).
"""
@inline function _xhm_s2s(Mf, sys, core)
    T = sys.T
    amp22 = _xas_amp_rd_val(Mf, core.amp22coeffs.amp_rd.gamma1, core.amp22coeffs.amp_rd.gamma2,
        core.amp22coeffs.amp_rd.gamma3, sys.coeffs22.fdamp, sys.coeffs22.fRD)
    phi22 = _xhm22_phase_full(sys.coeffs22, sys.timeshift, sys.phiref22, Mf)
    wf22R = Complex{T}(amp22 * cos(phi22), amp22 * sin(phi22)) *
            (core.amp22coeffs.amp0 * Mf^(T(-7.0) / T(6.0)))
    amplm = _xhm_rd_amp32_ansatz(Mf, core.rdamp32)
    philm = _xhm_spheroidal32_ansatzint(Mf, core.sph32.sph, core.sph32.phi0_S, core.fring32, core.fdamp32)
    return conj(core.mc2) * wf22R + conj(core.mc3) * amplm * Complex{T}(cos(philm), sin(philm))
end

"""
    _xhm_inter_phase32_freqs(sys, fAmpMatchIN, fEnd)

`(fcut, f1, f2, f3)`: `(3,2)`'s intermediate-phase collocation-point frequencies
(`IMRPhenomXHM_Intermediate_CollocPtsFreqs`'s `modeTag==32` branch, `_internals.c` lines 890-901).
`fcut = GetfcutInsp(...)` (the SAME `eta_factor*fMECOlm` rescaling `IMRPhenomXHM_intermediate.jl`'s
own `_xhm_inter_phase_colloc_freqs` uses for non-mixing modes -- NOT the same as `fAmpMatchIN`
itself, which is the unscaled `fMECOlm`); `fEnd = fAmpMatchIM = fRING22-0.5*fDAMP22`.
"""
@inline function _xhm_inter_phase32_freqs(sys, fAmpMatchIN, fEnd)
    T = sys.T
    @Tlit T begin
        eta_factor = 1.0 + 0.001 * (0.25 / sys.eta - 1.0)
        fcut = eta_factor * fAmpMatchIN
        f1 = (sqrt(3.0) * (fcut - fEnd) + 2.0 * (fcut + fEnd)) / 4.0
        f2 = (3.0 * fcut + fEnd) / 4.0
        f3 = (fcut + fEnd) / 2.0
        return fcut, f1, f2, f3
    end
end

"""
    _xhm_inter_phase32_dphidf(f, c, fring, fdamp) / _xhm_inter_phase32_ansatzint(f, c, fring, fdamp)

`(3,2)`'s 6-coefficient intermediate phase ansatz (`IMRPhenomXHM_Inter_Phase_Ansatz`/`_AnsatzInt`'s
`modeTag==32` branch, `_intermediate.c` lines 3273-3312): `dphi/df = c0 + c1/f + c2/f^2 + c3/f^3 +
c4/f^4 + cL*fdamp/(fdamp^2+(f-fring)^2)`, `phi = c0*f + c1*log(f) - c2/f - c3/(2f^2) - c4/(3f^3) +
cL*atan((f-fring)/fdamp)` -- the EXTRA `c3/f^3` term (vs. the 5-coefficient non-mixing ansatz,
`IMRPhenomXHM_intermediate.jl`'s own `_xhm_inter_phase_dphidf`/`_ansatzint`) is `(3,2)`-specific;
confirmed by direct reading, not assumed uniform across modes.
"""
@inline function _xhm_inter_phase32_dphidf(f, c, fring, fdamp)
    T = promote_type(typeof(f), typeof(fring), typeof(fdamp), typeof(c.c0))
    @Tlit T begin
        invf = 1.0 / f
        invf2 = invf * invf
        invf3 = invf2 * invf
        invf4 = invf2 * invf2
        return c.c0 + c.c1 * invf + c.c2 * invf2 + c.c3 * invf3 + c.c4 * invf4 +
               c.cL * fdamp / (fdamp * fdamp + (f - fring) * (f - fring))
    end
end

@inline function _xhm_inter_phase32_ansatzint(f, c, fring, fdamp)
    T = promote_type(typeof(f), typeof(fring), typeof(fdamp), typeof(c.c0))
    @Tlit T begin
        invf = 1.0 / f
        invf2 = invf * invf
        invf3 = invf2 * invf
        return c.c0 * f + c.c1 * log(f) - c.c2 * invf - T(0.5) * c.c3 * invf2 - (1.0 / 3.0) * c.c4 * invf3 +
               c.cL * atan((f - fring) / fdamp)
    end
end

"""
    _xhm_inter_phase32_coefficients(sys, core, fAmpMatchIN, fEnd)

`(3,2)`'s intermediate phase coefficients: a 6x6 linear solve (`IMRPhenomXHM_GetPhaseCoefficients`'s
`MixingOn==1` branch, `_internals.c` lines 2176-2253) combining the `p1`-`p4` collocation
fits (`_xhm_inter_phase_32_p1`-`p4`, already ported) with TWO more constraints extracted by
finite-differencing `SpheroidalToSpherical`'s own complex phase around `fEnd` (`fstep=1e-7`, matching
`_internals.c` lines 2183-2230 exactly): the wrapped phase VALUE at `fEnd` (`phi0RD`, not directly
used by the linear system itself but carried through for the `C1RD`/`CRD` glue), its first derivative
(`dphi0RD`, row 5 of the system, using the ansatz's ordinary VALUE basis since this row represents
"the phase-DERIVATIVE ansatz's value equals dphi0RD" -- every row of this system fits the
phase-derivative ansatz directly, not the phase itself), and its second derivative (`d2phi0RD`, row
6, using the DERIVATIVE of that basis). Branch-unwrapping (`argm/arg0/argp > 0 ? (...)-2*pi :
(...)`) matches `_internals.c`'s own `if(phase_args[i]>0) phase_args[i]-=2*LAL_PI` exactly.
"""
@inline function _xhm_inter_phase32_coefficients(sys, core, fAmpMatchIN, fEnd)
    T = sys.T
    dchi = sys.chi1 - sys.chi2
    deltaT = _xhm_inter_phase_deltaT(sys.eta, sys.StotR, dchi)
    p1v = _xhm_inter_phase_32_p1(sys.eta, sys.StotR, sys.chi1, sys.chi2) + deltaT
    p2v = _xhm_inter_phase_32_p2(sys.eta, sys.StotR, sys.chi1, sys.chi2) + deltaT
    p3v = _xhm_inter_phase_32_p3(sys.eta, sys.StotR, dchi) + deltaT
    p4v = _xhm_inter_phase_32_p4(sys.eta, sys.StotR, dchi) + deltaT

    fcut, f1, f2, f3 = _xhm_inter_phase32_freqs(sys, fAmpMatchIN, fEnd)

    # dphi0RD / d2phi0RD: the C source differences arg(S2S) with fstep = 1e-7, which for the second derivative is
    # roundoff / fstep^2 ~ 1e-2 relative noise -> the 6x6 solve below moved the (3,2) intermediate-phase coefficients by
    # ~1% under 1e-15 changes of the masses. Here: 5-point stencils with fstep = 1e-4 fEnd on the phase
    # RELATIVE to z(fEnd) (angle(z(f)/z0): small, no branch cut) and fstep = 1e-3 fEnd: arg(S2S) carries ~1e-13 rad of
    # roundoff (built from mode phases of O(100) rad), so d2 noise ~1e-13 / fstep^2 ~ 1e-5; truncation ~(fstep/fdamp)^4
    # ~ 1e-9. Differs from the C value by about the C value's own noise.
    fstep = T(1e-3) * fEnd
    z0 = _xhm_s2s(fEnd, sys, core)
    a2m = angle(_xhm_s2s(fEnd - 2 * fstep, sys, core) / z0)
    a1m = angle(_xhm_s2s(fEnd - fstep, sys, core) / z0)
    a1p = angle(_xhm_s2s(fEnd + fstep, sys, core) / z0)
    a2p = angle(_xhm_s2s(fEnd + 2 * fstep, sys, core) / z0)
    arg0 = angle(z0)
    arg0 = arg0 > 0 ? arg0 - 2 * T(pi) : arg0
    phi0RD = arg0
    dphi0RD = (-a2p + 8 * a1p - 8 * a1m + a2m) / (12 * fstep)
    d2phi0RD = (-a2p + 16 * a1p + 16 * a1m - a2m) / (12 * fstep * fstep)

    fring32, fdamp32 = core.fring32, core.fdamp32
    A = (_xhm_phase32_row(fcut, fring32, fdamp32), _xhm_phase32_row(f1, fring32, fdamp32),
         _xhm_phase32_row(f2, fring32, fdamp32), _xhm_phase32_row(f3, fring32, fdamp32),
         _xhm_phase32_row(fEnd, fring32, fdamp32), _xhm_phase32_rowderiv(fEnd, fring32, fdamp32))
    b = (T(p1v), T(p2v), T(p3v), T(p4v), dphi0RD, d2phi0RD)
    c = _xas_linsolve(A, b)
    return (c0 = c[1], cL = c[2], c1 = c[3], c2 = c[4], c4 = c[5], c3 = c[6], phi0RD = phi0RD, dphi0RD = dphi0RD)
end

# Rows of the (3,2) intermediate-phase system (`_xhm_inter_phase32_coefficients`): the ansatz terms at f and their f
# derivatives. Top-level functions, not closures: a closure over `T = sys.T` (a DataType VALUE) captured a pointer, and
# Enzyme's type analysis rejects that capture on Julia 1.13.1 ("illegal type analysis").
@inline function _xhm_phase32_row(f, fring32, fdamp32)
    T = typeof(f * fdamp32)
    L = fdamp32 / (fdamp32 * fdamp32 + (f - fring32) * (f - fring32))
    return (one(T), L, one(T) / f, one(T) / (f * f), one(T) / (f * f * f * f), one(T) / (f * f * f))
end
@inline function _xhm_phase32_rowderiv(f, fring32, fdamp32)
    T = typeof(f * fdamp32)
    num = -T(2.0) * fdamp32 * (f - fring32)
    den = (fdamp32 * fdamp32 + (f - fring32) * (f - fring32))^2
    return (zero(T), num / den, -one(T) / (f * f), -T(2.0) / (f * f * f), -T(4.0) / (f^5), -T(3.0) / (f^4))
end

"""
    _xhm_mode32_setup(sys)

The full `(3,2)` mode setup: [`_xhm_mode32_core`](@ref)'s mode-mixing ingredients, the
inspiral coefficients (identical machinery to the non-mixing modes -- `(3,2)`'s inspiral region is
NOT mode-mixed, confirmed by `IMRPhenomXHM_Amplitude_ModeMixing`/`_Phase_ModeMixing`'s own inspiral
branches calling the ordinary `IMRPhenomXHM_Inspiral_Amp_Ansatz`/`_Phase_AnsatzInt`), the
intermediate-region boundary bootstrapped from `SpheroidalToSpherical` (`V4 = |S2S(fAmpMatchIM)|`,
`d4` via the SAME 4th-order central finite difference `IMRPhenomXHM_inspiral.jl`'s own
`_xhm_insp_amp_nd_ansatz` uses, `df=10e-10` -- matching `IMRPhenomXHM_RD_Amp_NDAnsatz`'s role, not
ported standalone anywhere in this package (see `IMRPhenomXHM_ringdown.jl`), but its exact
finite-difference SCHEME is reused here), `(3,2)`'s intermediate amplitude coefficients (reusing
`IMRPhenomXHM_intermediate.jl`'s own general 8-coefficient `_xhm_inter_amp_coefficients` directly --
already handles `(l,m)=(3,2)` per that function's own signature, no `(3,2)`-specific amplitude
machinery needed beyond the boundary bootstrap), `(3,2)`'s intermediate phase coefficients
([`_xhm_inter_phase32_coefficients`](@ref)), the C1INSP/CINSP/C1RD/CRD glue (C1RD/CRD matched
against the FD-extracted `dphi0RD`/`phi0RD` directly, NOT a standalone ringdown ansatz -- matching
`_internals.c`'s own `pPhase->C1RD=Inter_Phase_Ansatz(fcutRD,...)-pPhase->dphi0RD` for the mixing
case specifically), and `deltaphiLM`.
"""
@inline function _xhm_mode32_setup(sys)
    T = sys.T
    core = _xhm_mode32_core(sys)

    inspcoeffs = _xhm_insp_amp_coefficients(3, 2, sys.eta, sys.chi1, sys.chi2)
    phi_lm, phiL_lm = _xhm_insp_phase_coefficients(3, 2, sys.eta, sys.chi1, sys.chi2)
    fAmpMatchIN = inspcoeffs.fcutInsp
    fAmpMatchIM = core.fAmpMatchIM

    F1 = fAmpMatchIN
    V1 = _xhm_insp_amp_ansatz(F1, inspcoeffs)
    d1 = _xhm_insp_amp_nd_ansatz(F1, inspcoeffs)

    df = T(1e-4) * fAmpMatchIM          # scaled step (C source: 10e-10, ~1e-7 roundoff noise; see _xhm_insp_amp_nd_ansatz)
    V4p2 = abs(_xhm_s2s(fAmpMatchIM + 2 * df, sys, core))
    V4p1 = abs(_xhm_s2s(fAmpMatchIM + df, sys, core))
    V4m1 = abs(_xhm_s2s(fAmpMatchIM - df, sys, core))
    V4m2 = abs(_xhm_s2s(fAmpMatchIM - 2 * df, sys, core))
    V4 = abs(_xhm_s2s(fAmpMatchIM, sys, core))
    d4 = (-V4p2 + 8 * V4p1 - 8 * V4m1 + V4m2) / (12 * df)

    intamp, fmid, invfhalf = _xhm_inter_amp_coefficients(
        3, 2, sys.eta, sys.Seta, sys.StotR, sys.S, sys.dchi / 2, F1, V1, d1, fAmpMatchIM, V4, d4)

    intphase32 = _xhm_inter_phase32_coefficients(sys, core, fAmpMatchIN, fAmpMatchIM)

    insp_phi_val = _xhm_insp_phase_ansatz_int(fAmpMatchIN, phi_lm, phiL_lm)
    insp_phi_deriv = _xhm_insp_phase_ansatz(fAmpMatchIN, phi_lm, phiL_lm)
    C1INSP = _xhm_inter_phase32_dphidf(fAmpMatchIN, intphase32, core.fring32, core.fdamp32) - insp_phi_deriv
    CINSP = -(C1INSP * fAmpMatchIN) + _xhm_inter_phase32_ansatzint(fAmpMatchIN, intphase32, core.fring32, core.fdamp32) - insp_phi_val
    C1RD = _xhm_inter_phase32_dphidf(fAmpMatchIM, intphase32, core.fring32, core.fdamp32) - intphase32.dphi0RD
    CRD = -C1RD * fAmpMatchIM + _xhm_inter_phase32_ansatzint(fAmpMatchIM, intphase32, core.fring32, core.fdamp32) - intphase32.phi0RD

    deltaphiLM = _xhm_deltaphilm(3, 2, sys.eta, sys.chi1, sys.chi2, sys.coeffs22, sys.timeshift,
        sys.phiref22, phi_lm, phiL_lm, C1INSP, CINSP)

    return (core = core, fAmpMatchIN = fAmpMatchIN, fAmpMatchIM = fAmpMatchIM, inspcoeffs = inspcoeffs,
        phi_lm = phi_lm, phiL_lm = phiL_lm, intamp = intamp, fmid = fmid, invfhalf = invfhalf,
        intphase32 = intphase32, C1INSP = C1INSP, CINSP = CINSP, C1RD = C1RD, CRD = CRD, deltaphiLM = deltaphiLM)
end

"""
    _xhm_hlm_32(f, sys, m32)

`h_{3,-2}(f)`: inspiral/intermediate regions use the standard real-amplitude/real-phase
reconstruction (identical structure to [`_xhm_hlm_nonmixing`](@ref), just with `(3,2)`'s own
mode-mixing-aware coefficients); the ringdown region (`Mf >= fAmpMatchIM`) uses
`_xhm_s2s(Mf,sys,core) * exp(i*(C1RD*Mf+CRD+deltaphiLM))` DIRECTLY as a complex value rather than
splitting into `cabs`/`carg(...)+correction` -- avoiding any phase-unwrapping concern entirely (see
[`_xhm_s2s`](@ref)'s own docstring) while remaining exactly equivalent to
`IMRPhenomXHM_Amplitude_ModeMixing`/`_Phase_ModeMixing`'s own `cabs(...)`/`carg(...)+correction`
construction (`Amp0*amp*exp(i*phi) == Amp0*|z|*exp(i*(carg(z)+correction)) == Amp0*z*exp(i*correction)`
since `exp(i*carg(z))=z/|z|` identically). `Amp0` gets the `(-1)^l` sign flip for `l=3` (odd), same
as every other mode.
"""
@inline function _xhm_hlm_32(f, sys, m32)
    T = sys.T
    Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * f
    core = m32.core
    h = if Mf < m32.fAmpMatchIN
        amp = _xhm_insp_amp_ansatz(Mf, m32.inspcoeffs)
        phi = _xhm_insp_phase_ansatz_int(Mf, m32.phi_lm, m32.phiL_lm) + m32.C1INSP * Mf + m32.CINSP + m32.deltaphiLM
        Complex{T}(amp * cos(phi), amp * sin(phi))
    elseif Mf < m32.fAmpMatchIM
        amp = _xhm_inter_amp_ansatz(Mf, m32.intamp, m32.fmid, m32.invfhalf)
        phi = _xhm_inter_phase32_ansatzint(Mf, m32.intphase32, core.fring32, core.fdamp32) + m32.deltaphiLM
        Complex{T}(amp * cos(phi), amp * sin(phi))
    else
        z = _xhm_s2s(Mf, sys, core)
        corr = m32.C1RD * Mf + m32.CRD + m32.deltaphiLM
        z * Complex{T}(cos(corr), sin(corr))
    end
    return -_xhm_amp0(sys) * h   # l=3 is always odd -> (-1)^l = -1
end

# ================================================================================================
# Spin-weighted spherical harmonics (s=-2), l=2,3,4, only the (l,m) combinations this file's 5-mode
# scope actually needs ((2,±1),(2,±2),(3,±2),(3,±3),(4,±4)). Ported from
# `XLALSpinWeightedSphericalHarmonic` (`lal/lib/utilities/SphericalHarmonics.c` lines 65-166,
# 418-421): `Y_lm(theta,phi) = fac_lm(theta) * exp(i*m*phi)`. The l=2 formulas duplicate (rather
# than reuse) `IMRPhenomXPrecession.jl`'s own inline `Y2m2`..`Y2p2` local variables, since those are
# local variables inside `_xp_twist_up22`, not standalone reusable functions, and that file's own
# convention hardcodes `phi=0` (this file needs general `phi`, see `_xhm_add_mode!` below).
# ================================================================================================

"""
    _xhm_swsh_fac(l, m, theta)

`fac` in `XLALSpinWeightedSphericalHarmonic`'s own notation (the real, `phi`-independent part of
`(-2)Y_lm(theta,phi)`) for EVERY `(l,m)` with `l in {2,3,4}`, `|m| <= l` (21 combinations).
Dispatches via `if`/`elseif` over `Integer` `(l,m)` literals, matching `_xhm_fring_fdamp`'s own
established convention (see that function's docstring) -- `l`/`m` are always small compile-time-
known-ish literals at this file's own call sites, never AD targets.

This file's own 5-mode aligned-spin sum ([`_xhm_add_mode!`](@ref)) only ever needs the 10
combinations `(2,±1),(2,±2),(3,±2),(3,±3),(4,±4)`. The remaining 11 (`(2,0)`, `(3,-1),(3,0),(3,1)`,
`(4,-3)..(4,3)`) exist for `IMRPhenomXPHM.jl`'s twisting-up, whose per-mode transfer-function sums
run over the FULL `m = -l..l` range for each `(l,mprime)` (`LALSimIMRPhenomXPHM.c`'s `Y2mA[5]`/
`Y3mA[7]`/`Y4mA[9]` tables) -- they live here rather than there so all of this package's
spin-weighted spherical harmonics stay in one place. All 21 transcribed from
`XLALSpinWeightedSphericalHarmonic`'s `s == -2` branch (`lal/lib/utilities/SphericalHarmonics.c`
lines 65-166).
"""
@inline function _xhm_swsh_fac(l::Integer, m::Integer, theta)
    T = typeof(theta)
    ct = cos(theta)
    ch = cos(theta / 2)
    sh = sin(theta / 2)
    st = sin(theta)
    if l == 2 && m == -2
        return sqrt(T(5.0) / (T(64.0) * T(pi))) * (one(T) - ct) * (one(T) - ct)
    elseif l == 2 && m == -1
        return sqrt(T(5.0) / (T(16.0) * T(pi))) * st * (one(T) - ct)
    elseif l == 2 && m == 0
        return sqrt(T(15.0) / (T(32.0) * T(pi))) * st * st
    elseif l == 2 && m == 1
        return sqrt(T(5.0) / (T(16.0) * T(pi))) * st * (one(T) + ct)
    elseif l == 2 && m == 2
        return sqrt(T(5.0) / (T(64.0) * T(pi))) * (one(T) + ct) * (one(T) + ct)
    elseif l == 3 && m == -3
        return sqrt(T(21.0) / (T(2.0) * T(pi))) * ch * sh^5
    elseif l == 3 && m == -2
        return sqrt(T(7.0) / (T(4.0) * T(pi))) * (T(2.0) + T(3.0) * ct) * sh^4
    elseif l == 3 && m == -1
        return sqrt(T(35.0) / (T(2.0) * T(pi))) * (st + T(4.0) * sin(T(2.0) * theta) - T(3.0) * sin(T(3.0) * theta)) / T(32.0)
    elseif l == 3 && m == 0
        return sqrt(T(105.0) / (T(2.0) * T(pi))) * ct * st * st / T(4.0)
    elseif l == 3 && m == 1
        return -sqrt(T(35.0) / (T(2.0) * T(pi))) * (st - T(4.0) * sin(T(2.0) * theta) - T(3.0) * sin(T(3.0) * theta)) / T(32.0)
    elseif l == 3 && m == 2
        return sqrt(T(7.0) / T(pi)) * ch^4 * (-T(2.0) + T(3.0) * ct) / T(2.0)
    elseif l == 3 && m == 3
        return -sqrt(T(21.0) / (T(2.0) * T(pi))) * ch^5 * sh
    elseif l == 4 && m == -4
        return T(3.0) * sqrt(T(7.0) / T(pi)) * ch^2 * sh^6
    elseif l == 4 && m == -3
        return T(3.0) * sqrt(T(7.0) / (T(2.0) * T(pi))) * ch * (one(T) + T(2.0) * ct) * sh^5
    elseif l == 4 && m == -2
        return T(3.0) * (T(9.0) + T(14.0) * ct + T(7.0) * cos(T(2.0) * theta)) * sh^4 / (T(4.0) * sqrt(T(pi)))
    elseif l == 4 && m == -1
        return T(3.0) * (T(3.0) * st + T(2.0) * sin(T(2.0) * theta) + T(7.0) * sin(T(3.0) * theta) - T(7.0) * sin(T(4.0) * theta)) /
               (T(32.0) * sqrt(T(2.0) * T(pi)))
    elseif l == 4 && m == 0
        return T(3.0) * sqrt(T(5.0) / (T(2.0) * T(pi))) * (T(5.0) + T(7.0) * cos(T(2.0) * theta)) * st * st / T(16.0)
    elseif l == 4 && m == 1
        return T(3.0) * (T(3.0) * st - T(2.0) * sin(T(2.0) * theta) + T(7.0) * sin(T(3.0) * theta) + T(7.0) * sin(T(4.0) * theta)) /
               (T(32.0) * sqrt(T(2.0) * T(pi)))
    elseif l == 4 && m == 2
        return T(3.0) * ch^4 * (T(9.0) - T(14.0) * ct + T(7.0) * cos(T(2.0) * theta)) / (T(4.0) * sqrt(T(pi)))
    elseif l == 4 && m == 3
        return -T(3.0) * sqrt(T(7.0) / (T(2.0) * T(pi))) * ch^5 * (-one(T) + T(2.0) * ct) * sh
    elseif l == 4 && m == 4
        return T(3.0) * sqrt(T(7.0) / T(pi)) * ch^6 * sh^2
    else
        # (a constant message: interpolating l, m does not compile in a device kernel where (l, m) is not folded --
        # GravitationalWaveBayes' XPHM per-walker kernel)
        throw(ArgumentError("_xhm_swsh_fac: unsupported (l, m): l = 2..4, |m| <= l"))
    end
end

"""
    _xhm_swsh(l, m, theta, phi)

`(-2)Y_lm(theta, phi) = _xhm_swsh_fac(l,m,theta) * exp(i*m*phi)` (`SphericalHarmonics.c` line 419,
`cpolar(1.0, m*phi)`).
"""
@inline function _xhm_swsh(l::Integer, m::Integer, theta, phi)
    T = promote_type(typeof(theta), typeof(phi))
    fac = _xhm_swsh_fac(l, m, theta)
    return Complex{T}(fac * cos(m * phi), fac * sin(m * phi))
end

# ================================================================================================
# Public API
# ================================================================================================

"""
    IMRPhenomXHM

Model-dispatch marker for `IMRPhenomXHM` (aligned-spin, 5-harmonic `(2,2),(2,1),(3,3),(3,2),(4,4)`
higher-mode model, no precession -- see this file's module docstring for the full scope and the
`Ampl`/`Phi`-vs-`(hp,hc)` design decision). Does NOT participate in `Ampl(model,...)`/`Phi(model,...)`
for the same reason `IMRPhenomXP` doesn't (see that struct's own docstring): there is no single
frequency-dependent phase that describes a genuine sum over independently-evolving modes. Use
[`h_lm`](@ref) for a single mode's complex value, or [`hphc`](@ref) for the actual `(hp, hc)` sum.
"""
struct IMRPhenomXHM end

const _XHM_MODES = ((2, 1), (3, 3), (3, 2), (4, 4))

"""
    _xhm_mc_eta(m1, m2)

Chirp mass and symmetric mass ratio from component masses (`m1 >= m2`, this package's `chi1`-is-
`m1`'s-spin convention throughout, same as `IMRPhenomXAS`/`IMRPhenomXP`).
"""
@inline function _xhm_mc_eta(m1, m2)
    T = promote_type(typeof(m1), typeof(m2))
    Mtot = m1 + m2
    eta = min(m1 * m2 / (Mtot * Mtot), T(0.25))   # rounding can give 0.25 + ulp near q = 1 (LAL nudges it back too)
    mc = eta^(T(3.0) / T(5.0)) * Mtot
    return mc, eta
end

"""
    h_lm(model::IMRPhenomXHM, l, m, f, m1, m2, chi1, chi2, dL; f_ref, phiRef, GMsun_over_c3, GMsun_over_c2_Gpc)

Single-mode complex frequency-domain strain `h_{l,-m}(f)` -- see this file's module docstring
("Working convention") for why this returns the `-m` mode for a POSITIVE `m` argument, matching
`XLALSimIMRPhenomXHMGenerateFDOneMode`'s own internal convention exactly (the natural quantity to
validate directly against LALSuite's own per-mode generator). `(l,m)` must be one of `(2,2),(2,1),
(3,3),(3,2),(4,4)` (this port's scope, matching LALSuite's own default `ModeArray`); anything else
throws `ArgumentError`. `f` may be a scalar (returns a scalar `Complex`) or an `AbstractVector`
(returns a same-length `Vector{Complex{T}}`), matching this package's `Ampl`/`Phi`/`twist_up22`
scalar-vs-array convention. `m1 >= m2`, `chi1` is the spin on `m1` (this package's convention
throughout); `dL` in Gpc.
"""
function h_lm(model::IMRPhenomXHM, l::Integer, m::Integer, f, m1, m2, chi1, chi2, dL;
    f_ref, phiRef = zero(promote_type(typeof(m1), typeof(m2), typeof(chi1), typeof(chi2))),
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    m > 0 || throw(ArgumentError("h_lm: m must be positive (this file's convention returns h_{l,-m} for positive m -- see module docstring)"))
    (l, m) == (2, 2) || (l, m) in _XHM_MODES ||
        throw(ArgumentError("h_lm: unsupported (l,m) = ($l,$m); only (2,2),(2,1),(3,3),(3,2),(4,4) are implemented"))

    mc, eta = _xhm_mc_eta(m1, m2)
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phiRef;
        GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc)

    if (l, m) == (2, 2)
        T = sys.T
        if f isa AbstractArray
            out = similar(f, Complex{T})
            @inbounds for i in eachindex(f)
                Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * T(f[i])
                amp = _xas_ampl_scalar(sys.amplcoeffs22, T(f[i]))
                phi = _xhm22_phase_full(sys.coeffs22, sys.timeshift, sys.phiref22, Mf)
                out[i] = T(_XP_HAS_MODE_NORMALIZATION) * Complex{T}(amp * cos(phi), amp * sin(phi))
            end
            return out
        else
            Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * T(f)
            amp = _xas_ampl_scalar(sys.amplcoeffs22, T(f))
            phi = _xhm22_phase_full(sys.coeffs22, sys.timeshift, sys.phiref22, Mf)
            return T(_XP_HAS_MODE_NORMALIZATION) * Complex{T}(amp * cos(phi), amp * sin(phi))
        end
    elseif (l, m) == (3, 2)
        m32 = _xhm_mode32_setup(sys)
        return f isa AbstractArray ? map(fi -> _xhm_hlm_32(fi, sys, m32), f) : _xhm_hlm_32(f, sys, m32)
    else
        ms = _xhm_mode_setup(l, m, sys)
        return f isa AbstractArray ? map(fi -> _xhm_hlm_nonmixing(l, fi, sys, ms), f) : _xhm_hlm_nonmixing(l, f, sys, ms)
    end
end

"""
    _xhm_add_mode!(hp, hc, l, m, hlm, theta, phi)

`IMRPhenomXHMFDAddMode`'s `sym=1` branch (`LALSimIMRPhenomXHM.c` lines 2306-2327): given `hlm =
h_{l,-m}(f)` (a single value, positive `m`), accumulates its contribution AND its `+m` partner's
(reconstructed via `h_{l,+m} = (-1)^l * conj(h_{l,-m})`) into `hp`/`hc` at a FIXED `phi = pi/2`
azimuthal angle (`LAL_PI_2`, hardcoded at every call site in `IMRPhenomXHM_MultiMode` -- NOT
`phiRef`, which instead enters through each mode's own `phi0`-dependent phase, see module
docstring). Returns the updated `(hp, hc)` pair rather than mutating in place (this package's
scalar/broadcast convention has no natural in-place target here, unlike LALSuite's own
pre-allocated `COMPLEX16FrequencySeries`).
"""
@inline function _xhm_add_mode!(hp, hc, l::Integer, m::Integer, hlm, theta, phi)
    minus1l = isodd(l) ? -1 : 1
    Ym = _xhm_swsh(l, -m, theta, phi)
    Ystar = conj(_xhm_swsh(l, m, theta, phi))
    factorp = (Ym + minus1l * Ystar) / 2
    factorc = im * (Ym - minus1l * Ystar) / 2
    return hp + factorp * hlm, hc + factorc * hlm
end

"""
    _xhm_hphc_scalar(fi, sys, ms21, ms33, ms44, m32, inclination, phi)

Non-allocating (beyond the per-mode `Complex` returns) `(hp, hc)` evaluation at a single frequency
`fi`, given every mode's pre-computed coefficient bundle. Factored out of [`hphc`](@ref) as a plain
top-level function (not a closure) so both the scalar and array branches share it without capturing
`sys`/`ms21`/etc. in a closure -- matching this package's general avoidance of closures on anything
resembling a per-frequency hot path (see [`_xhm_s2s`](@ref)'s own docstring for the same rationale
applied to `(3,2)`'s mode-mixing rotation).
"""
@inline function _xhm_hphc_scalar(fi, sys, ms21, ms33, ms44, m32, inclination, phi)
    T = sys.T
    Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * T(fi)
    amp22 = _xas_ampl_scalar(sys.amplcoeffs22, T(fi))
    phi22 = _xhm22_phase_full(sys.coeffs22, sys.timeshift, sys.phiref22, Mf)
    h22 = T(_XP_HAS_MODE_NORMALIZATION) * Complex{T}(amp22 * cos(phi22), amp22 * sin(phi22))
    h21 = _xhm_hlm_nonmixing(2, T(fi), sys, ms21)
    h33 = _xhm_hlm_nonmixing(3, T(fi), sys, ms33)
    h44 = _xhm_hlm_nonmixing(4, T(fi), sys, ms44)
    h32 = _xhm_hlm_32(T(fi), sys, m32)

    hp = zero(Complex{T}); hc = zero(Complex{T})
    hp, hc = _xhm_add_mode!(hp, hc, 2, 2, h22, inclination, phi)
    hp, hc = _xhm_add_mode!(hp, hc, 2, 1, h21, inclination, phi)
    hp, hc = _xhm_add_mode!(hp, hc, 3, 3, h33, inclination, phi)
    hp, hc = _xhm_add_mode!(hp, hc, 3, 2, h32, inclination, phi)
    hp, hc = _xhm_add_mode!(hp, hc, 4, 4, h44, inclination, phi)
    return hp, hc
end

"""
    hphc(model::IMRPhenomXHM, f, m1, m2, chi1, chi2, dL, inclination, phiRef; f_ref, GMsun_over_c3, GMsun_over_c2_Gpc)

End-to-end `IMRPhenomXHM` waveform: `(hp, hc)` summed over `(2,2),(2,1),(3,3),(3,2),(4,4)` (and
their `-m` partners, reconstructed via aligned-spin symmetry -- see [`_xhm_add_mode!`](@ref)),
matching `IMRPhenomXHM_MultiMode`'s own default-`ModeArray` behaviour (`LALSimIMRPhenomXHM.c` lines
1793-1911). `f` may be a scalar (returns a scalar `(hp, hc)` tuple) or an `AbstractVector` (returns
same-length `Vector{Complex{T}}`s), matching `twist_up22`'s own convention. `m1 >= m2`, `chi1` is
the spin on `m1`; `dL` in Gpc; `inclination` is the angle between the orbital angular momentum and
the line of sight (this is a non-precessing model, so `L` is fixed -- unlike `IMRPhenomXP`'s
`inclination`-at-`f_ref` convention, there is no frame-precession subtlety here); `phiRef` is the
reference orbital phase at `f_ref`.
"""
function hphc(model::IMRPhenomXHM, f, m1, m2, chi1, chi2, dL, inclination, phiRef;
    f_ref,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    mc, eta = _xhm_mc_eta(m1, m2)
    sys = _xhm_system_setup(mc, eta, chi1, chi2, dL, f_ref, phiRef;
        GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc)
    T = sys.T
    piHalf = T(pi) / T(2.0)
    inclinationT = T(inclination)

    ms21 = _xhm_mode_setup(2, 1, sys)
    ms33 = _xhm_mode_setup(3, 3, sys)
    ms44 = _xhm_mode_setup(4, 4, sys)
    m32 = _xhm_mode32_setup(sys)

    if f isa AbstractArray
        hp = similar(f, Complex{T})
        hc = similar(f, Complex{T})
        @inbounds for i in eachindex(f)
            hp[i], hc[i] = _xhm_hphc_scalar(f[i], sys, ms21, ms33, ms44, m32, inclinationT, piHalf)
        end
        return hp, hc
    else
        return _xhm_hphc_scalar(f, sys, ms21, ms33, ms44, m32, inclinationT, piHalf)
    end
end
