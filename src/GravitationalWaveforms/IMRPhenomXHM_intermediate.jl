"""
    IMRPhenomXHM intermediate-region amplitude/phase ansatze (this file)

The intermediate-region amplitude AND phase
ansatze for the `(l,m) = (2,1), (3,3), (3,2), (4,4)` higher harmonics that `IMRPhenomXHM` adds on
top of `IMRPhenomXAS`'s dominant `(2,2)` mode. Ported from LALSuite's
`LALSimIMRPhenomXHM_intermediate.c`/`.h`, plus the small amount of `LALSimIMRPhenomXHM_internals.c`
glue needed to evaluate these ansatze end-to-end (as in `IMRPhenomXHM_inspiral.jl`). Ringdown ansatze (`IMRPhenomXHM_ringdown.jl`) and mode-mixing/full assembly (`IMRPhenomXHM.jl`) are in their own files.

# A transcription pitfall
`_xhm_inter_amp_44_int2` originally contained a copy-paste error: its body was accidentally the
formula for `IMRPhenomXHM_inspiral.jl`'s `_xhm_insp_amp_44_iv2` (a DIFFERENT function, from the
neighbouring inspiral-region file, with the `sqroot` factors dropped) rather than
`IMRPhenomXHM_Inter_Amp_44_int2`'s own formula. This produced a `(4,4)`-mode-only, wildly wrong
intermediate collocation-point value (e.g. `1.17` where its immediate neighbours were `~0.3-0.4`),
which sent the degree-7 reconstruction polynomial into large-amplitude oscillation between
collocation points (a `~50-100%` relative-amplitude error across the interior of the band) despite
EXACTLY reproducing every one of its own (wrong) fitting constraints -- i.e. the bug was invisible
to any test that only checks "does the solved polynomial pass through its own inputs" (this
package's own such self-consistency unit tests all still passed with the bug present) and was only
caught by comparing against real LALSuite's independent numbers. Fixed by re-transcribing directly
from the actual `IMRPhenomXHM_Inter_Amp_44_int2` C source (`LALSimIMRPhenomXHM_intermediate.c`
lines 310-345) and re-verified with an automated numeric-literal cross-check against the C source
for every one of this file's 16 amplitude (`int1`-`int4`) and 24 phase (`p1`-`p6`) collocation-fit
functions (every remaining flagged "diff" was confirmed a benign extraction-tool artifact --
`pow(x,-1)` vs `x^(-1.0)` formatting, or `delta`/`sqrt(1-4eta)` computed once and reused in Julia
vs written inline multiple times in the C source -- not a real numeric discrepancy). This is
found by comparing directly with LALSuite.

# THE BIGGEST FINDING OF THIS PORT: the "delta0-delta5" closed-form solve is DEAD CODE under the
# actual default release, and this port does NOT implement it.
#
# The task's own source map (based on a first read of the C file) pointed at
# `IMRPhenomXHM_Intermediate_Amp_delta0`-`_delta5` (`LALSimIMRPhenomXHM_intermediate.c` lines
# ~728-1900, ~1170 lines of dense, mechanically-expanded 5th-order-polynomial-coefficient algebra
# with 8 different sub-cases each) as "the largest chunk" and the highest transcription-risk part
# of this file. Reading `LALSimIMRPhenomXHM_internals.c`'s actual call graph shows this is
# **entirely unreachable under the real default configuration**:
#
#   `IMRPhenomXHM_GetAmplitudeCoefficients` (`_internals.c` line 1146) branches on
#   `pWFHM->IMRPhenomXHMReleaseVersion`:
#     - `if (... != 122019)` (TRUE under the actual default 122022, as in the inspiral file): calls
#       `IMRPhenomXHM_Intermediate_Amp_Coefficients` (`_intermediate.c` line 2538), which sets up
#       and solves a **general N x N linear system via GSL LU decomposition** (`A x = b`, with
#       `A[row,j] = f_row^j * f_row^{-7/6}` for a value row or its frequency-derivative for a
#       derivative row) for coefficients `InterCoefficient[0..N-1]` of a **direct** polynomial
#       ansatz `Ansatz(f) = f^{-7/6} * sum_j InterCoefficient[j] * f^j`
#       (`IMRPhenomXHM_Intermediate_Amp_Ansatz`, `_intermediate.c` lines 1900-1911, the
#       `ReleaseVersion != 122019` branch).
#     - `else` (only `ReleaseVersion == 122019`, the OLD/legacy release, never hit by default):
#       this is the ONLY branch that calls `Update_Intermediate_Amplitude_Coefficients`,
#       `ChoosePolOrder`, and therefore the delta0-delta5 functions and the *reciprocal* ansatz
#       form `Ansatz(f) = ampNorm / polynomial(f)` (`_intermediate.c` lines 1913-1986, the
#       `ReleaseVersion == 122019` branch of `IMRPhenomXHM_Intermediate_Amp_Ansatz`).
#
# So under the real default configuration, delta0-delta5/ChoosePolOrder/
# `Update_Intermediate_Amplitude_Coefficients`/`IMRPhenomXHM_Intermediate_Amplitude_Veto`/
# `WavyPoints`/`CrossZeroP{2,3,4,5}` are ALL dead code -- a much bigger and more central instance
# of the "122019-only dead branch" pattern also found in the neighbouring inspiral file. **None of it is ported here.** The real intermediate
# amplitude ansatz actually used is a plain generic linear system solve, which this file implements
# using `IMRPhenomXAS.jl`'s own already-existing, already-validated generic Cramer's-rule solver
# (`_xas_linsolve`/`_xas_det`) -- reused directly rather than duplicated, exactly matching this
# package's established DRY convention (see `IMRPhenomXHM_inspiral.jl`'s reuse of
# `_xas_phi_coeffs`).
#
# The amplitude collocation-point-count digit code (`IMRPhenomXHMIntermediateAmpVersion`,
# `_internals.c` lines 312-341) is **211112** for `(3,3),(3,2),(4,4)` (6 "slots": left boundary
# value+derivative, `int1,int2,int3,int4` values, right boundary value+derivative = 8 unknowns,
# 8-term polynomial) and **110102** for `(2,1)` specifically ("the 21 behaves better if we skip the
# left derivative... we also drop two collocation points in the middle": left boundary VALUE ONLY,
# `int1` and `int3` only (`int2`,`int4` skipped), right boundary value+derivative = 5 unknowns,
# 5-term polynomial). Confirmed directly from `_internals.c`'s own inline comment documenting this
# digit encoding.
#
# # Phase intermediate ansatz: NOT release-version-switched (dead-code-free), but genuinely
# # depends on mode-mixing (`IMRPhenomXHM.jl`) for the `(3,2)` mode specifically
# Unlike amplitude, the phase collocation-point fits (`p1`-`p6`) have only ONE version in the C
# source (`case 122019`, with every function's own `default` branch literally saying "Recommended
# version is 122019") -- matching `_internals.c`'s own comment "HM Phase is unchanged respect to
# the old release" (line 354). So phase fits are NOT dead code; they are exactly what both release
# versions actually use.
#
# However, `IMRPhenomXHM_GetPhaseCoefficients` (`_internals.c` lines 1952-2464) branches the
# **coefficient-solve algorithm itself** on `pWFHM->MixingOn` (NOT on `ReleaseVersion`):
#   - `case 0` (`(2,1),(3,3),(4,4)` -- `MixingOn` is off for these per `_internals.c`'s own
#     comment "the MixingOn flag is automatically switched off for the modes 21,33,44"): solves a
#     plain 5x5 linear system directly from the `p1`-`p6` collocation-point fits (5 of the 6
#     selected via `cpoints_indices`, a small mass-ratio/spin-dependent selection rule) for
#     `(c0,cL,c1,c2,c4)` (`c3` stays 0). **Fully ported here** for these 3 modes.
#   - `case 1` (`(3,2)` only -- `MixingOn` "is automatically switched on when reconstructing the 32
#     mode"): solves a DIFFERENT 6x6 system that requires `SpheroidalToSpherical` (spherical-
#     spheroidal mode-mixing rotation, `_internals.c` line 2194) to obtain 2 of its 6 constraints
#     (a finite-difference 1st/2nd phase derivative of the ROTATED spheroidal-to-spherical
#     waveform at the ringdown-matching frequency) -- that machinery is in `IMRPhenomXHM.jl`.
#     **NOT ported here** -- see [`_xhm_inter_phase_coefficients`](@ref)'s
#     docstring; calling it for `(3,2)` throws `ArgumentError` rather than silently returning wrong
#     numbers. `(3,2)`'s `p1`-`p6` collocation-point FITS themselves (pure parameter-space fits,
#     mode-mixing-independent) ARE ported below.
#
# This same `MixingOn`-on-`(3,2)`-only mode-mixing dependency ALSO affects the amplitude ringdown
# BOUNDARY VALUE for `(3,2)` specifically (`IMRPhenomXHM_Intermediate_Amp_CollocationPoints`,
# `_intermediate.c` lines 2508-2513: `if (pWFHM->MixingOn == 0) ... IMRPhenomXHM_RD_Amp_Ansatz(...)
# else ... cabs(SpheroidalToSpherical(...))`) -- but since this port's amplitude coefficient solve
# already takes the ringdown boundary value/derivative `(V4, d4)` as caller-supplied numbers (see
# below, "boundary values this port does NOT compute"), the amplitude STRUCTURE is mode-agnostic
# and `(3,2)` amplitude support does not additionally require `IMRPhenomXHM.jl` -- only `(3,2)` PHASE does.

# Boundary values this file does NOT compute
The intermediate region's polynomial reconstruction needs the amplitude value (and, for most
modes, derivative) at BOTH edges: `F1` (inspiral/intermediate boundary, from
`IMRPhenomXHM_inspiral.jl`) and `F4` (intermediate/ringdown boundary, from
`IMRPhenomXHM_ringdown.jl`). Likewise the phase gluing constants `C1RD`/`CRD` need the
ringdown phase value/derivative at the same boundary. The
ringdown side lives elsewhere; **every function in this file that would need it takes the boundary
value(s) as an explicit numeric argument instead of computing them internally**
(`F4, V4, d4` for amplitude; `dphi0RD, phi0RD` for phase C1-continuity gluing) -- exactly mirroring
how `IMRPhenomXAS.jl`'s own intermediate-amplitude delta-solve takes `d1, d4` as pre-computed
numbers (there both sides happen to already be implemented in that same file, but the calling
convention -- boundary conditions in, coefficients out -- is the same). The inspiral-side boundary
(`F1, V1, d1`) IS computed internally here, using `IMRPhenomXHM_inspiral.jl`'s
`_xhm_insp_amp_ansatz`/`_xhm_insp_amp_nd_ansatz`.

# What this port deliberately does NOT implement (beyond the `(3,2)` phase boundary above)
- The `int0`/`dint0` "extra EMR collocation point" amplitude fits: reading the C source shows these
  have ONLY a `case 122018` branch (no `case 122022` at all -- calling them under the actual
  default release errors out in real LALSuite itself, `XLAL_ERROR_REAL8` on any other version).
  They are wired up only inside the `AmpEMR==1` block of the legacy (`ReleaseVersion==122019`)
  branch of `IMRPhenomXHM_GetAmplitudeCoefficients` (`_internals.c` lines 1694-1773) -- itself
  inside the same dead `else` branch as delta0-delta5. **Not ported.**
- The `(2,1)`, high-spin (`STotR >= 0.8`) PN-based override of the two innermost phase collocation
  VALUES (`_internals.c` lines 2078-2093, using `IMRPhenomX_dPhase_22`) -- a narrow special case
  affecting only near-extremal-spin `(2,1)` configurations. The `cpoints_indices` FREQUENCY
  selection logic that this same high-spin regime also triggers (`STotR>=0.8 && modeTag==21`,
  `_internals.c` line 2112) IS ported (cheap, a plain index swap). The value
  override is ported too (`_xhm_inter_phase_highspin21`, `_internals.c` lines 2076-2093); it needs the
  (2,2) coefficients, so [`_xhm_inter_phase_coefficients`](@ref) still throws for that case when
  called without `coeffs22`.
- The extreme-mass-ratio branches gated by `pWF22->eta < pWFHM->etaEMR` (`etaEMR=0.05`, i.e.
  roughly `q >~ 17`) in `IMRPhenomXHM_GetPhaseCoefficients`'s `cpoints_indices` selection and the
  `fcutInsp`/`fcutRD` EMR formulas -- out of this port's validated mass-ratio range (near-equal to
  GW190814-like ~1:9, `eta` down to ~0.09, comfortably above 0.05). The `cpoints_indices`
  SELECTION swap for `eta<etaEMR` is nonetheless ported (again, cheap and low-risk) but not
  exercised by the LALSuite validation grid.

# `(l,m)` <-> `emm`/`modeInt` bookkeeping: identical table to `IMRPhenomXHM_inspiral.jl`'s module
docstring (`(2,1)->modeInt=0,emm=1`; `(3,3)->modeInt=1,emm=3`; `(3,2)->modeInt=2,emm=2`;
`(4,4)->modeInt=3,emm=4`).
"""

# ================================================================================================
# AMPLITUDE: intermediate collocation-point parameter-space fits (`int1`-`int4`, `case 122022`
# only -- see module docstring for why `int0`/`dint0` are not ported). Ported from
# `IMRPhenomXHM_Inter_Amp_{21,33,32,44}_int{1,2,3,4}`, `LALSimIMRPhenomXHM_intermediate.c` lines
# 49-707. Arguments match the C source's own local variables per function (`S` is `pWF->STotR` for
# some functions and `pWF->chiPNHat` for others -- confirmed individually per function by reading
# the C source directly, NOT assumed uniform; see each function's own comment below).
# ================================================================================================

@inline function _xhm_inter_amp_21_int1(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return abs(
            delta * eta1 * (chidiff2 * (5.159755997682368 * eta1 - 30.293198248154948 * eta2 + 63.70715919820867 * eta3) + chidiff1 * (8.262642080222694 * eta1 - 415.88826990259116 * eta2 + 1427.5951158851076 * eta3)) +
            delta * eta1 * (18.55363583212328 - 66.46950491124205 * eta1 + 447.2214642597892 * eta2 - 1614.178472020212 * eta3 + 2199.614895727586 * eta4) +
            chidiff1 * eta5 * (-1698.841763891122 - 195.27885562092342 * S1 - 1.3098861736238572 * S2) +
            delta * eta1 * (chidiff1 * (34.17829404207186 * eta1 - 386.34587928670015 * eta2 + 1022.8553774274128 * eta3) * S1 + chidiff1 * (56.76554600963724 * eta1 - 491.4593694689354 * eta2 + 1016.6019654342113 * eta3) * S2) +
            delta * eta1 * S1 * (
                -8.276366844994188 * (1.0677538075697492 - 24.12941323757896 * eta1 + 516.7886322104276 * eta2 - 4389.799658723288 * eta3 + 16770.447637953577 * eta4 - 23896.392706809565 * eta5) -
                1.6908277400304084 * (3.4799140066657928 - 29.00026389706585 * eta1 + 114.8330693231833 * eta2 - 184.13091281984674 * eta3 + 592.300353344717 * eta4 - 2085.0821513466053 * eta5) * S1 -
                0.46006975902558517 * (-2.1663474937625975 + 826.026625945615 * eta1 - 17333.549622759732 * eta2 + 142904.08962903373 * eta3 - 528521.6231015554 * eta4 + 731179.456702448 * eta5) * S2
            )
        )
    end
end

@inline function _xhm_inter_amp_21_int2(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        return abs(
            delta * eta1 * (13.757856231617446 - 12.783698329428516 * eta1 + 12.048194546899204 * eta2) +
            chidiff1 * delta * eta1 * (15.107530092096438 * eta1 - 416.811753638553 * eta2 + 1333.6181181686939 * eta3) +
            chidiff1 * eta5 * (-1549.6199518612063 - 102.34716990474509 * S1 - 3.3637011939285015 * S2) +
            delta * eta1 * (chidiff1 * (36.358142200869295 * eta1 - 384.2123173145321 * eta2 + 984.6826660818275 * eta3) * S1 + chidiff1 * (4.159271594881928 * eta1 + 105.10911749116399 * eta2 - 639.190132707115 * eta3) * S2) +
            delta * eta1 * S1 * (
                -8.097876227116853 * (0.6569459700232806 + 9.861355377849485 * eta1 - 116.88834714736281 * eta2 + 593.8035334117192 * eta3 - 1063.0692862578455 * eta4) -
                1.0546375154878165 * (0.745557030602097 + 65.25215540635162 * eta1 - 902.5751736558435 * eta2 + 4350.442990924205 * eta3 - 7141.611333893155 * eta4) * S1 -
                0.5006664599166409 * (10.289020582277626 - 212.00728173197498 * eta1 + 2334.0029399672358 * eta2 - 11939.621138801092 * eta3 + 21974.8201355744 * eta4) * S2
            )
        )
    end
end

@inline function _xhm_inter_amp_21_int3(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        return abs(
            delta * eta1 * (13.318990196097973 - 21.755549987331054 * eta1 + 76.14884211156267 * eta2 - 127.62161159798488 * eta3) +
            chidiff1 * delta * eta1 * (17.704321326939414 * eta1 - 434.4390350012534 * eta2 + 1366.2408490833282 * eta3) +
            chidiff1 * delta * eta1 * (11.877985158418596 * eta1 - 131.04937626836355 * eta2 + 343.79587860999874 * eta3) * S1 +
            chidiff1 * eta5 * (-1522.8543551416456 - 16.639896279650678 * S1 + 3.0053086651515843 * S2) +
            delta * eta1 * S1 * (
                -8.665646058245033 * (0.7862132291286934 + 8.293609541933655 * eta1 - 111.70764910503321 * eta2 + 576.7172598056907 * eta3 - 1001.2370065269745 * eta4) -
                0.9459820574514348 * (1.309016452198605 + 48.94077040282239 * eta1 - 817.7854010574645 * eta2 + 4331.56002883546 * eta3 - 7518.309520232795 * eta4) * S1 -
                0.4308267743835775 * (9.970654092010587 - 302.9708323417439 * eta1 + 3662.099161055873 * eta2 - 17712.883990278668 * eta3 + 29480.158198408903 * eta4) * S2
            )
        )
    end
end

@inline function _xhm_inter_amp_21_int4(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        return abs(
            delta * eta1 * (13.094382343446163 - 22.831152256559523 * eta1 + 83.20619262213437 * eta2 - 139.25546924151664 * eta3) +
            chidiff1 * delta * eta1 * (20.120192352555357 * eta1 - 458.2592421214168 * eta2 + 1430.3698681181 * eta3) +
            chidiff1 * delta * eta1 * (12.925363020014743 * eta1 - 126.87194512915104 * eta2 + 280.6003655502327 * eta3) * S1 +
            chidiff1 * eta5 * (-1528.956015503355 + 74.44462583487345 * S1 - 2.2456928156392197 * S2) +
            delta * eta1 * S1 * (
                -9.499741513411829 * (0.912120958549489 + 2.400945118514037 * eta1 - 33.651192908287236 * eta2 + 166.04254881175257 * eta3 - 248.5050377498615 * eta4) -
                0.7850652143322492 * (1.534131218043425 + 60.81773903539479 * eta1 - 1032.1319480683567 * eta2 + 5381.481380750608 * eta3 - 9077.037917192794 * eta4) * S1 -
                0.21540359093306097 * (9.42805409480658 - 109.06544597367301 * eta1 + 385.8345793110262 * eta2 + 1889.9613367802453 * eta3 - 9835.416414460055 * eta4) * S2
            )
        )
    end
end

@inline function _xhm_inter_amp_33_int1(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        return chidiff1 * delta * eta1 * (-0.3516244197696068 * eta1 + 40.425151307421416 * eta2 - 148.3162618111991 * eta3) +
               delta * eta1 * (26.998512565991778 - 146.29035440932105 * eta1 + 914.5350366065115 * eta2 - 3047.513201789169 * eta3 + 3996.417635728702 * eta4) +
               chidiff1 * delta * eta1 * (5.575274516197629 * eta1 - 44.592719238427094 * eta2 + 99.91399033058927 * eta3) * S1 +
               delta * eta1 * S1 * (
                   -0.5383304368673182 * (-7.456619067234563 + 129.36947401891433 * eta1 - 843.7897535238325 * eta2 + 3507.3655567272644 * eta3 - 9675.194644814854 * eta4 + 11959.83533107835 * eta5) -
                   0.28042799223829407 * (-6.212827413930676 + 266.69059813274475 * eta1 - 4241.537539226717 * eta2 + 32634.43965039936 * eta3 - 119209.70783201039 * eta4 + 166056.27237509796 * eta5) * S1
               ) +
               chidiff1 * eta5 * (199.6863414922219 + 53.36849263931051 * S1 + 7.650565415855383 * S2)
    end
end

@inline function _xhm_inter_amp_33_int2(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        return delta * eta1 * (17.42562079069636 - 28.970875603981295 * eta1 + 50.726220750178435 * eta2) +
               chidiff1 * delta * eta1 * (-7.861956897615623 * eta1 + 93.45476935080045 * eta2 - 273.1170921735085 * eta3) +
               chidiff1 * delta * eta1 * (-0.3265505633310564 * eta1 - 9.861644053348053 * eta2 + 60.38649425562178 * eta3) * S1 +
               chidiff1 * eta5 * (234.13476431269862 + 51.2153901931183 * S1 - 10.05114600643587 * S2) +
               delta * eta1 * S1 * (
                   0.3104472390387834 * (6.073591341439855 + 169.85423386969634 * eta1 - 4964.199967099143 * eta2 + 42566.59565666228 * eta3 - 154255.3408672655 * eta4 + 205525.13910847943 * eta5) +
                   0.2295327944679772 * (19.236275867648594 - 354.7914372697625 * eta1 + 1876.408148917458 * eta2 + 2404.4151687877525 * eta3 - 41567.07396803811 * eta4 + 79210.33893514868 * eta5) * S1 +
                   0.30983324991828787 * (11.302200127272357 - 719.9854052004307 * eta1 + 13278.047199998868 * eta2 - 104863.50453518033 * eta3 + 376409.2335857397 * eta4 - 504089.07690692553 * eta5) * S2
               )
    end
end

@inline function _xhm_inter_amp_33_int3(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        return delta * eta1 * (14.555522136327964 - 12.799844096694798 * eta1 + 16.79500349318081 * eta2) +
               chidiff1 * delta * eta1 * (-16.292654447108134 * eta1 + 190.3516012682791 * eta2 - 562.0936797781519 * eta3) +
               chidiff1 * delta * eta1 * (-7.048898856045782 * eta1 + 49.941617405768135 * eta2 - 73.62033985436068 * eta3) * S1 +
               chidiff1 * eta5 * (263.5151703818307 + 44.408527093031566 * S1 + 10.457035444964653 * S2) +
               delta * eta1 * S1 * (
                   0.4590550434774332 * (3.0594364612798635 + 207.74562213604057 * eta1 - 5545.0086137386525 * eta2 + 50003.94075934942 * eta3 - 195187.55422847517 * eta4 + 282064.174913521 * eta5) +
                   0.657748992123043 * (5.57939137343977 - 124.06189543062042 * eta1 + 1276.6209573025596 * eta2 - 6999.7659193505915 * eta3 + 19714.675715229736 * eta4 - 20879.999628681435 * eta5) * S1 +
                   0.3695850566805098 * (6.077183107132255 - 498.95526910874986 * eta1 + 10426.348944657859 * eta2 - 91096.64982858274 * eta3 + 360950.6686625352 * eta4 - 534437.8832860565 * eta5) * S2
               )
    end
end

@inline function _xhm_inter_amp_33_int4(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return delta * eta1 * (13.312095699772305 - 7.449975618083432 * eta1 + 17.098576301150125 * eta2) +
               delta * eta1 * (chidiff1 * (-31.171150896110156 * eta1 + 371.1389274783572 * eta2 - 1103.1917047361735 * eta3) + chidiff2 * (32.78644599730888 * eta1 - 395.15713118955387 * eta2 + 1164.9282236341376 * eta3)) +
               chidiff1 * delta * eta1 * (-46.85669289852532 * eta1 + 522.3965959942979 * eta2 - 1485.5134187612182 * eta3) * S1 +
               chidiff1 * eta5 * (287.90444670305715 - 21.102665129433042 * chidiff2 + 7.635582066682054 * S1 - 29.471275170013012 * S2) +
               delta * eta1 * S1 * (
                   0.6893003654021495 * (3.1014226377197027 - 44.83989278653052 * eta1 + 565.3767256471909 * eta2 - 4797.429130246123 * eta3 + 19514.812242035154 * eta4 - 27679.226582207506 * eta5) +
                   0.7068016563068026 * (4.071212304920691 - 118.51094098279343 * eta1 + 1788.1730303291356 * eta2 - 13485.270489656365 * eta3 + 48603.96661003743 * eta4 - 65658.74746265226 * eta5) * S1 +
                   0.2181399561677432 * (-1.6754158383043574 + 303.9394443302189 * eta1 - 6857.936471898544 * eta2 + 59288.71069769708 * eta3 - 216137.90827404748 * eta4 + 277256.38289831823 * eta5) * S2
               )
    end
end

# 32 mode: `S` is `pWF->chiPNHat` for `int1`/`int3`, `pWF->STotR` for `int2`/`int4` -- confirmed
# individually by reading the C source (NOT a uniform convention across the 4 collocation points).
@inline function _xhm_inter_amp_32_int1(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return (chidiff2 * (-0.2341404256829785 * eta1 + 2.606326837996192 * eta2 - 8.68296921440857 * eta3) + chidiff1 * delta * (0.5454562486736877 * eta1 - 25.19759222940851 * eta2 + 73.40268975811729 * eta3)) * sqroot +
               chidiff1 * delta * (0.4422257616009941 * eta1 - 8.490112284851655 * eta2 + 32.22238925527844 * eta3) * S1 * sqroot +
               S1 * (0.7067243321652764 * (0.12885110296881636 + 9.608999847549535 * eta1 - 85.46581740280585 * eta2 + 325.71940024255775 * eta3 + 175.4194342269804 * eta4 - 1929.9084724384807 * eta5) +
                     0.1540566313813899 * (-0.3261041495083288 + 45.55785402900492 * eta1 - 827.591235943271 * eta2 + 7184.647314370326 * eta3 - 28804.241518798244 * eta4 + 43309.69769878964 * eta5) * S1) * sqroot +
               (480.0434256230109 * eta1 + 25346.341240810478 * eta2 - 99873.4707358776 * eta3 + 106683.98302194536 * eta4) * sqroot * (1.0 + 1082.6574834474493 * eta1 + 10083.297670051445 * eta2)^(-1.0)
    end
end

@inline function _xhm_inter_amp_32_int2(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return eta1 * (chidiff2 * (-4.175680729484314 * eta1 + 47.54281549129226 * eta2 - 128.88334273588077 * eta3) + chidiff1 * delta * (-0.18274358639599947 * eta1 - 71.01128541687838 * eta2 + 208.07105580635888 * eta3)) +
               eta1 * (4.760999387359598 - 38.57900689641654 * eta1 + 456.2188780552874 * eta2 - 4544.076411013166 * eta3 + 24956.9592553473 * eta4 - 69430.10468748478 * eta5 + 77839.74180254337 * eta6) +
               chidiff1 * delta * eta1 * (1.2198776533959694 * eta1 - 26.816651899746475 * eta2 + 68.72798751937934 * eta3) * S1 +
               eta1 * S1 * (1.5098291294292217 * (0.4844667556328104 + 9.848766999273414 * eta1 - 143.66427232396376 * eta2 + 856.9917885742416 * eta3 - 1633.3295758142904 * eta4) +
                            0.32413108737204144 * (2.835358206961064 - 62.37317183581803 * eta1 + 761.6103793011912 * eta2 - 3811.5047139343505 * eta3 + 6660.304740652403 * eta4) * S1)
    end
end

@inline function _xhm_inter_amp_32_int3(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return 3.881450518842405 * eta1 - 12.580316392558837 * eta2 + 1.7262466525848588 * eta3 +
               chidiff2 * (-7.065118823041031 * eta2 + 77.97950589523865 * eta3 - 203.65975422378446 * eta4) -
               58.408542930248046 * eta4 +
               chidiff1 * delta * (1.924723094787216 * eta2 - 90.92716917757797 * eta3 + 387.00162600306226 * eta4) +
               403.5748987560612 * eta5 +
               chidiff1 * delta * (-0.2566958540737833 * eta2 + 14.488550203412675 * eta3 - 26.46699529970884 * eta4) * S1 +
               S1 * (0.3650871458400108 * (71.57390929624825 * eta2 - 994.5272351916166 * eta3 + 6734.058809060536 * eta4 - 18580.859291282686 * eta5 + 16001.318492586077 * eta6) +
                     0.0960146077440495 * (451.74917589707513 * eta2 - 9719.470997418284 * eta3 + 83403.5743434538 * eta4 - 318877.43061174755 * eta5 + 451546.88775684836 * eta6) * S1 -
                     0.03985156529181297 * (-304.92981902871617 * eta2 + 3614.518459296278 * eta3 - 7859.4784979916085 * eta4 - 46454.57664737511 * eta5 + 162398.81483375572 * eta6) * S2)
    end
end

@inline function _xhm_inter_amp_32_int4(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return eta1 * (chidiff2 * (-8.572797326909152 * eta1 + 92.95723645687826 * eta2 - 236.2438921965621 * eta3) + chidiff1 * delta * (6.674358856924571 * eta1 - 171.4826985994883 * eta2 + 645.2760206304703 * eta3)) +
               eta1 * (3.921660532875504 - 16.57299637423352 * eta1 + 25.254017911686333 * eta2 - 143.41033155133266 * eta3 + 692.926425981414 * eta4) +
               chidiff1 * delta * eta1 * (-3.582040878719185 * eta1 + 57.75888914133383 * eta2 - 144.21651114700492 * eta3) * S1 +
               eta1 * S1 * (1.242750265695504 * (-0.522172424518215 + 25.168480118950065 * eta1 - 303.5223688400309 * eta2 + 1858.1518762309654 * eta3 - 3797.3561904195085 * eta4) +
                            0.2927045241764365 * (0.5056957789079993 - 15.488754837330958 * eta1 + 471.64047356915603 * eta2 - 3131.5783196211587 * eta3 + 6097.887891566872 * eta4) * S1)
    end
end

# 44 mode: `S` is `pWF->chiPNHat` for ALL FOUR collocation points (unlike 32).
@inline function _xhm_inter_amp_44_int1(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return eta1 * (chidiff1 * delta * (1.5378890240544967 * eta1 - 3.4499418893734903 * eta2 + 16.879953490422782 * eta3) + chidiff2 * (1.720226708214248 * eta1 - 11.87925165364241 * eta2 + 23.259283336239545 * eta3)) +
               eta1 * (8.790173464969538 - 64.95499142822892 * eta1 + 324.1998823562892 * eta2 - 1111.9864921907126 * eta3 + 1575.602443847111 * eta4) +
               eta1 * S1 * (-0.062333275821238224 * (-21.630297087123807 + 137.4395894877131 * eta1 + 64.92115530780129 * eta2 - 1013.1110639471394 * eta3) -
                            0.11014697070998722 * (4.149721483857751 - 108.6912882442823 * eta1 + 831.6073263887092 * eta2 - 1828.2527520190122 * eta3) * S1 -
                            0.07704777584463054 * (4.581767671445529 - 50.35070009227704 * eta1 + 344.9177692251726 * eta2 - 858.9168637051405 * eta3) * S2)
    end
end

@inline function _xhm_inter_amp_44_int2(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return eta1 * (chidiff1 * delta * (2.3123974306694057 * eta1 - 12.237594841284904 * eta2 + 44.78225529547671 * eta3) + chidiff2 * (2.9282931698944292 * eta1 - 25.624210264341933 * eta2 + 61.05270871360041 * eta3)) +
               eta1 * (6.98072197826729 - 46.81443520117986 * eta1 + 236.76146303619544 * eta2 - 920.358408667518 * eta3 + 1478.050456337336 * eta4) +
               eta1 * S1 * (-0.07801583359561987 * (-28.29972282146242 + 752.1603553640072 * eta1 - 10671.072606753183 * eta2 + 83447.0461509547 * eta3 - 350025.2112501252 * eta4 + 760889.6919776166 * eta5 - 702172.2934567826 * eta6) +
                            0.013159545629626014 * (91.1469833190294 - 3557.5003799977294 * eta1 + 52391.684517955284 * eta2 - 344254.9973814295 * eta3 + 1.0141877915334814e6 * eta4 - 1.1505186449682908e6 * eta5 + 268756.85659532435 * eta6) * S1)
    end
end

@inline function _xhm_inter_amp_44_int3(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return eta1 * (chidiff1 * delta * (-0.8765502142143329 * eta1 + 22.806632458441996 * eta2 - 43.675503209991184 * eta3) + chidiff2 * (0.48698617426180074 * eta1 - 4.302527065360426 * eta2 + 16.18571810759235 * eta3)) +
               eta1 * (6.379772583015967 - 44.10631039734796 * eta1 + 269.44092930942793 * eta2 - 1285.7635006711453 * eta3 + 2379.538739132234 * eta4) +
               eta1 * S1 * (-0.23316184683282615 * (-1.7279023138971559 - 23.606399143993716 * eta1 + 409.3387618483284 * eta2 - 1115.4147472977265 * eta3) -
                            0.09653777612560172 * (-5.310643306559746 - 2.1852511802701264 * eta1 + 541.1248219096527 * eta2 - 1815.7529908827103 * eta3) * S1 -
                            0.060477799540741804 * (-14.578189130145661 + 175.6116682068523 * eta1 - 569.4799973930861 * eta2 + 426.0861915646515 * eta3) * S2)
    end
end

@inline function _xhm_inter_amp_44_int4(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return eta1 * (chidiff1 * delta * (-2.461738962276138 * eta1 + 45.3240543970684 * eta2 - 112.2714974622516 * eta3) + chidiff2 * (0.9158352037567031 * eta1 - 8.724582331021695 * eta2 + 28.44633544874233 * eta3)) +
               eta1 * (6.098676337298138 - 45.42463610529546 * eta1 + 350.97192927929433 * eta2 - 2002.2013283876834 * eta3 + 4067.1685640401033 * eta4) +
               eta1 * S1 * (-0.36068516166901304 * (-2.120354236840677 - 47.56175350408845 * eta1 + 1618.4222330016048 * eta2 - 14925.514654896673 * eta3 + 60287.45399959349 * eta4 - 91269.3745059139 * eta5) -
                            0.09635801207669747 * (-11.824692837267394 + 371.7551657959369 * eta1 - 4176.398139238679 * eta2 + 16655.87939259747 * eta3 - 4102.218189945819 * eta4 - 67024.98285179552 * eta5) * S1 -
                            0.06565232123453196 * (-26.15227471380236 + 1869.0168486099005 * eta1 - 33951.35186039629 * eta2 + 253694.6032002248 * eta3 - 845341.6001856657 * eta4 + 1.0442282862506858e6 * eta5) * S2)
    end
end

"""
    _xhm_inter_amp_collocation_values(l, m, eta, delta, StotR, chiPNHat, chidiff)

`(int1, int2, int3, int4)` -- the 4 intermediate amplitude collocation-point values for mode
`(l,m)` at `fAmpMatchIN + k/5*(fAmpMatchIM-fAmpMatchIN)` for `k=1,2,3,4` (already `abs`-wrapped,
matching `fabs(pAmp->IntermediateAmpFits[idx](...))`, `_internals.c` line 1569/2496). Dispatches
each collocation point to the correct spin variable (`StotR` or `chiPNHat`) per that specific
function -- see this file's module docstring / each function's own comment; NOT a uniform
convention across points or modes.
"""
@inline function _xhm_inter_amp_collocation_values(l::Integer, m::Integer, eta, delta, StotR, chiPNHat, chidiff)
    if l == 2 && m == 1
        return abs(_xhm_inter_amp_21_int1(eta, delta, StotR, chidiff)),
               abs(_xhm_inter_amp_21_int2(eta, delta, StotR, chidiff)),
               abs(_xhm_inter_amp_21_int3(eta, delta, StotR, chidiff)),
               abs(_xhm_inter_amp_21_int4(eta, delta, StotR, chidiff))
    elseif l == 3 && m == 3
        return abs(_xhm_inter_amp_33_int1(eta, delta, StotR, chidiff)),
               abs(_xhm_inter_amp_33_int2(eta, delta, StotR, chidiff)),
               abs(_xhm_inter_amp_33_int3(eta, delta, StotR, chidiff)),
               abs(_xhm_inter_amp_33_int4(eta, delta, StotR, chidiff))
    elseif l == 3 && m == 2
        return abs(_xhm_inter_amp_32_int1(eta, delta, chiPNHat, chidiff)),
               abs(_xhm_inter_amp_32_int2(eta, delta, StotR, chidiff)),
               abs(_xhm_inter_amp_32_int3(eta, delta, chiPNHat, chidiff)),
               abs(_xhm_inter_amp_32_int4(eta, delta, StotR, chidiff))
    elseif l == 4 && m == 4
        return abs(_xhm_inter_amp_44_int1(eta, delta, chiPNHat, chidiff)),
               abs(_xhm_inter_amp_44_int2(eta, delta, chiPNHat, chidiff)),
               abs(_xhm_inter_amp_44_int3(eta, delta, chiPNHat, chidiff)),
               abs(_xhm_inter_amp_44_int4(eta, delta, chiPNHat, chidiff))
    else
        throw(ArgumentError("_xhm_inter_amp_collocation_values: unsupported (l,m) = ($l,$m); only (2,1),(3,3),(3,2),(4,4) are implemented"))
    end
end

# ================================================================================================
# AMPLITUDE: generic linear-system assembly + solve (the actual default-release reconstruction --
# see module docstring). Basis is `f^j * f^{-7/6}` for `j=0..N-1`; a "value" row uses the basis
# directly, a "derivative" row uses its `f`-derivative. Reuses `IMRPhenomXAS.jl`'s already-existing
# `_xas_linsolve`/`_xas_det` generic Cramer's-rule solver (DRY, no duplicated linear algebra).
# NOTE: `_xas_linsolve` is now straight-line Gaussian elimination with partial pivoting and
# `_xas_det` no longer exists (see `IMRPhenomXAS.jl`); the Cramer's-rule remarks in this section describe the
# solver as it was when the conditioning fix below was designed, and the centring/scaling remains valid.
#
# # A NUMERICAL-STABILITY FIX beyond what the C source does, needed specifically for `N=8`
# LALSuite's own C code (`_intermediate.c` lines 2571-2598) builds this system directly in RAW
# `f` (`Mf` is `O(0.03-0.15)` in this region): `A[row,j] = f_row^j * f_row^{-7/6}`. For the `N=5`
# `(2,1)` system this port's own testing found this numerically fine (residuals ~1e-13), but for
# the `N=8` systems (`(3,3),(3,2),(4,4)`) the raw-`f` Vandermonde matrix is catastrophically
# ill-conditioned in double precision -- empirically `cond(A) ~ 1e12-1e17` for realistic
# `(fAmpMatchIN, fAmpMatchIM)` boundaries (`f^7` at `f~0.03` spans ~10 orders of magnitude within a
# single row) -- and `_xas_det`'s recursive cofactor-expansion Cramer's rule (numerically
# appropriate for `IMRPhenomXAS.jl`'s own `N<=5` systems, but not designed for anything this
# ill-conditioned) loses essentially ALL accuracy at `N=8`: verified directly via `A*c - b` residual checks that came back `O(10-10^4)` against
# `O(1)` right-hand sides -- i.e. completely wrong answers, not just reduced precision. GSL's LU
# decomposition (what the real C code uses) is far more robust to this than naive Cramer's rule,
# but the underlying conditioning problem is real regardless of solver, and this package's
# Enzyme-safety constraints (see `IMRPhenomXAS.jl`'s module docstring) rule out reaching for
# `LinearAlgebra`'s (mutating, BLAS/LAPACK-backed) `\\` as a fix.
#
# The fix applied here: rescale the frequency variable to `x = (f - Fmid) * invFhalf` with `Fmid =
# (F1+F4)/2`, `invFhalf = 2/(F4-F1)` (so `x` spans `[-1,1]` across the fit interval) BEFORE
# building the Vandermonde matrix -- i.e. solve for the coefficients of `sum_j c_j x^j` rather than
# `sum_j c_j f^j` directly. This "center and scale" transform is the textbook fix for Vandermonde
# ill-conditioning (an even better-conditioned choice than merely scaling `x=f*K` without
# centering -- verified empirically: centering brings `cond(A)` down to ~1e2-1e3 for this port's
# validated configs, vs ~1e4-1e5 for scaling alone) while leaving `_xas_linsolve`/`_xas_det`
# themselves untouched and still exactly reused. The coefficients this file stores and evaluates
# are therefore coefficients of `x = (f-Fmid)*invFhalf`, NOT of raw `f` -- `(Fmid, invFhalf)` is
# carried alongside them in the returned tuple and consumed by
# [`_xhm_inter_amp_ansatz`](@ref)/[`_xhm_inter_amp_coefficients`](@ref)'s callers rather than
# converted back to a raw-`f` basis (converting back would reintroduce the same large-scale
# cancellation this rescaling exists to avoid).
# ================================================================================================

@inline function _xhm_inter_amp_value_row(f::T, fmid::T, invfhalf::T, ::Val{N}) where {T,N}
    x = (f - fmid) * invfhalf
    invf76 = f^(T(-7.0) / T(6.0))
    return ntuple(j -> x^(j - 1) * invf76, Val(N))
end

@inline function _xhm_inter_amp_deriv_row(f::T, fmid::T, invfhalf::T, ::Val{N}) where {T,N}
    # d/df[ x^(j-1) * f^{-7/6} ] with x = (f-fmid)*invfhalf:
    #   = (j-1)*invfhalf*x^(j-2)*f^{-7/6}  -  (7/6)*x^(j-1)*f^{-1}*f^{-7/6}
    x = (f - fmid) * invfhalf
    invf76 = f^(T(-7.0) / T(6.0))
    invf = T(1.0) / f
    return ntuple(Val(N)) do j
        idx = j - 1
        term1 = idx == 0 ? zero(T) : T(idx) * invfhalf * x^(idx - 1) * invf76
        term2 = -(T(7.0) / T(6.0)) * x^idx * invf76 * invf
        term1 + term2
    end
end

"""
    _xhm_inter_amp_coefficients(l, m, eta, delta, StotR, chiPNHat, chidiff, F1, V1, d1, F4, V4, d4)

`(c, fmid, invfhalf)`: `c` is the 5 (`(2,1)`) or 8 (`(3,3),(3,2),(4,4)`) `InterCoefficient[]`
polynomial coefficients (of the RESCALED variable `x = (f-fmid)*invfhalf`, see the section
docstring above for why) for mode `(l,m)`'s intermediate amplitude ansatz, matching
`IMRPhenomXHM_Intermediate_Amp_Coefficients` / `IMRPhenomXHM_Intermediate_Amp_CollocationPoints`
(`LALSimIMRPhenomXHM_intermediate.c` lines 2436-2629) under the actual default release (see module
docstring for why this is a plain linear solve, not delta0-delta5). `F1, V1, d1` are the
inspiral/intermediate boundary frequency, value and `f`-derivative (this port computes these
internally from `IMRPhenomXHM_inspiral.jl`'s ansatz -- see
[`_xhm_inter_amp_coefficients_from_inspiral`](@ref) for a convenience wrapper that does so). `F4,
V4, d4` are the intermediate/ringdown boundary frequency, value and `f`-derivative -- these
are caller-supplied (see module docstring). `c` is
returned as an `NTuple{8,T}`, zero-padded for `(2,1)`'s 5-term polynomial so
[`_xhm_inter_amp_ansatz`](@ref) can evaluate uniformly regardless of mode.
"""
@inline function _xhm_inter_amp_coefficients(l::Integer, m::Integer, eta, delta, StotR, chiPNHat, chidiff, F1, V1, d1, F4, V4, d4)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chiPNHat), typeof(chidiff),
        typeof(F1), typeof(V1), typeof(d1), typeof(F4), typeof(V4), typeof(d4))
    F1T, F4T = T(F1), T(F4)
    fmid = (F1T + F4T) / T(2.0)
    invfhalf = T(2.0) / (F4T - F1T)
    deltaf = (F4T - F1T) / T(5.0)
    int1, int2, int3, int4 = _xhm_inter_amp_collocation_values(l, m, eta, delta, StotR, chiPNHat, chidiff)

    if l == 2 && m == 1
        Fa = F1T + deltaf * T(1.0)
        Fb = F1T + deltaf * T(3.0)
        A = (
            _xhm_inter_amp_value_row(F1T, fmid, invfhalf, Val(5)),
            _xhm_inter_amp_value_row(Fa, fmid, invfhalf, Val(5)),
            _xhm_inter_amp_value_row(Fb, fmid, invfhalf, Val(5)),
            _xhm_inter_amp_value_row(F4T, fmid, invfhalf, Val(5)),
            _xhm_inter_amp_deriv_row(F4T, fmid, invfhalf, Val(5)),
        )
        b = (T(V1), T(int1), T(int3), T(V4), T(d4))
        c5 = _xas_linsolve(A, b)
        return (c5[1], c5[2], c5[3], c5[4], c5[5], zero(T), zero(T), zero(T)), fmid, invfhalf
    else
        Fa = F1T + deltaf * T(1.0)
        Fb = F1T + deltaf * T(2.0)
        Fc = F1T + deltaf * T(3.0)
        Fd = F1T + deltaf * T(4.0)
        A = (
            _xhm_inter_amp_value_row(F1T, fmid, invfhalf, Val(8)),
            _xhm_inter_amp_deriv_row(F1T, fmid, invfhalf, Val(8)),
            _xhm_inter_amp_value_row(Fa, fmid, invfhalf, Val(8)),
            _xhm_inter_amp_value_row(Fb, fmid, invfhalf, Val(8)),
            _xhm_inter_amp_value_row(Fc, fmid, invfhalf, Val(8)),
            _xhm_inter_amp_value_row(Fd, fmid, invfhalf, Val(8)),
            _xhm_inter_amp_value_row(F4T, fmid, invfhalf, Val(8)),
            _xhm_inter_amp_deriv_row(F4T, fmid, invfhalf, Val(8)),
        )
        b = (T(V1), T(d1), T(int1), T(int2), T(int3), T(int4), T(V4), T(d4))
        return _xas_linsolve(A, b), fmid, invfhalf
    end
end

"""
    _xhm_inter_amp_coefficients_from_inspiral(l, m, eta, delta, StotR, chiPNHat, chidiff, inspcoeffs, F4, V4, d4)

Convenience wrapper around [`_xhm_inter_amp_coefficients`](@ref) that computes the inspiral-side
boundary `(F1, V1, d1)` from `_xhm_insp_amp_coefficients`/`_xhm_insp_amp_ansatz`/
`_xhm_insp_amp_nd_ansatz` (`IMRPhenomXHM_inspiral.jl`) rather than requiring the caller to supply
them -- `F1 = inspcoeffs.fcutInsp` (matches `IMRPhenomXHM_Amplitude_fcutInsp`'s `q<20` branch,
`fcut = fMECOlm`, the only case this port's validated mass-ratio range exercises). For `(2,1)`
mode, `d1` is not computed (the ansatz doesn't use it -- the left-boundary derivative is skipped
for this mode, see module docstring) but is still evaluated for uniformity/simplicity; it is simply
unused by the `l==2,m==1` branch of `_xhm_inter_amp_coefficients`.
"""
@inline function _xhm_inter_amp_coefficients_from_inspiral(l::Integer, m::Integer, eta, delta, StotR, chiPNHat, chidiff, inspcoeffs, F4, V4, d4)
    F1 = inspcoeffs.fcutInsp
    V1 = _xhm_insp_amp_ansatz(F1, inspcoeffs)
    d1 = _xhm_insp_amp_nd_ansatz(F1, inspcoeffs)
    return _xhm_inter_amp_coefficients(l, m, eta, delta, StotR, chiPNHat, chidiff, F1, V1, d1, F4, V4, d4)
end

"""
    _xhm_inter_amp_ansatz(f, c, fmid, invfhalf)

`Ansatz(f) = f^{-7/6} * sum_{j=0}^{7} c[j+1] * ((f-fmid)*invfhalf)^j`, matching
`IMRPhenomXHM_Intermediate_Amp_Ansatz`'s `ReleaseVersion != 122019` branch
(`LALSimIMRPhenomXHM_intermediate.c` lines 1902-1911) EXCEPT evaluated in the rescaled, centered
variable `x=(f-fmid)*invfhalf` rather than raw `f` -- see the "AMPLITUDE: generic
linear-system..." section docstring above for why. `(c, fmid, invfhalf)` is the triple returned by
[`_xhm_inter_amp_coefficients`](@ref) (`c` zero-padded for `(2,1)`'s shorter polynomial, which
evaluates correctly here without any mode-specific branching since the padding terms vanish).
"""
@inline function _xhm_inter_amp_ansatz(f, c::NTuple{8,T}, fmid, invfhalf) where {T}
    x = (f - fmid) * invfhalf
    result = c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (c[6] + x * (c[7] + x * c[8]))))))
    return result * f^(T(-7.0) / T(6.0))
end

# ================================================================================================
# PHASE: intermediate collocation-point ("phase derivative", `p1`-`p6`) parameter-space fits.
# Ported from `IMRPhenomXHM_Inter_Phase_{21,33,32,44}_p{1..6}`,
# `LALSimIMRPhenomXHM_intermediate.c` lines 2642-3252. `p1`/`p2` need `chi1L`,`chi2L` individually
# (not just `dchi`); `p3`-`p6` only need `dchi` -- confirmed per-function by reading the C source
# (NOT assumed uniform). `S = pWF->STotR` throughout (unlike the amplitude `int` fits above).
# ================================================================================================

@inline function _xhm_inter_phase_21_p1(eta, S, chi1, chi2)
    T = promote_type(typeof(eta), typeof(S), typeof(chi1), typeof(chi2))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 4045.84 + 7.63226 / eta - 1956.93 * eta - 23428.1 * eta2 + 369153.0 * eta3 - 2.28832e6 * eta4 + 6.82533e6 * eta5 - 7.86254e6 * eta6
        eqSpin = -347.273 * S + 83.5428 * S2 - 355.67 * S3 + (4.44457 * S + 16.5548 * S2 + 13.6971 * S3) / eta + eta * (-79.761 * S - 355.299 * S2 + 1114.51 * S3 - 1077.75 * S4) + 92.6654 * S4 + eta2 * (-619.837 * S - 722.787 * S2 + 2392.73 * S3 + 2689.18 * S4)
        uneqSpin = (918.976 * chi1 * delta - 918.976 * chi2 * delta) * eta + (91.7679 * chi1 * delta - 91.7679 * chi2 * delta) * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_33_p1(eta, S, chi1, chi2)
    T = promote_type(typeof(eta), typeof(S), typeof(chi1), typeof(chi2))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        noSpin = 4360.19 + 4.27128 / eta - 8727.4 * eta + 18485.9 * eta2 + 371303.00000000006 * eta3 - 3.22792e6 * eta4 + 1.01799e7 * eta5 - 1.15659e7 * eta6
        eqSpin = ((11.6635 - 251.579 * eta - 3255.6400000000003 * eta2 + 19614.6 * eta3 - 34860.2 * eta4) * S + (14.8017 + 204.025 * eta - 5421.92 * eta2 + 36587.3 * eta3 - 74299.5 * eta4) * S2) / eta
        uneqSpin = eta * (223.65100000000004 * chi1 * delta * (3.9201300240106223 + eta) - 223.65100000000004 * chi2 * delta * (3.9201300240106223 + eta))
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_32_p1(eta, S, chi1, chi2)
    T = promote_type(typeof(eta), typeof(S), typeof(chi1), typeof(chi2))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 4414.11 + 4.21564 / eta - 10687.8 * eta + 58234.6 * eta2 - 64068.40000000001 * eta3 - 704442.0 * eta4 + 2.86393e6 * eta5 - 3.26362e6 * eta6
        eqSpin = ((6.39833 - 610.267 * eta + 2095.72 * eta2 - 3970.89 * eta3) * S + (22.956700000000005 - 99.1551 * eta + 331.593 * eta2 - 794.79 * eta3) * S2 + (10.4333 + 43.8812 * eta - 541.261 * eta2 + 294.289 * eta3) * S3 + eta * (106.047 - 1569.0299999999997 * eta + 4810.61 * eta2) * S4) / eta
        uneqSpin = 132.244 * delta * eta * (chi1 * (6.227738120444028 - eta) + chi2 * (-6.227738120444028 + eta))
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_44_p1(eta, S, chi1, chi2)
    T = promote_type(typeof(eta), typeof(S), typeof(chi1), typeof(chi2))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        noSpin = 4349.66 + 4.34125 / eta - 8202.33 * eta + 5534.1 * eta2 + 536500.0 * eta3 - 4.33197e6 * eta4 + 1.37792e7 * eta5 - 1.60802e7 * eta6
        eqSpin = ((12.0704 - 528.098 * eta + 1822.9100000000003 * eta2 - 9349.73 * eta3 + 17900.9 * eta4) * S + (10.4092 + 253.334 * eta - 5452.04 * eta2 + 35416.6 * eta3 - 71523.0 * eta4) * S2 + eta * (492.60300000000007 - 9508.5 * eta + 57303.4 * eta2 - 109418.0 * eta3) * S3) / eta
        uneqSpin = -262.143 * delta * eta * (chi1 * (-3.0782778864970646 - eta) + chi2 * (3.0782778864970646 + eta))
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_21_p2(eta, S, chi1, chi2)
    T = promote_type(typeof(eta), typeof(S), typeof(chi1), typeof(chi2))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta3 * eta2
        eta6 = eta4 * eta2
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 3509.09 + 0.91868 / eta + 194.72 * eta - 27556.2 * eta2 + 369153.0 * eta3 - 2.28832e6 * eta4 + 6.82533e6 * eta5 - 7.86254e6 * eta6
        eqSpin = ((0.7083999999999999 - 60.1611 * eta + 131.815 * eta2 - 619.837 * eta3) * S + (6.104720000000001 - 59.2068 * eta + 278.588 * eta2 - 722.787 * eta3) * S2 + (5.7791 + 117.913 * eta - 1180.4 * eta2 + 2392.73 * eta3) * S3 + eta * (92.6654 - 1077.75 * eta + 2689.18 * eta2) * S4) / eta
        uneqSpin = -91.7679 * delta * eta * (chi1 * (-1.6012352903357276 - eta) + chi2 * (1.6012352903357276 + eta))
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_33_p2(eta, S, chi1, chi2)
    T = promote_type(typeof(eta), typeof(S), typeof(chi1), typeof(chi2))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        noSpin = 3797.06 + 0.786684 / eta - 2397.09 * eta - 25514.0 * eta2 + 518314.99999999994 * eta3 - 3.41708e6 * eta4 + 1.01799e7 * eta5 - 1.15659e7 * eta6
        eqSpin = ((6.7812399999999995 + 39.4668 * eta - 3520.37 * eta2 + 19614.6 * eta3 - 34860.2 * eta4) * S + (4.80384 + 293.215 * eta - 5914.61 * eta2 + 36587.3 * eta3 - 74299.5 * eta4) * S2) / eta
        uneqSpin = -223.65100000000004 * delta * eta * (chi1 * (-1.3095134830606614 - eta) + chi2 * (1.3095134830606614 + eta))
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_32_p2(eta, S, chi1, chi2)
    T = promote_type(typeof(eta), typeof(S), typeof(chi1), typeof(chi2))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 3980.7 + 0.956703 / eta - 6202.38 * eta + 29218.1 * eta2 + 24484.2 * eta3 - 807629.0 * eta4 + 2.86393e6 * eta5 - 3.26362e6 * eta6
        eqSpin = ((1.92692 - 226.825 * eta + 75.246 * eta2 + 1291.56 * eta3) * S + (15.328700000000001 - 99.1551 * eta + 608.328 * eta2 - 2402.94 * eta3) * S2 + (10.4333 + 43.8812 * eta - 541.261 * eta2 + 294.289 * eta3) * S3 + eta * (106.047 - 1569.0299999999997 * eta + 4810.61 * eta2) * S4) / eta
        uneqSpin = 132.244 * delta * eta * (chi1 * (2.5769789177580837 - eta) + chi2 * (-2.5769789177580837 + eta))
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_44_p2(eta, S, chi1, chi2)
    T = promote_type(typeof(eta), typeof(S), typeof(chi1), typeof(chi2))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        noSpin = 3804.19 + 0.66144 / eta - 2421.77 * eta - 33475.8 * eta2 + 665951.0 * eta3 - 4.50145e6 * eta4 + 1.37792e7 * eta5 - 1.60802e7 * eta6
        eqSpin = ((5.83038 - 172.047 * eta + 926.576 * eta2 - 7676.87 * eta3 + 17900.9 * eta4) * S + (6.17601 + 253.334 * eta - 5672.02 * eta2 + 35722.1 * eta3 - 71523.0 * eta4) * S2 + eta * (492.60300000000007 - 9508.5 * eta + 57303.4 * eta2 - 109418.0 * eta3) * S3) / eta
        uneqSpin = -262.143 * delta * eta * (chi1 * (-1.0543062374352932 - eta) + chi2 * (1.0543062374352932 + eta))
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_21_p3(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta2 * eta2
        eta5 = eta2 * eta3
        eta6 = eta3 * eta3
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 3241.68 + 890.016 * eta - 28651.9 * eta2 + 369153.0 * eta3 - 2.28832e6 * eta4 + 6.82533e6 * eta5 - 7.86254e6 * eta6
        eqSpin = (-2.2484 + 187.641 * eta - 619.837 * eta2) * S + (3.22603 + 166.323 * eta - 722.787 * eta2) * S2 + (117.913 - 1094.59 * eta + 2392.73 * eta2) * S3 + (92.6654 - 1077.75 * eta + 2689.18 * eta2) * S4
        uneqSpin = 91.7679 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_33_p3(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        noSpin = 3321.83 + 1796.03 * eta - 52406.1 * eta2 + 605028.0 * eta3 - 3.52532e6 * eta4 + 1.01799e7 * eta5 - 1.15659e7 * eta6
        eqSpin = (223.601 - 3714.77 * eta + 19614.6 * eta2 - 34860.2 * eta3) * S + (314.317 - 5906.46 * eta + 36587.3 * eta2 - 74299.5 * eta3) * S2
        uneqSpin = 223.651 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_32_p3(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 3416.57 + 2308.63 * eta - 84042.9 * eta2 + 1.01936e6 * eta3 - 6.0644e6 * eta4 + 1.76399e7 * eta5 - 2.0065e7 * eta6
        eqSpin = (24.6295 - 282.354 * eta - 2582.55 * eta2 + 12750.0 * eta3) * S + (433.675 - 8775.86 * eta + 56407.8 * eta2 - 114798.0 * eta3) * S2 + (559.705 - 10627.4 * eta + 61581.0 * eta2 - 114029.0 * eta3) * S3 + (106.047 - 1569.03 * eta + 4810.61 * eta2) * S4
        uneqSpin = 63.9466 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_44_p3(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        noSpin = 3308.97 + 2353.58 * eta - 66340.1 * eta2 + 777272.0 * eta3 - 4.64438e6 * eta4 + 1.37792e7 * eta5 - 1.60802e7 * eta6
        eqSpin = (-21.5697 + 926.576 * eta - 7989.26 * eta2 + 17900.9 * eta3) * S + (353.539 - 6403.24 * eta + 37599.5 * eta2 - 71523.0 * eta3) * S2 + (492.603 - 9508.5 * eta + 57303.4 * eta2 - 109418.0 * eta3) * S3
        uneqSpin = 262.143 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_21_p4(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 3160.88 + 974.355 * eta - 28932.5 * eta2 + 369780.0 * eta3 - 2.28832e6 * eta4 + 6.82533e6 * eta5 - 7.86254e6 * eta6
        eqSpin = (26.3355 - 196.851 * eta + 438.401 * eta2) * S + (45.9957 - 256.248 * eta + 117.563 * eta2) * S2 + (-20.0261 + 467.057 * eta - 1613.0 * eta2) * S3 + (-61.7446 + 577.057 * eta - 1096.81 * eta2) * S4
        uneqSpin = 65.3326 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_33_p4(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        noSpin = 3239.44 - 661.15 * eta + 5139.79 * eta2 + 3456.2 * eta3 - 248477.0 * eta4 + 1.17255e6 * eta5 - 1.70363e6 * eta6
        eqSpin = (225.859 - 4150.09 * eta + 24364.0 * eta2 - 46537.3 * eta3) * S + (35.2439 - 994.971 * eta + 8953.98 * eta2 - 23603.5 * eta3) * S2 + (-310.489 + 5946.15 * eta - 35337.1 * eta2 + 67102.4 * eta3) * S3
        uneqSpin = 30.484 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_32_p4(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 3307.49 - 476.909 * eta - 5980.37 * eta2 + 127610.0 * eta3 - 919108.0 * eta4 + 2.86393e6 * eta5 - 3.26362e6 * eta6
        eqSpin = (-5.02553 - 282.354 * eta + 1291.56 * eta2) * S + (-43.8823 + 740.123 * eta - 2402.94 * eta2) * S2 + (43.8812 - 370.362 * eta + 294.289 * eta2) * S3 + (106.047 - 1569.03 * eta + 4810.61 * eta2) * S4
        uneqSpin = -132.244 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_44_p4(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        noSpin = 3245.63 - 928.56 * eta + 8463.89 * eta2 - 17422.6 * eta3 - 165169.0 * eta4 + 908279.0 * eta5 - 1.31138e6 * eta6
        eqSpin = (32.506 - 590.293 * eta + 3536.61 * eta2 - 6758.52 * eta3) * S + (-25.7716 + 738.141 * eta - 4867.87 * eta2 + 9129.45 * eta3) * S2 + (-15.7439 + 620.695 * eta - 4679.24 * eta2 + 9582.58 * eta3) * S3
        uneqSpin = 87.0832 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_21_p5(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 3102.36 + 315.911 * eta - 1688.26 * eta2 + 3635.76 * eta3
        eqSpin = (-23.0959 + 320.93 * eta - 1029.76 * eta2) * S + (-49.5435 + 826.816 * eta - 3079.39 * eta2) * S2 + (40.7054 - 365.842 * eta + 1094.11 * eta2) * S3 + (81.8379 - 1243.26 * eta + 4689.22 * eta2) * S4
        uneqSpin = 119.014 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_33_p5(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        noSpin = 3114.3 + 2143.06 * eta - 49428.3 * eta2 + 563997.0 * eta3 - 3.35991e6 * eta4 + 9.99745e6 * eta5 - 1.17123e7 * eta6
        eqSpin = (190.051 - 3705.08 * eta + 23046.2 * eta2 - 46537.3 * eta3) * S + (63.6615 - 1414.2 * eta + 10166.1 * eta2 - 23603.5 * eta3) * S2 + (-257.524 + 5179.97 * eta - 33001.4 * eta2 + 67102.4 * eta3) * S3
        uneqSpin = 54.9833 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_32_p5(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        eta7 = eta6 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 3259.03 - 3967.58 * eta + 111203.0 * eta2 - 1.81883e6 * eta3 + 1.73811e7 * eta4 - 9.56988e7 * eta5 + 2.75056e8 * eta6 - 3.15866e8 * eta7
        eqSpin = (19.7509 - 1104.53 * eta + 3810.18 * eta2) * S + (-230.07 + 2314.51 * eta - 5944.49 * eta2) * S2 + (-201.633 + 2183.43 * eta - 6233.99 * eta2) * S3 + (106.047 - 1569.03 * eta + 4810.61 * eta2) * S4
        uneqSpin = 112.714 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_44_p5(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        eta7 = eta6 * eta
        S2 = S * S
        S3 = S2 * S
        noSpin = 3108.38 + 3722.46 * eta - 119588.0 * eta2 + 1.92148e6 * eta3 - 1.69796e7 * eta4 + 8.39194e7 * eta5 - 2.17143e8 * eta6 + 2.2829700000000003e8 * eta7
        eqSpin = (118.319 - 529.854 * eta) * eta * S + (21.0314 - 240.648 * eta + 516.333 * eta2) * S2 + (20.3384 - 356.241 * eta + 999.417 * eta2) * S3
        uneqSpin = 97.1364 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_21_p6(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 3089.18 + 4.89194 * eta + 190.008 * eta2 - 255.245 * eta3
        eqSpin = (2.96997 + 57.1612 * eta - 432.223 * eta2) * S + (-18.8929 + 630.516 * eta - 2804.66 * eta2) * S2 + (-24.6193 + 549.085 * eta2) * S3 + (-12.8798 - 722.674 * eta + 3967.43 * eta2) * S4
        uneqSpin = 74.0984 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_33_p6(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        noSpin = 3111.46 + 384.121 * eta - 13003.6 * eta2 + 179537.0 * eta3 - 1.19313e6 * eta4 + 3.79886e6 * eta5 - 4.64858e6 * eta6
        eqSpin = (182.864 - 3834.22 * eta + 24532.9 * eta2 - 50165.9 * eta3) * S + (21.0158 - 746.957 * eta + 6701.33 * eta2 - 17842.3 * eta3) * S2 + (-292.855 + 5886.62 * eta - 37382.4 * eta2 + 75501.8 * eta3) * S3
        uneqSpin = 75.5162 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_32_p6(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        eta7 = eta6 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = 3259.03 - 3967.58 * eta + 111203.0 * eta2 - 1.81883e6 * eta3 + 1.73811e7 * eta4 - 9.56988e7 * eta5 + 2.75056e8 * eta6 - 3.15866e8 * eta7
        eqSpin = (19.7509 - 1104.53 * eta + 3810.18 * eta2) * S + (-230.07 + 2314.51 * eta - 5944.49 * eta2) * S2 + (-201.633 + 2183.43 * eta - 6233.99 * eta2) * S3 + (106.047 - 1569.03 * eta + 4810.61 * eta2) * S4
        uneqSpin = 112.714 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_inter_phase_44_p6(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        eta5 = eta4 * eta
        eta6 = eta5 * eta
        S2 = S * S
        S3 = S2 * S
        noSpin = 3096.03 + 986.752 * eta - 20371.1 * eta2 + 220332.0 * eta3 - 1.31523e6 * eta4 + 4.29193e6 * eta5 - 6.01179e6 * eta6
        eqSpin = (-9.96292 - 118.526 * eta + 2255.76 * eta2 - 6758.52 * eta3) * S + (-14.4869 + 370.039 * eta - 3605.8 * eta2 + 9129.45 * eta3) * S2 + (17.0209 + 70.1931 * eta - 3070.08 * eta2 + 9582.58 * eta3) * S3
        uneqSpin = 23.0759 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

"""
    _xhm_inter_phase_collocation_values(l, m, eta, S, chi1, chi2)

`(p1, p2, p3, p4, p5, p6)` -- the 6 intermediate phase-derivative collocation-point values for
mode `(l,m)`, matching `pPhase->IntermediatePhaseFits[modeint*6+i](...)`
(`_internals.c` line 1999). `S = pWF->STotR`, `chi1,chi2 = pWF->chi1L,chi2L`, `dchi = chi1-chi2`.
"""
@inline function _xhm_inter_phase_collocation_values(l::Integer, m::Integer, eta, S, chi1, chi2)
    dchi = chi1 - chi2
    if l == 2 && m == 1
        return _xhm_inter_phase_21_p1(eta, S, chi1, chi2), _xhm_inter_phase_21_p2(eta, S, chi1, chi2),
               _xhm_inter_phase_21_p3(eta, S, dchi), _xhm_inter_phase_21_p4(eta, S, dchi),
               _xhm_inter_phase_21_p5(eta, S, dchi), _xhm_inter_phase_21_p6(eta, S, dchi)
    elseif l == 3 && m == 3
        return _xhm_inter_phase_33_p1(eta, S, chi1, chi2), _xhm_inter_phase_33_p2(eta, S, chi1, chi2),
               _xhm_inter_phase_33_p3(eta, S, dchi), _xhm_inter_phase_33_p4(eta, S, dchi),
               _xhm_inter_phase_33_p5(eta, S, dchi), _xhm_inter_phase_33_p6(eta, S, dchi)
    elseif l == 3 && m == 2
        return _xhm_inter_phase_32_p1(eta, S, chi1, chi2), _xhm_inter_phase_32_p2(eta, S, chi1, chi2),
               _xhm_inter_phase_32_p3(eta, S, dchi), _xhm_inter_phase_32_p4(eta, S, dchi),
               _xhm_inter_phase_32_p5(eta, S, dchi), _xhm_inter_phase_32_p6(eta, S, dchi)
    elseif l == 4 && m == 4
        return _xhm_inter_phase_44_p1(eta, S, chi1, chi2), _xhm_inter_phase_44_p2(eta, S, chi1, chi2),
               _xhm_inter_phase_44_p3(eta, S, dchi), _xhm_inter_phase_44_p4(eta, S, dchi),
               _xhm_inter_phase_44_p5(eta, S, dchi), _xhm_inter_phase_44_p6(eta, S, dchi)
    else
        throw(ArgumentError("_xhm_inter_phase_collocation_values: unsupported (l,m) = ($l,$m); only (2,1),(3,3),(3,2),(4,4) are implemented"))
    end
end

# ================================================================================================
# PHASE: intermediate collocation-point frequencies, matching
# `IMRPhenomXHM_Intermediate_CollocPtsFreqs` (`LALSimIMRPhenomXHM_internals.c` lines 876-927,
# `case 122019`/`case 122022` -- identical formulas in both, no version branching) and `GetfcutInsp`
# (lines 867-872). Only the non-`(3,2)` branch is implemented (`(3,2)` is in `IMRPhenomXHM.jl`, see module
# docstring) -- calling with `l==3,m==2` throws.
# ================================================================================================

"""
    _xhm_inter_phase_colloc_freqs(l, m, fMECOlm, fring, fdamp, eta)

`(fcut, f1, f2, f3, f4, f5, fPhaseMatchIN, fPhaseMatchIM)` -- `fcut` is `GetfcutInsp`'s value (used
as the frequency of `p1`), `f1..f5` are `p2..p6`'s frequencies, `fPhaseMatchIN = fMECOlm` and
`fPhaseMatchIM = fring - fdamp` are the intermediate region's own matching frequencies (used for
the C1-continuity gluing to the neighbouring regions, NOT necessarily equal to `fcut`/`f5`). `fring,
fdamp` are the mode's own (`l,m`) ringdown frequency/damping-time (from
[`_xhm_fring_fdamp`](@ref)), NOT the dominant `(2,2)` mode's.
"""
@inline function _xhm_inter_phase_colloc_freqs(l::Integer, m::Integer, fMECOlm, fring, fdamp, eta)
    if l == 3 && m == 2
        throw(ArgumentError("_xhm_inter_phase_colloc_freqs: (3,2) intermediate phase collocation-point frequencies need IMRPhenomXHM.jl's mode-mixing machinery (SpheroidalToSpherical) -- not implemented, see module docstring"))
    end
    T = promote_type(typeof(fMECOlm), typeof(fring), typeof(fdamp), typeof(eta))
    @Tlit T begin
        eta_factor = 1.0 + 0.001 * (0.25 / eta - 1.0)
        fcut = eta_factor * fMECOlm
        f1 = (sqrt(3.0) * (fcut - fring) + 2.0 * (fcut + fring)) / 4.0
        f2 = (3.0 * fcut + fring) / 4.0
        f3 = (fcut + fring) / 2.0
        f4 = (fcut + 3.0 * fring) / 4.0
        f5 = (fcut + 7.0 * fring) / 8.0
        fPhaseMatchIN = fMECOlm
        fPhaseMatchIM = fring - fdamp
    end
    return fcut, f1, f2, f3, f4, f5, fPhaseMatchIN, fPhaseMatchIM
end

"""
    _xhm_inter_phase_cpoints_indices(l, m, eta, StotR, etaEMR=0.05)

The 5 collocation-point indices (into the 6-element `(p1,...,p6)`/frequency tuples, 1-based)
actually used to build the 5x5 linear system for `(2,1),(3,3),(4,4)` (`MixingOn=0`), matching
`_internals.c` lines 2096-2121's `cpoints_indices` selection. `(3,2)` is not supported here (see
[`_xhm_inter_phase_colloc_freqs`](@ref)). `(2,1)` with `StotR >= 0.8` uses (1,2,3,5,6) together with the
value override `_xhm_inter_phase_highspin21`.
"""
@inline function _xhm_inter_phase_cpoints_indices(l::Integer, m::Integer, eta, StotR, etaEMR=0.05)
    if l == 3 && m == 2
        throw(ArgumentError("_xhm_inter_phase_cpoints_indices: (3,2) needs IMRPhenomXHM.jl's mode-mixing machinery, see module docstring"))
    end
    emm_eq_ell = (l == m)
    if (eta < etaEMR) || (emm_eq_ell && StotR >= 0.8) || (l == 3 && m == 3 && StotR < 0)
        return (1, 2, 4, 5, 6)
    elseif l == 2 && m == 1 && StotR >= 0.8     # `_internals.c` line 2112 (with the value override, see below)
        return (1, 2, 3, 5, 6)
    else
        return (1, 2, 3, 4, 6)
    end
end

"""
    _xhm_inter_phase_deltaT(eta, StotR, dchi)

`pWFHM->DeltaT = -2*pi*(500 + psi4tostrain)` (`IMRPhenomXHM_SetHMWaveformVariables`,
`_internals.c` line 465, `psi4tostrain = XLALSimIMRPhenomXPsi4ToStrain(eta, StotR, dchi)`): the
same time-alignment constant every mode's intermediate-region phase-derivative collocation VALUES
get shifted by (see [`_xhm_inter_phase_coefficients`](@ref)'s docstring for where and why). `psi4tostrain` is mode-independent (same fit regardless of `(l,m)`),
matching `IMRPhenomXHM.jl`'s own, separately-needed `_xhm_psi4_to_strain` -- reused here directly
rather than duplicated (Julia resolves this call at run time, so the cross-file, later-in-include-
order reference is safe: by the time any code actually calls this function the whole `GravitationalWaveforms`
module, `IMRPhenomXHM.jl` included, is already loaded).
"""
@inline function _xhm_inter_phase_deltaT(eta, StotR, dchi)
    T = promote_type(typeof(eta), typeof(StotR), typeof(dchi))
    psi4tostrain = _xhm_psi4_to_strain(eta, StotR, dchi)
    @Tlit T begin
        return -2.0 * pi * (500.0 + psi4tostrain)
    end
end

# High-spin (2,1) override (`IMRPhenomXHM_GetPhaseCoefficients`, `_internals.c` lines 2076-2093): for StotR >= 0.8 the
# first two collocation values (after the DeltaT shift) are re-anchored on the third with the (2,2) inspiral phase
# derivative's differences at the (2/m)-scaled collocation frequencies, "to avoid sharp transitions for the 21 mode by
# extending the inspiral region into the intermediate one". Needs the (2,2) coefficients (`coeffs22`); without them the
# case is refused as before.
@inline _xhm_inter_phase_highspin21(l, m, eta, StotR, pvals, freqs, ::Nothing) =
    (l == 2 && m == 1 && StotR >= 0.8) ?
        throw(ArgumentError("_xhm_inter_phase_coefficients: (2,1) with StotR >= 0.8 needs coeffs22 (the near-extremal-spin override)")) :
        pvals
@inline function _xhm_inter_phase_highspin21(l, m, eta, StotR, pvals, freqs, coeffs22)
    (l == 2 && m == 1 && StotR >= 0.8) || return pvals
    two_over_m = 2 / m
    iv1 = _xhm22_dphase_raw(coeffs22, two_over_m * freqs[1]) / eta
    iv2 = _xhm22_dphase_raw(coeffs22, two_over_m * freqs[2]) / eta
    iv3 = _xhm22_dphase_raw(coeffs22, two_over_m * freqs[3]) / eta
    p2 = pvals[3] + (iv2 - iv3)
    p1 = p2 + (iv1 - iv2)
    return (p1, p2, pvals[3], pvals[4], pvals[5], pvals[6])
end

"""
    _xhm_inter_phase_coefficients(l, m, eta, StotR, chi1, chi2, fMECOlm, fring, fdamp)

`(c0, cL, c1, c2, c4)` for mode `(l,m)`'s intermediate phase-derivative ansatz, matching the
`MixingOn == 0` branch of `IMRPhenomXHM_GetPhaseCoefficients` (`_internals.c` lines 2126-2174,
2273-2287) -- `(2,1),(3,3),(4,4)` only (`(3,2)` is in `IMRPhenomXHM.jl`, see module docstring; this
throws `ArgumentError`). `fring, fdamp` are the mode's own `(l,m)` ringdown values (from
[`_xhm_fring_fdamp`](@ref)).

**The `DeltaT` shift matters**: `IMRPhenomXHM_GetPhaseCoefficients` adds `pWFHM->DeltaT` (see
[`_xhm_inter_phase_deltaT`](@ref)) to EVERY intermediate collocation-point phase-DERIVATIVE value
BEFORE solving the linear system (`_internals.c` lines 1997-2003, "time-shift wf so that modes peak
around t=0" -- runs unconditionally, before the `MixingOn` switch, for every mode). Because the linear system's basis has a constant `1.0` term in
every row (the `c0` column), adding this SAME constant to every right-hand-side value shifts the
solved `c0` by EXACTLY that constant and leaves every other coefficient unchanged (a clean linear-
algebra fact, not approximate) -- so omitting it would give a single, uniform, mode-INDEPENDENT
(same fit function `XLALSimIMRPhenomXPsi4ToStrain` regardless of `(l,m)`) but config-DEPENDENT
additive offset to `dPhi/df` across the ENTIRE assembled waveform (inspiral AND ringdown too, since
`C1INSP`/`C1RD` are matched against this same intermediate reference and inherit it
via the C1-continuity glue). Only an absolute `h_lm`-vs-`SimIMRPhenomXHMGenerateFDOneMode` comparison
shows this; a shape-only comparison with the median offset removed does not.
"""
@inline function _xhm_inter_phase_coefficients(l::Integer, m::Integer, eta, StotR, chi1, chi2, fMECOlm, fring, fdamp,
                                               coeffs22=nothing)
    if l == 3 && m == 2
        throw(ArgumentError("_xhm_inter_phase_coefficients: (3,2) intermediate phase reconstruction needs IMRPhenomXHM.jl's mode-mixing machinery (SpheroidalToSpherical) -- not implemented, see module docstring"))
    end
    T = promote_type(typeof(eta), typeof(StotR), typeof(chi1), typeof(chi2), typeof(fMECOlm), typeof(fring), typeof(fdamp))

    pvals0 = _xhm_inter_phase_collocation_values(l, m, eta, StotR, chi1, chi2)
    deltaT = _xhm_inter_phase_deltaT(eta, StotR, chi1 - chi2)
    # `ntuple(f, Val(6))` over a DIFFERENTLY-NAMED source (`pvals0`, not reusing `pvals`) instead
    # of `map(f, ::NTuple{6})` writing back into `pvals` itself: the original `pvals = map(v -> v +
    # deltaT, pvals)` self-reassignment made the closure capture `pvals` as a boxed (`Core.Box`,
    # effectively `Any`-typed) variable -- Julia's classic "closure over a variable reassigned in
    # its own enclosing scope" footgun, invisible on CPU (just a perf hit) but fatal on GPU
    # backends (confirmed by a real `InvalidIRError`, "unsupported dynamic function invocation",
    # cascading into every downstream use of `pvals` including line ~1328 below). Renaming the
    # source breaks the self-reference, so `pvals` stays a plain concretely-typed `NTuple{6,T}`.
    pvals1 = ntuple(k -> pvals0[k] + deltaT, Val(6))
    fcut, f1, f2, f3, f4, f5, _, _ = _xhm_inter_phase_colloc_freqs(l, m, fMECOlm, fring, fdamp, eta)
    freqs = (fcut, f1, f2, f3, f4, f5)
    pvals = _xhm_inter_phase_highspin21(l, m, eta, StotR, pvals1, freqs, coeffs22)
    idx = _xhm_inter_phase_cpoints_indices(l, m, eta, StotR)

    fringT = T(fring)
    fdampT = T(fdamp)
    row(f0) = begin
        f = T(f0)
        fm1 = T(1.0) / f
        fm2 = fm1 * fm1
        lorentz = fdampT / (fdampT * fdampT + (f - fringT) * (f - fringT))
        (T(1.0), lorentz, fm1, fm2, fm2 * fm2)
    end

    A = ntuple(k -> row(freqs[idx[k]]), Val(5))
    b = ntuple(k -> T(pvals[idx[k]]), Val(5))
    c = _xas_linsolve(A, b)
    return (c[1], c[2], c[3], c[4], c[5])
end

"""
    _xhm_inter_phase_dphidf(f, l, m, c, fring, fdamp)

`dPhi/df` intermediate ansatz (`IMRPhenomXHM_Inter_Phase_Ansatz`,
`LALSimIMRPhenomXHM_intermediate.c` lines 3285-3312, `modeTag != 32` branch): `c0 + c1/f + c2/f^2 +
c4/f^4 + cL*fdamp/(fdamp^2+(f-fring)^2)`. `c = (c0,cL,c1,c2,c4)` from
[`_xhm_inter_phase_coefficients`](@ref).
"""
@inline function _xhm_inter_phase_dphidf(f, c, fring, fdamp)
    T = promote_type(typeof(f), typeof(fring), typeof(fdamp), typeof(c[1]))
    @Tlit T begin
        c0, cL, c1, c2, c4 = c
        invf = 1.0 / f
        invf2 = invf * invf
        invf4 = invf2 * invf2
        return c0 + c1 * invf + c2 * invf2 + c4 * invf4 + cL * fdamp / (fdamp * fdamp + (f - fring) * (f - fring))
    end
end

"""
    _xhm_inter_phase_ansatzint(f, c, fring, fdamp)

`Phi(f)` -- the antiderivative of [`_xhm_inter_phase_dphidf`](@ref), matching
`IMRPhenomXHM_Inter_Phase_AnsatzInt`'s `modeTag != 32` branch
(`LALSimIMRPhenomXHM_intermediate.c` lines 3258-3283): `c0*f + c1*log(f) - c2/f - c4/(3f^3) +
cL*atan((f-fring)/fdamp)`.
"""
@inline function _xhm_inter_phase_ansatzint(f, c, fring, fdamp)
    T = promote_type(typeof(f), typeof(fring), typeof(fdamp), typeof(c[1]))
    @Tlit T begin
        c0, cL, c1, c2, c4 = c
        invf = 1.0 / f
        invf3 = invf * invf * invf
        return c0 * f + c1 * log(f) - c2 * invf - (1.0 / 3.0) * c4 * invf3 + cL * atan((f - fring) / fdamp)
    end
end

"""
    _xhm_inter_phase_glue(c, fring, fdamp, fcutInsp, insp_phi, insp_dphi, fcutRD, dphi0RD, phi0RD)

`(C1INSP, CINSP, C1RD, CRD)`: the C1-continuity additive/linear-in-`f` correction constants that
glue the raw intermediate-phase ansatz to the neighbouring inspiral and
ringdown (`dphi0RD, phi0RD` are caller-supplied, see module docstring)
regions, matching `_internals.c` lines 2323-2331 (`C1INSP = Inter_Ansatz(fcutInsp) -
Insp_Ansatz(fcutInsp)`, etc). `insp_phi, insp_dphi` are the inspiral phase
ansatz's value/derivative at `fcutInsp` (`IMRPhenomXHM_inspiral.jl`); this function does
not compute them itself to keep the file's dependency direction one-way (intermediate depends on
inspiral, not vice versa).
"""
@inline function _xhm_inter_phase_glue(c, fring, fdamp, fcutInsp, insp_dphi, insp_phi, fcutRD, dphi0RD, phi0RD)
    C1INSP = _xhm_inter_phase_dphidf(fcutInsp, c, fring, fdamp) - insp_dphi
    CINSP = -(C1INSP * fcutInsp) + _xhm_inter_phase_ansatzint(fcutInsp, c, fring, fdamp) - insp_phi
    C1RD = _xhm_inter_phase_dphidf(fcutRD, c, fring, fdamp) - dphi0RD
    CRD = -C1RD * fcutRD + _xhm_inter_phase_ansatzint(fcutRD, c, fring, fdamp) - phi0RD
    return C1INSP, CINSP, C1RD, CRD
end
