"""
    IMRPhenomXHM inspiral-region amplitude/phase ansatze (this file)

The inspiral-region amplitude AND phase ansatze
for the `(l,m) = (2,1), (3,3), (3,2), (4,4)` higher harmonics that `IMRPhenomXHM` adds on top of
`IMRPhenomXAS`'s existing dominant `(2,2)` mode. Ported directly from LALSuite's
`LALSimIMRPhenomXHM_inspiral.c`/`.h` (local checkout:
`lalsuite/lalsimulation/lib/LALSimIMRPhenomXHM_inspiral.c`), plus the small amount of
`LALSimIMRPhenomXHM_internals.c` glue needed to actually evaluate these ansatze end-to-end (the
`.h` only declares fit/ansatz *pieces* -- the "how do I get from `(eta,chi1,chi2)` to the `phi[]`/
`InspiralCoefficient[]` arrays these ansatze consume" wiring lives in `_internals.c`; without it
there would be nothing testable). Ringdown QNM fits (`IMRPhenomXHM_qnm.jl`), the
intermediate-region ansatz (`IMRPhenomXHM_intermediate.jl`) and mode-mixing/full assembly (`IMRPhenomXHM.jl`) are in their own files.

# Working in dimensionless `Mf`, not physical Hz
Exactly like the LALSuite functions they port, every function in this file operates on the
dimensionless frequency `Mf` (total-mass-scaled), not physical Hz -- there is no `mc`/`dL`-aware
`Phi`/`Ampl` wrapper here (that full-waveform assembly, including the physical-unit conversion and
the spin-weighted-spherical-harmonic mode sum, is in `IMRPhenomXHM.jl`). This matches how
`IMRPhenomXHM_Inspiral_Amp_Ansatz`/`IMRPhenomXHM_Inspiral_Phase_Ansatz` themselves are written in
LALSuite -- they are called from a fully physical-unit-free part of the C code.

# The C source's TWO configuration eras, and which one this port targets
`LALSimIMRPhenomXHM_inspiral.c`'s functions are version-switched (`InspAmpFlag`/`InspPhaseFlag`
arguments, `pWFHM->IMRPhenomXHM*Version` fields) between an older "122018/122019" calibration and
a newer "122022" one. `PhenomXHMReleaseVersion`'s default (confirmed by reading
`LALSimInspiralWaveformParams.c`: `DEFINE_INSERT_FUNC(PhenomXHMReleaseVersion, INT4,
"PhenomXHMReleaseVersion", 122022)`) is **122022** -- so that is what real LALSuite actually
generates unless a caller explicitly overrides it, and it is what this port targets and what the
LALSuite comparison validates against. The legacy 122018/122019 branches are NOT ported
(a real, deliberate scope decision, not an oversight -- see "Dead code under the default release"
below for the specific functions this excludes).

# Three genuine dead-code findings under the DEFAULT (122022) release (confirmed by reading the
# source, not assumed) -- each is a real judgment call worth flagging for re-verification:

1. **`IMRPhenomXHM_Inspiral_PNAmp_21Ansatz` and `Get21PNAmplitudeCoefficients` (the `(2,1)`-mode's
   alternate TaylorT4-based ansatz) are unreachable under the default release.** They only run when
   `pWFHM->useFAmpPN == 1`, and `LALSimIMRPhenomXHM_internals.c` line 102 initializes
   `wf->useFAmpPN = 0` unconditionally; it is only ever set to `1` inside the
   `IMRPhenomXHMReleaseVersion == 122019` branch (line 270, for extreme mass ratios). Since the
   default release is 122022, `useFAmpPN` stays `0` for every mode including `(2,1)`, so the
   generic `IMRPhenomXHM_Inspiral_PNAmp_Ansatz` (the `cabs(...)`-based power-series form) is what
   actually runs for `(2,1)` too, using the `else` branch of `case 21` inside
   `IMRPhenomXHM_GetPNAmplitudeCoefficients` (NOT `Get21PNAmplitudeCoefficients`). **Not ported**:
   `IMRPhenomXHM_Inspiral_PNAmp_21Ansatz`, `Get21PNAmplitudeCoefficients`. Ported instead: the
   `(2,1)`-mode `else`-branch PN coefficients (`_xhm_pn_amplitude_coefficients`'s `modeTag == 21`
   case), which the default release always uses.
2. **`IMRPhenomXHM_Inspiral_Amplitude_Veto`/`WavyPoints` are wired up ONLY in the legacy release
   path.** `IMRPhenomXHM_Get_Inspiral_Amp_Coefficients` (the entry point the 122022 release calls,
   `LALSimIMRPhenomXHM_internals.c` line ~1229) contains only a `// Do Vetos? set final
   pWFHM->IMRPhenomXHMInspiralAmpVersion` comment with no actual call; the veto/wavy-point logic
   that *is* wired up (lines ~1419-1448) sits inside the `else` (`IMRPhenomXHMReleaseVersion ==
   122019`) branch of `IMRPhenomXHM_GetAmplitudeCoefficients`. So under the default release, all 3
   inspiral collocation points are unconditionally used (`IMRPhenomXHMInspiralAmpVersion == 123`
   always) -- the veto/wavy-point downgrade path this file's own comment gestures at simply never
   fires. **Ported anyway** (`_xhm_inspiral_amplitude_veto`/`_xhm_wavy_points`) since
   they are cheap, self-contained, pure functions --
   documented here as dead code under the configuration validated here.
3. **The phenomenological "lambda" phase parameter-space fits
   (`IMRPhenomXHM_Insp_Phase_{21,33,32,44}_lambda`) are only used for `eta <= 0.01`
   (roughly `q >~ 98`).** `IMRPhenomXHM_GetPhaseCoefficients` (`_internals.c` lines ~2038-2052):
   `if (pWF22->eta > 0.01) { pPhase->LambdaPN = IMRPhenomXHM_Insp_Phase_LambdaPN(eta, modeTag); }
   else { pPhase->LambdaPN = InspiralPhaseFits[modeInt](pWF22, ...); }` -- i.e. for every mass
   ratio in this port's validated range (near-equal-mass through GW190814-like ~1:9, all comfortably
   `eta > 0.01`), the closed-form `LambdaPN` (`IMRPhenomXHM_Insp_Phase_LambdaPN`, ported below as
   `_xhm_insp_phase_lambdapn`) is what actually runs, NOT the phenomenological fits. **Ported
   anyway** (`_xhm_insp_phase_21_lambda`/`_33_lambda`/`_32_lambda`/`_44_lambda`) but they are not exercised by, and not validated via, the LALSuite
   comparison (which only covers `eta > 0.01` configurations) -- a future extreme-mass-ratio task
   would need its own dedicated check of these four functions.

# What the phase ansatz needs from `IMRPhenomXAS.jl`, and why that's the right way to get it
`IMRPhenomXHM_GetPhaseCoefficients` (`_internals.c` lines 2010-2054) builds each higher mode's
inspiral phase coefficients (`pPhase->phi[]`/`phiL[]`) by *rescaling* the dominant `(2,2)` mode's
own already-computed TaylorF2(+pseudo-PN) inspiral phase coefficients
(`phi0..phi8`/`phi5L`/`phi6L`/`phi8L`/`sigma1..sigma4`/`phiNorm`) -- Appendix D of the IMRPhenomXHM
paper (arXiv:2001.10914). Those coefficients are EXACTLY what `IMRPhenomXAS.jl`'s
[`_xas_phi_coeffs`](@ref) already computes and exposes (its `.phi_ins` NamedTuple field) as part of
its own already-LALSuite-validated inspiral phase. Rather than re-deriving/duplicating that ~60-line
block here (drift risk, exactly the kind of thing this package's own conventions warn against; see
e.g. `IMRPhenomXPrecession.jl`'s reuse of `_xas`/normalization-fix precedent), this file calls
`_xas_phi_coeffs` directly and extracts `.phi_ins`. Two LALSuite struct fields referenced by the
rescaling code (`pPhase22->phi_minus2`, `phi_minus1`, `phi9`, `phi9L`, `sigma5`) are confirmed
**always zero** under the actual default configuration: `LALSimIMRPhenomX_internals.c` initializes
them to `0.0` and only sets them nonzero under `IMRPhenomXInspiralPhaseVersion` values 105/114/115,
while `LALSimInspiralWaveformParams.c` confirms the default is **104**
(`DEFINE_INSERT_FUNC(PhenomXInspiralPhaseVersion, INT4, "InsPhaseVersion", 104)`, `NPseudoPN = 4`
under 104 per `LALSimIMRPhenomX_internals.c` line ~1368) -- exactly the 4-pseudo-PN-coefficient
ansatz `IMRPhenomXAS.jl` already implements. So treating those fields as identically zero here is
not an approximation; it is the exact behaviour of real LALSuite's default configuration.

# Amplitude/phase inspiral-intermediate matching frequency
Both the amplitude and phase inspiral ansatze cut off at the SAME frequency under the default
release: `IMRPhenomXHM_Amplitude_fcutInsp`'s `case 122022` (`_internals.c` lines 782-794) reduces
to `fcut = pWFHM->fMECOlm` whenever `pWF22->q < 20` (true for this port's entire validated grid,
which only goes up to ~1:9); `IMRPhenomXHM_Intermediate_CollocPtsFreqs` unconditionally sets
`pPhase->fPhaseMatchIN = pWFHM->fMECOlm` too (line 926, no version gate at all). `fMECOlm` itself is
`wf22->fMECO * emm * 0.5` (`_internals.c` line 99) -- the SAME `(2,2)`-mode MECO frequency
`IMRPhenomXAS.jl`'s [`_xas_fMECO`](@ref) already computes, just rescaled by the mode's own `emm/2`.
The `q >= 20` extreme-mass-ratio branch of `fcutInsp` (a separate closed-form fit, `fcutEMR`) is
**not ported** here -- out of this port's validated mass-ratio range, and would need its own
dedicated validation if a future task needs it.

# `(l,m)` <-> `emm`/`modeInt`/`modeTag` bookkeeping (confirmed via
`IMRPhenomXHM_SetHMWaveformVariables`, `_internals.c` lines 94-137)
| (l,m)  | `modeTag` (=10l+m) | `modeInt` (0-based array index) | `emm` |
|--------|--------------------|----------------------------------|-------|
| (2,1)  | 21                 | 0                                | 1     |
| (3,3)  | 33                 | 1                                | 3     |
| (3,2)  | 32                 | 2                                | 2     |
| (4,4)  | 44                 | 3                                | 4     |
Every function in this file dispatches on `(l,m)` via a plain `if`/`elseif` chain over `Int`
literals (matching `IMRPhenomXHM_qnm.jl`'s `_xhm_fring_fdamp` convention), not a generated/`Val`-
based dispatch.
"""

# ================================================================================================
# AMPLITUDE: collocation-point parameter-space fits ("iv1"/"iv2"/"iv3" = value at
# 0.5/0.75/1.0 * fcutInsp), `case 122022` only (the default `IMRPhenomXHMInspiralAmpFitsVersion`
# -- see module docstring). Ported from `IMRPhenomXHM_Insp_Amp_{21,33,32,44}_iv{1,2,3}`,
# `LALSimIMRPhenomXHM_inspiral.c` lines 48-537. Arguments match the C source's local variables:
# `eta = pWF->eta`, `delta = pWF->delta` (= this package's `Seta`), `S = pWF->chiPNHat`,
# `chidiff = pWF->dchi_half` (= half of `chi1 - chi2`).
# ================================================================================================

@inline function _xhm_insp_amp_21_iv1(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        return abs(
            chidiff1 * eta1^5 * (-3962.5020052272976 + 987.635855365408 * S1 - 134.98527058315528 * S2) +
            delta * (19.30531354642419 + 16.6640319856064 * eta1 - 120.58166037019478 * eta2 + 220.77233521626252 * eta3) * sqroot +
            chidiff1 * delta * (31.364509907424765 * eta1 - 843.6414532232126 * eta2 + 2638.3077554662905 * eta3) * sqroot +
            chidiff1 * delta * (32.374226994179054 * eta1 - 202.86279451816662 * eta2 + 347.1621871204769 * eta3) * S1 * sqroot +
            delta * S1 * (
                -16.75726972301224 * (1.1787350890261943 - 7.812073811917883 * eta1 + 99.47071002831267 * eta2 - 500.4821414428368 * eta3 + 876.4704270866478 * eta4) +
                2.3439955698372663 * (0.9373952326655807 + 7.176140122833879 * eta1 - 279.6409723479635 * eta2 + 2178.375177755584 * eta3 - 4768.212511142035 * eta4) * S1
            ) * sqroot
        )
    end
end

@inline function _xhm_insp_amp_21_iv2(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
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
            chidiff1 * eta5 * (-2898.9172078672705 + 580.9465034962822 * S1 + 22.251142639924076 * S2) +
            delta * (
                chidiff2 * (-18.541685007214625 * eta1 + 166.7427445020744 * eta2 - 417.5186332459383 * eta3) +
                chidiff1 * (41.61457952037761 * eta1 - 779.9151607638761 * eta2 + 2308.6520892707795 * eta3)
            ) * sqroot +
            delta * (11.414934585404561 + 30.883118528233638 * eta1 - 260.9979123967537 * eta2 + 1046.3187137392433 * eta3 - 1556.9475493549746 * eta4) * sqroot +
            delta * S1 * (
                -10.809007068469844 * (1.1408749895922659 - 18.140470190766937 * eta1 + 368.25127088896744 * eta2 - 3064.7291458207815 * eta3 + 11501.848278358668 * eta4 - 16075.676528787526 * eta5) +
                1.0088254664333147 * (1.2322739396680107 - 192.2461213084741 * eta1 + 4257.760834055382 * eta2 - 35561.24587952242 * eta3 + 130764.22485304279 * eta4 - 177907.92440833704 * eta5) * S1
            ) * sqroot +
            delta * (
                chidiff1 * (36.88578491943111 * eta1 - 321.2569602623214 * eta2 + 748.6659668096737 * eta3) * S1 +
                chidiff1 * (-95.42418611585117 * eta1 + 1217.338674959742 * eta2 - 3656.192371615541 * eta3) * S2
            ) * sqroot
        )
    end
end

@inline function _xhm_insp_amp_21_iv3(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        return abs(
            chidiff1 * eta5 * (-2282.9983216879655 + 157.94791186394787 * S1 + 16.379731479465033 * S2) +
            chidiff1 * delta * (21.935833431534224 * eta1 - 460.7130131927895 * eta2 + 1350.476411541137 * eta3) * sqroot +
            delta * (5.390240326328237 + 69.01761987509603 * eta1 - 568.0027716789259 * eta2 + 2435.4098320959706 * eta3 - 3914.3390484239667 * eta4) * sqroot +
            chidiff1 * delta * (29.731007410186827 * eta1 - 372.09609843131386 * eta2 + 1034.4897198648962 * eta3) * S1 * sqroot +
            delta * S1 * (
                -7.1976397556450715 * (0.7603360145475428 - 6.587249958654174 * eta1 + 120.87934060776237 * eta2 - 635.1835857158857 * eta3 + 1109.0598539312573 * eta4) -
                0.0811847192323969 * (7.951454648295709 + 517.4039644814231 * eta1 - 9548.970156895082 * eta2 + 52586.63520999897 * eta3 - 93272.17990295641 * eta4) * S1 -
                0.28384547935698246 * (-0.8870770459576875 + 180.0378964169756 * eta1 - 2707.9572896559484 * eta2 + 14158.178124971111 * eta3 - 24507.800226675925 * eta4) * S2
            ) * sqroot
        )
    end
end

@inline function _xhm_insp_amp_33_iv1(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        return chidiff1 * eta5 * (155.1434307076563 + 26.852777193715088 * S1 + 1.4157230717300835 * S2) +
               chidiff1 * delta * (6.296698171560171 * eta1 + 15.81328761563562 * eta2 - 141.85538063933927 * eta3) * sqroot +
               delta * (20.94372147101354 + 68.14577638017842 * eta1 - 898.470298591732 * eta2 + 4598.64854748635 * eta3 - 8113.199260593833 * eta4) * sqroot +
               chidiff1 * delta * (29.221863857271703 * eta1 - 348.1658322276406 * eta2 + 965.4670353331536 * eta3) * S1 * sqroot +
               delta * S1 * (
                   -9.753610761811967 * (1.7819678168496158 - 44.07982999150369 * eta1 + 750.8933447725581 * eta2 - 5652.44754829634 * eta3 + 19794.855873435758 * eta4 - 26407.40988450443 * eta5) +
                   0.014210376114848208 * (-196.97328616330392 + 7264.159472864562 * eta1 - 125763.47850622259 * eta2 + 1.1458022059130718e6 * eta3 - 4.948175330328345e6 * eta4 + 7.911048294733888e6 * eta5) * S1 -
                   0.26859293613553986 * (-8.029069605349488 + 888.7768796633982 * eta1 - 16664.276483466252 * eta2 + 128973.72291098491 * eta3 - 462437.2690007375 * eta4 + 639989.1197424605 * eta5) * S2
               ) * sqroot
    end
end

@inline function _xhm_insp_amp_33_iv2(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        return chidiff1 * eta5 * (161.62678370819597 + 37.141092711336846 * S1 - 0.16889712161410445 * S2) +
               chidiff1 * delta * (3.4895829486899825 * eta1 + 51.07954458810889 * eta2 - 249.71072528701757 * eta3) * sqroot +
               delta * (12.501397517602173 + 35.75290806646574 * eta1 - 357.6437296928763 * eta2 + 1773.8883882162215 * eta3 - 3100.2396041211605 * eta4) * sqroot +
               chidiff1 * delta * (13.854211287141906 * eta1 - 135.54916401086845 * eta2 + 327.2467193417936 * eta3) * S1 * sqroot +
               delta * S1 * (
                   -5.2580116732827085 * (1.7794900975289085 - 48.20753331991333 * eta1 + 861.1650630146937 * eta2 - 6879.681319382729 * eta3 + 25678.53964955809 * eta4 - 36383.824902258915 * eta5) +
                   0.028627002336747746 * (-50.57295946557892 + 734.7581857539398 * eta1 - 2287.0465658878725 * eta2 + 15062.821881048358 * eta3 - 168311.2370167227 * eta4 + 454655.37836367317 * eta5) * S1 -
                   0.15528289788512326 * (-12.738184090548508 + 1129.44485109116 * eta1 - 25091.14888164863 * eta2 + 231384.03447562453 * eta3 - 953010.5908118751 * eta4 + 1.4516597366230418e6 * eta5) * S2
               ) * sqroot
    end
end

@inline function _xhm_insp_amp_33_iv3(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        eta7 = eta1 * eta6
        S1 = S
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return chidiff1 * delta * (-0.5869777957488564 * eta1 + 32.65536124256588 * eta2 - 110.10276573567405 * eta3) +
               chidiff1 * delta * (3.524800489907584 * eta1 - 40.26479860265549 * eta2 + 113.77466499598913 * eta3) * S1 +
               delta * S1 * (
                   -1.2846335585108297 * (0.09991079016763821 + 1.37856806162599 * eta1 + 23.26434219690476 * eta2 - 34.842921754693386 * eta3 - 70.83896459998664 * eta4) -
                   0.03496714763391888 * (-0.230558571912664 + 188.38585449575902 * eta1 - 3736.1574640444287 * eta2 + 22714.70643022915 * eta3 - 43221.0453556626 * eta4) * S1
               ) +
               chidiff1 * eta7 * (2667.3441342894776 + 47.94869769580204 * chidiff2 + 793.5988192446642 * S1 + 293.89657731755483 * S2) +
               delta * (5.148353856800232 + 148.98231189649468 * eta1 - 2774.5868652930294 * eta2 + 29052.156454239772 * eta3 - 162498.31493332976 * eta4 + 460912.76402476896 * eta5 - 521279.50781871413 * eta6) * sqroot
    end
end

@inline function _xhm_insp_amp_32_iv1(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return (
                   chidiff1 * delta * (-0.739317114582042 * eta1 - 47.473246070362634 * eta2 + 278.9717709112207 * eta3 - 566.6420939162068 * eta4) +
                   chidiff2 * (-0.5873680378268906 * eta1 + 6.692187014925888 * eta2 - 24.37776782232888 * eta3 + 23.783684827838247 * eta4)
               ) * sqroot +
               (3.2940434453819694 + 4.94285331708559 * eta1 - 343.3143244815765 * eta2 + 3585.9269057886418 * eta3 - 19279.186145681153 * eta4 + 51904.91007211022 * eta5 - 55436.68857586653 * eta6) * sqroot +
               chidiff1 * delta * (12.488240781993923 * eta1 - 209.32038774208385 * eta2 + 1160.9833883184604 * eta3 - 2069.5349737049073 * eta4) * S1 * sqroot +
               S1 * (
                   0.6343034651912586 * (-2.5844888818001737 + 78.98200041834092 * eta1 - 1087.6241783616488 * eta2 + 7616.234910399297 * eta3 - 24776.529123239357 * eta4 + 30602.210950069973 * eta5) -
                   0.062088720220899465 * (6.5586380356588565 + 36.01386705325694 * eta1 - 3124.4712274775407 * eta2 + 33822.437731298516 * eta3 - 138572.93700180828 * eta4 + 198366.10615196894 * eta5) * S1
               ) * sqroot
    end
end

@inline function _xhm_insp_amp_32_iv2(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return (
                   chidiff2 * (-0.03940151060321499 * eta1 + 1.9034209537174116 * eta2 - 8.78587250202154 * eta3) +
                   chidiff1 * delta * (-1.704299788495861 * eta1 - 4.923510922214181 * eta2 + 0.36790005839460627 * eta3)
               ) * sqroot +
               (2.2911849711339123 - 5.1846950040514335 * eta1 + 60.10368251688146 * eta2 - 1139.110227749627 * eta3 + 7970.929280907627 * eta4 - 25472.73682092519 * eta5 + 30950.67053883646 * eta6) * sqroot +
               S1 * (
                   0.7718201508695763 * (-1.3012906461000349 + 26.432880113146012 * eta1 - 186.5001124789369 * eta2 + 712.9101229418721 * eta3 - 970.2126139442341 * eta4) +
                   0.04832734931068797 * (-5.9999628512498315 + 78.98681284391004 * eta1 + 1.8360177574514709 * eta2 - 2537.636347529708 * eta3 + 6858.003573909322 * eta4) * S1
               ) * sqroot
    end
end

@inline function _xhm_insp_amp_32_iv3(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return (
                   chidiff2 * (-0.6358511175987503 * eta1 + 5.555088747533164 * eta2 - 14.078156877577733 * eta3) +
                   chidiff1 * delta * (0.23205448591711159 * eta1 - 19.46049432345157 * eta2 + 36.20685853857613 * eta3)
               ) * sqroot +
               (1.1525594672495008 + 7.380126197972549 * eta1 - 17.51265776660515 * eta2 - 976.9940395257111 * eta3 + 8880.536804741967 * eta4 - 30849.228936891763 * eta5 + 38785.53683146884 * eta6) * sqroot +
               chidiff1 * delta * (1.904350804857431 * eta1 - 25.565242391371093 * eta2 + 80.67120303906654 * eta3) * S1 * sqroot +
               S1 * (
                   0.785171689871352 * (-0.4634745514643032 + 18.70856733065619 * eta1 - 167.9231114864569 * eta2 + 744.7699462372949 * eta3 - 1115.008825153004 * eta4) +
                   0.13469300326662165 * (-2.7311391326835133 + 72.17373498208947 * eta1 - 483.7040402103785 * eta2 + 1136.8367114738041 * eta3 - 472.02962341590774 * eta4) * S1
               ) * sqroot
    end
end

@inline function _xhm_insp_amp_44_iv1(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
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
        return (
                   chidiff1 * delta * (0.5697308729057493 * eta1 + 8.895576813118867 * eta2 - 34.98399465240273 * eta3) +
                   chidiff2 * (1.6370346538130884 * eta1 - 14.597095790380884 * eta2 + 33.182723737396294 * eta3)
               ) * sqroot +
               (5.2601381002242595 - 3.557926105832778 * eta1 - 138.9749850448088 * eta2 + 603.7453704122706 * eta3 - 923.5495700703648 * eta4) * sqroot +
               S1 * (
                   -0.41839636169678796 * (5.143510231379954 + 104.62892421207803 * eta1 - 4232.508174045782 * eta2 + 50694.024801783446 * eta3 - 283097.33358214336 * eta4 + 758333.2655404843 * eta5 - 788783.0559069642 * eta6) -
                   0.05653522061311774 * (5.605483124564013 + 694.00652410087 * eta1 - 17551.398321516353 * eta2 + 165236.6480734229 * eta3 - 761661.9645651339 * eta4 + 1.7440315410044065e6 * eta5 - 1.6010489769238676e6 * eta6) * S1 -
                   0.023693246676754775 * (16.437107575918503 - 2911.2154288136217 * eta1 + 89338.32554683842 * eta2 - 1.0803340811860575e6 * eta3 + 6.255666490084672e6 * eta4 - 1.7434160932177313e7 * eta5 + 1.883460394974573e7 * eta6) * S2
               ) * sqroot
    end
end

@inline function _xhm_insp_amp_44_iv2(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
        eta1 = eta
        eta2 = eta1 * eta1
        eta3 = eta1 * eta2
        eta4 = eta1 * eta3
        eta5 = eta1 * eta4
        eta6 = eta1 * eta5
        S1 = S
        S2 = S1 * S1
        S3 = S1 * S2
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return (
                   chidiff2 * (-0.8318312659717388 * eta1 + 7.6541168007977864 * eta2 - 16.648660653220123 * eta3) +
                   chidiff1 * delta * (2.214478316304753 * eta1 - 7.028104574328955 * eta2 + 5.56587823143958 * eta3)
               ) * sqroot +
               (3.173191054680422 + 6.707695566702527 * eta1 - 155.22519772642607 * eta2 + 604.0067075996933 * eta3 - 876.5048298377644 * eta4) * sqroot +
               chidiff1 * delta * (4.749663394334708 * eta1 - 42.62996105525792 * eta2 + 97.01712147349483 * eta3) * S1 * sqroot +
               S1 * (
                   -0.2627203100303006 * (6.460396349297595 - 52.82425783851536 * eta1 - 552.1725902144143 * eta2 + 12546.255587592654 * eta3 - 81525.50289542897 * eta4 + 227254.37897941095 * eta5 - 234487.3875219032 * eta6) -
                   0.008424003742397579 * (-109.26773035716548 + 15514.571912666677 * eta1 - 408022.6805482195 * eta2 + 4.620165968920881e6 * eta3 - 2.6446950627957724e7 * eta4 + 7.539643948937692e7 * eta5 - 8.510662871580401e7 * eta6) * S1 -
                   0.008830881730801855 * (-37.49992494976597 + 1359.7883958101172 * eta1 - 23328.560285901796 * eta2 + 260027.4121353132 * eta3 - 1.723865744472182e6 * eta4 + 5.858455766230802e6 * eta5 - 7.756341721552802e6 * eta6) * S2 -
                   0.027167813927224657 * (34.281932237450256 - 3312.7658728016568 * eta1 + 84126.14531363266 * eta2 - 956052.0170024392 * eta3 + 5.570748509263883e6 * eta4 - 1.6270212243584689e7 * eta5 + 1.8855858173287075e7 * eta6) * S3
               ) * sqroot
    end
end

@inline function _xhm_insp_amp_44_iv3(eta, delta, S, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(S), typeof(chidiff))
    @Tlit T begin
        sqroot = sqrt(eta)
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
        return (
                   chidiff1 * delta * (1.4739380748149558 * eta1 + 0.06541707987699942 * eta2 - 9.473290540936633 * eta3) +
                   chidiff2 * (-0.3640838331639651 * eta1 + 3.7369795937033756 * eta2 - 8.709159662885131 * eta3)
               ) * sqroot +
               (1.7335503724888923 + 12.656614578053683 * eta1 - 139.6610487470118 * eta2 + 456.78649322753824 * eta3 - 599.2709938848282 * eta4) * sqroot +
               chidiff1 * delta * (2.3532739003216254 * eta1 - 21.37216554136868 * eta2 + 53.35003268489743 * eta3) * S1 * sqroot +
               S1 * (
                   -0.15782329022461472 * (6.0309399412954345 - 229.16361598098678 * eta1 + 3777.477006415653 * eta2 - 31109.307191210424 * eta3 + 139319.8239886073 * eta4 - 324891.4001578353 * eta5 + 307714.3954026392 * eta6) -
                   0.03050157254864058 * (4.232861441291087 + 1609.4251694451375 * eta1 - 51213.27604422822 * eta2 + 612317.1751155312 * eta3 - 3.5589766538499263e6 * eta4 + 1.0147654212772278e7 * eta5 - 1.138861230369246e7 * eta6) * S1 -
                   0.026407497690308382 * (-17.184685557542196 + 744.4743953122965 * eta1 - 10494.512487701073 * eta2 + 66150.52694069289 * eta3 - 184787.79377504133 * eta4 + 148102.4257785174 * eta5 + 128167.89151782403 * eta6) * S2
               ) * sqroot
    end
end

"""
    _xhm_insp_amp_collocation_values(l, m, eta, delta, S, chidiff)

`(iv1, iv2, iv3)` -- the 3 inspiral amplitude collocation-point values for mode `(l,m)`, i.e.
`abs(fit(eta,delta,S,chidiff))` for that mode's `_iv1`/`_iv2`/`_iv3` parameter-space fits (the
`abs` matches `IMRPhenomXHM_Inspiral_Amp_CollocationPoints`, `_internals.c` line 885:
`pAmp->CollocationPointsValuesAmplitudeInsp[i] = fabs(pAmp->InspiralAmpFits[...](...))`).
"""
@inline function _xhm_insp_amp_collocation_values(l::Integer, m::Integer, eta, delta, S, chidiff)
    if l == 2 && m == 1
        return abs(_xhm_insp_amp_21_iv1(eta, delta, S, chidiff)),
               abs(_xhm_insp_amp_21_iv2(eta, delta, S, chidiff)),
               abs(_xhm_insp_amp_21_iv3(eta, delta, S, chidiff))
    elseif l == 3 && m == 3
        return abs(_xhm_insp_amp_33_iv1(eta, delta, S, chidiff)),
               abs(_xhm_insp_amp_33_iv2(eta, delta, S, chidiff)),
               abs(_xhm_insp_amp_33_iv3(eta, delta, S, chidiff))
    elseif l == 3 && m == 2
        return abs(_xhm_insp_amp_32_iv1(eta, delta, S, chidiff)),
               abs(_xhm_insp_amp_32_iv2(eta, delta, S, chidiff)),
               abs(_xhm_insp_amp_32_iv3(eta, delta, S, chidiff))
    elseif l == 4 && m == 4
        return abs(_xhm_insp_amp_44_iv1(eta, delta, S, chidiff)),
               abs(_xhm_insp_amp_44_iv2(eta, delta, S, chidiff)),
               abs(_xhm_insp_amp_44_iv3(eta, delta, S, chidiff))
    else
        throw(ArgumentError("_xhm_insp_amp_collocation_values: unsupported (l,m) = ($l,$m); only (2,1),(3,3),(3,2),(4,4) are implemented"))
    end
end

# ================================================================================================
# AMPLITUDE: Post-Newtonian (complex) coefficients of the Fourier-domain inspiral PN series, per
# mode. Ported from `IMRPhenomXHM_GetPNAmplitudeCoefficients`, `LALSimIMRPhenomXHM_internals.c`
# lines 969-1057 -- specifically the `useFAmpPN == 0` (default) branch of `case 21`, plus the
# unconditional `case 33`/`case 32`/`case 44` branches (see module docstring, dead-code finding 1,
# for why `case 21`'s `useFAmpPN == 1` branch and `Get21PNAmplitudeCoefficients` are not ported).
# `chiS = (chi1L+chi2L)/2`, `chiA = (chi1L-chi2L)/2` (`pWFHM->chi_s`/`chi_a`, `_internals.c` lines
# 389-390); `delta` = this package's `Seta`.
# ================================================================================================

"""
    _xhm_pn_amplitude_coefficients(l, m, eta, delta, chiS, chiA)

`(pnInitial, pnOneThird, pnTwoThirds, pnThreeThirds, pnFourThirds, pnFiveThirds, pnSixThirds)` --
the (generally complex) Fourier-domain PN inspiral-amplitude series coefficients for mode `(l,m)`,
each already multiplied by the mode's own `(2/emm)^{k/3}` rescaling power (matching
`powers_of_2d1`/`_2d3`/`_2d4` in the C source; `(3,2)` needs none since `emm=2` makes `2/emm=1`).
"""
@inline function _xhm_pn_amplitude_coefficients(l::Integer, m::Integer, eta, delta, chiS, chiA)
    Tr = promote_type(typeof(eta), typeof(delta), typeof(chiS), typeof(chiA))
    T = Complex{Tr}
    zeroT = zero(T)
    if l == 2 && m == 1
        @Tlit Tr begin
            p2 = 2.0
            p2_13 = cbrt(p2)
            p2_23 = p2_13 * p2_13
            p2_43 = p2_23 * p2_23
            p2_53 = p2_43 * p2_13
            p2_2 = p2 * p2
            eta2 = eta * eta
            pnInitial = zeroT
            pnOneThird = T(delta * pi^(1.0 / 3.0) * p2_13)
            pnTwoThirds = T((-3.0 * (chiA + chiS * delta)) / 2.0 * pi^(2.0 / 3.0) * p2_23)
            pnThreeThirds = T((335.0 * delta + 1404.0 * delta * eta) / 672.0 * pi * p2)
            pnFourThirds = (T(3427.0 * chiA + 3427.0 * chiS * delta - 8404.0 * chiA * eta - 3860.0 * chiS * delta * eta - 1344.0 * delta * pi) -
                             im * T(672.0 * delta + 672.0 * delta * log(16.0))) / T(1344.0) * T(pi^(4.0 / 3.0) * p2_43)
            pnFiveThirds = T(
                (-155965824.0 * chiA * chiS - 964357.0 * delta + 432843264.0 * chiA * chiS * eta - 23670792.0 * delta * eta +
                 24385536.0 * chiA * pi + 24385536.0 * chiS * delta * pi - 77982912.0 * delta * chiA^2 + 81285120.0 * delta * eta * chiA^2 -
                 77982912.0 * delta * chiS^2 + 39626496.0 * delta * eta * chiS^2 + 21535920.0 * delta * eta2) / 8.128512e6 * pi^(5.0 / 3.0) * p2_53
            )
            pnSixThirds = (
                T(143063173.0 * chiA + 143063173.0 * chiS * delta - 546199608.0 * chiA * eta - 169191096.0 * chiS * delta * eta -
                  9898560.0 * delta * pi + 20176128.0 * delta * eta * pi + 61725888.0 * chiS * delta * chiA^2 - 81285120.0 * chiS * delta * eta * chiA^2 +
                  20575296.0 * chiA^3 - 81285120.0 * eta * chiA^3 + 61725888.0 * chiA * chiS^2 - 165618432.0 * chiA * eta * chiS^2 +
                  20575296.0 * delta * chiS^3 - 1016064.0 * delta * eta * chiS^3 + 128873808.0 * chiA * eta2 - 3859632.0 * chiS * delta * eta2) -
                im * T(1350720.0 * delta + 72043776.0 * delta * eta + 5402880.0 * delta * log(2.0) + 17224704.0 * delta * eta * log(2.0))
            ) / T(5.419008e6) * T(pi^2 * p2_2)
        end
        return pnInitial, pnOneThird, pnTwoThirds, pnThreeThirds, pnFourThirds, pnFiveThirds, pnSixThirds
    elseif l == 3 && m == 3
        @Tlit Tr begin
            p23 = 2.0 / 3.0
            p23_13 = cbrt(p23)
            p23_23 = p23_13 * p23_13
            p23_43 = p23_23 * p23_23
            p23_53 = p23_43 * p23_13
            p23_2 = p23 * p23
            eta2 = eta * eta
            pnInitial = zeroT
            pnOneThird = T(delta * pi^(1.0 / 3.0) * p23_13)
            pnTwoThirds = zeroT
            pnThreeThirds = T((-1945.0 * delta + 2268.0 * delta * eta) / 672.0 * pi * p23)
            pnFourThirds = (T(325.0 * chiA + 325.0 * chiS * delta - 1120.0 * chiA * eta - 80.0 * chiS * delta * eta + 120.0 * delta * pi) +
                             im * T(-504.0 * delta + 720.0 * delta * log(1.5))) / T(120.0) * T(pi^(4.0 / 3.0) * p23_43)
            pnFiveThirds = T(
                (-2263282560.0 * chiA * chiS - 1077664867.0 * delta + 9053130240.0 * chiA * chiS * eta - 5926068792.0 * delta * eta -
                 1131641280.0 * delta * chiA^2 + 4470681600.0 * delta * eta * chiA^2 - 1131641280.0 * delta * chiS^2 + 55883520.0 * delta * eta * chiS^2 +
                 2966264784.0 * delta * eta2) / 4.4706816e8 * pi^(5.0 / 3.0) * p23_53
            )
            pnSixThirds = (
                T(22007835.0 * chiA + 22007835.0 * chiS * delta - 80190540.0 * chiA * eta - 31722300.0 * chiS * delta * eta -
                  9193500.0 * delta * pi + 17826480.0 * delta * eta * pi - 12428640.0 * chiA * eta2 - 6078240.0 * chiS * delta * eta2) +
                im * T(26467560.0 * delta - 98774368.0 * delta * eta - 37810800.0 * delta * log(1.5) + 37558080.0 * delta * eta * log(1.5))
            ) / T(2.17728e6) * T(pi^2 * p23_2)
        end
        return pnInitial, pnOneThird, pnTwoThirds, pnThreeThirds, pnFourThirds, pnFiveThirds, pnSixThirds
    elseif l == 3 && m == 2
        @Tlit Tr begin
            eta2 = eta * eta
            eta3 = eta2 * eta
            pnInitial = zeroT
            pnOneThird = zeroT
            pnTwoThirds = T((-1.0 + 3.0 * eta) * pi^(2.0 / 3.0))
            pnThreeThirds = T(-4.0 * chiS * eta * pi)
            pnFourThirds = T((10471.0 - 61625.0 * eta + 82460.0 * eta2) / 10080.0 * pi^(4.0 / 3.0))
            pnFiveThirds = (T(-3955.0 * chiS - 3955.0 * chiA * delta - 12600.0 * chiS * eta2 + 10810.0 * chiS * eta + 11865.0 * chiA * delta * eta) +
                             im * T(2520.0 - 11088.0 * eta)) / T(840.0) * T(pi^(5.0 / 3.0))
            pnSixThirds = T(
                (824173699.0 + 2263282560.0 * chiA * chiS * delta - 26069649.0 * eta - 15209631360.0 * chiA * chiS * delta * eta +
                 3576545280.0 * chiS * eta * pi + 1131641280.0 * chiA^2 - 7865605440.0 * eta * chiA^2 + 1131641280.0 * chiS^2 -
                 11870591040.0 * eta * chiS^2 - 13202119896.0 * eta2 + 13412044800.0 * chiA^2 * eta2 + 5830513920.0 * chiS^2 * eta2 +
                 5907445488.0 * eta3) / 4.4706816e8 * pi^2
            )
        end
        return pnInitial, pnOneThird, pnTwoThirds, pnThreeThirds, pnFourThirds, pnFiveThirds, pnSixThirds
    elseif l == 4 && m == 4
        @Tlit Tr begin
            p05 = 0.5
            p05_23 = p05^(2.0 / 3.0)
            p05_43 = p05^(4.0 / 3.0)
            p05_53 = p05^(5.0 / 3.0)
            p05_2 = p05 * p05
            eta2 = eta * eta
            eta3 = eta2 * eta
            pnInitial = zeroT
            pnOneThird = zeroT
            pnTwoThirds = T((1.0 - 3.0 * eta) * pi^(2.0 / 3.0) * p05_23)
            pnThreeThirds = zeroT
            pnFourThirds = T((-158383.0 + 641105.0 * eta - 446460.0 * eta2) / 36960.0 * pi^(4.0 / 3.0) * p05_43)
            pnFiveThirds = (T(565.0 * chiS + 565.0 * chiA * delta - 2075.0 * chiS * eta - 1695.0 * chiA * delta * eta + 240.0 * pi - 720.0 * eta * pi + 1140.0 * chiS * eta2) +
                             im * T(-1008.0 + 3579.0 * eta + 960.0 * log(2.0) - 2880.0 * eta * log(2.0))) / T(120.0) * T(pi^(5.0 / 3.0) * p05_53)
            pnSixThirds = T(
                (7888301437.0 - 147113366400.0 * chiA * chiS * delta - 745140957231.0 * eta + 441340099200.0 * chiA * chiS * delta * eta -
                 73556683200.0 * chiA^2 + 511264353600.0 * eta * chiA^2 - 73556683200.0 * chiS^2 + 224302478400.0 * eta * chiS^2 +
                 2271682065240.0 * eta2 - 871782912000.0 * chiA^2 * eta2 - 10897286400.0 * chiS^2 * eta2 - 805075876080.0 * eta3) /
                2.90594304e10 * pi^2 * p05_2
            )
        end
        return pnInitial, pnOneThird, pnTwoThirds, pnThreeThirds, pnFourThirds, pnFiveThirds, pnSixThirds
    else
        throw(ArgumentError("_xhm_pn_amplitude_coefficients: unsupported (l,m) = ($l,$m); only (2,1),(3,3),(3,2),(4,4) are implemented"))
    end
end

"""
    _xhm_pnamp_ansatz(Mf, pn, PNglobalfactor, ampNorm)

`IMRPhenomXHM_Inspiral_PNAmp_Ansatz`'s generic (non-`(2,1)`-special-cased) branch,
`LALSimIMRPhenomXHM_inspiral.c` lines 668-693: the Fourier-domain PN amplitude series, evaluated
via `cabs` of the complex power series, rescaled by `PNglobalfactor * Mf^{-7/6} * ampNorm`
(`pAmp->InspRescaleFactor == 0`, the default -- see module docstring). `pn` is the 7-tuple from
[`_xhm_pn_amplitude_coefficients`](@ref).
"""
@inline function _xhm_pnamp_ansatz(Mf, pn, PNglobalfactor, ampNorm)
    T = promote_type(typeof(Mf), typeof(PNglobalfactor), typeof(ampNorm), real(typeof(pn[1])))
    @Tlit T begin
        series = pn[1] + Mf^(1.0 / 3.0) * pn[2] + Mf^(2.0 / 3.0) * pn[3] + Mf * pn[4] +
                 Mf^(4.0 / 3.0) * pn[5] + Mf^(5.0 / 3.0) * pn[6] + Mf^2.0 * pn[7]
        return abs(series) * PNglobalfactor * Mf^(-7.0 / 6.0) * ampNorm
    end
end

# ================================================================================================
# AMPLITUDE: legacy `rho1`/`rho2`/`rho3` pseudo-PN coefficient solvers (shared across modes),
# ported from `IMRPhenomXHM_Inspiral_Amp_rho{1,2,3}`, `LALSimIMRPhenomXHM_inspiral.c` lines
# 561-657. `version` = 0,1,2,3 (number of collocation points used in the reconstruction, NOT the
# same numbering as `IMRPhenomXHMInspiralAmpVersion`'s digit-code below). These are called ONLY
# from the legacy (`IMRPhenomXHMReleaseVersion == 122019`) release path in real LALSuite
# (`_internals.c` lines 1499-1501) -- dead code under the default 122022 release validated
# here (see module docstring), ported for completeness.
# ================================================================================================

@inline function _xhm_inspiral_amp_rho1(v1, v2, v3, fcutInsp, f1, f2, f3, version::Integer)
    T = promote_type(typeof(v1), typeof(fcutInsp))
    if version == 3
        return (fcutInsp^(7.0 / 3.0) * (-(f1^3.0 * f3^(8.0 / 3.0) * v2) + f1^(8.0 / 3.0) * f3^3.0 * v2 +
                f2^3.0 * (f3^(8.0 / 3.0) * v1 - f1^(8.0 / 3.0) * v3) + f2^(8.0 / 3.0) * (-(f3^3.0 * v1) + f1^3.0 * v3))) /
               (f1^(7.0 / 3.0) * (f1^(1.0 / 3.0) - f2^(1.0 / 3.0)) * f2^(7.0 / 3.0) * (f1^(1.0 / 3.0) - f3^(1.0 / 3.0)) *
                (f2^(1.0 / 3.0) - f3^(1.0 / 3.0)) * f3^(7.0 / 3.0))
    elseif version == 2
        return (fcutInsp^(7.0 / 3.0) * (-(f2^(8.0 / 3.0) * v1) + f1^(8.0 / 3.0) * v2)) /
               (f1^(7.0 / 3.0) * (f1^(1.0 / 3.0) - f2^(1.0 / 3.0)) * f2^(7.0 / 3.0))
    elseif version == 1
        return (fcutInsp^(7.0 / 3.0) * v1) / f1^(7.0 / 3.0)
    elseif version == 0
        return zero(T)
    else
        throw(ArgumentError("_xhm_inspiral_amp_rho1: version must be 0,1,2,3 (got $version)"))
    end
end

@inline function _xhm_inspiral_amp_rho2(v1, v2, v3, fcutInsp, f1, f2, f3, version::Integer)
    T = promote_type(typeof(v1), typeof(fcutInsp))
    if version == 3
        return (-(fcutInsp^(8.0 / 3.0) * (-(f1^3.0 * f3^(7.0 / 3.0) * v2) + f1^(7.0 / 3.0) * f3^3.0 * v2 +
                f2^3.0 * (f3^(7.0 / 3.0) * v1 - f1^(7.0 / 3.0) * v3) + f2^(7.0 / 3.0) * (-(f3^3.0 * v1) + f1^3.0 * v3)))) /
               (f1^(7.0 / 3.0) * (f1^(1.0 / 3.0) - f2^(1.0 / 3.0)) * f2^(7.0 / 3.0) * (f1^(1.0 / 3.0) - f3^(1.0 / 3.0)) *
                (f2^(1.0 / 3.0) - f3^(1.0 / 3.0)) * f3^(7.0 / 3.0))
    elseif version == 2
        return (-(fcutInsp^(8.0 / 3.0) * (-(f2^(7.0 / 3.0) * v1) + f1^(7.0 / 3.0) * v2))) /
               (f1^(7.0 / 3.0) * (f1^(1.0 / 3.0) - f2^(1.0 / 3.0)) * f2^(7.0 / 3.0))
    elseif version == 1 || version == 0
        return zero(T)
    else
        throw(ArgumentError("_xhm_inspiral_amp_rho2: version must be 0,1,2,3 (got $version)"))
    end
end

@inline function _xhm_inspiral_amp_rho3(v1, v2, v3, fcutInsp, f1, f2, f3, version::Integer)
    T = promote_type(typeof(v1), typeof(fcutInsp))
    if version == 3
        return (fcutInsp^3.0 * (f1^(7.0 / 3.0) * (-f1^(1.0 / 3.0) + f3^(1.0 / 3.0)) * f3^(7.0 / 3.0) * v2 +
                f2^(7.0 / 3.0) * (-(f3^(8.0 / 3.0) * v1) + f1^(8.0 / 3.0) * v3) + f2^(8.0 / 3.0) * (f3^(7.0 / 3.0) * v1 - f1^(7.0 / 3.0) * v3))) /
               (f1^(7.0 / 3.0) * (f1^(1.0 / 3.0) - f2^(1.0 / 3.0)) * f2^(7.0 / 3.0) * (f1^(1.0 / 3.0) - f3^(1.0 / 3.0)) *
                (f2^(1.0 / 3.0) - f3^(1.0 / 3.0)) * f3^(7.0 / 3.0))
    elseif version == 2 || version == 1 || version == 0
        return zero(T)
    else
        throw(ArgumentError("_xhm_inspiral_amp_rho3: version must be 0,1,2,3 (got $version)"))
    end
end

# ================================================================================================
# AMPLITUDE: default-release (122022) pseudo-PN coefficient solve, ported from
# `IMRPhenomXHM_Inspiral_Amp_Coefficients`, `LALSimIMRPhenomXHM_inspiral.c` lines 893-943. Only
# `case 123` (use all 3 collocation points) is ported -- the ONLY case actually reachable under the
# default release, since the veto mechanism that would select a different case never fires (see
# module docstring, dead-code finding 2).
# ================================================================================================

"""
    _xhm_inspiral_amp_coefficients_123(v1, v2, v3, finsp, f1, f2, f3)

`(IC0, IC1, IC2)` -- the 3 pseudo-PN coefficients (`pAmp->InspiralCoefficient[0..2]`) for the
`case 123` branch (all 3 collocation points used, the only case reachable under the default
122022 release). `finsp` is the inspiral/intermediate matching frequency (`pAmp->fAmpMatchIN`,
numerically identical to `f3` here since `f3 = 1.0*fAmpMatchIN`, kept as a separate argument to
mirror the C source's own `powers_of_finsp` faithfully). `v1,v2,v3` are
`(CollocationPointsValuesAmplitudeInsp[i] - PNAmplitudeInsp[i]) / PNdominant * f_i^{7/6}`, computed
by the caller (`_xhm_insp_amp_coefficients`).
"""
@inline function _xhm_inspiral_amp_coefficients_123(v1, v2, v3, finsp, f1, f2, f3)
    T = promote_type(typeof(v1), typeof(finsp))
    @Tlit T begin
        denom = f1^(7.0 / 3.0) * (f1^(1.0 / 3.0) - f2^(1.0 / 3.0)) * f2^(7.0 / 3.0) *
                (f1^(1.0 / 3.0) - f3^(1.0 / 3.0)) * (f2^(1.0 / 3.0) - f3^(1.0 / 3.0)) * f3^(7.0 / 3.0)
        IC0 = finsp^(7.0 / 3.0) * (
            -(f1^3.0 * f3^(8.0 / 3.0) * v2) + f1^(8.0 / 3.0) * f3^3.0 * v2 +
            f2^3.0 * (f3^(8.0 / 3.0) * v1 - f1^(8.0 / 3.0) * v3) + f2^(8.0 / 3.0) * (-(f3^3.0 * v1) + f1^3.0 * v3)
        ) / denom
        IC1 = finsp^(8.0 / 3.0) * (
            f1^3.0 * f3^(7.0 / 3.0) * v2 - f1^(7.0 / 3.0) * f3^3.0 * v2 +
            f2^3.0 * (-(f3^(7.0 / 3.0) * v1) + f1^(7.0 / 3.0) * v3) + f2^(7.0 / 3.0) * (f3^3.0 * v1 - f1^3.0 * v3)
        ) / denom
        IC2 = finsp^3.0 * (
            f1^(7.0 / 3.0) * (-f1^(1.0 / 3.0) + f3^(1.0 / 3.0)) * f3^(7.0 / 3.0) * v2 +
            f2^(7.0 / 3.0) * (-(f3^(8.0 / 3.0) * v1) + f1^(8.0 / 3.0) * v3) + f2^(8.0 / 3.0) * (f3^(7.0 / 3.0) * v1 - f1^(7.0 / 3.0) * v3)
        ) / denom
        return IC0, IC1, IC2
    end
end

"""
    _xhm_wavy_points(p1, p2, p3)

Direct port of `WavyPoints`, `LALSimIMRPhenomXHM_inspiral.c` lines 833-839: `true` iff `p2` is a
strict local extremum of `(p1,p2,p3)` (collocation points misbehaving outside the calibration
region). NOT wired into `_xhm_insp_amp_coefficients` below -- see module docstring, dead-code
finding 2 (unreachable under the default 122022 release); ported standalone for completeness / a
future task's use.
"""
@inline _xhm_wavy_points(p1, p2, p3) = (p1 > p2 && p2 < p3) || (p1 < p2 && p2 > p3)

"""
    _xhm_inspiral_amplitude_veto(iv1, iv2, iv3, f1, f2, f3, ampNorm)

Direct (pure, non-mutating) port of `IMRPhenomXHM_Inspiral_Amplitude_Veto`,
`LALSimIMRPhenomXHM_inspiral.c` lines 802-830: zeroes out any collocation-point value below
`0.2/ampNorm * f_i^{7/6}` and reports which points survived, as `(iv1', iv2', iv3',
n_surviving)`. The C version also mutates `pWFHM->IMRPhenomXHMInspiralAmpVersion` in place; that
bookkeeping is left to the caller here. NOT wired into `_xhm_insp_amp_coefficients` below -- see
module docstring, dead-code finding 2 (unreachable under the default 122022 release); ported
standalone for completeness / a future task's use.
"""
@inline function _xhm_inspiral_amplitude_veto(iv1, iv2, iv3, f1, f2, f3, ampNorm)
    T = promote_type(typeof(iv1), typeof(f1), typeof(ampNorm))
    threshold = T(0.2) / ampNorm
    ok1 = iv1 >= threshold * f1^(7.0 / 6.0)
    ok2 = iv2 >= threshold * f2^(7.0 / 6.0)
    ok3 = iv3 >= threshold * f3^(7.0 / 6.0)
    return (ok1 ? iv1 : zero(T)), (ok2 ? iv2 : zero(T)), (ok3 ? iv3 : zero(T)), count((ok1, ok2, ok3))
end

# ================================================================================================
# AMPLITUDE: top-level coefficient bundle + ansatz evaluation.
# ================================================================================================

"""
    _xhm_insp_amp_coefficients(l, m, eta, chi1, chi2)

Everything [`_xhm_insp_amp_ansatz`](@ref) needs to evaluate mode `(l,m)`'s inspiral amplitude
ansatz at any `Mf`, computed once (matching `IMRPhenomXHM_Get_Inspiral_Amp_Coefficients`,
`LALSimIMRPhenomXHM_inspiral.c` lines 841-870, called from `IMRPhenomXHM_GetAmplitudeCoefficients`,
`LALSimIMRPhenomXHM_internals.c` lines 1220-1229, the default (122022) release path). `chi1`,
`chi2` follow this package's `m1 >= m2`, "chi1 is the spin on m1" convention throughout.
"""
@inline function _xhm_insp_amp_coefficients(l::Integer, m::Integer, eta, chi1, chi2)
    T = promote_type(typeof(eta), typeof(chi1), typeof(chi2))
    @Tlit T begin
        Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
        mm1 = 0.5 * (1.0 + Seta)
        mm2 = 0.5 * (1.0 - Seta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))  # chiPNHat
        chiS = 0.5 * (chi1 + chi2)
        chiA = 0.5 * (chi1 - chi2)
        dchi = chi1 - chi2
        chidiff = 0.5 * dchi

        emm = m == 1 ? 1.0 : m == 2 ? 2.0 : m == 3 ? 3.0 : 4.0  # see module docstring's (l,m) table
    end

    fMECO22 = _xas_fMECO(eta, S, dchi, Seta)
    fcutInsp = fMECO22 * emm * T(0.5)

    f1, f2, f3 = fcutInsp * T(0.5), fcutInsp * T(0.75), fcutInsp

    iv1, iv2, iv3 = _xhm_insp_amp_collocation_values(l, m, eta, Seta, S, chidiff)

    pn = _xhm_pn_amplitude_coefficients(l, m, eta, Seta, chiS, chiA)

    ampNorm = sqrt(T(2.0) * eta / T(3.0)) * T(pi)^(T(-1.0) / T(6.0))
    prefactor = if l == 2 && m == 1
        sqrt(T(2.0)) / T(3.0)
    elseif l == 3 && m == 3
        T(0.75) * sqrt(T(5.0) / T(7.0))
    elseif l == 3 && m == 2
        sqrt(T(5.0) / T(7.0)) / T(3.0)
    else # l == 4 && m == 4
        T(4.0) * sqrt(T(2.0)) / T(9.0) * sqrt(T(5.0) / T(7.0))
    end
    PNglobalfactor = (T(2.0) / emm)^(T(-7.0) / T(6.0)) * prefactor
    PNdominant = ampNorm * (T(2.0) / emm)^(T(-7.0) / T(6.0))

    PN1 = _xhm_pnamp_ansatz(f1, pn, PNglobalfactor, ampNorm)
    PN2 = _xhm_pnamp_ansatz(f2, pn, PNglobalfactor, ampNorm)
    PN3 = _xhm_pnamp_ansatz(f3, pn, PNglobalfactor, ampNorm)

    v1 = (iv1 - PN1) / PNdominant * f1^(T(7.0) / T(6.0))
    v2 = (iv2 - PN2) / PNdominant * f2^(T(7.0) / T(6.0))
    v3 = (iv3 - PN3) / PNdominant * f3^(T(7.0) / T(6.0))

    IC0, IC1, IC2 = _xhm_inspiral_amp_coefficients_123(v1, v2, v3, fcutInsp, f1, f2, f3)

    return (l = l, m = m, fcutInsp = fcutInsp, pn = pn, PNglobalfactor = PNglobalfactor,
        PNdominant = PNdominant, ampNorm = ampNorm, IC = (IC0, IC1, IC2))
end

"""
    _xhm_insp_amp_ansatz(Mf, coeffs)

The full inspiral amplitude ansatz (`IMRPhenomXHM_Inspiral_Amp_Ansatz`'s "new release"/default
branch, `LALSimIMRPhenomXHM_inspiral.c` lines 759-770: PN series + pseudo-PN correction, no
`RescaleFactor` division since `pAmp->InspRescaleFactor == 0` by default), evaluated at
dimensionless frequency `Mf` using the coefficient bundle from [`_xhm_insp_amp_coefficients`](@ref).
"""
@inline function _xhm_insp_amp_ansatz(Mf, coeffs)
    IC0, IC1, IC2 = coeffs.IC
    fc = coeffs.fcutInsp
    T = promote_type(typeof(Mf), typeof(fc))
    @Tlit T begin
        pnAmp = _xhm_pnamp_ansatz(Mf, coeffs.pn, coeffs.PNglobalfactor, coeffs.ampNorm)
        pseudo = (Mf^(7.0 / 3.0) / fc^(7.0 / 3.0) * IC0 + Mf^(8.0 / 3.0) / fc^(8.0 / 3.0) * IC1 + Mf^3.0 / fc^3.0 * IC2) *
                 Mf^(-7.0 / 6.0) * coeffs.PNdominant
        return pnAmp + pseudo
    end
end

"""
    _xhm_insp_amp_nd_ansatz(Mf, coeffs)

4th-order-accurate central finite-difference derivative `d(InspAmp)/dMf`, matching
`IMRPhenomXHM_Inspiral_Amp_NDAnsatz`, `LALSimIMRPhenomXHM_inspiral.c` lines 777-798 (used to get a derivative estimate at the inspiral/
intermediate boundary when an analytic one isn't used instead). Uses the exact
same 5-point stencil as the C source, but with a SCALED step: the C source's `df = 10e-10` (absolute,
at Mf ~ 0.01-0.1) amplifies Float64 roundoff to ~1e-7 relative, which made the XHM likelihood jump by ~1e-6..1e-5
under 1e-15 changes of the masses (no smooth gradient). `df = 1e-4 Mf`: roundoff ~1e-12, truncation ~1e-16
(the ansatz is a smooth power series in Mf). Differs from the C value by about the C value's own noise.
"""
@inline function _xhm_insp_amp_nd_ansatz(Mf, coeffs)
    T = promote_type(typeof(Mf), typeof(coeffs.fcutInsp))
    df = T(1e-4) * Mf
    fun2R = _xhm_insp_amp_ansatz(Mf + 2 * df, coeffs)
    funR = _xhm_insp_amp_ansatz(Mf + df, coeffs)
    funL = _xhm_insp_amp_ansatz(Mf - df, coeffs)
    fun2L = _xhm_insp_amp_ansatz(Mf - 2 * df, coeffs)
    return (-fun2R + 8 * funR - 8 * funL + fun2L) / (12 * df)
end

# ================================================================================================
# PHASE: parameter-space fits for `lambda_lm` (the weighted phase difference between mode `(l,m)`
# and the `(2,2)` mode, Eq. 4.10-4.12 of arXiv:2001.10914). Ported from
# `IMRPhenomXHM_Insp_Phase_{21,33,32,44}_lambda`, `LALSimIMRPhenomXHM_inspiral.c` lines 956-1046
# (the only version present in the source, `case 122019`). NOTE: `S = pWF->STotR` here (NOT
# `chiPNHat`, unlike the amplitude `iv` fits above), and `dchi = chi1L - chi2L` (the FULL
# difference, NOT `dchi_half`) -- confirmed by re-reading the C source directly rather than
# assuming the amplitude fits' convention carries over. Only exercised (see module docstring,
# dead-code finding 3) for `eta <= 0.01`, outside the validated mass-ratio range.
# ================================================================================================

@inline function _xhm_insp_phase_21_lambda(eta, STotR, dchi)
    T = promote_type(typeof(eta), typeof(STotR), typeof(dchi))
    @Tlit T begin
        S = STotR
        eta2 = eta * eta
        eta3 = eta2 * eta
        S2 = S * S
        noSpin = 13.664473636545068 - 170.08866400251395 * eta + 3535.657736681598 * eta2 -
                 26847.690494515424 * eta3 + 96463.68163125668 * eta2 * eta2 - 133820.89317471132 * eta2 * eta3
        eqSpin = (S * (18.52571430563905 - 41.55066592130464 * S +
                       eta3 * (83493.24265292779 + 16501.749243703132 * S - 149700.4915210766 * S2) +
                       eta * (3642.5891077598003 + 1198.4163078715173 * S - 6961.484805326852 * S2) +
                       33.8697137964237 * S2 +
                       eta2 * (-35031.361998480075 - 7233.191207000735 * S + 62149.00902591944 * S2))) /
                 (6.880288191574696 + 1.0 * S)
        uneqSpin = -134.27742343186577 * dchi * sqrt(1.0 - 4.0 * eta) * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_insp_phase_33_lambda(eta, STotR, dchi)
    T = promote_type(typeof(eta), typeof(STotR), typeof(dchi))
    @Tlit T begin
        S = STotR
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta
        eta3 = eta2 * eta
        noSpin = 4.1138398568400705 + 9.772510519809892 * eta - 103.92956504520747 * eta2 + 242.3428625556764 * eta3
        eqSpin = ((-0.13253553909611435 + 26.644159828590055 * eta - 105.09339163109497 * eta2) * S) /
                 (1.0 + 0.11322426762297967 * S)
        uneqSpin = -19.705359163581168 * dchi * eta2 * delta
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_insp_phase_32_lambda(eta, STotR, dchi, chi1L, chi2L)
    T = promote_type(typeof(eta), typeof(STotR), typeof(dchi), typeof(chi1L), typeof(chi2L))
    @Tlit T begin
        S = STotR
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        S2 = S * S
        S3 = S2 * S
        S4 = S3 * S
        noSpin = (9.913819875501506 + 18.424900617803107 * eta - 574.8672384388947 * eta2 +
                  2671.7813055097877 * eta3 - 6244.001932443913 * eta4) / (1.0 - 0.9103118343073325 * eta)
        eqSpin = (-4.367632806613781 + 245.06757304950986 * eta - 2233.9319708029775 * eta2 + 5894.355429022858 * eta3) * S +
                 (-1.375112297530783 - 1876.760129419146 * eta + 17608.172965575013 * eta2 - 40928.07304790013 * eta3) * S2 +
                 (-1.28324755577382 - 138.36970336658558 * eta + 708.1455154504333 * eta2 - 273.23750933544176 * eta3) * S3 +
                 (1.8403161863444328 + 2009.7361967331492 * eta - 18636.271414571278 * eta2 + 42379.205045791656 * eta3) * S4
        uneqSpin = dchi * sqrt(1.0 - 4.0 * eta) * eta2 *
                   (-105.34550407768225 - 1566.1242344157668 * chi1L * eta + 1566.1242344157668 * chi2L * eta + 2155.472229664981 * eta * S)
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_insp_phase_44_lambda(eta, STotR, dchi)
    T = promote_type(typeof(eta), typeof(STotR), typeof(dchi))
    @Tlit T begin
        S = STotR
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        S2 = S * S
        noSpin = 5.254484747463392 - 21.277760168559862 * eta + 160.43721442910618 * eta2 -
                 1162.954360723399 * eta3 + 1685.5912722190276 * eta4 - 1538.6661348106031 * eta4 * eta
        eqSpin = (0.007067861615983771 - 10.945895160727437 * eta + 246.8787141453734 * eta2 - 810.7773268493444 * eta3) * S +
                 (0.17447830920234977 + 4.530539154777984 * eta - 176.4987316167203 * eta2 + 621.6920322846844 * eta3) * S2
        uneqSpin = -8.384066369867833 * dchi * sqrt(1.0 - 4.0 * eta) * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

"""
    _xhm_insp_phase_lambdapn(eta, l, m)

`IMRPhenomXHM_Insp_Phase_LambdaPN`'s closed-form linear-in-`f` phase correction from the complex
part of the PN amplitude (Eq. 4.9 of arXiv:2001.10914, truncated at linear order), ported from
`LALSimIMRPhenomXHM_inspiral.c` lines 1051-1081. This IS exercised under the default release for
every mass ratio in this port's validated range (`eta > 0.01`, see module docstring) -- unlike the
`_lambda` parameter-space fits above.
"""
@inline function _xhm_insp_phase_lambdapn(eta, l::Integer, m::Integer)
    T = typeof(eta)
    @Tlit T begin
        PI = T(pi)
        output = if l == 2 && m == 1
            2.0 * PI * (-0.5 - 2.0 * log(2.0))
        elseif l == 3 && m == 3
            2.0 / 3.0 * PI * (-21.0 / 5.0 + 6.0 * log(1.5))
        elseif l == 3 && m == 2
            -((2376.0 * PI * (-5.0 + 22.0 * eta)) / (-3960.0 + 11880.0 * eta))
        elseif l == 4 && m == 4
            45045.0 * PI * (336.0 - 1193.0 * eta + 320.0 * (-1.0 + 3.0 * eta) * log(2.0)) / (2.0 * (-1801800.0 + 5405400.0 * eta))
        else
            zero(T)
        end
        return -1.0 * output
    end
end

# ================================================================================================
# PHASE: rescale IMRPhenomXAS's own (already-validated) `(2,2)`-mode inspiral phase coefficients
# into mode `(l,m)`'s `phi[]`/`phiL[]` (Appendix D of arXiv:2001.10914). Ported from
# `IMRPhenomXHM_GetPhaseCoefficients`, `LALSimIMRPhenomXHM_internals.c` lines 2010-2054 -- see the
# module docstring for why `phi_minus2`/`phi_minus1`/`phi9`/`phi9L`/`sigma5` are correctly treated
# as identically zero here, and why reusing `IMRPhenomXAS.jl`'s `_xas_phi_coeffs` directly (rather
# than re-deriving the same ~15-coefficient block) is the right call.
# ================================================================================================

"""
    _xhm_insp_phase_coefficients(l, m, eta, chi1, chi2)

`(phi, phiL)` -- two 15-element tuples (index `i=1..15` <-> LALSuite's C array index `i-1=0..14`,
frequency power `Mf^{(i-8)/3}`), the rescaled+`LambdaPN`-corrected inspiral phase ansatz
coefficients for mode `(l,m)`, ready for [`_xhm_insp_phase_ansatz`](@ref)/
[`_xhm_insp_phase_ansatz_int`](@ref).
"""
@inline function _xhm_insp_phase_coefficients(l::Integer, m::Integer, eta, chi1, chi2)
    T = promote_type(typeof(eta), typeof(chi1), typeof(chi2))
    pc = _xas_phi_coeffs(IMRPhenomXAS(), one(T), eta, chi1, chi2)
    pi_ = pc.phi_ins

    @Tlit T begin
        emm = m == 1 ? 1.0 : m == 2 ? 2.0 : m == 3 ? 3.0 : 4.0
        m_over_2 = emm * 0.5
        fact = pi_.phiN / eta
        zeroT = zero(T)

        # phenXnonLog[0..14], phenXLog[0..14] (0-based, C convention), NOT yet *fact:
        # idx0=phi_minus2=0, idx1=phi_minus1=0, idx2=phi0, idx3=phi1, idx4=phi2, idx5=phi3,
        # idx6=phi4, idx7=phi5, idx8=phi6, idx9=phi7, idx10=phi8, idx11=phi9=0, idx12..14=0
        nonLog = (zeroT, zeroT, pi_.phi0, pi_.phi1, pi_.phi2, pi_.phi3, pi_.phi4, pi_.phi5,
            pi_.phi6, pi_.phi7, pi_.phi8, zeroT, zeroT, zeroT, zeroT)
        Log = (zeroT, zeroT, zeroT, zeroT, zeroT, zeroT, zeroT, pi_.phi5L,
            pi_.phi6L, zeroT, pi_.phi8L, zeroT, zeroT, zeroT, zeroT)
        # pseudoPN[0..14]: idx10=sigma1, idx11=sigma2, idx12=sigma3, idx13=sigma4, idx14=sigma5=0
        pseudo = (zeroT, zeroT, zeroT, zeroT, zeroT, zeroT, zeroT, zeroT, zeroT, zeroT,
            pi_.sigma1, pi_.sigma2, pi_.sigma3, pi_.sigma4, zeroT)

        logm2 = log(m_over_2)
        # `Val(15)` (not a plain `Int` literal) -- the non-`Val` form of `ntuple` doesn't
        # guarantee compile-time unrolling and falls back to a heap-allocating generic path on
        # GPU backends (confirmed by a real `InvalidIRError` porting this to `CUDABackend()`,
        # complete with `jl_alloc_genericmemory_unchecked`/`collect` in the stacktrace). See also `_xas_phase_CV`/
        # `_xas_amp_CV` in `IMRPhenomXAS.jl` for the same fix applied earlier.
        phi = ntuple(Val(15)) do i1
            i = i1 - 1  # 0-based C index
            nl = nonLog[i1] * fact
            lg = Log[i1] * fact
            ps = pseudo[i1] * fact
            (nl + ps - lg * logm2) * m_over_2^((10.0 - i) / 3.0)
        end
        phiL = ntuple(Val(15)) do i1
            i = i1 - 1
            (Log[i1] * fact) * m_over_2^((10.0 - i) / 3.0)
        end
    end

    lambdaPN = _xhm_insp_phase_lambdapn(eta, l, m)
    phi = Base.setindex(phi, phi[11] + lambdaPN, 11)  # idx10 (0-based) -> Julia index 11

    return phi, phiL
end

"""
    _xhm_insp_phase_ansatz_int(Mf, phi, phiL)

The inspiral orbital phase itself, `IMRPhenomXHM_Inspiral_Phase_AnsatzInt`,
`LALSimIMRPhenomXHM_inspiral.c` lines 1091-1100: `sum_i (phi[i] + phiL[i]*log(Mf)) * Mf^{(i-7)/3}`
for `i=0..14` (0-based C indexing; `phi`/`phiL` here are 1-based Julia tuples from
[`_xhm_insp_phase_coefficients`](@ref)).
"""
@inline function _xhm_insp_phase_ansatz_int(Mf, phi, phiL)
    T = promote_type(typeof(Mf), typeof(phi[1]))
    @Tlit T begin
        logMf = log(Mf)
        s = zero(T)
        for i1 in 1:15
            i = i1 - 1
            s += (phi[i1] + phiL[i1] * logMf) * Mf^((i - 7.0) / 3.0)
        end
        return s
    end
end

"""
    _xhm_insp_phase_ansatz(Mf, phi, phiL)

`d(_xhm_insp_phase_ansatz_int)/dMf`, `IMRPhenomXHM_Inspiral_Phase_Ansatz`,
`LALSimIMRPhenomXHM_inspiral.c` lines 1102-1112: the analytic frequency-derivative used to build
the ansatz (matched to TaylorF2's `dphi/df` at collocation points), NOT itself the phase.
"""
@inline function _xhm_insp_phase_ansatz(Mf, phi, phiL)
    T = promote_type(typeof(Mf), typeof(phi[1]))
    @Tlit T begin
        logMf = log(Mf)
        s = zero(T)
        for i1 in 1:15
            i = i1 - 1
            n = (i - 7.0) / 3.0
            s += ((phi[i1] + phiL[i1] * logMf) * n + phiL[i1]) * Mf^(n - 1.0)
        end
        return s
    end
end
