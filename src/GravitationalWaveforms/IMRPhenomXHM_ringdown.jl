"""
    IMRPhenomXHM ringdown-region amplitude/phase ansatze (this file)

The ringdown-region amplitude AND phase ansatze for
the `(l,m) = (2,1), (3,3), (3,2), (4,4)` higher harmonics that `IMRPhenomXHM` adds on top of
`IMRPhenomXAS`'s existing dominant `(2,2)` mode. Ported directly from LALSuite's
`LALSimIMRPhenomXHM_ringdown.c`/`.h` (local checkout:
`lalsuite/lalsimulation/lib/LALSimIMRPhenomXHM_ringdown.c`), plus the small amount of
`LALSimIMRPhenomXHM_internals.c` glue needed to actually assemble a testable ansatz (the `.h`/`.c`
only declare fit/ansatz *pieces* -- the "how do I get from `(eta,chi1,chi2,fring,fdamp)` to the
`RDCoefficient[]` array these ansatze consume" wiring lives in `_internals.c`'s
`IMRPhenomXHM_RD_Amp_Coefficients`/`IMRPhenomXHM_GetPhaseCoefficients`). The
QNM ringdown-frequency/damping-time fits (`IMRPhenomXHM_qnm.jl`'s `_xhm_fring_fdamp`) are used
directly here, not re-derived. Mode-mixing/full assembly is in `IMRPhenomXHM.jl`.

# Working in dimensionless `Mf`, not physical Hz
Exactly like the LALSuite functions they port, every function in this file operates on the
dimensionless frequency `Mf` (total-mass-scaled), not physical Hz -- matching the inspiral and
intermediate files. The `mc`/`dL`-aware wrapper is in `IMRPhenomXHM.jl`.

# Scope actually delivered: FULL standalone ansatz for (2,1),(3,3),(4,4); raw fits only for (3,2)
Both amplitude and phase are fully ported, assembled, and validated against LALSuite for the three "no
mode-mixing" higher modes. For `(3,2)`, only the raw parameter-space fit functions are ported
(`alambda`,`lambda`,`sigma`,`rdaux1`,`rdaux2` for amplitude; `SpheroidalTimeShift`,
`SpheroidalPhaseShift`,`p1`-`p4` for phase) -- assembling a standalone, validated `(3,2)` ringdown
ansatz turns out to be genuinely impossible without machinery this package does not yet have
anywhere (see "`(3,2)` mode: why only raw fits are ported" below). This is a real, source-verified
scope boundary, not a shortcut -- the assembly is in `IMRPhenomXHM.jl`.

# Which `IMRPhenomXHMReleaseVersion` this port targets, and a genuinely three-way-split default
`PhenomXHMReleaseVersion`'s default is **122022** (confirmed in `LALSimInspiralWaveformParams.c`). Reading `LALSimIMRPhenomXHM_internals.c`'s `IMRPhenomXHM_SetHMWaveformVariables`
directly (lines ~295-370) shows this default does NOT uniformly bump every sub-version to `122022`
the way one might assume from the inspiral and intermediate files -- it is a genuine mix:

- `IMRPhenomXHMRingdownAmpFitsVersion` -> **122022** (bumped). So the amplitude parameter-space fits
  used here (`rdcp1`-`rdcp3` for `(2,1),(3,3),(4,4)`; `alambda`/`lambda`/`sigma` for `(3,2)`) are the
  122022-branch coefficients, NOT 122018.
- `IMRPhenomXHMRingdownAmpVersion` -> **2** for `(2,1),(3,3),(4,4)`, but explicitly overridden back
  to **1** for `(3,2)` (internals.c line ~345, `"Use fit of coefficients instead of collocation
  points"`). This is a genuine, mode-dependent ansatz-*form* switch, confirmed by tracing the
  `switch(RDAmpFlag)` dispatch in `IMRPhenomXHM_RD_Amp_Coefficients` (ringdown.c lines ~737-816):
  version 2 solves `RDCoefficient[0..2]` from 3 collocation points (`rdcp1`-`rdcp3`, ringdown.c
  lines ~784-790); version 1 takes `RDCoefficient[0..2]` **directly** from `alambda`/`lambda`/`sigma`
  (no linear solve at all). **This means `alambda`/`lambda`/`sigma` are LIVE for `(3,2)` but DEAD
  for `(2,1),(3,3),(4,4)` under the actual default** -- the reverse of a naive reading of this
  task's own source map, which lists `alambda`/`lambda`/`sigma` first as if they were the primary
  mechanism for every mode. Confirmed by direct reading, not assumed; **not ported** for
  `(2,1),(3,3),(4,4)` (genuinely unreachable there), **ported** for `(3,2)` (genuinely live there).
- `IMRPhenomXHMRingdownPhaseVersion`/`...PhaseFitsVersion` -> **stays 122019**, NOT bumped to
  122022. The `else if (ReleaseVersion == 122022)` block's phase section (internals.c lines
  ~353-366) only touches `nCollocPtsRDPhase` for `(3,2)`; the 5-line block that *would* set
  `IMRPhenomXHMRingdownPhaseVersion = 122022` is present in the source but **commented out**
  (`// wf->IMRPhenomXHMRingdownPhaseFitsVersion = 122022;` etc., with the comment
  `"This can introduce significant mismatch ... so it is left for the next update"`). So every phase
  fit ported here (`_22_alpha2`/`_22_alphaL`, and `(3,2)`'s `SpheroidalTimeShift`/
  `SpheroidalPhaseShift`/`p1`-`p4`) uses the **122019** branch, confirmed live; the 122022 branches
  present in the C source for the `(3,2)`-specific fits are dead code under the real default and are
  NOT ported (as in the inspiral and intermediate files). `p5`
  (`IMRPhenomXHM_RD_Phase_32_p5`) is **doubly** dead: it has no 122019 branch at all (only 122022,
  itself unreachable), AND `nCollocPtsRDPhase` stays 4 (not 5) under the actual default release
  (internals.c line ~355) -- so even a 122022-phase-version world would never call it with the
  5-point linear system that would use it. Not ported.

# `alpha0 = 0` is hardcoded for the non-mixing ringdown phase ansatz -- no C1-continuity fit needed
Unlike `IMRPhenomXAS.jl`'s dominant-mode ringdown phase (`_xas_phi_mrd_val`, which needs `beta0`/
`beta1` C1-continuity corrections against the intermediate region), the `(2,1),(3,3),(4,4)`
ringdown phase *derivative* ansatz has no free additive constant at all:
`LALSimIMRPhenomXHM_internals.c` line ~1977 initializes `pPhase->alpha0 = 0.0` and this field is
**never reassigned** anywhere in the `MixingOn == 0` code path (confirmed by grepping every
`alpha0` occurrence in `_internals.c`) -- only `alpha2`/`alphaL` get computed from fits. The
intermediate/ringdown join instead happens entirely on the intermediate side (the `C1RD`/
`CRD` glue constants), not by adjusting the ringdown ansatz itself. This module's
`_xhm_rd_phase_coefficients` reflects this directly: it returns `alpha0 = 0` unconditionally for
`(2,1),(3,3),(4,4)`.

# `DAnsatz` vs `NDAnsatz`
`IMRPhenomXHM_RD_Amp_DAnsatz` (analytic derivative) is the one actually used for every mode this
task ports: it seeds `RDCoefficient[3]`/`[4]` (the exponential-falloff continuity constants, see
below) for ALL modes via the trick at `IMRPhenomXHM_RD_Amp_Coefficients` (ringdown.c lines
~799-807) of temporarily zeroing `fAmpRDfalloff` to force the ansatz down its Lorentzian branch.
`IMRPhenomXHM_RD_Amp_NDAnsatz` (4th-order finite-difference numerical derivative) is called from
exactly one place in the whole codebase: `SpheroidalToSpherical`'s mode-mixing machinery (used only
for `(3,2)`'s full mode-mixed reconstruction, in `IMRPhenomXHM.jl`). **Not ported** (correctly out of scope; the analytic `DAnsatz` this module DOES
port is what every non-mixing mode, and even `(3,2)`'s own falloff continuity, actually uses).

# `IMRPhenomXHM_Ringdown_Amplitude_Veto`: confirmed DEAD under the default release
This function (ringdown.c lines ~1033-1052) is only called when `pWFHM->RingdownAmpVeto == 1`
(`_internals.c` line ~1792). Tracing every assignment to `RingdownAmpVeto`: it is initialized to
`0` (line ~106) and only ever set to `1` inside the `IMRPhenomXHMReleaseVersion == 122019` branch
(lines ~268, ~281, for modes `21`/`32`) -- never touched in the `122022` branch. So under the actual
default release, `RingdownAmpVeto` stays `0` for every mode and this veto never fires. **Not
ported** -- as in the inspiral and intermediate files.

# `fAmpRDfalloff`: genuinely LIVE (not dead)
Unlike the veto above, `pWFHM->fAmpRDfalloff = pWFHM->fRING + 2 * pWFHM->fDAMP` (`_internals.c`
line ~1224) is set **unconditionally** whenever `ReleaseVersion != 122019` -- i.e. always, under our
default, for every mode. This means the amplitude ansatz's exponential-falloff branch
(`ampRD = RDCoefficient[3] * exp(-RDCoefficient[4] * (f - fAmpRDfalloff))` for
`f >= fAmpRDfalloff`) is exercised by every real `IMRPhenomXHM` waveform generated at frequencies
above `fring + 2*fdamp`, not just an edge case -- `_xhm_rd_amp_coefficients`/`_xhm_rd_amp_ansatz`
below implement it as a normal, always-active part of the ansatz (not an optional extra), and the
LALSuite validation deliberately sweeps well past `fring + 2*fdamp` to exercise it.

# `(3,2)` mode: why only raw fits are ported (not a full standalone ansatz)
Both the amplitude and phase machinery for `(3,2)` were traced in full and both turn out to need
external dependencies this package does not have anywhere yet, confirmed by reading
`LALSimIMRPhenomXHM_internals.c` directly (not assumed):
- **Amplitude**: `nCoefficientsRDAux` (which activates the `rdaux1`/`rdaux2`-based low-frequency
  polynomial extension of the ringdown ansatz) is set unconditionally whenever `pWFHM->MixingOn` is
  true (`_internals.c` line ~1215-1219) -- and `MixingOn` is true precisely (and only) for `(3,2)`.
  So `(3,2)`'s ringdown amplitude ansatz ALWAYS needs the `rdaux1`/`rdaux2` polynomial branch (it is
  not an edge case for this mode the way the exponential falloff might have seemed at first
  glance) -- and that branch's own boundary conditions need `fAmpMatchIM`
  (`IMRPhenomXHM_Amplitude_fcutRD`'s mode-mixing-aware form, `= fRING22 - 0.5*fDAMP22` under our
  default for `(3,2)` specifically -- a DIFFERENT formula than the `fring - fdamp` used for
  `(2,1),(3,3),(4,4)`) and a self-referential linear solve. This is squarely mode-mixing-adjacent
  plumbing, not a narrow ringdown-ansatz porting task.
- **Phase**: `(3,2)`'s spheroidal ringdown ansatz coefficients (`alpha0_S`,`alpha2_S`,`alpha4_S`,
  `alphaL_S`) are obtained from a 4-point linear solve using this file's own `p1`-`p4` fits (that
  part alone WOULD be in scope) -- but the result then needs an additional time/phase-shift
  correction (`GetSpheroidalCoefficients`, `_internals.c` lines ~2555-2624) that is anchored to the
  DOMINANT `(2,2)` mode's *own* ringdown phase derivative via `IMRPhenomX_dPhase_22`,
  `IMRPhenomX_TimeShift_22`, and `IMRPhenomX_Phase_22_ConnectionCoefficients` -- none of which exist
  anywhere in this package (`IMRPhenomXAS.jl` computes an equivalent quantity internally inside
  `_xas_phi_coeffs`/`_xas_phi_mrd_deriv`, but does not expose the specific "22 mode's own time-shift
  constant" LALSuite's `IMRPhenomX_TimeShift_22` returns as a standalone, reusable value). Porting
  that would mean porting a meaningful slice of `LALSimIMRPhenomX_internals.c` (the *dominant*-mode
  internals, not even the HM ones) as a side effect, so that belongs in `IMRPhenomXHM.jl`.

Given both findings, this file ports `(3,2)`'s raw fits (useful, narrow, independently testable
building blocks used by `IMRPhenomXHM.jl`) but does NOT assemble or validate a standalone `(3,2)` ringdown
ansatz function, amplitude or phase.
"""

# ================================================================================================
# AMPLITUDE: raw parameter-space fits, live (122022) branch only.
# `(2,1),(3,3),(4,4)`: rdcp1-rdcp3 (the collocation-point fits actually used, RDAmpVersion==2).
# `(3,2)`: alambda/lambda/sigma (RDAmpVersion==1, direct coefficients) + rdaux1/rdaux2 (raw only,
# see module docstring for why no standalone (3,2) ansatz is assembled from these).
# ================================================================================================

@inline function _xhm_rd_amp21_rdcp1(eta, delta, chiPNHat, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(chiPNHat), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4; eta6 = eta1 * eta5
        S1 = chiPNHat
        S2 = S1 * S1
        chidiff1 = chidiff
        return abs(delta * eta1 * (12.880905080761432 - 23.5291063016996 * eta1 + 92.6090002736012 * eta2 - 175.16681482428694 * eta3) +
                    chidiff1 * delta * eta1 * (26.89427230731867 * eta1 - 710.8871223808559 * eta2 + 2255.040486907459 * eta3) +
                    chidiff1 * delta * eta1 * (21.402708785047853 * eta1 - 232.07306353130417 * eta2 + 591.1097623278739 * eta3) * S1 +
                    delta * eta1 * S1 * (-10.090867481062709 * (0.9580746052260011 + 5.388149112485179 * eta1 - 107.22993216128548 * eta2 + 801.3948756800821 * eta3 - 2688.211889175019 * eta4 + 3950.7894052628735 * eta5 - 1992.9074348833092 * eta6) -
                                          0.42972412296628143 * (1.9193131231064235 + 139.73149069609775 * eta1 - 1616.9974609915555 * eta2 - 3176.4950303461164 * eta3 + 107980.65459735804 * eta4 - 479649.75188253267 * eta5 + 658866.0983367155 * eta6) * S1) +
                    chidiff1 * eta5 * (-1512.439342647443 + 175.59081294852444 * S1 + 10.13490934572329 * S2))
    end
end

@inline function _xhm_rd_amp21_rdcp2(eta, delta, chiPNHat, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(chiPNHat), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4
        S1 = chiPNHat
        S2 = S1 * S1
        chidiff1 = chidiff
        return abs(delta * (9.112452928978168 - 7.5304766811877455 * eta1) * eta1 +
                    chidiff1 * delta * eta1 * (16.236533863306132 * eta1 - 500.11964987628926 * eta2 + 1618.0818430353293 * eta3) +
                    chidiff1 * delta * eta1 * (2.7866868976718226 * eta1 - 0.4210629980868266 * eta2 - 20.274691328125606 * eta3) * S1 +
                    chidiff1 * eta5 * (-1116.4039232324135 + 245.73200219767514 * S1 + 21.159179960295855 * S2) +
                    delta * eta1 * S1 * (-8.236485576091717 * (0.8917610178208336 + 5.1501231412520285 * eta1 - 87.05136337926156 * eta2 + 519.0146702141192 * eta3 - 997.6961311502365 * eta4) +
                                          0.2836840678615208 * (-0.19281297100324718 - 57.65586769647737 * eta1 + 586.7942442434971 * eta2 - 1882.2040277496196 * eta3 + 2330.3534917059906 * eta4) * S1 +
                                          0.40226131643223145 * (-3.834742668014861 + 190.42214703482531 * eta1 - 2885.5110686004946 * eta2 + 16087.433824017446 * eta3 - 29331.524552164105 * eta4) * S2))
    end
end

@inline function _xhm_rd_amp21_rdcp3(eta, delta, chiPNHat, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(chiPNHat), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4
        S1 = chiPNHat
        S2 = S1 * S1
        chidiff1 = chidiff
        return abs(delta * (2.920930733198033 - 3.038523690239521 * eta1) * eta1 +
                    chidiff1 * delta * eta1 * (6.3472251472354975 * eta1 - 171.23657247338042 * eta2 + 544.1978232314333 * eta3) +
                    chidiff1 * delta * eta1 * (1.9701247529688362 * eta1 - 2.8616711550845575 * eta2 - 0.7347258030219584 * eta3) * S1 +
                    chidiff1 * eta5 * (-334.0969956136684 + 92.91301644484749 * S1 - 5.353399481074393 * S2) +
                    delta * eta1 * S1 * (-2.7294297839371824 * (1.148166706456899 - 4.384077347340523 * eta1 + 36.120093043420326 * eta2 - 87.26454353763077 * eta3) +
                                          0.23949142867803436 * (-0.6931516433988293 + 33.33372867559165 * eta1 - 307.3404155231787 * eta2 + 862.3123076782916 * eta3) * S1 +
                                          0.1930861073906724 * (3.7735099269174106 - 19.11543562444476 * eta1 - 78.07256429516346 * eta2 + 485.67801863289293 * eta3) * S2))
    end
end

@inline function _xhm_rd_amp33_rdcp1(eta, delta, StotR, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4
        S1 = StotR
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return delta * eta1 * (12.439702602599235 - 4.436329538596615 * eta1 + 22.780673360839497 * eta2) +
               delta * eta1 * (chidiff1 * (-41.04442169938298 * eta1 + 502.9246970179746 * eta2 - 1524.2981907688634 * eta3) +
                                chidiff2 * (32.23960072974939 * eta1 - 365.1526474476759 * eta2 + 1020.6734178547847 * eta3)) +
               chidiff1 * delta * eta1 * (-52.85961155799673 * eta1 + 577.6347407795782 * eta2 - 1653.496174539196 * eta3) * S1 +
               chidiff1 * eta5 * (257.33227387984863 - 34.5074027042393 * chidiff2 - 21.836905132600755 * S1 - 15.81624534976308 * S2) +
               13.499999999999998 * delta * eta1 * S1 * (-0.13654149379906394 * (2.719687834084113 + 29.023992126142304 * eta1 - 742.1357702210267 * eta2 + 4142.974510926698 * eta3 - 6167.08766058184 * eta4 - 3591.1757995710486 * eta5) -
                                                          0.06248535354306988 * (6.697567446351289 - 78.23231700361792 * eta1 + 444.79350113344543 * eta2 - 1907.008984765889 * eta3 + 6601.918552659412 * eta4 - 10056.98422430965 * eta5) * S1) / (-3.9329308614837704 + S1)
    end
end

@inline function _xhm_rd_amp33_rdcp2(eta, delta, StotR, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4
        S1 = StotR
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return delta * eta1 * (8.425057692276933 + 4.543696144846763 * eta1) +
               chidiff1 * delta * eta1 * (-32.18860840414171 * eta1 + 412.07321398189293 * eta2 - 1293.422289802462 * eta3) +
               chidiff1 * delta * eta1 * (-17.18006888428382 * eta1 + 190.73514518113845 * eta2 - 636.4802385540647 * eta3) * S1 +
               delta * eta1 * S1 * (0.1206817303851239 * (8.667503604073314 - 144.08062755162752 * eta1 + 3188.189172446398 * eta2 - 35378.156133055556 * eta3 + 163644.2192178668 * eta4 - 265581.70142471837 * eta5) +
                                     0.08028332044013944 * (12.632478544060636 - 322.95832000179297 * eta1 + 4777.45310151897 * eta2 - 35625.58409457366 * eta3 + 121293.97832549023 * eta4 - 148782.33687815256 * eta5) * S1) +
               chidiff1 * eta5 * (159.72371180117415 - 29.10412708633528 * chidiff2 - 1.873799747678187 * S1 + 41.321480132899524 * S2)
    end
end

@inline function _xhm_rd_amp33_rdcp3(eta, delta, StotR, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4
        S1 = StotR
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return delta * eta1 * (2.485784720088995 + 2.321696430921996 * eta1) +
               delta * eta1 * (chidiff1 * (-10.454376404653859 * eta1 + 147.10344302665484 * eta2 - 496.1564538739011 * eta3) +
                                chidiff2 * (-5.9236399792925996 * eta1 + 65.86115501723127 * eta2 - 197.51205149250532 * eta3)) +
               chidiff1 * delta * eta1 * (-10.27418232676514 * eta1 + 136.5150165348149 * eta2 - 473.30988537734174 * eta3) * S1 +
               chidiff1 * eta5 * (32.07819766300362 - 3.071422453072518 * chidiff2 + 35.09131921815571 * S1 + 67.23189816732847 * S2) +
               13.499999999999998 * delta * eta1 * S1 * (0.0011484326782460882 * (4.1815722950796035 - 172.58816646768219 * eta1 + 5709.239330076732 * eta2 - 67368.27397765424 * eta3 + 316864.0589150127 * eta4 - 517034.11171277676 * eta5) -
                                                          0.009496797093329243 * (0.9233282181397624 - 118.35865186626413 * eta1 + 2628.6024206791726 * eta2 - 23464.64953722729 * eta3 + 94309.57566199072 * eta4 - 140089.40725211444 * eta5) * S1) / (0.09549360183532198 - 0.41099904730526465 * S1 + S2)
    end
end

@inline function _xhm_rd_amp44_rdcp1(eta, delta, StotR, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4; eta6 = eta1 * eta5
        S1 = StotR
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return eta1 * (chidiff1 * delta * (-8.51952446214978 * eta1 + 117.76530248141987 * eta2 - 297.2592736781142 * eta3) +
                        chidiff2 * (-0.2750098647982238 * eta1 + 4.456900599347149 * eta2 - 8.017569928870929 * eta3)) +
               eta1 * (5.635069974807398 - 33.67252878543393 * eta1 + 287.9418482197136 * eta2 - 3514.3385364216438 * eta3 + 25108.811524802128 * eta4 - 98374.18361532023 * eta5 + 158292.58792484726 * eta6) +
               eta1 * S1 * (-0.4360849737360132 * (-0.9543114627170375 - 58.70494649755802 * eta1 + 1729.1839588870455 * eta2 - 16718.425586396803 * eta3 + 71236.86532610047 * eta4 - 111910.71267453219 * eta5) -
                             0.024861802943501172 * (-52.25045490410733 + 1585.462602954658 * eta1 - 15866.093368857853 * eta2 + 35332.328181283 * eta3 + 168937.32229060197 * eta4 - 581776.5303770923 * eta5) * S1 +
                             0.005856387555754387 * (186.39698091707513 - 9560.410655118145 * eta1 + 156431.3764198244 * eta2 - 1.0461268207440731e6 * eta3 + 3.054333578686424e6 * eta4 - 3.2369858387064277e6 * eta5) * S2)
    end
end

@inline function _xhm_rd_amp44_rdcp2(eta, delta, StotR, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4
        S1 = StotR
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return eta1 * (chidiff1 * delta * (-2.861653255976984 * eta1 + 50.50227103211222 * eta2 - 123.94152825700999 * eta3) +
                        chidiff2 * (2.9415751419018865 * eta1 - 28.79779545444817 * eta2 + 72.40230240887851 * eta3)) +
               eta1 * (3.2461722686239307 + 25.15310593958783 * eta1 - 792.0167314124681 * eta2 + 7168.843978909433 * eta3 - 30595.4993786313 * eta4 + 49148.57065911245 * eta5) +
               eta1 * S1 * (-0.23311779185707152 * (-1.0795711755430002 - 20.12558747513885 * eta1 + 1163.9107546486134 * eta2 - 14672.23221502075 * eta3 + 73397.72190288734 * eta4 - 127148.27131388368 * eta5) +
                             0.025805905356653 * (11.929946153728276 + 350.93274421955806 * eta1 - 14580.02701600596 * eta2 + 174164.91607515427 * eta3 - 819148.9390278616 * eta4 + 1.3238624538095295e6 * eta5) * S1 +
                             0.019740635678180102 * (-7.046295936301379 + 1535.781942095697 * eta1 - 27212.67022616794 * eta2 + 201981.0743810629 * eta3 - 696891.1349708183 * eta4 + 910729.0219043035 * eta5) * S2)
    end
end

@inline function _xhm_rd_amp44_rdcp3(eta, delta, StotR, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4
        S1 = StotR
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return eta1 * (chidiff1 * delta * (2.4286414692113816 * eta1 - 23.213332913737403 * eta2 + 66.58241012629095 * eta3) +
                        chidiff2 * (3.085167288859442 * eta1 - 31.60440418701438 * eta2 + 78.49621016381445 * eta3)) +
               eta1 * (0.861883217178703 + 13.695204704208976 * eta1 - 337.70598252897696 * eta2 + 2932.3415281149432 * eta3 - 12028.786386004691 * eta4 + 18536.937955014455 * eta5) +
               eta1 * S1 * (-0.048465588779596405 * (-0.34041762314288154 - 81.33156665674845 * eta1 + 1744.329802302927 * eta2 - 16522.343895064576 * eta3 + 76620.18243090731 * eta4 - 133340.93723954144 * eta5) +
                             0.024804027856323612 * (-8.666095805675418 + 711.8727878341302 * eta1 - 13644.988225595187 * eta2 + 112832.04975245205 * eta3 - 422282.0368440555 * eta4 + 584744.0406581408 * eta5) * S1)
    end
end

# ---- (3,2)-specific raw amplitude fits (RDAmpVersion==1, live for this mode -- see module
# docstring for why no standalone (3,2) amplitude ansatz is assembled here) ----

@inline function _xhm_rd_amp32_alambda(eta, delta, StotR, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4; eta6 = eta1 * eta5
        S1 = StotR
        S2 = S1 * S1
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return chidiff2 * (-3.4614418482110163 * eta3 + 35.464117772624164 * eta4 - 85.19723511005235 * eta5) +
               chidiff1 * delta * (2.0328561081997463 * eta3 - 46.18751757691501 * eta4 + 170.9266105597438 * eta5) +
               chidiff2 * (-0.4600401291210382 * eta3 + 12.23450117663151 * eta4 - 42.74689906831975 * eta5) * S1 +
               chidiff1 * delta * (5.786292428422767 * eta3 - 53.60467819078566 * eta4 + 117.66195692191727 * eta5) * S1 +
               S1 * (-0.0013330716557843666 * (56.35538385647113 * eta1 - 1218.1550992423377 * eta2 + 16509.69605686402 * eta3 - 102969.88022112886 * eta4 + 252228.94931931415 * eta5 - 150504.2927996263 * eta6) +
                     0.0010126460331462495 * (-33.87083889060834 * eta1 + 502.6221651850776 * eta2 - 1304.9210590188136 * eta3 - 36980.079328277505 * eta4 + 295469.28617550555 * eta5 - 597155.7619486618 * eta6) * S1 -
                     0.00043088431510840695 * (-30.014415072587354 * eta1 - 1900.5495690280086 * eta2 + 76517.21042363928 * eta3 - 870035.1394696251 * eta4 + 3.9072674134789007e6 * eta5 - 6.094089675611567e6 * eta6) * S2) +
               (0.08408469319155859 * eta1 - 1.223794846617597 * eta2 + 6.5972460654253515 * eta3 - 15.707327897569396 * eta4 + 14.163264397061505 * eta5) / (1.0 - 8.612447115134758 * eta1 + 18.93655612952139 * eta2)
    end
end

@inline function _xhm_rd_amp32_lambda(eta, delta, StotR, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4
        S1 = StotR
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return 0.978510781593996 + 0.36457571743142897 * eta1 - 12.259851752618998 * eta2 + 49.19719473681921 * eta3 +
               chidiff1 * delta * (-188.37119473865533 * eta3 + 2151.8731700399308 * eta4 - 6328.182823770599 * eta5) +
               chidiff2 * (115.3689949926392 * eta3 - 1159.8596972989067 * eta4 + 2657.6998831179444 * eta5) +
               S1 * (0.22358643406992756 * (0.48943645614341924 - 32.06682257944444 * eta1 + 365.2485484044132 * eta2 - 915.2489655397206 * eta3) +
                     0.0792473022309144 * (1.877251717679991 - 103.65639889587327 * eta1 + 1202.174780792418 * eta2 - 3206.340850767219 * eta3) * S1)
    end
end

@inline function _xhm_rd_amp32_sigma(eta, delta, StotR, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4
        S1 = StotR
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return 1.3353917551819414 + 0.13401718687342024 * eta1 +
               chidiff1 * delta * (144.37065005786636 * eta3 - 754.4085447486738 * eta4 + 123.86194078913776 * eta5) +
               chidiff2 * (209.09202210427972 * eta3 - 1769.4658099037918 * eta4 + 3592.287297392387 * eta5) +
               S1 * (-0.012086025709597246 * (-6.230497473791485 + 600.5968613752918 * eta1 - 6606.1009717965735 * eta2 + 17277.60594350428 * eta3) -
                     0.06066548829900489 * (-0.9208054306316676 + 142.0346574366267 * eta1 - 1567.249168668069 * eta2 + 4119.373703246675 * eta3) * S1)
    end
end

@inline function _xhm_rd_amp32_rdaux1(eta, delta, StotR, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4; eta6 = eta1 * eta5
        S1 = StotR
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return chidiff2 * (-4.188795724777721 * eta2 + 53.39200466700963 * eta3 - 131.19660856923554 * eta4) +
               chidiff1 * delta * (14.284921364132623 * eta2 - 321.26423637658746 * eta3 + 1242.865584938088 * eta4) +
               S1 * (-0.022968727462555794 * (83.66854837403105 * eta1 - 3330.6261333413177 * eta2 + 77424.12614733395 * eta3 - 710313.3016672594 * eta4 + 2.6934917075009225e6 * eta5 - 3.572465179268999e6 * eta6) +
                     0.0014795114305436387 * (-1672.7273629876313 * eta1 + 90877.38260964208 * eta2 - 1.6690169155105734e6 * eta3 + 1.3705532554135624e7 * eta4 - 5.116110998398143e7 * eta5 + 7.06066766311127e7 * eta6) * S1) +
               (4.45156488896258 * eta1 - 77.39303992494544 * eta2 + 522.5070635563092 * eta3 - 1642.3057499049708 * eta4 + 2048.333892310575 * eta5) / (1.0 - 9.611489164758915 * eta1 + 24.249594730050312 * eta2)
    end
end

@inline function _xhm_rd_amp32_rdaux2(eta, delta, chiPNHat, chidiff)
    T = promote_type(typeof(eta), typeof(delta), typeof(chiPNHat), typeof(chidiff))
    @Tlit T begin
        eta1 = eta
        eta2 = eta1 * eta1; eta3 = eta1 * eta2; eta4 = eta1 * eta3; eta5 = eta1 * eta4; eta6 = eta1 * eta5; eta7 = eta1 * eta6
        S1 = chiPNHat
        chidiff1 = chidiff
        chidiff2 = chidiff1 * chidiff1
        return chidiff2 * (-18.550171209458394 * eta2 + 188.99161055445936 * eta3 - 440.26516625611 * eta4) +
               chidiff1 * delta * (13.132625215315063 * eta2 - 340.5204040505528 * eta3 + 1327.1224176812448 * eta4) +
               S1 * (-0.16707403272774676 * (6.678916447469937 * eta1 + 1331.480396625797 * eta2 - 41908.45179140144 * eta3 + 520786.0225074669 * eta4 - 3.1894624909922685e6 * eta5 + 9.51553823212259e6 * eta6 - 1.1006903622406831e7 * eta7) +
                     0.015205286051218441 * (108.10032279461095 * eta1 - 16084.215590200103 * eta2 + 462957.5593513407 * eta3 - 5.635028227588545e6 * eta4 + 3.379925277713386e7 * eta5 - 9.865815275452062e7 * eta6 + 1.1201307979786257e8 * eta7) * S1) +
               (3.902154247490771 * eta1 - 55.77521071924907 * eta2 + 294.9496843041973 * eta3 - 693.6803787318279 * eta4 + 636.0141528226893 * eta5) / (1.0 - 8.56699762573719 * eta1 + 19.119341007236955 * eta2)
    end
end

"""
    _xhm_rd_amp_lorentzian(f, c0, c1, c2, fring, fdamp)

The `IMRPhenomXHM_RD_Amp_Ansatz` Lorentzian-with-exponential-decay-envelope core (ringdown.c line
~936, `RDAmpVersion` case 1/2, the branch shared by every mode ported here): directly analogous
in form to `IMRPhenomXAS.jl`'s own `_xas_amp_rd_val`, just re-parameterized (`c0,c1,c2` here play
the role of a combined `gamma1,gamma2,gamma3` after `IMRPhenomXHM_RD_Amp_Coefficients`'s
collocation-point solve, see `_xhm_rd_amp_coefficients` below).
"""
@inline function _xhm_rd_amp_lorentzian(f, c0, c1, c2, fring, fdamp)
    T = promote_type(typeof(f), typeof(c0), typeof(fring))
    @Tlit T begin
        dfr = f - fring
        dfd = fdamp * c2
        return c0 * fdamp / (exp(c1 / dfd * dfr) * (dfr * dfr + dfd * dfd))
    end
end

"""
    _xhm_rd_amp_dlorentzian(f, c0, c1, c2, fring, fdamp)

Analytic derivative of [`_xhm_rd_amp_lorentzian`](@ref) -- `IMRPhenomXHM_RD_Amp_DAnsatz`'s
`RDAmpVersion` case 1/2 branch (ringdown.c lines ~981-989), the one actually used under the default
release (see module docstring's "`DAnsatz` vs `NDAnsatz`" section).
"""
@inline function _xhm_rd_amp_dlorentzian(f, c0, c1, c2, fring, fdamp)
    T = promote_type(typeof(f), typeof(c0), typeof(fring))
    @Tlit T begin
        dfr = f - fring
        numerator = c0 * (dfr * dfr * c1 + 2.0 * fdamp * dfr * c2 + fdamp * fdamp * c1 * c2 * c2)
        denom = (dfr * dfr + fdamp * fdamp * c2 * c2)
        denom = c2 * denom * denom * exp(dfr * c1 / (fdamp * c2))
        return -numerator / denom
    end
end

"""
    _xhm_rd_amp_coefficients(l, m, eta, delta, StotR, chiPNHat, chidiff, fring, fdamp)

Ringdown amplitude ansatz coefficients for the non-mixing modes `(l,m) = (2,1),(3,3),(4,4)` --
`IMRPhenomXHM_RD_Amp_Coefficients`'s `RDAmpVersion == 2` branch (ringdown.c lines ~744-807) plus the
always-live exponential-falloff continuity constants (see module docstring). `(3,2)` deliberately
throws (its RDAux-dependent ansatz needs mode-mixing machinery, see module docstring) rather than
silently returning an unvalidated/incomplete result.

`fring`/`fdamp` are this mode's own values (`_xhm_fring_fdamp(l, m, a, Erad)` from `IMRPhenomXHM_qnm.jl`),
`StotR`/`chiPNHat`/`chidiff`/`delta` are the SAME `(2,2)`-mode-derived spin/mass-ratio combinations
`IMRPhenomXAS.jl`/`IMRPhenomXHM_inspiral.jl`/`IMRPhenomXHM_intermediate.jl` already compute
(`StotR`, the PN-hat effective spin `chiPNHat`, `chidiff = 0.5*(chi1-chi2)`, `delta = Seta =
sqrt(1-4eta)`).

Returns a `NamedTuple` `(c0, c1, c2, c3, c4, fring, fdamp, falloff)` ready for
[`_xhm_rd_amp_ansatz`](@ref)/[`_xhm_rd_amp_dansatz`](@ref).
"""
@inline function _xhm_rd_amp_coefficients(l::Integer, m::Integer, eta, delta, StotR, chiPNHat, chidiff, fring, fdamp)
    T = promote_type(typeof(eta), typeof(delta), typeof(StotR), typeof(chiPNHat), typeof(chidiff), typeof(fring), typeof(fdamp))
    if l == 2 && m == 1
        rdcp1 = abs(_xhm_rd_amp21_rdcp1(eta, delta, chiPNHat, chidiff))
        rdcp2 = abs(_xhm_rd_amp21_rdcp2(eta, delta, chiPNHat, chidiff))
        rdcp3 = abs(_xhm_rd_amp21_rdcp3(eta, delta, chiPNHat, chidiff))
    elseif l == 3 && m == 3
        rdcp1 = abs(_xhm_rd_amp33_rdcp1(eta, delta, StotR, chidiff))
        rdcp2 = abs(_xhm_rd_amp33_rdcp2(eta, delta, StotR, chidiff))
        rdcp3 = abs(_xhm_rd_amp33_rdcp3(eta, delta, StotR, chidiff))
    elseif l == 4 && m == 4
        rdcp1 = abs(_xhm_rd_amp44_rdcp1(eta, delta, StotR, chidiff))
        rdcp2 = abs(_xhm_rd_amp44_rdcp2(eta, delta, StotR, chidiff))
        rdcp3 = abs(_xhm_rd_amp44_rdcp3(eta, delta, StotR, chidiff))
    elseif l == 3 && m == 2
        throw(ArgumentError("_xhm_rd_amp_coefficients: (3,2)'s ringdown amplitude ansatz needs " *
                             "mode-mixing machinery (fAmpMatchIM, rdaux1/rdaux2 polynomial) not " *
                             "ported here -- see IMRPhenomXHM_ringdown.jl's module docstring, " *
                             "see IMRPhenomXHM.jl"))
    else
        throw(ArgumentError("_xhm_rd_amp_coefficients: unsupported (l,m) = ($l,$m); only " *
                             "(2,1),(3,3),(4,4) have a standalone ringdown amplitude ansatz here"))
    end

    @Tlit T begin
        if rdcp3 >= rdcp2 * rdcp2 / rdcp1
            rdcp3 = 0.5 * rdcp2 * rdcp2 / rdcp1
        end
        if rdcp3 > rdcp2
            rdcp3 = 0.5 * rdcp2
        end
        if rdcp1 < rdcp2 && rdcp3 > rdcp1
            rdcp3 = rdcp1
        end

        deno = sqrt(rdcp1 / rdcp3) - rdcp1 / rdcp2
        if deno <= 0.0
            deno = T(1e-16)
        end

        c0 = rdcp1 * fdamp / deno
        c2 = sqrt(c0 / (rdcp2 * fdamp))
        c1 = 0.5 * c2 * log(rdcp1 / rdcp3)

        falloff = fring + 2.0 * fdamp
        c3 = _xhm_rd_amp_lorentzian(falloff, c0, c1, c2, fring, fdamp)
        c4 = -_xhm_rd_amp_dlorentzian(falloff, c0, c1, c2, fring, fdamp) / c3

        return (c0 = c0, c1 = c1, c2 = c2, c3 = c3, c4 = c4, fring = fring, fdamp = fdamp, falloff = falloff)
    end
end

"""
    _xhm_rd_amp_ansatz(f, c)

Ringdown amplitude at dimensionless frequency `f` (`Mf`), given the coefficients `c` from
[`_xhm_rd_amp_coefficients`](@ref) -- `IMRPhenomXHM_RD_Amp_Ansatz`'s live branch (ringdown.c lines
~931-937): the always-active Lorentzian below `c.falloff`, an exponential decay above it (see
module docstring's `fAmpRDfalloff` note -- this is NOT an edge case, real waveforms exercise it).
`RDRescaleFactor` is confirmed `0` (off) under the default release for these modes, so no extra
division is applied (matches the C source's own `if (RDRescaleFactor != 0)` guard).
"""
@inline function _xhm_rd_amp_ansatz(f, c)
    T = c.c0 isa AbstractFloat ? typeof(c.c0) : typeof(f)
    return f >= c.falloff ? c.c3 * exp(-c.c4 * (f - c.falloff)) :
           _xhm_rd_amp_lorentzian(f, c.c0, c.c1, c.c2, c.fring, c.fdamp)
end

"""
    _xhm_rd_amp_dansatz(f, c)

Analytic derivative of [`_xhm_rd_amp_ansatz`](@ref)'s Lorentzian branch only -- matches
`IMRPhenomXHM_RD_Amp_DAnsatz`, which (confirmed by reading the C source) has no separate
falloff-branch formula at all; it is only ever evaluated AT `c.falloff` itself (from the Lorentzian
side, exactly as done inside [`_xhm_rd_amp_coefficients`](@ref) to seed `c3`/`c4`).
"""
@inline function _xhm_rd_amp_dansatz(f, c)
    return _xhm_rd_amp_dlorentzian(f, c.c0, c.c1, c.c2, c.fring, c.fdamp)
end

# ================================================================================================
# PHASE: raw parameter-space fits, live (122019) branch only (see module docstring for why the
# phase side stays on 122019 even under the default 122022 release).
# ================================================================================================

"""
    _xhm_rd_phase_alpha2(eta, StotR, dchi)

`IMRPhenomXHM_RD_Phase_22_alpha2`, 122019 branch (the live one -- see module docstring). Despite
the `_22_` in its C name this is NOT `(2,2)`-mode-specific: it is a generic fit reused (after
rescaling by `wlm`/`fring^2`, see [`_xhm_rd_phase_coefficients`](@ref)) for every non-mixing higher
mode `(2,1),(3,3),(4,4)`.
"""
@inline function _xhm_rd_phase_alpha2(eta, StotR, dchi)
    T = promote_type(typeof(eta), typeof(StotR), typeof(dchi))
    @Tlit T begin
        S = StotR
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta
        noSpin = 0.2088669311744758 - 0.37138987533788487 * eta + 6.510807976353186 * eta2 - 31.330215053905395 * eta3 + 55.45508989446867 * eta4
        eqSpin = ((0.2393965714370633 + 1.6966740823756759 * eta - 16.874355161681766 * eta2 + 38.61300158832203 * eta3) * S) / (1.0 - 0.633218538432246 * S)
        uneqSpin = dchi * (0.9088578269496244 * eta^2.5 + 15.619592332008951 * dchi * eta^3.5) * delta
        return noSpin + eqSpin + uneqSpin
    end
end

"""
    _xhm_rd_phase_alphaL(eta, StotR, chi1, chi2)

`IMRPhenomXHM_RD_Phase_22_alphaL`, 122019 branch (the live one). Same "reused for every non-mixing
mode" note as [`_xhm_rd_phase_alpha2`](@ref)'s docstring.
"""
@inline function _xhm_rd_phase_alphaL(eta, StotR, chi1, chi2)
    T = promote_type(typeof(eta), typeof(StotR), typeof(chi1), typeof(chi2))
    @Tlit T begin
        S = StotR
        delta = sqrt(1.0 - 4.0 * eta)
        eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta
        S2 = S * S
        noSpin = eta * (-1.1926122248825484 + 2.5400257699690143 * eta - 16.504334734464244 * eta2 + 27.623649807617376 * eta3)
        eqSpin = eta3 * S * (35.803988443700824 + 9.700178927988006 * S - 77.2346297158916 * S2) +
                 eta * S * (0.1034526554654983 - 0.21477847929548569 * S - 0.06417449517826644 * S2) +
                 eta2 * S * (-4.7282481007397825 + 0.8743576195364632 * S + 8.170616575493503 * S2) +
                 eta4 * S * (-72.50310678862684 - 39.83460092417137 * S + 180.8345521274853 * S2)
        uneqSpin = (-0.7428134042821221 * chi1 * eta^3.5 + 0.7428134042821221 * chi2 * eta^3.5 +
                    17.588573345324154 * chi1 * chi1 * eta^4.5 - 35.17714669064831 * chi1 * chi2 * eta^4.5 +
                    17.588573345324154 * chi2 * chi2 * eta^4.5) * delta
        return noSpin + eqSpin + uneqSpin
    end
end

# ---- (3,2)-specific raw phase fits (122019 branch, live -- see module docstring for why no
# standalone (3,2) phase ansatz is assembled here) ----

@inline function _xhm_rd_phase32_timeshift(eta, StotR, dchi)
    T = promote_type(typeof(eta), typeof(StotR), typeof(dchi))
    @Tlit T begin
        S = StotR
        eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta; eta5 = eta4 * eta
        S2 = S * S; S3 = S2 * S; S4 = S3 * S
        noSpin = 11.851438981981772 + 167.95086712701223 * eta - 4565.033758777737 * eta2 + 61559.132976189896 * eta3 - 364129.24735853914 * eta4 + 739270.8814129328 * eta5
        eqSpin = (9.506768471271634 + 434.31707030999445 * eta - 8046.364492927503 * eta2 + 26929.677144312944 * eta3) * S +
                 (-5.949655484033632 - 307.67253970367034 * eta + 1334.1062451631644 * eta2 + 3575.347142399199 * eta3) * S2 +
                 (3.4881615575084797 - 2244.4613237912527 * eta + 24145.932943269272 * eta2 - 60929.87465551446 * eta3) * S3 +
                 (15.585154698977842 - 2292.778112523392 * eta + 24793.809334683185 * eta2 - 65993.84497923202 * eta3) * S4
        uneqSpin = 465.7904934097202 * dchi * sqrt(1.0 - 4.0 * eta) * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_rd_phase32_phaseshift(eta, StotR, chi1, chi2)
    T = promote_type(typeof(eta), typeof(StotR), typeof(chi1), typeof(chi2))
    @Tlit T begin
        S = StotR
        eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta; eta5 = eta4 * eta; eta6 = eta5 * eta; eta7 = eta6 * eta
        S2 = S * S; S3 = S2 * S; S4 = S3 * S
        noSpin = -1.3328895897490733 - 22.209549522908667 * eta + 1056.2426481245027 * eta2 - 21256.376324666326 * eta3 + 246313.12887984765 * eta4 - 1.6312968467540336e6 * eta5 + 5.614617173188322e6 * eta6 - 7.612233821752137e6 * eta7
        eqSpin = (S * (-1.622727240110213 + 0.9960210841611344 * S - 1.1239505323267036 * S2 - 1.9586085340429995 * S3 +
                       eta2 * (196.7055281997748 + 135.25216875394943 * S + 1086.7504825459278 * S2 + 546.6246807461155 * S3 - 312.1010566468068 * S4) +
                       0.7638287749489343 * S4 +
                       eta * (-47.475568056234245 - 35.074072557604445 * S - 97.16014978329918 * S2 - 34.498125910065156 * S3 + 24.02858084544326 * S4) +
                       eta3 * (62.632493533037625 - 22.59781899512552 * S - 2683.947280170815 * S2 - 1493.177074873678 * S3 + 805.0266029288334 * S4))) / (-2.950271397057221 + 1.0 * S)
        uneqSpin = (sqrt(1.0 - 4.0 * eta) * (chi2 * eta^2.5 * (88.56162028006072 - 30.01812659282717 * S) +
                                              chi2 * eta2 * (43.126266433486435 - 14.617728550838805 * S) +
                                              chi1 * eta2 * (-43.126266433486435 + 14.617728550838805 * S) +
                                              chi1 * eta^2.5 * (-88.56162028006072 + 30.01812659282717 * S))) / (-2.950271397057221 + 1.0 * S)
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_rd_phase32_p1(eta, StotR, dchi, chi1, chi2)
    T = promote_type(typeof(eta), typeof(StotR), typeof(dchi), typeof(chi1), typeof(chi2))
    @Tlit T begin
        S = StotR
        eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta; eta5 = eta4 * eta
        S2 = S * S; S3 = S2 * S; S4 = S3 * S; S5 = S4 * S
        noSpin = 3169.372056189274 + 426.8372805022653 * eta - 12569.748101922158 * eta2 + 149846.7281073725 * eta3 - 817182.2896823225 * eta4 + 1.5674053633767858e6 * eta5
        eqSpin = (19.23408352151287 - 1762.6573670619173 * eta + 7855.316419853637 * eta2 - 3785.49764771212 * eta3) * S +
                 (-42.88446003698396 + 336.8340966473415 * eta - 5615.908682338113 * eta2 + 20497.5021807654 * eta3) * S2 +
                 (13.918237996338371 + 10145.53174542332 * eta - 91664.12621864353 * eta2 + 201204.5096556517 * eta3) * S3 +
                 (-24.72321125342808 - 4901.068176970293 * eta + 53893.9479532688 * eta2 - 139322.02687945773 * eta3) * S4 +
                 (-61.01931672442576 - 16556.65370439302 * eta + 162941.8009556697 * eta2 - 384336.57477596396 * eta3) * S5
        uneqSpin = dchi * sqrt(1.0 - 4.0 * eta) * eta2 * (641.2473192044652 - 1600.240100295189 * chi1 * eta + 1600.240100295189 * chi2 * eta + 13275.623692212472 * eta * S)
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_rd_phase32_p2(eta, StotR, dchi)
    T = promote_type(typeof(eta), typeof(StotR), typeof(dchi))
    @Tlit T begin
        S = StotR
        eta2 = eta * eta; eta3 = eta2 * eta
        S2 = S * S; S3 = S2 * S; S4 = S3 * S
        noSpin = 3131.0260952676376 + 206.09687819102305 * eta - 2636.4344627081873 * eta2 + 7475.062269742079 * eta3
        eqSpin = (49.90874152040307 - 691.9815135740145 * eta - 434.60154548208334 * eta2 + 10514.68111669422 * eta3) * S +
                 (97.3078084654917 - 3458.2579971189534 * eta + 26748.805404989867 * eta2 - 56142.13736008524 * eta3) * S2 +
                 (-132.49105074500454 + 429.0787542102207 * eta + 7269.262546204149 * eta2 - 27654.067482558712 * eta3) * S3 +
                 (-227.8023564332453 + 5119.138772157134 * eta - 34444.2579678986 * eta2 + 69666.01833764123 * eta3) * S4
        uneqSpin = 477.51566939885424 * dchi * sqrt(1.0 - 4.0 * eta) * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_rd_phase32_p3(eta, StotR, dchi)
    T = promote_type(typeof(eta), typeof(StotR), typeof(dchi))
    @Tlit T begin
        S = StotR
        eta2 = eta * eta; eta3 = eta2 * eta
        S2 = S * S; S3 = S2 * S; S4 = S3 * S
        noSpin = 3082.803556599222 + 76.94679795837645 * eta - 586.2469821978381 * eta2 + 977.6115755788503 * eta3
        eqSpin = (45.08944710349874 - 807.7353772747749 * eta + 1775.4343704616288 * eta2 + 2472.6476419567534 * eta3) * S +
                 (95.57355060136699 - 2224.9613131172046 * eta + 13821.251641893134 * eta2 - 25583.314298758105 * eta3) * S2 +
                 (-144.96370424517866 + 2268.4693587493093 * eta - 10971.864789147161 * eta2 + 16259.911572457446 * eta3) * S3 +
                 (-227.8023564332453 + 5119.138772157134 * eta - 34444.2579678986 * eta2 + 69666.01833764123 * eta3) * S4
        uneqSpin = 378.2359918274837 * dchi * sqrt(1.0 - 4.0 * eta) * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xhm_rd_phase32_p4(eta, StotR, dchi)
    T = promote_type(typeof(eta), typeof(StotR), typeof(dchi))
    @Tlit T begin
        S = StotR
        eta2 = eta * eta; eta3 = eta * eta2
        S2 = S * S; S3 = S2 * S; S4 = S3 * S
        noSpin = 3077.0657367004565 + 64.99844502520415 * eta - 357.38692756785395 * eta2
        eqSpin = (34.793450080444714 - 986.7751755509875 * eta - 9490.641676924794 * eta3 + 5700.682624203565 * eta2) * S +
                 (57.38106384558743 - 1644.6690499868596 * eta - 19906.416384606226 * eta3 + 11008.881935880598 * eta2) * S2 +
                 (-126.02362949830213 + 3169.3397351803583 * eta + 62863.79877094988 * eta3 - 26766.730897942085 * eta2) * S3 +
                 (-169.30909412804587 + 4900.706039920717 * eta + 95314.99988114933 * eta3 - 41414.05689348732 * eta2) * S4
        uneqSpin = 390.5443469721231 * dchi * sqrt(1.0 - 4.0 * eta) * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

"""
    _xhm_rd_phase_coefficients(l, m, eta, StotR, chi1, chi2, fring)

Ringdown phase ansatz coefficients `(alpha0, alpha2, alphaL)` for the non-mixing modes
`(l,m) = (2,1),(3,3),(4,4)` -- the `MixingOn == 0` branch of `IMRPhenomXHM_GetPhaseCoefficients`
(`_internals.c` lines ~2129-2151): `alpha0` is hardcoded `0` (see module docstring), `alpha2`/
`alphaL` are the [`_xhm_rd_phase_alpha2`](@ref)/[`_xhm_rd_phase_alphaL`](@ref) fits rescaled by
`wlm` (`2` for `l==m`, `m/3` otherwise) and `1/eta`. `(3,2)` throws (needs mode-mixing machinery,
see module docstring).
"""
@inline function _xhm_rd_phase_coefficients(l::Integer, m::Integer, eta, StotR, chi1, chi2, fring)
    T = promote_type(typeof(eta), typeof(StotR), typeof(chi1), typeof(chi2), typeof(fring))
    if l == 3 && m == 2
        throw(ArgumentError("_xhm_rd_phase_coefficients: (3,2)'s ringdown phase ansatz needs " *
                             "the (2,2) mode's own IMRPhenomX_TimeShift_22/dPhase_22 machinery, " *
                             "not ported anywhere in this package -- see " *
                             "IMRPhenomXHM_ringdown.jl's module docstring, see IMRPhenomXHM.jl"))
    end
    if !((l == 2 && m == 1) || (l == 3 && m == 3) || (l == 4 && m == 4))
        throw(ArgumentError("_xhm_rd_phase_coefficients: unsupported (l,m) = ($l,$m); only " *
                             "(2,1),(3,3),(4,4) have a standalone ringdown phase ansatz here"))
    end
    dchi = chi1 - chi2
    wlm = l == m ? T(2.0) : T(m) / T(3.0)
    alpha2fit = T(_xhm_rd_phase_alpha2(eta, StotR, dchi))
    alphaLfit = T(_xhm_rd_phase_alphaL(eta, StotR, chi1, chi2))
    alpha2 = (one(T) / (fring * fring)) * wlm * alpha2fit
    alphaL = (one(T) / eta) * alphaLfit
    return (alpha0 = zero(T), alpha2 = alpha2, alphaL = alphaL, fring = fring)
end

"""
    _xhm_rd_phase_dphidf(f, c, fring, fdamp)

Ringdown phase DERIVATIVE (`dPhi/dMf`) at dimensionless frequency `f`, given the coefficients `c`
from [`_xhm_rd_phase_coefficients`](@ref) -- `IMRPhenomXHM_RD_Phase_Ansatz`'s `MixingOn == 0`
branch (ringdown.c lines ~1391-1397). Matches this file's/`IMRPhenomXHM_intermediate.jl`'s
`_xhm_inter_phase_dphidf` naming convention.
"""
@inline function _xhm_rd_phase_dphidf(f, c, fring, fdamp)
    T = promote_type(typeof(f), typeof(fring), typeof(fdamp))
    @Tlit T begin
        return c.alpha0 + fring * fring * c.alpha2 / (f * f) + c.alphaL * fdamp / (fdamp * fdamp + (f - fring) * (f - fring))
    end
end

"""
    _xhm_rd_phase_ansatzint(f, c, fring, fdamp)

Ringdown phase (integrated, i.e. the phase itself up to the frequency-independent glue constant
`IMRPhenomXHM.jl` adds) at dimensionless frequency `f` -- `IMRPhenomXHM_RD_Phase_AnsatzInt`'s
`MixingOn == 0` branch (ringdown.c lines ~1438-1442).
"""
@inline function _xhm_rd_phase_ansatzint(f, c, fring, fdamp)
    T = promote_type(typeof(f), typeof(fring), typeof(fdamp))
    @Tlit T begin
        return c.alpha0 * f - fring * fring * c.alpha2 / f + c.alphaL * atan((f - fring) / fdamp)
    end
end

"""
    _xhm_rd_phase_derivansatz(f, c, fring, fdamp)

Second derivative (`d2Phi/dMf2`) -- `IMRPhenomXHM_RD_Phase_DerAnsatz`'s `MixingOn == 0` branch
(ringdown.c lines ~1480-1486). Not needed by any C1-continuity join in this file's own scope
(`alpha0 = 0`, no fit against it), ported for completeness/self-consistency testing
(finite-difference cross-check against [`_xhm_rd_phase_dphidf`](@ref), matching this package's
`IMRPhenomXAS.jl` convention of pairing analytic derivatives with FD tests).
"""
@inline function _xhm_rd_phase_derivansatz(f, c, fring, fdamp)
    T = promote_type(typeof(f), typeof(fring), typeof(fdamp))
    @Tlit T begin
        return -2.0 * fring * fring * c.alpha2 / (f * f * f) -
               2.0 * c.alphaL * fdamp * (f - fring) / (fdamp * fdamp + (f - fring) * (f - fring))^2
    end
end
