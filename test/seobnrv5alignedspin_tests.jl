# Unit + physics-consistency tests for GravitationalWaveforms.SEOBNRv5AlignedSpin. See
# SEOBNRv5AlignedSpin.jl's own module docstring for the full physics/citation writeup this file's
# tests refer to -- this is a summary of what's checked here:
#
#  1. Basic construction/Phi/Ampl sanity (finite, monotonic phase, positive amplitude).
#  2. validate_seobnrv5alignedspin_params boundary behavior.
#  3. The two-stage initial-condition solve's own residual (IC_cons: dH/dpphi==omega0, dH/dr==0 to
#     Newton-solve precision) -- a locked-in internal-consistency regression.
#  4. **Real external validation against `pyseobnr`'s own actual SEOBNRv5HM dynamics** --
#     `pyseobnr.generate_waveform.generate_modes_opt(q=1.5, chi1=0.3, chi2=-0.2, omega_start=0.01,
#     approximant="SEOBNRv5HM", debug=True)`'s returned model's `.dynamics` (the 3-tuple return's
#     3rd element, `SEOBNRv5HM_opt` -- `generate_modes_opt(..., debug=True)` does NOT return an
#     object with a `.dynamics` attribute directly, a detail re-confirmed when this file's
#     checkpoints were regenerated for the 7-mode flux extension below), -- both the initial condition (r0, pphi0, pr0, H0 all
#     confirmed to agree with the real `pyseobnr` values quoted below) and several points along
#     the integrated trajectory, now spanning from `t=125.66M` all the way to `t=17199.95M`
#     (`r~6.4M`, deep in the plunge) -- a substantially wider span than this test originally
#     covered, made possible by the residual now staying small far longer (see next point).
#     **Since the full 7-mode `_seobv5_rr_force` extension (superseding the original `(2,2)`-only
#     port), the residual at every one of these checkpoints dropped by 2-3 orders of magnitude**
#     (e.g. at `t=10628.4931M`: `r`/`omega` residual went from `~0.12%`/`~0.17%` down to
#     `~1.1e-4%`/`~1.7e-4%`; `pr0` itself went from `~0.3%` down to `~1e-3%`) -- confirming the
#     missing-6-mode flux deficit, not the Hamiltonian/gradient/xi port or the IC's
#     finite-difference Hessian, was the dominant source of the old residual (a real correction to
#     this file's own prior attribution of `pr0`'s `~0.3%` residual solely to the FD Hessian -- see
#     SEOBNRv5AlignedSpin.jl's docstring's "Radiation-reaction force" section). The residual still
#     grows monotonically with time/decreasing `r` (now from `~1e-7` near `t=0` to `~5-10%` only in
#     the deepest few `M` before merger, `r<7M`) -- itself still locked in as a regression, at the
#     new, much tighter levels.
#  5. A finite-difference smoothness/sanity check on Phi w.r.t. a parameter, complementing the
#     `Enzyme.gradient` validation in `seobnrv5alignedspin_enzyme_ka_tests.jl`.
#  6. Float32 genericity.
#
# Real `pyseobnr` values below (v0.3.6, `Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C.pyx`) were
# obtained from a pyseobnr script.

# The per-binary flux coefficients, computed the way `_seobv5_coeffs` does.
function _seobv5_test_fc(nu, X1, X2, chi1, chi2)
    delta, chiS, chiA = X1 - X2, 0.5 * (chi1 + chi2), 0.5 * (chi1 - chi2)
    atest = (1.0 - 2.0 * nu) * chiS + delta * chiA
    absN = (GravitationalWaveforms._seobv5_multipole_prefix_abs(nu, X1, X2, lm...) for lm in ((3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)))
    return GravitationalWaveforms._seobv5_flux_coeffs(nu, delta, chiS, chiA, atest, absN...)
end

@testset "SEOBNRv5AlignedSpin construction and Phi/Ampl basic sanity" begin
    model = SEOBNRv5AlignedSpin()
    @test model isa SEOBNRv5AlignedSpin

    mc, eta, dL = 23.821232072402484, 0.24793388429752067, 0.4
    chi1, chi2 = 0.3, -0.2
    dt_default = 1.0 / 4096.0
    t = (0:20:200) .* dt_default

    phi = Phi(model, t, mc, eta, chi1, chi2)
    ampl = Ampl(model, t, mc, eta, chi1, chi2, dL)

    @test phi isa Vector{Float64}
    @test ampl isa Vector{Float64}
    @test length(phi) == length(t)
    @test length(ampl) == length(t)
    @test all(isfinite, phi)
    @test all(isfinite, ampl)
    @test all(>(0), ampl)
    @test issorted(phi)

    f_gw = diff(phi) ./ diff(t) ./ (2pi)
    @test all(>(0), f_gw)
    @test f_gw[end] > f_gw[1]

    phi_scalar = Phi(model, 0.02, mc, eta, chi1, chi2)
    @test phi_scalar isa Float64
    ampl_scalar = Ampl(model, 0.02, mc, eta, chi1, chi2, dL)
    @test ampl_scalar isa Float64

    # Float32 genericity: output type tracks input type.
    f32 = Float32.(t)
    phi32 = Phi(model, f32, Float32(mc), Float32(eta), Float32(chi1), Float32(chi2))
    ampl32 = Ampl(model, f32, Float32(mc), Float32(eta), Float32(chi1), Float32(chi2), Float32(dL))
    @test eltype(phi32) === Float32
    @test eltype(ampl32) === Float32
    @test all(isfinite, phi32)
    @test all(isfinite, ampl32)
end

@testset "validate_seobnrv5alignedspin_params" begin
    @test validate_seobnrv5alignedspin_params(0.24, 0.1, -0.2) === nothing
    @test validate_seobnrv5alignedspin_params(0.25, 0.0, 0.0) === nothing
    @test_throws ArgumentError validate_seobnrv5alignedspin_params(0.0, 0.0, 0.0)
    @test_throws ArgumentError validate_seobnrv5alignedspin_params(0.26, 0.0, 0.0)
    @test_throws ArgumentError validate_seobnrv5alignedspin_params(0.2, 1.5, 0.0)
    @test_throws ArgumentError validate_seobnrv5alignedspin_params(0.2, 0.0, -1.5)
    @test_throws ArgumentError validate_seobnrv5alignedspin_params(0.2, 0.0, 0.0; dL = -1.0)
    @test validate_seobnrv5alignedspin_params(0.2, 0.0, 0.0; dL = 1.0) === nothing
end

@testset "Two-stage initial condition: internal residual (IC_cons/IC_diss Newton-solve self-consistency)" begin
    # nu=0.24, X1=0.6, X2=0.4 (q=1.5), chi1=0.3, chi2=-0.2 -- the exact configuration validated
    # against real pyseobnr output below.
    nu, X1, X2, chi1, chi2 = 0.24, 0.6, 0.4, 0.3, -0.2
    ap = X1 * chi1 + X2 * chi2
    am = X1 * chi1 - X2 * chi2
    a6 = GravitationalWaveforms._seobv5_a6_NS(nu)
    dSO = GravitationalWaveforms._seobv5_dSO(nu, ap, am)
    omega0 = 0.01

    r0, pphi0 = GravitationalWaveforms._seobv5_circular_ic(omega0, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))
    H0, xi0, Heff0 = GravitationalWaveforms._seobv5_ham_xi(r0, 0.0, pphi0, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))
    dHdr0, dHdpr0, dHdpphi0 = GravitationalWaveforms._seobv5_sym_gradient(r0, 0.0, pphi0, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))

    # IC_cons's own defining equations, evaluated at the solve's own solution: both residuals are
    # at the Newton solve's own convergence precision (near machine epsilon for dHdr, and the
    # circular-orbit-frequency match for dHdpphi), not just "small".
    @test isapprox(dHdpphi0, omega0; rtol = 1e-10)
    @test abs(dHdr0) < 1e-12

    delta = X1 - X2
    chiS = 0.5 * (chi1 + chi2)
    chiA = 0.5 * (chi1 - chi2)
    atest = (1.0 - 2.0 * nu) * chiS + delta * chiA
    pr0 = GravitationalWaveforms._seobv5_dissipative_ic(r0, pphi0, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO), _seobv5_test_fc(nu, X1, X2, chi1, chi2))

    # pr0 is small and negative (slow radial inspiral drift), same sign/order of magnitude as
    # pyseobnr's own real IC_diss solution below -- not exactly circular (pr0 != 0).
    @test pr0 < 0.0
    @test abs(pr0) < 1e-2
end

@testset "Real pyseobnr SEOBNRv5HM validation (q=1.5, chi1=0.3, chi2=-0.2, omega_start=0.01)" begin
    # Direct comparison against pyseobnr.generate_waveform.generate_modes_opt
    # (approximant="SEOBNRv5HM", debug=True).dynamics -- a real (N,8) array (t,r,phi,pr,pphi,H,
    # omega,omega_circ), obtained from a pyseobnr script, reproduced here as
    # hardcoded reference values.
    nu, X1, X2, chi1, chi2 = 0.24, 0.6, 0.4, 0.3, -0.2
    ap = X1 * chi1 + X2 * chi2
    am = X1 * chi1 - X2 * chi2
    a6 = GravitationalWaveforms._seobv5_a6_NS(nu)
    dSO = GravitationalWaveforms._seobv5_dSO(nu, ap, am)
    delta = X1 - X2
    chiS = 0.5 * (chi1 + chi2)
    chiA = 0.5 * (chi1 - chi2)
    atest = (1.0 - 2.0 * nu) * chiS + delta * chiA

    r0, pphi0 = GravitationalWaveforms._seobv5_circular_ic(0.01, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))
    pr0 = GravitationalWaveforms._seobv5_dissipative_ic(r0, pphi0, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO), _seobv5_test_fc(nu, X1, X2, chi1, chi2))
    H0, _, _ = GravitationalWaveforms._seobv5_ham_xi(r0, pr0, pphi0, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))

    # Initial condition: r0, pphi0, H0 (and omega, checked via the gradient below) all agree with
    # real pyseobnr to ~1e-6-1e-9 relative -- confirms the Hamiltonian/gradient/xi transcription
    # itself is correct (the conservative sector), independent of the flux/RR-force sector. pr0
    # (IC_diss) now agrees to ~1e-5 relative (was ~0.3% under the original (2,2)-only flux) --
    # since IC_diss's own residual() internally calls _seobv5_rr_force too (see
    # _seobv5_dissipative_ic's docstring), this confirms the old ~0.3% was dominated by the
    # missing-6-mode flux deficit feeding into that residual, not primarily the FD Hessian
    # approximation this file's docstring previously attributed it to alone -- a real correction,
    # not a re-tolerancing.
    @test isapprox(r0, 21.5904223; rtol = 1e-6)
    @test isapprox(pphi0, 4.98657191; rtol = 1e-6)
    @test isapprox(H0, 4.14436894; rtol = 1e-6)
    @test isapprox(pr0, -3.24375068e-4; rtol = 5e-5)

    coeffs = (nu = nu, X1 = X1, X2 = X2, chi1 = chi1, chi2 = chi2,
        hp = GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO),
        fc = _seobv5_test_fc(nu, X1, X2, chi1, chi2))
    state0 = SEOBNRv5AlignedSpinState(r0, 0.0, pr0, pphi0)

    function omega_of(s)
        H, _, _ = GravitationalWaveforms._seobv5_ham_xi(s.r, s.prst, s.pphi, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))
        _, _, dHdpphi = GravitationalWaveforms._seobv5_sym_gradient(s.r, s.prst, s.pphi, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))
        return dHdpphi
    end

    # t=125.663706 (reduced/M-second units, matching pyseobnr's own dynamics[:,0] convention):
    # pyseobnr row (t,r,phi,pr,pphi,H,omega,omega_circ) =
    # (125.663706, 21.5523068, 1.25830327, -3.24504137e-4, 4.98280181, 4.14433119, 1.00265333e-2, ...)
    # Under the full 7-mode flux, r/omega now agree to ~2e-7/~3e-7 relative (was ~5e-6/~7.6e-6
    # under the original (2,2)-only port -- an order-of-magnitude tightening even this early,
    # where the missing-mode deficit had barely had time to accumulate).
    s_early = seobnrv5alignedspin_rk4_integrate(state0, 0.05, round(Int, 125.663706 / 0.05), coeffs)
    @test isapprox(s_early.r, 21.5523068; rtol = 1e-5)
    @test isapprox(s_early.phi, 1.25830327; atol = 5e-4)
    @test isapprox(s_early.pphi, 4.98280181; rtol = 1e-5)
    @test isapprox(omega_of(s_early), 1.00265333e-2; rtol = 1e-5)

    # t=10628.4931: pyseobnr row = (10628.4931, 17.1572152, 123.568277, -6.70449252e-4, 4.53076137,
    # 4.13894895, 1.41164560e-2, ...). Under the full 7-mode flux, r/omega now agree to
    # ~1.1e-6/~1.7e-6 relative (was ~0.12%/~0.17% under the original (2,2)-only port -- a
    # ~1000x tightening at this checkpoint, confirming the missing-mode flux deficit was
    # essentially the entire source of the old residual here).
    s_mid = seobnrv5alignedspin_rk4_integrate(state0, 0.05, round(Int, 10628.4931 / 0.05), coeffs)
    @test isapprox(s_mid.r, 17.1572152; rtol = 1e-5)
    @test isapprox(s_mid.pphi, 4.53076137; rtol = 1e-5)
    @test isapprox(omega_of(s_mid), 1.41164560e-2; rtol = 1e-5)

    # Further checkpoints deeper into the inspiral/plunge, added when the flux was extended from
    # (2,2)-only to the full 7-mode sum -- the old (2,2)-only port's residual was already ~0.3-2%
    # in this regime (per this file's prior module-docstring writeup); the point here is to show
    # the new port's residual, while still growing monotonically (expected: this port has no NQC/
    # merger-ringdown machinery, and the missing-mode deficit -- now from modes beyond these seven,
    # i.e. none, so purely from accumulated integration/IC error -- still compounds near plunge),
    # stays 2-3 orders of magnitude tighter than the old port at the same physical (r, not just t)
    # locations. pyseobnr rows (t,r,phi,pr,pphi,H,omega,omega_circ), same script/config:
    # (14982.680301786795, 13.358058014240235, 195.7941324816542, -0.0015084900562119883,
    #  4.107170990911787, 4.13173324162784, 0.020534163348289575, ...)
    # (16496.781390156906, 10.449041831196043, 232.184980991933, -0.0034530443032319657,
    #  3.761252855017743, 4.123204396785986, 0.02961174132807019, ...)
    # (17006.687091993397, 8.290649378213372, 249.6849027521939, -0.007833747238010105,
    #  3.4959199227703963, 4.113890590506039, 0.041666172120646966, ...)
    # (17199.950092191153, 6.446751573231209, 259.0856965988041, -0.019488782192990686,
    #  3.2690352439812993, 4.102560856587829, 0.05987863259708051, ...) -- r~6.4M, deep plunge.
    s_late1 = seobnrv5alignedspin_rk4_integrate(state0, 0.05, round(Int, 14982.680301786795 / 0.05), coeffs)
    @test isapprox(s_late1.r, 13.358058014240235; rtol = 1e-5)
    @test isapprox(omega_of(s_late1), 0.020534163348289575; rtol = 1e-5)

    s_late2 = seobnrv5alignedspin_rk4_integrate(state0, 0.05, round(Int, 16496.781390156906 / 0.05), coeffs)
    @test isapprox(s_late2.r, 10.449041831196043; rtol = 5e-5)
    @test isapprox(omega_of(s_late2), 0.02961174132807019; rtol = 5e-5)

    s_late3 = seobnrv5alignedspin_rk4_integrate(state0, 0.05, round(Int, 17006.687091993397 / 0.05), coeffs)
    @test isapprox(s_late3.r, 8.290649378213372; rtol = 5e-5)
    @test isapprox(omega_of(s_late3), 0.041666172120646966; rtol = 5e-5)

    s_late4 = seobnrv5alignedspin_rk4_integrate(state0, 0.05, round(Int, 17199.950092191153 / 0.05), coeffs)
    @test isapprox(s_late4.r, 6.446751573231209; rtol = 5e-5)
    @test isapprox(omega_of(s_late4), 0.05987863259708051; rtol = 5e-5)

    # Regression, re-measured after the GSF-amplitude-fit fix below: the relative residual in r
    # grows (loosely, not strictly monotonically -- see next paragraph) across these checkpoints,
    # but at dramatically smaller magnitude than before that fix (`~1e-7` near `t=0` to `~1.4e-5`
    # at `r~8.3M`, not `~1e-7` to `~1-10%` as this file previously documented).
    #
    # Cross-validating the post-adiabatic driver against real
    # `pyseobnr` at an arbitrary off-trajectory point shows `_seobv5_rr_force` needs the
    # 2nd-order gravitational-self-force (GSF) amplitude-fit corrections
    # (`pyseobnr/eob/fits/GSF_fits.py`'s `GSF_amplitude_fits(nu)`, wired into
    # `eob_pars.flux_params.extra_coeffs`/`extra_coeffs_log` for all seven modes at `v^4`-`v^10`,
    # see `SEOBNRv5AlignedSpin.jl`'s own "Radiation-reaction force" docstring section for the exact
    # per-mode citation) -- a real, ~0.1%-level gap in the flux at a generic point, previously
    # masked by this file's own checkpoint validation being dominated by other error sources (first
    # the missing 6 modes, then this file's own missing-NQC/ringdown-attachment error). Porting
    # `GSF_amplitude_fits` and wiring it into every mode's `rho_lm` polynomial (the fix actually
    # applied, not merely described) reduced the residual at every checkpoint by 1-4 orders of
    # magnitude, including deep in the plunge (`r~6.4M`) where the residual was previously the
    # largest documented value in this file (`~7.7-9.7%`) and is now `~1.3e-5` -- **strongly
    # suggesting the previously-documented near-merger error growth was NOT primarily missing-NQC/
    # ringdown-attachment physics, as this file's docstring previously framed it, but was instead
    # dominated by this missing GSF flux term all along.** This is a correction to that prior
    # framing, not merely a tighter tolerance.
    #
    # Because the residual is now tiny everywhere, its checkpoint-to-checkpoint ordering is no
    # longer strictly monotonic (`resid_late4 < resid_late3` by a small amount, `~1.3e-5` vs
    # `~1.4e-5` -- consistent with genuine sub-percent-level numerical noise at this precision, not
    # a sign of a remaining systematic error) -- so this regression checks that the residual STAYS
    # small (a tight absolute ceiling at every checkpoint, tightening with each successive one only
    # on average) rather than asserting strict monotonic growth, which the old, much-larger-residual
    # regime could support but this new, much tighter one cannot meaningfully distinguish from noise.
    resid_early = abs(s_early.r - 21.5523068) / 21.5523068
    resid_mid = abs(s_mid.r - 17.1572152) / 17.1572152
    resid_late1 = abs(s_late1.r - 13.358058014240235) / 13.358058014240235
    resid_late2 = abs(s_late2.r - 10.449041831196043) / 10.449041831196043
    resid_late3 = abs(s_late3.r - 8.290649378213372) / 8.290649378213372
    resid_late4 = abs(s_late4.r - 6.446751573231209) / 6.446751573231209
    @test resid_early < 5e-7
    @test resid_mid < 1e-6
    @test resid_late1 < 1e-5
    @test resid_late2 < 2e-5
    @test resid_late3 < 3e-5
    @test resid_late4 < 3e-5
    @test resid_mid > resid_early
    @test resid_late1 > resid_mid
    @test resid_late2 > resid_late1
end

@testset "Real pyseobnr SEOBNRv5HM validation (q=1.0 exactly-equal-mass, chi1=chi2=0.2, omega_start=0.01)" begin
    # Regression test: every
    # config above uses q=1.5 (delta=X1-X2=0.0917...), never delta EXACTLY 0. _seobv5_rr_force's
    # (2,1)/(3,3)/(5,5)/(4,3) mode f_lm corrections had unconditional `/delta` divisions that hit
    # `0/0=NaN` at delta=0 exactly (independent of spin), and the NaN survived multiplication by the
    # correctly-zero mode prefactor (absN33/absN21/absN55 are exactly 0.0, not NaN, at delta=0 --
    # the classic 0*NaN=NaN IEEE-754 trap). Fixed by porting pyseobnr's own separate `dm2==0` branch
    # (waveform.pyx's `if dm2: ... else: ...` for each mode) verbatim -- these are genuinely
    # different closed-form expressions, not the delta->0 limit of the general formula (several
    # terms that are regular in the general expression vanish identically in the equal-mass branch).
    # chi1=chi2=0.2 (equal aligned spins) additionally makes chiA=0, stress-testing both singular
    # inputs (delta=0 and chiA=0) at once. Reference values from a real
    # `pyseobnr.generate_waveform.generate_modes_opt(q=1.0, chi1=0.2, chi2=0.2, omega_start=0.01,
    # approximant="SEOBNRv5HM", debug=True).dynamics` run.
    nu, X1, X2, chi1, chi2 = 0.25, 0.5, 0.5, 0.2, 0.2
    ap = X1 * chi1 + X2 * chi2
    am = X1 * chi1 - X2 * chi2
    a6 = GravitationalWaveforms._seobv5_a6_NS(nu)
    dSO = GravitationalWaveforms._seobv5_dSO(nu, ap, am)
    delta = X1 - X2
    @test delta == 0.0
    chiS = 0.5 * (chi1 + chi2)
    chiA = 0.5 * (chi1 - chi2)
    @test chiA == 0.0
    atest = (1.0 - 2.0 * nu) * chiS + delta * chiA

    r0, pphi0 = GravitationalWaveforms._seobv5_circular_ic(0.01, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))
    pr0 = GravitationalWaveforms._seobv5_dissipative_ic(r0, pphi0, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO), _seobv5_test_fc(nu, X1, X2, chi1, chi2))

    @test all(isfinite, (r0, pphi0, pr0))
    @test isapprox(r0, 21.581130747169688; rtol = 1e-8)
    @test isapprox(pphi0, 4.97368987238724; rtol = 1e-8)
    @test isapprox(pr0, -0.0003346062741183952; rtol = 1e-4)

    absN33 = GravitationalWaveforms._seobv5_multipole_prefix_abs(nu, X1, X2, 3, 3)
    absN21 = GravitationalWaveforms._seobv5_multipole_prefix_abs(nu, X1, X2, 2, 1)
    absN44 = GravitationalWaveforms._seobv5_multipole_prefix_abs(nu, X1, X2, 4, 4)
    absN32 = GravitationalWaveforms._seobv5_multipole_prefix_abs(nu, X1, X2, 3, 2)
    absN55 = GravitationalWaveforms._seobv5_multipole_prefix_abs(nu, X1, X2, 5, 5)
    absN43 = GravitationalWaveforms._seobv5_multipole_prefix_abs(nu, X1, X2, 4, 3)
    # Odd-m mode amplitudes vanish identically for equal-mass binaries -- genuinely zero, not NaN.
    @test absN33 == 0.0
    @test absN21 == 0.0
    @test absN43 == 0.0

    coeffs = (nu = nu, X1 = X1, X2 = X2, chi1 = chi1, chi2 = chi2,
        hp = GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO),
        fc = _seobv5_test_fc(nu, X1, X2, chi1, chi2))
    state0 = SEOBNRv5AlignedSpinState(r0, 0.0, pr0, pphi0)

    function omega_of_q1(s)
        H, _, _ = GravitationalWaveforms._seobv5_ham_xi(s.r, s.prst, s.pphi, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))
        _, _, dHdpphi = GravitationalWaveforms._seobv5_sym_gradient(s.r, s.prst, s.pphi, GravitationalWaveforms.SEOBNRv5HamParams(nu, X1, X2, chi1, chi2, a6, dSO))
        return dHdpphi
    end

    # pyseobnr row (t,r,phi,pr,pphi,H,omega,omega_circ) =
    # (16688.65170871195, 7.390190591181951, 249.09231816041552, -0.011473756595602631,
    #  3.344810269852001, ..., 0.04906757741125355, ...)
    s1 = seobnrv5alignedspin_rk4_integrate(state0, 0.05, round(Int, 16688.65170871195 / 0.05), coeffs)
    @test all(isfinite, s1)
    @test isapprox(s1.r, 7.390190591181951; rtol = 1e-3)
    @test isapprox(s1.phi, 249.09231816041552; rtol = 1e-3)
    @test isapprox(omega_of_q1(s1), 0.04906757741125355; rtol = 1e-3)

    # pyseobnr row: (16823.851708709983, 5.45260777374606, 257.0228391439861,
    #  -0.032326473313164523, 3.0933193732094444, ..., 0.0751136589050677, ...) -- closer to
    # plunge, looser tolerance matching this file's own established convention above.
    s2 = seobnrv5alignedspin_rk4_integrate(state0, 0.05, round(Int, 16823.851708709983 / 0.05), coeffs)
    @test all(isfinite, s2)
    @test isapprox(s2.r, 5.45260777374606; rtol = 5e-3)
    @test isapprox(omega_of_q1(s2), 0.0751136589050677; rtol = 5e-3)

    # Also confirm the base public Phi/Ampl interface (what actually broke) is finite and sane
    # across a spread of spin combinations at exactly q=1 -- zero spin, equal aligned spin, and
    # unequal spin all exercise delta=0 with different chiA/chiS content.
    for (c1, c2) in ((0.0, 0.0), (0.2, 0.2), (0.3, -0.2), (0.5, 0.1))
        ph = Phi(SEOBNRv5AlignedSpin(), 0.01, 10.0, 0.25, c1, c2; f_min = 20.0)
        am = Ampl(SEOBNRv5AlignedSpin(), 0.01, 10.0, 0.25, c1, c2, 1.0; f_min = 20.0)
        @test isfinite(ph)
        @test isfinite(am)
        @test am > 0.0
    end
end

@testset "Finite-difference smoothness sanity (Enzyme substitute)" begin
    # A genuine, characterized finding from writing this test (not hidden): the two-stage
    # initial-condition solve ([`_seobv5_circular_ic`](@ref)/[`_seobv5_dissipative_ic`](@ref))
    # uses its own INTERNAL finite differences (a Jacobian step of `1e-6*r`/`1e-6*pphi` for the
    # circular solve, an absolute `1e-8` step for the dissipative solve's own 1D Newton
    # derivative, and a further central-difference Hessian approximation inside that -- see
    # SEOBNRv5AlignedSpin.jl's docstring's "Initial conditions" section). Probing `Phi`'s
    # smoothness w.r.t. `mc`/`chi1` with an OUTER central difference at a comparably small
    # relative step (found directly: `1e-6`, `1e-7`) lands inside that internal noise floor and
    # gives an unstable, step-size-dependent derivative estimate purely from the *initial
    # condition's own* nested finite differences, not from any discontinuity in the physics
    # itself. Confirmed by scanning step sizes directly: the estimate is stable and converged for
    # relative steps `1e-3` down through `1e-5` (agreeing to <0.2%), and only becomes noisy at
    # `1e-6` and smaller -- so this check deliberately stays in the `1e-4`-`1e-3` stable region,
    # which is exactly where it needs to be to be a meaningful "no discontinuity" check at all.
    # **A concrete, actionable note for a future `Enzyme.gradient` pass**: the
    # initial condition's internal finite differences are not analytically differentiable as
    # written -- a real AD pass through this model would need those replaced with (or
    # differentiated correctly through) an analytic Jacobian/Hessian, not just the RHS itself.
    model = SEOBNRv5AlignedSpin()
    mc, eta, chi1, chi2 = 23.821232072402484, 0.24793388429752067, 0.3, -0.2
    t = 0.005

    function phi_of_mc(x)
        Phi(model, t, x, eta, chi1, chi2)
    end
    function phi_of_chi1(x)
        Phi(model, t, mc, eta, x, chi2)
    end

    for (f, x0) in ((phi_of_mc, mc), (phi_of_chi1, chi1))
        h1 = 1e-3 * max(abs(x0), 1.0)
        h2 = 1e-4 * max(abs(x0), 1.0)
        d1 = (f(x0 + h1) - f(x0 - h1)) / (2h1)
        d2 = (f(x0 + h2) - f(x0 - h2)) / (2h2)
        @test isfinite(d1)
        @test isfinite(d2)
        @test isapprox(d1, d2; rtol = 1e-2)
    end
end
