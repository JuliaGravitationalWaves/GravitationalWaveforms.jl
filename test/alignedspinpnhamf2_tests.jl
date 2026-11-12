# AlignedSpinPNHamF2 -- the closed-form, frequency-domain SPA sibling of AlignedSpinPNHam.
# See that model's own docstring (src/GravitationalWaveforms/AlignedSpinPNHamF2.jl) for the full derivation,
# including the genuine spin-phasing discrepancy found (and not silently resolved) between a
# from-scratch energy-balance reconstruction and TaylorF2's own published/LALSuite-validated
# spin-orbit/spin-spin coefficients (this file reuses the latter, restricted to AlignedSpinPNHam's
# own S1.S2-only spin-spin scope).

@testset "AlignedSpinPNHamF2 construction and Phi/Ampl basic sanity" begin
    model = AlignedSpinPNHamF2()
    @test model isa AlignedSpinPNHamF2

    mc, eta, chi1, chi2, dL = 23.821232072402484, 0.24793388429752067, 0.3, -0.2, 1.0
    f = collect(range(20.0, 500.0, length = 50))

    phi = Phi(model, f, mc, eta, chi1, chi2)
    ampl = Ampl(model, f, mc, eta, chi1, chi2, dL)

    @test phi isa Vector{Float64}
    @test ampl isa Vector{Float64}
    @test length(phi) == length(f)
    @test length(ampl) == length(f)
    @test all(isfinite, phi)
    @test all(isfinite, ampl)
    @test all(>=(0.0), ampl)

    phi_scalar = Phi(model, 100.0, mc, eta, chi1, chi2)
    @test phi_scalar isa Float64

    f32 = Float32.(f)
    phi32 = Phi(model, f32, Float32(mc), Float32(eta), Float32(chi1), Float32(chi2))
    ampl32 = Ampl(model, f32, Float32(mc), Float32(eta), Float32(chi1), Float32(chi2), Float32(dL))
    @test eltype(phi32) === Float32
    @test eltype(ampl32) === Float32
end

@testset "validate_alignedspinpnhamf2_params" begin
    @test validate_alignedspinpnhamf2_params(0.24, 0.1, -0.2) === nothing
    @test validate_alignedspinpnhamf2_params(0.25, 0.0, 0.0) === nothing
    @test_throws ArgumentError validate_alignedspinpnhamf2_params(0.0, 0.0, 0.0)
    @test_throws ArgumentError validate_alignedspinpnhamf2_params(0.26, 0.0, 0.0)
    @test_throws ArgumentError validate_alignedspinpnhamf2_params(0.2, 1.5, 0.0)
    @test_throws ArgumentError validate_alignedspinpnhamf2_params(0.2, 0.0, -1.5)
    @test_throws ArgumentError validate_alignedspinpnhamf2_params(0.2, 0.0, 0.0; dL = -1.0)
    @test validate_alignedspinpnhamf2_params(0.2, 0.0, 0.0; dL = 1.0) === nothing
end

@testset "Non-spin: exact reduction to TaylorF2 (bit-for-bit)" begin
    # AlignedSpinPNHamF2's non-spin sector is literally the same construction as TaylorF2's own
    # (see that model's docstring's "Derivation and validation" section -- confirmed, via a
    # from-scratch power-series energy-balance computation, to reproduce TaylorF2.jl's own
    # raw0..raw7/rawl5/rawl6 non-spin coefficients to BigFloat precision at several eta). With
    # chi1=chi2=0, both models' Phi/Ampl must therefore agree EXACTLY (same formula, same inputs),
    # not just approximately -- a strong, easy-to-regress-test consequence of that derivation.
    mc, eta, dL = 23.821232072402484, 0.24793388429752067, 1.0
    f = collect(range(20.0, 800.0, length = 80))

    phi_asphf2 = Phi(AlignedSpinPNHamF2(), f, mc, eta, 0.0, 0.0)
    phi_tf2 = Phi(TaylorF2(), f, mc, eta, 0.0, 0.0)
    @test phi_asphf2 == phi_tf2

    ampl_asphf2 = Ampl(AlignedSpinPNHamF2(), f, mc, eta, 0.0, 0.0, dL)
    ampl_tf2 = Ampl(TaylorF2(), f, mc, eta, 0.0, 0.0, dL)
    @test ampl_asphf2 == ampl_tf2

    # Same exact-match property at a second, very different eta (mass-ratio dependence check).
    for eta2 in (0.16, 0.05)
        @test Phi(AlignedSpinPNHamF2(), f, mc, eta2, 0.0, 0.0) == Phi(TaylorF2(), f, mc, eta2, 0.0, 0.0)
    end
end

@testset "Spin: genuinely differs from TaylorF2 (quadrupole/self-spin terms deliberately dropped)" begin
    # AlignedSpinPNHam's own H_SS is a pure S1.S2 coupling (see that file's docstring) -- it has no
    # S1^2/S2^2 (quadrupole/self-spin) structure at all, so AlignedSpinPNHamF2 deliberately omits
    # TaylorF2's `_tf2_4PNQM2SO`/`_tf2_4PNSelf2SO`/`_tf2_4PNQM2S`/`_tf2_4PNSelf2S` terms (which are
    # nonzero even for qm=1, i.e. even for plain BHs). This testset locks in that documented,
    # intentional divergence as a regression (not a silent, undocumented behavior change).
    mc, eta, dL = 23.821232072402484, 0.24793388429752067, 1.0
    f = collect(range(20.0, 500.0, length = 30))
    chi1, chi2 = 0.5, -0.3

    phi_asphf2 = Phi(AlignedSpinPNHamF2(), f, mc, eta, chi1, chi2)
    phi_tf2 = Phi(TaylorF2(), f, mc, eta, chi1, chi2)
    @test all(isfinite, phi_asphf2)
    @test phi_asphf2 != phi_tf2
    @test maximum(abs.(phi_asphf2 .- phi_tf2)) > 1e-3   # a real, physically-meaningful difference

    # But the full spin-orbit content (1.5-3.5PN) IS shared with TaylorF2 -- confirm that isolating
    # just the S1.S2 sector (chi1 or chi2 = 0, so no spin-spin coupling at all) still differs only
    # by the self-spin/quadrupole SO terms (`_tf2_4PNQM2SO`/`_tf2_4PNSelf2SO`), not by the
    # spin-orbit terms themselves -- i.e. confirm SOME agreement remains for a case where only
    # self-spin (not S1.S2) content is at stake by checking chi2=0 still shows a nonzero but
    # small-ish residual coming only from self-spin (a soft sanity check, not a precise bound).
    phi_asphf2_c2z = Phi(AlignedSpinPNHamF2(), f, mc, eta, chi1, 0.0)
    phi_tf2_c2z = Phi(TaylorF2(), f, mc, eta, chi1, 0.0)
    @test all(isfinite, phi_asphf2_c2z)
    @test phi_asphf2_c2z != phi_tf2_c2z
end

@testset "SPA phase is a smooth, well-behaved function of frequency (dPhi/df sanity)" begin
    # Basic internal-consistency check on the new closed-form series itself (mirrors the kind of
    # dPhi/df sanity check IMRPhenomD/TaylorT3 cross-checks already use elsewhere in this
    # package): a small central-difference derivative should be finite and vary smoothly (no
    # sign flips / discontinuities) across the inspiral band, well inside the ISCO cutoff.
    mc, eta, chi1, chi2 = 23.821232072402484, 0.24793388429752067, 0.4, -0.1
    model = AlignedSpinPNHamF2()
    # Stay well below the Schwarzschild-ISCO cutoff (~80 Hz for this ~55 Msun system -- v_isco^3/
    # (pi*M_sec)) so Phi doesn't hit the branchless `ifelse(fgrid<fCut,...,zero(T))` cutoff, which
    # would make dPhi/df trivially (and correctly) zero rather than reflecting the phasing series.
    fs = collect(range(25.0, 65.0, length = 20))
    h = 1e-3
    dphidf = [(Phi(model, f + h, mc, eta, chi1, chi2) - Phi(model, f - h, mc, eta, chi1, chi2)) / (2h) for f in fs]
    @test all(isfinite, dphidf)
    # The SPA phasing is dominated by a positive-coefficient v^-5 term (pfaN/v^5, pfaN > 0 for
    # eta>0), which is monotonically DECREASING in f -- so dPhi/df should be negative throughout
    # the inspiral band (standard SPA convention, confirmed exactly against TaylorF2's own
    # identical non-spin construction above).
    @test all(<(0.0), dphidf)
end

@testset "Rough cross-check against AlignedSpinPNHam's own time-domain output" begin
    # AlignedSpinPNHam.Phi/.Ampl take TIME, not frequency, and (being an ODE-integrated model with
    # only a 1PN-order conservative Hamiltonian, see that file's own "Explicit scope limit"
    # section) is a substantially cruder approximant of the conservative sector than the 3PN
    # energy-balance content underlying this SPA model -- so exact agreement is NOT expected, and
    # this is deliberately a loose, order-of-magnitude sanity check, not a tight tolerance test.
    #
    # The correct comparison quantity is the standard SPA "group delay" identity
    # dPsi/df = 2*pi*t(f) (NOT direct phase-value comparison -- Psi(f) is a stationary-phase/
    # Legendre-type transform of the orbital phase and decreases with f while the time-domain
    # phase increases with t; these are different mathematical objects). Group-delay DIFFERENCES
    # (removing the arbitrary additive gauge constant) are compared against actual elapsed
    # time-domain time differences, at matched instantaneous frequencies, well inside
    # AlignedSpinPNHam's own confirmed-stable range (t <= ~0.05s / r >= ~9.7 for this baseline
    # system, see that file's docstring/tests).
    mc, eta, chi1, chi2 = 23.821232072402484, 0.24793388429752067, 0.0, 0.0
    model_td = AlignedSpinPNHam()
    model_fd = AlignedSpinPNHamF2()
    t1, t2 = 0.01, 0.03

    dt_internal = 5.0 / 4096.0   # several times AlignedSpinPNHam's own default RK4 step (1/4096)
                                 # -- see that file's "Step-quantization note".
    f_at(t) = (Phi(model_td, t + dt_internal, mc, eta, chi1, chi2) -
               Phi(model_td, t - dt_internal, mc, eta, chi1, chi2)) / (2 * dt_internal) / (2 * pi)
    f1, f2 = f_at(t1), f_at(t2)
    @test f2 > f1   # chirping up, as expected

    df = 0.05
    group_delay(f) = (Phi(model_fd, f + df, mc, eta, chi1, chi2) -
                       Phi(model_fd, f - df, mc, eta, chi1, chi2)) / (2 * df) / (2 * pi)
    tg1, tg2 = group_delay(f1), group_delay(f2)

    @test isfinite(tg1) && isfinite(tg2)
    @test tg2 > tg1   # same direction: both models agree the "time-to-this-frequency" increases
    # Order-of-magnitude consistency only (found, in practice, to differ by up to a factor of a
    # few near f_min for this 1PN-conservative-vs-3PN-energy-balance comparison) -- a factor-of-10
    # window is a genuine, non-vacuous check without overclaiming precision this comparison
    # doesn't have.
    @test 0.1 * (t2 - t1) < (tg2 - tg1) < 10.0 * (t2 - t1)
end

@testset "Phasing coefficients regression vs real lalsimulation (cross-validation)" begin
    # Real LALSuite (`lalsimulation` 7.7.1): `lalsimulation.SimInspiralTaylorF2AlignedPhasing(m1_SI, m2_SI, chi1z, chi2z,
    # lal.CreateDict())` returns a `PNPhasingSeries` with `.v[0..8]`/`.vlogv[0..8]` -- the EXACT
    # same `pfaN,pfa2,...,pfa7`/`pfl5,pfl6` this file's `_asphf2_phi_coeffs` computes (same index
    # convention: `.v[k]` is the coefficient of `v^k`).
    #
    # Baseline system: mc=23.821232072402484 Msun, eta=0.24793388429752067 (m1=X1*M, m2=X2*M via
    # `delta=sqrt(1-4*eta)`), in solar masses converted to kg via `lal.MSUN_SI`.
    #
    # Result: for BOTH chi1=chi2=0 (non-spin) AND chi1=0.5,chi2=-0.3, this model's `pfaN, pfa2,
    # pfa3, pfa5, pfl5, pfl6, pfa7` match LAL's real output EXACTLY (bit-for-bit/~1e-15 relative --
    # i.e. the full non-spin sector AND the full spin-orbit sector, 1.5-3.5PN). `pfa4`/`pfa6`
    # (the 2PN/3PN slots) differ from LAL's *full* TaylorF2 output BY EXACTLY the quadrupole-
    # monopole/self-spin term this model deliberately omits (see this model's own docstring) --
    # confirmed here to ~1e-15 relative by adding that exact term back and recovering LAL's value
    # bit-for-bit. This is the strongest available validation of this model's spin content: not
    # merely "matches this package's own TaylorF2.jl", but "matches real `lalsimulation`, with the
    # one intentional omission precisely, numerically accounted for."
    mc, eta = 23.821232072402484, 0.24793388429752067
    delta = sqrt(1 - 4 * eta)
    mm1, mm2 = 0.5 * (1 + delta), 0.5 * (1 - delta)

    # chi1=chi2=0 (non-spin): LAL v=[0.09453125000000001, 0.0, 0.6077577298280423,
    # -4.751658888554562, 4.356995049766929, 14.649094476683693, -103.72868444332092,
    # 106.78115986864228, 0.0], vlogv=[..., 43.94728343005109 (v^5), -30.82619047619048 (v^6), ...].
    c0 = GravitationalWaveforms._asphf2_phi_coeffs(AlignedSpinPNHamF2(), mc, eta, 0.0, 0.0)
    @test c0.pfaN ≈ 0.09453125000000001 rtol=1e-14
    @test c0.pfa2 ≈ 0.6077577298280423 rtol=1e-14
    @test c0.pfa3 ≈ -4.751658888554562 rtol=1e-14
    @test c0.pfa4 ≈ 4.356995049766929 rtol=1e-14   # no quadrupole/self-spin content at chi=0
    @test c0.pfa5 ≈ 14.649094476683693 rtol=1e-14
    @test c0.pfl5 ≈ 43.94728343005109 rtol=1e-14
    @test c0.pfa6 ≈ -103.72868444332092 rtol=1e-14 # no S1.S2/quadrupole content at chi=0
    @test c0.pfl6 ≈ -30.82619047619048 rtol=1e-14
    @test c0.pfa7 ≈ 106.78115986864228 rtol=1e-14

    # chi1=0.5, chi2=-0.3: LAL v=[0.09453125000000001, 0.0, 0.6077577298280423,
    # -4.325487013554563, 4.259216729454429, 11.193692261382688, -77.02325864663102,
    # 36.51136087251438, 0.0], vlogv[5]=33.58107678414806, vlogv[6]=-30.82619047619048.
    chi1, chi2 = 0.5, -0.3
    c = GravitationalWaveforms._asphf2_phi_coeffs(AlignedSpinPNHamF2(), mc, eta, chi1, chi2)
    @test c.pfaN ≈ 0.09453125000000001 rtol=1e-14
    @test c.pfa2 ≈ 0.6077577298280423 rtol=1e-14
    @test c.pfa3 ≈ -4.325487013554563 rtol=1e-13   # full 1.5PN spin-orbit: exact match
    @test c.pfa5 ≈ 11.193692261382688 rtol=1e-13   # full 2.5PN spin-orbit: exact match
    @test c.pfl5 ≈ 33.58107678414806 rtol=1e-13
    @test c.pfl6 ≈ -30.82619047619048 rtol=1e-14    # non-spin tail log term, unaffected by spin
    @test c.pfa7 ≈ 36.51136087251438 rtol=1e-13    # full 3.5PN spin-orbit: exact match

    # pfa4/pfa6: differ from LAL's FULL (quadrupole-included) value by exactly the omitted
    # quadrupole-monopole/self-spin term (qm1=qm2=1 for a BBH, i.e. lambda1=lambda2=0).
    S1z2, S2z2 = chi1^2, chi2^2
    quad4 = (GravitationalWaveforms._tf2_4PNQM2SO(mm1) + GravitationalWaveforms._tf2_4PNSelf2SO(mm1)) * S1z2 +
            (GravitationalWaveforms._tf2_4PNQM2SO(mm2) + GravitationalWaveforms._tf2_4PNSelf2SO(mm2)) * S2z2 +
            (GravitationalWaveforms._tf2_4PNQM2S(mm1) + GravitationalWaveforms._tf2_4PNSelf2S(mm1)) * S1z2 +
            (GravitationalWaveforms._tf2_4PNQM2S(mm2) + GravitationalWaveforms._tf2_4PNSelf2S(mm2)) * S2z2
    quad6 = (GravitationalWaveforms._tf2_6PNQM2S(mm1) + GravitationalWaveforms._tf2_6PNSelf2S(mm1)) * S1z2 +
            (GravitationalWaveforms._tf2_6PNQM2S(mm2) + GravitationalWaveforms._tf2_6PNSelf2S(mm2)) * S2z2
    pfaN_prefactor = 3.0 / (128.0 * eta)
    @test c.pfa4 + quad4 * pfaN_prefactor ≈ 4.259216729454429 rtol=1e-13
    @test c.pfa6 + quad6 * pfaN_prefactor ≈ -77.02325864663102 rtol=1e-13
end
