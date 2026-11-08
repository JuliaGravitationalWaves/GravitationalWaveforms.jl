# Enzyme regression test for the two Enzyme reverse-mode NaN singularities guarded in
# `_xp_precession_setup` (`src/GravitationalWaveforms/IMRPhenomXPrecession.jl`):
#
# 1. `phiJ = atan(J0y, J0x)` -- `atan2`'s own `0/0` derivative at its
#    origin, live-reachable because GravitationalWaveBayes.jl's NUTS walker initialization starts every
#    IMRPhenomXP/IMRPhenomXPHM walker at exactly zero in-plane spin.
# 2. `thetaJ = acos(J0z/J0safe)` -- found empirically THIS session while validating fix (1): the
#    SAME aligned-spin configuration drives `acos`'s argument to exactly `+-1`, its own derivative
#    pole, and this poisons the Enzyme reverse-mode gradient of the ENTIRE `_xp_precession_setup`
#    function (every field, not just `thetaJ`'s own read-out) -- confirmed empirically that fix (1)
#    alone was necessary but NOT sufficient; both are required together.
#
# Both fixes are architecturally different: `phiJ`'s fix substitutes a fixed nonzero point
# `(1e-6, 0)` for `(J0x, J0y)` below a tiny radius threshold (safe because `atan(0,1e-6)` happens to
# equal `atan(0,0)`, so the forward value is unperturbed even at the pole). `thetaJ`'s fix instead
# replaces `acos(z/r)` with the mathematically equivalent, pole-free `atan2(sqrt(r^2-z^2), z)` --
# an epsilon-clamp on `acos`'s argument was tried first and reverted, since any margin large enough
# to keep the derivative finite-but-not-huge measurably perturbed `thetaJ`'s own value at exact
# alignment, breaking `test_twisting_up.jl`'s exact (`rtol=1e-10`) aligned-spin-limit closed-form
# comparison.
using Enzyme

@testset "_xp_precession_setup: Enzyme.gradient finite at the aligned-spin (zero in-plane spin) singularity" begin
    m1, m2 = 30.0, 20.0
    chi1z, chi2z = 0.3, -0.2
    fref, incl, phiRef = 20.0, 0.4, 0.1

    function phiJ_of(chi1x, chi1y, chi2x, chi2y)
        setup = GravitationalWaveforms._xp_precession_setup(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, fref, incl, phiRef)
        return setup.phiJ
    end
    function thetaJ_of(chi1x, chi1y, chi2x, chi2y)
        setup = GravitationalWaveforms._xp_precession_setup(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, fref, incl, phiRef)
        return setup.thetaJ
    end

    @testset "at the exact singular point (all in-plane spins zero)" begin
        @test phiJ_of(0.0, 0.0, 0.0, 0.0) == 0.0
        @test thetaJ_of(0.0, 0.0, 0.0, 0.0) == 0.0   # exact -- confirms no forward-value perturbation

        # Enzyme.gradient(Reverse, f, a, b, c, d) returns the flat tuple (df/da, df/db, df/dc, df/dd)
        # directly -- NOT wrapped in another level -- confirmed empirically here (a wrong
        # assumption about an extra wrapping layer caused a spurious BoundsError the first time
        # this test was written).
        g_phiJ = Enzyme.gradient(Enzyme.Reverse, phiJ_of, 0.0, 0.0, 0.0, 0.0)
        g_thetaJ = Enzyme.gradient(Enzyme.Reverse, thetaJ_of, 0.0, 0.0, 0.0, 0.0)
        @test all(isfinite, g_phiJ)
        @test all(isfinite, g_thetaJ)
    end

    @testset "away from the singularity: matches finite differences" begin
        chi1x0, chi1y0, chi2x0, chi2y0 = 0.2, 0.1, 0.05, -0.15
        h = 1.0e-6

        g_phiJ = Enzyme.gradient(Enzyme.Reverse, phiJ_of, chi1x0, chi1y0, chi2x0, chi2y0)
        fd_phiJ_x = (phiJ_of(chi1x0 + h, chi1y0, chi2x0, chi2y0) - phiJ_of(chi1x0 - h, chi1y0, chi2x0, chi2y0)) / (2h)
        fd_phiJ_y = (phiJ_of(chi1x0, chi1y0 + h, chi2x0, chi2y0) - phiJ_of(chi1x0, chi1y0 - h, chi2x0, chi2y0)) / (2h)
        @test isapprox(g_phiJ[1], fd_phiJ_x; rtol = 1e-4)
        @test isapprox(g_phiJ[2], fd_phiJ_y; rtol = 1e-4)

        g_thetaJ = Enzyme.gradient(Enzyme.Reverse, thetaJ_of, chi1x0, chi1y0, chi2x0, chi2y0)
        fd_thetaJ_x = (thetaJ_of(chi1x0 + h, chi1y0, chi2x0, chi2y0) - thetaJ_of(chi1x0 - h, chi1y0, chi2x0, chi2y0)) / (2h)
        fd_thetaJ_y = (thetaJ_of(chi1x0, chi1y0 + h, chi2x0, chi2y0) - thetaJ_of(chi1x0, chi1y0 - h, chi2x0, chi2y0)) / (2h)
        @test isapprox(g_thetaJ[1], fd_thetaJ_x; rtol = 1e-4)
        @test isapprox(g_thetaJ[2], fd_thetaJ_y; rtol = 1e-4)
    end
end

# MSA (PrecVersion 223) at equal mass: the S^2 cubic has a double root (|S| conserved) at q = 1, and
# numerically within ~|chi_min| of it, where acos' = Inf; psi1 ~ 1/delta^2 and psi2 ~ 1/(1-q)^4 are +-Inf at q = 1 inside
# switched-off branches; LAL's S1L_pav + S2L_pav is rounding noise / 1e-16 at q = 1. Before the safe-input fixes
# ~75% of random configurations had NaN gradients at 1 - q = 1e-8 and all at q = 1 (where NUTS's init clamp puts
# walkers with q > 1); the q = 1 value also jumped ~0.1 from the q -> 1 limit.
@testset "_xp_precession_setup(223): finite, continuous gradients at q -> 1" begin
    function msa_scal(m1, m2, c1x, c1y, c1z, c2x, c2y, c2z)
        s = GravitationalWaveforms._xp_precession_setup(m1, m2, c1x, c1y, c1z, c2x, c2y, c2z, 20.0, 0.6, 1.1; prec_version = Val(223))
        acc = s.final_spin + s.thetaJN + s.alpha0 + s.epsilon0 + s.zeta_polarization
        for v in (0.25, 0.3, 0.38, 0.45)
            acc += sum(GravitationalWaveforms._xp_euler_angles(v, s))
        end
        return acc
    end
    at_q(q) = (65.0 / (1 + q), 65.0 - 65.0 / (1 + q))
    spins = ((0.3, -0.2, 0.1, -0.4, 0.3, -0.2),           # generic
             (0.8226, -0.1202, -0.5, -0.0015, 0.0009, -0.0001),  # |chi2| ~ 2e-3: NaN from 1 - q = 1e-5 before
             (0.4, 0.1, -0.2, 1e-6, -1e-6, 0.0),          # |chi2| ~ 1e-6: NaN from 1 - q = 1e-2 before
             (0.0, 0.0, 0.4, 0.0, 0.0, -0.3))             # aligned: reverse NaN at q = 0.8 before
    for sp in spins
        g99 = Enzyme.gradient(Enzyme.Reverse, msa_scal, at_q(0.99)..., sp...)
        for q in (0.8, 1 - 1e-5, 1 - 1e-8, 1 - 1e-12, 1.0)
            gr = Enzyme.gradient(Enzyme.Reverse, msa_scal, at_q(q)..., sp...)
            gf = Enzyme.gradient(Enzyme.Forward, msa_scal, at_q(q)..., sp...)
            @test all(isfinite, gr)
            @test all(isfinite, gf)
            @test collect(gr) ≈ collect(gf) rtol = 1e-8 atol = 1e-6   # aligned: abs/safe-point subgradients differ ~1e-7
            q > 0.9 && @test sqrt(sum(abs2, gr)) < 10 * sqrt(sum(abs2, g99))
        end
        @test msa_scal(at_q(1.0)..., sp...) ≈ msa_scal(at_q(1 - 1e-8)..., sp...) atol = 1e-5
    end
end
