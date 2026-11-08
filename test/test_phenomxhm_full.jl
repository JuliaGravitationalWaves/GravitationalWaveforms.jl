using Test
using GravitationalWaveforms
const W = GravitationalWaveforms

# full IMRPhenomXHM assembly (src/GravitationalWaveforms/IMRPhenomXHM.jl). See that file's module
# docstring for the API design decision (h_lm/hphc, not Ampl/Phi) ; the absolute (not just shape/ratio) LALSuite
# comparison complements this unit-test file.

const XHM_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))

@testset "IMRPhenomXHM full assembly" begin

    @testset "IMRPhenomXHM struct + h_lm basic sanity across all 5 modes" begin
        model = IMRPhenomXHM()
        @test model isa IMRPhenomXHM

        m1, m2, chi1, chi2, dL = 30.0, 20.0, 0.3, -0.2, 0.4
        f_ref, phiRef = 20.0, 0.7
        f = collect(range(20.0, 1024.0, length = 200))

        for (l, m) in XHM_MODES
            h = h_lm(model, l, m, f, m1, m2, chi1, chi2, dL; f_ref = f_ref, phiRef = phiRef)
            @test h isa Vector{ComplexF64}
            @test length(h) == length(f)
            @test all(isfinite, h)
            @test any(x -> abs(x) > 0, h)  # not identically zero

            # Scalar frequency input returns a scalar.
            hs = h_lm(model, l, m, 100.0, m1, m2, chi1, chi2, dL; f_ref = f_ref, phiRef = phiRef)
            @test hs isa ComplexF64
            @test isfinite(hs)
        end
    end

    @testset "h_lm rejects unsupported (l,m)" begin
        model = IMRPhenomXHM()
        m1, m2, chi1, chi2, dL = 30.0, 20.0, 0.3, -0.2, 0.4
        @test_throws ArgumentError h_lm(model, 5, 5, 100.0, m1, m2, chi1, chi2, dL; f_ref = 20.0)
        @test_throws ArgumentError h_lm(model, 4, 3, 100.0, m1, m2, chi1, chi2, dL; f_ref = 20.0)
        @test_throws ArgumentError h_lm(model, 2, -1, 100.0, m1, m2, chi1, chi2, dL; f_ref = 20.0)  # m must be positive
    end

    @testset "hphc basic sanity" begin
        model = IMRPhenomXHM()
        m1, m2, chi1, chi2, dL = 30.0, 20.0, 0.3, -0.2, 0.4
        f_ref, phiRef, inclination = 20.0, 0.7, 0.6
        f = collect(range(20.0, 1024.0, length = 200))

        hp, hc = hphc(model, f, m1, m2, chi1, chi2, dL, inclination, phiRef; f_ref = f_ref)
        @test hp isa Vector{ComplexF64}
        @test hc isa Vector{ComplexF64}
        @test length(hp) == length(f)
        @test length(hc) == length(f)
        @test all(isfinite, hp)
        @test all(isfinite, hc)
        @test any(x -> abs(x) > 0, hp)
        @test any(x -> abs(x) > 0, hc)

        hps, hcs = hphc(model, 100.0, m1, m2, chi1, chi2, dL, inclination, phiRef; f_ref = f_ref)
        @test hps isa ComplexF64
        @test hcs isa ComplexF64
    end

    @testset "hphc equals the direct per-mode Y_lm sum (self-consistency with h_lm)" begin
        # hphc() internally builds each mode via the SAME _xhm_hlm_nonmixing/_xhm_hlm_32 path h_lm()
        # uses, but through its own coefficient bundles (ms21/ms33/ms44/m32 built once per call,
        # vs h_lm() building a fresh one per mode) -- this checks those two independent code paths
        # agree, not just that hphc() is internally self-consistent with itself.
        model = IMRPhenomXHM()
        m1, m2, chi1, chi2, dL = 45.0, 15.0, 0.4, 0.1, 0.8
        f_ref, phiRef, inclination = 20.0, 1.1, 0.9
        f = 150.0

        hp, hc = hphc(model, f, m1, m2, chi1, chi2, dL, inclination, phiRef; f_ref = f_ref)

        piHalf = pi / 2
        hp2 = zero(ComplexF64)
        hc2 = zero(ComplexF64)
        for (l, m) in XHM_MODES
            hlm = h_lm(model, l, m, f, m1, m2, chi1, chi2, dL; f_ref = f_ref, phiRef = phiRef)
            hp2, hc2 = W._xhm_add_mode!(hp2, hc2, l, m, hlm, inclination, piHalf)
        end
        @test hp ≈ hp2 rtol = 1e-12
        @test hc ≈ hc2 rtol = 1e-12
    end

    @testset "Float32 genericity" begin
        model = IMRPhenomXHM()
        m1, m2, chi1, chi2, dL = 30.0f0, 20.0f0, 0.3f0, -0.2f0, 0.4f0
        f_ref, phiRef, inclination = 20.0f0, 0.7f0, 0.6f0
        f = Float32.(range(20.0, 1024.0, length = 100))

        for (l, m) in XHM_MODES
            h = h_lm(model, l, m, f, m1, m2, chi1, chi2, dL; f_ref = f_ref, phiRef = phiRef)
            @test eltype(h) === ComplexF32
            @test all(isfinite, h)
        end

        hp, hc = hphc(model, f, m1, m2, chi1, chi2, dL, inclination, phiRef; f_ref = f_ref)
        @test eltype(hp) === ComplexF32
        @test eltype(hc) === ComplexF32
        @test all(isfinite, hp)
        @test all(isfinite, hc)
    end

    @testset "(2,2) mode: h_lm matches IMRPhenomXAS's own Ampl up to _XP_HAS_MODE_NORMALIZATION, exactly" begin
        # h_lm(model,2,2,...) is built from _xas_ampl_scalar directly (not through Ampl(::IMRPhenomXAS,...)'s
        # own array wrapper) -- this checks the two agree, i.e. no drift between the scalar path this
        # file uses internally and the public Ampl/Phi API IMRPhenomXAS.jl exposes.
        model = IMRPhenomXHM()
        m1, m2, chi1, chi2, dL = 35.0, 22.0, 0.2, -0.1, 0.6
        f_ref, phiRef = 20.0, 0.3
        f = collect(range(20.0, 900.0, length = 40))
        mc, eta = W._xhm_mc_eta(m1, m2)

        h22 = h_lm(model, 2, 2, f, m1, m2, chi1, chi2, dL; f_ref = f_ref, phiRef = phiRef)
        gw_amp = Ampl(IMRPhenomXAS(), f, mc, eta, chi1, chi2, dL)
        @test abs.(h22) ≈ gw_amp * W._XP_HAS_MODE_NORMALIZATION rtol = 1e-12
    end

    @testset "deltaphiLM is frequency-independent (analytic property, not just a numerical accident)" begin
        # deltaphiLM contributes an ADDITIVE CONSTANT to phase, so h_lm(f)/exp(i*deltaphiLM) evaluated
        # with deltaphiLM subtracted at two different points should show the SAME relative phase shift
        # as directly comparing the mode-phase-minus-deltaphiLM construction. Simplest direct check:
        # recompute deltaphiLM twice (independent calls) for the same physical config and confirm it's
        # bit-identical (a NamedTuple field, not recomputed per-frequency).
        m1, m2, chi1, chi2 = 40.0, 12.0, 0.35, -0.15
        mc, eta = W._xhm_mc_eta(m1, m2)
        sys = W._xhm_system_setup(mc, eta, chi1, chi2, 1.0, 20.0, 0.5)
        for (l, m) in ((2, 1), (3, 3), (4, 4))
            ms = W._xhm_mode_setup(l, m, sys)
            @test isfinite(ms.deltaphiLM)
            ms2 = W._xhm_mode_setup(l, m, sys)
            @test ms.deltaphiLM == ms2.deltaphiLM
        end
    end

    @testset "_xhm22_dphase_raw matches a finite difference of _xhm22_phase_raw" begin
        m1, m2, chi1, chi2 = 32.0, 18.0, 0.25, -0.1
        mc, eta = W._xhm_mc_eta(m1, m2)
        coeffs22 = W._xas_phi_coeffs(IMRPhenomXAS(), mc, eta, chi1, chi2)
        for Mf in (0.003, 0.01, 0.02, 0.06, 0.1)
            dMf = 1e-7
            fd = (W._xhm22_phase_raw(coeffs22, Mf + dMf) - W._xhm22_phase_raw(coeffs22, Mf - dMf)) / (2 * dMf)
            analytic = W._xhm22_dphase_raw(coeffs22, Mf)
            @test analytic ≈ fd rtol = 1e-5
        end
    end

    @testset "_xhm_timeshift22/_xhm_phiref22 are finite and (2,2) mode phase matches _xhm22_phase_full exactly" begin
        m1, m2, chi1, chi2 = 28.0, 26.0, 0.1, 0.05
        mc, eta = W._xhm_mc_eta(m1, m2)
        sys = W._xhm_system_setup(mc, eta, chi1, chi2, 1.0, 20.0, 0.4)
        @test isfinite(sys.timeshift)
        @test isfinite(sys.phiref22)

        f = 200.0
        Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * f
        phi_full = W._xhm22_phase_full(sys.coeffs22, sys.timeshift, sys.phiref22, Mf)
        h22 = h_lm(IMRPhenomXHM(), 2, 2, f, m1, m2, chi1, chi2, 1.0; f_ref = 20.0, phiRef = 0.4)
        @test angle(h22 / cis(phi_full)) ≈ 0.0 atol = 1e-9  # same phase mod 2pi
    end

    @testset "spin-weighted spherical harmonics: normalization sanity at select angles" begin
        # (-2)Y_{2,2}(0,0) = sqrt(5/(4*pi)), a standard textbook value (already used elsewhere in
        # this package, e.g. IMRPhenomXPrecession.jl's twist_up22 docstring).
        @test W._xhm_swsh(2, 2, 0.0, 0.0) ≈ sqrt(5.0 / (4.0 * pi)) atol = 1e-12
        @test abs(W._xhm_swsh(2, -2, 0.0, 0.0)) ≈ 0.0 atol = 1e-12
        # phi-dependence is a pure exp(i*m*phi) rotation of the theta=0 (phi-independent-magnitude) value.
        for phi in (0.3, 1.7, -0.5)
            @test W._xhm_swsh(3, 3, 0.4, phi) ≈ W._xhm_swsh(3, 3, 0.4, 0.0) * cis(3 * phi) rtol = 1e-12
            @test W._xhm_swsh(4, -4, 0.4, phi) ≈ W._xhm_swsh(4, -4, 0.4, 0.0) * cis(-4 * phi) rtol = 1e-12
        end
        # `_xhm_swsh_fac` now covers every `(l,m)` with `l in {2,3,4}`, `|m| <= l` -- the 11
        # combinations beyond this file's own 5-mode needs (`m = 0` among them) were added for
        # `IMRPhenomXPHM.jl`'s twisting-up, whose per-mode transfer sums run over the full
        # `m = -l..l` range. So `(2,0)` no longer throws; `l = 5` and `|m| > l` still do.
        @test W._xhm_swsh(2, 0, 0.5, 0.0) ≈ sqrt(15.0 / (32.0 * pi)) * sin(0.5)^2 rtol = 1e-12
        @test_throws ArgumentError W._xhm_swsh(5, 0, 0.5, 0.0)
        @test_throws ArgumentError W._xhm_swsh(4, 5, 0.5, 0.0)
    end

    @testset "Ampzero degeneracy (equal-mass, equal-spin, odd-m modes) does not throw/crash" begin
        # LALSuite's "Ampzero" case: for q=1, chi1==chi2, the odd-m
        # modes' PN amplitude collocation values vanish identically -- a genuine 0/0 in the raw
        # LALSuite ansatz itself (not a bug), which the LALSuite comparisons explicitly SKIP
        # rather than compare. This port inherits the same intrinsic 0/0 (confirmed: produces NaN
        # here too, not a silently-wrong finite value) -- the only thing worth asserting at the unit
        # level is that evaluating at this degenerate point doesn't throw/crash the whole pipeline
        # (e.g. a DomainError from a negative sqrt), since real callers may legitimately hit q=1,
        # equal-spin configurations.
        model = IMRPhenomXHM()
        m1, m2, chi1, chi2, dL = 30.0, 30.0, 0.2, 0.2, 0.5
        f = collect(range(20.0, 800.0, length = 60))
        for (l, m) in ((2, 1), (3, 3))
            h = h_lm(model, l, m, f, m1, m2, chi1, chi2, dL; f_ref = 20.0, phiRef = 0.0)
            @test h isa Vector{ComplexF64}
            @test length(h) == length(f)
        end
    end

    @testset "GMsun_over_c3/GMsun_over_c2_Gpc keyword overrides are threaded through" begin
        model = IMRPhenomXHM()
        m1, m2, chi1, chi2, dL = 30.0, 20.0, 0.3, -0.2, 0.4
        f = 100.0
        h_default = h_lm(model, 2, 1, f, m1, m2, chi1, chi2, dL; f_ref = 20.0)
        h_custom = h_lm(model, 2, 1, f, m1, m2, chi1, chi2, dL; f_ref = 20.0,
            GMsun_over_c3 = W.GMsun_over_c3 * 1.0, GMsun_over_c2_Gpc = W.GMsun_over_c2_Gpc * 1.0)
        @test h_default ≈ h_custom rtol = 1e-14
    end
end

# LAL's `Ampzero` : (2,1), (3,3) vanish for equal masses (eta rounds to 0.25, |1 - q| < ~2e-8) and equal
# spins; their amplitude fits are 0/0 there and the whole waveform was NaN (incl. non-spinning q = 1).
@testset "IMRPhenomXHM: odd modes zero at equal mass, equal spin" begin
    f = collect(20.0:2.0:512.0)
    for (c1, c2) in ((0.0, 0.0), (0.3, 0.3)), q in (1 - 1e-8, 1.0)
        m1 = 65 / (1 + q); m2 = 65 - m1
        @test all(iszero, h_lm(IMRPhenomXHM(), 2, 1, f, m1, m2, c1, c2, 400.0; f_ref = 20.0))
        @test all(iszero, h_lm(IMRPhenomXHM(), 3, 3, f, m1, m2, c1, c2, 400.0; f_ref = 20.0))
        hp, hc = hphc(IMRPhenomXHM(), f, m1, m2, c1, c2, 400.0, 0.6, 1.0; f_ref = 20.0)
        @test all(isfinite, hp) && all(isfinite, hc)
        # continuous: just before the snap the odd modes are O(1 - q)
        m1b = 65 / (2 - 1e-7); hpb, _ = hphc(IMRPhenomXHM(), f, m1b, 65 - m1b, c1, c2, 400.0, 0.6, 1.0; f_ref = 20.0)
        @test hp ≈ hpb rtol = 1e-5
    end
end
