using Test
using GravitationalWaveforms

# `_xphm_twist_up_mode`/`_xphm_hphc_scalar` are intentionally unexported (matching
# `_xp_twist_up22`/`_xhm_hphc_scalar`'s own split); `IMRPhenomXPHM`/`hphc` are the public entry
# points.
const _xphm_twist = GravitationalWaveforms._xphm_twist_up_mode
const _xphm_setup = GravitationalWaveforms._xp_precession_setup

const _XPHM_MODES = ((2, 2), (2, 1), (3, 3), (3, 2), (4, 4))

# Rebuild one aligned-spin co-precessing-frame mode `h_{l,-mprime}` exactly the way
# `_xphm_hphc_scalar` does internally, so the per-mode tests below feed the twisting-up function
# the same input the real code path does (rather than a synthetic stand-in).
function _xphm_test_hlm(l, mprime, fi, sys, ms21, ms33, ms44, m32)
    if (l, mprime) == (2, 2)
        Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * fi
        amp = GravitationalWaveforms._xas_ampl_scalar(sys.amplcoeffs22, fi)
        ph = GravitationalWaveforms._xhm22_phase_full(sys.coeffs22, sys.timeshift, sys.phiref22, Mf)
        return GravitationalWaveforms._XP_HAS_MODE_NORMALIZATION * (amp * cis(ph))
    elseif (l, mprime) == (2, 1)
        return GravitationalWaveforms._xhm_hlm_nonmixing(2, fi, sys, ms21)
    elseif (l, mprime) == (3, 3)
        return GravitationalWaveforms._xhm_hlm_nonmixing(3, fi, sys, ms33)
    elseif (l, mprime) == (4, 4)
        return GravitationalWaveforms._xhm_hlm_nonmixing(4, fi, sys, ms44)
    else
        return GravitationalWaveforms._xhm_hlm_32(fi, sys, m32)
    end
end

@testset "IMRPhenomXPHM (precessing + higher-order modes)" begin

    @testset "marker struct and basic sanity" begin
        model = IMRPhenomXPHM()
        @test model isa IMRPhenomXPHM

        f = collect(range(20.0, 512.0, length = 60))
        hp, hc = hphc(model, f, 35.0, 12.0, 0.4, 0.1, 0.2, -0.3, 0.2, -0.1, 20.0, 0.8, 1.2, 1.0)
        @test hp isa Vector{ComplexF64}
        @test hc isa Vector{ComplexF64}
        @test length(hp) == length(f) && length(hc) == length(f)
        @test all(isfinite, hp) && all(isfinite, hc)
        @test maximum(abs, hp) > 0.0 && maximum(abs, hc) > 0.0

        # Scalar-vs-vector convention (this package's `Ampl`/`Phi`/`twist_up22`/`hphc` contract).
        hp_s, hc_s = hphc(model, f[10], 35.0, 12.0, 0.4, 0.1, 0.2, -0.3, 0.2, -0.1, 20.0, 0.8, 1.2, 1.0)
        @test hp_s isa ComplexF64 && hc_s isa ComplexF64
        @test hp_s ≈ hp[10] rtol=1e-13
        @test hc_s ≈ hc[10] rtol=1e-13

        # A genuinely precessing configuration must NOT satisfy the aligned-spin `hc == -i*hp`
        # identity (which `_xp_apply_zeta_rotation`'s docstring proves is exactly preserved for
        # aligned spins) -- i.e. the twisting-up is doing real work here, not passing through.
        @test maximum(abs.(hc .+ im .* hp)) > 1e-3 * maximum(abs.(hp))
    end

    @testset "unsupported (l,mprime) throws" begin
        setup = _xphm_setup(35.0, 12.0, 0.4, 0.1, 0.2, -0.3, 0.2, -0.1, 20.0, 0.8, 1.2)
        # (4,3) is present in LALSuite's own IMRPhenomXPHMTwistUp but marked "only used when
        # twisting PhenomHM" -- deliberately out of scope here, see IMRPhenomXPHM.jl's docstring.
        @test_throws ArgumentError _xphm_twist(4, 3, 0.2, setup, 1.0 + 0.0im)
        @test_throws ArgumentError _xphm_twist(5, 5, 0.2, setup, 1.0 + 0.0im)
    end

    @testset "(2,2) branch delegates to _xp_twist_up22 bit-for-bit" begin
        # `mprime = 2` means `v_mprime == v`, so the (2,2) twisting-up IS `_xp_twist_up22` -- the
        # already-LALSuite-validated IMRPhenomXP path, reused rather than re-derived.
        setup = _xphm_setup(35.0, 12.0, 0.4, 0.1, 0.2, -0.3, 0.2, -0.1, 20.0, 0.8, 1.2)
        Mtot = 47.0
        for f in range(20.0, 512.0, length = 20)
            v = cbrt(pi * Mtot * GravitationalWaveforms.GMsun_over_c3 * f)
            for hAS in ((3.0 + 1.5im), (-0.2 - 0.9im))
                @test _xphm_twist(2, 2, v, setup, hAS) === GravitationalWaveforms._xp_twist_up22(v, setup, hAS)
            end
        end
    end

    @testset "aligned-spin + phiRef=0 limit reproduces IMRPhenomXHM exactly" begin
        # NNLO (PrecVersion 102) only: LAL 223 (MSA) does NOT reduce to XHM for aligned spins (checked against
        # LAL 6.2.0: 5-112% differences; ours reproduces that, test_phenomxp_msa.jl).
        # With `chi1x=chi1y=chi2x=chi2y=0`: `Sperp == 0`, so `cos_beta == 1` (beta == 0) at every
        # frequency, every Wigner-d collapses to `d^l_{±mprime,±mprime} = 1`, and (because the
        # `alpha_i` and `epsilon_i` NNLO coefficients coincide identically at `chi_p == 0`)
        # `alpha(v) - epsilon(v) == alpha0 - epsilon0` is frequency-INDEPENDENT. At `phiRef == 0`
        # the residual constant is exactly the `exp(-i*mprime*pi/2)` factor `_xhm_add_mode!`'s own
        # hardcoded `phi = pi/2` azimuth supplies, and `zeta_polarization == pi` makes the final
        # rotation the identity -- so the two models must agree to machine precision, mode by mode
        # and in total. This exercises every Wigner-d table, every spherical harmonic and every
        # per-`l` sign pattern simultaneously.
        #
        # `phiRef = 0` is used here because `hphc(::IMRPhenomXPHM, ...)` generates the co-precessing
        # modes at `phi0 = 0` regardless of the user's `phiRef` (LALSuite's Convention-1 behaviour,
        # `LALSimIMRPhenomX_precession.c` line 836 -- see the source docstring), while
        # `hphc(::IMRPhenomXHM, ...)` uses the `phiRef` it is given. The two therefore coincide
        # exactly only at `phiRef = 0`; the `phiRef != 0` testset below covers the general case.
        f = collect(range(20.0, 512.0, length = 200))
        for (m1, m2, chi1z, chi2z, incl) in ((35.0, 12.0, 0.2, -0.1, 0.8),
                                             (30.0, 25.0, 0.3, -0.2, 0.4),
                                             (63.0, 7.0, 0.6, 0.2, 1.9))
            hpP, hcP = hphc(IMRPhenomXPHM(102), f, m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z,
                20.0, incl, 0.0, 1.0)
            hpX, hcX = hphc(IMRPhenomXHM(), f, m1, m2, chi1z, chi2z, 1.0, incl, 0.0; f_ref = 20.0)
            @test maximum(abs.(hpP .- hpX)) / maximum(abs.(hpX)) < 1e-13
            @test maximum(abs.(hcP .- hcX)) / maximum(abs.(hcX)) < 1e-13
        end
    end

    @testset "aligned-spin limit at phiRef != 0: per-mode exp(i*mprime*phiRef) offset" begin
        # In the aligned-spin limit each twisted mode equals `IMRPhenomXHM`'s own `_xhm_add_mode!`
        # contribution for that mode times a FREQUENCY-INDEPENDENT `exp(i*mprime*phiRef)`, when the
        # aligned-spin `sys` is built at the SAME `phiRef` that the precession setup saw. The
        # frequency-independence is the substantive claim (a frequency-dependent residual would mean
        # the Euler angles were entering wrongly); the closed form for the constant is checked too.
        #
        # This `exp(i*mprime*phiRef)` is precisely why `hphc(::IMRPhenomXPHM, ...)` must NOT pass the
        # user's `phiRef` into `_xhm_system_setup`: the factor is MPRIME-DEPENDENT, so feeding
        # `phiRef` in both places leaves the five modes mutually misaligned (measured against
        # LALSuite: ~7% amplitude and ~2.4 rad phase error). The next testset guards that directly.
        m1, m2, chi1z, chi2z = 35.0, 12.0, 0.2, -0.1
        incl, phiRef, f_ref, dL = 0.8, 1.2, 20.0, 1.0
        setup = _xphm_setup(m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z, f_ref, incl, phiRef)
        @test setup.chi_p == 0.0

        mc, eta = GravitationalWaveforms._xhm_mc_eta(m1, m2)
        sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1z, chi2z, dL, f_ref, phiRef)
        ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
        ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
        ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
        m32 = GravitationalWaveforms._xhm_mode32_setup(sys)

        for (l, mprime) in _XPHM_MODES
            ratios = ComplexF64[]
            for fi in range(20.0, 512.0, length = 40)
                Mf = sys.coeffs22.M * sys.coeffs22.GMsun_over_c3 * fi
                @test GravitationalWaveforms._xp_cos_beta(cbrt(pi * Mf * 2 / mprime), setup) == 1.0
                hlm = _xphm_test_hlm(l, mprime, fi, sys, ms21, ms33, ms44, m32)
                hp1, hc1 = _xphm_twist(l, mprime, cbrt(pi * Mf * 2 / mprime), setup, hlm)
                hp2, hc2 = GravitationalWaveforms._xhm_add_mode!(zero(ComplexF64), zero(ComplexF64),
                    l, mprime, hlm, incl, pi / 2)
                push!(ratios, hp1 / hp2)
                @test hc1 / hc2 ≈ hp1 / hp2 rtol=1e-11
            end
            expected = cis(mprime * phiRef)
            @test all(r -> isapprox(r, expected; rtol = 1e-11), ratios)
        end
    end

    @testset "co-precessing modes are generated at phi0 = 0, not at the user's phiRef" begin
        # Direct regression guard on the Convention-1 `pWF->phi0 = 0` override
        # (`LALSimIMRPhenomX_precession.c` lines 828-838). In the aligned-spin limit,
        # `hphc(::IMRPhenomXPHM, ..., phiRef)` must equal the explicit five-mode sum built from
        # `IMRPhenomXHM` modes evaluated at phiRef = 0 (NOT at `phiRef`), each multiplied by that
        # mode's `exp(i*mprime*phiRef)` twisting factor. Passing the user's `phiRef` into
        # `_xhm_system_setup` instead breaks this identity -- which is exactly the bug this test
        # exists to catch, and which cost ~7% amplitude / ~2.4 rad phase against real LALSuite.
        m1, m2, chi1z, chi2z = 35.0, 12.0, 0.2, -0.1
        incl, f_ref, dL = 0.8, 20.0, 1.0
        f = collect(range(20.0, 400.0, length = 80))

        for phiRef in (0.0, 0.7, 1.2, -2.1)
            setup = _xphm_setup(m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z, f_ref, incl, phiRef)
            mc, eta = GravitationalWaveforms._xhm_mc_eta(m1, m2)
            # phiRef = 0 here, deliberately -- that is the whole point of this test.
            sys = GravitationalWaveforms._xhm_system_setup(mc, eta, chi1z, chi2z, dL, f_ref, 0.0)
            ms21 = GravitationalWaveforms._xhm_mode_setup(2, 1, sys)
            ms33 = GravitationalWaveforms._xhm_mode_setup(3, 3, sys)
            ms44 = GravitationalWaveforms._xhm_mode_setup(4, 4, sys)
            m32 = GravitationalWaveforms._xhm_mode32_setup(sys)

            expect_hp = ComplexF64[]
            expect_hc = ComplexF64[]
            for fi in f
                hp = zero(ComplexF64); hc = zero(ComplexF64)
                for (l, mprime) in _XPHM_MODES
                    hlm = _xphm_test_hlm(l, mprime, fi, sys, ms21, ms33, ms44, m32)
                    a, b = GravitationalWaveforms._xhm_add_mode!(zero(ComplexF64), zero(ComplexF64),
                        l, mprime, hlm, incl, pi / 2)
                    tw = cis(mprime * phiRef)
                    hp += tw * a
                    hc += tw * b
                end
                a, b = GravitationalWaveforms._xp_apply_zeta_rotation(hp, hc, setup.zeta_polarization)
                push!(expect_hp, a); push!(expect_hc, b)
            end

            got_hp, got_hc = hphc(IMRPhenomXPHM(102), f, m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z,
                f_ref, incl, phiRef, dL)
            @test maximum(abs.(got_hp .- expect_hp)) / maximum(abs.(expect_hp)) < 1e-12
            @test maximum(abs.(got_hc .- expect_hc)) / maximum(abs.(expect_hc)) < 1e-12
        end
    end

    @testset "Float32 genericity" begin
        f32 = Float32.(collect(range(20.0, 512.0, length = 30)))
        hp, hc = hphc(IMRPhenomXPHM(), f32, 35.0f0, 12.0f0, 0.4f0, 0.1f0, 0.2f0,
            -0.3f0, 0.2f0, -0.1f0, 20.0f0, 0.8f0, 1.2f0, 1.0f0)
        @test eltype(hp) === ComplexF32
        @test eltype(hc) === ComplexF32
        @test all(isfinite, hp) && all(isfinite, hc)
    end

    @testset "spin-weighted spherical harmonics: newly added (l,m) branches" begin
        # `_xhm_swsh_fac` gained the 11 (l,m) combinations IMRPhenomXPHM's full m = -l..l transfer
        # sums need. Check them against the standard orthonormality-independent identity
        # `(-2)Y_{l,-m}(pi - theta, 0) = (-1)^l * (-2)Y_{l,m}(theta, 0)` (the parity relation the
        # `h_{l,-m} = (-1)^l conj(h_{l,m})` mode symmetry rests on), which is an independent check
        # rather than a restatement of the transcribed formulas.
        for theta in (0.3, 0.8, 1.5, 2.4)
            for l in 2:4, m in (-l):l
                lhs = GravitationalWaveforms._xhm_swsh_fac(l, -m, pi - theta)
                rhs = (isodd(l) ? -1.0 : 1.0) * GravitationalWaveforms._xhm_swsh_fac(l, m, theta)
                @test lhs ≈ rhs atol=1e-13
            end
        end
        @test_throws ArgumentError GravitationalWaveforms._xhm_swsh_fac(5, 0, 0.5)
        @test_throws ArgumentError GravitationalWaveforms._xhm_swsh_fac(4, 5, 0.5)
    end
end
