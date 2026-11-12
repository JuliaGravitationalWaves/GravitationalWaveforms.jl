using Test
using GravitationalWaveforms

# `_xp_twist_up22` is intentionally unexported (matching `_xp_precession_setup`'s split), so it's
# accessed qualified. `twist_up22`/`IMRPhenomXP` are the public, exported entry points.
const _twist_scalar = GravitationalWaveforms._xp_twist_up22
const _setup = GravitationalWaveforms._xp_precession_setup

# `_v_of_f` as in test_precession_frame.jl: reused when that file ran first in this process (runtests.jl order),
# defined here when the parallel runner puts this file in another process.
if !@isdefined(_v_of_f)
    function _v_of_f(m1, m2, f)
        T = promote_type(typeof(m1), typeof(m2), typeof(f))
        return cbrt(T(pi) * (m1 + m2) * T(GravitationalWaveforms.GMsun_over_c3) * f)
    end
end

# Aligned-spin-limit closed form for hp_sum/hc_sum, hand-derived from `_xp_twist_up22`'s own
# five-term sum (see that function's docstring): when cos_beta == +1 exactly (cBetah=1, sBetah=0,
# the L+SL>0 case), every Wigner-d coefficient vanishes except d2m2[m=-2]=1 and d22[m=2]=1, so
#   hp_sum = exp(2i*alpha)*(Y2m2 + Y22),  hc_sum = i*exp(2i*alpha)*(Y2m2 - Y22).
# hp/hc are then eps_phase_hP = exp(-2i*epsilon)*hAS/2 times these sums.
function _aligned_spin_limit_hphc(v, setup, hAS)
    T = setup.T
    alpha = GravitationalWaveforms._xp_pn_euler_alpha_nnlo(v, setup)
    epsilon = GravitationalWaveforms._xp_pn_euler_epsilon_nnlo(v, setup)
    theta = setup.thetaJN
    ct, st = cos(theta), sin(theta)
    Y2m2 = sqrt(5.0 / (64.0 * pi)) * (1.0 - ct)^2
    Y22 = sqrt(5.0 / (64.0 * pi)) * (1.0 + ct)^2
    hp_sum = cis(2 * alpha) * (Y2m2 + Y22)
    hc_sum = im * cis(2 * alpha) * (Y2m2 - Y22)
    eps_phase_hP = cis(-2 * epsilon) * Complex{T}(hAS) / 2
    return eps_phase_hP * hp_sum, eps_phase_hP * hc_sum
end

@testset "IMRPhenomXPrecession (twisting-up)" begin

    @testset "Aligned-spin limit: hp/hc reduce EXACTLY to the hand-derived closed form" begin
        # cos_beta == +1 (L+SL > 0, the typical case for moderate spins) for all three configs.
        for (m1, m2, chi1z, chi2z) in ((30.0, 20.0, 0.3, -0.2), (40.0, 10.0, -0.1, 0.5), (25.0, 24.0, 0.0, 0.0))
            f_ref = 20.0
            inclination = 0.5
            phiRef = 0.7
            setup = _setup(m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z, f_ref, inclination, phiRef)
            @test setup.Sperp == 0.0  # precondition for the closed form below

            for f in range(20.0, 512.0, length = 25)
                v = _v_of_f(m1, m2, f)
                cb = GravitationalWaveforms._xp_cos_beta(v, setup)
                @test cb == 1.0  # confirms we're in the beta==0 branch of the closed form

                hAS = (0.7 + 0.2im) * exp(1.3im * f / 100)  # arbitrary nontrivial synthetic hAS
                hp, hc = _twist_scalar(v, setup, hAS)
                hp_expected, hc_expected = _aligned_spin_limit_hphc(v, setup, hAS)

                @test hp ≈ hp_expected rtol=1e-12
                @test hc ≈ hc_expected rtol=1e-12
            end
        end
    end

    @testset "twist_up22 end-to-end aligned-spin limit matches IMRPhenomXAS's own hAS decomposition" begin
        # With Sperp == 0 (aligned spins), the closed form above further specializes since alpha,
        # epsilon, thetaJN are all real numbers computed by the same setup; cross-check the
        # public `twist_up22` wrapper (which internally calls Ampl/Phi(::IMRPhenomXAS,...) to build
        # hAS) against the same hand-derived closed form, using the actual hAS it computes.
        model = IMRPhenomXP()
        m1, m2 = 30.0, 20.0
        chi1z, chi2z = 0.3, -0.2
        f_ref = 20.0
        inclination = 0.5
        phiRef = 0.7
        dL = 1.0
        setup = _setup(m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z, f_ref, inclination, phiRef)

        mc = setup.eta^0.6 * (m1 + m2)
        f = collect(range(20.0, 512.0, length = 30))
        ampl = Ampl(IMRPhenomXAS(), f, mc, setup.eta, chi1z, chi2z, dL)
        phi = Phi(IMRPhenomXAS(), f, mc, setup.eta, chi1z, chi2z; phiRef = phiRef)

        hp, hc = twist_up22(model, f, m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z,
            f_ref, inclination, phiRef, dL)

        @test hp isa Vector{ComplexF64}
        @test hc isa Vector{ComplexF64}
        @test length(hp) == length(f)

        for i in eachindex(f)
            v = _v_of_f(m1, m2, f[i])
            # `twist_up22` internally converts IMRPhenomXAS's face-on-normalized Ampl*exp(iPhi) into
            # the raw (2,2)-mode convention `_xp_twist_up22`/`IMRPhenomXPTwistUp22` expect via
            # `_XP_HAS_MODE_NORMALIZATION` (see that constant's docstring) -- apply the same
            # conversion here so this is a fair apples-to-apples comparison against the internal
            # formula, not a re-introduction of the bug that constant fixes.
            hAS = ampl[i] * cis(phi[i]) * GravitationalWaveforms._XP_HAS_MODE_NORMALIZATION
            hp_expected, hc_expected = _aligned_spin_limit_hphc(v, setup, hAS)
            @test hp[i] ≈ hp_expected rtol=1e-10
            @test hc[i] ≈ hc_expected rtol=1e-10
        end
    end

    @testset "Face-on (theta_jn=0) normalization matches IMRPhenomXAS's own already-LALSuite-validated hplus" begin
        # This is the externally-anchored check that catches what the previous testset's internal
        # self-consistency check cannot: at theta_jn=0 (face-on) with Sperp=0 (aligned spins, so
        # cos_beta=1 exactly, no precession), the twisted-up hp/hc must equal IMRPhenomXAS's own
        # Ampl*exp(iPhi) UP TO the `exp(-2i*zeta_polarization)` overall phase factor
        # `_xp_apply_zeta_rotation` applies (see that function's docstring, and
        # `_xp_precession_setup`'s `zeta_polarization` field docstring) -- NOT exactly equal to
        # `h`/`-i*h` with no extra phase, since `zeta_polarization` is generically nonzero even at
        # zero in-plane spin (it depends on `inclination`/`phiRef`/`kappa`, not on `chi_p`) and
        # this port's own algebra shows a rotation applied to a `hc=-i*hp` signal always preserves
        # that exact ratio while multiplying BOTH by `exp(-2i*zeta)` -- confirmed against REAL
        # LALSuite `IMRPhenomXP` output (not just this port's own internals): at `inclination=0`,
        # aligned spins, `|LAL hplus|/|GravitationalWaveBayes Ampl*exp(iPhi)| = 1` to machine precision and
        # `LAL_hplus/h == LAL_hcross/(-i*h)` exactly at every frequency checked (diagnostic,
        # `m1=30,m2=20,chi1=0.3,chi2=-0.2,phiRef=0.7`) -- i.e. real LALSuite genuinely does apply
        # this same extra phase at face-on, it is not an artifact of this port. An EARLIER version
        # of this test asserted `hp==h`/`hc==-i*h` with no phase factor at all -- written before
        # `zeta_polarization` was implemented, and WRONG once it was (a fresh `Pkg.test()` run
        # caught the resulting failure).
        model = IMRPhenomXP()
        m1, m2 = 30.0, 20.0
        chi1z, chi2z = 0.3, -0.2
        f_ref = 20.0
        inclination = 0.0   # face-on
        phiRef = 0.7
        dL = 1.0
        setup = _setup(m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z, f_ref, inclination, phiRef)
        @test setup.thetaJN == 0.0
        @test setup.Sperp == 0.0

        mc = setup.eta^0.6 * (m1 + m2)
        f = collect(range(20.0, 512.0, length = 30))
        ampl = Ampl(IMRPhenomXAS(), f, mc, setup.eta, chi1z, chi2z, dL)
        phi = Phi(IMRPhenomXAS(), f, mc, setup.eta, chi1z, chi2z; phiRef = phiRef)

        hp, hc = twist_up22(model, f, m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z,
            f_ref, inclination, phiRef, dL)

        zeta_phase = cis(-2 * setup.zeta_polarization)
        for i in eachindex(f)
            h_faceon = ampl[i] * cis(phi[i]) * zeta_phase
            @test hp[i] ≈ h_faceon rtol=1e-10
            @test hc[i] ≈ -im * h_faceon rtol=1e-10
        end
        # And the ratio hc/hp itself, independent of the overall phase, is exactly -i regardless
        # -- this is the part of the ORIGINAL test's intent that remains valid and is the more
        # fundamental invariant (a pure polarization-frame rotation can never break this ratio for
        # an aligned-spin/non-precessing signal, proven algebraically in this port's own
        # `_xp_apply_zeta_rotation` docstring).
        for i in eachindex(f)
            @test hc[i] / hp[i] ≈ -im rtol=1e-10
        end
    end

    @testset "Energy/amplitude sanity for a genuinely precessing configuration" begin
        configs = (
            (m1 = 35.0, m2 = 12.0, chi1x = 0.4, chi1y = 0.1, chi1z = 0.2, chi2x = -0.3, chi2y = 0.2, chi2z = -0.1),
            (m1 = 60.0, m2 = 8.0, chi1x = 0.6, chi1y = -0.3, chi1z = 0.1, chi2x = 0.1, chi2y = 0.5, chi2z = 0.0),
            (m1 = 20.0, m2 = 18.0, chi1x = 0.2, chi1y = 0.6, chi1z = -0.4, chi2x = -0.5, chi2y = -0.1, chi2z = 0.3),
        )
        f_ref = 20.0
        inclination = 0.8
        phiRef = 1.2
        dL = 1.0
        model = IMRPhenomXP()
        f = collect(range(20.0, 1024.0, length = 60))

        for c in configs
            setup = _setup(c.m1, c.m2, c.chi1x, c.chi1y, c.chi1z, c.chi2x, c.chi2y, c.chi2z, f_ref, inclination, phiRef)
            @test setup.Sperp > 0  # genuinely precessing

            mc = setup.eta^0.6 * (c.m1 + c.m2)
            # `twist_up22` internally overrides the ringdown/damping final spin with the
            # PRECESSING one -- this reference amplitude
            # must use the same override, or a shifted `fRING` alone (not precession redistributing
            # power) would show up as a spurious violation of this sanity bound right at ringdown.
            final_spin = GravitationalWaveforms._xas_precessing_final_spin(setup.eta, c.chi1z, c.chi2z, setup.chi_p)
            ampl = Ampl(IMRPhenomXAS(), f, mc, setup.eta, c.chi1z, c.chi2z, dL;
                final_spin_override = final_spin)

            hp, hc = twist_up22(model, f, c.m1, c.m2, c.chi1x, c.chi1y, c.chi1z,
                c.chi2x, c.chi2y, c.chi2z, f_ref, inclination, phiRef, dL)

            @test all(isfinite, hp)
            @test all(isfinite, hc)

            for i in eachindex(f)
                pow = abs2(hp[i]) + abs2(hc[i])
                @test isfinite(pow)
                # Precession redistributes power between polarizations/across the band; it should
                # not manufacture or destroy it wildly -- require the same order of magnitude as
                # |hAS|^2 (generously: within a factor of 10 either way).
                as_pow = abs2(ampl[i])
                @test 0.1 * as_pow <= pow <= 10.0 * as_pow
            end
        end
    end

    @testset "Finite/no-NaN across frequency sweep and precessing configs" begin
        configs = (
            (m1 = 35.0, m2 = 12.0, chi1x = 0.4, chi1y = 0.1, chi1z = 0.2, chi2x = -0.3, chi2y = 0.2, chi2z = -0.1),
            (m1 = 60.0, m2 = 8.0, chi1x = 0.6, chi1y = -0.3, chi1z = 0.1, chi2x = 0.1, chi2y = 0.5, chi2z = 0.0),
            (m1 = 20.0, m2 = 18.0, chi1x = 0.2, chi1y = 0.6, chi1z = -0.4, chi2x = -0.5, chi2y = -0.1, chi2z = 0.3),
        )
        f_ref = 20.0
        inclination = 0.8
        phiRef = 1.2
        dL = 1.0
        model = IMRPhenomXP()

        for c in configs
            for f in range(20.0, 1024.0, length = 15)
                hp, hc = twist_up22(model, f, c.m1, c.m2, c.chi1x, c.chi1y, c.chi1z,
                    c.chi2x, c.chi2y, c.chi2z, f_ref, inclination, phiRef, dL)
                @test hp isa ComplexF64
                @test hc isa ComplexF64
                @test isfinite(hp)
                @test isfinite(hc)
            end
        end
    end

    @testset "Float32/Float64 type genericity" begin
        for T in (Float32, Float64)
            m1, m2 = T(35.0), T(12.0)
            chi1x, chi1y, chi1z = T(0.4), T(0.1), T(0.2)
            chi2x, chi2y, chi2z = T(-0.3), T(0.2), T(-0.1)
            f_ref = T(20.0)
            inclination = T(0.8)
            phiRef = T(1.2)
            dL = T(1.0)
            model = IMRPhenomXP()

            setup = _setup(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination, phiRef)
            @test setup.T === T

            v = _v_of_f(m1, m2, T(100.0))
            hAS = Complex{T}(T(0.5), T(0.1))
            hp, hc = _twist_scalar(v, setup, hAS)
            @test hp isa Complex{T}
            @test hc isa Complex{T}
            @test isfinite(hp) && isfinite(hc)

            hp_s, hc_s = twist_up22(model, T(100.0), m1, m2, chi1x, chi1y, chi1z,
                chi2x, chi2y, chi2z, f_ref, inclination, phiRef, dL)
            @test hp_s isa Complex{T}
            @test hc_s isa Complex{T}

            f = collect(range(T(20.0), T(1024.0), length = 10))
            hp_v, hc_v = twist_up22(model, f, m1, m2, chi1x, chi1y, chi1z,
                chi2x, chi2y, chi2z, f_ref, inclination, phiRef, dL)
            @test eltype(hp_v) === Complex{T}
            @test eltype(hc_v) === Complex{T}
            @test all(isfinite, hp_v)
            @test all(isfinite, hc_v)
        end
    end

end
