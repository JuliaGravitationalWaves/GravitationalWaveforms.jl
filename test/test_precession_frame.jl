using Test
using GravitationalWaveforms

# `_xp_precession_setup`/`_xp_cos_beta`/`_xp_pn_euler_alpha_nnlo`/`_xp_pn_euler_epsilon_nnlo` are
# intentionally unexported (matching this package's `_xas_phi_coeffs`/`_xas_phi_scalar` internal
# split -- see IMRPhenomXPrecession.jl's module docstring), so they're accessed qualified.
const _setup = GravitationalWaveforms._xp_precession_setup
const _cos_beta = GravitationalWaveforms._xp_cos_beta
const _alpha_nnlo = GravitationalWaveforms._xp_pn_euler_alpha_nnlo
const _epsilon_nnlo = GravitationalWaveforms._xp_pn_euler_epsilon_nnlo

# v(f) = (pi*M*f)^(1/3) in geometric units, matching IMRPhenomXPrecession.jl's own convention
# (Mf = (m1+m2)*GMsun_over_c3*f, omega = pi*Mf, v = cbrt(omega)).
function _v_of_f(m1, m2, f)
    T = promote_type(typeof(m1), typeof(m2), typeof(f))
    GMsun_over_c3 = T(GravitationalWaveforms.GMsun_over_c3)
    Mf = (m1 + m2) * GMsun_over_c3 * f
    return cbrt(T(pi) * Mf)
end

@testset "IMRPhenomXPrecession (precession-frame setup)" begin

    @testset "Aligned-spin limit: Sperp, chi_p exactly zero; cos_beta == copysign(1, L+SL)" begin
        for (m1, m2, chi1z, chi2z) in ((30.0, 20.0, 0.3, -0.2), (40.0, 10.0, -0.1, 0.5), (25.0, 24.0, 0.0, 0.0))
            f_ref = 20.0
            inclination = 0.5
            phiRef = 0.7
            setup = _setup(m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z, f_ref, inclination, phiRef)

            @test setup.Sperp == 0.0
            @test setup.chi_p == 0.0

            for f in range(20.0, 512.0, length = 25)
                v = _v_of_f(m1, m2, f)
                cb = _cos_beta(v, setup)
                L = GravitationalWaveforms._xp_L_pn_ansatz(v, setup.eta / v, setup.Lcoeffs...)
                @test cb == copysign(1.0, L + setup.SL)
            end
        end
    end

    @testset "thetaJN reduces to inclination in the aligned-spin limit" begin
        m1, m2 = 30.0, 20.0
        chi1z, chi2z = 0.3, -0.2
        f_ref = 20.0
        for inclination in (0.0, 0.3, 1.0, 2.0, pi)
            phiRef = 0.9
            setup = _setup(m1, m2, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z, f_ref, inclination, phiRef)
            # J0z > 0 for this mild-spin, moderate-mass-ratio configuration (L dominates SL), so
            # J0 is parallel (not anti-parallel) to Lhat and thetaJN == inclination exactly (up to
            # floating point acos/cos round-trip error).
            @test setup.J0z > 0
            @test isapprox(setup.thetaJN, inclination; atol = 1.0e-10)
        end
    end

    @testset "Precessing configurations: alpha/epsilon/cos_beta finite, cos_beta in [-1,1]" begin
        configs = (
            (m1 = 35.0, m2 = 12.0, chi1x = 0.4, chi1y = 0.1, chi1z = 0.2, chi2x = -0.3, chi2y = 0.2, chi2z = -0.1),
            (m1 = 60.0, m2 = 8.0, chi1x = 0.6, chi1y = -0.3, chi1z = 0.1, chi2x = 0.1, chi2y = 0.5, chi2z = 0.0),
            (m1 = 20.0, m2 = 18.0, chi1x = 0.2, chi1y = 0.6, chi1z = -0.4, chi2x = -0.5, chi2y = -0.1, chi2z = 0.3),
        )
        f_ref = 20.0
        inclination = 0.8
        phiRef = 1.2

        for c in configs
            setup = _setup(c.m1, c.m2, c.chi1x, c.chi1y, c.chi1z, c.chi2x, c.chi2y, c.chi2z, f_ref, inclination, phiRef)

            @test setup.Sperp > 0
            @test setup.chi_p > 0
            @test isfinite(setup.thetaJN)
            @test isfinite(setup.phiJ)
            @test isfinite(setup.alpha_offset)
            @test isfinite(setup.epsilon_offset)

            for f in range(20.0, 512.0, length = 40)
                v = _v_of_f(c.m1, c.m2, f)
                cb = _cos_beta(v, setup)
                al = _alpha_nnlo(v, setup)
                ep = _epsilon_nnlo(v, setup)

                @test isfinite(cb)
                @test isfinite(al)
                @test isfinite(ep)
                @test -1.0 <= cb <= 1.0
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

            setup = _setup(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination, phiRef)
            @test setup.T === T
            @test setup.Sperp isa T
            @test setup.chi_p isa T
            @test setup.thetaJN isa T

            for f in range(T(20.0), T(512.0), length = 10)
                v = _v_of_f(m1, m2, T(f))
                cb = _cos_beta(v, setup)
                al = _alpha_nnlo(v, setup)
                ep = _epsilon_nnlo(v, setup)
                @test cb isa T
                @test al isa T
                @test ep isa T
                @test isfinite(cb) && isfinite(al) && isfinite(ep)
                @test -one(T) <= cb <= one(T)
            end
        end
    end

    @testset "validate_phenomxp_precession_params" begin
        @test validate_phenomxp_precession_params(30.0, 20.0, 0.1, 0.0, 0.2, 0.0, 0.1, -0.2) === nothing
        @test validate_phenomxp_precession_params(30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; f_ref = 20.0) === nothing
        @test_throws ArgumentError validate_phenomxp_precession_params(20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) # m1 < m2
        @test_throws ArgumentError validate_phenomxp_precession_params(30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # m2 <= 0
        @test_throws ArgumentError validate_phenomxp_precession_params(30.0, 20.0, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0) # |chi1| > 1
        @test_throws ArgumentError validate_phenomxp_precession_params(30.0, 20.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9) # |chi2| > 1
        @test_throws ArgumentError validate_phenomxp_precession_params(30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; f_ref = -1.0)
    end

end
