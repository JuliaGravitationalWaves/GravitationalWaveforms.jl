# Enzyme validation for GravitationalWaveforms.SEOBNRv5AlignedSpinF2 (the
# frequency-domain SPA model), mirroring seobnrv5alignedspin_enzyme_ka_tests.jl
# and alignedspinpnham_enzyme_ka_tests.jl's own established shape exactly.
#
# What is differentiated, and why: SEOBNRv5AlignedSpinF2.jl's own module docstring is explicit that
# `_seobv5f2_coeffs` (which runs the Float64-only two-stage IC Newton solve and the Float64-only
# `seobnrv5alignedspin_postadiabatic_dynamics_fixed` driver to build the dense `(ts,phis,omegas)`
# tables) is NOT claimed Enzyme-differentiable -- exactly the same "IC solve is out of scope, only
# the analytic hot path is claimed differentiable" precedent `SEOBNRv5AlignedSpin`'s own Enzyme test
# (`seobnrv5alignedspin_enzyme_ka_tests.jl`) already established for the base model's two-stage IC
# solve. So this file differentiates `_seobv5f2_phi_scalar`/`_seobv5f2_ampl_scalar` -- the genuinely
# new hot path this model adds (Hermite bracket search + closed-form quadratic root solve +
# amplitude Jacobian) -- w.r.t. `coeffs.M_sec`, holding the dense tables fixed (computed once, real
# `SEOBNRv5AlignedSpin` physics, real PA-fixed driver, not a toy proxy).
#
# `M_sec` is not an arbitrary choice: it is the one scalar `coeffs` field that participates in BOTH
# the target-omega computation feeding the bracket search (`omega_target = pi*M_sec*f`, which
# selects the bracket index `i` via `searchsortedlast` -- a genuinely discontinuous,
# data-dependent branch) AND the physical-unit conversions in the final formulas (`t_star_phys =
# t_star*M_sec`, `dOmegaGW_dt_phys = 2*domega_dt/M_sec^2`). A finite, correct-to-leading-order
# derivative through this composition (away from the measure-zero set of `f`/`M_sec` combinations
# that sit exactly on a bracket boundary) is precisely the "does the interpolation's
# data-dependent bracket search compose with Enzyme" question flagged, unresolved, as this model's
# largest engineering risk in its own module docstring.
#
# Both `Enzyme.Reverse` and `Enzyme.Forward` are tried -- this is a small, CPU-only, isolated scalar
# test (no CUDA/`ptxas` involved, no ~1000-line CSE Hamiltonian in the differentiated call graph
# itself, since the dense tables are frozen inputs), so there is no a priori reason to prefer
# forward-mode here the way `SEOBNRv5AlignedSpin`'s own GPU sub-test needed to -- both modes are
# cheap enough to just check.

using Enzyme

@testset "SEOBNRv5AlignedSpinF2: Enzyme through _seobv5f2_phi_scalar/_seobv5f2_ampl_scalar matches finite differences" begin
    model = SEOBNRv5AlignedSpinF2()
    mc, eta, chi1, chi2, dL, fmin = 28.0961, 0.2479, 0.3, -0.2, 0.4, 20.0
    coeffs = GravitationalWaveforms._seobv5f2_coeffs(model, mc, eta, chi1, chi2; f_min = fmin, NMAX = Val(256))
    @test coeffs.success
    f_target = coeffs.fmin + 0.4 * (coeffs.fmax - coeffs.fmin)  # well inside the covered band,
    # away from both the fmin/fmax cutoff edges and (checked below) any bracket-index transition

    function phi_of_M_sec(M_sec_val, coeffs, f)
        c2 = merge(coeffs, (M_sec = M_sec_val,))
        return GravitationalWaveforms._seobv5f2_phi_scalar(c2, f)
    end
    function ampl_of_M_sec(M_sec_val, coeffs, f, dL)
        c2 = merge(coeffs, (M_sec = M_sec_val,))
        return GravitationalWaveforms._seobv5f2_ampl_scalar(c2, f, dL, GravitationalWaveforms.GMsun_over_c2_Gpc)
    end

    h = 1.0e-7 * coeffs.M_sec
    fd_phi = (phi_of_M_sec(coeffs.M_sec + h, coeffs, f_target) -
              phi_of_M_sec(coeffs.M_sec - h, coeffs, f_target)) / (2h)
    fd_ampl = (ampl_of_M_sec(coeffs.M_sec + h, coeffs, f_target, dL) -
               ampl_of_M_sec(coeffs.M_sec - h, coeffs, f_target, dL)) / (2h)

    @testset "Reverse mode" begin
        g_phi = Enzyme.gradient(Enzyme.Reverse, Const(phi_of_M_sec), coeffs.M_sec, Const(coeffs), Const(f_target))
        g_ampl = Enzyme.gradient(Enzyme.Reverse, Const(ampl_of_M_sec), coeffs.M_sec, Const(coeffs), Const(f_target), Const(dL))
        @test isfinite(g_phi[1])
        @test isfinite(g_ampl[1])
        @test isapprox(g_phi[1], fd_phi; rtol = 1e-4, atol = 1e-6)
        @test isapprox(g_ampl[1], fd_ampl; rtol = 1e-4, atol = 1e-6)
    end

    @testset "Forward mode" begin
        dphi = Enzyme.autodiff(Enzyme.Forward, Const(phi_of_M_sec), Duplicated(coeffs.M_sec, 1.0), Const(coeffs), Const(f_target))
        dampl = Enzyme.autodiff(Enzyme.Forward, Const(ampl_of_M_sec), Duplicated(coeffs.M_sec, 1.0), Const(coeffs), Const(f_target), Const(dL))
        @test isfinite(dphi[1])
        @test isfinite(dampl[1])
        @test isapprox(dphi[1], fd_phi; rtol = 1e-4, atol = 1e-6)
        @test isapprox(dampl[1], fd_ampl; rtol = 1e-4, atol = 1e-6)
    end

    @testset "Enzyme composes across a genuine bracket-index change (varying f, not M_sec)" begin
        # A second, independent check of the same underlying concern: differentiate w.r.t. f
        # itself (also feeds the bracket search) at several frequencies spanning different PA
        # table bracket indices, not just one point.
        function phi_of_f(f_val, coeffs)
            return GravitationalWaveforms._seobv5f2_phi_scalar(coeffs, f_val)
        end
        test_fs = range(coeffs.fmin + 0.5, coeffs.fmax - 0.5; length = 8)
        for f in test_fs
            hf = 1.0e-6
            fd = (phi_of_f(f + hf, coeffs) - phi_of_f(f - hf, coeffs)) / (2hf)
            g = Enzyme.gradient(Enzyme.Reverse, Const(phi_of_f), f, Const(coeffs))
            @test isfinite(g[1])
            @test isapprox(g[1], fd; rtol = 1e-3, atol = 1e-4)
        end
    end
end
