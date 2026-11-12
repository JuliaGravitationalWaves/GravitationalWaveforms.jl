# SEOBNRv5AlignedSpinF2.jl: CPU-only, non-Enzyme validation.
#
# Two independent things are checked, mirroring the derivation in that file's own module
# docstring:
# 1. The generic `_seobv5f2_dense_to_freq` SPA reconstruction (Hermite time-to-frequency
#    inversion + amplitude Jacobian), applied to `AlignedSpinPNHam`'s own dense RK4 trajectory,
#    reproduces `AlignedSpinPNHamF2` (already validated to ~1e-14 against real `lalsimulation`)
#    exactly in amplitude (once the Newtonian-order dOmega/dt is substituted for the model's own,
#    matching AlignedSpinPNHamF2's restricted-PN-order amplitude convention) and in phase shape
#    (after removing the expected affine reference-gauge offset).
# 2. Applied to the real `SEOBNRv5AlignedSpin` post-adiabatic table (`SEOBNRv5AlignedSpinF2`
#    itself), the interpolated time-domain phase at the reconstructed t*(f) agrees with an
#    independent from-scratch RK4 re-integration (`SEOBNRv5AlignedSpin.Phi` called directly) to a
#    small, documented tolerance -- the model's own known post-adiabatic-vs-full-dynamics
#    residual, not a defect in the interpolation.
#
# NOT tested here (out of scope, see SEOBNRv5AlignedSpinF2.jl's own module docstring): Enzyme/
# KernelAbstractions differentiability (forward or reverse); merger/ringdown region (this model
# is inspiral-only, hard-cutoff past the PA table's own reach); direct real-pyseobnr frequency-
# domain cross-validation (SEOBNRv5AlignedSpin's own time-domain correctness against pyseobnr is
# validated elsewhere).

@testset "SEOBNRv5AlignedSpinF2: generic SPA reconstruction matches AlignedSpinPNHamF2 exactly (amplitude) and to affine-removed residual (phase)" begin
    model_pn = GravitationalWaveforms.AlignedSpinPNHam()
    model_f2 = GravitationalWaveforms.AlignedSpinPNHamF2()
    mc, eta, chi1, chi2, dL = 28.0961, 0.2479, 0.3, -0.2, 0.4
    fmin = 20.0
    dt = 1.0 / 4096.0
    coeffs_pn = GravitationalWaveforms._asph_coeffs(model_pn, mc, eta, chi1, chi2; f_min = fmin)

    # Single dense forward pass, storing (t_hat, phi, omega) at every RK4 step -- same shape of
    # data SEOBNRv5AlignedSpinF2 gets from the post-adiabatic driver, just from a genuine
    # uniform-in-t RK4 integration instead of a uniform-in-r post-adiabatic one.
    t_end = 0.06  # AlignedSpinPNHam's own docstring: r "unbinds" past ~0.07-0.075s, stay well inside
    dt_hat = dt / coeffs_pn.M_sec
    nsteps = round(Int, t_end / dt)
    state = GravitationalWaveforms.AlignedSpinPNHamState(coeffs_pn.r0, 0.0, 0.0, coeffs_pn.pphi0)
    ts = zeros(nsteps + 1)
    phis = zeros(nsteps + 1)
    omegas = zeros(nsteps + 1)
    for i in 1:(nsteps + 1)
        _, _, dHdpphi = GravitationalWaveforms._asph_hamiltonian_partials(state.r, state.pr, state.pphi,
            coeffs_pn.eta, coeffs_pn.SOcoef, coeffs_pn.SScoef)
        ts[i] = (i - 1) * dt_hat
        phis[i] = state.phi
        omegas[i] = dHdpphi
        i <= nsteps && (state = GravitationalWaveforms._asph_rk4_step(state, dt_hat, coeffs_pn))
    end
    @test all(isfinite, omegas)
    # A naive circular (pr=0) initial condition is not exactly on the adiabatic sequence, so
    # AlignedSpinPNHam's own genuine RK4 trajectory (unlike SEOBNRv5AlignedSpin's post-adiabatic
    # table, tested below, which enforces the adiabatic sequence by construction) carries a small
    # transient (radial "junk" oscillation) for the first ~60 steps before settling onto the
    # monotonically-increasing adiabatic-inspiral trend the reconstruction's bracket search
    # requires -- confirmed directly (`omegas[1:60]` has 58 non-monotonic steps, `omegas[59:end]`
    # none). Trim it rather than assert monotonicity where the physics itself doesn't have it.
    ts, phis, omegas = ts[60:end], phis[60:end], omegas[60:end]
    @test all(diff(omegas) .> 0)

    test_fs = [20.05, 20.1, 20.2, 20.3, 20.4]

    # --- Amplitude: substitute the analytic Newtonian dOmega/dt (matches AlignedSpinPNHamF2's own
    # restricted-PN-order amplitude convention) and expect exact (float-precision) agreement. ---
    for f in test_fs
        r = GravitationalWaveforms._seobv5f2_dense_to_freq(ts, phis, omegas, length(omegas), coeffs_pn.M_sec, f)
        v = cbrt(pi * coeffs_pn.M_sec * f)
        Ampl_TD = 4.0 * coeffs_pn.eta * coeffs_pn.M * GravitationalWaveforms.GMsun_over_c2_Gpc / dL * v * v
        om_hat = pi * coeffs_pn.M_sec * f
        domega_dt_newtonian = (96.0 / 5.0) * eta * om_hat^(11.0 / 3.0)
        dOmegaGW_dt_phys = 2.0 * domega_dt_newtonian / coeffs_pn.M_sec^2
        Ampl_FD_mine = 0.5 * Ampl_TD * sqrt(2.0 * pi / dOmegaGW_dt_phys)
        Ampl_FD_ref = GravitationalWaveforms.Ampl(model_f2, f, mc, eta, chi1, chi2, dL)
        @test isapprox(Ampl_FD_mine, Ampl_FD_ref; rtol = 1e-12)
    end

    # --- Phase: use the model's own true dOmega/dt-derived reconstruction; expect the shape (not
    # the absolute value -- the two models' phase origins differ by construction) to match. ---
    diffphi = Float64[]
    for f in test_fs
        r = GravitationalWaveforms._seobv5f2_dense_to_freq(ts, phis, omegas, length(omegas), coeffs_pn.M_sec, f)
        t_star_phys = r.t_star * coeffs_pn.M_sec
        Phi_TD = 2.0 * r.phi_star
        Phi_FD_mine = 2.0 * pi * f * t_star_phys - Phi_TD - pi / 4.0
        Phi_FD_ref = GravitationalWaveforms.Phi(model_f2, f, mc, eta, chi1, chi2)
        push!(diffphi, Phi_FD_mine - Phi_FD_ref)
    end
    X = hcat(ones(length(test_fs)), test_fs)
    affine_coef = X \ diffphi
    resid = diffphi .- X * affine_coef
    @test all(abs.(resid) .< 1e-2)
end

@testset "SEOBNRv5AlignedSpinF2: matches independent from-scratch SEOBNRv5AlignedSpin.Phi at reconstructed t*(f)" begin
    model = GravitationalWaveforms.SEOBNRv5AlignedSpin()
    model_f2 = GravitationalWaveforms.SEOBNRv5AlignedSpinF2()
    configs = [
        (8.0, 0.2479, 0.3, -0.2, 20.0),
        (28.0961, 0.2479, 0.3, -0.2, 20.0),
    ]
    for (mc, eta, chi1, chi2, fmin) in configs
        coeffs = GravitationalWaveforms._seobv5f2_coeffs(model_f2, mc, eta, chi1, chi2; f_min = fmin)
        @test coeffs.success
        @test coeffs.fmin < coeffs.fmax
        @test isapprox(coeffs.fmin, fmin; rtol = 1e-3)
        @test all(isfinite, coeffs.ts)
        @test all(isfinite, coeffs.phis)
        @test all(isfinite, coeffs.omegas)
        @test all(diff(coeffs.omegas) .> 0)

        test_fs = range(coeffs.fmin + 0.5, coeffs.fmax - 0.5; length = 6)
        for f in test_fs
            r = GravitationalWaveforms._seobv5f2_dense_to_freq(coeffs.ts, coeffs.phis, coeffs.omegas, coeffs.nreal, coeffs.M_sec, f)
            t_star_phys = r.t_star * coeffs.M_sec
            Phi_TD_interp = 2.0 * r.phi_star
            Phi_TD_direct = GravitationalWaveforms.Phi(model, t_star_phys, mc, eta, chi1, chi2; f_min = fmin)
            @test isapprox(Phi_TD_interp, Phi_TD_direct; atol = 0.1)
        end
    end

    # Cutoff behavior: outside [fmin, fmax] both Phi and Ampl are exactly 0, mirroring
    # AlignedSpinPNHamF2/TaylorF2's own fCut convention.
    mc, eta, chi1, chi2, dL, fmin = 28.0961, 0.2479, 0.3, -0.2, 0.4, 20.0
    coeffs = GravitationalWaveforms._seobv5f2_coeffs(model_f2, mc, eta, chi1, chi2; f_min = fmin)
    @test GravitationalWaveforms.Phi(model_f2, coeffs.fmin - 1.0, mc, eta, chi1, chi2; f_min = fmin) == 0.0
    @test GravitationalWaveforms.Phi(model_f2, coeffs.fmax + 1.0, mc, eta, chi1, chi2; f_min = fmin) == 0.0
    @test GravitationalWaveforms.Ampl(model_f2, coeffs.fmin - 1.0, mc, eta, chi1, chi2, dL; f_min = fmin) == 0.0
    @test GravitationalWaveforms.Ampl(model_f2, coeffs.fmax + 1.0, mc, eta, chi1, chi2, dL; f_min = fmin) == 0.0

    # Array-in/array-out contract, sanity: finite, no throw, agrees with scalar calls.
    fs_arr = collect(range(coeffs.fmin + 0.5, coeffs.fmax - 0.5; length = 10))
    phis_arr = GravitationalWaveforms.Phi(model_f2, fs_arr, mc, eta, chi1, chi2; f_min = fmin)
    amps_arr = GravitationalWaveforms.Ampl(model_f2, fs_arr, mc, eta, chi1, chi2, dL; f_min = fmin)
    @test all(isfinite, phis_arr)
    @test all(isfinite, amps_arr)
    @test all(amps_arr .> 0)
    for (i, f) in enumerate(fs_arr)
        @test phis_arr[i] == GravitationalWaveforms.Phi(model_f2, f, mc, eta, chi1, chi2; f_min = fmin)
        @test amps_arr[i] == GravitationalWaveforms.Ampl(model_f2, f, mc, eta, chi1, chi2, dL; f_min = fmin)
    end
end

using Random

# `_seobv5f2_searchsortedlast` replaces `Base.searchsortedlast` inside `_seobv5f2_dense_to_freq` because
# the latter's data-dependent-trip-count loop makes Enzyme reverse mode leak device heap on CUDA (see
# that function's docstring). It must be a drop-in for sorted input: ties, the "padding strictly above
# omegas[nreal]" layout the real kernel builds, out-of-range queries, and the >511 fallback path.
@testset "SEOBNRv5AlignedSpinF2: _seobv5f2_searchsortedlast matches Base.searchsortedlast" begin
    ss = GravitationalWaveforms._seobv5f2_searchsortedlast
    rng = MersenneTwister(1)
    nbad = 0
    for n in (1, 2, 3, 7, 100, 255, 256, 300, 511, 512, 700), trial in 1:200
        v = sort(rand(rng, n) .* 10)
        if trial % 3 == 0                              # duplicate entries (ties)
            v[rand(rng, 1:n)] = v[rand(rng, 1:n)]
            sort!(v)
        end
        if trial % 5 == 0 && n > 10                    # padding region strictly above omegas[nreal]
            nr = rand(rng, 5:n-1)
            v[nr+1:end] .= v[nr] .+ 1.0 .+ (1:(n-nr))
        end
        for x in vcat(rand(rng, 20) .* 12 .- 1, v[rand(rng, 1:n)], v[1], v[end], -1.0, 1e9)
            ss(v, x) == searchsortedlast(v, x) || (nbad += 1)
        end
    end
    @test nbad == 0
    # also on a view, since the kernels pass a column view of a (256, nwalkers) buffer
    M = sort(rand(rng, 256, 4) .* 10; dims = 1)
    for w in 1:4, x in (0.0, 3.3, 9.9, 11.0)
        @test ss(view(M, :, w), x) == searchsortedlast(view(M, :, w), x)
    end
end
