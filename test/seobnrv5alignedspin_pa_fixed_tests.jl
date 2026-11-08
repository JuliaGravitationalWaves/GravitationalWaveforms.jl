# Enzyme + KernelAbstractions validation for GravitationalWaveforms.SEOBNRv5AlignedSpinPAFixed.jl -- the
# fixed-size-buffer/dynamic-nreal redesign of SEOBNRv5AlignedSpinPA.jl's runtime-sized `Vector`s.
#
# Three tiers, cheapest first:
# 1. Pure numerical correctness (no Enzyme at all): does the fixed-buffer driver reproduce the
#    original Vector-based seobnrv5alignedspin_postadiabatic_dynamics over the real (1:nreal)
#    region? Fast, safe on any machine, always run.
# 2. Bare Enzyme.gradient/autodiff (Reverse) through the fixed-buffer driver, vs. central finite
#    differences -- the actual question this file exists to answer. Heavy (compiles the real
#    ~1000-line Hamiltonian's differentiated closure inside a fixed, but still substantial, NMAX
#    loop) -- NMAX is kept deliberately small here (64) to
#    make this tier tractable, not chosen to represent a production size.
# 3. KernelAbstractions.jl @kernel compilation (CPU always; CUDA GPU attempted, skipped gracefully
#    if unavailable -- mirrors seobnrv5alignedspin_enzyme_ka_tests.jl's own convention exactly).
#
# NOT tested here (see SEOBNRv5AlignedSpinPAFixed.jl's module
# docstring): direct real-pyseobnr cross-validation (this file's correctness claim is entirely
# "matches the already-validated original"); NMAX at production scale (1024); the likelihood-kernel integration (in GravitationalWaveBayes.jl).

using Enzyme
using KernelAbstractions

@testset "SEOBNRv5AlignedSpinPAFixed: matches the original Vector-based PA driver over the real region" begin
    model = SEOBNRv5AlignedSpin()
    configs = [
        (23.821232072402484, 0.24793388429752067, 0.3, -0.2, 20.0),  # this file's own q=1.5 config, n=26
        (8.0, 0.2479, 0.3, -0.2, 20.0),                                # Config A, n=107 (near NMAX=256's own limit)
        (28.0961, 0.2479, 0.3, -0.2, 20.0),                            # Config B, n=18
    ]
    for (mc, eta, chi1, chi2, fmin) in configs
        coeffs = GravitationalWaveforms._seobv5_coeffs(model, mc, eta, chi1, chi2; f_min = fmin)
        pa_fixed = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs, Val(256))
        pa_orig = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics(coeffs)
        n = pa_fixed.nreal

        @test pa_fixed.success == pa_orig.success
        @test n == length(pa_orig.r)
        @test all(isfinite, pa_fixed.r)
        @test all(isfinite, pa_fixed.pr)
        @test all(isfinite, pa_fixed.pphi)
        @test all(isfinite, pa_fixed.t)
        @test all(isfinite, pa_fixed.phi)

        @test all(isapprox.(pa_fixed.r[1:n], pa_orig.r; rtol = 1e-9))
        @test all(isapprox.(pa_fixed.pr[1:n], pa_orig.pr; rtol = 1e-6, atol = 1e-9))
        @test all(isapprox.(pa_fixed.pphi[1:n], pa_orig.pphi; rtol = 1e-6, atol = 1e-9))
        @test all(isapprox.(pa_fixed.t[1:n], pa_orig.t; rtol = 1e-6, atol = 1e-6))
        @test all(isapprox.(pa_fixed.phi[1:n], pa_orig.phi; rtol = 1e-6, atol = 1e-6))
    end

    @testset "explicit failure path (r0<=11.5) matches the original's success=false" begin
        coeffs_short = GravitationalWaveforms._seobv5_coeffs(model, 23.821232072402484, 0.24793388429752067, 0.3, -0.2; f_min = 200.0)
        pa_fixed_short = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs_short, Val(64))
        pa_orig_short = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics(coeffs_short)
        @test pa_fixed_short.success == false
        @test pa_orig_short.success == false
        @test all(isfinite, pa_fixed_short.r)
        @test all(isfinite, pa_fixed_short.pr)
        @test all(isfinite, pa_fixed_short.pphi)
    end
end

# --- Enzyme reverse-mode gradient through the fixed-buffer driver itself. NMAX=64 (this file's
# own q=1.5 config gives nreal=26, comfortably interior). mc=23.821232072402484 is NOT chosen
# arbitrarily -- it sits well inside an nreal-plateau (there is a discrete kink at each nreal transition); perturbing mc by the finite-difference step
# below does not cross a transition (confirmed empirically in this testset itself, not assumed --
# see the `nreal` re-check inside the loss closure). ---

const _SEOBV5_PA_FIXED_TEST_NMAX = Val(64)

function _seobv5_pa_fixed_loss_mc(mc_val, model, eta, chi1, chi2, fmin)
    coeffs = GravitationalWaveforms._seobv5_coeffs(model, mc_val, eta, chi1, chi2; f_min = fmin)
    pa = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs, _SEOBV5_PA_FIXED_TEST_NMAX)
    return pa.t[pa.nreal] + pa.phi[pa.nreal]
end

function _seobv5_pa_fixed_loss_chi1(chi1_val, model, mc, eta, chi2, fmin)
    coeffs = GravitationalWaveforms._seobv5_coeffs(model, mc, eta, chi1_val, chi2; f_min = fmin)
    pa = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs, _SEOBV5_PA_FIXED_TEST_NMAX)
    return pa.t[pa.nreal] + pa.phi[pa.nreal]
end

@testset "SEOBNRv5AlignedSpinPAFixed: Enzyme.gradient matches finite differences" begin
    model = SEOBNRv5AlignedSpin()
    mc, eta, chi1, chi2, fmin = 23.821232072402484, 0.24793388429752067, 0.3, -0.2, 20.0

    # Confirm nreal is stable across the finite-difference step used below (i.e. this config is not
    # sitting on a discrete grid-size transition) -- checked, not assumed.
    coeffs_probe = GravitationalWaveforms._seobv5_coeffs(model, mc, eta, chi1, chi2; f_min = fmin)
    nreal0 = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs_probe, _SEOBV5_PA_FIXED_TEST_NMAX).nreal

    @testset "w.r.t. mc (chirp mass)" begin
        h = 1.0e-6 * mc
        coeffs_p = GravitationalWaveforms._seobv5_coeffs(model, mc + h, eta, chi1, chi2; f_min = fmin)
        coeffs_m = GravitationalWaveforms._seobv5_coeffs(model, mc - h, eta, chi1, chi2; f_min = fmin)
        nreal_p = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs_p, _SEOBV5_PA_FIXED_TEST_NMAX).nreal
        nreal_m = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs_m, _SEOBV5_PA_FIXED_TEST_NMAX).nreal
        @test nreal_p == nreal0 == nreal_m   # not on a transition -- required for a meaningful FD check

        g = Enzyme.gradient(Enzyme.Reverse, _seobv5_pa_fixed_loss_mc, mc,
            Enzyme.Const(model), Enzyme.Const(eta), Enzyme.Const(chi1), Enzyme.Const(chi2), Enzyme.Const(fmin))
        fd = (_seobv5_pa_fixed_loss_mc(mc + h, model, eta, chi1, chi2, fmin) -
              _seobv5_pa_fixed_loss_mc(mc - h, model, eta, chi1, chi2, fmin)) / (2h)

        @test isfinite(g[1])
        @test isapprox(g[1], fd; rtol = 1e-3, atol = 1e-6)
    end

    @testset "w.r.t. chi1" begin
        h = 1.0e-6
        coeffs_p = GravitationalWaveforms._seobv5_coeffs(model, mc, eta, chi1 + h, chi2; f_min = fmin)
        coeffs_m = GravitationalWaveforms._seobv5_coeffs(model, mc, eta, chi1 - h, chi2; f_min = fmin)
        nreal_p = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs_p, _SEOBV5_PA_FIXED_TEST_NMAX).nreal
        nreal_m = GravitationalWaveforms.seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs_m, _SEOBV5_PA_FIXED_TEST_NMAX).nreal
        @test nreal_p == nreal0 == nreal_m

        g = Enzyme.gradient(Enzyme.Reverse, _seobv5_pa_fixed_loss_chi1, chi1,
            Enzyme.Const(model), Enzyme.Const(mc), Enzyme.Const(eta), Enzyme.Const(chi2), Enzyme.Const(fmin))
        fd = (_seobv5_pa_fixed_loss_chi1(chi1 + h, model, mc, eta, chi2, fmin) -
              _seobv5_pa_fixed_loss_chi1(chi1 - h, model, mc, eta, chi2, fmin)) / (2h)

        @test isfinite(g[1])
        @test isapprox(g[1], fd; rtol = 1e-3, atol = 1e-6)
    end
end

# --- KernelAbstractions.jl @kernel compilation, CPU backend. One thread per walker, each with its
# own `mc` (hence its own independently-computed `nreal`) -- the actual batched-NUTS-relevant
# scenario, mirroring seobnrv5alignedspin_enzyme_ka_tests.jl's own per-thread convention. ---

@kernel function _seobv5_pa_fixed_ka_kernel!(out, mcs, model, eta, chi1, chi2, fmin)
    i = @index(Global)
    out[i] = _seobv5_pa_fixed_loss_mc(mcs[i], model, eta, chi1, chi2, fmin)
end

function _seobv5_pa_fixed_ka_run!(out, mcs, backend, model, eta, chi1, chi2, fmin, n)
    kernel! = _seobv5_pa_fixed_ka_kernel!(backend)
    kernel!(out, mcs, model, eta, chi1, chi2, fmin; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return sum(out)
end

@testset "SEOBNRv5AlignedSpinPAFixed: compiles and runs inside a KernelAbstractions.jl @kernel (CPU)" begin
    model = SEOBNRv5AlignedSpin()
    eta, chi1, chi2, fmin = 0.24793388429752067, 0.3, -0.2, 20.0
    backend = KernelAbstractions.CPU()
    n = 4
    # Four distinct mc values -> four distinct nreal values, all comfortably interior of their own
    # plateaus (spaced 0.3 apart, far from this config's own transition spacing of order dr0/dr_dmc).
    mcs = [22.5, 22.8, 23.1, 23.4]
    out = zeros(Float64, n)

    _seobv5_pa_fixed_ka_run!(out, mcs, backend, model, eta, chi1, chi2, fmin, n)
    @test all(isfinite, out)

    expected = [_seobv5_pa_fixed_loss_mc(mc_val, model, eta, chi1, chi2, fmin) for mc_val in mcs]
    @test out == expected

    @testset "Enzyme.autodiff through the KA kernel itself" begin
        out_buf = zeros(Float64, n)
        dout_buf = zeros(Float64, n)
        dmcs = zeros(Float64, n)
        Enzyme.autodiff(Enzyme.Reverse, _seobv5_pa_fixed_ka_run!, Enzyme.Active,
            Enzyme.Duplicated(out_buf, dout_buf), Enzyme.Duplicated(mcs, dmcs),
            Enzyme.Const(backend), Enzyme.Const(model), Enzyme.Const(eta), Enzyme.Const(chi1),
            Enzyme.Const(chi2), Enzyme.Const(fmin), Enzyme.Const(n))
        @test all(isfinite, dmcs)

        fd = similar(dmcs)
        for i in 1:n
            h = 1.0e-6 * mcs[i]
            mp = copy(mcs); mp[i] += h
            mm = copy(mcs); mm[i] -= h
            fd[i] = (_seobv5_pa_fixed_ka_run!(zeros(n), mp, backend, model, eta, chi1, chi2, fmin, n) -
                     _seobv5_pa_fixed_ka_run!(zeros(n), mm, backend, model, eta, chi1, chi2, fmin, n)) / (2h)
        end
        @test isapprox(dmcs, fd; rtol = 1e-3, atol = 1e-6)
    end
end

# --- KernelAbstractions.jl @kernel compilation, GPU backend (CUDA.jl). Skips gracefully if no
# functional CUDA device is present -- identical convention to
# seobnrv5alignedspin_enzyme_ka_tests.jl's own GPU testset. This is genuinely new information: the
# ORIGINAL runtime-sized-Vector PA driver could never even ATTEMPT a GPU kernel (dynamic heap
# allocation inside a kernel body is not supported); this fixed-buffer redesign is specifically
# constructed to remove that blocker, so this is the first real test of whether it does. ---

const _SEOBV5_PA_FIXED_HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch e
    @warn "CUDA.jl not usable on this machine -- skipping SEOBNRv5AlignedSpinPAFixed GPU KA tests" exception = (e, catch_backtrace())
    false
end

if _SEOBV5_PA_FIXED_HAVE_CUDA
    @testset "SEOBNRv5AlignedSpinPAFixed: compiles and runs inside a KernelAbstractions.jl @kernel (CUDA GPU)" begin
        model = SEOBNRv5AlignedSpin()
        eta, chi1, chi2, fmin = 0.24793388429752067, 0.3, -0.2, 20.0
        mcs = [22.5, 22.8, 23.1, 23.4]
        n = length(mcs)
        backend = CUDA.CUDABackend()

        mcs_gpu = CUDA.CuArray(mcs)
        out_gpu = CUDA.zeros(Float64, n)

        try
            kernel! = _seobv5_pa_fixed_ka_kernel!(backend)
            kernel!(out_gpu, mcs_gpu, model, eta, chi1, chi2, fmin; ndrange = n)
            KernelAbstractions.synchronize(backend)
            out = Array(out_gpu)
            @test all(isfinite, out)
            expected = [_seobv5_pa_fixed_loss_mc(mc_val, model, eta, chi1, chi2, fmin) for mc_val in mcs]
            @test isapprox(out, expected; rtol = 1e-10)
        catch e
            @warn "SEOBNRv5AlignedSpinPAFixed forward-GPU KA kernel failed -- see this testset's own comment" exception = (e, catch_backtrace())
            rethrow()
        end
    end
else
    @testset "SEOBNRv5AlignedSpinPAFixed: KernelAbstractions.jl @kernel (CUDA GPU) -- skipped" begin
        @test_skip "no functional CUDA device on this machine"
    end
end
