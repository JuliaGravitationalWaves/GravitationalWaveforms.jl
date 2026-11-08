# KernelAbstractions.jl / CUDA.jl validation for GravitationalWaveforms.SEOBNRv5AlignedSpinF2's own
# `_seobv5f2_coeffs` -- the fixed-buffer rewrite (like
# SEOBNRv5AlignedSpinPAFixed.jl's own PA driver rewrite; see that file's module docstring) that replaced its prior
# `Vector{Float64}(undef, n)` allocations (the confirmed, pinpointed CUDA blocker -- binned-kernel
# NUTS itself has no CUDA blocker; this allocation did) with `SVector{NMAX,Float64}` buffers built
# directly from `seobnrv5alignedspin_postadiabatic_dynamics_fixed`'s own already-CUDA-validated
# `t`/`phi` output plus a freshly-computed, uniform-cost `omegas` buffer whose padding region
# (`i>nreal`) is forced strictly increasing so `_seobv5f2_dense_to_freq`'s `searchsortedlast` can
# run over the FULL fixed-size buffer with no dynamic-length array anywhere in the hot path.
#
# Mirrors `seobnrv5alignedspin_pa_fixed_tests.jl`'s own tiered KA structure exactly: CPU KA kernel
# first (always run), CUDA GPU KA kernel second (skipped gracefully if no functional device).
#
# NOT tested here: Enzyme differentiability inside the KA kernel
# (`seobnrv5alignedspinf2_enzyme_ka_tests.jl` already covers Enzyme through the reconstruction hot
# path on a CPU-built `coeffs`, holding table construction outside the AD graph, exactly as
# `SEOBNRv5AlignedSpin`'s own precedent does -- this file is about the KA/CUDA *compilation*
# question for `_seobv5f2_coeffs` itself, not autodiff through it). Full `mc`-through-`_seobv5f2_coeffs`
# Enzyme differentiability is a separate question, unaffected either way by
# this file's results.

using KernelAbstractions

# NMAX=64: this file's own q=1.5-family config (mc~22.5-23.4) gives nreal in the mid-20s (same
# family `seobnrv5alignedspin_pa_fixed_tests.jl` uses at its own NMAX=64), comfortably interior --
# kept small deliberately to keep this file's own compile time tractable (see
# `SEOBNRv5AlignedSpinPAFixed.jl`'s own docstring for the super-linear NMAX-vs-compile-time scaling
# this inherits: NMAX=256 already measured ~28.6s for the PA driver alone).
const _SEOBV5F2_KA_TEST_NMAX = Val(64)

@kernel function _seobv5f2_ka_kernel!(out, mcs, model, eta, chi1, chi2, fmin)
    i = @index(Global)
    coeffs = GravitationalWaveforms._seobv5f2_coeffs(model, mcs[i], eta, chi1, chi2; f_min = fmin, NMAX = _SEOBV5F2_KA_TEST_NMAX)
    f_mid = coeffs.fmin + 0.4 * (coeffs.fmax - coeffs.fmin)
    out[i] = GravitationalWaveforms._seobv5f2_phi_scalar(coeffs, f_mid)
end

function _seobv5f2_ka_loss(mc_val, model, eta, chi1, chi2, fmin)
    coeffs = GravitationalWaveforms._seobv5f2_coeffs(model, mc_val, eta, chi1, chi2; f_min = fmin, NMAX = _SEOBV5F2_KA_TEST_NMAX)
    f_mid = coeffs.fmin + 0.4 * (coeffs.fmax - coeffs.fmin)
    return GravitationalWaveforms._seobv5f2_phi_scalar(coeffs, f_mid)
end

@testset "SEOBNRv5AlignedSpinF2: _seobv5f2_coeffs compiles and runs inside a KernelAbstractions.jl @kernel (CPU)" begin
    model = GravitationalWaveforms.SEOBNRv5AlignedSpinF2()
    eta, chi1, chi2, fmin = 0.24793388429752067, 0.3, -0.2, 20.0
    # Four distinct mc values -> four distinct nreal values, each independently computed per-thread
    # (the actual batched-NUTS-relevant scenario), all comfortably interior of NMAX=64.
    mcs = [22.5, 22.8, 23.1, 23.4]
    n = length(mcs)
    backend = KernelAbstractions.CPU()
    out = zeros(Float64, n)

    kernel! = _seobv5f2_ka_kernel!(backend)
    kernel!(out, mcs, model, eta, chi1, chi2, fmin; ndrange = n)
    KernelAbstractions.synchronize(backend)

    @test all(isfinite, out)
    expected = [_seobv5f2_ka_loss(mc_val, model, eta, chi1, chi2, fmin) for mc_val in mcs]
    @test out == expected
end

# --- CUDA GPU backend. Skips gracefully if no functional CUDA device is present -- identical
# convention to `seobnrv5alignedspin_pa_fixed_tests.jl`'s own GPU testset. This is genuinely new
# information: the ORIGINAL runtime-sized-Vector `_seobv5f2_coeffs` could never even ATTEMPT a GPU
# kernel; this fixed-buffer redesign is specifically constructed to remove that blocker, so this is
# the first real test of whether it does. ---

const _SEOBV5F2_KA_HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch e
    @warn "CUDA.jl not usable on this machine -- skipping SEOBNRv5AlignedSpinF2 GPU KA tests" exception = (e, catch_backtrace())
    false
end

if _SEOBV5F2_KA_HAVE_CUDA
    @testset "SEOBNRv5AlignedSpinF2: _seobv5f2_coeffs compiles and runs inside a KernelAbstractions.jl @kernel (CUDA GPU)" begin
        model = GravitationalWaveforms.SEOBNRv5AlignedSpinF2()
        eta, chi1, chi2, fmin = 0.24793388429752067, 0.3, -0.2, 20.0
        mcs = [22.5, 22.8, 23.1, 23.4]
        n = length(mcs)
        backend = CUDA.CUDABackend()

        mcs_gpu = CUDA.CuArray(mcs)
        out_gpu = CUDA.zeros(Float64, n)

        try
            kernel! = _seobv5f2_ka_kernel!(backend)
            kernel!(out_gpu, mcs_gpu, model, eta, chi1, chi2, fmin; ndrange = n)
            KernelAbstractions.synchronize(backend)
            out = Array(out_gpu)
            @test all(isfinite, out)
            expected = [_seobv5f2_ka_loss(mc_val, model, eta, chi1, chi2, fmin) for mc_val in mcs]
            @test isapprox(out, expected; rtol = 1e-10)
        catch e
            @warn "SEOBNRv5AlignedSpinF2 forward-GPU KA kernel failed -- see this testset's own comment" exception = (e, catch_backtrace())
            rethrow()
        end
    end
else
    @testset "SEOBNRv5AlignedSpinF2: KernelAbstractions.jl @kernel (CUDA GPU) -- skipped" begin
        @test_skip "no functional CUDA device on this machine"
    end
end
