# Enzyme + KernelAbstractions validation for GravitationalWaveforms.PrecessingSpinPNHam,
# mirroring alignedspinpnham_enzyme_ka_tests.jl's and odetoy_enzyme_ka_tests.jl's structure.
#
# What is differentiated, and why: PrecessingSpinPNHam.jl's own docstring identifies
# `precessingspinpnham_rk4_integrate`/`_psph_rhs` -- the 11-component `(phase,v,LNhat,S1,S2)` core
# -- as what a validation pass should differentiate/kernelize, explicitly NOT `evolve` (which adds
# an `E1`-companion RK4 loop and a `Complex{T}`-valued twisting-up transform,
# `_psph_twist_up22`, on top; see this file's own "Complex{T} in
# evolve" note below). `_psph_coeffs`
# is, like `AlignedSpinPNHam`'s `_asph_coeffs`, a one-time-per-waveform, plain-`Float64`,
# non-`@Tlit`-generic host-side precompute (mass fractions, spin-precession/domega-series
# coefficients, initial state) -- not part of the hot loop -- so, exactly as in
# `alignedspinpnham_enzyme_ka_tests.jl`, `coeffs` is built once via the real `_psph_coeffs` and
# `precessingspinpnham_rk4_integrate` is differentiated w.r.t. one of its own `coeffs` fields
# directly (`eta`, then `wdot3S1O`, the leading spin-orbit `domega/dt` coefficient for body 1).
#
# Complex{T} in evolve: `_psph_twist_up22`'s spin-weighted-spherical-harmonic/Wigner-d twisting-up
# algebra genuinely uses `Complex{T}` (`cis`, `im`, `hAS::Complex{T}`) -- but this is NOT part of
# the 11-component core state's own RHS/integrator (`_psph_core`/`_psph_rhs` are exactly the same
# real-valued FieldVector machinery as AlignedSpinPNHam/ODEToy) and is explicitly out of this
# file's own documented Enzyme/KernelAbstractions readiness scope (see PrecessingSpinPNHam.jl's
# docstring's "Enzyme/KernelAbstractions readiness" section, which only discusses
# `SVector`+`cross` allocation-freedom for the core state, not `evolve`). So `evolve` itself is not
# exercised here -- only `precessingspinpnham_rk4_integrate`, consistent with the scope this file's
# own docstring sets and with this validation pass's general policy of differentiating exactly the
# function each model's own docstring names as the Enzyme target, not guessing a broader one.
#
# A real, characterized Enzyme finding carried over from alignedspinpnham_enzyme_ka_tests.jl (see
# that file's own comment for the full writeup): passing the full `coeffs` NamedTuple (which
# carries a `T::DataType` field from `_psph_coeffs`) as an explicit `Enzyme.Const(coeffs)`
# argument to `Enzyme.autodiff` raises `EnzymeRuntimeActivityError: ... mismatched activity
# between its rooted and non-rooted fields` -- a DataType field mixed with Float64 fields in one
# struct/NamedTuple argument. Fix, applied throughout this file's KA-kernel testsets (not the
# plain `Enzyme.gradient`-via-closure testset above, which is unaffected -- closing over `coeffs`
# rather than passing it as an explicit autodiff argument never hits this): pass only a
# DataType-free `coeffs_min` (exactly the fields `_psph_core` reads) plus the initial-condition
# scalars as separate plain arguments.

using Enzyme
using KernelAbstractions

const _PSPH_M1 = 30.0
const _PSPH_M2 = 20.0
const _PSPH_CHI1 = (0.3, 0.1, 0.2)
const _PSPH_CHI2 = (-0.1, 0.2, 0.4)
const _PSPH_FREF = 20.0
const _PSPH_DT = 1.0 / 4096.0
const _PSPH_T = 0.03

@testset "PrecessingSpinPNHam: Enzyme.gradient through precessingspinpnham_rk4_integrate matches finite differences" begin
    coeffs = GravitationalWaveforms._psph_coeffs(PrecessingSpinPNHam(), _PSPH_M1, _PSPH_M2,
        _PSPH_CHI1..., _PSPH_CHI2...; f_ref = _PSPH_FREF)
    nsteps = round(Int, _PSPH_T / _PSPH_DT)
    dt_hat = _PSPH_DT / coeffs.M_sec

    @testset "w.r.t. coeffs.eta" begin
        function final_phase_eta(eta_val)
            c2 = merge(coeffs, (eta = eta_val,))
            T = typeof(eta_val)
            state0 = PrecessingSpinPNHamState(zero(T), coeffs.v0,
                coeffs.LNhx0, coeffs.LNhy0, coeffs.LNhz0,
                coeffs.S1x0, coeffs.S1y0, coeffs.S1z0,
                coeffs.S2x0, coeffs.S2y0, coeffs.S2z0)
            final = precessingspinpnham_rk4_integrate(state0, dt_hat, nsteps, c2)
            return 2 * final.phase
        end

        g = Enzyme.gradient(Enzyme.Reverse, final_phase_eta, coeffs.eta)

        h = 1.0e-6 * coeffs.eta
        fd = (final_phase_eta(coeffs.eta + h) - final_phase_eta(coeffs.eta - h)) / (2h)

        @test isfinite(g[1])
        @test isapprox(g[1], fd; rtol = 1e-4, atol = 1e-6)
    end

    @testset "w.r.t. coeffs.wdot3S1O (leading spin-orbit domega/dt coefficient)" begin
        function final_phase_wdot3S1O(w_val)
            c2 = merge(coeffs, (wdot3S1O = w_val,))
            T = typeof(w_val)
            state0 = PrecessingSpinPNHamState(zero(T), coeffs.v0,
                coeffs.LNhx0, coeffs.LNhy0, coeffs.LNhz0,
                coeffs.S1x0, coeffs.S1y0, coeffs.S1z0,
                coeffs.S2x0, coeffs.S2y0, coeffs.S2z0)
            final = precessingspinpnham_rk4_integrate(state0, dt_hat, nsteps, c2)
            return 2 * final.phase
        end

        g = Enzyme.gradient(Enzyme.Reverse, final_phase_wdot3S1O, coeffs.wdot3S1O)

        h = 1.0e-6 * max(abs(coeffs.wdot3S1O), 1.0)
        fd = (final_phase_wdot3S1O(coeffs.wdot3S1O + h) - final_phase_wdot3S1O(coeffs.wdot3S1O - h)) / (2h)

        @test isfinite(g[1])
        @test isapprox(g[1], fd; rtol = 1e-4, atol = 1e-6)
    end
end

# --- KernelAbstractions.jl @kernel compilation, CPU backend. Batches over an array of `eta`
# values, holding everything else fixed. Top-level, non-closure functions throughout (see
# odetoy_enzyme_ka_tests.jl's own comment for why -- boxed @testset-local closures break Enzyme's
# activity analysis when differentiated through a KA kernel). `coeffs_min` (DataType-free, only
# the fields `_psph_core` reads) plus the 10 initial-state scalars are passed instead of the full
# `coeffs` NamedTuple -- see file header comment for the Enzyme finding this works around. ---

@kernel function _psph_ka_test_kernel!(out, etas, coeffs_min, v0, LNhx0, LNhy0, LNhz0,
        S1x0, S1y0, S1z0, S2x0, S2y0, S2z0, dt_hat, nsteps)
    i = @index(Global)
    T = eltype(etas)
    c2 = merge(coeffs_min, (eta = etas[i],))
    state0 = PrecessingSpinPNHamState(zero(T), T(v0), T(LNhx0), T(LNhy0), T(LNhz0),
        T(S1x0), T(S1y0), T(S1z0), T(S2x0), T(S2y0), T(S2z0))
    final = precessingspinpnham_rk4_integrate(state0, dt_hat, nsteps, c2)
    out[i] = 2 * final.phase
end

function _psph_ka_test_run!(out, etas, backend, coeffs_min, v0, LNhx0, LNhy0, LNhz0,
        S1x0, S1y0, S1z0, S2x0, S2y0, S2z0, dt_hat, nsteps, n)
    kernel! = _psph_ka_test_kernel!(backend)
    kernel!(out, etas, coeffs_min, v0, LNhx0, LNhy0, LNhz0, S1x0, S1y0, S1z0, S2x0, S2y0, S2z0,
        dt_hat, nsteps; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return sum(out)
end

function _psph_final_phase_wrt_eta_ref(eta_val, coeffs_min, v0, LNhx0, LNhy0, LNhz0,
        S1x0, S1y0, S1z0, S2x0, S2y0, S2z0, dt_hat, nsteps)
    c2 = merge(coeffs_min, (eta = eta_val,))
    T = typeof(eta_val)
    state0 = PrecessingSpinPNHamState(zero(T), T(v0), T(LNhx0), T(LNhy0), T(LNhz0),
        T(S1x0), T(S1y0), T(S1z0), T(S2x0), T(S2y0), T(S2z0))
    final = precessingspinpnham_rk4_integrate(state0, dt_hat, nsteps, c2)
    return 2 * final.phase
end

function _psph_coeffs_min(coeffs)
    return (eta = coeffs.eta, S1dot3 = coeffs.S1dot3, S2dot3 = coeffs.S2dot3,
        wdotnewt = coeffs.wdotnewt, wdotcoeff2 = coeffs.wdotcoeff2, wdotcoeff3 = coeffs.wdotcoeff3,
        wdot3S1O = coeffs.wdot3S1O, wdot3S2O = coeffs.wdot3S2O)
end

@testset "PrecessingSpinPNHam: compiles and runs inside a KernelAbstractions.jl @kernel (CPU)" begin
    coeffs = GravitationalWaveforms._psph_coeffs(PrecessingSpinPNHam(), _PSPH_M1, _PSPH_M2,
        _PSPH_CHI1..., _PSPH_CHI2...; f_ref = _PSPH_FREF)
    coeffs_min = _psph_coeffs_min(coeffs)
    nsteps = round(Int, _PSPH_T / _PSPH_DT)
    dt_hat = _PSPH_DT / coeffs.M_sec
    backend = KernelAbstractions.CPU()
    n = 8
    etas = collect(range(0.15, 0.2499, length = n))
    out = zeros(Float64, n)
    ic = (coeffs.v0, coeffs.LNhx0, coeffs.LNhy0, coeffs.LNhz0,
        coeffs.S1x0, coeffs.S1y0, coeffs.S1z0, coeffs.S2x0, coeffs.S2y0, coeffs.S2z0)

    _psph_ka_test_run!(out, etas, backend, coeffs_min, ic..., dt_hat, nsteps, n)

    @test all(isfinite, out)

    expected = [_psph_final_phase_wrt_eta_ref(eta_val, coeffs_min, ic..., dt_hat, nsteps) for eta_val in etas]
    @test out == expected

    @testset "Enzyme.autodiff through the KA kernel itself" begin
        out_buf = zeros(Float64, n)
        dout_buf = zeros(Float64, n)
        deta = zeros(Float64, n)
        Enzyme.autodiff(Enzyme.Reverse, _psph_ka_test_run!, Enzyme.Active,
            Enzyme.Duplicated(out_buf, dout_buf), Enzyme.Duplicated(etas, deta),
            Enzyme.Const(backend), Enzyme.Const(coeffs_min),
            Enzyme.Const(ic[1]), Enzyme.Const(ic[2]), Enzyme.Const(ic[3]), Enzyme.Const(ic[4]),
            Enzyme.Const(ic[5]), Enzyme.Const(ic[6]), Enzyme.Const(ic[7]), Enzyme.Const(ic[8]),
            Enzyme.Const(ic[9]), Enzyme.Const(ic[10]),
            Enzyme.Const(dt_hat), Enzyme.Const(nsteps), Enzyme.Const(n))
        @test all(isfinite, deta)

        fd = similar(deta)
        for i in 1:n
            h = 1.0e-6 * etas[i]
            ep = copy(etas); ep[i] += h
            em = copy(etas); em[i] -= h
            fd[i] = (_psph_ka_test_run!(zeros(n), ep, backend, coeffs_min, ic..., dt_hat, nsteps, n) -
                     _psph_ka_test_run!(zeros(n), em, backend, coeffs_min, ic..., dt_hat, nsteps, n)) / (2h)
        end
        @test isapprox(deta, fd; rtol = 1e-4, atol = 1e-6)
    end
end

# --- KernelAbstractions.jl @kernel compilation, GPU backend (CUDA.jl, a test-only
# extras/targets dependency).
# Skips gracefully if no functional CUDA device is present. ---

const _PSPH_HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch e
    @warn "CUDA.jl not usable on this machine -- skipping PrecessingSpinPNHam GPU KA tests" exception = (e, catch_backtrace())
    false
end

# Top-level (not inside any @testset), used only by the GPU Enzyme.autodiff-through-kernel
# subtest below. See alignedspinpnham_enzyme_ka_tests.jl's own GPU testset comment for the full
# writeup of the two real Enzyme+KernelAbstractions-on-CUDA findings this design works around:
# (1) an on-device `merge` of a Const-captured struct with one Active per-thread scalar makes the
# whole merged value activity-dependent, which a GPU kernel LAUNCH (unlike a CPU one) cannot
# accommodate ("Active kernel arguments not supported on GPU"); (2) even with (1) fixed, a bare
# `Float64` scalar reaching the `kernel!(...)` call gets independently reclassified `Active` by
# KernelAbstractions' CUDA-specific Enzyme rule regardless of the outer autodiff() call's own
# `Const` annotation. Fix for both: every per-thread-varying AND every shared-but-`Float64`-typed
# quantity (`eta` and the whole initial condition / `dt_hat`) is folded into ONE per-thread struct,
# uploaded whole as a `Duplicated` array -- only `nsteps::Int` remains a bare kernel-launch scalar.
@kernel function _psph_ka_gpu_diff_kernel!(out, coeffs_arr, nsteps)
    i = @index(Global)
    c2 = coeffs_arr[i]
    T = typeof(c2.eta)
    state0 = PrecessingSpinPNHamState(zero(T), T(c2.v0), T(c2.LNhx0), T(c2.LNhy0), T(c2.LNhz0),
        T(c2.S1x0), T(c2.S1y0), T(c2.S1z0), T(c2.S2x0), T(c2.S2y0), T(c2.S2z0))
    final = precessingspinpnham_rk4_integrate(state0, c2.dt_hat, nsteps, c2)
    out[i] = 2 * final.phase
end
function _psph_ka_gpu_diff_run!(out, coeffs_arr, backend, nsteps, n)
    kernel! = _psph_ka_gpu_diff_kernel!(backend)
    kernel!(out, coeffs_arr, nsteps; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return sum(out)
end

if _PSPH_HAVE_CUDA
    @testset "PrecessingSpinPNHam: compiles and runs inside a KernelAbstractions.jl @kernel (CUDA GPU)" begin
        coeffs = GravitationalWaveforms._psph_coeffs(PrecessingSpinPNHam(), _PSPH_M1, _PSPH_M2,
            _PSPH_CHI1..., _PSPH_CHI2...; f_ref = _PSPH_FREF)
        coeffs_min = _psph_coeffs_min(coeffs)
        nsteps = round(Int, _PSPH_T / _PSPH_DT)
        dt_hat = _PSPH_DT / coeffs.M_sec
        backend = CUDA.CUDABackend()
        n = 8
        etas = collect(range(0.15, 0.2499, length = n))
        etas_gpu = CUDA.CuArray(etas)
        out_gpu = CUDA.zeros(Float64, n)
        ic = (coeffs.v0, coeffs.LNhx0, coeffs.LNhy0, coeffs.LNhz0,
            coeffs.S1x0, coeffs.S1y0, coeffs.S1z0, coeffs.S2x0, coeffs.S2y0, coeffs.S2z0)

        _psph_ka_test_run!(out_gpu, etas_gpu, backend, coeffs_min, ic..., dt_hat, nsteps, n)
        out = Array(out_gpu)

        @test all(isfinite, out)
        expected = [_psph_final_phase_wrt_eta_ref(eta_val, coeffs_min, ic..., dt_hat, nsteps) for eta_val in etas]
        @test isapprox(out, expected; rtol = 1e-10)
        @test out_gpu isa CUDA.CuArray

        @testset "Enzyme.autodiff through the CUDA KA kernel itself" begin
            # No Complex{T} anywhere in _psph_core/_psph_rhs/precessingspinpnham_rk4_integrate
            # (confirmed directly by reading PrecessingSpinPNHam.jl -- Complex{T} appears only in
            # `evolve`'s own twisting-up transform, deliberately not exercised here, see file
            # header comment) -- so GravitationalWaveBayes.jl's known Complex{T}+Enzyme.Reverse+CUDABackend()
            # TypeAnalysis.cpp crash class was never expected here, and was NOT what was hit; see
            # the top-level comment above this testset's own kernel/run function definitions for
            # the two real (different) Enzyme+KernelAbstractions-on-CUDA findings this test's
            # design works around instead.
            coeffs_arr_host = [merge(coeffs_min, (eta = eta_val, v0 = ic[1], LNhx0 = ic[2],
                LNhy0 = ic[3], LNhz0 = ic[4], S1x0 = ic[5], S1y0 = ic[6], S1z0 = ic[7],
                S2x0 = ic[8], S2y0 = ic[9], S2z0 = ic[10], dt_hat = dt_hat)) for eta_val in etas]
            coeffs_arr_gpu = CUDA.CuArray(coeffs_arr_host)
            zero_c = merge(coeffs_min, (eta = 0.0, S1dot3 = 0.0, S2dot3 = 0.0, wdotnewt = 0.0,
                wdotcoeff2 = 0.0, wdotcoeff3 = 0.0, wdot3S1O = 0.0, wdot3S2O = 0.0,
                v0 = 0.0, LNhx0 = 0.0, LNhy0 = 0.0, LNhz0 = 0.0,
                S1x0 = 0.0, S1y0 = 0.0, S1z0 = 0.0, S2x0 = 0.0, S2y0 = 0.0, S2z0 = 0.0, dt_hat = 0.0))
            dcoeffs_arr_gpu = CUDA.CuArray(fill(zero_c, n))

            out_buf = CUDA.zeros(Float64, n)
            dout_buf = CUDA.zeros(Float64, n)  # NOT ones() -- return activity is already Active
            Enzyme.autodiff(Enzyme.Reverse, _psph_ka_gpu_diff_run!, Enzyme.Active,
                Enzyme.Duplicated(out_buf, dout_buf), Enzyme.Duplicated(coeffs_arr_gpu, dcoeffs_arr_gpu),
                Enzyme.Const(backend), Enzyme.Const(nsteps), Enzyme.Const(n))
            dcoeffs_arr = Array(dcoeffs_arr_gpu)
            deta = [dc.eta for dc in dcoeffs_arr]
            @test all(isfinite, deta)

            fd = similar(deta)
            for i in 1:n
                h = 1.0e-6 * etas[i]
                fd[i] = (_psph_final_phase_wrt_eta_ref(etas[i] + h, coeffs_min, ic..., dt_hat, nsteps) -
                         _psph_final_phase_wrt_eta_ref(etas[i] - h, coeffs_min, ic..., dt_hat, nsteps)) / (2h)
            end
            @test isapprox(deta, fd; rtol = 1e-4, atol = 1e-6)
        end
    end
else
    @testset "PrecessingSpinPNHam: KernelAbstractions.jl @kernel (CUDA GPU) -- skipped" begin
        @test_skip "no functional CUDA device on this machine"
    end
end
