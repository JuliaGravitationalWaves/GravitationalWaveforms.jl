# Enzyme + KernelAbstractions validation for GravitationalWaveforms.AlignedSpinPNHam,
# mirroring `odetoy_enzyme_ka_tests.jl`'s structure/idioms exactly (top-level, non-closure
# functions for anything that gets Enzyme.autodiff'd through a KA kernel -- see that file's own
# comment for exactly why closures over @testset-local variables get boxed and break Enzyme's
# activity analysis; plain `Enzyme.gradient` calls stay as in-testset closures, matching
# `odetoy_enzyme_ka_tests.jl`'s own first testset, which that file's comments confirm does NOT hit
# the boxing issue).
#
# What is differentiated, and why: AlignedSpinPNHam.jl's own docstring identifies
# `alignedspinpnham_rk4_integrate`/`_asph_rhs` (not `Phi`/`_asph_coeffs`) as "the function a later
# Enzyme/KernelAbstractions validation pass will differentiate/kernelize" -- deliberately, since
# `_asph_coeffs` computes its flux-series/spin coefficients and the circular-orbit initial
# condition via a fixed, plain-`Float64`, non-`@Tlit`-generic pipeline (that file's own docstring:
# "the flux coefficient formulas below have no natural @Tlit-friendly form...") that is a
# one-time-per-waveform host-side precompute, not part of the hot loop. So here, exactly like
# `odetoy_enzyme_ka_tests.jl`'s own `Mc_sec`, a `coeffs` NamedTuple is built ONCE (ordinary
# Float64, via the real `_asph_coeffs`) and then `alignedspinpnham_rk4_integrate` itself is
# differentiated w.r.t. one of its `coeffs` fields directly (`eta`, then `SOcoef`) -- a real,
# non-trivial exercise of the actual hot-path RHS (`_asph_hamiltonian_partials`/`_asph_flux`,
# `cbrt`/`log`/`max`/`eps` included), matching finite differences to the same tolerance bar
# `odetoy_enzyme_ka_tests.jl` uses.

using Enzyme
using KernelAbstractions

const _ASPH_MC = 23.821232072402484
const _ASPH_ETA = 0.24793388429752067
const _ASPH_CHI1 = 0.3
const _ASPH_CHI2 = -0.2
const _ASPH_FMIN = 20.0
const _ASPH_DT = 1.0 / 4096.0
const _ASPH_T = 0.03   # well inside the ~0.07-0.075s PN-Hamiltonian-breakdown safe window

@testset "AlignedSpinPNHam: Enzyme.gradient through alignedspinpnham_rk4_integrate matches finite differences" begin
    coeffs = GravitationalWaveforms._asph_coeffs(AlignedSpinPNHam(), _ASPH_MC, _ASPH_ETA, _ASPH_CHI1, _ASPH_CHI2;
        f_min = _ASPH_FMIN)
    nsteps = round(Int, _ASPH_T / _ASPH_DT)
    dt_hat = _ASPH_DT / coeffs.M_sec

    @testset "w.r.t. coeffs.eta" begin
        function final_phase_eta(eta_val)
            c2 = merge(coeffs, (eta = eta_val,))
            state0 = AlignedSpinPNHamState(coeffs.r0, zero(eta_val), zero(eta_val), coeffs.pphi0)
            final = alignedspinpnham_rk4_integrate(state0, dt_hat, nsteps, c2)
            return 2 * final.phi
        end

        g = Enzyme.gradient(Enzyme.Reverse, final_phase_eta, coeffs.eta)

        h = 1.0e-6 * coeffs.eta
        fd = (final_phase_eta(coeffs.eta + h) - final_phase_eta(coeffs.eta - h)) / (2h)

        @test isfinite(g[1])
        @test isapprox(g[1], fd; rtol = 1e-4, atol = 1e-6)
    end

    @testset "w.r.t. coeffs.SOcoef (spin-orbit Hamiltonian coefficient)" begin
        function final_phase_SOcoef(SOcoef_val)
            c2 = merge(coeffs, (SOcoef = SOcoef_val,))
            T = typeof(SOcoef_val)
            state0 = AlignedSpinPNHamState(coeffs.r0, zero(T), zero(T), coeffs.pphi0)
            final = alignedspinpnham_rk4_integrate(state0, dt_hat, nsteps, c2)
            return 2 * final.phi
        end

        g = Enzyme.gradient(Enzyme.Reverse, final_phase_SOcoef, coeffs.SOcoef)

        h = 1.0e-6 * max(abs(coeffs.SOcoef), 1.0)
        fd = (final_phase_SOcoef(coeffs.SOcoef + h) - final_phase_SOcoef(coeffs.SOcoef - h)) / (2h)

        @test isfinite(g[1])
        @test isapprox(g[1], fd; rtol = 1e-4, atol = 1e-6)
    end
end

# --- KernelAbstractions.jl @kernel compilation, CPU backend. Batches over an array of `eta`
# values (mirroring odetoy_enzyme_ka_tests.jl's `mc_secs` array), holding everything else in
# `coeffs` fixed. Top-level, non-closure functions throughout (see file header comment).
#
# A real, characterized Enzyme finding hit while building this test (not hidden): passing the
# full `coeffs` NamedTuple straight through as an explicit `Enzyme.Const(coeffs)` argument to
# `Enzyme.autodiff` (as opposed to merely closing over it, which the plain `Enzyme.gradient`
# testset above does and which works fine) raises
# `EnzymeRuntimeActivityError: ... Custom rule was passed an argument with mismatched activity
# between its rooted and non-rooted fields ... Non-rooted types: (Float64 x19)
# (differentiable/active) - Rooted types: (DataType,) (constant/inactive)`. The cause: `coeffs`
# (from `_asph_coeffs`) carries a `T::DataType` field (`T = Float64`) alongside its many `Float64`
# fields -- Enzyme's static activity analysis refuses to treat a struct/NamedTuple argument as
# uniformly Const when one of its fields is a "rooted" type (`DataType`, a boxed/pointer-like
# value) and the rest are plain (`Float64`) bits, even though every field truly is constant here.
# Fix: strip the `DataType` field before passing a coeffs-like NamedTuple as an explicit
# Enzyme.autodiff argument -- a `coeffs_min` containing only the fields `_asph_rhs` actually reads
# (`eta`, `SOcoef`, `SScoef`, `fluxcoeffs`), all-Float64/NamedTuple-of-Float64, has no such mixed
# rooting and differentiates cleanly (confirmed below). This is purely an Enzyme argument-passing
# shape issue, not a physics bug -- the identical computation, expressed as a closure over the
# same (T-carrying) `coeffs`, already differentiated correctly above.

@kernel function _asph_ka_test_kernel!(out, etas, coeffs_min, r0, pphi0, dt_hat, nsteps)
    i = @index(Global)
    T = eltype(etas)
    c2 = merge(coeffs_min, (eta = etas[i],))
    state0 = AlignedSpinPNHamState(T(r0), zero(T), zero(T), T(pphi0))
    final = alignedspinpnham_rk4_integrate(state0, dt_hat, nsteps, c2)
    out[i] = 2 * final.phi
end

function _asph_ka_test_run!(out, etas, backend, coeffs_min, r0, pphi0, dt_hat, nsteps, n)
    kernel! = _asph_ka_test_kernel!(backend)
    kernel!(out, etas, coeffs_min, r0, pphi0, dt_hat, nsteps; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return sum(out)
end

# Plain (non-kernel, non-Enzyme) reference used only for the "kernel computes the same thing"
# cross-check below.
function _asph_final_phase_wrt_eta_ref(eta_val, coeffs_min, r0, pphi0, dt_hat, nsteps)
    c2 = merge(coeffs_min, (eta = eta_val,))
    T = typeof(eta_val)
    state0 = AlignedSpinPNHamState(T(r0), zero(T), zero(T), T(pphi0))
    final = alignedspinpnham_rk4_integrate(state0, dt_hat, nsteps, c2)
    return 2 * final.phi
end

@testset "AlignedSpinPNHam: compiles and runs inside a KernelAbstractions.jl @kernel (CPU)" begin
    coeffs = GravitationalWaveforms._asph_coeffs(AlignedSpinPNHam(), _ASPH_MC, _ASPH_ETA, _ASPH_CHI1, _ASPH_CHI2;
        f_min = _ASPH_FMIN)
    coeffs_min = (eta = coeffs.eta, SOcoef = coeffs.SOcoef, SScoef = coeffs.SScoef,
        fluxcoeffs = coeffs.fluxcoeffs)
    nsteps = round(Int, _ASPH_T / _ASPH_DT)
    dt_hat = _ASPH_DT / coeffs.M_sec
    backend = KernelAbstractions.CPU()
    n = 8
    etas = collect(range(0.15, 0.2499, length = n))  # stays in (0,0.25], AlignedSpinPNHam's own domain
    out = zeros(Float64, n)

    _asph_ka_test_run!(out, etas, backend, coeffs_min, coeffs.r0, coeffs.pphi0, dt_hat, nsteps, n)

    @test all(isfinite, out)

    expected = [_asph_final_phase_wrt_eta_ref(eta_val, coeffs_min, coeffs.r0, coeffs.pphi0, dt_hat, nsteps) for eta_val in etas]
    @test out == expected

    @testset "Enzyme.autodiff through the KA kernel itself" begin
        out_buf = zeros(Float64, n)
        dout_buf = zeros(Float64, n)
        deta = zeros(Float64, n)
        Enzyme.autodiff(Enzyme.Reverse, _asph_ka_test_run!, Enzyme.Active,
            Enzyme.Duplicated(out_buf, dout_buf), Enzyme.Duplicated(etas, deta),
            Enzyme.Const(backend), Enzyme.Const(coeffs_min), Enzyme.Const(coeffs.r0), Enzyme.Const(coeffs.pphi0),
            Enzyme.Const(dt_hat), Enzyme.Const(nsteps), Enzyme.Const(n))
        @test all(isfinite, deta)

        fd = similar(deta)
        for i in 1:n
            h = 1.0e-6 * etas[i]
            ep = copy(etas); ep[i] += h
            em = copy(etas); em[i] -= h
            fd[i] = (_asph_ka_test_run!(zeros(n), ep, backend, coeffs_min, coeffs.r0, coeffs.pphi0, dt_hat, nsteps, n) -
                     _asph_ka_test_run!(zeros(n), em, backend, coeffs_min, coeffs.r0, coeffs.pphi0, dt_hat, nsteps, n)) / (2h)
        end
        @test isapprox(deta, fd; rtol = 1e-4, atol = 1e-6)
    end
end

# --- KernelAbstractions.jl @kernel compilation, GPU backend (CUDA.jl, added as a test-only
# extras/targets dependency). Skips gracefully (not an error) if no
# functional CUDA device is present, mirroring how a machine without a GPU should still be able to
# run `Pkg.test()` on the CPU-only testsets above. ---

const _ASPH_HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch e
    @warn "CUDA.jl not usable on this machine -- skipping AlignedSpinPNHam GPU KA tests" exception = (e, catch_backtrace())
    false
end

# Top-level (not inside any @testset -- see file header / odetoy_enzyme_ka_tests.jl's own comment
# for why), used only by the GPU Enzyme.autodiff-through-kernel subtest below: a per-thread
# `coeffs`-like value read directly from a `Duplicated` array element, with NO on-device
# `merge`/construction mixing Const and Active pieces -- see that subtest's own comment for the
# first real Enzyme+KernelAbstractions GPU finding this design avoids. `r0`/`pphi0`/`dt_hat` are
# folded into the SAME per-thread struct (as fields whose shadow just stays zero) rather than
# passed as separate bare `Float64` arguments to the kernel launch -- see that subtest's comment
# for the SECOND, distinct finding this additionally works around (a bare `Float64` reaching the
# `kernel!(...)` call, even one the outer `Enzyme.autodiff` call marks `Const`, gets independently
# reclassified `Active` by KernelAbstractions' own CUDA-specific Enzyme rule, which then refuses
# it) -- so the only scalar left in the kernel launch's own argument list is `nsteps::Int`, a type
# Enzyme never treats as differentiable regardless of dataflow.
@kernel function _asph_ka_gpu_diff_kernel!(out, coeffs_arr, nsteps)
    i = @index(Global)
    c2 = coeffs_arr[i]
    T = typeof(c2.eta)
    state0 = AlignedSpinPNHamState(T(c2.r0), zero(T), zero(T), T(c2.pphi0))
    final = alignedspinpnham_rk4_integrate(state0, c2.dt_hat, nsteps, c2)
    out[i] = 2 * final.phi
end
function _asph_ka_gpu_diff_run!(out, coeffs_arr, backend, nsteps, n)
    kernel! = _asph_ka_gpu_diff_kernel!(backend)
    kernel!(out, coeffs_arr, nsteps; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return sum(out)
end

if _ASPH_HAVE_CUDA
    @testset "AlignedSpinPNHam: compiles and runs inside a KernelAbstractions.jl @kernel (CUDA GPU)" begin
        coeffs = GravitationalWaveforms._asph_coeffs(AlignedSpinPNHam(), _ASPH_MC, _ASPH_ETA, _ASPH_CHI1, _ASPH_CHI2;
            f_min = _ASPH_FMIN)
        coeffs_min = (eta = coeffs.eta, SOcoef = coeffs.SOcoef, SScoef = coeffs.SScoef,
            fluxcoeffs = coeffs.fluxcoeffs)
        nsteps = round(Int, _ASPH_T / _ASPH_DT)
        dt_hat = _ASPH_DT / coeffs.M_sec
        backend = CUDA.CUDABackend()
        n = 8
        etas = collect(range(0.15, 0.2499, length = n))
        etas_gpu = CUDA.CuArray(etas)
        out_gpu = CUDA.zeros(Float64, n)

        _asph_ka_test_run!(out_gpu, etas_gpu, backend, coeffs_min, coeffs.r0, coeffs.pphi0, dt_hat, nsteps, n)
        out = Array(out_gpu)

        @test all(isfinite, out)
        expected = [_asph_final_phase_wrt_eta_ref(eta_val, coeffs_min, coeffs.r0, coeffs.pphi0, dt_hat, nsteps) for eta_val in etas]
        @test isapprox(out, expected; rtol = 1e-10)
        @test out_gpu isa CUDA.CuArray

        @testset "Enzyme.autodiff through the CUDA KA kernel itself" begin
            # No Complex{T} anywhere in _asph_rhs/_asph_hamiltonian_partials/_asph_flux (confirmed
            # directly by reading AlignedSpinPNHam.jl) -- so GravitationalWaveBayes.jl's known
            # Complex{T}+Enzyme.Reverse+CUDABackend() TypeAnalysis.cpp crash class was never
            # expected here, and was NOT what was hit.
            #
            # A second real, distinct Enzyme+KernelAbstractions finding WAS hit while building
            # this test (not hidden): the CPU KA-kernel testset above builds each thread's
            # per-eta `coeffs` by `merge`-ing a Const-captured `coeffs_min` with one genuinely
            # `Duplicated`/active per-thread scalar (`etas[i]`) INSIDE the kernel body, and Enzyme
            # differentiates that cleanly on the CPU backend. The identical pattern on the CUDA
            # backend instead raises `error("Active kernel arguments not supported on GPU")` from
            # `KernelAbstractions/ext/EnzymeCore08Ext.jl`'s own `augmented_primal` -- a deliberate,
            # informative guard, not a crash. Root cause: building a new immutable NamedTuple from
            # one active scalar plus several Const fields makes the *whole merged value* activity-
            # dependent from Enzyme's point of view; a CPU reverse pass can carry that via an
            # ordinary register-based (`Active`) derivative, but a GPU kernel LAUNCH has no
            # register-return channel across the host/device boundary -- only `Const`/`Duplicated`
            # (device-memory-backed) argument kinds are supported for an actual `@kernel` call on
            # CUDABackend, so KernelAbstractions' own extension explicitly refuses the `Active`
            # case rather than mishandling it. This is a real, GPU-specific Enzyme/KernelAbstractions
            # shape constraint (not a physics bug, and not the known Complex{T} crash class) --
            # documented here since it is exactly the kind of finding this validation pass exists
            # to surface.
            #
            # Third finding, on top of the two documented above: even with (1)'s fix applied (no
            # on-device merge), a plain `Float64` scalar (`r0`/`pphi0`/`dt_hat`) reaching the
            # `kernel!(...)` call -- even though the OUTER `Enzyme.autodiff` call marks it
            # `Const` -- gets independently reclassified `Active` by KernelAbstractions' CUDA-
            # specific Enzyme rule for the kernel launch itself (confirmed directly: the
            # `augmented_primal` error message's own argument-type list showed
            # `Active{Float64}, Active{Float64}, Active{Float64}` for exactly `r0`/`pphi0`/
            # `dt_hat`, despite both being passed as `Enzyme.Const(...)` at the autodiff() call
            # site) -- i.e. Enzyme's activity inference for a custom-rule call boundary like this
            # one is NOT simply inherited from an outer, already-resolved annotation. Fix: fold
            # `r0`/`pphi0`/`dt_hat` as extra fields into the SAME per-thread `Duplicated` struct
            # (their shadow just stays zero) so no bare differentiable-typed scalar reaches the
            # kernel-launch argument list at all -- only `nsteps::Int` remains, a type Enzyme never
            # treats as differentiable regardless of dataflow, and that one already resolved to
            # `Const` correctly in every attempt above.
            coeffs_arr_host = [merge(coeffs_min, (eta = eta_val, r0 = coeffs.r0, pphi0 = coeffs.pphi0,
                dt_hat = dt_hat)) for eta_val in etas]
            coeffs_arr_gpu = CUDA.CuArray(coeffs_arr_host)
            zero_c = merge(coeffs_min, (eta = 0.0, SOcoef = 0.0, SScoef = 0.0,
                fluxcoeffs = map(zero, coeffs_min.fluxcoeffs), r0 = 0.0, pphi0 = 0.0, dt_hat = 0.0))
            dcoeffs_arr_gpu = CUDA.CuArray(fill(zero_c, n))

            out_buf = CUDA.zeros(Float64, n)
            dout_buf = CUDA.zeros(Float64, n)  # NOT ones() -- return activity is already Active (see below)
            Enzyme.autodiff(Enzyme.Reverse, _asph_ka_gpu_diff_run!, Enzyme.Active,
                Enzyme.Duplicated(out_buf, dout_buf), Enzyme.Duplicated(coeffs_arr_gpu, dcoeffs_arr_gpu),
                Enzyme.Const(backend), Enzyme.Const(nsteps), Enzyme.Const(n))
            dcoeffs_arr = Array(dcoeffs_arr_gpu)
            deta = [dc.eta for dc in dcoeffs_arr]
            @test all(isfinite, deta)

            fd = similar(deta)
            for i in 1:n
                h = 1.0e-6 * etas[i]
                fd[i] = (_asph_final_phase_wrt_eta_ref(etas[i] + h, coeffs_min, coeffs.r0, coeffs.pphi0, dt_hat, nsteps) -
                         _asph_final_phase_wrt_eta_ref(etas[i] - h, coeffs_min, coeffs.r0, coeffs.pphi0, dt_hat, nsteps)) / (2h)
            end
            @test isapprox(deta, fd; rtol = 1e-4, atol = 1e-6)
        end
    end
else
    @testset "AlignedSpinPNHam: KernelAbstractions.jl @kernel (CUDA GPU) -- skipped" begin
        @test_skip "no functional CUDA device on this machine"
    end
end
