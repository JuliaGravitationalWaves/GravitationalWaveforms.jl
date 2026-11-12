# Enzyme + KernelAbstractions validation for GravitationalWaveforms.ODEToy. This is the primary thing this scaffolding model exists to
# prove: that `StaticArrays`-based state + a fixed-step, allocation-free RK4 loop (see
# ODEToy.jl's `_odetoy_rhs`/`_odetoy_rk4_step`/`odetoy_rk4_integrate`) is (a) differentiable
# end-to-end with `Enzyme.autodiff`/`Enzyme.gradient`, matching finite differences, and (b)
# compiles and runs inside a `KernelAbstractions.jl` `@kernel` -- the exact pattern
# `GravitationalWaveBayes.jl` uses for `IMRPhenomD`'s
# `_phi_scalar`/`_ampl_scalar` scalar core, now confirmed for a genuinely ODE-integrated model
# too.
#
# `Enzyme`/`KernelAbstractions` are test-only dependencies (see Project.toml's `[extras]`/
# `[targets]`) -- GravitationalWaveforms.jl itself stays dependency-light (only `StaticArrays`, needed for
# `ODEToyState`); differentiating/kernelizing this physics is a downstream concern
# (`GravitationalWaveBayes.jl`'s), this package just needs to *not get in the way* of it, which is what these
# tests confirm.
#
# Runtime note: like `GravitationalWaveBayes.jl`'s own `test_enzyme_gradient.jl`/`test_enzyme_ka.jl`, Enzyme's
# first-call compilation of a reverse-mode rule for this code is slow (observed ~80-150s per
# `@testset` below during development) -- this is normal Enzyme compile overhead, not a hang or
# a performance regression in the physics code itself.

using Enzyme
using KernelAbstractions

# Defined at file top level (not inside a `@testset`/`let` block) and taking every quantity as an
# explicit argument -- no closed-over outer-scope variables. This matters, not just style:
# `@testset` bodies are their own local scope, so a nested `function` defined inside one that
# *closes over* a `@testset`-local variable (a first version of the KA-kernel Enzyme test below
# closed over `backend`, `omega0`, `dt`, `nsteps`, `n` this way) gets that variable boxed by
# Julia's compiler (it can't prove the closure doesn't need to see later reassignments) --  and a
# boxed capture is exactly the kind of "can this be written through this reference?" ambiguity
# Enzyme's activity analysis refuses to guess about, raising
# `EnzymeMutabilityException: Function argument passed to autodiff cannot be proven readonly`
# pointing at the *boxed capture itself* (confirmed directly: the exact same body, moved to a
# real top-level function with all inputs as plain arguments -- `backend` passed as
# `Enzyme.Const(backend)` -- differentiates cleanly, and faster too, presumably since nothing is
# boxed for type inference either).
@kernel function _odetoy_ka_test_kernel!(out, mc_secs, omega0, dt, nsteps)
    i = @index(Global)
    state0 = ODEToyState(zero(eltype(mc_secs)), omega0)
    final = odetoy_rk4_integrate(state0, dt, nsteps, mc_secs[i])
    out[i] = 2 * final.phase
end

function _odetoy_ka_test_run!(out, mc_secs, backend, omega0, dt, nsteps, n)
    kernel! = _odetoy_ka_test_kernel!(backend)
    kernel!(out, mc_secs, omega0, dt, nsteps; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return sum(out)
end

@testset "ODEToy: Enzyme.gradient through odetoy_rk4_integrate matches finite differences" begin
    mc, omega0, dt = 25.0, pi * 20.0, 1.0 / 4096.0
    Mc_sec = mc * GravitationalWaveforms.GMsun_over_c3
    nsteps = 2048  # t ~= 0.5s, comfortably below this (mc, omega0)'s tc ~= 1.03s

    # Differentiate the final GW phase (2*orbital phase) w.r.t. chirp mass (in seconds) --
    # exactly the kind of scalar-in/scalar-out, allocation-free call graph a real likelihood
    # gradient through this physics would need.
    function final_phase(Mc_sec)
        state0 = ODEToyState(zero(Mc_sec), omega0)
        final = odetoy_rk4_integrate(state0, dt, nsteps, Mc_sec)
        return 2 * final.phase
    end

    g = Enzyme.gradient(Enzyme.Reverse, final_phase, Mc_sec)

    h = 1.0e-8 * Mc_sec
    fd = (final_phase(Mc_sec + h) - final_phase(Mc_sec - h)) / (2h)

    @test isfinite(g[1])
    @test isapprox(g[1], fd; rtol = 1e-4, atol = 1e-6)
end

@testset "ODEToy: compiles and runs inside a KernelAbstractions.jl @kernel (CPU)" begin
    omega0, dt = pi * 20.0, 1.0 / 4096.0
    nsteps = 2048
    backend = KernelAbstractions.CPU()
    n = 8
    # Chirp masses kept in a range that stays well inside every element's own coalescence time
    # at nsteps*dt ~= 0.5s (the largest mc here has tc ~= 0.7s, still comfortably above).
    mc_secs = collect(range(1.0e-4, 1.7e-4, length = n))
    out = zeros(Float64, n)

    _odetoy_ka_test_run!(out, mc_secs, backend, omega0, dt, nsteps, n)

    @test all(isfinite, out)

    # Cross-check bit-for-bit against direct (non-kernel) calls to the same integrator -- the
    # kernel must not silently compute something different.
    expected = [2 * odetoy_rk4_integrate(ODEToyState(0.0, omega0), dt, nsteps, mc).phase for mc in mc_secs]
    @test out == expected

    @testset "Enzyme.autodiff through the KA kernel itself" begin
        # `out` must be preallocated *outside* the function Enzyme differentiates and passed in
        # as `Duplicated` (mutated in place), not freshly allocated inside it -- and that function
        # must be a real top-level function (see the module-level comment above
        # `_odetoy_ka_test_kernel!`/`_odetoy_ka_test_run!`), not a closure over this
        # `@testset`'s local variables. Two distinct, real Enzyme findings were hit and fixed
        # while building this test, both `EnzymeMutabilityException: Function argument passed to
        # autodiff cannot be proven readonly`, from two different causes:
        #   1. First draft allocated `out = zeros(...)` freshly inside the differentiated
        #      function on every call -- Enzyme's activity analysis couldn't prove that
        #      freshly-allocated, then-summed array was safe to treat as an ordinary temporary.
        #      Fix: preallocate `out` outside and pass it in as `Duplicated` (mutated in place),
        #      the standard Enzyme in-place idiom, matching `GravitationalWaveBayes.jl`'s own
        #      `_fused_waveform_likelihood_kernel!` pattern.
        #   2. Second draft (this file's own first version) fixed (1) but defined the
        #      differentiated function *inside* this `@testset` block, closing over `backend`/
        #      `omega0`/`dt`/`nsteps`/`n` from the enclosing scope -- Julia boxes such captures
        #      (it can't prove they're never reassigned), and a boxed capture is exactly the kind
        #      of "can this be written through?" ambiguity Enzyme's activity analysis refuses to
        #      guess about. Fix: hoist the function to module top level with every quantity as an
        #      explicit argument, `backend` passed as `Enzyme.Const(backend)`.
        # Neither was a hang or a dead end -- both are exactly the kind of reportable
        # Enzyme-shape finding this scaffolding step exists to surface.
        out_buf = zeros(Float64, n)
        dout_buf = zeros(Float64, n)
        dmc = zeros(Float64, n)
        Enzyme.autodiff(Enzyme.Reverse, _odetoy_ka_test_run!, Enzyme.Active,
            Enzyme.Duplicated(out_buf, dout_buf), Enzyme.Duplicated(mc_secs, dmc),
            Enzyme.Const(backend), Enzyme.Const(omega0), Enzyme.Const(dt), Enzyme.Const(nsteps), Enzyme.Const(n))
        @test all(isfinite, dmc)

        fd = similar(dmc)
        for i in 1:n
            h = 1.0e-6 * mc_secs[i]
            mp = copy(mc_secs); mp[i] += h
            mm = copy(mc_secs); mm[i] -= h
            fd[i] = (_odetoy_ka_test_run!(zeros(n), mp, backend, omega0, dt, nsteps, n) -
                     _odetoy_ka_test_run!(zeros(n), mm, backend, omega0, dt, nsteps, n)) / (2h)
        end
        @test isapprox(dmc, fd; rtol = 1e-4, atol = 1e-6)
    end
end
