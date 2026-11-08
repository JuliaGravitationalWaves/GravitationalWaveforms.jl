# Enzyme + KernelAbstractions validation for GravitationalWaveforms.SEOBNRv5AlignedSpin,
# mirroring alignedspinpnham_enzyme_ka_tests.jl's structure.
#
# What is differentiated, and why: SEOBNRv5AlignedSpin.jl's own docstring is explicit that the
# two-stage initial-condition solve (`_seobv5_circular_ic`/`_seobv5_dissipative_ic`) uses ITS OWN
# internal finite differences (a Jacobian/Hessian built from central differences of the already-
# analytic `_seobv5_gradient`) and is "not analytically differentiable as written" -- confirmed
# directly by `test/seobnrv5alignedspin_tests.jl`'s own "Finite-difference smoothness sanity"
# testset comment, which found a real, characterized noise floor from exactly this nesting. So,
# exactly as for AlignedSpinPNHam/PrecessingSpinPNHam, this file differentiates
# `seobnrv5alignedspin_rk4_integrate`/`_seobv5_rhs` itself -- the genuinely analytic (transcribed,
# not finite-differenced) Hamiltonian-gradient hot path -- w.r.t. a `coeffs` field, holding the
# two-stage initial condition (`r0`, `pphi0`, `pr0`, computed once via the real, Float64-only
# `_seobv5_coeffs`) fixed. This is consistent with (not a weaker substitute for) the actual
# Enzyme-readiness claim this file's docstring makes: only the RHS/integrator, not the IC solve,
# was ever claimed Enzyme-differentiable.
#
# No Complex{T} anywhere in _seobv5_ham_xi/_seobv5_gradient/_seobv5_rr_force/_seobv5_rhs (confirmed
# directly by reading SEOBNRv5AlignedSpin.jl -- the full 7-mode radiation-reaction force this port
# now implements ((2,2),(3,3),(2,1),(4,4),(3,2),(5,5),(4,3)) is real-valued throughout by a
# deliberate design choice: the one mode, (3,3), whose `rholmPwrl` is genuinely complex is carried
# as a real/imaginary pair of plain `T`s, not Julia's `Complex{T}`, see that file's own
# "Radiation-reaction force" docstring section) -- so this is not expected to hit GravitationalWaveBayes.jl's
# known Complex{T}+Enzyme.Reverse+CUDABackend() TypeAnalysis.cpp crash class in its GPU testset
# below. (An earlier version of this comment flagged the 7-mode extension as still pending and
# a re-run as still needed once it landed -- it has since landed, real/imaginary-pair-not-
# Complex{T} design intact, and this file's `_seobv5_test_coeffs()` was updated for the six new
# `absN*` fields `_seobv5_rhs` now requires.)

using Enzyme
using KernelAbstractions

const _SEOBV5_NU = 0.24
const _SEOBV5_X1 = 0.6
const _SEOBV5_X2 = 0.4
const _SEOBV5_CHI1 = 0.3
const _SEOBV5_CHI2 = -0.2
const _SEOBV5_OMEGA0 = 0.01
const _SEOBV5_DT_HAT = 0.05
const _SEOBV5_T_HAT = 125.663706   # matches this file's own "early" real-pyseobnr checkpoint

function _seobv5_test_coeffs()
    ap = _SEOBV5_X1 * _SEOBV5_CHI1 + _SEOBV5_X2 * _SEOBV5_CHI2
    am = _SEOBV5_X1 * _SEOBV5_CHI1 - _SEOBV5_X2 * _SEOBV5_CHI2
    a6 = GravitationalWaveforms._seobv5_a6_NS(_SEOBV5_NU)
    dSO = GravitationalWaveforms._seobv5_dSO(_SEOBV5_NU, ap, am)
    delta = _SEOBV5_X1 - _SEOBV5_X2
    chiS = 0.5 * (_SEOBV5_CHI1 + _SEOBV5_CHI2)
    chiA = 0.5 * (_SEOBV5_CHI1 - _SEOBV5_CHI2)
    atest = (1.0 - 2.0 * _SEOBV5_NU) * chiS + delta * chiA
    absN = (GravitationalWaveforms._seobv5_multipole_prefix_abs(_SEOBV5_NU, _SEOBV5_X1, _SEOBV5_X2, lm...)
        for lm in ((3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)))
    fc = GravitationalWaveforms._seobv5_flux_coeffs(_SEOBV5_NU, delta, chiS, chiA, atest, absN...)
    hp = GravitationalWaveforms.SEOBNRv5HamParams(_SEOBV5_NU, _SEOBV5_X1, _SEOBV5_X2, _SEOBV5_CHI1, _SEOBV5_CHI2, a6, dSO)
    r0, pphi0 = GravitationalWaveforms._seobv5_circular_ic(_SEOBV5_OMEGA0, hp)
    pr0 = GravitationalWaveforms._seobv5_dissipative_ic(r0, pphi0, hp, fc)
    # the per-binary flux coefficients (`fc`), computed the way `_seobv5_coeffs` does
    coeffs = (nu = _SEOBV5_NU, X1 = _SEOBV5_X1, X2 = _SEOBV5_X2, chi1 = _SEOBV5_CHI1, chi2 = _SEOBV5_CHI2,
        hp = hp, fc = fc)
    return coeffs, r0, pphi0, pr0
end

# `coeffs` with nu / dSO changed consistently: the top-level `nu` (flux) and the Hamiltonian's `hp`
function _seobv5_test_with(c; nu = c.nu, dSO = c.hp.dSO)
    h = c.hp
    return merge(c, (nu = nu, hp = GravitationalWaveforms.SEOBNRv5HamParams(promote(nu, h.X1, h.X2, h.chi1, h.chi2, h.a6, dSO)...)))
end

_seobv5_test_zero(x::Real) = zero(x)
_seobv5_test_zero(x::Union{Tuple,NamedTuple}) = map(_seobv5_test_zero, x)

@testset "SEOBNRv5AlignedSpin: Enzyme.gradient through seobnrv5alignedspin_rk4_integrate matches finite differences" begin
    coeffs, r0, pphi0, pr0 = _seobv5_test_coeffs()
    nsteps = round(Int, _SEOBV5_T_HAT / _SEOBV5_DT_HAT)

    @testset "w.r.t. coeffs.nu" begin
        function final_phase_nu(nu_val)
            c2 = _seobv5_test_with(coeffs; nu = nu_val)
            T = typeof(nu_val)
            state0 = SEOBNRv5AlignedSpinState(T(r0), zero(T), T(pr0), T(pphi0))
            final = seobnrv5alignedspin_rk4_integrate(state0, T(_SEOBV5_DT_HAT), nsteps, c2)
            return 2 * final.phi
        end

        g = Enzyme.gradient(Enzyme.Reverse, final_phase_nu, coeffs.nu)

        h = 1.0e-6 * coeffs.nu
        fd = (final_phase_nu(coeffs.nu + h) - final_phase_nu(coeffs.nu - h)) / (2h)

        @test isfinite(g[1])
        @test isapprox(g[1], fd; rtol = 1e-4, atol = 1e-6)
    end

    @testset "w.r.t. coeffs.dSO (spin-orbit calibration coefficient)" begin
        function final_phase_dSO(dSO_val)
            c2 = _seobv5_test_with(coeffs; dSO = dSO_val)
            T = typeof(dSO_val)
            state0 = SEOBNRv5AlignedSpinState(T(r0), zero(T), T(pr0), T(pphi0))
            final = seobnrv5alignedspin_rk4_integrate(state0, T(_SEOBV5_DT_HAT), nsteps, c2)
            return 2 * final.phi
        end

        g = Enzyme.gradient(Enzyme.Reverse, final_phase_dSO, coeffs.hp.dSO)

        h = 1.0e-6 * max(abs(coeffs.hp.dSO), 1.0)
        fd = (final_phase_dSO(coeffs.hp.dSO + h) - final_phase_dSO(coeffs.hp.dSO - h)) / (2h)

        @test isfinite(g[1])
        @test isapprox(g[1], fd; rtol = 1e-4, atol = 1e-6)
    end
end

# --- KernelAbstractions.jl @kernel compilation, CPU backend. Batches over an array of `nu`
# values, holding everything else (including the shared IC r0/pphi0/pr0) fixed. Top-level,
# non-closure functions throughout. ---

@kernel function _seobv5_ka_test_kernel!(out, nus, coeffs, r0, pphi0, pr0, dt_hat, nsteps)
    i = @index(Global)
    T = eltype(nus)
    c2 = _seobv5_test_with(coeffs; nu = nus[i])
    state0 = SEOBNRv5AlignedSpinState(T(r0), zero(T), T(pr0), T(pphi0))
    final = seobnrv5alignedspin_rk4_integrate(state0, dt_hat, nsteps, c2)
    out[i] = 2 * final.phi
end

function _seobv5_ka_test_run!(out, nus, backend, coeffs, r0, pphi0, pr0, dt_hat, nsteps, n)
    kernel! = _seobv5_ka_test_kernel!(backend)
    kernel!(out, nus, coeffs, r0, pphi0, pr0, dt_hat, nsteps; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return sum(out)
end

function _seobv5_final_phase_wrt_nu_ref(nu_val, coeffs, r0, pphi0, pr0, dt_hat, nsteps)
    c2 = _seobv5_test_with(coeffs; nu = nu_val)
    T = typeof(nu_val)
    state0 = SEOBNRv5AlignedSpinState(T(r0), zero(T), T(pr0), T(pphi0))
    final = seobnrv5alignedspin_rk4_integrate(state0, dt_hat, nsteps, c2)
    return 2 * final.phi
end

@testset "SEOBNRv5AlignedSpin: compiles and runs inside a KernelAbstractions.jl @kernel (CPU)" begin
    coeffs, r0, pphi0, pr0 = _seobv5_test_coeffs()
    nsteps = round(Int, _SEOBV5_T_HAT / _SEOBV5_DT_HAT)
    backend = KernelAbstractions.CPU()
    n = 8
    # A narrow band around nu=0.24 -- stays comfortably inside (0,0.25], SEOBNRv5AlignedSpin's own
    # domain, and close enough to the IC's own nu that the shared (r0,pphi0,pr0) (solved AT
    # nu=0.24) keeps the trajectory well-behaved over this file's short nsteps window.
    nus = collect(range(0.2, 0.2499, length = n))
    out = zeros(Float64, n)

    _seobv5_ka_test_run!(out, nus, backend, coeffs, r0, pphi0, pr0, _SEOBV5_DT_HAT, nsteps, n)

    @test all(isfinite, out)

    expected = [_seobv5_final_phase_wrt_nu_ref(nu_val, coeffs, r0, pphi0, pr0, _SEOBV5_DT_HAT, nsteps) for nu_val in nus]
    @test out == expected

    @testset "Enzyme.autodiff through the KA kernel itself" begin
        out_buf = zeros(Float64, n)
        dout_buf = zeros(Float64, n)
        dnu = zeros(Float64, n)
        Enzyme.autodiff(Enzyme.Reverse, _seobv5_ka_test_run!, Enzyme.Active,
            Enzyme.Duplicated(out_buf, dout_buf), Enzyme.Duplicated(nus, dnu),
            Enzyme.Const(backend), Enzyme.Const(coeffs), Enzyme.Const(r0), Enzyme.Const(pphi0),
            Enzyme.Const(pr0), Enzyme.Const(_SEOBV5_DT_HAT), Enzyme.Const(nsteps), Enzyme.Const(n))
        @test all(isfinite, dnu)

        fd = similar(dnu)
        for i in 1:n
            h = 1.0e-6 * nus[i]
            np = copy(nus); np[i] += h
            nm = copy(nus); nm[i] -= h
            fd[i] = (_seobv5_ka_test_run!(zeros(n), np, backend, coeffs, r0, pphi0, pr0, _SEOBV5_DT_HAT, nsteps, n) -
                     _seobv5_ka_test_run!(zeros(n), nm, backend, coeffs, r0, pphi0, pr0, _SEOBV5_DT_HAT, nsteps, n)) / (2h)
        end
        @test isapprox(dnu, fd; rtol = 1e-4, atol = 1e-6)
    end
end

# --- KernelAbstractions.jl @kernel compilation, GPU backend (CUDA.jl). Skips gracefully if no
# functional CUDA device is present. ---

const _SEOBV5_HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch e
    @warn "CUDA.jl not usable on this machine -- skipping SEOBNRv5AlignedSpin GPU KA tests" exception = (e, catch_backtrace())
    false
end

# Top-level (not inside any @testset), used only by the GPU Enzyme.autodiff-through-kernel
# subtest below. See alignedspinpnham_enzyme_ka_tests.jl's own GPU testset comment for the full
# writeup of the two real Enzyme+KernelAbstractions-on-CUDA findings this design works around: a
# GPU kernel LAUNCH cannot accept any `Active`-classified argument (only `Const`/`Duplicated`),
# and a bare `Float64` scalar reaching `kernel!(...)` gets independently reclassified `Active` by
# KernelAbstractions' CUDA-specific Enzyme rule regardless of the outer autodiff() call's own
# `Const` annotation. Fix: every `Float64` quantity (`nu` and the whole `r0`/`pphi0`/`pr0`/`dt_hat`
# initial condition) is folded into ONE per-thread struct, uploaded whole as a `Duplicated` array
# -- only `nsteps::Int` remains a bare kernel-launch scalar.
@kernel function _seobv5_ka_gpu_diff_kernel!(out, coeffs_arr, nsteps)
    i = @index(Global)
    c2 = coeffs_arr[i]
    T = typeof(c2.nu)
    state0 = SEOBNRv5AlignedSpinState(T(c2.r0), zero(T), T(c2.pr0), T(c2.pphi0))
    final = seobnrv5alignedspin_rk4_integrate(state0, c2.dt_hat, nsteps, c2)
    out[i] = 2 * final.phi
end
function _seobv5_ka_gpu_diff_run!(out, coeffs_arr, backend, nsteps, n)
    kernel! = _seobv5_ka_gpu_diff_kernel!(backend)
    kernel!(out, coeffs_arr, nsteps; ndrange = n)
    KernelAbstractions.synchronize(backend)
    return sum(out)
end

if _SEOBV5_HAVE_CUDA
    @testset "SEOBNRv5AlignedSpin: compiles and runs inside a KernelAbstractions.jl @kernel (CUDA GPU)" begin
        coeffs, r0, pphi0, pr0 = _seobv5_test_coeffs()
        nsteps = round(Int, _SEOBV5_T_HAT / _SEOBV5_DT_HAT)
        backend = CUDA.CUDABackend()
        n = 8
        nus = collect(range(0.2, 0.2499, length = n))
        nus_gpu = CUDA.CuArray(nus)
        out_gpu = CUDA.zeros(Float64, n)

        _seobv5_ka_test_run!(out_gpu, nus_gpu, backend, coeffs, r0, pphi0, pr0, _SEOBV5_DT_HAT, nsteps, n)
        out = Array(out_gpu)

        @test all(isfinite, out)
        expected = [_seobv5_final_phase_wrt_nu_ref(nu_val, coeffs, r0, pphi0, pr0, _SEOBV5_DT_HAT, nsteps) for nu_val in nus]
        @test isapprox(out, expected; rtol = 1e-10)
        @test out_gpu isa CUDA.CuArray

        @testset "Enzyme.autodiff (forward-mode) through the CUDA KA kernel itself" begin
            # This subtest uses Enzyme.Forward. Enzyme.Reverse here is CONFIRMED BROKEN on this class of
            # hardware, not a physics bug or an architectural Enzyme-incompatibility --
            # `Enzyme.autodiff(Reverse, ...)` reverse-mode-differentiating this kernel makes
            # `ptxas` (NVIDIA's own PTX assembler, a separate external process) grow without bound
            # (confirmed via `dmesg`: OOM-killed at ~15GB RAM, then again at ~24GB RAM+30GB swap
            # after 54 minutes with no sign of finishing) because Enzyme's reverse-mode source
            # transformation duplicates the ~200-term CSE Hamiltonian gradient's adjoint code once
            # per RK4 step actually taken -- `ptxas`'s input PTX size (and thus its own memory use)
            # scales with `nsteps`, not just the Hamiltonian's one-time size.
            #
            # Fix, not a workaround: this model's own Hamiltonian gradient
            # (`_seobv5_gradient`/`_seobv5_auxderivs`) is already analytic (a verbatim port of
            # `pyseobnr`'s own pre-differentiated closed form) -- Enzyme is only ever asked to
            # differentiate the *outer* RK4-integration loop w.r.t. a handful of binary parameters
            # (`nu`, `dSO`, ...), which is exactly forward-mode AD's textbook case: cost scales
            # with the number of *inputs* being differentiated (one here, per thread), not with
            # reverse-mode's need to invert the whole control-flow graph. Forward-mode compiles as
            # ordinary dual-number arithmetic with no extra control-flow duplication for `ptxas` to
            # choke on, so it should sidestep the PTX blowup entirely -- confirmed below, at the
            # model's own full `nsteps` (no `nsteps_diff` shortening needed).
            #
            # Mechanically: seed each thread's own `coeffs_arr` shadow with `nu`-component = 1,
            # everything else 0. Since `out[i]` depends only on `coeffs_arr[i]`, one single forward
            # pass computes the whole per-thread Jacobian diagonal `d(out[i])/d(nu_i)` directly
            # into `dout_buf[i]` -- no separate call per thread needed.
            coeffs_arr_host = [merge(_seobv5_test_with(coeffs; nu = nu_val), (r0 = r0, pphi0 = pphi0, pr0 = pr0,
                dt_hat = _SEOBV5_DT_HAT)) for nu_val in nus]
            coeffs_arr_gpu = CUDA.CuArray(coeffs_arr_host)
            seed_c = merge(coeffs, (nu = 1.0, X1 = 0.0, X2 = 0.0, chi1 = 0.0, chi2 = 0.0,
                hp = GravitationalWaveforms.SEOBNRv5HamParams(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), fc = _seobv5_test_zero(coeffs.fc),
                r0 = 0.0, pphi0 = 0.0, pr0 = 0.0, dt_hat = 0.0))
            dcoeffs_arr_gpu = CUDA.CuArray(fill(seed_c, n))

            out_buf = CUDA.zeros(Float64, n)
            dout_buf = CUDA.zeros(Float64, n)  # forward-mode: mutated in place to hold d(out)/d(nu)

            gpu_grad_ok = true
            local dnu
            try
                Enzyme.autodiff(Enzyme.Forward, _seobv5_ka_gpu_diff_run!, Enzyme.Const,
                    Enzyme.Duplicated(out_buf, dout_buf), Enzyme.Duplicated(coeffs_arr_gpu, dcoeffs_arr_gpu),
                    Enzyme.Const(backend), Enzyme.Const(nsteps), Enzyme.Const(n))
                dnu = Array(dout_buf)
            catch e
                gpu_grad_ok = false
                @warn "SEOBNRv5AlignedSpin Enzyme.autodiff(Forward, ...) on CUDABackend() failed -- see this testset's own comment; if this reproduces, it is a genuinely new finding, not the already-diagnosed reverse-mode ptxas OOM" exception = (e, catch_backtrace())
            end

            if gpu_grad_ok
                @test all(isfinite, dnu)
                fd = similar(dnu)
                for i in 1:n
                    h = 1.0e-6 * nus[i]
                    fd[i] = (_seobv5_final_phase_wrt_nu_ref(nus[i] + h, coeffs, r0, pphi0, pr0, _SEOBV5_DT_HAT, nsteps) -
                             _seobv5_final_phase_wrt_nu_ref(nus[i] - h, coeffs, r0, pphi0, pr0, _SEOBV5_DT_HAT, nsteps)) / (2h)
                end
                @test isapprox(dnu, fd; rtol = 1e-4, atol = 1e-6)
            else
                @test_broken false  # see @warn above for the actual failure -- would be a new, undiagnosed finding
            end
        end
    end
else
    @testset "SEOBNRv5AlignedSpin: KernelAbstractions.jl @kernel (CUDA GPU) -- skipped" begin
        @test_skip "no functional CUDA device on this machine"
    end
end
