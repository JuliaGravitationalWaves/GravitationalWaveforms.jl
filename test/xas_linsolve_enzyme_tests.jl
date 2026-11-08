# Regression tests for `GravitationalWaveforms._xas_linsolve` (IMRPhenomXAS.jl), the small dense solver behind every
# collocation-point phase/amplitude reconstruction in IMRPhenomXAS and IMRPhenomXHM (N = 4, 5, 6 and 8 there).
#
# Why this file exists: the previous implementation (recursive Laplace-expansion Cramer's rule over
# `NTuple`s) was mis-differentiated by Enzyme in REVERSE mode at N = 6 (relative error 1.0 on Julia 1.12.7, 1.5e-2 on
# 1.13.0, on a well-conditioned matrix; forward mode was correct) and crashed Enzyme's LLVM verification for N >= 7 on
# Julia 1.13 ("PHI nodes not grouped at top of basic block"). IMRPhenomXHM's (3,2) mode is the only mode that solves a
# 6x6 system, and its reverse-mode derivative was wrong by up to >100x, giving 0.1-3 % errors in the full log-likelihood
# gradient. The old gradient test for XHM (in GravitationalWaveBayes.jl) used zero strain, so it
# passed vacuously. These tests use well-conditioned (diagonally dominant) matrices, so central finite differences are
# a trustworthy reference and any Enzyme discrepancy is a real bug, not conditioning noise.

using Test
using GravitationalWaveforms
using Enzyme
using KernelAbstractions

const _LS = GravitationalWaveforms._xas_linsolve

# identity matrix without importing LinearAlgebra into the test namespace
_ls_eye(N) = [i == k ? 1.0 : 0.0 for i in 1:N, k in 1:N]

# Diagonally dominant test system, built from a flat parameter vector `x` (length N*N + N).
_ls_mkA(x, ::Val{N}) where {N} = ntuple(i -> ntuple(k -> x[(i - 1) * N + k] + (i == k ? 2.0 * N : 0.0), N), N)
_ls_obj(x::Vector{Float64}, v::Val{N}) where {N} = sum(_LS(_ls_mkA(x, v), ntuple(i -> x[N * N + i], N)))
# A weighted (non-symmetric) scalar objective, so the gradient is not accidentally structured.
_ls_obj_w(x::Vector{Float64}, v::Val{N}) where {N} = sum(ntuple(j -> (j + 0.5) * _LS(_ls_mkA(x, v), ntuple(i -> x[N * N + i], N))[j]^2, N))

@testset "_xas_linsolve" begin
    @testset "solves A x = b (including a matrix that needs pivoting), N = $N" for N in (4, 5, 6, 7, 8, 9)
        # zero leading entry forces a row exchange; entries are otherwise O(1) and the system is nonsingular
        A = [i == k ? 0.0 : 1.0 / (i + 2k) + (i + k == N + 1 ? 2.0 : 0.0) for i in 1:N, k in 1:N]
        A[1, 1] = 0.0
        b = collect(1.0:N)
        x = collect(_LS(ntuple(i -> ntuple(k -> A[i, k], N), N), ntuple(i -> b[i], N)))
        @test isapprox(x, A \ b; rtol = 1e-9)
        # a generic diagonally dominant system, plus Float32 genericity
        Ad = A .+ 5.0 .* _ls_eye(N)
        xd = collect(_LS(ntuple(i -> ntuple(k -> Ad[i, k], N), N), ntuple(i -> b[i], N)))
        @test isapprox(xd, Ad \ b; rtol = 1e-9)
        x32 = _LS(ntuple(i -> ntuple(k -> Float32(Ad[i, k]), N), N), ntuple(i -> Float32(b[i]), N))
        @test eltype(x32) === Float32 && isapprox(collect(Float64, x32), Ad \ b; rtol = 1e-3)
    end

    @testset "reverse-mode Enzyme gradient matches central differences, N = $N" for N in (4, 5, 6, 7, 8, 9)
        v = Val(N)
        x = 0.1 .* cos.(1.0:(N * N + N))
        for obj in (_ls_obj, _ls_obj_w)
            dx = zeros(length(x))
            Enzyme.autodiff(Enzyme.Reverse, obj, Enzyme.Active, Enzyme.Duplicated(copy(x), dx), Enzyme.Const(v))
            fd = map(1:length(x)) do i
                h = 1e-6
                xp = copy(x); xp[i] += h
                xm = copy(x); xm[i] -= h
                (obj(xp, v) - obj(xm, v)) / (2h)
            end
            @test maximum(abs.(dx .- fd)) <= 1e-6 * maximum(abs.(fd))
        end
    end
end

# --- CUDA: the solver is straight-line scalar code, so it must compile and give the host answer inside a GPU kernel.
# Skips gracefully when no functional CUDA device is present (same convention as the other CUDA test files).
const _LS_HAVE_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch e
    @warn "CUDA.jl not usable on this machine -- skipping _xas_linsolve GPU test" exception = (e, catch_backtrace())
    false
end

@kernel function _ls_cuda_kernel!(out, ::Val{N}, scale) where {N}
    w = @index(Global)
    A = ntuple(i -> ntuple(k -> (i == k ? 2.0 * N : 0.0) + scale[w] * cos(1.0 * ((i - 1) * N + k)) * 0.1, N), N)
    b = ntuple(i -> scale[w] * sin(1.0 * i), N)
    x = _LS(A, b)
    for j in 1:N
        out[j, w] = x[j]
    end
end

if _LS_HAVE_CUDA
    @testset "_xas_linsolve inside a KernelAbstractions @kernel (CUDA), N = $N" for N in (4, 6, 8)
        scale = [1.0, 0.5, 2.0]
        backend = CUDA.CUDABackend()
        out_gpu = CUDA.zeros(Float64, N, length(scale))
        _ls_cuda_kernel!(backend)(out_gpu, Val(N), CUDA.CuArray(scale); ndrange = length(scale))
        KernelAbstractions.synchronize(backend)
        out = Array(out_gpu)
        for (w, s) in enumerate(scale)
            A = [(i == k ? 2.0 * N : 0.0) + s * cos(1.0 * ((i - 1) * N + k)) * 0.1 for i in 1:N, k in 1:N]
            b = [s * sin(1.0 * i) for i in 1:N]
            @test isapprox(out[:, w], A \ b; rtol = 1e-9)
        end
    end
end
