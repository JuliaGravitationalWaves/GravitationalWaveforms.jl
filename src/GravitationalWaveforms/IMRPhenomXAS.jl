export IMRPhenomXAS, validate_phenomxas_params

"""
    IMRPhenomXAS

IMRPhenomXAS (Pratten et al. 2020, arXiv:2001.11412): aligned-spin, dominant (2,2)-mode-only
successor to IMRPhenomD. Uses a richer collocation-point-based ansatz (more calibration
points, an improved final-spin/radiated-energy fit, and a rational-function intermediate
amplitude ansatz) but follows the exact same architectural pattern as `IMRPhenomD` in this
package: `struct IMRPhenomXAS end`, and `Phi(model::IMRPhenomXAS, ...)` / `Ampl(model::IMRPhenomXAS,
...)` methods dispatching on it.

Ported from `ripple` (github.com/tedwards2412/ripple, JAX; `src/ripplegw/waveforms/
IMRPhenomXAS.py` and `IMRPhenomX_utils.py`), itself validated against LALSimulation
(`LALSimIMRPhenomX*.c`). Several formulas were spot-checked directly against LALSimulation's C
source during porting (`LALSimIMRPhenomXUtilities.c`, `LALSimIMRPhenomX_qnm.c`) . Only the dominant (2,2) harmonic is
implemented; IMRPhenomXHM (higher harmonics) is in `IMRPhenomXHM.jl`.

# Differences from `IMRPhenomD`'s pattern worth noting
- The final-spin and radiated-energy fits (`_xas_a_Erad`) are the more accurate
  Jimenez-Forteza-et-al.-style fits used by the X family, not IMRPhenomD's simpler linear-in-aeff
  proxy — a genuinely different (better) calibration, not a refactor of IMRPhenomD's formula.
- The phase has **two** C1-continuity join points (inspiral/intermediate at `f1`,
  intermediate/ringdown at `f2`), each fixed up with a `(value, derivative)`-matched
  linear-in-frequency correction (`alpha0,alpha1` and `beta0,beta1`), generalizing IMRPhenomD's
  single-join-point `C1Int`/`C2Int`/`C1MRD`/`C2MRD` pattern to two joins.
- The intermediate **amplitude** ansatz is a *reciprocal* rational function
  (`f^(7/6)/(delta0+delta1 f+...+delta4 f^4)`), not IMRPhenomD's direct quintic polynomial — this
  is a deliberate difference in the XAS paper, not a porting choice.
- Several phenomenological coefficients are reconstructed by solving small (4x4/5x5) linear
  systems against collocation-point values (`_xas_linsolve`/`_xas_det`, a generic
  determinant/Cramer's-rule solver over fixed-size tuples — pure, non-mutating, and thus
  Enzyme- and broadcast-safe by construction, unlike a BLAS/LAPACK-backed `\\`), rather than
  IMRPhenomD's closed-form-only coefficients. Where ripple's own Python source already gives the
  closed-form solution (inspiral amplitude `rho1..rho3`, intermediate amplitude
  `delta0..delta4`), those closed forms are transcribed directly instead of re-solving.
- Internal `(value, derivative)` matching at the join frequencies uses hand-derived analytic
  derivatives (matching this package's existing IMRPhenomD.jl convention of computing `DPhiIns`/
  `DPhiInt`/`DPhiMRDVal` etc. analytically) rather than nested automatic differentiation, to
  avoid compounding this codebase's documented Enzyme/broadcast fragility (see `@Tlit`'s
  docstring in TypeGenericity.jl) with a second AD system. Each analytic derivative has a
  paired finite-difference consistency test — see `test/test_phenomxas_phase.jl` and
  `test/test_phenomxas_amplitude.jl`.
"""
struct IMRPhenomXAS end

"""
    validate_phenomxas_params(eta, chi1, chi2; dL = nothing)

Host-side sanity check for IMRPhenomXAS parameter domains, mirroring `validate_phenomd_params`:
`0 < eta <= 0.25`, `-1 <= chi1,chi2 <= 1`, and (if `dL` is given) `dL > 0`. Throws
`ArgumentError` on violation, returns `nothing` otherwise. Intentionally not called from `Phi`/
`Ampl` themselves — see `validate_phenomd_params`'s docstring for the rationale (hot path,
Enzyme/GPU-kernel target).
"""
function validate_phenomxas_params(eta, chi1, chi2; dL = nothing)
    0.0 < eta <= 0.25 || throw(ArgumentError("Invalid eta = $eta: must be in (0, 0.25]"))
    -1.0 <= chi1 <= 1.0 || throw(ArgumentError("Invalid chi1 = $chi1: spin must be in [-1, 1]"))
    -1.0 <= chi2 <= 1.0 || throw(ArgumentError("Invalid chi2 = $chi2: spin must be in [-1, 1]"))
    if dL !== nothing
        dL > 0.0 || throw(ArgumentError("Invalid dL = $dL: luminosity distance must be positive (and must be in Gpc, see Ampl's docstring)"))
    end
    return nothing
end

# ============================================================================================
# Generic small linear-system solver (used to reconstruct several phase coefficients from
# collocation-point values, mirroring ripple's `jnp.linalg.solve` calls).
#
# Implemented as straight-line, generated (`@generated`, one body per `(N,T)`) Gaussian elimination with
# partial pivoting over `NTuple`s, rather than calling into `LinearAlgebra`'s BLAS/LAPACK-backed `\` on a
# `Matrix{T}`. N is 4, 5, 6 or 8 here (called a handful of times per `Phi`/`h_lm` setup, not per frequency
# point), so cost is irrelevant; the payoff is that this is plain scalar arithmetic with no arrays, no mutation,
# no recursion, no closures and no library calls, so it is Enzyme-differentiable and GPU-kernel-portable by
# construction, with no risk of hitting this codebase's documented Enzyme/broadcast/BLAS fragility (see
# `@Tlit`'s docstring). Row exchanges are branch-free (`ifelse`), so the body stays branch-free straight-line code.
#
# WHY NOT RECURSIVE CRAMER'S RULE. An earlier implementation was a recursive
# Laplace-expansion determinant + Cramer's rule built from `ntuple` closures. Enzyme (0.13.204, Julia 1.12.7 and
# 1.13.0) mis-differentiated it in REVERSE mode at N = 6 (relative error 1.0 on 1.12.7, 1.5e-2 on 1.13, on a
# well-conditioned matrix; forward mode and the determinant alone were correct) and aborted in LLVM verification
# ("PHI nodes not grouped at top of basic block") for N >= 7 on Julia 1.13. `IMRPhenomXHM`'s (3,2) mode is the only
# mode that solves a 6x6 system, and its reverse-mode derivative was wrong by up to >100x against a 256-bit BigFloat
# reference, giving 0.1-3 % errors in the full log-likelihood gradient. The straight-line solver below gives correct
# reverse-mode gradients for N = 4..9 on both Julia versions, no 1.13 abort, and is at least as accurate as the old
# solver in Float64 (vs BigFloat over 100 random points: IMRPhenomXAS phase 2e-11 -> 6e-13 median, XHM (2,2) 3e-13 ->
# 3e-14; (3,3)/(4,4) unchanged; the (3,2) mode stays limited to ~1e-4 by the conditioning of its 6x6 phase system,
# whichever solver is used). Regression tests: `test/xas_linsolve_enzyme_tests.jl`.
# ============================================================================================

# `b::Tuple{T,Vararg{T}}` (not `NTuple{N,T}`): at least one element, so `T` is bound for every signature (Aqua
# unbound_args); the length is checked against N below.
@generated function _xas_linsolve(A::NTuple{N,NTuple{N,T}}, b::Tuple{T,Vararg{T}}) where {N,T}
    msg = "_xas_linsolve: $N x $N system, right-hand side of length $(fieldcount(b))"
    fieldcount(b) == N || return :(throw(DimensionMismatch($msg)))
    a(i, k) = Symbol(:a_, i, :_, k)
    y(i) = Symbol(:y_, i)
    x(i) = Symbol(:x_, i)
    body = Expr(:block)
    for i in 1:N, k in 1:N
        push!(body.args, :($(a(i, k)) = A[$i][$k]))
    end
    for i in 1:N
        push!(body.args, :($(y(i)) = b[$i]))
    end
    for p in 1:(N - 1)
        # partial pivoting: bring the largest |a_ip| (i >= p) to row p by successive conditional swaps
        for i in (p + 1):N
            c = Symbol(:c_, p, :_, i)
            push!(body.args, :($c = abs($(a(i, p))) > abs($(a(p, p)))))
            for k in p:N
                t = Symbol(:t_, p, :_, i, :_, k)
                push!(body.args, :($t = $(a(p, k))))
                push!(body.args, :($(a(p, k)) = ifelse($c, $(a(i, k)), $t)))
                push!(body.args, :($(a(i, k)) = ifelse($c, $t, $(a(i, k)))))
            end
            ty = Symbol(:ty_, p, :_, i)
            push!(body.args, :($ty = $(y(p))))
            push!(body.args, :($(y(p)) = ifelse($c, $(y(i)), $ty)))
            push!(body.args, :($(y(i)) = ifelse($c, $ty, $(y(i)))))
        end
        for i in (p + 1):N
            m = Symbol(:m_, i, :_, p)
            push!(body.args, :($m = $(a(i, p)) / $(a(p, p))))
            for k in (p + 1):N
                push!(body.args, :($(a(i, k)) = $(a(i, k)) - $m * $(a(p, k))))
            end
            push!(body.args, :($(y(i)) = $(y(i)) - $m * $(y(p))))
        end
    end
    for i in N:-1:1
        acc = y(i)
        for k in (i + 1):N
            acc = :($acc - $(a(i, k)) * $(x(k)))
        end
        push!(body.args, :($(x(i)) = $acc / $(a(i, i))))
    end
    push!(body.args, Expr(:tuple, [x(i) for i in 1:N]...))
    return body
end

# ============================================================================================
# Generic collocation-value combination functions (fits of a phenomenological coefficient as a
# function of eta and spin, over a flat tuple of numeric fit coefficients). Directly mirrors
# `nospin_CV`/`Eqspin_CV`/`Uneqspin_CV`/`Amp_Nospin_CV`/`Amp_Eqspin_CV`/`Amp_Uneqspin_CV` in
# ripple's `IMRPhenomX_utils.py`. No bare `Float64` literals appear here (only integer literals
# used as `^` exponents, which do not trigger `@Tlit`'s promotion trap), so these are already
# fully T-generic without needing `@Tlit`.
# ============================================================================================

@inline function _xas_nospin_CV(c::NTuple{10,T}, eta) where {T}
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    num = c[1] + c[2] * eta + c[3] * eta2 + c[4] * eta3 + c[5] * eta4 + c[6] * eta5
    den = c[7] + c[8] * eta + c[9] * eta2 + c[10] * eta3
    return num / den
end

@inline function _xas_eqspin_CV(c::NTuple{29,T}, eta, S) where {T}
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    numerator = S * (
        c[1] + c[2] * S + c[3] * S2 + c[4] * S3 + c[5] * S4 +
        eta * (c[6] + c[7] * S + c[8] * S2 + c[9] * S3 + c[10] * S4) +
        eta2 * (c[11] + c[12] * S + c[13] * S2 + c[14] * S3 + c[15] * S4) +
        eta3 * (c[16] + c[17] * S + c[18] * S2 + c[19] * S3 + c[20] * S4) +
        eta4 * (c[21] + c[22] * S + c[23] * S2 + c[24] * S3 + c[25] * S4)
    )
    denominator = c[26] + c[27] * S + c[28] * S2 + c[29] * S3
    return numerator / denominator
end

@inline function _xas_uneqspin_CV(c::NTuple{10,T}, eta, S, chia, Seta) where {T}
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    return chia * Seta * eta * (
        c[1] + c[2] * eta + c[3] * eta2 + c[4] * eta3 + c[5] * eta4 + c[6] * eta5 +
        c[7] * S + c[8] * S * eta2 + c[9] * S * eta3
    ) + c[10] * chia * chia * eta
end

@inline function _xas_amp_nospin_CV(c::NTuple{8,T}, eta) where {T}
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    num = c[1] + c[2] * eta + c[3] * eta2 + c[4] * eta3 + c[5] * eta4
    den = c[6] + c[7] * eta + c[8] * eta2
    return num / den
end

@inline function _xas_amp_eqspin_CV(c::NTuple{28,T}, eta, S) where {T}
    eta2 = eta * eta
    eta3 = eta2 * eta
    S2 = S * S
    S3 = S2 * S
    S4 = S3 * S
    S5 = S4 * S
    numS0 = c[1] + c[2] * eta + c[3] * eta2 + c[4] * eta3
    numS1 = c[5] + c[6] * eta + c[7] * eta2 + c[8] * eta3
    numS2 = c[9] + c[10] * eta + c[11] * eta2 + c[12] * eta3
    numS3 = c[13] + c[14] * eta + c[15] * eta2 + c[16] * eta3
    numS4 = c[17] + c[18] * eta + c[19] * eta2 + c[20] * eta3
    numS5 = c[21] + c[22] * eta + c[23] * eta2 + c[24] * eta3
    den = c[25] + c[26] * S + c[27] * eta + c[28] * S2
    return (numS0 + numS1 * S + numS2 * S2 + numS3 * S3 + numS4 * S4 + numS5 * S5) / den
end

@inline function _xas_amp_uneqspin_CV(c::NTuple{6,T}, eta, chia, Seta) where {T}
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    eta5 = eta4 * eta
    return chia * Seta * (c[1] + c[2] * eta + c[3] * eta2 + c[4] * eta3 + c[5] * eta4 + c[6] * eta5)
end

# Row-level wrappers: `row` is a full 49-tuple (phase) or 42-tuple (amplitude) coefficient-table
# row, already cast to the working type T; these slice it into the nospin/eqspin/uneqspin parts
# (fixed layout: phase rows are 10+29+10, amp rows are 8+28+6, matching `eqspin_indx`/
# `uneqspin_indx` = 10/39 and `amp_eqspin_indx`/`amp_uneqspin_indx` = 8/36 in ripple) and sum
# the three contributions, exactly as ripple's `nospin_CV(...) + Eqspin_CV(...) +
# Uneqspin_CV(...)` call pattern does at each collocation-value use site.
@inline function _xas_phase_CV(row::NTuple{49,T}, eta, S, chia, Seta) where {T}
    # `Val(N)` (not a plain `Int` literal) is required for `ntuple` to compile-time-unroll --
    # the non-`Val` form doesn't guarantee `row[i]`'s index stays a compile-time constant, which
    # GPUCompiler.jl needs (confirmed by a real `InvalidIRError`/`jl_f_throw_methoderror` porting
    # this to `CUDABackend()`; every other `ntuple` call site in this codebase already uses `Val`).
    ns = ntuple(i -> row[i], Val(10))
    eq = ntuple(i -> row[10+i], Val(29))
    un = ntuple(i -> row[39+i], Val(10))
    return _xas_nospin_CV(ns, eta) + _xas_eqspin_CV(eq, eta, S) + _xas_uneqspin_CV(un, eta, S, chia, Seta)
end

@inline function _xas_amp_CV(row::NTuple{42,T}, eta, S, chia, Seta) where {T}
    ns = ntuple(i -> row[i], Val(8))
    eq = ntuple(i -> row[8+i], Val(28))
    un = ntuple(i -> row[36+i], Val(6))
    return _xas_amp_nospin_CV(ns, eta) + _xas_amp_eqspin_CV(eq, eta, S) + _xas_amp_uneqspin_CV(un, eta, chia, Seta)
end

# ============================================================================================
# Cutoff/reference frequencies (dimensionless, Mf): ringdown fRD, damping fdamp, MECO fMECO,
# and ISCO fISCO. Ported from `get_cutoff_fMs` in ripple's `IMRPhenomX_utils.py`; the
# rational-in-final-spin fRD/fdamp formula and the fMECO leading coefficients were spot-checked
# against LALSimulation's `LALSimIMRPhenomX_qnm.c` / `LALSimIMRPhenomXUtilities.c` directly
# during porting (byte-for-byte match on several distinctive constants).
# ============================================================================================

# Final spin `a` (Jimenez-Forteza-et-al.-style fit) and radiated energy `Erad`, both functions
# of (eta, Seta, dchi, StotR) only -- NOT the same `aeff`/`Erad` IMRPhenomD.jl uses (those are a
# simpler linear-in-aeff proxy); this is XAS's own, more accurate calibration.
@inline function _xas_a_Erad(eta, Seta, dchi, StotR)
    T = promote_type(typeof(eta), typeof(Seta), typeof(dchi), typeof(StotR))
    @Tlit T begin
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        mm1 = 0.5 * (1.0 + Seta)
        mm2 = 0.5 * (1.0 - Seta)
        StotR2 = StotR * StotR
        StotR3 = StotR2 * StotR
        dchi2 = dchi * dchi

        a = (
            (3.4641016151377544 * eta + 20.0830030082033 * eta2 - 12.333573402277912 * eta3) /
            (1.0 + 7.2388440419467335 * eta)
            +
            (
                (mm1^2 + mm2^2) * StotR +
                (
                    (-0.8561951310209386 * eta - 0.09939065676370885 * eta2 + 1.668810429851045 * eta3) * StotR +
                    (0.5881660363307388 * eta - 2.149269067519131 * eta2 + 3.4768263932898678 * eta3) * StotR2 +
                    (0.142443244743048 * eta - 0.9598353840147513 * eta2 + 1.9595643107593743 * eta3) * StotR3
                ) / (1.0 + (-0.9142232693081653 + 2.3191363426522633 * eta - 9.710576749140989 * eta3) * StotR)
            )
            +
            (
                0.3223660562764661 * dchi * Seta * (1.0 + 9.332575956437443 * eta) * eta2 -
                0.059808322561702126 * dchi2 * eta3 +
                2.3170397514509933 * dchi * Seta * (1.0 - 3.2624649875884852 * eta) * eta3 * StotR
            )
        )

        Erad = (
            (
                (0.057190958417936644 * eta + 0.5609904135313374 * eta2 - 0.84667563764404 * eta3 + 3.145145224278187 * eta4) *
                (
                    1.0 +
                    (-0.13084389181783257 - 1.1387311580238488 * eta + 5.49074464410971 * eta2) * StotR +
                    (-0.17762802148331427 + 2.176667900182948 * eta2) * StotR2 +
                    (-0.6320191645391563 + 4.952698546796005 * eta - 10.023747993978121 * eta2) * StotR3
                )
            ) / (1.0 + (-0.9919475346968611 + 0.367620218664352 * eta + 4.274567337924067 * eta2) * StotR)
        ) + (
            -0.09803730445895877 * dchi * Seta * (1.0 - 3.2283713377939134 * eta) * eta2 +
            0.01118530335431078 * dchi2 * eta3 -
            0.01978238971523653 * dchi * Seta * (1.0 - 4.91667749015812 * eta) * eta * StotR
        )

        return a, Erad
    end
end

@inline function _xas_fRD_fdamp(a, Erad)
    T = promote_type(typeof(a), typeof(Erad))
    @Tlit T begin
        a2 = a * a
        a3 = a2 * a
        a4 = a3 * a
        a5 = a4 * a
        a6 = a5 * a
        a7 = a6 * a
        fRD = (
            (
                0.05947169566573468 - 0.14989771215394762 * a + 0.09535606290986028 * a2 +
                0.02260924869042963 * a3 - 0.02501704155363241 * a4 - 0.005852438240997211 * a5 +
                0.0027489038393367993 * a6 + 0.0005821983163192694 * a7
            ) / (
                1.0 - 2.8570126619966296 * a + 2.373335413978394 * a2 -
                0.6036964688511505 * a4 + 0.0873798215084077 * a6
            )
        ) / (1.0 - Erad)
        fdamp = (
            (
                0.014158792290965177 - 0.036989395871554566 * a + 0.026822526296575368 * a2 +
                0.0008490933750566702 * a3 - 0.004843996907020524 * a4 -
                0.00014745235759327472 * a5 + 0.0001504546201236794 * a6
            ) / (
                1.0 - 2.5900842798681376 * a + 1.8952576220623967 * a2 -
                0.31416610693042507 * a4 + 0.009002719412204133 * a6
            )
        ) / (1.0 - Erad)
        return fRD, fdamp
    end
end

@inline function _xas_fISCO(a)
    T = typeof(a)
    @Tlit T begin
        a2 = a * a
        Z1 = 1.0 + cbrt(1.0 - a2) * (cbrt(1.0 + a) + cbrt(1.0 - a))
        Z1 = min(Z1, 3.0)
        Z2 = sqrt(3.0 * a2 + Z1 * Z1)
        rISCO = 3.0 + Z2 - sign(a) * sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))
        rISCO3o2 = rISCO * sqrt(rISCO)
        OmegaISCO = 1.0 / (rISCO3o2 + a)
        return OmegaISCO / pi
    end
end

"""
    _xas_precessing_final_spin(eta, chi1, chi2, chi_p)

Precessing-binary final spin magnitude, `XLALSimIMRPhenomXPrecessingFinalSpin2017`
(`LALSimIMRPhenomXUtilities.c` lines 225-251, the default `PhenomXPFinalSpinMod = 0` LALSuite
uses for both `IMRPhenomXP` and `IMRPhenomXPHM`): combines the ALIGNED-spin final-spin fit already
in [`_xas_a_Erad`](@ref) (confirmed, coefficient-for-coefficient, to be the exact same fit as
`XLALSimIMRPhenomXFinalSpin2017`) with the in-plane spin's contribution to the final black hole's
spin magnitude, `Sperp = chi_p * m1n^2` (`m1n = m1/(m1+m2)`, i.e. LALSuite's `q_factor` under this
package's existing `m1 >= m2` convention, so the C source's `m1 >= m2` branch is the only one that
can ever apply here): `af = copysign(1, af_parallel) * sqrt(Sperp^2 + af_parallel^2)`.

**Only `fRING`/`fDAMP` should be recomputed from this value, not radiated energy `Erad`, `fISCO` or
`fMECO`** -- LALSuite leaves those on the aligned-spin fit even when using the precessing final
spin for ringdown (`LALSimIMRPhenomX_precession.c` lines 1482-1519: `pWF->afinal = pWF->afinal_prec`
feeds `pWF->fRING`/`pWF->fDAMP` via `evaluate_QNMfit_fring22`/`_fdamp22` immediately below, but
`pWF->fISCO = XLALSimIMRPhenomXfISCO(pWF->afinal)` is commented out there -- `fISCO` was already
set earlier, from the aligned-spin final spin, in `IMRPhenomXSetWaveformVariables`, and is never
touched again). Callers (`twist_up22` for `IMRPhenomXP`, `hphc(::IMRPhenomXPHM, ...)`) therefore
pass this value only as the `final_spin_override` fed to [`_xas_fRD_fdamp`](@ref)'s `a` argument
(via `Phi`/`Ampl(::IMRPhenomXAS, ...)`'s and `_xhm_system_setup`'s own `final_spin_override`
keyword), never in place of the `a` used for `_xas_fISCO`/`_xhm_fring_fdamp`'s `Erad`.
"""
@inline function _xas_precessing_final_spin(eta, chi1, chi2, chi_p)
    T = promote_type(typeof(eta), typeof(chi1), typeof(chi2), typeof(chi_p))
    @Tlit T begin
        Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
        mm1 = 0.5 * (1.0 + Seta)
        mm2 = 0.5 * (1.0 - Seta)
        StotR = (mm1 * mm1 * chi1 + mm2 * mm2 * chi2) / (mm1 * mm1 + mm2 * mm2)
        a_parallel, _ = _xas_a_Erad(eta, Seta, chi1 - chi2, StotR)
        Sperp = chi_p * mm1 * mm1
        return copysign(one(T), a_parallel) * sqrt(Sperp * Sperp + a_parallel * a_parallel)
    end
end

@inline function _xas_fMECO(eta, S, dchi, Seta)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi), typeof(Seta))
    @Tlit T begin
        eta2 = eta * eta
        eta3 = eta2 * eta
        eta4 = eta3 * eta
        S2 = S * S
        S3 = S2 * S
        dchi2 = dchi * dchi

        term1 = (
            0.018744340279608845 + 0.0077903147004616865 * eta + 0.003940354686136861 * eta2 -
            0.00006693930988501673 * eta3
        ) / (1.0 - 0.10423384680638834 * eta)

        num2 = S * (
            0.00027180386951683135 - 0.00002585252361022052 * S +
            eta4 * (-0.0006807631931297156 + 0.022386313074011715 * S - 0.0230825153005985 * S2) +
            eta2 * (0.00036556167661117023 - 0.000010021140796150737 * S - 0.00038216081981505285 * S2) +
            eta * (0.00024422562796266645 - 0.00001049013062611254 * S - 0.00035182990586857726 * S2) +
            eta3 * (-0.0005418851224505745 + 0.000030679548774047616 * S + 4.038390455349854e-6 * S2) -
            0.00007547517256664526 * S2
        )
        den2 = 0.026666543809890402 +
               (-0.014590539285641243 - 0.012429476486138982 * eta + 1.4861197211952053 * eta4 +
                0.025066696514373803 * eta2 + 0.005146809717492324 * eta3) * S +
               (-0.0058684526275074025 - 0.02876774751921441 * eta - 2.551566872093786 * eta4 -
                0.019641378027236502 * eta2 - 0.001956646166089053 * eta3) * S2 +
               (0.003507640638496499 + 0.014176504653145768 * eta + 1.0 * eta4 +
                0.012622225233586283 * eta2 - 0.00767768214056772 * eta3) * S3

        term3 = dchi2 * (0.00034375176678815234 + 0.000016343732281057392 * eta) * eta2 +
                dchi * Seta * eta * (
                    0.08064665214195679 * eta2 +
                    eta * (-0.028476219509487793 - 0.005746537021035632 * S) -
                    0.0011713735642446144 * S
                )

        return term1 + num2 / den2 + term3
    end
end

# ============================================================================================
# Phase ansatz pieces (value and, where needed for C1-continuity matching, analytic
# derivative). Ported from `get_inspiral_phase`/`get_intermediate_raw_phase`/
# `get_mergerringdown_raw_phase` in ripple's `IMRPhenomXAS.py`. `_val`/`_deriv` pairs are
# cross-checked against finite differences in test/test_phenomxas_phase.jl.
# ============================================================================================

@inline function _xas_phi_ins_val(x, phi0, phi1, phi2, phi3, phi4, phi5, phi5L, phi6, phi6L, phi7, phi8, phi8L, sigma1, sigma2, sigma3, sigma4, phiN)
    T = promote_type(typeof(x), typeof(phi2), typeof(sigma1))
    @Tlit T begin
        lx = log(x)
        phi_TF2 = phi0 + phi1 * x^(1.0 / 3.0) + phi2 * x^(2.0 / 3.0) + phi3 * x +
                  phi4 * x^(4.0 / 3.0) + phi5 * x^(5.0 / 3.0) + phi5L * x^(5.0 / 3.0) * lx +
                  phi6 * x^2.0 + phi6L * x^2.0 * lx + phi7 * x^(7.0 / 3.0) +
                  phi8 * x^(8.0 / 3.0) + phi8L * x^(8.0 / 3.0) * lx
        phi_Ins = phi_TF2 + sigma1 * x^(8.0 / 3.0) + sigma2 * x^3.0 + sigma3 * x^(10.0 / 3.0) + sigma4 * x^(11.0 / 3.0)
        return phiN * x^(-5.0 / 3.0) * phi_Ins
    end
end

@inline function _xas_phi_ins_deriv(x, phi0, phi1, phi2, phi3, phi4, phi5, phi5L, phi6, phi6L, phi7, phi8, phi8L, sigma1, sigma2, sigma3, sigma4, phiN)
    T = promote_type(typeof(x), typeof(phi2), typeof(sigma1))
    @Tlit T begin
        lx = log(x)
        phi_TF2 = phi0 + phi1 * x^(1.0 / 3.0) + phi2 * x^(2.0 / 3.0) + phi3 * x +
                  phi4 * x^(4.0 / 3.0) + phi5 * x^(5.0 / 3.0) + phi5L * x^(5.0 / 3.0) * lx +
                  phi6 * x^2.0 + phi6L * x^2.0 * lx + phi7 * x^(7.0 / 3.0) +
                  phi8 * x^(8.0 / 3.0) + phi8L * x^(8.0 / 3.0) * lx
        phi_Ins = phi_TF2 + sigma1 * x^(8.0 / 3.0) + sigma2 * x^3.0 + sigma3 * x^(10.0 / 3.0) + sigma4 * x^(11.0 / 3.0)

        dphi_TF2 = phi1 * (1.0 / 3.0) * x^(-2.0 / 3.0) + phi2 * (2.0 / 3.0) * x^(-1.0 / 3.0) + phi3 +
                   phi4 * (4.0 / 3.0) * x^(1.0 / 3.0) + phi5 * (5.0 / 3.0) * x^(2.0 / 3.0) +
                   phi5L * ((5.0 / 3.0) * x^(2.0 / 3.0) * lx + x^(2.0 / 3.0)) +
                   phi6 * 2.0 * x + phi6L * (2.0 * x * lx + x) +
                   phi7 * (7.0 / 3.0) * x^(4.0 / 3.0) + phi8 * (8.0 / 3.0) * x^(5.0 / 3.0) +
                   phi8L * ((8.0 / 3.0) * x^(5.0 / 3.0) * lx + x^(5.0 / 3.0))
        dphi_Ins = dphi_TF2 + sigma1 * (8.0 / 3.0) * x^(5.0 / 3.0) + sigma2 * 3.0 * x^2.0 +
                   sigma3 * (10.0 / 3.0) * x^(7.0 / 3.0) + sigma4 * (11.0 / 3.0) * x^(8.0 / 3.0)

        return phiN * ((-5.0 / 3.0) * x^(-8.0 / 3.0) * phi_Ins + x^(-5.0 / 3.0) * dphi_Ins)
    end
end

@inline function _xas_phi_int_val(x, b0, b1, b2, b3, b4, cL, fdamp, fRD)
    T = promote_type(typeof(x), typeof(b0), typeof(cL))
    @Tlit T begin
        return b0 * x + b1 * log(x) - b2 / x - b3 / (2.0 * x^2.0) - b4 / (3.0 * x^3.0) +
               (2.0 * cL / fdamp) * atan((x - fRD) / (2.0 * fdamp))
    end
end

@inline function _xas_phi_int_deriv(x, b0, b1, b2, b3, b4, cL, fdamp, fRD)
    T = promote_type(typeof(x), typeof(b0), typeof(cL))
    @Tlit T begin
        return b0 + b1 / x + b2 / x^2.0 + b3 / x^3.0 + b4 / x^4.0 +
               4.0 * cL / (4.0 * fdamp^2.0 + (x - fRD)^2.0)
    end
end

@inline function _xas_phi_mrd_val(x, c0, c1, c2, c4, cL, fdamp, fRD)
    T = promote_type(typeof(x), typeof(c0), typeof(cL))
    @Tlit T begin
        cLovfda = cL / fdamp
        return c0 * x + 1.5 * c1 * x^(2.0 / 3.0) - c2 / x - (c4 / 3.0) / x^3.0 +
               cLovfda * atan((x - fRD) / fdamp)
    end
end

@inline function _xas_phi_mrd_deriv(x, c0, c1, c2, c4, cL, fdamp, fRD)
    T = promote_type(typeof(x), typeof(c0), typeof(cL))
    @Tlit T begin
        return c0 + c1 * x^(-1.0 / 3.0) + c2 / x^2.0 + c4 / x^4.0 +
               cL / (fdamp^2.0 + (x - fRD)^2.0)
    end
end

# ============================================================================================
# Amplitude ansatz pieces. Ported from `get_inspiral_Amp`/`get_mergerringdown_Amp` in ripple's
# `IMRPhenomXAS.py`; `_xas_amp_rd_deriv` reuses the exact derivative formula this package's own
# IMRPhenomD.jl already computes (and tests) for the structurally-identical
# exp(...)*Lorentzian ringdown-amplitude ansatz (IMRPhenomD's `d2`), just re-parameterized —
# verified algebraically equivalent during porting, not re-derived from scratch.
# ============================================================================================

@inline function _xas_amp_ins_val(x, A2, A3, A4, A5, A6, rho1, rho2, rho3)
    T = promote_type(typeof(x), typeof(A2), typeof(rho1))
    @Tlit T begin
        return 1.0 + A2 * x^(2.0 / 3.0) + A3 * x + A4 * x^(4.0 / 3.0) + A5 * x^(5.0 / 3.0) +
               A6 * x^2.0 + rho1 * x^(7.0 / 3.0) + rho2 * x^(8.0 / 3.0) + rho3 * x^3.0
    end
end

@inline function _xas_amp_ins_deriv(x, A2, A3, A4, A5, A6, rho1, rho2, rho3)
    T = promote_type(typeof(x), typeof(A2), typeof(rho1))
    @Tlit T begin
        return A2 * (2.0 / 3.0) * x^(-1.0 / 3.0) + A3 + A4 * (4.0 / 3.0) * x^(1.0 / 3.0) +
               A5 * (5.0 / 3.0) * x^(2.0 / 3.0) + A6 * 2.0 * x + rho1 * (7.0 / 3.0) * x^(4.0 / 3.0) +
               rho2 * (8.0 / 3.0) * x^(5.0 / 3.0) + rho3 * 3.0 * x^2.0
    end
end

@inline function _xas_amp_rd_val(x, gamma1, gamma2, gamma3, fdamp, fRD)
    T = promote_type(typeof(x), typeof(gamma1), typeof(fdamp))
    @Tlit T begin
        gammaR = gamma2 / (fdamp * gamma3)
        gammaD2 = (gamma3 * fdamp) * (gamma3 * fdamp)
        gammaD13 = fdamp * gamma1 * gamma3
        return exp(-(x - fRD) * gammaR) * gammaD13 / ((x - fRD) * (x - fRD) + gammaD2)
    end
end

# Same closed form as IMRPhenomD.jl's own `d2` (derivative of the same exp*Lorentzian ansatz at
# its ringdown-region matching point), re-parameterized to XAS's own fRD/fdamp/gamma1-3 and
# evaluated at a general `x` rather than the fixed `f3Interm`. See this file's module docstring.
@inline function _xas_amp_rd_deriv(x, gamma1, gamma2, gamma3, fdamp, fRD)
    T = promote_type(typeof(x), typeof(gamma1), typeof(fdamp))
    @Tlit T begin
        u = x - fRD
        D = fdamp * gamma3 * fdamp * gamma3
        numer = (-2.0 * fdamp * u * gamma3 * gamma1) / (u * u + D) - gamma2 * gamma1
        denom = exp(u * gamma2 / (fdamp * gamma3)) * (u * u + D)
        return numer / denom
    end
end
"""
    _xas_phi_coeffs(model::IMRPhenomXAS, mc, eta, chi1, chi2; kwargs...)

Frequency-independent part of the IMRPhenomXAS phase calculation, factored out of `Phi` exactly as
[`_phi_coeffs`](@ref) is for IMRPhenomD (see that function's docstring for the full rationale: this
split is both the architecturally correct design for a per-GPU-thread kernel — recomputing this
whole collocation-point/linear-solve block for every frequency bin would be a severe,
avoidable performance regression — and empirically required for `Enzyme.autodiff` to complete on
a KernelAbstractions kernel evaluating the phase, since the array-allocating `Phi` hangs
`Enzyme.Reverse` compilation indefinitely inside a `@kernel` body). Returns a `NamedTuple`
holding everything [`_xas_phi_scalar`](@ref) needs to evaluate the phase at a single frequency;
none of these fields depend on frequency.

Unlike IMRPhenomD (one join point), XAS has **two** C1-continuity join points (`f1`:
inspiral/intermediate, `f2`: intermediate/ringdown), so the returned coefficients include both
`(alpha0, alpha1)` (inspiral/intermediate join correction) and `(beta0, beta1)`
(intermediate/ringdown join correction), grouped alongside the ansatz coefficients they apply to
in nested `phi_ins`/`phi_int`/`phi_mrd` NamedTuples for readability.
"""
@inline function _xas_phi_coeffs(model::IMRPhenomXAS,
    mc,
    eta,
    chi1,
    chi2;
    fcutPar = 0.3,
    t0 = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
    phiRef = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
    GMsun_over_c3 = GMsun_over_c3,
    final_spin_override = NaN,
)
    # No @assert here: this is the hot path (per-walker GPU kernel call / Enzyme target).
    # Parameter-domain checks belong once, at the sampler/prior level -- call
    # `validate_phenomxas_params` explicitly for host-side sanity checks.
    T = promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2), typeof(t0), typeof(phiRef))

    # Keyword-derived scalars aren't reached by the `@Tlit` rewrite below (it only rewrites
    # literal numbers written directly in this file's source) -- cast them explicitly, matching
    # IMRPhenomD.jl's `_phi_coeffs` convention.
    fcutPar = T(fcutPar)
    GMsun_over_c3 = T(GMsun_over_c3)
    t0 = T(t0)
    phiRef = T(phiRef)
    final_spin_override = T(final_spin_override)

    phase_table = map(row -> T.(row), PhenomXAS_phase_coeff_table)

@Tlit T begin
    M = mc / (eta^(3.0 / 5.0))
    eta2 = eta * eta
    eta3 = eta2 * eta

    Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
    mm1 = 0.5 * (1.0 + Seta)
    mm2 = 0.5 * (1.0 - Seta)
    chi_eff = mm1 * chi1 + mm2 * chi2
    S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
    StotR = (mm1 * mm1 * chi1 + mm2 * mm2 * chi2) / (mm1 * mm1 + mm2 * mm2)
    chia = chi1 - chi2

    a, Erad = _xas_a_Erad(eta, Seta, chia, StotR)
    # `final_spin_override` (see `_xas_precessing_final_spin`'s docstring) only ever replaces the
    # ringdown/damping-frequency input `a` -- `fISCO` below deliberately keeps using the ALIGNED
    # `a`, matching LALSuite's own `IMRPhenomXPrecessingFinalSpinDeviation` (`fISCO` is set once,
    # earlier, from the aligned-spin final spin, and never recomputed from `afinal_prec`).
    a_ringdown = ifelse(isnan(final_spin_override), a, final_spin_override)
    fRD, fdamp = _xas_fRD_fdamp(a_ringdown, Erad)
    fISCO = _xas_fISCO(a)
    fMECO = _xas_fMECO(eta, S, chia, Seta)

    fIMmatch = 0.6 * (0.5 * fRD + fISCO)
    fINmatch = fMECO
    deltafMs_join = (fIMmatch - fINmatch) * 0.03
    f1 = fINmatch - deltafMs_join
    f2 = fIMmatch + 0.5 * deltafMs_join

    chi1L2 = chi1 * chi1
    chi1L3 = chi1L2 * chi1
    chi2L2 = chi2 * chi2
    chi2L3 = chi2L2 * chi2
    chi1L2L = chi1 * chi2

    # ---- TaylorF2(+pseudo-PN) inspiral phase coefficients (ported from ripple's
    # get_inspiral_phase; `Seta` plays the role of ripple's `delta`) ----
    phi0 = 1.0
    phi1 = 0.0
    phi2 = (3715.0 / 756.0 + (55.0 * eta) / 9.0) * pi^(2.0 / 3.0)
    phi3 = -16.0 * pi^2.0 +
           ((113.0 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) - 76.0 * (chi1 + chi2) * eta) / 6.0) * pi
    phi4 = (15293365.0 / 508032.0 + (27145.0 * eta) / 504.0 + (3085.0 * eta2) / 72.0) * pi^(4.0 / 3.0) +
           ((-5.0 * (81.0 * chi1L2 * (1.0 + Seta - 2.0 * eta) + 316.0 * chi1L2L * eta - 81.0 * chi2L2 * (-1.0 + Seta + 2.0 * eta))) / 16.0) * pi^(4.0 / 3.0)
    phi5 = 0.0
    phi5L = ((5.0 * (46374.0 - 6552.0 * eta) * pi) / 4536.0) * pi^(5.0 / 3.0) +
            ((-732985.0 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) - 560.0 * (-1213.0 * (chi1 + chi2) + 63.0 * (chi1 - chi2) * Seta) * eta + 85680.0 * (chi1 + chi2) * eta2) / 4536.0) * pi^(5.0 / 3.0)
    phi6L = (-6848.0 / 63.0) * pi^2.0
    phi6 = (
               11583231236531.0 / 4.69421568e9 -
               (5.0 * eta * (3147553127.0 + 588.0 * eta * (-45633.0 + 102260.0 * eta))) / 3.048192e6 -
               (6848.0 * 0.577215664901532860606512090082402431) / 21.0 -
               (640.0 * pi^2.0) / 3.0 +
               (2255.0 * eta * pi^2.0) / 12.0 -
               (13696.0 * log(2.0)) / 21.0 -
               (6848.0 * log(pi)) / 63.0
           ) * pi^2.0 +
           ((5.0 * (227.0 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) - 156.0 * (chi1 + chi2) * eta) * pi) / 3.0) * pi^2.0 +
           ((5.0 * (20.0 * chi1L2L * eta * (11763.0 + 12488.0 * eta) + 7.0 * chi2L2 * (-15103.0 * (-1.0 + Seta) + 2.0 * (-21683.0 + 6580.0 * Seta) * eta - 9808.0 * eta2) - 7.0 * chi1L2 * (-15103.0 * (1.0 + Seta) + 2.0 * (21683.0 + 6580.0 * Seta) * eta + 9808.0 * eta2))) / 4032.0) * pi^2.0
    phi7 = ((5.0 * (15419335.0 + 168.0 * (75703.0 - 29618.0 * eta) * eta) * pi) / 254016.0) * pi^(7.0 / 3.0) +
           ((5.0 * (-5030016755.0 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) + 4.0 * (2113331119.0 * (chi1 + chi2) + 675484362.0 * (chi1 - chi2) * Seta) * eta - 1008.0 * (208433.0 * (chi1 + chi2) + 25011.0 * (chi1 - chi2) * Seta) * eta2 + 90514368.0 * (chi1 + chi2) * eta3)) / 6.096384e6) * pi^(7.0 / 3.0) +
           (-5.0 * (57.0 * chi1L2 * (1.0 + Seta - 2.0 * eta) + 220.0 * chi1L2L * eta - 57.0 * chi2L2 * (-1.0 + Seta + 2.0 * eta)) * pi) * pi^(7.0 / 3.0) +
           ((14585.0 * (-(chi2L3 * (-1.0 + Seta)) + chi1L3 * (1.0 + Seta)) - 5.0 * (chi2L3 * (8819.0 - 2985.0 * Seta) + 8439.0 * chi1 * chi2L2 * (-1.0 + Seta) - 8439.0 * chi1L2 * chi2 * (1.0 + Seta) + chi1L3 * (8819.0 + 2985.0 * Seta)) * eta + 40.0 * (chi1 + chi2) * (17.0 * chi1L2 - 14.0 * chi1L2L + 17.0 * chi2L2) * eta2) / 48.0) * pi^(7.0 / 3.0)
    phi8coef = (-5.0 * (1263141.0 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) - 2.0 * (794075.0 * (chi1 + chi2) + 178533.0 * (chi1 - chi2) * Seta) * eta + 94344.0 * (chi1 + chi2) * eta2) * pi) / 9072.0
    phi8 = (phi8coef * (-1.0 + log(pi))) * pi^(8.0 / 3.0)
    phi8L = phi8coef * pi^(8.0 / 3.0)

    phiN = -(3.0 * pi^(-5.0 / 3.0)) / 128.0

    # ---- sigma1..4 (inspiral pseudo-PN coefficients) from a 4-collocation-point fit ----
    fMs_PhaseInsMin = 0.0026
    fMs_PhaseInsMax = 1.020 * fMECO
    deltax_ins = fMs_PhaseInsMax - fMs_PhaseInsMin
    xmin_ins = fMs_PhaseInsMin
    CP_phase_Ins0 = 0.0 * deltax_ins + xmin_ins
    CP_phase_Ins1 = 0.25 * deltax_ins + xmin_ins
    CP_phase_Ins2 = 0.75 * deltax_ins + xmin_ins
    CP_phase_Ins3 = 1.0 * deltax_ins + xmin_ins

    CV_phase_Ins0 = _xas_phase_CV(phase_table[1], eta, S, chia, Seta)
    CV_phase_Ins1 = _xas_phase_CV(phase_table[2], eta, S, chia, Seta)
    CV_phase_Ins2 = _xas_phase_CV(phase_table[3], eta, S, chia, Seta)
    CV_phase_Ins3 = _xas_phase_CV(phase_table[4], eta, S, chia, Seta)
    CV_phase_Ins0 = CV_phase_Ins0 + CV_phase_Ins2
    CV_phase_Ins1 = CV_phase_Ins1 + CV_phase_Ins2
    CV_phase_Ins3 = CV_phase_Ins3 + CV_phase_Ins2

    A_ins = (
        (one(T), CP_phase_Ins0^(1.0 / 3.0), CP_phase_Ins0^(2.0 / 3.0), CP_phase_Ins0),
        (one(T), CP_phase_Ins1^(1.0 / 3.0), CP_phase_Ins1^(2.0 / 3.0), CP_phase_Ins1),
        (one(T), CP_phase_Ins2^(1.0 / 3.0), CP_phase_Ins2^(2.0 / 3.0), CP_phase_Ins2),
        (one(T), CP_phase_Ins3^(1.0 / 3.0), CP_phase_Ins3^(2.0 / 3.0), CP_phase_Ins3),
    )
    b_ins = (CV_phase_Ins0, CV_phase_Ins1, CV_phase_Ins2, CV_phase_Ins3)
    coeffs_Ins = _xas_linsolve(A_ins, b_ins)
    sigma1 = (-5.0 / 3.0) * coeffs_Ins[1]
    sigma2 = (-5.0 / 4.0) * coeffs_Ins[2]
    sigma3 = -1.0 * coeffs_Ins[3]
    sigma4 = (-5.0 / 6.0) * coeffs_Ins[4]

    phi_ins_f1 = _xas_phi_ins_val(f1, phi0, phi1, phi2, phi3, phi4, phi5, phi5L, phi6, phi6L, phi7, phi8, phi8L, sigma1, sigma2, sigma3, sigma4, phiN)
    dphi_ins_f1 = _xas_phi_ins_deriv(f1, phi0, phi1, phi2, phi3, phi4, phi5, phi5L, phi6, phi6L, phi7, phi8, phi8L, sigma1, sigma2, sigma3, sigma4, phiN)

    # ---- Merger-ringdown phase (c0,c1,c2,c4,cRD) from a 5-collocation-point fit ----
    dphase0 = 5.0 / (128.0 * pi^(5.0 / 3.0))
    fMs_PhaseRDMin = fIMmatch
    fMs_PhaseRDMax = fRD + 1.25 * fdamp
    deltax_rd = fMs_PhaseRDMax - fMs_PhaseRDMin
    xmin_rd = fMs_PhaseRDMin
    inv2sqrt2 = 1.0 / (2.0 * sqrt(2.0))
    CP_phase_RD0 = 0.0 * deltax_rd + xmin_rd
    CP_phase_RD1 = (0.5 - inv2sqrt2) * deltax_rd + xmin_rd
    CP_phase_RD2 = 0.5 * deltax_rd + xmin_rd
    CP_phase_RD3 = fRD
    CP_phase_RD4 = 1.0 * deltax_rd + xmin_rd

    CV_phase_RD0 = _xas_phase_CV(phase_table[9], eta, StotR, chia, Seta)
    CV_phase_RD1 = _xas_phase_CV(phase_table[10], eta, StotR, chia, Seta)
    CV_phase_RD2 = _xas_phase_CV(phase_table[11], eta, StotR, chia, Seta)
    CV_phase_RD3 = _xas_phase_CV(phase_table[12], eta, StotR, chia, Seta)
    CV_phase_RD4 = _xas_phase_CV(phase_table[13], eta, StotR, chia, Seta)
    CV_phase_RD4 = CV_phase_RD4 + CV_phase_RD3
    CV_phase_RD2 = CV_phase_RD2 + CV_phase_RD3
    CV_phase_RD1 = CV_phase_RD1 + CV_phase_RD3
    CV_phase_RD0 = CV_phase_RD0 + CV_phase_RD1

    A_rd = (
        (one(T), CP_phase_RD0^(-1.0 / 3.0), CP_phase_RD0^(-2.0), CP_phase_RD0^(-4.0), -dphase0 / (fdamp * fdamp + (CP_phase_RD0 - fRD) * (CP_phase_RD0 - fRD))),
        (one(T), CP_phase_RD1^(-1.0 / 3.0), CP_phase_RD1^(-2.0), CP_phase_RD1^(-4.0), -dphase0 / (fdamp * fdamp + (CP_phase_RD1 - fRD) * (CP_phase_RD1 - fRD))),
        (one(T), CP_phase_RD2^(-1.0 / 3.0), CP_phase_RD2^(-2.0), CP_phase_RD2^(-4.0), -dphase0 / (fdamp * fdamp + (CP_phase_RD2 - fRD) * (CP_phase_RD2 - fRD))),
        (one(T), CP_phase_RD3^(-1.0 / 3.0), CP_phase_RD3^(-2.0), CP_phase_RD3^(-4.0), -dphase0 / (fdamp * fdamp + (CP_phase_RD3 - fRD) * (CP_phase_RD3 - fRD))),
        (one(T), CP_phase_RD4^(-1.0 / 3.0), CP_phase_RD4^(-2.0), CP_phase_RD4^(-4.0), -dphase0 / (fdamp * fdamp + (CP_phase_RD4 - fRD) * (CP_phase_RD4 - fRD))),
    )
    b_rd = (CV_phase_RD0, CV_phase_RD1, CV_phase_RD2, CV_phase_RD3, CV_phase_RD4)
    coeffs_RD = _xas_linsolve(A_rd, b_rd)
    c0 = coeffs_RD[1]
    c1 = coeffs_RD[2]
    c2 = coeffs_RD[3]
    c4 = coeffs_RD[4]
    cRD = coeffs_RD[5]
    cL = -dphase0 * cRD

    phi_mrd_f2 = _xas_phi_mrd_val(f2, c0, c1, c2, c4, cL, fdamp, fRD)
    dphi_mrd_f2 = _xas_phi_mrd_deriv(f2, c0, c1, c2, c4, cL, fdamp, fRD)

    # ---- Intermediate phase (b0..b4) from a 5-collocation-point fit ----
    deltax_int = f2 - f1
    xmin_int = f1
    CP_phase_Int0 = 0.0 * deltax_int + xmin_int
    CP_phase_Int1 = (0.5 - inv2sqrt2) * deltax_int + xmin_int
    CP_phase_Int2 = 0.5 * deltax_int + xmin_int
    CP_phase_Int3 = (0.5 + inv2sqrt2) * deltax_int + xmin_int
    CP_phase_Int4 = 1.0 * deltax_int + xmin_int

    v2IMmRDv4 = _xas_phase_CV(phase_table[5], eta, StotR, chia, Seta)
    v3IMmRDv4 = _xas_phase_CV(phase_table[6], eta, StotR, chia, Seta)
    v2IM = _xas_phase_CV(phase_table[7], eta, StotR, chia, Seta)
    d43 = _xas_phase_CV(phase_table[8], eta, StotR, chia, Seta)

    CV_phase_Int0 = dphi_ins_f1
    CV_phase_Int1 = 0.75 * (v2IMmRDv4 + CV_phase_RD3) + 0.25 * v2IM
    CV_phase_Int2 = v3IMmRDv4 + CV_phase_RD3
    CV_phase_Int3 = d43 + CV_phase_Int2
    CV_phase_Int4 = CV_phase_RD0

    r0 = fRD / CP_phase_Int0
    r1 = fRD / CP_phase_Int1
    r2 = fRD / CP_phase_Int2
    r3 = fRD / CP_phase_Int3
    r4 = fRD / CP_phase_Int4
    A_int = (
        (one(T), r0, r0 * r0, r0 * r0 * r0, r0 * r0 * r0 * r0),
        (one(T), r1, r1 * r1, r1 * r1 * r1, r1 * r1 * r1 * r1),
        (one(T), r2, r2 * r2, r2 * r2 * r2, r2 * r2 * r2 * r2),
        (one(T), r3, r3 * r3, r3 * r3 * r3, r3 * r3 * r3 * r3),
        (one(T), r4, r4 * r4, r4 * r4 * r4, r4 * r4 * r4 * r4),
    )
    b_int = (
        CV_phase_Int0 - (4.0 * cL) / (4.0 * fdamp * fdamp + (CP_phase_Int0 - fRD) * (CP_phase_Int0 - fRD)),
        CV_phase_Int1 - (4.0 * cL) / (4.0 * fdamp * fdamp + (CP_phase_Int1 - fRD) * (CP_phase_Int1 - fRD)),
        CV_phase_Int2 - (4.0 * cL) / (4.0 * fdamp * fdamp + (CP_phase_Int2 - fRD) * (CP_phase_Int2 - fRD)),
        CV_phase_Int3 - (4.0 * cL) / (4.0 * fdamp * fdamp + (CP_phase_Int3 - fRD) * (CP_phase_Int3 - fRD)),
        CV_phase_Int4 - (4.0 * cL) / (4.0 * fdamp * fdamp + (CP_phase_Int4 - fRD) * (CP_phase_Int4 - fRD)),
    )
    coeffs_Int = _xas_linsolve(A_int, b_int)
    b0 = coeffs_Int[1]
    b1 = coeffs_Int[2] * fRD
    b2 = coeffs_Int[3] * fRD * fRD
    b3 = coeffs_Int[4] * fRD * fRD * fRD
    b4 = coeffs_Int[5] * fRD * fRD * fRD * fRD

    # ---- C1 continuity: inspiral/intermediate join at f1, intermediate/ringdown join at f2 ----
    phi_int_f1_raw = _xas_phi_int_val(f1, b0, b1, b2, b3, b4, cL, fdamp, fRD)
    dphi_int_f1_raw = _xas_phi_int_deriv(f1, b0, b1, b2, b3, b4, cL, fdamp, fRD)
    alpha1 = dphi_ins_f1 - dphi_int_f1_raw
    alpha0 = phi_ins_f1 - phi_int_f1_raw - alpha1 * f1

    phi_int_f2_raw = _xas_phi_int_val(f2, b0, b1, b2, b3, b4, cL, fdamp, fRD)
    dphi_int_f2_raw = _xas_phi_int_deriv(f2, b0, b1, b2, b3, b4, cL, fdamp, fRD)
    phi_int_f2_corrected = phi_int_f2_raw + alpha1 * f2 + alpha0
    dphi_int_f2_corrected = dphi_int_f2_raw + alpha1

    beta1 = dphi_int_f2_corrected - dphi_mrd_f2
    beta0 = phi_int_f2_corrected - phi_mrd_f2 - beta1 * f2

    return (
        T = T, M = M, GMsun_over_c3 = GMsun_over_c3, f1 = f1, f2 = f2, fcutPar = fcutPar,
        eta = eta, t0 = t0, phiRef = phiRef, fdamp = fdamp, fRD = fRD,
        phi_ins = (
            phi0 = phi0, phi1 = phi1, phi2 = phi2, phi3 = phi3, phi4 = phi4, phi5 = phi5,
            phi5L = phi5L, phi6 = phi6, phi6L = phi6L, phi7 = phi7, phi8 = phi8, phi8L = phi8L,
            sigma1 = sigma1, sigma2 = sigma2, sigma3 = sigma3, sigma4 = sigma4, phiN = phiN,
        ),
        phi_int = (b0 = b0, b1 = b1, b2 = b2, b3 = b3, b4 = b4, alpha0 = alpha0, alpha1 = alpha1),
        phi_mrd = (c0 = c0, c1 = c1, c2 = c2, c4 = c4, cL = cL, beta0 = beta0, beta1 = beta1),
    )
end # @Tlit T
end

@inline function _xas_linb_fit(eta, S, dchi, delta)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi), typeof(delta))
    @Tlit T begin
        eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta; eta5 = eta4 * eta; eta6 = eta5 * eta
        S2 = S * S; S3 = S2 * S; S4 = S3 * S
        noSpin = 3155.1635543201924 + 1257.9949740608242 * eta - 32243.28428870599 * eta2 +
                 347213.65466875216 * eta3 - 1.9223851649491738e6 * eta4 +
                 5.3035911346921865e6 * eta5 - 5.789128656876938e6 * eta6
        eqSpin = (-24.181508118588667 + 115.49264174560281 * eta - 380.19778216022763 * eta2) * S +
                 (24.72585609641552 - 328.3762360751952 * eta + 725.6024119989094 * eta2) * S2 +
                 (23.404604124552 - 646.3410199799737 * eta + 1941.8836639529036 * eta2) * S3 +
                 (-12.814828278938885 - 325.92980012408367 * eta + 1320.102640190539 * eta2) * S4
        uneqSpin = -148.17317525117338 * dchi * delta * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

@inline function _xas_psi4_to_strain_fit(eta, S, dchi)
    T = promote_type(typeof(eta), typeof(S), typeof(dchi))
    @Tlit T begin
        eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta3 * eta
        S2 = S * S; S3 = S2 * S; S4 = S3 * S
        noSpin = 13.39320482758057 - 175.42481512989315 * eta + 2097.425116152503 * eta2 -
                 9862.84178637907 * eta3 + 16026.897939722587 * eta4
        eqSpin = (4.7895602776763 - 163.04871764530466 * eta + 609.5575850476959 * eta2) * S +
                 (1.3934428041390161 - 97.51812681228478 * eta + 376.9200932531847 * eta2) * S2 +
                 (15.649521097877374 + 137.33317057388916 * eta - 755.9566456906406 * eta2) * S3 +
                 (13.097315867845788 + 149.30405703643288 * eta - 764.5242164872267 * eta2) * S4
        uneqSpin = 105.37711654943146 * dchi * sqrt(1.0 - 4.0 * eta) * eta2
        return noSpin + eqSpin + uneqSpin
    end
end

"""
    _xas_hybrid_time_shift(eta, chi1, chi2; final_spin_override = NaN)

Ringdown-region merger-time alignment constant matching LALSuite's `IMRPhenomX_TimeShift_22`
hybrid-alignment convention (`LALSimIMRPhenomX_internals.c`), ready to pass directly as `Phi`'s
`t0` keyword (already sign-corrected: `Phi` applies `-t0 * Mf`, LALSuite's own `linb` fit is added
as `+linb * Mf`, so this returns the negated LALSuite `tshift`).

Combines two parameter-space fits calibrated directly on NR hybrids (`_xas_linb_fit`,
`XLALSimIMRPhenomXLinb`; `_xas_psi4_to_strain_fit`, `XLALSimIMRPhenomXPsi4ToStrain`) with this
package's own ringdown phase-derivative ansatz — including its `beta1` intermediate/ringdown
continuity correction (the analogue of LALSuite's `C2MRD`, `LALSimIMRPhenomX_internals.c:2598`,
`2696`; the earlier "ringdown ansatz alone, evaluated at `f2`" approximation omitted this,
leaving a ~0.0004 rad/Hz `dPhi/df` residual) — evaluated at `fRD - fdamp`, plus LALSuite's own
hardcoded `500M` psi4-peak-to-waveform-end offset. Matches LALSuite's actual `IMRPhenomXAS`/
`IMRPhenomXP` time alignment to machine precision (`dPhi/df` residual ~2e-13 rad/Hz, against LALSuite), not derived from first
principles here.

`final_spin_override`: same convention as [`_xas_phi_coeffs`](@ref)'s own keyword — pass the
precessing final spin (from [`_xas_precessing_final_spin`](@ref)) for `IMRPhenomXP`/
`IMRPhenomXPHM` twisting-up, leave the default `NaN` for plain aligned-spin `IMRPhenomXAS`.
"""
@inline function _xas_hybrid_time_shift(eta, chi1, chi2; final_spin_override = NaN)
    T = promote_type(typeof(eta), typeof(chi1), typeof(chi2))
    eta = T(eta)
    chi1 = T(chi1)
    chi2 = T(chi2)
    final_spin_override = T(final_spin_override)
@Tlit T begin
    Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
    mm1 = 0.5 * (1.0 + Seta)
    mm2 = 0.5 * (1.0 - Seta)
    StotR = (mm1 * mm1 * chi1 + mm2 * mm2 * chi2) / (mm1 * mm1 + mm2 * mm2)
    dchi = chi1 - chi2

    coeffs = _xas_phi_coeffs(IMRPhenomXAS(), one(T), eta, chi1, chi2;
        final_spin_override = final_spin_override)
    fRD, fdamp = coeffs.fRD, coeffs.fdamp
    c0, c1, c2, c4, cL = coeffs.phi_mrd.c0, coeffs.phi_mrd.c1, coeffs.phi_mrd.c2,
        coeffs.phi_mrd.c4, coeffs.phi_mrd.cL
    beta1 = coeffs.phi_mrd.beta1

    frefFit = fRD - fdamp
    dphi22Ref = (_xas_phi_mrd_deriv(frefFit, c0, c1, c2, c4, cL, fdamp, fRD) + beta1) / eta

    linb = _xas_linb_fit(eta, StotR, dchi, Seta)
    psi4tostrain = _xas_psi4_to_strain_fit(eta, StotR, dchi)
    tshift = linb - dphi22Ref - 2.0 * pi * (500.0 + psi4tostrain)
    return -tshift
end # @Tlit T
end

"""
    _xas_phi_scalar(coeffs, f)

Non-allocating IMRPhenomXAS phase evaluation at a single frequency `f` (not a vector), using the
frequency-independent coefficients from [`_xas_phi_coeffs`](@ref) -- the XAS analogue of
IMRPhenomD's [`_phi_scalar`](@ref). This is the scalar core that `Phi` broadcasts over an
`AbstractVector` of frequencies (preserving the existing array-in/array-out API for all current
callers), and that the fused KernelAbstractions likelihood kernel calls directly in a per-thread
frequency loop with zero array allocation.

Generalizes IMRPhenomD's single-join-point three-way `ifelse` (inspiral / intermediate / ringdown,
split at `fInsJoin`/`fMRDJoin`) to XAS's two join points `f1`/`f2`, with the intermediate and
ringdown branches picking up the `(alpha0, alpha1)`/`(beta0, beta1)` linear-in-frequency
C1-continuity corrections computed once in `_xas_phi_coeffs`.
"""
@inline function _xas_phi_scalar(coeffs, f)
    T = coeffs.T
@Tlit T begin
    f_safe = max(f, eps(T))
    fgrid = coeffs.M * coeffs.GMsun_over_c3 * f_safe

    phis = ifelse(
        fgrid < coeffs.f1,
        _xas_phi_ins_val(fgrid,
            coeffs.phi_ins.phi0, coeffs.phi_ins.phi1, coeffs.phi_ins.phi2, coeffs.phi_ins.phi3,
            coeffs.phi_ins.phi4, coeffs.phi_ins.phi5, coeffs.phi_ins.phi5L, coeffs.phi_ins.phi6,
            coeffs.phi_ins.phi6L, coeffs.phi_ins.phi7, coeffs.phi_ins.phi8, coeffs.phi_ins.phi8L,
            coeffs.phi_ins.sigma1, coeffs.phi_ins.sigma2, coeffs.phi_ins.sigma3, coeffs.phi_ins.sigma4,
            coeffs.phi_ins.phiN),
        ifelse(
            fgrid < coeffs.f2,
            _xas_phi_int_val(fgrid, coeffs.phi_int.b0, coeffs.phi_int.b1, coeffs.phi_int.b2,
                coeffs.phi_int.b3, coeffs.phi_int.b4, coeffs.phi_mrd.cL, coeffs.fdamp, coeffs.fRD) +
            coeffs.phi_int.alpha1 * fgrid + coeffs.phi_int.alpha0,
            ifelse(
                fgrid < coeffs.fcutPar,
                _xas_phi_mrd_val(fgrid, coeffs.phi_mrd.c0, coeffs.phi_mrd.c1, coeffs.phi_mrd.c2,
                    coeffs.phi_mrd.c4, coeffs.phi_mrd.cL, coeffs.fdamp, coeffs.fRD) +
                coeffs.phi_mrd.beta0 + coeffs.phi_mrd.beta1 * fgrid,
                zero(T),
            ),
        ),
    ) / coeffs.eta

    return phis + ifelse(fgrid < coeffs.fcutPar, -coeffs.t0 * (fgrid - zero(T)) - coeffs.phiRef, zero(T))
end # @Tlit T
end

"""
    Phi(model::IMRPhenomXAS, f, mc, eta, chi1, chi2; kwargs...)

IMRPhenomXAS frequency-domain phase, dominant (2,2) mode, following the same call contract and
`t0`/`phiRef` external-alignment convention as `Phi(::IMRPhenomD, ...)` (no internal
time-of-coalescence alignment is performed; see that function's docstring for the general
`Ampl`/`Phi` contract this package uses). `f` may be a scalar (returns a scalar, non-allocating)
or an `AbstractVector` (returns a same-length `Vector`) -- computes the frequency-independent
coefficients once via [`_xas_phi_coeffs`](@ref), then evaluates [`_xas_phi_scalar`](@ref) at each
frequency. Callers evaluating many frequencies for one set of parameters where per-call
allocation matters (e.g. inside a GPU kernel) should call `_xas_phi_coeffs` once and
`_xas_phi_scalar` per point directly instead of going through this array-oriented wrapper.

Implementation note: like `Phi(::IMRPhenomD, ...)`, this deliberately uses an explicit
`similar`-then-loop instead of broadcasting `_xas_phi_scalar` over `f`, for the same
Enzyme-activity-analysis reason documented on that function.

No parameter-domain assertions run here — this is the hot path; see `validate_phenomxas_params`
for an opt-in host-side check.

# Keyword arguments
- `fcutPar = 0.3`: dimensionless (Mf) frequency beyond which the waveform is defined to be
  zero. XAS's own natural validity cutoff (ripple/LALSimulation's `fM_CUT`), differing from
  IMRPhenomD's `fcutPar = 0.2` default.
- `t0`, `phiRef`: external linear-in-frequency / constant phase offsets, applied exactly as in
  `Phi(::IMRPhenomD, ...)`.
- `GMsun_over_c3`: same constant/units convention as `Phi(::IMRPhenomD, ...)`.
- `final_spin_override = NaN`: when finite, replaces the ringdown/damping-frequency final spin `a`
  (not `fISCO`) -- see [`_xas_precessing_final_spin`](@ref)'s docstring. Used by `IMRPhenomXP`/
  `IMRPhenomXPHM` to apply LALSuite's precessing final spin; leave at the default `NaN` for the
  ordinary aligned-spin model (the default recovers the pre-existing behavior exactly).
"""
function Phi(model::IMRPhenomXAS,
    f,
    mc,
    eta,
    chi1,
    chi2;
    fcutPar = 0.3,
    t0 = zero(eltype(f)),
    phiRef = zero(eltype(f)),
    GMsun_over_c3 = GMsun_over_c3,
    final_spin_override = NaN,
)
    coeffs = _xas_phi_coeffs(model, mc, eta, chi1, chi2;
        fcutPar = fcutPar, t0 = t0, phiRef = phiRef, GMsun_over_c3 = GMsun_over_c3,
        final_spin_override = final_spin_override)
    if f isa AbstractArray
        out = similar(f, coeffs.T)
        @inbounds for i in eachindex(f)
            out[i] = _xas_phi_scalar(coeffs, f[i])
        end
        return out
    else
        return _xas_phi_scalar(coeffs, f)
    end
end

"""
    _xas_ampl_coeffs(model::IMRPhenomXAS, mc, eta, chi1, chi2, dL; kwargs...)

Frequency-independent part of the IMRPhenomXAS amplitude calculation, factored out of `Ampl` for the
same reasons as [`_xas_phi_coeffs`](@ref) (see its docstring, and IMRPhenomD's
[`_ampl_coeffs`](@ref)): computing this block once per waveform instead of once per frequency bin
is both the correct design for a per-GPU-thread kernel and empirically required for
`Enzyme.autodiff` to complete on a KernelAbstractions kernel that evaluates the amplitude.
"""
@inline function _xas_ampl_coeffs(model::IMRPhenomXAS,
    mc,
    eta,
    chi1,
    chi2,
    dL;
    fcutPar = 0.3,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
    final_spin_override = NaN,
)
    # No @assert here: this is the hot path (per-walker GPU kernel call / Enzyme target).
    # Call `validate_phenomxas_params(eta, chi1, chi2; dL=dL)` explicitly for host-side checks.
    T = promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2), typeof(dL))

    fcutPar = T(fcutPar)
    GMsun_over_c3 = T(GMsun_over_c3)
    GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc)
    final_spin_override = T(final_spin_override)

    amp_table = map(row -> T.(row), PhenomXAS_amp_coeff_table)

@Tlit T begin
    M = mc / (eta^(3.0 / 5.0))
    eta2 = eta * eta

    Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
    mm1 = 0.5 * (1.0 + Seta)
    mm2 = 0.5 * (1.0 - Seta)
    chi_eff = mm1 * chi1 + mm2 * chi2
    S = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
    StotR = (mm1 * mm1 * chi1 + mm2 * mm2 * chi2) / (mm1 * mm1 + mm2 * mm2)
    chia = chi1 - chi2

    a, Erad = _xas_a_Erad(eta, Seta, chia, StotR)
    # See the matching comment in `_xas_phi_coeffs`: only fRD/fdamp are overridden, not fISCO.
    a_ringdown = ifelse(isnan(final_spin_override), a, final_spin_override)
    fRD, fdamp = _xas_fRD_fdamp(a_ringdown, Erad)
    fISCO = _xas_fISCO(a)
    fMECO = _xas_fMECO(eta, S, chia, Seta)

    fA1 = fMECO + 0.25 * (fISCO - fMECO)

    # ---- Inspiral amplitude: closed-form PN coefficients A2..A6 + pseudo-PN rho1..rho3 ----
    A2 = ((-969.0 + 1804.0 * eta) / 672.0) * pi^(2.0 / 3.0)
    A3 = ((81.0 * (chi1 + chi2) + 81.0 * chi1 * Seta - 81.0 * chi2 * Seta - 44.0 * (chi1 + chi2) * eta) / 48.0) * pi
    A4 = (
        (-27312085.0 - 10287648.0 * chi1^2.0 * (1.0 + Seta) +
         24.0 * (
             428652.0 * chi2^2.0 * (-1.0 + Seta) +
             (-1975055.0 + 10584.0 * (81.0 * chi1^2.0 - 94.0 * chi1 * chi2 + 81.0 * chi2^2.0)) * eta +
             1473794.0 * eta2
         )) / 8.128512e6
    ) * pi^(4.0 / 3.0)
    A5 = (
        (-6048.0 * chi1^2.0 * chi1 * (-1.0 - Seta + (3.0 + Seta) * eta) +
         chi2 * (-((287213.0 + 6048.0 * chi2^2.0) * (-1.0 + Seta)) +
                 4.0 * (-93414.0 + 1512.0 * chi2^2.0 * (-3.0 + Seta) + 2083.0 * Seta) * eta -
                 35632.0 * eta2) +
         chi1 * (287213.0 * (1.0 + Seta) - 4.0 * eta * (93414.0 + 2083.0 * Seta + 8908.0 * eta)) +
         42840.0 * (-1.0 + 4.0 * eta) * pi) / 32256.0
    ) * pi^(5.0 / 3.0)
    A6 = (
        (-1242641879927.0 +
         12.0 * (
             28.0 * (-3248849057.0 + 11088.0 * (163199.0 * chi1^2.0 - 266498.0 * chi1 * chi2 + 163199.0 * chi2^2.0)) * eta2 +
             27026893936.0 * eta2 * eta -
             116424.0 * (
                 147117.0 * (-(chi2^2.0 * (-1.0 + Seta)) + chi1^2.0 * (1.0 + Seta)) +
                 60928.0 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) * pi
             ) +
             eta * (
                 545384828789.0 -
                 77616.0 * (
                     638642.0 * chi1 * chi2 + chi1^2.0 * (-158633.0 + 282718.0 * Seta) -
                     chi2^2.0 * (158633.0 + 282718.0 * Seta) - 107520.0 * (chi1 + chi2) * pi +
                     275520.0 * pi^2.0
                 )
             )
         )) / 6.0085960704e10
    ) * pi^2.0

    CV_Amp_Ins0 = _xas_amp_CV(amp_table[1], eta, S, chia, Seta)
    CV_Amp_Ins1 = _xas_amp_CV(amp_table[2], eta, S, chia, Seta)
    CV_Amp_Ins2 = _xas_amp_CV(amp_table[3], eta, S, chia, Seta)

    CP_Amp_Ins0 = 0.50 * fA1
    CP_Amp_Ins1 = 0.75 * fA1
    CP_Amp_Ins2 = 1.00 * fA1

    cbrt0 = cbrt(CP_Amp_Ins0)
    cbrt1 = cbrt(CP_Amp_Ins1)
    cbrt2 = cbrt(CP_Amp_Ins2)
    rho_den = CP_Amp_Ins0^(7.0 / 3.0) * (cbrt0 - cbrt1) * CP_Amp_Ins1^(7.0 / 3.0) *
              (cbrt0 - cbrt2) * (cbrt1 - cbrt2) * CP_Amp_Ins2^(7.0 / 3.0)

    rho1 = (
        -(CP_Amp_Ins1^(8.0 / 3.0) * CP_Amp_Ins2^3.0 * CV_Amp_Ins0) +
        CP_Amp_Ins1^3.0 * CP_Amp_Ins2^(8.0 / 3.0) * CV_Amp_Ins0 +
        CP_Amp_Ins0^(8.0 / 3.0) * CP_Amp_Ins2^3.0 * CV_Amp_Ins1 -
        CP_Amp_Ins0^3.0 * CP_Amp_Ins2^(8.0 / 3.0) * CV_Amp_Ins1 -
        CP_Amp_Ins0^(8.0 / 3.0) * CP_Amp_Ins1^3.0 * CV_Amp_Ins2 +
        CP_Amp_Ins0^3.0 * CP_Amp_Ins1^(8.0 / 3.0) * CV_Amp_Ins2
    ) / rho_den
    rho2 = (
        CP_Amp_Ins1^(7.0 / 3.0) * CP_Amp_Ins2^3.0 * CV_Amp_Ins0 -
        CP_Amp_Ins1^3.0 * CP_Amp_Ins2^(7.0 / 3.0) * CV_Amp_Ins0 -
        CP_Amp_Ins0^(7.0 / 3.0) * CP_Amp_Ins2^3.0 * CV_Amp_Ins1 +
        CP_Amp_Ins0^3.0 * CP_Amp_Ins2^(7.0 / 3.0) * CV_Amp_Ins1 +
        CP_Amp_Ins0^(7.0 / 3.0) * CP_Amp_Ins1^3.0 * CV_Amp_Ins2 -
        CP_Amp_Ins0^3.0 * CP_Amp_Ins1^(7.0 / 3.0) * CV_Amp_Ins2
    ) / rho_den
    rho3 = (
        CP_Amp_Ins1^(8.0 / 3.0) * CP_Amp_Ins2^(7.0 / 3.0) * CV_Amp_Ins0 -
        CP_Amp_Ins1^(7.0 / 3.0) * CP_Amp_Ins2^(8.0 / 3.0) * CV_Amp_Ins0 -
        CP_Amp_Ins0^(8.0 / 3.0) * CP_Amp_Ins2^(7.0 / 3.0) * CV_Amp_Ins1 +
        CP_Amp_Ins0^(7.0 / 3.0) * CP_Amp_Ins2^(8.0 / 3.0) * CV_Amp_Ins1 +
        CP_Amp_Ins0^(8.0 / 3.0) * CP_Amp_Ins1^(7.0 / 3.0) * CV_Amp_Ins2 -
        CP_Amp_Ins0^(7.0 / 3.0) * CP_Amp_Ins1^(8.0 / 3.0) * CV_Amp_Ins2
    ) / rho_den

    # ---- Merger-ringdown amplitude: gamma1..3, v1RD, and the RD-region matching frequency ----
    gamma2 = _xas_amp_CV(amp_table[5], eta, StotR, chia, Seta)
    gamma3 = _xas_amp_CV(amp_table[6], eta, StotR, chia, Seta)
    v1RD = _xas_amp_CV(amp_table[7], eta, StotR, chia, Seta)

    fA4 = if gamma2 <= 1.0
        abs(fRD + fdamp * gamma3 * (sqrt(1.0 - gamma2 * gamma2) - 1.0) / gamma2)
    else
        abs(fRD - fdamp * gamma3 / gamma2)
    end

    gamma1 = (v1RD / (fdamp * gamma3)) *
             (fA4 * fA4 - 2.0 * fA4 * fRD + fRD * fRD + fdamp * fdamp * gamma3 * gamma3) *
             exp((fA4 - fRD) * gamma2 / (fdamp * gamma3))

    # ---- Intermediate amplitude: reciprocal-rational-function fit via boundary
    # value+derivative matching (delta0..delta4), ported directly from ripple's closed-form
    # reconstruction (get_intermediate_Amp) ----
    inspFA1 = _xas_amp_ins_val(fA1, A2, A3, A4, A5, A6, rho1, rho2, rho3)
    rawd1 = _xas_amp_ins_deriv(fA1, A2, A3, A4, A5, A6, rho1, rho2, rho3)
    rdFA4 = _xas_amp_rd_val(fA4, gamma1, gamma2, gamma3, fdamp, fRD)
    rawd4 = _xas_amp_rd_deriv(fA4, gamma1, gamma2, gamma3, fdamp, fRD)

    d1 = (7.0 / 6.0) * fA1^(1.0 / 6.0) / inspFA1 - fA1^(7.0 / 6.0) * rawd1 / (inspFA1 * inspFA1)
    d4 = (7.0 / 6.0) * fA4^(1.0 / 6.0) / rdFA4 - fA4^(7.0 / 6.0) * rawd4 / (rdFA4 * rdFA4)

    fAmid = fA1 + 0.5 * (fA4 - fA1)

    V1 = fA1^(7.0 / 6.0) / inspFA1
    V4 = fA4^(7.0 / 6.0) / rdFA4
    V2raw = _xas_amp_CV(amp_table[4], eta, StotR, chia, Seta)
    V2 = 1.0 / V2raw

    FMs1 = fA1
    FMs2 = fAmid
    FMs4 = fA4

    F12 = FMs1 * FMs1
    F13 = F12 * FMs1
    F14 = F13 * FMs1
    F15 = F14 * FMs1
    F22 = FMs2 * FMs2
    F23 = F22 * FMs2
    F24 = F23 * FMs2
    F42 = FMs4 * FMs4
    F43 = F42 * FMs4
    F44 = F43 * FMs4
    F45 = F44 * FMs4

    F1mF2 = FMs1 - FMs2
    F1mF4 = FMs1 - FMs4
    F2mF4 = FMs2 - FMs4
    F1mF22 = F1mF2 * F1mF2
    F2mF42 = F2mF4 * F2mF4
    F1mF43 = F1mF4 * F1mF4 * F1mF4

    intAmp_denom = F1mF22 * F1mF43 * F2mF42

    delta0 = (
        -(d4 * F12 * F1mF22 * F1mF4 * FMs2 * F2mF4 * FMs4) +
        d1 * FMs1 * F1mF2 * F1mF4 * FMs2 * F2mF42 * F42 +
        F42 * (
            FMs2 * F2mF42 * (-4.0 * F12 + 3.0 * FMs1 * FMs2 + 2.0 * FMs1 * FMs4 - FMs2 * FMs4) * V1 +
            F12 * F1mF43 * V2
        ) +
        F12 * F1mF22 * FMs2 * (FMs1 * FMs2 - 2.0 * FMs1 * FMs4 - 3.0 * FMs2 * FMs4 + 4.0 * F42) * V4
    ) / intAmp_denom

    delta1 = (
        d4 * FMs1 * F1mF22 * F1mF4 * F2mF4 * (2.0 * FMs2 * FMs4 + FMs1 * (FMs2 + FMs4)) +
        FMs4 * (
            -(d1 * F1mF2 * F1mF4 * F2mF42 * (2.0 * FMs1 * FMs2 + (FMs1 + FMs2) * FMs4)) -
            2.0 * FMs1 * (
                F44 * (V1 - V2) + 3.0 * F24 * (V1 - V4) + F14 * (V2 - V4) +
                4.0 * F23 * FMs4 * (-V1 + V4) + 2.0 * F13 * FMs4 * (-V2 + V4) +
                FMs1 * (2.0 * F43 * (-V1 + V2) + 6.0 * F22 * FMs4 * (V1 - V4) + 4.0 * F23 * (-V1 + V4))
            )
        )
    ) / intAmp_denom

    delta2 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4 * (F12 + FMs2 * FMs4 + 2.0 * FMs1 * (FMs2 + FMs4))) +
        d1 * F1mF2 * F1mF4 * F2mF42 * (FMs1 * FMs2 + 2.0 * (FMs1 + FMs2) * FMs4 + F42) -
        4.0 * F12 * F23 * V1 + 3.0 * FMs1 * F24 * V1 - 4.0 * FMs1 * F23 * FMs4 * V1 + 3.0 * F24 * FMs4 * V1 +
        12.0 * F12 * FMs2 * F42 * V1 - 4.0 * F23 * F42 * V1 - 8.0 * F12 * F43 * V1 + FMs1 * F44 * V1 + F45 * V1 +
        F15 * V2 + F14 * FMs4 * V2 - 8.0 * F13 * F42 * V2 + 8.0 * F12 * F43 * V2 - FMs1 * F44 * V2 - F45 * V2 -
        F1mF22 * (
            F13 + FMs2 * (3.0 * FMs2 - 4.0 * FMs4) * FMs4 + F12 * (2.0 * FMs2 + FMs4) +
            FMs1 * (3.0 * FMs2 - 4.0 * FMs4) * (FMs2 + 2.0 * FMs4)
        ) * V4
    ) / intAmp_denom

    delta3 = (
        d4 * F1mF22 * F1mF4 * F2mF4 * (2.0 * FMs1 + FMs2 + FMs4) -
        d1 * F1mF2 * F1mF4 * F2mF42 * (FMs1 + FMs2 + 2.0 * FMs4) +
        2.0 * (
            F44 * (-V1 + V2) + 2.0 * F12 * F2mF42 * (V1 - V4) + 2.0 * F22 * F42 * (V1 - V4) +
            2.0 * F13 * FMs4 * (V2 - V4) + F24 * (-V1 + V4) + F14 * (-V2 + V4) +
            2.0 * FMs1 * FMs4 * (F42 * (V1 - V2) + F22 * (V1 - V4) + 2.0 * FMs2 * FMs4 * (-V1 + V4))
        )
    ) / intAmp_denom

    delta4 = (
        -(d4 * F1mF22 * F1mF4 * F2mF4) +
        d1 * F1mF2 * F1mF4 * F2mF42 -
        3.0 * FMs1 * F22 * V1 + 2.0 * F23 * V1 + 6.0 * FMs1 * FMs2 * FMs4 * V1 - 3.0 * F22 * FMs4 * V1 -
        3.0 * FMs1 * F42 * V1 + F43 * V1 +
        F13 * V2 - 3.0 * F12 * FMs4 * V2 + 3.0 * FMs1 * F42 * V2 - F43 * V2 -
        F1mF22 * (FMs1 + 2.0 * FMs2 - 3.0 * FMs4) * V4
    ) / intAmp_denom

    Overallamp = 2.0 * sqrt(5.0 / (64.0 * pi)) * M * GMsun_over_c2_Gpc * M * GMsun_over_c3 / dL
    amp0 = sqrt(2.0 * eta / 3.0) * pi^(-1.0 / 6.0)

    return (
        T = T, M = M, GMsun_over_c3 = GMsun_over_c3, fA1 = fA1, fA4 = fA4, fcutPar = fcutPar,
        fdamp = fdamp, fRD = fRD,
        amp_ins = (A2 = A2, A3 = A3, A4 = A4, A5 = A5, A6 = A6, rho1 = rho1, rho2 = rho2, rho3 = rho3),
        amp_int = (delta0 = delta0, delta1 = delta1, delta2 = delta2, delta3 = delta3, delta4 = delta4),
        amp_rd = (gamma1 = gamma1, gamma2 = gamma2, gamma3 = gamma3),
        Overallamp = Overallamp, amp0 = amp0,
    )
end # @Tlit T
end

"""
    _xas_ampl_scalar(coeffs, f)

Non-allocating IMRPhenomXAS amplitude evaluation at a single frequency `f` (not a vector), using
the frequency-independent coefficients from [`_xas_ampl_coeffs`](@ref) -- the XAS analogue of
IMRPhenomD's [`_ampl_scalar`](@ref). This is the scalar core that `Ampl` broadcasts over an
`AbstractVector` of frequencies (preserving the existing array-in/array-out API for all current
callers), and that the fused KernelAbstractions likelihood kernel calls directly in a per-thread
frequency loop with zero array allocation.
"""
@inline function _xas_ampl_scalar(coeffs, f)
    T = coeffs.T
@Tlit T begin
    f_safe = max(f, eps(T))
    fgrid = coeffs.M * coeffs.GMsun_over_c3 * f_safe

    amplitudeIMR = ifelse(
        fgrid < coeffs.fA1,
        _xas_amp_ins_val(fgrid, coeffs.amp_ins.A2, coeffs.amp_ins.A3, coeffs.amp_ins.A4,
            coeffs.amp_ins.A5, coeffs.amp_ins.A6, coeffs.amp_ins.rho1, coeffs.amp_ins.rho2, coeffs.amp_ins.rho3),
        ifelse(
            fgrid < coeffs.fA4,
            fgrid^(7.0 / 6.0) / (
                coeffs.amp_int.delta0 + fgrid * (
                    coeffs.amp_int.delta1 + fgrid * (
                        coeffs.amp_int.delta2 + fgrid * (coeffs.amp_int.delta3 + fgrid * coeffs.amp_int.delta4)
                    )
                )
            ),
            ifelse(
                fgrid < coeffs.fcutPar,
                _xas_amp_rd_val(fgrid, coeffs.amp_rd.gamma1, coeffs.amp_rd.gamma2, coeffs.amp_rd.gamma3,
                    coeffs.fdamp, coeffs.fRD),
                zero(T),
            ),
        ),
    )

    return coeffs.Overallamp * coeffs.amp0 * (fgrid^(-7.0 / 6.0)) * amplitudeIMR
end # @Tlit T
end

"""
    Ampl(model::IMRPhenomXAS, f, mc, eta, chi1, chi2, dL; kwargs...)

IMRPhenomXAS frequency-domain amplitude, dominant (2,2) mode. Same overall normalization
convention (SPA-style, `dL` in Gpc) as `Ampl(::IMRPhenomD, ...)` — this package's `Overallamp`
and `amp0` factors are structurally identical between IMRPhenomD and XAS (both port the same
`2*sqrt(5/(64*pi))*M^2/dL` / `sqrt(2*eta/3)*pi^(-1/6)` normalization; only the pseudo-PN shape
function differs), so the Gpc-unit convention already validated for IMRPhenomD against LALSuite
(see `Ampl(::IMRPhenomD, ...)`'s docstring) carries over unchanged. `f` may be a scalar (returns a
scalar, non-allocating) or an `AbstractVector` (returns a same-length `Vector`) -- computes the
frequency-independent coefficients once via [`_xas_ampl_coeffs`](@ref), then evaluates
[`_xas_ampl_scalar`](@ref) at each frequency. Callers evaluating many frequencies for one set of
parameters where per-call allocation matters (e.g. inside a GPU kernel) should call
`_xas_ampl_coeffs` once and `_xas_ampl_scalar` per point directly instead of going through this
array-oriented wrapper.

Implementation note: like `Ampl(::IMRPhenomD, ...)`, this deliberately uses an explicit
`similar`-then-loop instead of broadcasting `_xas_ampl_scalar` over `f`, for the same
Enzyme-activity-analysis reason documented on that function.

No parameter-domain assertions run here — this is the hot path; see `validate_phenomxas_params`.

# Keyword arguments
- `fcutPar = 0.3`: see `Phi(::IMRPhenomXAS, ...)`.
- `GMsun_over_c3`, `GMsun_over_c2_Gpc`: same convention as `Ampl(::IMRPhenomD, ...)`.
- `final_spin_override = NaN`: see `Phi(::IMRPhenomXAS, ...)`'s docstring.
"""
function Ampl(model::IMRPhenomXAS,
    f,
    mc,
    eta,
    chi1,
    chi2,
    dL;
    fcutPar = 0.3,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
    final_spin_override = NaN,
)
    coeffs = _xas_ampl_coeffs(model, mc, eta, chi1, chi2, dL;
        fcutPar = fcutPar, GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
        final_spin_override = final_spin_override)
    if f isa AbstractArray
        out = similar(f, coeffs.T)
        @inbounds for i in eachindex(f)
            out[i] = _xas_ampl_scalar(coeffs, f[i])
        end
        return out
    else
        return _xas_ampl_scalar(coeffs, f)
    end
end
const PhenomXAS_amp_coeff_table = (
    (
        (  # Coeffs for CV_Amp_Ins0 (ind 0)
            -0.015178276424448592,  # No spin
            -0.06098548699809163,
            0.4845148547154606,
            0.0,
            0.0,
            1.0,
            0.09799277215675059,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            0.02300153747158323,
            0.0,
            0.10495263104245876,
            0.0,
            0.01761591799745109,
            0.0,
            -0.14404522791467844,
            0.0,
            0.0,
            0.04834642258922544,
            -0.14189350657140673,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.7340448493183307,
            0.0,
            0.0,
            0.0,  # Uneq spin
            0.0,
            0.0,
            0.0,
            0.0018724905795891192,
            34.90874132485147,
        ),
        (  # Coeffs for CV_Amp_Ins1 (ind 1)
            -0.058572000924124644,  # No spin
            -1.1970535595488723,
            8.4630293045015,
            0.0,
            0.0,
            1.0,
            15.430818840453686,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            -0.08746408292050666,
            -0.20646621646484237,
            0.788717372588848,
            0.0,
            -0.018924013869130434,
            -0.21291764491897636,
            +0.8282888482429105,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.332123330797879,
            1.0,
            0.0,
            0.0,
            0.0,  # Uneq spin
            0.0,
            0.0,
            0.0,
            0.004389995099201855,
            105.84553997647659,
        ),
        (  # Coeffs for CV_Amp_Ins2 (ind 2)
            -0.16212854591357853,  # No spin
            1.617404703616985,
            -3.186012733446088,
            5.629598195000046,
            0.0,
            1.0,
            0.04507019231274476,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            1.0055835408962206,
            -4.127597118865669,
            18.353433894421833,
            -41.0378120175805,
            -0.31443470118113853,
            5.215501942120774,
            -18.80590889704093,
            19.099315016873643,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            5.852706459485663,
            -5.717874483424523,
            0.0,
            1.0,
            0.0,  # Uneq spin
            0.0,
            0.0,
            0.0,
            0.05575955418803233,
            208.92352600701068,
        ),
        (  # Coeffs for V2 (ind 3)
            1.4873184918202145,  # No spin
            1974.6112656679577,
            27563.641024162127,
            -19837.908020966777,
            0.0,
            1.0,
            143.29004876335128,
            458.4097306093354,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            27.952730865904343,
            -365.55631765202895,
            0.0,
            1612.2681322644232,
            3.2646808851249016,
            -260.3494489873286,
            3011.446602208493,
            -6962.675551371755,
            -19.38970173389662,
            0.0,
            0.0,
            1486.4658089990298,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            12.647425554323242,
            -10.540154508599963,
            0.0,
            1.0,
            0.0,  # Uneq spin
            0.0,
            -0.016404056649860943,
            -296.473359655246,
            0.0,
            0.0,
        ),
        (  # Coeffs for gamma2 (ind 4)
            0.8312293675316895,  # No spin
            7.480371544268765,
            -18.256121237800397,
            0.0,
            0.0,
            1.0,
            10.915453595496611,
            -30.578409433912874,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            0.5869408584532747,
            -0.1467158405070222,
            0.25295441250444334,
            0.0,
            0.031852563636196894,
            -2.8489481072076472,
            4.6849496672664594,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.8775263105069953,
            -3.41755361841226,
            0.0,
            1.0,
            0.0,  # Uneq spin
            -0.00548054788508203,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs for gamma3 (ind 5)
            1.3666000000000007,  # No spin
            -4.091333144596439,
            2.109081209912545,
            -4.222259944408823,
            0.0,
            1.0,
            -2.7440263888207594,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            0.07179105336478316,
            -0.8752427297525086,
            2.331724812782498,
            0.0,
            -0.05633734476062242,
            0.4168560229353532,
            -0.6330998412809531,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,  # Uneq spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs for v1RD (ind 6)
            0.03689164742964719,  # No spin
            25.417967754401182,
            162.52904393600332,
            0.0,
            0.0,
            1.0,
            61.19874463331437,
            -29.628854485544874,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            -0.14352506969368556,
            -4.805034453745424,
            27.675454081988036,
            -48.31945248941757,
            0.026356911108320547,
            1.11147906765112,
            -2.398327419614959,
            -3.751501972663298,
            0.19967405175523437,
            6.176053843938542,
            -47.99096500250743,
            81.9290740950083,
            -0.05292913111731128,
            -0.2874540719094058,
            -5.104257870393138,
            30.491948143930266,
            -0.18147275151697131,
            -8.990840289951514,
            72.08174136362386,
            -132.77982622925845,
            -1.4160870461211452,
            1.0,
            0.0,
            0.0,
            0.0,  # Uneq spin
            0.0,
            -0.04426571511345366,
            0.0,
            0.0,
            0.0,
        ),
    )
)
const PhenomXAS_phase_coeff_table = (
    (
        (  # Coeffs collocation point 0 of the inspiral phase (ind 0)
            -17294.000000000007,  # No spin
            -19943.076428555978,
            483033.0998073767,
            0.0,
            0.0,
            0.0,
            1.0,
            4.460294035404433,
            0.0,
            0.0,
            68384.62786426462,  # Eq spin
            67663.42759836042,
            -2179.3505885609297,
            19703.894135534803,
            32614.091002011017,
            -58475.33302037833,
            62190.404951852535,
            18298.307770807573,
            -303141.1945565486,
            0.0,
            -148368.4954044637,
            -758386.5685734496,
            -137991.37032619823,
            1.0765877367729193e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0412979553629143,
            1.0,
            0.0,
            0.0,
            12017.062595934838,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs collocation point 1 of the inspiral phase (ind 1)
            -7579.300000000004,  # No spin
            -120297.86185566607,
            1.1694356931282217e6,
            -557253.0066989232,
            0.0,
            0.0,
            1.0,
            18.53018618227582,
            0.0,
            0.0,
            -27089.36915061857,  # Eq spin
            -66228.9369155027,
            -44331.41741405198,
            0.0,
            0.0,
            50644.13475990821,
            157036.45676788126,
            126736.43159783827,
            0.0,
            0.0,
            150022.21343386435,
            -50166.382087278434,
            -399712.22891153296,
            0.0,
            0.0,
            -593633.5370110178,
            -325423.99477314285,
            +847483.2999508682,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.5232497464826662,
            -3.062957826830017,
            -1.130185486082531,
            1.0,
            3843.083992827935,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs collocation point 2 of the inspiral phase (ind 2)
            15415.000000000007,  # No spin
            873401.6255736464,
            376665.64637025696,
            -3.9719980569125614e6,
            8.913612508054944e6,
            0.0,
            1.0,
            46.83697749859996,
            0.0,
            0.0,
            397951.95299014193,  # Eq spin
            -207180.42746987,
            -130668.37221912303,
            0.0,
            0.0,
            -1.0053073129700898e6,
            1.235279439281927e6,
            -174952.69161683554,
            0.0,
            0.0,
            -1.9826323844247842e6,
            208349.45742548333,
            895372.155565861,
            0.0,
            0.0,
            4.662143741417853e6,
            -584728.050612325,
            -1.6894189124921719e6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -9.675704197652225,
            3.5804521763363075,
            2.5298346636273306,
            1.0,
            -24708.109411857182,  # UnEq Spin
            24703.28267342699,
            0.0,
            0.0,
            0.0,
            0.0,
            47752.17032707405,
            0.0,
            0.0,
            -1296.9289110696955,
        ),
        (  # Coeffs collocation point 3 of the inspiral phase (ind 3)
            2439.000000000001,  # No spin
            -31133.52170083207,
            28867.73328134167,
            0.0,
            0.0,
            0.0,
            1.0,
            0.41143032589262585,
            0.0,
            0.0,
            16116.057657391262,  # Eq spin
            9861.635308837876,
            0.0,
            0.0,
            0.0,
            -82355.86732027541,
            -25843.06175439942,
            0.0,
            0.0,
            0.0,
            229284.04542668918,
            117410.37432997991,
            0.0,
            0.0,
            0.0,
            -375818.0132734753,
            -386247.80765802023,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -3.7385208695213668,
            0.25294420589064653,
            1.0,
            0.0,
            194.5554531509207,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs collocation point 0 of the intermediate phase (ind 4)
            0.0,  # No Spin
            0.9951733419499662,
            101.21991715215253,
            632.4731389009143,
            0.0,
            0.0,
            0.00016803066316882238,
            0.11412314719189287,
            1.8413983770369362,
            1.0,
            18.694178521101332,  # Eq spin
            16.89845522539974,
            0.3612417066833153,
            0.0,
            0.0,
            -697.6773920613674,
            0.0,
            -147.53381808989846,
            0.0,
            0.0,
            0.0,
            4941.31613710257,
            0.0,
            0.0,
            0.0,
            3531.552143264721,
            -14302.70838220423,
            178.85850322465944,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            2.965640445745779,
            -2.7706595614504725,
            1.0,
            0.0,
            0.0,  # UnEq Spin
            356.74395864902294,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1693.326644293169,
            0.0,
        ),
        (  # Coeffs collocation point 1 of the intermediate phase (ind 5)
            0.0,  # No Spin
            -5.126358906504587,
            -227.46830225846668,
            688.3609087244353,
            -751.4184178636324,
            0.0,
            -0.004551938711031158,
            -0.7811680872741462,
            1.0,
            0.0,
            0.1549280856660919,  # Eq spin
            -0.9539250460041732,
            -2.84311102369862,
            0.0,
            0.0,
            73.79645135116367,
            0.0,
            -8.13494176717772,
            0.0,
            0.0,
            0.0,
            -539.4071941841604,
            0.0,
            0.0,
            0.0,
            -936.3740515136005,
            1862.9097047992134,
            224.77581754671272,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.5308507364054487,
            1.0,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            0.0,
            2993.3598520496153,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs collocation point 1 of the intermediate phase (ind 6)
            -82.54500000000004,  # No Spin
            -5.58197349185435e6,
            -3.5225742421184325e8,
            1.4667258334378073e9,
            0.0,  #
            0.0,  #
            1.0,
            66757.12830903867,
            5.385164380400193e6,
            2.5176585751772933e6,
            19.416719811164853,  # Eq spin
            -36.066611959079935,
            -0.8612656616290079,
            5.95010003393006,
            4.984750041013893,
            207.69898051583655,
            -132.88417400679026,
            -17.671713040498304,
            29.071788188638315,
            37.462217031512786,
            170.97203068800542,
            -107.41099349364234,
            0.0,
            -647.8103976942541,
            0.0,
            -1365.1499998427248,
            1152.425940764218,
            415.7134909564443,
            1897.5444343138167,
            -866.283566780576,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.1492259468169692,
            1.0,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            7343.130973149263,
            -20486.813161100774,
            0.0,
            0.0,
            0.0,
            515.9898508588834,
            0.0,
            0.0,
        ),
        (  # Coeffs collocation point 2 of the intermediate phase (ind 7)
            0.4248820426833804,  # No Spin
            -906.746595921514,
            -282820.39946006844,
            -967049.2793750163,
            670077.5414916876,
            0.0,
            1.0,
            1670.9440812294847,
            19783.077247023448,
            0.0,
            0.22814271667259703,  # Eq spin
            1.1366593671801855,
            0.4818323187946999,
            0.0,
            0.0,
            12.840649528989287,
            0.0,
            -61.17248283184154,
            0.0,
            0.0,
            -711.8532052499075,
            269.9234918621958,
            941.6974723887743,
            0.0,
            0.0,
            3499.432393555856,
            -877.8811492839261,
            -4974.189172654984,
            0.0,
            0.0,
            -4939.642457025497,
            -227.7672020783411,
            8745.201037897836,
            0.0,
            0.0,
            -1.2442293719740283,
            1.0,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            -514.8494071830514,
            1493.3851099678195,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs collocation point 0 of the merger ringdown phase (ind 8)
            0.0,  # No spin
            0.7207992174994245,
            -1.237332073800276,
            6.086871214811216,
            0.0,
            0.0,
            0.006851189888541745,
            0.06099184229137391,
            -0.15500218299268662,
            1.0,
            0.06519048552628343,  # Eq spin
            0.0,
            0.20035146870472367,
            0.0,
            -0.2697933899920511,
            -25.25397971063995,
            -5.215945111216946,
            -0.28745205203100666,
            5.7756520242745735,
            +4.917070939324979,
            +58.59408241189781,
            +153.95945758807616,
            0.0,
            -43.97332874253772,
            -11.61488280763592,
            +160.14971486043524,
            -693.0504179144295,
            0.0,
            0.0,
            0.0,
            -308.62513664956975,
            +835.1725103648205,
            -47.56042058800358,
            +338.7263666984089,
            -22.384949087140086,
            1.0,
            -0.6628745847248266,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            -23.504907495268824,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs collocation point 1 of the merger ringdown phase (ind 9)
            0.0,  # No spin
            -9.460253118496386,
            +9.429314399633007,
            +64.69109972468395,
            0.0,
            0.0,
            -0.0670554310666559,
            -0.09987544893382533,
            1.0,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.04497628581617564,
            0.0,
            0.0,
            17.36495157980372,
            0.0,
            0.0,
            0.0,
            0.0,
            -191.00932194869588,
            -62.997389062600035,
            +64.42947340363101,
            0.0,
            0.0,
            930.3458437154668,
            +808.457330742532,
            0.0,
            0.0,
            0.0,
            -774.3633787391745,
            -2177.554979351284,
            -1031.846477275069,
            0.0,
            0.0,
            1.0,
            -0.7267610313751913,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            -36.66374091965371,
            91.60477826830407,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs collocation point 2 of the merger ringdown phase (ind 10)
            0.0,  # No spin
            -8.506898502692536,
            +13.936621412517798,
            0.0,
            0.0,
            0.0,
            -0.40919671232073945,
            1.0,
            0.0,
            0.0,
            0.0,  # Eq spin
            0.046849371468156265,
            0.0,
            0.0,
            0.0,
            1.7280582989361533,
            0.0,
            18.41570325463385,
            -13.743271480938104,
            0.0,
            73.8367329022058,
            0.0,
            -95.57802408341716,
            +215.78111099820157,
            0.0,
            -27.976989112929353,
            +6.404060932334562,
            +109.04824706217418,
            -633.1966645925428,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.6862449113932192,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            0.0,
            0.0,
            641.8965762829259,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs collocation point 3 of the merger ringdown phase (ind 11)
            -85.86062966719405,  # No spin
            -4616.740713893726,
            -4925.756920247186,
            +7732.064464348168,
            +12828.269960300782,
            -39783.51698102803,
            1.0,
            +50.206318806624004,
            0.0,
            0.0,
            33.335857451144356,  # Eq spin
            -36.49019206094966,
            -3.835967351280833,
            2.302712009652155,
            1.6533417657003922,
            -69.19412903018717,
            26.580344399838758,
            -15.399770764623746,
            31.231253209893488,
            97.69027029734173,
            93.64156367505917,
            -18.184492163348665,
            423.48863373726243,
            -104.36120236420928,
            -719.8775484010988,
            0.0,
            1497.3545918387515,
            -101.72731770500685,
            0.0,
            0.0,
            1075.8686153198323,
            -3443.0233614187396,
            -4253.974688619423,
            -608.2901586790335,
            5064.173605639933,
            -1.3705601055555852,
            1.0,
            0.0,
            0.0,
            22.363215261437862,  # UnEq Spin
            156.08206945239374,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (  # Coeffs collocation point 4 of the merger ringdown phase (ind 12)
            0.0,  # No spin
            7.05731400277692,
            22.455288821807095,
            119.43820622871043,
            0.0,
            0.0,
            0.26026709603623255,
            1.0,
            0.0,
            0.0,
            0.0,  # Eq spin
            0.0,
            0.0,
            0.0,
            0.0,
            -7.9407123129681425,
            9.486783128047414,
            0.0,
            0.0,
            0.0,
            134.88158268621922,
            -56.05992404859163,
            0.0,
            0.0,
            0.0,
            -316.26970506215554,
            90.31815139272628,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            -0.7162058321905909,
            0.0,
            0.0,
            0.0,  # UnEq Spin
            0.0,
            43.82713604567481,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    )
)
