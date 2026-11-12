using StaticArrays: SVector, MVector

export seobnrv5alignedspin_postadiabatic_dynamics_fixed

"""
    SEOBNRv5AlignedSpinPAFixed

The differentiable, GPU-compatible form of `SEOBNRv5AlignedSpinPA.jl`: every runtime-sized `Vector{Float64}` in the original PA driver
is replaced here by a compile-time-fixed-size `StaticArrays.SVector{NMAX,Float64}`/`MVector`, with
a *dynamic* `Int` (`nreal<=NMAX`, computed exactly as the original's `r_size`) driving `ifelse`/
dynamic-tuple-indexed selection between boundary and interior stencil rows -- a pattern
validated against real
finite differences (including the REAL [`_SEOBV5_FINDIFF9_COEFFS`](@ref) coefficient table, at grid
sizes bracketing this repo's own previously-measured real configurations, 18 and 107 points).

# What is (and is not) validated by this file's existence

**Numerically validated against the original `Vector`-based implementation**
([`seobnrv5alignedspin_postadiabatic_dynamics`](@ref)) -- see
`test/seobnrv5alignedspin_pa_fixed_tests.jl` -- across every configuration that file's own test
suite already uses. This is a *storage/architecture* change only: same algorithm, same coefficient
tables, same physics function calls: `t`/`r`/`phi`/`pr`/`pphi` over the real (`1:nreal`) region
should reproduce the original to floating-point precision, not merely "close."

**NOT yet re-validated here against real `pyseobnr` output directly** -- that validation lives on
the original `Vector`-based implementation (`test/seobnrv5alignedspin_pa_tests.jl`'s own real-
`pyseobnr` cross-checks); this file's own correctness claim is entirely "matches that
already-validated implementation," a weaker but still meaningful transitive guarantee.

**Enzyme reverse-mode / `KernelAbstractions.jl` validation is this file's reason for
existing**; see its gated test file.

# One accepted, deliberate divergence from the original: `_seobv5_pphi_eqn`'s discriminant clamp

The original [`_seobv5_pphi_eqn`](@ref) `throw`s a `DomainError` when the PA quadratic's
discriminant `D` is negative beyond a `1e-10` numerical tolerance -- a genuine, intentional sanity
check on physically-meaningful grid points. This file's own driver computes EVERY index in
`1:NMAX` unconditionally, including padding indices `>nreal` that hold a fixed, physically-inert
placeholder radius (`_SEOBV5_PA_R_PAD = 50.0`, chosen to be smooth/singularity-free for every
Hamiltonian term regardless of the actual binary's masses/spins) whose own results are masked out
before anything downstream reads them -- a `throw` inside that discarded-but-still-executed
padding computation would crash the whole call for reasons that have nothing to do with the real
physics, and a `throw` inside a KernelAbstractions/Enzyme hot path is a correctness/compilation
hazard in its own right (divergent, side-effecting control flow). [`_seobv5_pphi_eqn_fixed`](@ref)
therefore replaces the `throw` with an unconditional branch-free clamp, `D = max(D, 0.0)`. Whether
this ever actually changes behavior on the REAL (`i<=nreal`) region -- as opposed to merely making
the discarded padding region crash-safe -- is checked empirically in this file's own numerical
validation test (comparing against the original's `throw`-based real-region output), not assumed.
"""
struct SEOBNRv5AlignedSpinPAFixed end

const _SEOBV5_PA_NMAX = 256
const _SEOBV5_PA_R_PAD = 50.0

@inline function _seobv5_pa_clampidx(j::Int, ::Val{NMAX}) where {NMAX}
    return min(max(j, 1), NMAX)
end

"""
    _seobv5_findiff_derivative9_fixed(y::SVector{NMAX,Float64}, nreal::Int, h::Float64) -> SVector{NMAX,Float64}

Fixed-buffer analog of [`_seobv5_findiff_derivative9`](@ref): identical
[`_SEOBV5_FINDIFF9_COEFFS`](@ref) 9-row table and three-region structure (static left rows at
`i<=4`; a single interior row for `5<=i<=nreal-4`; right rows selected by the *dynamic* tuple
index `i-nreal+9` for `nreal-3<=i<=nreal`), operating on a compile-time-fixed `NMAX`-length buffer
with the true grid length `nreal<=NMAX` passed as a runtime `Int`. Elements past `nreal` are
computed (same cost as every other lane, required for `KernelAbstractions.jl`/Enzyme uniformity)
but masked to `0`. 
"""
@inline function _seobv5_findiff_derivative9_fixed(y::SVector{NMAX,Float64}, nreal::Int, h::Float64) where {NMAX}
    C = _SEOBV5_FINDIFF9_COEFFS
    dydx = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:NMAX
        active = i <= nreal
        is_left = i <= 4
        is_right = (i >= nreal - 3) && (i <= nreal)

        left_row = C[clamp(i, 1, 4)]
        right_row = C[clamp(i - nreal + 9, 6, 9)]
        mid_row = C[5]

        s_left = 0.0
        s_mid = 0.0
        s_right = 0.0
        @inbounds for k in 1:9
            s_left += left_row[k] * y[_seobv5_pa_clampidx(k, Val(NMAX))]
            s_mid += mid_row[k] * y[_seobv5_pa_clampidx(i - 5 + k, Val(NMAX))]
            s_right += right_row[k] * y[_seobv5_pa_clampidx(nreal - 9 + k, Val(NMAX))]
        end
        val = ifelse(is_left, s_left, ifelse(is_right, s_right, s_mid))
        dydx[i] = ifelse(active, val / h, 0.0)
    end
    return SVector(dydx)
end

"""
    _seobv5_cumulative_integral7_fixed(y::SVector{NMAX,Float64}, nreal::Int, h::Float64) -> SVector{NMAX,Float64}

Fixed-buffer analog of [`_seobv5_cumulative_integral7`](@ref): same [`_SEOBV5_CUMINT7_COEFFS`](@ref)
7-row table, same structure (`integral[1]=0`; static near-left rows at static output indices
`2,3,4` using the FIXED window `y[1:8]`; a single interior row for `5<=i<=nreal-3`; near-right rows
at *dynamic* output indices `nreal-2,nreal-1,nreal` using the FIXED-relative-to-`nreal` window
`y[nreal-7:nreal]`) -- verbatim port of the original's own two fixed-window special-case blocks
plus its shifting-window interior block, onto the same fixed-`NMAX`/dynamic-`nreal` buffer
convention as [`_seobv5_findiff_derivative9_fixed`](@ref). Per-index contributions past `nreal`
(and index `1`) are `0` before the final `cumsum`, so the returned array is exactly flat
(equal to the true definite integral at `nreal`) for every padding index -- harmless, since only
index `nreal` is ever read downstream ([`seobnrv5alignedspin_postadiabatic_dynamics_fixed`](@ref)'s
own `t`/`phi`).
"""
@inline function _seobv5_cumulative_integral7_fixed(y::SVector{NMAX,Float64}, nreal::Int, h::Float64) where {NMAX}
    ROW = _SEOBV5_CUMINT7_COEFFS
    contrib = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:NMAX
        active = i <= nreal
        is_near_left = (i >= 2) && (i <= 4)
        is_near_right = (i >= nreal - 2) && (i <= nreal)
        is_interior = (i >= 5) && (i <= nreal - 3)

        left_row = ROW[clamp(i - 1, 1, 3)]
        right_row = ROW[clamp(i - nreal + 7, 5, 7)]
        mid_row = ROW[4]

        s_left = 0.0
        s_mid = 0.0
        s_right = 0.0
        @inbounds for k in 1:8
            s_left += left_row[k] * y[_seobv5_pa_clampidx(k, Val(NMAX))]
            s_mid += mid_row[k] * y[_seobv5_pa_clampidx(i - 5 + k, Val(NMAX))]
            s_right += right_row[k] * y[_seobv5_pa_clampidx(nreal - 8 + k, Val(NMAX))]
        end
        val = ifelse(is_near_left, s_left, ifelse(is_near_right, s_right, ifelse(is_interior, s_mid, 0.0)))
        contrib[i] = ifelse(active, val * h, 0.0)
    end
    # NOT `cumsum(contrib)`: Base's generic `cumsum` on an `MVector` returns a plain, dynamically-
    # sized `Vector` (StaticArrays does not specialize it for `MVector`), so `SVector(cumsum(...))`
    # forced a runtime, not compile-time, size conversion right here -- confirmed (not guessed) to
    # be the exact site Enzyme's reverse-mode failed on ("cannot statically prove the type of a
    # value being differentiated"). A hand-rolled
    # cumulative sum that never leaves `MVector{NMAX,Float64}` keeps the size in the type throughout.
    result = zero(MVector{NMAX,Float64})
    acc = 0.0
    @inbounds for i in 1:NMAX
        acc += contrib[i]
        result[i] = acc
    end
    return SVector(result)
end

"""
    _seobv5_pa_grid_fixed(r0, dr, nreal, ::Val{NMAX}) -> SVector{NMAX,Float64}

The fixed-buffer PA radial grid: `r[i] = r0-(i-1)*dr` for the real region `i<=nreal` (identical to
the original's `collect(range(r0,r_final;length=r_size))`, up to the usual floating-point
round-off from an explicit arithmetic recurrence vs. `range`'s own construction -- checked
empirically in the numerical validation test, not assumed negligible), and a FIXED, physically-inert
placeholder (`_SEOBV5_PA_R_PAD`) for every padding index `>nreal` -- deliberately NOT a continuation
of the same arithmetic progression (which would drive `r` through zero/negative for a large
`NMAX-nreal` gap, e.g. `nreal=18,NMAX=1024`, hitting real singularities in the Hamiltonian's
`log(r)`/`1/r^k` terms) -- see this file's own module docstring for why a crash-safe but physically
meaningless value is what the padding region actually needs.
"""
@inline function _seobv5_pa_grid_fixed(r0::Float64, dr::Float64, nreal::Int, ::Val{NMAX}) where {NMAX}
    r = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:NMAX
        r[i] = ifelse(i <= nreal, r0 - (i - 1) * dr, _SEOBV5_PA_R_PAD)
    end
    return SVector(r)
end

"""
    _seobv5_adiabatic_j0_fixed(r::SVector{NMAX,Float64}, hp, nreal = NMAX) -> SVector{NMAX,Float64}

Fixed-buffer analog of [`_seobv5_adiabatic_j0`](@ref): identical per-point fixed-8-iteration
Newton solve (uniform across every index, no boundary-dependent logic at all -- the simplest of
the four fixed-buffer conversions), just applied to every index `1:NMAX` including the padding
region's `_SEOBV5_PA_R_PAD` radius (which converges to some finite, physically meaningless but
FINITE angular momentum there, discarded by every caller).

`nreal` (default `NMAX`, i.e. every lane) limits the solve to lanes `i <= nreal`; later lanes get
the inert `sqrt(r[i])` instead. Nothing in the real region reads those lanes, so the real-region
result is unchanged -- see [`seobnrv5alignedspin_postadiabatic_dynamics_fixed`](@ref)'s
`skip_padding`.
"""
function _seobv5_adiabatic_j0_fixed(r::SVector{NMAX,Float64}, hp, nreal::Int = NMAX) where {NMAX}
    pphi = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:NMAX
        if i > nreal
            pphi[i] = sqrt(r[i])
            continue
        end
        ri = r[i]
        p = sqrt(ri)
        for _ in 1:8     # exact derivative: converged to round-off after 4 (20 with the finite-difference one)
            dHdr, deriv = _seobv5_sym_j0(ri, p, hp)
            p -= dHdr / deriv
        end
        pphi[i] = p
    end
    return SVector(pphi)
end

"""
    _seobv5_pphi_eqn_fixed(pphi_trial, r, pr, dpr_dr, hp, fc) -> Float64

[`_seobv5_pphi_eqn`](@ref) with the negative-discriminant handling an unconditional branch-free clamp
(`D = max(D, 0.0)`) instead of a numerical-tolerance check plus `throw` -- see this file's own module docstring
("One accepted, deliberate divergence...") for why. Same quadratic ([`_seobv5_pphi_quadratic`](@ref)).
"""
@inline function _seobv5_pphi_eqn_fixed(pphi_trial::Float64, r::Float64, pr::Float64, dpr_dr::Float64, hp, fc)
    C_2, C_1, D = _seobv5_pphi_quadratic(pphi_trial, r, pr, dpr_dr, hp, fc)
    return (-C_1 - sqrt(max(D, 0.0))) / (2.0 * C_2)
end

"""
    _seobv5_postadiabatic_solution_fixed(r, nreal, pphi_in, hp, fc, dr; order=8) -> (pr, pphi)

Fixed-buffer analog of [`_seobv5_postadiabatic_solution!`](@ref): identical 8-pass (default)
alternating sweep, [`_seobv5_pr_eqn`](@ref) reused verbatim (no `throw`/branch hazard in its own
body, confirmed by inspection), [`_seobv5_pphi_eqn_fixed`](@ref) in place of
[`_seobv5_pphi_eqn`](@ref).

`skip_padding=true` skips both per-lane equations for `i > nreal` (those lanes keep their inputs:
`pr = 0`, `pphi = pphi_in`); the real region is unchanged, since no real-lane stencil reads them.
Default `false` keeps every-lane evaluation for callers that rely on it (GravitationalWaveBayes'
`SEOBNRv5AlignedSpinF2Adjoint` recomputes this primal itself).
"""
function _seobv5_postadiabatic_solution_fixed(r::SVector{NMAX,Float64}, nreal::Int, pphi_in::SVector{NMAX,Float64},
        hp, fc, dr::Float64; order::Int = 8, skip_padding::Bool = false) where {NMAX}
    nlanes = skip_padding ? nreal : NMAX
    pr = zero(MVector{NMAX,Float64})
    pphi = MVector(pphi_in)
    for n in 1:order
        if isodd(n)
            dpphi_dr = -_seobv5_findiff_derivative9_fixed(SVector(pphi), nreal, dr)
            @inbounds for i in 1:NMAX
                i > nlanes && continue
                # Deep in the padding region (i>nreal), every point shares the identical placeholder
                # r=_SEOBV5_PA_R_PAD and (after the adiabatic solve) the identical pphi, so ANY
                # finite-difference derivative stencil there is EXACTLY 0.0 (a basic stencil-
                # coefficient identity: coefficients summing to zero applied to a constant sequence).
                # _seobv5_pr_eqn divides by dpphi_dr, so an unguarded exact 0.0 there is a genuine
                # 0/0 (or x/0) -> NaN/Inf trap in the discarded-but-still-executed padding lanes --
                # confirmed empirically (this is what motivated this comment, not a hypothetical).
                # Substitute a safe nonzero placeholder for masked lanes; the result is discarded by
                # every caller regardless, so its exact value doesn't matter, only its finiteness.
                dpphi_dr_i = ifelse(i <= nreal, dpphi_dr[i], 1.0)
                pr[i] = _seobv5_pr_eqn(pr[i], r[i], pphi[i], dpphi_dr_i, hp, fc)
            end
        else
            dpr_dr = -_seobv5_findiff_derivative9_fixed(SVector(pr), nreal, dr)
            @inbounds for i in 1:NMAX
                i > nlanes && continue
                # dpr_dr only enters _seobv5_pphi_eqn_fixed multiplicatively, not as a divisor, so
                # an exact 0.0 here can't itself produce a NaN/Inf the way dpphi_dr's division does
                # above -- but guarding it the same way costs nothing and keeps the two branches
                # symmetric/defensive against any future change to _seobv5_pphi_eqn_fixed's own body.
                dpr_dr_i = ifelse(i <= nreal, dpr_dr[i], 1.0)
                pphi[i] = _seobv5_pphi_eqn_fixed(pphi[i], r[i], pr[i], dpr_dr_i, hp, fc)
            end
        end
    end
    return SVector(pr), SVector(pphi)
end

"""
    seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs, ::Val{NMAX}=Val(1024); order=8, dr0=0.2) -> NamedTuple

Fixed-buffer analog of [`seobnrv5alignedspin_postadiabatic_dynamics`](@ref) -- see this file's own
module docstring for the full architecture and validation status. Returns a `NamedTuple`:
`success::Bool` (identical semantics to the original: `false` exactly when `r_size<=4` or
`r0<=11.5`); `nreal::Int` (the real grid length, `<=NMAX`, well-defined and safe to read even when
`success` is `false`, unlike the original's early-return-without-computing-anything design); and
`t,r,phi,pr,pphi::SVector{NMAX,Float64}`, always fully computed (even on `success=false` or
`nreal<NMAX`) with every index `>nreal` masked/flat per each stencil's own docstring -- callers
must slice `[1:nreal]` (or just read index `nreal` for `t[end]`/`r[end]`-style original usage) and
must check `success` before trusting the result at all, exactly as for the original.

**`NMAX` sizing is a genuine, measured cost tradeoff, not a free knob** -- compiling
this driver against the REAL ~1000-line Hamiltonian scales **super-linearly** with `NMAX` --
`NMAX=64`: ~3.4s: `NMAX=128`: ~6.1s; `NMAX=256`: ~28.6s; `NMAX=512`: ~194s; `NMAX=1024` did not
finish in over 25 minutes of wall-clock (killed, not a hang -- steady 100% single-core CPU and flat
~7GB RSS the whole time, consistent with a genuinely huge amount of LLVM IR to optimize, most likely
from the compiler unrolling or otherwise duplicating the large inlined Hamiltonian body across the
`NMAX`-trip loop, though this was not separately confirmed via IR inspection). Each `NMAX` value is
its own monomorphized method (`Val{NMAX}` is a type parameter), so this cost is paid ONCE per
distinct `NMAX` per Julia session (JIT-cached thereafter), not per call -- but it rules out
`NMAX=1024` (this file's original, since-revised default) as impractical for ordinary use, and makes
`NMAX=512` a real (~3 minute) cost even where it would otherwise be the natural choice. The current
default (`_SEOBV5_PA_NMAX=256`) covers this repo's own q=1.5/8.0/28.1 test configs (`r_size` up to
107) with room to spare, but NOT this repo's own low-mass/low-`f_min` probe (`r_size` up to 507)
-- configurations needing more grid points than `NMAX` allows are
silently truncated (masked at `NMAX`, not an error) under the current design, a real limitation, not
merely an inconvenience. Suppressing whatever compiler behavior drives this scaling (candidate:
`@noinline` barriers around the Hamiltonian calls inside these loops, trading runtime speed for
compile time) has not been attempted.
"""
function seobnrv5alignedspin_postadiabatic_dynamics_fixed(coeffs, ::Val{NMAX} = Val(_SEOBV5_PA_NMAX);
        order::Int = 8, dr0::Float64 = 0.2, skip_padding::Bool = true) where {NMAX}
    hp = _seobv5_f64(coeffs.hp)
    nu, X1, X2, chi1, chi2 = hp.nu, hp.X1, hp.X2, hp.chi1, hp.chi2
    fc = _seobv5_f64(coeffs.fc)
    r0 = Float64(coeffs.r0)

    chi_eff = X1 * chi1 + X2 * chi2
    r_final_prefactor = 2.7 + chi_eff * (1.0 - 4.0 * nu)
    # Out of the physical domain the ISCO fit can hit x^(1/3) of a negative number (DomainError); the
    # table is marked unsuccessful below anyway, so evaluate it at harmless placeholder values there.
    in_domain = (0.0 < nu <= 0.25) & (abs(chi1) <= 1.0) & (abs(chi2) <= 1.0)
    r_isco, _ = in_domain ? _seobv5_kerr_isco(chi1, chi2, X1, X2) : _seobv5_kerr_isco(0.0, 0.0, 0.5, 0.5)
    r_final = max(10.0, r_final_prefactor * r_isco)

    # Non-physical inputs (|chi| > 1, eta > 1/4, ...) make r0 or r_isco NaN; `ceil(Int, NaN)` would throw.
    # Report them as an unsuccessful table instead (callers already reject `success == false`).
    nreal_span = (r0 - r_final) / dr0
    nreal_raw = isfinite(nreal_span) ? ceil(Int, clamp(nreal_span, -1.0e6, 1.0e6)) : 0
    # Outside the physical domain (|chi| > 1, eta > 1/4) the potentials still evaluate to finite numbers,
    # so without this the table would silently report success for meaningless physics.
    success = in_domain & !(nreal_raw <= 4 || r0 <= 11.5)
    nreal_bumped = nreal_raw < 10 ? 10 : nreal_raw
    nreal = success ? min(nreal_bumped, NMAX) : 10

    # When `success` is false, `r0` itself is exactly the reason (e.g. r0<=11.5, sometimes as
    # extreme as r0~2.9 -- deep inside/near the horizon, a regime these Hamiltonian potentials were
    # never designed for) -- confirmed empirically (not a hypothetical) that evaluating the real
    # physics there produces NaN from the very first grid point, not just in a discarded padding
    # tail. Since this branch's own output is explicitly documented as untrustworthy without a
    # `success` check anyway, substitute an always-valid, physically-inert grid (same philosophy as
    # `_SEOBV5_PA_R_PAD`) rather than the genuinely-invalid real `r0`/`r_final` -- guarantees
    # finiteness regardless of caller discipline, the same Enzyme-safety property the padding-region
    # fix above provides for the `nreal<i<=NMAX` tail.
    r0_grid = success ? r0 : 50.0
    r_final_grid = success ? r_final : 40.0
    dr = (r0_grid - r_final_grid) / (nreal - 1)
    r = _seobv5_pa_grid_fixed(r0_grid, dr, nreal, Val(NMAX))

    # `skip_padding` (default on): padding lanes (i > nreal) skip all per-lane physics.
    # No real-lane stencil or quadrature term reads them, so `t`/`phi` and the real-region
    # `pr`/`pphi` are bit-identical to computing every lane, but the cost scales with `nreal`
    # instead of `NMAX` (18 of 256 lanes for GW150914). Loop trip counts stay the constant `NMAX`
    # (the branch is inside the body), so there is still no data-dependent trip count for Enzyme's
    # reverse pass to cache on GPU (see `_seobv5f2_searchsortedlast`). The returned padding-lane
    # `pr`/`pphi` then hold inert placeholders (0 / sqrt(r)) rather than physics values.
    nlanes = skip_padding ? nreal : NMAX
    pphi0 = _seobv5_adiabatic_j0_fixed(r, hp, nlanes)
    pr, pphi = _seobv5_postadiabatic_solution_fixed(r, nreal, pphi0, hp, fc, dr; order = order,
        skip_padding = skip_padding)

    dt_dr = zero(MVector{NMAX,Float64})
    dphi_dr = zero(MVector{NMAX,Float64})
    @inbounds for i in 1:NMAX
        i > nlanes && continue
        xi, dHdpr, dHdpphi = _seobv5_sym_tphi(r[i], pr[i], pphi[i], hp)
        dH_dpr_times_xi = dHdpr * xi
        dt_dr[i] = 1.0 / dH_dpr_times_xi
        dphi_dr[i] = dHdpphi / dH_dpr_times_xi
    end

    t = _seobv5_cumulative_integral7_fixed(SVector(dt_dr), nreal, -dr)
    phi = _seobv5_cumulative_integral7_fixed(SVector(dphi_dr), nreal, -dr)

    return (success = success, nreal = nreal, t = t, r = r, phi = phi, pr = pr, pphi = pphi, r_final = r_final)
end
