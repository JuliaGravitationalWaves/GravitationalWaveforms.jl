using StaticArrays: SVector, MVector
using LinearAlgebra: cross, dot, norm

include("SEOBNRv5PHM_pn_generated.jl")

"""
    SEOBNRv5PHMPNTable

The SEOBNRv5PHM post-Newtonian spin evolution (milestone P1 of GravitationalWaveBayes' `SEOBNRv5PHM_DESIGN.md`) as a fixed-size
node table in the orbital frequency: pyseobnr integrates `prec_eqns_20102022` in time (DOP853, rtol 1e-10) from
the spins given at `omega_ref`, backwards to `0.9 omega_start` and forwards until omega stops increasing or exceeds
0.35, then fits a cubic spline in omega to the samples; the EOB dynamics read spins, `LN_hat` and `L` from it.

Here the independent variable is `x = omega^(-p/3)` (= v^(-p); decreasing in omega), integrated with a 6th-order
Runge-Kutta method on a uniform grid of `M` nodes (`M - 1` steps of size `h`) from `x_lo = (0.9 omega_start)^(-p/3)`
to `x_hi = 0.35^(-p/3)`, `M = ceil(|x_hi - x_lo| / H) + 1` for the target step `H` (`_seobv5phm_pn_hstep`) (at most the storage size `N`, so
a short inspiral takes few steps and only one longer than `(N - 1) H` gets coarser ones), with the spins given at
`omega_ref` first carried to `x_lo` by `ceil(|x_lo - x_ref| / h)` (at most `N`) steps. Values between nodes are cubic Hermite interpolants of the state `(LN_hat, S1, S2)`
using the exact derivatives `dz/dx` at the nodes. Sizes are fixed (no data-dependent growth); the node data live in
caller-owned 9 x N matrices (`_seobv5phm_pn_fill!`, `_seobv5phm_pn_view`), e.g. one slice per walker of a device array.

The default `p = 3` (`x = 1/omega`) spaces the nodes by roughly equal precession phase: the leading-order
precession rate is `~ v^5` and `dt/dv ~ v^(-9)`, so the phase per unit `v^(-1)` grows like `v^(-2)` and is
nearly constant per unit `v^(-3)`. With `p = 1` (uniform in `1/v`) a 256-node table of a long, low-frequency
evolution (q = 4, omega 0.0059 .. 0.35, ~9 precession cycles) missed pyseobnr's spins by 2.7e-3.

If omega stops increasing before 0.35 (domega/dt <= 0, which pyseobnr's stopping test catches), the table ends at
the last node before it: `n` is the number of valid nodes and `omega_end` the last valid frequency (pyseobnr's
`omegaPN_f`, which also ends the EOB evolution).
"""
struct SEOBNRv5PHMPNTable{T,G,C}
    x::G                              # node values of x = omega^(-p/3) (a `_PNGrid`), increasing index = decreasing x
    z::C                              # z[i]: (LN_hat, S1, S2) at node i (S_i = chi_i m_i^2, M = 1), a `_PNCols`
    dz::C                             # dz[i]: dz/dx at node i
    n::Int                            # valid nodes
    omega_end::T
    m1::T
    m2::T
    p::Float64
end

# The uniform node grid x_i = x1 + (i - 1) h, without storage.
struct _PNGrid{T}
    x1::T
    h::T
end
@inline Base.getindex(g::_PNGrid, i::Integer) = g.x1 + (i - 1) * g.h

# Node vectors as the columns of a 9 x N matrix (a heap Matrix on the CPU, a view of global memory on a device).
struct _PNCols{A}
    A::A
end
@inline Base.getindex(c::_PNCols, i::Integer) = @inbounds SVector{9}(c.A[1, i], c.A[2, i], c.A[3, i], c.A[4, i],
    c.A[5, i], c.A[6, i], c.A[7, i], c.A[8, i], c.A[9, i])

const _SEOBV5PHM_PN_OMEGA_MAX = 0.35
const _SEOBV5PHM_PN_P = 3.0
const _SEOBV5PHM_PN_N = 512
const _SEOBV5PHM_PN_H0 = 4.2
const _SEOBV5PHM_PN_HMAX = 0.6

# Target step in x = 1/omega: the leading-order spin-orbit precession phase per unit x is constant along the
# inspiral, (5/96)(2 + 3q/2) for the smaller body's spin (q = m_big/m_small), so a fixed phase per step is
# h ~ 1/(2 + 3q/2). H0, HMAX: every pyseobnr dynamics reference at its N -> infinity agreement (q = 8 needs h = 0.3,
# near-extremal equal-mass spins h = 0.6).
@inline _seobv5phm_pn_hstep(m1, m2) =
    min(_SEOBV5PHM_PN_H0 / (2 + 3 * max(m1, m2) / (2 * min(m1, m2))), _SEOBV5PHM_PN_HMAX)

# nodes of the table from x_lo to x_hi at target step H (the step count is piecewise constant in the parameters)
@inline _seobv5phm_pn_m(x_lo, x_hi, N, H) = clamp(ceil(Int, abs(x_hi - x_lo) / H) + 1, 2, N)

@inline _seobv5phm_pn_x(omega, p) = p == 3 ? inv(omega) : omega^(-p / 3)
@inline _seobv5phm_pn_om(x, p) = p == 3 ? inv(x) : x^(-3 / p)

# d(LN_hat, S1, S2, omega)/dt -> d(LN_hat, S1, S2)/dx at x = omega^(-p/3); also returns domega/dt
@inline function _seobv5phm_pn_dx(z9::SVector{9,T}, x, p, nu, m1, m2) where {T}
    omega = _seobv5phm_pn_om(x, p)
    z = SVector(z9[1], z9[2], z9[3], z9[4], z9[5], z9[6], z9[7], z9[8], z9[9], omega)
    d = _seobv5phm_pn_rhs(z, nu, m1, m2, m1, m2)
    wdot = d[10]
    # pyseobnr's RHS normalizes LN_hat in place, i.e. the solver state is projected back to |LN_hat| = 1 after
    # every step; prec_eqns' LN_hat rate has a component along LN_hat (the dm * Lh * (Lh . S1 x S2) term), so the
    # equivalent ODE (step -> 0) keeps only the perpendicular part. Without this the precession runs at
    # 1/|LN_hat| times the rate and drifts from pyseobnr's solution (2.6e-5 in LN_hat over q = 1.5's inspiral).
    Lh = SVector(z9[1], z9[2], z9[3])
    Lh = Lh / norm(Lh)
    Ld = SVector(d[1], d[2], d[3])
    Ld = Ld - dot(Lh, Ld) * Lh
    # dx/dt = -(p/3) omega^(-p/3 - 1) domega/dt = -(p/3) (x / omega) domega/dt
    s = -3 * omega / (p * x * wdot)
    return SVector(Ld[1] * s, Ld[2] * s, Ld[3] * s, d[4] * s, d[5] * s, d[6] * s, d[7] * s, d[8] * s, d[9] * s), wdot
end

# One table step: Butcher's 7-stage 6th-order Runge-Kutta method (rational coefficients). At the table's step sizes
# the PN step error dominates the spin error against pyseobnr; order 6 reaches it with half RK4's nodes.
@inline function _seobv5phm_pn_step(z, x, h, p, nu, m1, m2)
    k1, w1 = _seobv5phm_pn_dx(z, x, p, nu, m1, m2)
    k2, w2 = _seobv5phm_pn_dx(z + (h / 3) * k1, x + h / 3, p, nu, m1, m2)
    k3, w3 = _seobv5phm_pn_dx(z + (2h / 3) * k2, x + 2h / 3, p, nu, m1, m2)
    k4, w4 = _seobv5phm_pn_dx(z + (h / 12) * (k1 + 4 * k2 - k3), x + h / 3, p, nu, m1, m2)
    k5, w5 = _seobv5phm_pn_dx(z + (h / 16) * (-k1 + 18 * k2 - 3 * k3 - 6 * k4), x + h / 2, p, nu, m1, m2)
    k6, w6 = _seobv5phm_pn_dx(z + (h / 8) * (9 * k2 - 3 * k3 - 6 * k4 + 4 * k5), x + h / 2, p, nu, m1, m2)
    k7, w7 = _seobv5phm_pn_dx(z + (h / 44) * (9 * k1 - 36 * k2 + 63 * k3 + 72 * k4 - 64 * k6), x + h, p, nu, m1, m2)
    ok = (w1 > 0) & (w2 > 0) & (w3 > 0) & (w4 > 0) & (w5 > 0) & (w6 > 0) & (w7 > 0)
    return z + (h / 120) * (11 * (k1 + k7) + 81 * (k3 + k4) - 32 * (k5 + k6)), ok
end

"""
    seobnrv5phm_pn_table(m1, m2, chi1, chi2, omega_ref, omega_start; N = 512, p = 3, H) -> SEOBNRv5PHMPNTable

PN spin evolution of a binary with masses `m1 + m2 = 1` and dimensionless spin vectors `chi1`, `chi2` (in the frame
where `LN_hat = z` at `omega_ref`), tabulated from `0.9 omega_start` (pyseobnr's start for the postadiabatic
inspiral) to the end of the PN evolution. See [`SEOBNRv5PHMPNTable`](@ref).
"""
function seobnrv5phm_pn_table(m1, m2, chi1, chi2, omega_ref, omega_start; N::Int = _SEOBV5PHM_PN_N, p = _SEOBV5PHM_PN_P,
        H = _seobv5phm_pn_hstep(m1, m2))
    return _seobv5phm_pn_table(m1, m2, SVector{3}(chi1), SVector{3}(chi2), omega_ref, omega_start, Val(N), p, H)
end

function _seobv5phm_pn_table(m1, m2, chi1::SVector{3}, chi2::SVector{3}, omega_ref, omega_start, ::Val{N},
        p = _SEOBV5PHM_PN_P, H = _seobv5phm_pn_hstep(m1, m2)) where {N}
    T = promote_type(typeof(m1), typeof(m2), eltype(chi1), eltype(chi2), typeof(omega_ref), typeof(omega_start))
    Z = zeros(T, 9, N)
    DZ = zeros(T, 9, N)
    x1, h, n, omega_end = _seobv5phm_pn_fill!(Z, DZ, m1, m2, chi1, chi2, omega_ref, omega_start, N, p; H)
    return SEOBNRv5PHMPNTable(_PNGrid(x1, h), _PNCols(Z), _PNCols(DZ), n, omega_end, T(m1), T(m2), Float64(p))
end

"""
    _seobv5phm_pn_fill!(Z, DZ, m1, m2, chi1, chi2, omega_ref, omega_start, N, p; H, ZC = nothing) -> (x1, h, n, omega_end)

The table's integration (see [`SEOBNRv5PHMPNTable`](@ref)) writing the node states and their derivatives into the
columns of `Z`, `DZ` (9 x N; any writable matrix, e.g. a walker's slice of a device array). Columns past `n` hold
the last state and zero derivatives. Wrap the result with [`_seobv5phm_pn_view`](@ref). `ZC` (9 x N, optional):
receives the start state of every step that carries the spins from `omega_ref` to the table start (for a reverse
sweep).
"""
function _seobv5phm_pn_fill!(Z, DZ, m1, m2, chi1, chi2, omega_ref, omega_start, N, p; H = _seobv5phm_pn_hstep(m1, m2), ZC = nothing)
    T = eltype(Z)
    p = Float64(p)
    nu = m1 * m2 / (m1 + m2)^2
    z0 = SVector{9,T}(0, 0, 1, chi1[1] * m1^2, chi1[2] * m1^2, chi1[3] * m1^2, chi2[1] * m2^2, chi2[2] * m2^2,
        chi2[3] * m2^2)
    x_lo = _seobv5phm_pn_x(T(0.9) * omega_start, p)
    x_ref = _seobv5phm_pn_x(T(omega_ref), p)
    x_hi = _seobv5phm_pn_x(T(_SEOBV5PHM_PN_OMEGA_MAX), p)
    # carry the spins from omega_ref to the table start (backwards in time unless omega_ref < 0.9 omega_start)
    M = _seobv5phm_pn_m(x_lo, x_hi, N, H)
    h = (x_hi - x_lo) / (M - 1)
    z = z0
    NB = clamp(ceil(Int, abs((x_lo - x_ref) / h)), 1, N)
    hb = (x_lo - x_ref) / NB
    for i in 0:NB-1
        if ZC !== nothing
            @inbounds for c in 1:9
                ZC[c, i+1] = z[c]
            end
        end
        z, _ = _seobv5phm_pn_step(z, x_ref + i * hb, hb, p, nu, m1, m2)
    end
    n = M
    done = false
    @inbounds for i in 1:N
        done |= i > M
        x = x_lo + (min(i, M) - 1) * h                  # (padding past M: zeroed below, kept inside the range)
        d, _ = _seobv5phm_pn_dx(z, x, p, nu, m1, m2)
        d = ifelse(done, zero(d), d)
        for c in 1:9
            Z[c, i] = z[c]
            DZ[c, i] = d[c]
        end
        if !done && i < M
            znew, ok = _seobv5phm_pn_step(z, x, h, p, nu, m1, m2)
            if ok && all(isfinite, znew)
                z = znew
            else
                n = i
                done = true
            end
        end
    end
    return x_lo, h, n, _seobv5phm_pn_om(x_lo + (n - 1) * h, p)
end

# A table over caller-owned storage (Z, DZ as filled by `_seobv5phm_pn_fill!`).
@inline _seobv5phm_pn_view(Z, DZ, x1, h, n, omega_end, m1, m2, p) =
    SEOBNRv5PHMPNTable(_PNGrid(x1, h), _PNCols(Z), _PNCols(DZ), n, omega_end, m1, m2, Float64(p))

# The table interval of `omega` (clamped to the table range): the cubic `seobnrv5phm_pn_state` uses there.
@inline function _seobv5phm_pn_interval(tab::SEOBNRv5PHMPNTable, omega)
    y = (_seobv5phm_pn_x(omega, tab.p) - tab.x[1]) / (tab.x[2] - tab.x[1])
    y = clamp(y, zero(y), oftype(y, tab.n - 1))
    return min(floor(Int, y), tab.n - 2) + 1
end

"""
    seobnrv5phm_pn_state(tab, omega[, i]) -> (LN_hat, S1, S2)

Cubic Hermite interpolation of the table's state at `omega` (clamped to the table range). `LN_hat` is returned
normalized (as pyseobnr's `build_splines_PN` renormalizes its samples). With `i`, the cubic of table interval `i`,
also outside it (the plunge's RK stages: the spins of a step from one interval, see `_seobv5phm_rk4_inc`).
"""
@inline seobnrv5phm_pn_state(tab::SEOBNRv5PHMPNTable, omega) =
    seobnrv5phm_pn_state(tab, omega, _seobv5phm_pn_interval(tab, omega))
@inline function seobnrv5phm_pn_state(tab::SEOBNRv5PHMPNTable, omega, i::Integer)
    xq = _seobv5phm_pn_x(omega, tab.p)
    h = tab.x[2] - tab.x[1]
    y = (xq - tab.x[1]) / h
    y = clamp(y, zero(y), oftype(y, tab.n - 1))
    s = y - (i - 1)
    s2 = s * s
    s3 = s2 * s
    h00 = 2s3 - 3s2 + 1
    h10 = s3 - 2s2 + s
    h01 = -2s3 + 3s2
    h11 = s3 - s2
    z = h00 * tab.z[i] + (h10 * h) * tab.dz[i] + h01 * tab.z[i+1] + (h11 * h) * tab.dz[i+1]
    Lh = SVector(z[1], z[2], z[3])
    Lh = Lh / norm(Lh)
    return Lh, SVector(z[4], z[5], z[6]), SVector(z[7], z[8], z[9])
end

"""
    seobnrv5phm_pn_everything(tab, omega[, i]) -> SVector{16}

The 16 quantities of pyseobnr's `build_splines_PN(...)["everything"]` at `omega`: `chi1_LN, chi2_LN, chi1_L,
chi2_L, chi1_v (3), chi2_v (3), LN_hat (3), L_vec (3)`, computed from the interpolated state (of interval `i`).
"""
@inline function seobnrv5phm_pn_everything(tab::SEOBNRv5PHMPNTable, omega, i...)
    m1, m2 = tab.m1, tab.m2
    Lh, S1, S2 = seobnrv5phm_pn_state(tab, omega, i...)
    nu = m1 * m2 / (m1 + m2)^2
    X1 = m1 / (m1 + m2)
    X2 = m2 / (m1 + m2)
    q = m1 / m2
    v = omega^(one(omega) / 3)
    chi1 = S1 / m1^2
    chi2 = S2 / m2^2
    lNS1 = dot(S1, Lh)
    lNS2 = dot(S2, Lh)
    L = _seobv5phm_Lvec(q, nu, X1, X2, v, S1, S2, Lh, dot(S1, S1), dot(S2, S2), dot(S1, S2), lNS1, lNS2)
    Lhat = L / norm(L)
    return SVector(dot(chi1, Lh), dot(chi2, Lh), dot(chi1, Lhat), dot(chi2, Lhat), chi1..., chi2..., Lh..., L...)
end
