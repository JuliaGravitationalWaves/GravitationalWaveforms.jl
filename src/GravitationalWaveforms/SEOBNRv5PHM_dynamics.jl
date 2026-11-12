using StaticArrays: SVector
using LinearAlgebra: dot, norm

export seobnrv5phm_dynamics

# sqrt(u) with a finite derivative at u = 0 (the limit 0 there): the in-plane spin magnitudes vanish for spins along L,
# where sqrt's derivative is infinite and a reverse pass gives 0 * Inf = NaN. Safe-denominator form, so neither branch
# forms a NaN (GPU code may evaluate both).
@inline _seobv5phm_ssqrt(u) = ifelse(u > floatmin(typeof(u)), sqrt(ifelse(u > floatmin(typeof(u)), u, one(u))), zero(u))

# log, sqrt and non-integer powers for the generated Hamiltonian (`SEOBNRv5_ham_symbolic_generated.jl`): NaN outside their
# domain, as pyseobnr's C code, instead of a DomainError (which on a GPU aborts the whole batch). An RK4 stage near the
# end of the plunge can leave the Hamiltonian's domain (seen at m1 = 55.4, m2 = 15.9, r ~ 1.52 with dr/dt -> 0 and
# p_r* running away); `_seobv5phm_plunge!` then stops on the non-finite state (stop = 2, pyseobnr's check_terminal).
@inline _seobv5phm_nlog(x) = x < 0 ? oftype(x, NaN) : log(x)
@inline _seobv5phm_nsqrt(x) = x < 0 ? oftype(x, NaN) : sqrt(x)
@inline _seobv5phm_npow(x, y) = x < 0 ? oftype(x, NaN) : x^y

"""
SEOBNRv5PHM quasi-precessing EOB dynamics (milestone P2 of GravitationalWaveBayes' `SEOBNRv5PHM_DESIGN.md`).

pyseobnr 0.3.7's default precessing route (`GenerateWaveform`, `postadiabatic = True`):
`compute_combined_dynamics_exp_v1` in `eob/dynamics/postadiabatic_C_prec.pyx`, then `compute_dynamics_prec_opt`
(`integrate_ode_prec.py`). The orbital dynamics are the aligned-spin EOB equations of motion in `(r, phi, p_r*, p_phi)`
with the orbit-averaged precessing Hamiltonian `Hamiltonian_v5PHM` ([`_seobv5phm_ham_call`](@ref) and its derivatives,
generated from `gen/symbolic/hamiltonians/v5phm.jl`) and the SEOBNRv5 flux, where the spins entering both are read
from the PN spin evolution ([`SEOBNRv5PHMPNTable`](@ref), milestone P1) at the current orbital frequency:
the spin vectors (the Hamiltonian only uses `|chi_i|^2` and `chi_1 . chi_2`), their projections onto `LN_hat`
(`chi_i_LN`: Hamiltonian, flux spin combinations, `dSO` calibration fit) and onto `L_hat` (`chi_i_L`).

1. **Initial conditions** at `omega_start` (`computeIC_augm`): `(r0, pphi0)` from `dH/dpphi = omega_start`,
   `dH/dr = 0` (2-d Newton with the exact Hessian), `pr0` from `rdot = Fphi / (xi dpphi/dr)` with
   `dpphi/dr = -H_rr / H_r,pphi` ([`_seobv5phm_ic`](@ref)), spins at `omega_start`.
2. **Radial grid** (`compute_postadiabatic_dynamics`): from `r0` to `r_final = max(11, (2.7 + (chi_eff -
   chi_perp_eff/2)(1 - 4 nu)) r_ISCO)`, `r_ISCO` of the aligned HBR2016 remnant of the projected spins; the
   spacing gives 20 nodes per precession cycle (cycles of `LN_hat` up to `r_final`, from the PN evolution),
   clamped to `0.05 <= dr <= 0.1`; spin-aligned binaries (zero in-plane spin components) use `dr = 0.1`. The
   PA is skipped (full ODE from the initial conditions, as pyseobnr) if `ceil((r0 - r_final)/0.1) <= 10`,
   `r0 <= 11.5` or `r_final >= r0`.
3. **Postadiabatic solution**, analytic, 8 orders (`compute_postadiabatic_solution`): `pphi` from the adiabatic
   condition `dH/dr = 0` at `pr = 0`, then alternately `pr` (odd orders, [`_seobv5phm_pr_eqn`](@ref)) and `pphi`
   (even orders, [`_seobv5phm_pphi_eqn`](@ref)) with `dpphi/dr`, `dpr/dr` from a not-a-knot cubic spline.
   The spins at a node are taken at the orbital frequency of the previous order, exactly in pyseobnr's order
   (see [`_seobv5phm_pa_solution!`](@ref)). Time and phase are the not-a-knot spline integrals of `dt/dr`,
   `dphi/dr` (scipy's `InterpolatedUnivariateSpline` interpolant is the same spline).
4. **ODE** from the last PA node (`compute_dynamics_prec_opt`): pyseobnr integrates with RKF45 (tolerances 1e-8)
   and updates the spins after every accepted step at that step's orbital frequency (spins constant within a step).
   Here: classical RK4 with the spins read at each stage's own orbital frequency (`_seobv5phm_rk4_inc`; pyseobnr's
   per-step update is an O(h) error at RK4's step sizes), a fixed number of steps (`NRK`, steps after the stop are
   masked) and the step `dt = clamp(min(c_om / Omega, c_r r / |dr/dt|), 0.02, 10)` (0.3, 0.01); stopping conditions as
   `check_terminal` (orbital-frequency peak, `r < 1.4`, for `r < 3`: `dr/dt > 0`, `dpr/dt > 0`,
   `omega_circ > 1`; `omega > omega_PN_end`; `|a_+|, |a_-| > 1`). The step that triggers a stop is kept, as in
   pyseobnr (its post-peak point is used for the peak fit).

Differences from pyseobnr by construction: the spins come from the P1 table (pyseobnr's DOP853 + cubic spline
differs by up to ~1e-4 in the spin components), and the ODE uses RK4 steps with stage spins instead of RKF45's with
per-step spins (pyseobnr's trajectory depends on its step sizes through that update at a similar level; ours converges
with the step rule: ||dh||^2/||h||^2 <= 2.3e-5 against a 4x finer rule over 32 binaries, was 4.3e-4).
"""
struct SEOBNRv5PHMDynamics end

const _SEOBV5PHM_NPA = 512           # maximum PA nodes (pyseobnr: up to (r0 - r_final) / 0.05)
const _SEOBV5PHM_NRK = 1024          # ODE steps (fixed trip count)
# ODE step rule dt = clamp(min(C_OM / Omega, C_R r / |dr/dt|), 0.02, 10) (the default of every caller, incl. the GravitationalWaveBayes
# device dynamics and its reverse)
const _SEOBV5PHM_C_OM = 0.3
const _SEOBV5PHM_C_R = 0.01

# ---------------------------------------------------------------------------------------------------
# Per-binary constants and the spins at an orbital frequency
# ---------------------------------------------------------------------------------------------------

"""
    _seobv5phm_base(m1, m2, chi1, chi2, omega_ref, omega_start; N = 512, p = 3, H)

Per-binary constants (`m1 + m2 = 1`, `m1 >= m2`; spin vectors at `omega_ref` in the frame with `LN_hat = z`):
`nu, X1, X2, delta, a6`, the Newtonian multipole magnitudes the flux needs, and the PN spin table.
"""
function _seobv5phm_base(m1, m2, chi1, chi2, omega_ref, omega_start; N::Int = _SEOBV5PHM_PN_N, p = _SEOBV5PHM_PN_P,
        H = _seobv5phm_pn_hstep(m1, m2))
    tab = _seobv5phm_pn_table(m1, m2, SVector{3}(chi1), SVector{3}(chi2), omega_ref, omega_start, Val(N), p, H)
    return _seobv5phm_base_tab(m1, m2, tab, omega_ref, omega_start)
end

# The per-binary constants around an existing PN table (e.g. a device walker's view, `_seobv5phm_pn_view`).
@inline function _seobv5phm_base_tab(m1, m2, tab, omega_ref, omega_start)
    nu = m1 * m2
    X1, X2 = m1, m2
    absN = (_seobv5_multipole_prefix_abs(nu, X1, X2, 3, 3), _seobv5_multipole_prefix_abs(nu, X1, X2, 2, 1),
        _seobv5_multipole_prefix_abs(nu, X1, X2, 4, 4), _seobv5_multipole_prefix_abs(nu, X1, X2, 3, 2),
        _seobv5_multipole_prefix_abs(nu, X1, X2, 5, 5), _seobv5_multipole_prefix_abs(nu, X1, X2, 4, 3))
    return (nu = nu, X1 = X1, X2 = X2, delta = m1 - m2, a6 = _seobv5_a6_NS(nu), absN = absN, tab = tab,
        omega_ref = omega_ref, omega_start = omega_start)
end

"""
    _seobv5phm_spins(b, omega[, i])

Everything the Hamiltonian and the flux read from the PN evolution at orbital frequency `omega`: spin vectors
`c1`, `c2`, projections `chi1LN, chi2LN` (onto `LN_hat`) and `chi1L, chi2L` (onto `L_hat`), `LNhat`, the
calibration `dSO(nu, a_+, a_-)` of the `LN_hat` projections and the flux coefficients `fc` of
`chi_S, chi_A` = half sum/difference of the `LN_hat` projections (pyseobnr `update_spins`). With `i`, from the PN
table's interval `i` (see `seobnrv5phm_pn_state`).
"""
@inline function _seobv5phm_spins(b, omega, i...)
    e = seobnrv5phm_pn_everything(b.tab, omega, i...)
    chi1LN, chi2LN, chi1L, chi2L = e[1], e[2], e[3], e[4]
    nu, X1, X2, delta = b.nu, b.X1, b.X2, b.delta
    ap = X1 * chi1LN + X2 * chi2LN
    am = X1 * chi1LN - X2 * chi2LN
    chiS = (chi1LN + chi2LN) / 2
    chiA = (chi1LN - chi2LN) / 2
    atest = (1 - 2 * nu) * chiS + delta * chiA
    fc = _seobv5_flux_coeffs(nu, delta, chiS, chiA, atest, b.absN...)
    return (c1 = SVector(e[5], e[6], e[7]), c2 = SVector(e[8], e[9], e[10]), chi1LN = chi1LN, chi2LN = chi2LN,
        chi1L = chi1L, chi2L = chi2L, LNhat = SVector(e[11], e[12], e[13]), ap = ap, am = am,
        dSO = _seobv5_dSO(nu, ap, am), fc = fc)
end

# The generated Hamiltonian's argument list at (r, prst, pphi) with spins `s`
@inline _seobv5phm_hargs(r, prst, pphi, s, b) = (r, prst, pphi, s.c1[1], s.c1[2], s.c1[3], s.c2[1], s.c2[2],
    s.c2[3], s.chi1LN, s.chi2LN, s.chi1L, s.chi2L, b.nu, b.X1, b.X2, b.delta, b.a6, s.dSO)

# Heff from H = sqrt(1 + 2 nu (Heff - 1)) / nu (the source term pyseobnr's flux rebuilds from H)
@inline _seobv5phm_heff(H, nu) = ((nu * H)^2 - 1) / (2 * nu) + 1

"""
    _seobv5phm_rhs(r, prst, pphi, s, b) -> (rdot, phidot, prdot, pphidot, omega_circ)

`get_rhs_prec` (`rhs_precessing.pyx`) with spins `s`: `rdot = xi dH/dpr`, `phidot = dH/dpphi`,
`prdot = -xi dH/dr + F_r`, `pphidot = F_phi`.
"""
@inline function _seobv5phm_rhs(r, prst, pphi, s, b)
    dHdr, dHdpr, dHdpphi, H, xi, om_c = _seobv5phm_sym_rhs(_seobv5phm_hargs(r, prst, pphi, s, b)...)
    Fr, Fphi = _seobv5_rr_force(prst, pphi, dHdpphi, om_c, _seobv5phm_heff(H, b.nu), H, b.nu, s.fc)
    return xi * dHdpr, dHdpphi, -xi * dHdr + Fr, Fphi, om_c
end

# ---------------------------------------------------------------------------------------------------
# Initial conditions (computeIC_augm)
# ---------------------------------------------------------------------------------------------------

"""
    _seobv5phm_ic(b, s, omega; newton = 8, secant = 6) -> (r0, pphi0, pr0)

`computeIC_augm` (`initial_conditions_aligned_precessing.py`) with fixed spins `s`: Newton on
`(dH/dpphi - omega, dH/dr) = 0` in `(r, pphi)` from the Newtonian guess (pyseobnr: `scipy.optimize.root`, tol
1e-12), then `pr` from `dH/dpr = Fphi / (xi dpphi/dr)`, `dpphi/dr = -H_rr / H_r,pphi` at `(r0, pr, pphi0)`
(pyseobnr: Brent on `[-0.03, 0]`), here by secant iterations from `pr = 0`. Fixed iteration counts.
"""
# (counts: round-off floor reached after 4 Newton / 4 secant steps over q 1-20, |chi| < 0.99, omega 0.002-0.06)
function _seobv5phm_ic(b, s, omega; newton::Int = 8, secant::Int = 6)
    r = omega^(-oftype(omega, 2) / 3)
    j = sqrt(r)
    z = zero(r)
    for _ in 1:newton
        dHdr, dHdpphi, Hrr, Hrj, Hjj = _seobv5phm_sym_circ_hessian(_seobv5phm_hargs(r, z, j, s, b)...)
        f1 = dHdpphi - omega
        f2 = dHdr
        det = Hrj * Hrj - Hjj * Hrr        # J = [Hrj Hjj; Hrr Hrj] for (f1, f2) in (r, j)
        dr = (Hrj * f1 - Hjj * f2) / det
        dj = (Hrj * f2 - Hrr * f1) / det
        r -= dr
        j -= dj
    end
    # (no closure over the reassigned r, j: boxed captures are type-unstable and fail in GPU kernels)
    p0 = zero(r)
    p1 = oftype(r, -1e-4)
    g0 = _seobv5phm_ic_diss(p0, r, j, s, b)
    g1 = _seobv5phm_ic_diss(p1, r, j, s, b)
    for _ in 1:secant
        den = g1 - g0
        p2 = ifelse(den == 0, p1, p1 - g1 * (p1 - p0) / ifelse(den == 0, one(den), den))
        p0, g0 = p1, g1
        p1 = p2
        g1 = _seobv5phm_ic_diss(p1, r, j, s, b)
    end
    return r, j, p1
end

@inline function _seobv5phm_ic_diss(pr, r, j, s, b)
    H, xi, dHdpr, dHdpphi, om_c, Hrr, Hrj = _seobv5phm_sym_diss_terms(_seobv5phm_hargs(r, pr, j, s, b)...)
    dLdr = -Hrr / Hrj
    _, Fphi = _seobv5_rr_force(pr, j, dHdpphi, om_c, _seobv5phm_heff(H, b.nu), H, b.nu, s.fc)
    return Fphi / (xi * dLdr) - dHdpr
end

# ---------------------------------------------------------------------------------------------------
# Not-a-knot cubic spline on a uniform grid: node slopes and cumulative integral
# ---------------------------------------------------------------------------------------------------

"""
    _seobv5phm_spline_slopes!(s, y, h, n, w)

Node derivatives of the not-a-knot cubic spline through `y[1:n]` on a uniform grid of (signed) spacing `h`
(scipy `CubicSpline(..., bc_type="not-a-knot")`, and FITPACK's interpolating `k = 3` spline, which has the same
knots): `s_1 + 2 s_2 = (5 m_1 + m_2)/2`, `s_{i-1} + 4 s_i + s_{i+1} = 3 (m_{i-1} + m_i)`,
`2 s_{n-1} + s_n = (m_{n-2} + 5 m_{n-1})/2` with `m_i = (y_{i+1} - y_i)/h`; Thomas algorithm (`w` scratch).
"""
function _seobv5phm_spline_slopes!(s, y, h, n, w)
    @inbounds begin
        m1 = (y[2] - y[1]) / h
        m2 = (y[3] - y[2]) / h
        # forward sweep: w[i] = modified super-diagonal, s[i] = modified rhs
        w[1] = 2 * one(h)
        s[1] = (5 * m1 + m2) / 2
        for i in 2:n-1
            mi1 = (y[i] - y[i-1]) / h
            mi = (y[i+1] - y[i]) / h
            den = 4 - w[i-1]
            w[i] = 1 / den
            s[i] = (3 * (mi1 + mi) - s[i-1]) / den
        end
        mn2 = (y[n-1] - y[n-2]) / h
        mn1 = (y[n] - y[n-1]) / h
        den = 1 - 2 * w[n-1]
        s[n] = ((mn2 + 5 * mn1) / 2 - 2 * s[n-1]) / den
        for i in n-1:-1:1
            s[i] = s[i] - w[i] * s[i+1]
        end
    end
    return s
end

"""
    _seobv5phm_spline_cumint!(I, y, s, h, n)

`I[i] = integral of the Hermite cubic with node values `y` and slopes `s` from node 1 to node i` (signed `h`).
"""
function _seobv5phm_spline_cumint!(I, y, s, h, n)
    @inbounds begin
        I[1] = zero(eltype(I))
        for i in 1:n-1
            I[i+1] = I[i] + h / 2 * (y[i] + y[i+1]) + h * h / 12 * (s[i] - s[i+1])
        end
    end
    return I
end

# ---------------------------------------------------------------------------------------------------
# Postadiabatic equations (analytic, pr_eqn_analytic / pphi_eqn_analytic)
# ---------------------------------------------------------------------------------------------------

"""
    _seobv5phm_j0(r, s, b; newton = 8) -> pphi

Adiabatic angular momentum: `dH/dr (r, 0, pphi) = 0` by Newton from `sqrt(r)` with the exact `H_r,pphi`
(`compute_adiabatic_solution`, pyseobnr: `scipy.optimize.root`, tol 1e-12).
"""
@inline function _seobv5phm_j0(r, s, b; newton::Int = 8)      # converged to round-off after 5
    j = sqrt(r)
    z = zero(r)
    for _ in 1:newton
        dHdr, Hrj = _seobv5phm_sym_j0(_seobv5phm_hargs(r, z, j, s, b)...)
        j -= dHdr / Hrj
    end
    return j
end

"""
    _seobv5phm_pr_eqn(pr, r, pphi, dpphi_dr, s, b) -> (pr_new, omega)

`pr_eqn_analytic`: `pr = xi/(2(1 + Bnp)) (F_phi / (dpphi/dr) 2 nu H Heven / A - xi dQ/dpr)` with everything at
`(r, pr, pphi)` (the previous order's `pr`); also returns `omega = dH/dpphi` there.
"""
@inline function _seobv5phm_pr_eqn(pr, r, pphi, dpphi_dr, s, b)
    H, xi, A, Bnp, Heven, Hodd, dQdprst, om, om_c = _seobv5phm_sym_pr_terms(_seobv5phm_hargs(r, pr, pphi, s, b)...)
    nu = b.nu
    _, Fphi = _seobv5_rr_force(pr, pphi, om, om_c, _seobv5phm_heff(H, nu), H, nu, s.fc)
    return xi / (2 * (1 + Bnp)) * (Fphi / dpphi_dr * 2 * nu * H * Heven / A - xi * dQdprst), om
end

"""
    _seobv5phm_pphi_eqn(pphi, r, pr, dpr_dr, s, b) -> (pphi_new, omega)

`pphi_eqn_analytic`: the root `(-C1 - sqrt(D)) / (2 C2)` of the quadratic in `pphi` from `dpr/dr dH/dpr + dH/dr -
F_r / xi = 0`, coefficients at `(r, pr, pphi)` (the previous order's `pphi`); `D` is clamped at 0 (pyseobnr
raises for `D < -1e-10`, which only happens far outside the inspiral); also returns `omega = dH/dpphi` there.
"""
@inline function _seobv5phm_pphi_eqn(pphi, r, pr, dpr_dr, s, b)
    H, xi, A, Bnp, Bnpa, Q, Heven, Hodd, Bp, dAdr, dBnpdr, dBnpadr, dxidr, dQdr, dQdprst, dHodddr, dBpdr, om, om_c =
        _seobv5phm_sym_pphi_terms(_seobv5phm_hargs(r, pr, pphi, s, b)...)
    nu = b.nu
    drdt = A / (2 * nu * H * Heven) * (2 * pr / xi * (1 + Bnp) + xi * dQdprst)
    _, Fphi = _seobv5_rr_force(pr, pphi, om, om_c, _seobv5phm_heff(H, nu), H, nu, s.fc)
    pr2 = pr * pr
    xi2 = xi * xi
    xi3 = xi2 * xi
    r2 = r * r
    r3 = r2 * r
    lap = s.chi1L * b.X1 + s.chi2L * b.X2
    lap2 = lap * lap
    C2 = dAdr * (Bp / r2 + lap2 / r2 * Bnpa) + A * (-2 * Bp / r3 - 2 * lap2 / r3 * Bnpa + lap2 / r2 * dBnpadr + dBpdr / r2)
    C1 = dHodddr / pphi * 2 * Heven
    C0a = dAdr * (1 + pr2 / xi2 * (1 + Bnp) + Q) + A * (dQdr - 2 * pr2 / xi3 * dxidr * (1 + Bnp) + pr2 / xi2 * dBnpdr)
    tmp = 2 * Heven * nu * H / xi
    C0 = C0a + dpr_dr * drdt * tmp - pr / pphi * tmp * Fphi
    D = max(C1 * C1 - 4 * C2 * C0, zero(C1))
    return (-C1 - sqrt(D)) / (2 * C2), om
end

# ---------------------------------------------------------------------------------------------------
# Radial grid
# ---------------------------------------------------------------------------------------------------

"""
    _seobv5phm_prec_cycles(tab, omega_f)

`compute_prec_cycles`: number of precession cycles of `LN_hat` from the start of the PN evolution to orbital
frequency `omega_f`, `(1/2pi) int |dLN_hat/dt| dt`, here `int |dLN_hat/dx| |dx|` over the table nodes (trapezoid).
"""
function _seobv5phm_prec_cycles(tab::SEOBNRv5PHMPNTable, omega_f)
    xf = _seobv5phm_pn_x(omega_f, tab.p)
    h = tab.x[2] - tab.x[1]
    acc = zero(promote_type(typeof(tab.omega_end), typeof(omega_f)))
    g(i) = norm(SVector(tab.dz[i][1], tab.dz[i][2], tab.dz[i][3])) * abs(h)
    @inbounds for i in 1:tab.n-1
        xi, xj = tab.x[i], tab.x[i+1]          # decreasing x
        if xj >= xf
            acc += (g(i) + g(i + 1)) / 2
        elseif xi > xf
            f = (xi - xf) / (xi - xj)
            gf = g(i) + f * (g(i + 1) - g(i))
            acc += f * (g(i) + gf) / 2
        end
    end
    return acc / (2 * pi)
end

"""
    _seobv5phm_pa_grid(b, s0, r0) -> (success, n, r_final)

`compute_postadiabatic_dynamics`' grid decision with spins `s0` at `omega_start` (see the module notes).
"""
# the PA grid's end radius
@inline function _seobv5phm_pa_rfinal(b, s0, r0)
    nu, X1, X2 = b.nu, b.X1, b.X2
    ap = s0.ap
    chi_perp_eff = _seobv5phm_ssqrt(sum(abs2, X1 * s0.c1 + X2 * s0.c2 - ap * s0.LNhat))
    r_isco, _ = _seobv5_kerr_isco(s0.chi1LN, s0.chi2LN, X1, X2)
    return max(oftype(r0, 11), (oftype(r0, 2.7) + (ap - chi_perp_eff / 2) * (1 - 4 * nu)) * r_isco)
end
function _seobv5phm_pa_grid(b, s0, r0)
    r_final = _seobv5phm_pa_rfinal(b, s0, r0)
    r_range = r0 - r_final
    r_size = isfinite(r_range) ? ceil(Int, clamp(r_range / 0.1, -1e6, 1e6)) : 0
    success = !(r_size <= 10 || r0 <= 11.5 || r_final >= r0)
    chi_in_plane = s0.c1[1]^2 + s0.c1[2]^2 + s0.c2[1]^2 + s0.c2[2]^2
    n = 0
    if success
        if chi_in_plane > 0
            n = ceil(Int, 20 * _seobv5phm_prec_cycles(b.tab, r_final^(-oftype(r0, 1.5))))
        end
        if n == 0
            n = ceil(Int, r_range / 0.1)
        else
            dr = r_range / n
            if dr < 0.05
                n = ceil(Int, r_range / 0.05)
            elseif dr > 0.1
                n = ceil(Int, r_range / 0.1)
            end
        end
    end
    return success, n, r_final
end

# ---------------------------------------------------------------------------------------------------
# Postadiabatic solution
# ---------------------------------------------------------------------------------------------------

"""
    _seobv5phm_pa_solution!(pr, pphi, om, r, n, b, d, w; order = 8)

`compute_adiabatic_solution` + `compute_postadiabatic_solution` on the grid `r[1:n]` (uniform, decreasing; `d`, `w`
scratch of length >= n). Spin
bookkeeping as pyseobnr: the adiabatic `pphi` uses spins at `omega = r^(-3/2)`; then `om[i] = dH/dpphi(r, 0,
pphi)` with those spins. Each order: spins at the current `om[i]`; the odd/even equation (evaluated at the previous
`pr`/`pphi`) also returns `dH/dpphi` there, which replaces `om[i]` (pyseobnr updates the frequency array in place);
after the sweep, `om[i] = dH/dpphi(r, pr, pphi)` with spins at that replaced value.
"""
function _seobv5phm_pa_solution!(pr, pphi, om, r, n, b, d, w; order::Int = 8)
    h = r[2] - r[1]
    @inbounds for i in 1:n
        pphi[i], om[i] = _seobv5phm_pa_adiabatic(r[i], b)
        pr[i] = zero(eltype(pr))
    end
    for k in 1:order
        _seobv5phm_spline_slopes!(d, isodd(k) ? pphi : pr, h, n, w)
        @inbounds for i in 1:n
            pr[i], pphi[i], om[i] = _seobv5phm_pa_update(k, r[i], pr[i], pphi[i], om[i], d[i], b)
        end
    end
    return pr, pphi, om
end

# The PA pieces node by node (the device runs them node-parallel; their reverse is per node):
# adiabatic start at r: (pphi, omega) with spins at r^(-3/2), then omega = dH/dpphi(r, 0, pphi) with those spins.
@inline function _seobv5phm_pa_adiabatic(r, b)
    T = typeof(r)
    s = _seobv5phm_spins(b, r^(-T(1.5)))
    pphi = _seobv5phm_j0(r, s, b)
    return pphi, _seobv5phm_sym_omega_circ(_seobv5phm_hargs(r, zero(T), pphi, s, b)...)
end
# The same from its converged pphi (`jc`, held constant): one more Newton step, which leaves it unchanged and carries
# the implicit-function derivative (for a device reverse without the Newton loop on its tape).
@inline function _seobv5phm_pa_adiabatic_from(r, b, jc)
    T = typeof(r)
    s = _seobv5phm_spins(b, r^(-T(1.5)))
    dHdr, Hrj = _seobv5phm_sym_j0(_seobv5phm_hargs(r, zero(T), jc, s, b)...)
    pphi = jc - dHdr / Hrj
    return pphi, _seobv5phm_sym_omega_circ(_seobv5phm_hargs(r, zero(T), pphi, s, b)...)
end
# order k at one node, with d = the slope (d pphi/dr for odd k, d pr/dr for even k) of the previous order's spline:
# the odd (pr) / even (pphi) equation with spins at om, which also returns dH/dpphi there; then om = dH/dpphi at the
# new state with spins at that value.
@inline function _seobv5phm_pa_update(k, r, pr, pphi, om, d, b)
    s = _seobv5phm_spins(b, om)
    if isodd(k)
        pr, om = _seobv5phm_pr_eqn(pr, r, pphi, d, s, b)
    else
        pphi, om = _seobv5phm_pphi_eqn(pphi, r, pr, d, s, b)
    end
    s = _seobv5phm_spins(b, om)
    return pr, pphi, _seobv5phm_ham_omega(_seobv5phm_hargs(r, pr, pphi, s, b)...)
end
# dt/dr and dphi/dr at a PA node (spins at om)
@inline function _seobv5phm_pa_dtdr(r, pr, pp, om, b)
    s = _seobv5phm_spins(b, om)
    xi, dHdpr, dHdpphi = _seobv5phm_sym_tphi(_seobv5phm_hargs(r, pr, pp, s, b)...)
    return 1 / (dHdpr * xi), dHdpphi / (dHdpr * xi)
end
# The RK4 increments of (r, phi, pr, pphi) over a step h from (r, pr, pphi) with first-stage rates k1 (spins s, read
# at a frequency in PN-table interval i). The later stages read the spins at their own orbital frequency, predicted as
# dH/dpphi at the stage's state with the previous stage's spins, from interval i's cubic (one table window per step
# for the GravitationalWaveBayes device reverse). pyseobnr keeps the step's start spins for all stages (harmless at its RKF45
# tolerances); with RK4's steps that is an O(h) error, up to 4e-4 in ||dh||^2/||h||^2 at the default step rule.
@inline function _seobv5phm_stage(r, pr, pp, sp, i, b)
    s = _seobv5phm_spins(b, _seobv5phm_ham_omega(_seobv5phm_hargs(r, pr, pp, sp, b)...), i)
    return _seobv5phm_rhs(r, pr, pp, s, b), s
end
@inline function _seobv5phm_rk4_inc(r, pr, pp, h, k1, s, i, b)
    k1r, k1f, k1p, k1j = k1
    (k2r, k2f, k2p, k2j, _), s2 = _seobv5phm_stage(r + h / 2 * k1r, pr + h / 2 * k1p, pp + h / 2 * k1j, s, i, b)
    # stage 3 (the same time as stage 2) with stage 2's spins, stage 4's at Omega extrapolated from stages 1 and 3:
    # both O(h^2) in the spins' frequency, without a spin read / Hamiltonian evaluation each (a spin read costs two
    # right-hand sides: the flux coefficients)
    k3r, k3f, k3p, k3j, _ = _seobv5phm_rhs(r + h / 2 * k2r, pr + h / 2 * k2p, pp + h / 2 * k2j, s2, b)
    k4r, k4f, k4p, k4j, _ = _seobv5phm_rhs(r + h * k3r, pr + h * k3p, pp + h * k3j, _seobv5phm_spins(b, 2k3f - k1f, i), b)
    return h / 6 * (k1r + 2k2r + 2k3r + k4r), h / 6 * (k1f + 2k2f + 2k3f + k4f), h / 6 * (k1p + 2k2p + 2k3p + k4p),
        h / 6 * (k1j + 2k2j + 2k3j + k4j)
end
# One plunge step from (r, pr, pphi) with spins s read at om_s: the step size (from the state's rates) and the RK4
# increments.
@inline function _seobv5phm_plunge_step(sr, spr, spp, s, om_s, b, c_om, c_r)
    T = typeof(sr)
    k1 = _seobv5phm_rhs(sr, spr, spp, s, b)
    hstep = clamp(min(c_om / k1[2], c_r * sr / abs(k1[1])), T(0.02), T(10))
    return hstep, _seobv5phm_rk4_inc(sr, spr, spp, hstep, k1, s, _seobv5phm_pn_interval(b.tab, om_s), b)...
end

"""
    seobnrv5phm_dynamics(m1, m2, chi1, chi2, omega_ref, omega_start; NPA = 512, NRK = 1024, c_om = 0.3, c_r = 0.01)

The SEOBNRv5PHM orbital dynamics (see [`SEOBNRv5PHMDynamics`](@ref)) for masses `m1 + m2 = 1` and spin vectors
`chi1`, `chi2` at `omega_ref` (frame with `LN_hat = z`). Returns a NamedTuple of node vectors (length
`NPA + NRK`, valid `1:n`, `t` padded with unit steps beyond; `1:npa` are PA nodes): `t, r, phi, pr, pphi` with their
time derivatives `dr, dphi, dpr, dpphi`, `omega_s` (the orbital frequency the node's spins were read at),
`omega_circ`, plus `pa_success`, `r_final`, `r0, pphi0, pr0`, `stop` (stop reason, pyseobnr's `check_terminal`
codes; 0 if the step budget ran out) and the base `b`. Allocates and calls [`_seobv5phm_dynamics!`](@ref).
"""
function seobnrv5phm_dynamics(m1, m2, chi1, chi2, omega_ref, omega_start; NPA::Int = _SEOBV5PHM_NPA,
        NRK::Int = _SEOBV5PHM_NRK, c_om = _SEOBV5PHM_C_OM, c_r = _SEOBV5PHM_C_R, order::Int = 8,
        N::Int = _SEOBV5PHM_PN_N, p = _SEOBV5PHM_PN_P, H = _seobv5phm_pn_hstep(m1, m2))
    b = _seobv5phm_base(m1, m2, chi1, chi2, omega_ref, omega_start; N, p, H)
    T = promote_type(typeof(b.nu), typeof(b.tab.omega_end))
    ND = NPA + NRK
    B = (t = zeros(T, ND), r = zeros(T, ND), phi = zeros(T, ND), pr = zeros(T, ND), pphi = zeros(T, ND),
        dr = zeros(T, ND), dphi = zeros(T, ND), dpr = zeros(T, ND), dpphi = zeros(T, ND), omega_s = zeros(T, ND),
        omega_circ = zeros(T, ND), s1 = zeros(T, NPA), s2 = zeros(T, NPA), s3 = zeros(T, NPA), s4 = zeros(T, NPA))
    sc = _seobv5phm_dynamics!(B, b, T(omega_start); NPA = NPA, NRK = NRK, c_om = c_om, c_r = c_r, order = order)
    return (t = B.t, r = B.r, phi = B.phi, pr = B.pr, pphi = B.pphi, dr = B.dr, dphi = B.dphi, dpr = B.dpr,
        dpphi = B.dpphi, omega_s = B.omega_s, omega_circ = B.omega_circ, n = sc.n, npa = sc.npa,
        pa_success = sc.pa_success, r_final = sc.r_final, r0 = sc.r0, pphi0 = sc.pphi0, pr0 = sc.pr0, stop = sc.stop, b = b)
end

"""
    _seobv5phm_dynamics!(B, b, omega_start; NPA, NRK, c_om, c_r, order) -> NamedTuple of scalars

The dynamics of [`seobnrv5phm_dynamics`](@ref) into caller-owned buffers `B` (fields `t, r, phi, pr, pphi, dr, dphi,
dpr, dpphi, omega_s, omega_circ` of length >= `NPA + NRK`, scratch `s1 .. s4` of length >= `NPA`; e.g. one device
walker's views of global arrays), with the per-binary constants `b` ([`_seobv5phm_base_tab`](@ref)). Returns
`(n, npa, pa_success, r_final, r0, pphi0, pr0, stop)`. No allocation.
"""
function _seobv5phm_dynamics!(B, b, omega_start; NPA::Int = _SEOBV5PHM_NPA, NRK::Int = _SEOBV5PHM_NRK,
        c_om = _SEOBV5PHM_C_OM, c_r = _SEOBV5PHM_C_R, order::Int = 8)
    T = typeof(omega_start)
    t, r, ph, pr, pp = B.t, B.r, B.phi, B.pr, B.pphi
    dr, dph, dpr, dpp, oms, omc = B.dr, B.dphi, B.dpr, B.dpphi, B.omega_s, B.omega_circ
    ND = NPA + NRK
    @inbounds for i in 1:ND
        t[i] = zero(T); r[i] = zero(T); ph[i] = zero(T); pr[i] = zero(T); pp[i] = zero(T)
        dr[i] = zero(T); dph[i] = zero(T); dpr[i] = zero(T); dpp[i] = zero(T); oms[i] = zero(T); omc[i] = zero(T)
    end
    r0, pphi0, pr0, pa_success, npa, r_final = _seobv5phm_dyn_start(b, omega_start, NPA)
    if pa_success
        hgrid = (r_final - r0) / (npa - 1)
        @inbounds for i in 1:npa
            r[i] = r0 + (i - 1) * hgrid
        end
        _seobv5phm_pa_solution!(view(pr, 1:npa), view(pp, 1:npa), view(oms, 1:npa), view(r, 1:npa), npa, b,
            B.s1, B.s2; order = order)
        dtdr, dphidr, sl, w = B.s1, B.s2, B.s3, B.s4
        @inbounds for i in 1:npa
            dtdr[i], dphidr[i] = _seobv5phm_pa_dtdr(r[i], pr[i], pp[i], oms[i], b)
        end
        _seobv5phm_spline_slopes!(sl, dtdr, hgrid, npa, w)
        _seobv5phm_spline_cumint!(t, dtdr, sl, hgrid, npa)
        _seobv5phm_spline_slopes!(sl, dphidr, hgrid, npa, w)
        _seobv5phm_spline_cumint!(ph, dphidr, sl, hgrid, npa)
        @inbounds for i in 1:npa
            dr[i], dph[i], dpr[i], dpp[i], omc[i] = _seobv5phm_rhs(r[i], pr[i], pp[i], _seobv5phm_spins(b, oms[i]), b)
        end
    else
        npa = 1
        r[1], pr[1], pp[1] = r0, pr0, pphi0
        oms[1] = omega_start
        dr[1], dph[1], dpr[1], dpp[1], omc[1] = _seobv5phm_rhs(r0, pr0, pphi0, _seobv5phm_spins(b, omega_start), b)
    end
    n, stop = _seobv5phm_plunge!(B, b, npa, pa_success ? dph[npa] : omega_start, NRK, c_om, c_r)
    @inbounds for i in n+1:ND
        t[i] = t[n] + (i - n)      # strictly increasing padding for bracket searches
    end
    return (n = n, npa = npa, pa_success = pa_success, r_final = r_final, r0 = r0, pphi0 = pphi0, pr0 = pr0, stop = stop)
end

# Initial conditions and the PA grid decision: (r0, pphi0, pr0, pa_success, npa, r_final).
@inline function _seobv5phm_dyn_start(b, omega_start, NPA)
    s0 = _seobv5phm_spins(b, omega_start)
    r0, pphi0, pr0 = _seobv5phm_ic(b, s0, omega_start)
    pa_success, npa, r_final = _seobv5phm_pa_grid(b, s0, r0)
    return r0, pphi0, pr0, pa_success, min(npa, NPA), r_final
end
# (r0, pphi0, pr0, r_final) of the above from the converged initial conditions (held constant): one more Newton step
# in (r, pphi) and one in pr (slope by central differences), which leave them unchanged and carry the
# implicit-function derivatives -- for a device reverse without the solver loops on its tape.
@inline function _seobv5phm_dyn_start_from(b, omega_start, r0c, pphi0c, pr0c)
    s = _seobv5phm_spins(b, omega_start)
    dHdr, dHdpphi, Hrr, Hrj, Hjj = _seobv5phm_sym_circ_hessian(_seobv5phm_hargs(r0c, zero(r0c), pphi0c, s, b)...)
    f1 = dHdpphi - omega_start
    det = Hrj * Hrj - Hjj * Hrr
    r = r0c - (Hrj * f1 - Hjj * dHdr) / det
    j = pphi0c - (Hrj * dHdr - Hrr * f1) / det
    dp = 1e-6 * abs(pr0c) + 1e-12
    slope = (_seobv5phm_ic_diss(pr0c + dp, r, j, s, b) - _seobv5phm_ic_diss(pr0c - dp, r, j, s, b)) / (2dp)
    pr = pr0c - _seobv5phm_ic_diss(pr0c, r, j, s, b) / slope
    return r, j, pr, _seobv5phm_pa_rfinal(b, s, r)
end

"""
    _seobv5phm_plunge!(B, b, n0, om0, NRK, c_om, c_r) -> (n, stop)

The ODE from node `n0` of `B` (a step's first stage reads the spins at the frequency of the step's start, the later
stages at their own, see `_seobv5phm_rk4_inc`; the first step's at `om0` = dH/dpphi at the last PA node, or
omega_start): fixed-count RK4 with
`_seobv5phm_plunge_step`, nodes `n0 + 1 ..`, stop conditions of pyseobnr's `check_terminal`.
"""
function _seobv5phm_plunge!(B, b, n0, om0, NRK, c_om, c_r)
    t, r, ph, pr, pp = B.t, B.r, B.phi, B.pr, B.pphi
    dr, dph, dpr, dpp, oms, omc = B.dr, B.dphi, B.dpr, B.dpphi, B.omega_s, B.omega_circ
    omPN = b.tab.omega_end
    n = n0
    om_prev = om0
    s = _seobv5phm_spins(b, om_prev)
    sr, sph, spr, spp, st = r[n], ph[n], pr[n], pp[n], t[n]
    r_prev = sr
    stop = 0
    @inbounds for k in 1:NRK
        stop != 0 && break
        hstep, Dr, Dph, Dpr, Dpp = _seobv5phm_plunge_step(sr, spr, spp, s, om_prev, b, c_om, c_r)
        nr = sr + Dr; nph = sph + Dph; npr = spr + Dpr; npp = spp + Dpp
        if !(isfinite(nr) & isfinite(npr) & isfinite(npp))
            stop = 2
            break
        end
        a, om, c, d, oc = _seobv5phm_rhs(nr, npr, npp, s, b)
        # a non-finite rate at the new state (outside the Hamiltonian's domain): drop the step and stop, as pyseobnr
        # (`integrate_ode_prec`: a NaN omega removes the last point)
        if !(isfinite(a) & isfinite(om) & isfinite(c) & isfinite(d) & isfinite(oc))
            stop = 2
            break
        end
        n += 1
        st += hstep
        t[n] = st; r[n] = nr; ph[n] = nph; pr[n] = npr; pp[n] = npp
        dr[n] = a; dph[n] = om; dpr[n] = c; dpp[n] = d; omc[n] = oc
        oms[n] = om_prev    # this step ran with spins at the previous step's frequency
        # check_terminal (pyseobnr evaluates it for r <= 6) and the spin-extrapolation guard
        if nr <= 6
            stop = nr <= 1.4 ? 1 : !isfinite(om) ? 2 : om < om_prev ? 3 :
                   (nr < 3 && a > 0) ? 5 : (nr < 3 && c > 0) ? 6 : (nr < 3 && oc > 1) ? 7 :
                   (nr < 3 && nr > r_prev) ? 8 : om > omPN ? 9 : 0
        end
        stop != 0 && break
        s = _seobv5phm_spins(b, om)
        if abs(s.ap) > 1 || abs(s.am) > 1
            stop = 10
            break
        end
        om_prev = om
        sr, sph, spr, spp = nr, nph, npr, npp
        r_prev = nr
    end
    return n, stop
end
