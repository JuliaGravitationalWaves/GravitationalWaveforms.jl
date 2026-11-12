export seobnrv5alignedspin_postadiabatic_dynamics, seobnrv5alignedspin_combined_dynamics

"""
    SEOBNRv5AlignedSpinPA

Post-adiabatic (PA) approximation for
`SEOBNRv5AlignedSpin`, extending that file rather than replacing anything in it.

# What PA is

PA re-solves the *same* Hamiltonian/RR-force dynamics `SEOBNRv5AlignedSpin.jl` already has, but with
`r` (not `t`) as the independent variable: on a fixed radial grid `r0:dr0:r_final`, it solves for the
adiabatic (`epsilon=0`) angular momentum `j0(r)` (Eq. 2.5 of arXiv:2105.06983, via `dH/dr=0` at
`pr*=0`), then alternately corrects `pr*(r)` (odd orders, Eq. 2.6) and `pphi(r)` (even orders, Eq.
2.7) for a fixed 8 passes, and finally recovers `t(r)`/`phi(r)` by a cumulative quadrature of
`dt/dr`/`dphi/dr` over the already-tabulated grid -- no time-stepping anywhere in this phase. This
file ports the **"analytic"** variant (`postadiabatic_C_fast.pyx`) specifically -- closed-form
algebra (a rational expression for `pr`, a quadratic formula for `pphi`), not `postadiabatic_C.pyx`'s
"numeric" root-finding variant (the actual
`pyseobnr` production default, and the better fit to this package's Enzyme/KernelAbstractions-
readiness discipline).

# Provenance

Ported from these `pyseobnr` sources:
- `pyseobnr/eob/hamiltonian/Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C.pyx:1697-1899` -- the `auxderivs`
  cpdef method ([`_seobv5_auxderivs`](@ref)); generated from the Hamiltonian source
  (`gen/symbolic/generate.jl`), checked against pyseobnr's values in the tests.
- `pyseobnr/eob/utils/nr_utils.py:15-202` -- `bbh_final_spin_non_precessing_HBR2016`'s `"M3J4"`
  branch ([`_seobv5_final_spin_hbr2016`](@ref)), and `calc_isco_radius` (used inline by
  [`_seobv5_kerr_isco`](@ref)).
- `pyseobnr/eob/dynamics/postadiabatic_C.pyx:265-283` -- `Kerr_ISCO` ([`_seobv5_kerr_isco`](@ref)).
- `pyseobnr/eob/dynamics/postadiabatic_C.pyx:32-260` -- the order-9 finite-difference-derivative
  stencil (`fin_diff_coeffs_order_9`/`fin_diff_derivative`,
  [`_seobv5_findiff_derivative9`](@ref)) and the order-7 cumulative-quadrature stencil
  (`interpolated_integral_order_7`/`cumulative_integral`, [`_seobv5_cumulative_integral7`](@ref)).
- `pyseobnr/eob/dynamics/postadiabatic_C.pyx:296-322` -- `j0_eqn`/`compute_adiabatic_solution`
  ([`_seobv5_adiabatic_j0`](@ref)), Eq. 2.5.
- `pyseobnr/eob/dynamics/postadiabatic_C_fast.pyx:1-268` -- `pr_eqn`/`compute_pr` (Eq. 2.6,
  [`_seobv5_pr_eqn`](@ref)), `pphi_eqn`/`compute_pphi` (Eq. 2.7, [`_seobv5_pphi_eqn`](@ref)),
  `compute_postadiabatic_solution` (the 8-pass driver, [`_seobv5_postadiabatic_solution!`](@ref)),
  `compute_postadiabatic_dynamics` (grid construction, handoff radius, validity check, quadrature,
  [`seobnrv5alignedspin_postadiabatic_dynamics`](@ref)), `compute_combined_dynamics` (the PA/ODE
  splice and explicit fallback, [`seobnrv5alignedspin_combined_dynamics`](@ref)).

The "analytic" variant's own grid spacing is
`dr0=0.2` (`postadiabatic_C_fast.pyx`'s `compute_postadiabatic_dynamics`, `cdef double dr0 = 0.2`)
-- **not** `0.3`, the value in the *other* ("numeric",
`postadiabatic_C.pyx`) implementation; the two variants use
different default grid spacings. Also, `postadiabatic_C_fast.pyx`'s `compute_postadiabatic_dynamics`
bumps `r_size` up to a minimum of `10` (not just checking `r_size<=4`) when `4 < r_size < 10`,
necessary for correctness here since the order-9 finite-difference stencil needs >= 9 points.

# Runtime-sized-array architecture -- CPU-only, plain `Vector{Float64}`, non-Enzyme

Every PA quantity (`r`, `pr`, `pphi`, `t`, `phi`, and intermediate derivative arrays) is a plain
`Vector{Float64}` whose length (`r_size`) is a runtime value (a function of `r0`, `chi_eff`, `nu`
via the IC solve and the HBR2016/Kerr-ISCO handoff radius) -- genuinely new architectural territory
for this package, since every other model here keeps
its dynamical state as a compile-time-fixed-size `StaticArrays.FieldVector`. This implementation is
**deliberately plain Julia `Vector`-based, `Float64`-only, and never passed through
`Enzyme.gradient`/`KernelAbstractions.jl` `@kernel`**. Ordinary Julia `Vector` allocation, indexing, and the finite-difference/cumulative-
quadrature stencils below all work correctly and are straightforward to write (no genuine blocker
found) -- but whether this composes with `Enzyme.jl`/`KernelAbstractions.jl` remains **open, not
resolved here**: a runtime-sized-array reverse-mode gradient (through a variable-length stencil, or
through the fixed-iteration Newton solve inside [`_seobv5_adiabatic_j0`](@ref)) was not attempted.
"""
struct SEOBNRv5AlignedSpinPA end

# =============================================================================================
# HBR2016 M3J4 final-spin fit + Kerr-ISCO handoff-radius formula. Verbatim from
# pyseobnr/eob/utils/nr_utils.py (calc_isco_radius, _bbh_HBR2016_ell's "M3J4" branch,
# bbh_final_spin_non_precessing_HBR2016) and pyseobnr/eob/dynamics/postadiabatic_C.pyx:265-283
# (Kerr_ISCO). Both m1/m2 arguments below are the same X1/X2 mass-fraction convention (summing to
# 1) every other function in this file uses -- confirmed directly: postadiabatic_C_fast.pyx's own
# `compute_postadiabatic_dynamics` calls `Kerr_ISCO(chi_1, chi_2, m_1, m_2)` with the same `m_1,
# m_2` it elsewhere uses as `nu = m_1*m_2/(m_1+m_2)**2` and `chi_eff = m_1*chi_1+m_2*chi_2` (i.e.
# `m_1+m_2` need not literally be 1 in general, but always is in this port's own convention, exactly
# as for `_seobv5_ham_xi`/`_seobv5_gradient`/`_seobv5_auxderivs`). ===============================
# =============================================================================================

"""
    _seobv5_calc_isco_radius(a) -> Float64

Kerr ISCO radius as a function of the (dimensionless) Kerr spin parameter `a`, Eq. 2.5/2.8 of Ori &
Thorne, Phys. Rev. D 62, 124022 (2000) -- verbatim port of `nr_utils.py`'s `calc_isco_radius`
(lines 15-33), used both by [`_seobv5_final_spin_hbr2016`](@ref) (evaluated at the *effective*
spin `aeff`, an intermediate quantity in the fit, not a real horizon) and by
[`_seobv5_kerr_isco`](@ref) (evaluated at the real remnant spin `af`).
"""
@inline function _seobv5_calc_isco_radius(a)
    a = min(a, 1.0)
    z1 = 1.0 + (1.0 - a^2)^(1.0 / 3.0) * ((1.0 + a)^(1.0 / 3.0) + (1.0 - a)^(1.0 / 3.0))
    z2 = sqrt(3.0 * a^2 + z1^2)
    return 3.0 + z2 - sign(a) * sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
end

"""
    _seobv5_final_spin_hbr2016(X1, X2, chi1, chi2) -> Float64

The (signed) dimensionless remnant spin from aligned-spin BBH merger, Hofmann, Barausse & Rezzolla,
ApJL 825, L19 (2016) ("HBR2016"), the 20-coefficient `"M3J4"` fit version specifically (the version
`pyseobnr`'s own `Kerr_ISCO` hardcodes, `postadiabatic_C.pyx:275`). Verbatim port of
`nr_utils.py`'s `bbh_final_spin_non_precessing_HBR2016` (lines 174-202) composed with
`_bbh_HBR2016_ell`'s `"M3J4"` branch (lines 51-172, `nJ=4`/20-coefficient case only -- the `"M1J2"`/
`"M3J3"` branches, needed only for the other two fit versions `pyseobnr` itself never selects here,
are not ported). `X1`/`X2` are mass fractions (`X1+X2=1`, `X1>=X2`, this file's usual convention);
`q = X2/X1 = m2/m1` matches `nr_utils.py`'s own `q = m2/m1` exactly (`_bbh_HBR2016_setup`, line 51).
One-time-per-waveform, plain `Float64` (mirrors `_seobv5_a6_NS`/`_seobv5_dSO`).

**Validation**: a closed-form-vs-closed-form check against real `pyseobnr`
(`bbh_final_spin_non_precessing_HBR2016(0.6, 0.4, 0.3, -0.2, version="M3J4")`) at `X1=0.6,X2=0.4,chi1=0.3,chi2=-0.2` (this file's usual `q=1.5` validation
configuration): real `pyseobnr` gives `af=0.717896193825361`; this port gives the same value to
every printed digit (`0.717896193825361`) -- exact agreement, not merely close, confirming the
transcription (checked directly, not assumed -- see `test/seobnrv5alignedspin_pa_tests.jl`'s own
`"HBR2016 final spin"` testset for the locked-in regression).
"""
@inline function _seobv5_final_spin_hbr2016(X1, X2, chi1, chi2)
    q = X2 / X1
    atot = (chi1 + q * q * chi2) / ((1.0 + q) * (1.0 + q))
    return atot + _seobv5_hbr2016_ell(X1, X2, chi1, chi2) / (1.0 / q + 2.0 + q)
end

# `_bbh_HBR2016_ell` (`nr_utils.py`, "M3J4"): the orbital angular momentum term of the HBR2016 fits, of the spin
# components along L (shared by the aligned and the precessing final-spin fits).
@inline function _seobv5_hbr2016_ell(X1, X2, chi1, chi2)
    q = X2 / X1

    k00 = -5.97723
    k01 = 3.39221
    k02 = 4.48865
    k03 = -5.77101
    k04 = -13.0459
    k10 = 35.1278
    k11 = -72.9336
    k12 = -86.0036
    k13 = 93.7371
    k14 = 200.975
    k20 = -146.822
    k21 = 387.184
    k22 = 447.009
    k23 = -467.383
    k24 = -884.339
    k30 = 223.911
    k31 = -648.502
    k32 = -697.177
    k33 = 753.738
    k34 = 1166.89
    xi_hbr = 0.474046

    eta = X1 * X2 / (X1 + X2)^2

    atot = (chi1 + q * q * chi2) / ((1.0 + q) * (1.0 + q))
    aeff = atot + xi_hbr * eta * (chi1 + chi2)

    r_isco = _seobv5_calc_isco_radius(aeff)
    e_isco = sqrt(1.0 - 2.0 / 3.0 / r_isco)
    l_isco = 2.0 / (3.0 * sqrt(3.0)) * (1.0 + 2.0 * sqrt(3.0 * r_isco - 2.0))

    aeff2 = aeff * aeff
    aeff3 = aeff2 * aeff
    eta2 = eta * eta
    eta3 = eta2 * eta

    ksum = k00 + k01 * aeff + k02 * aeff2 + eta * (k10 + k11 * aeff + k12 * aeff2)
    ksum = ksum + (k03 + eta * k13) * aeff3 + eta2 * (k20 + k21 * aeff + k22 * aeff2 + k23 * aeff3) +
           eta3 * (k30 + k31 * aeff + k32 * aeff2 + k33 * aeff3)
    ksum = ksum + (k04 + eta * k14 + eta2 * k24 + eta3 * k34) * aeff3 * aeff

    return abs((l_isco - 2.0 * atot * (e_isco - 1.0)) + eta * ksum)
end

"""
    _seobv5_kerr_isco(chi1, chi2, X1, X2) -> (r_isco, L_isco)

The Kerr ISCO radius and angular momentum for the black hole remnant of this binary's merger, per
its HBR2016-predicted spin ([`_seobv5_final_spin_hbr2016`](@ref)). Verbatim port of
`postadiabatic_C.pyx`'s `Kerr_ISCO` (lines 265-283); only `r_isco` is actually used downstream (the
PA handoff radius, [`seobnrv5alignedspin_postadiabatic_dynamics`](@ref)) -- confirmed directly that
`compute_postadiabatic_dynamics`/`compute_combined_dynamics` only ever read `Kerr_ISCO(...)[0]`,
never `[1]`, so `L_isco` here is transcribed for citation completeness but is dead code exactly as
it is in `pyseobnr` itself, and was not separately validated against real `pyseobnr` output (unlike
`r_isco`, see [`_seobv5_final_spin_hbr2016`](@ref)'s own validation note).

**Validation**: real `pyseobnr` (`Kerr_ISCO(0.3, -0.2, 0.6, 0.4)`) gives `r_isco=3.3105655850677906`,
`L_isco=2.0410944472880925` at this file's usual `q=1.5` configuration; this port gives the same
values to every printed digit -- exact agreement.
"""
@inline function _seobv5_kerr_isco(chi1, chi2, X1, X2)
    a = _seobv5_final_spin_hbr2016(X1, X2, chi1, chi2)
    z1 = 1.0 + (1.0 - a^2)^(1.0 / 3.0) * ((1.0 + a)^(1.0 / 3.0) + (1.0 - a)^(1.0 / 3.0))
    z2 = sqrt(3.0 * a^2 + z1^2)
    r_isco = 3.0 + z2 - sign(a) * sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))
    L_isco = 2.0 / (3.0 * sqrt(3.0)) * (1.0 + 2.0 * sqrt(3.0 * r_isco) - 2.0)
    return (r_isco, L_isco)
end

# =============================================================================================
# Runtime-sized-array machinery -- an 8th-order finite-difference derivative stencil
# and a 7th-order cumulative-quadrature integral, both over a `Vector{Float64}` whose length is a
# runtime value (the PA radial grid size), not known at compile time. CPU-only, plain Julia
# `Vector`, never passed through Enzyme/KernelAbstractions -- see this file's own module docstring
# for the full architectural discussion. Verbatim port of `postadiabatic_C.pyx`'s
# `fin_diff_coeffs_order_9`/`fin_diff_derivative` (lines 32-260) and
# `interpolated_integral_order_7`/`cumulative_integral` (lines 159-260, 861-897, `order=7` branch
# only -- the `order=3`/`order=5` branches, never used by `compute_postadiabatic_dynamics` (which
# always calls `cumulative_integral(r, dt_dr)`/`cumulative_integral(r, dphi_dr)` at the function's
# own `order=7` default), are not ported). Both were re-derived from `numpy.convolve`'s own
# "valid"-mode definition by hand: `convolve(reverse(row), slice, "valid")` with two equal-length
# arrays reduces to a plain `dot(row, slice)`, which is what the explicit loops below implement. ===
# =============================================================================================

const _SEOBV5_FINDIFF9_COEFFS = (
    (-761.0/280.0, 8.0, -14.0, 56.0/3.0, -35.0/2.0, 56.0/5.0, -14.0/3.0, 8.0/7.0, -1.0/8.0),
    (-1.0/8.0, -223.0/140.0, 7.0/2.0, -7.0/2.0, 35.0/12.0, -7.0/4.0, 7.0/10.0, -1.0/6.0, 1.0/56.0),
    (1.0/56.0, -2.0/7.0, -19.0/20.0, 2.0, -5.0/4.0, 2.0/3.0, -1.0/4.0, 2.0/35.0, -1.0/168.0),
    (-1.0/168.0, 1.0/14.0, -1.0/2.0, -9.0/20.0, 5.0/4.0, -1.0/2.0, 1.0/6.0, -1.0/28.0, 1.0/280.0),
    (1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 0.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0),
    (-1.0/280.0, 1.0/28.0, -1.0/6.0, 1.0/2.0, -5.0/4.0, 9.0/20.0, 1.0/2.0, -1.0/14.0, 1.0/168.0),
    (1.0/168.0, -2.0/35.0, 1.0/4.0, -2.0/3.0, 5.0/4.0, -2.0, 19.0/20.0, 2.0/7.0, -1.0/56.0),
    (-1.0/56.0, 1.0/6.0, -7.0/10.0, 7.0/4.0, -35.0/12.0, 7.0/2.0, -7.0/2.0, 223.0/140.0, 1.0/8.0),
    (1.0/8.0, -8.0/7.0, 14.0/3.0, -56.0/5.0, 35.0/2.0, -56.0/3.0, 14.0, -8.0, 761.0/280.0),
)

const _SEOBV5_CUMINT7_COEFFS = (
    (5257.0/17280.0, 139849.0/120960.0, -(4511.0/4480.0), 123133.0/120960.0, -(88547.0/120960.0), 1537.0/4480.0, -(11351.0/120960.0), 275.0/24192.0),
    (-(275.0/24192.0), 5311.0/13440.0, 11261.0/13440.0, -(44797.0/120960.0), 2987.0/13440.0, -(1283.0/13440.0), 2999.0/120960.0, -(13.0/4480.0)),
    (13.0/4480.0, -(4183.0/120960.0), 6403.0/13440.0, 9077.0/13440.0, -(20227.0/120960.0), 803.0/13440.0, -(191.0/13440.0), 191.0/120960.0),
    (-(191.0/120960.0), 1879.0/120960.0, -(353.0/4480.0), 68323.0/120960.0, 68323.0/120960.0, -(353.0/4480.0), 1879.0/120960.0, -(191.0/120960.0)),
    (191.0/120960.0, -(191.0/13440.0), 803.0/13440.0, -(20227.0/120960.0), 9077.0/13440.0, 6403.0/13440.0, -(4183.0/120960.0), 13.0/4480.0),
    (-(13.0/4480.0), 2999.0/120960.0, -(1283.0/13440.0), 2987.0/13440.0, -(44797.0/120960.0), 11261.0/13440.0, 5311.0/13440.0, -(275.0/24192.0)),
    (275.0/24192.0, -(11351.0/120960.0), 1537.0/4480.0, -(88547.0/120960.0), 123133.0/120960.0, -(4511.0/4480.0), 139849.0/120960.0, 5257.0/17280.0),
)

"""
    _seobv5_findiff_derivative9(x::Vector{Float64}, y::Vector{Float64}) -> Vector{Float64}

8th-order finite-difference derivative `dy/dx` on an equally-spaced (`length(x) >= 9`) grid, with
correctly-handled asymmetric endpoint stencils -- verbatim port of `postadiabatic_C.pyx`'s
`fin_diff_derivative` (lines 234-260). Plain `Vector{Float64}`, allocates one output vector (not
allocation-free -- unlike the rest of this package's hot-path RHS code, this runs once per PA order
per waveform, not once per RK4 substep, so this is not this file's hot path). Divides by
`abs(x[2]-x[1])` always (`fabs(h)` in the source), **regardless of whether `x` is increasing or
decreasing** -- `SEOBNRv5AlignedSpinPA`'s own radial grid runs from `r0` down to `r_final`
(decreasing), so callers apply an explicit sign flip afterward exactly where `pyseobnr` itself does
(`compute_pr`'s `dpphi_dr = -fin_diff_derivative(r, pphi)`, `compute_pphi`'s `dpr_dr =
-fin_diff_derivative(r, pr)`) -- see [`_seobv5_postadiabatic_solution!`](@ref).
"""
function _seobv5_findiff_derivative9(x::Vector{Float64}, y::Vector{Float64})
    N = length(x)
    N >= 9 || throw(ArgumentError("_seobv5_findiff_derivative9 needs length(x) >= 9, got $N"))
    C = _SEOBV5_FINDIFF9_COEFFS
    h = abs(x[2] - x[1])
    dydx = Vector{Float64}(undef, N)
    @inbounds for i in 1:4
        s = 0.0
        for k in 1:9
            s += C[i][k] * y[k]
        end
        dydx[i] = s
    end
    @inbounds for i in 5:(N - 4)
        s = 0.0
        row = C[5]
        for k in 1:9
            s += row[k] * y[i - 5 + k]
        end
        dydx[i] = s
    end
    @inbounds for i in (N - 3):N
        row = C[i - N + 9]
        s = 0.0
        for k in 1:9
            s += row[k] * y[N - 9 + k]
        end
        dydx[i] = s
    end
    dydx ./= h
    return dydx
end

"""
    _seobv5_cumulative_integral7(x::Vector{Float64}, y::Vector{Float64}) -> Vector{Float64}

Cumulative integral of `y` w.r.t. `x` (`length(x) >= 8`, equally spaced), 7th-order accurate --
verbatim port of `postadiabatic_C.pyx`'s `cumulative_integral` (lines 861-897, `order=7` branch,
the only order `compute_postadiabatic_dynamics` itself ever calls with). `integral[1] == 0`
(matches the source's own `integral[0]=0`); `integral[i]` is the definite integral from `x[1]` to
`x[i]`. **Uses the true signed step `x[2]-x[1]`** (not `abs`, unlike
[`_seobv5_findiff_derivative9`](@ref)) -- correct for `SEOBNRv5AlignedSpinPA`'s own decreasing-`r`
grid, since a negative step here correctly makes `t(r)`/`phi(r)` increase as `r` decreases, exactly
matching `pyseobnr`'s own convention (confirmed directly: `cumulative_integral`'s source computes
`h = x[1]-x[0]` with no `fabs`, unlike `fin_diff_derivative`'s explicit `fabs(x[1]-x[0])`).
"""
function _seobv5_cumulative_integral7(x::Vector{Float64}, y::Vector{Float64})
    N = length(x)
    N >= 8 || throw(ArgumentError("_seobv5_cumulative_integral7 needs length(x) >= 8, got $N"))
    ROW = _SEOBV5_CUMINT7_COEFFS
    h = x[2] - x[1]
    integral = zeros(Float64, N)
    @inbounds for (jrow, idx) in ((1, 2), (2, 3), (3, 4))
        s = 0.0
        row = ROW[jrow]
        for k in 1:8
            s += row[k] * y[k]
        end
        integral[idx] = s
    end
    @inbounds for i in 5:(N - 3)
        s = 0.0
        row = ROW[4]
        for k in 1:8
            s += row[k] * y[i - 5 + k]
        end
        integral[i] = s
    end
    @inbounds for (jrow, idx) in ((5, N - 2), (6, N - 1), (7, N))
        s = 0.0
        row = ROW[jrow]
        for k in 1:8
            s += row[k] * y[N - 8 + k]
        end
        integral[idx] = s
    end
    integral .*= h
    return cumsum(integral)
end

# =============================================================================================
# The order-by-order PA driver -- the adiabatic j0(r) order-0 solve (Eq. 2.5, a
# fixed-iteration 1D Newton solve mirroring _seobv5_circular_ic's own precedent, replacing
# pyseobnr's own scipy.optimize.root call) and the closed-form pr_eqn/pphi_eqn (Eq. 2.6/2.7,
# postadiabatic_C_fast.pyx's "analytic" variant), plus the fixed 8-pass alternating sweep. All
# plain Float64 (mirrors _seobv5_circular_ic/_seobv5_dissipative_ic's own precedent), operating on
# plain Vector{Float64}s of runtime-determined length. ====================
# =============================================================================================

"""
    _seobv5_adiabatic_j0(r, hp) -> Vector{Float64}

Eq. 2.5 of arXiv:2105.06983 (`j0_eqn`/`compute_adiabatic_solution`, `postadiabatic_C.pyx:296-322`):
at each grid point `r[i]`, the adiabatic (`epsilon=0`) angular momentum `j0(r)` solving
`dH/dr(r,prst=0,pphi)=0` for `pphi` -- the circular-orbit angular momentum at that `r`, treating
`r` as a fixed label rather than a dynamical variable. `pyseobnr` itself uses `scipy.optimize.root`
(adaptive tolerance); this port uses a **fixed-iteration (8-iteration) 1D Newton solve** instead,
mirroring [`_seobv5_circular_ic`](@ref)'s own already-established fixed-iteration-Newton precedent
in this file (not `scipy`'s adaptive approach).
Seeded from `Newtonian_j0(r) = sqrt(r)` (`postadiabatic_C.pyx:284-288`), exactly as `pyseobnr` seeds
its own root-find. The Newton derivative (`d(dH/dr)/dpphi`) is exact ([`_seobv5_sym_j0`](@ref)).
"""
function _seobv5_adiabatic_j0(r::Vector{Float64}, hp)
    N = length(r)
    pphi = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        ri = r[i]
        p = sqrt(ri)
        for _ in 1:8     # exact derivative: converged to round-off after 4 (20 with the finite-difference one)
            dHdr, deriv = _seobv5_sym_j0(ri, p, hp)
            p -= dHdr / deriv
        end
        pphi[i] = p
    end
    return pphi
end

"""
    _seobv5_pr_eqn(pr_trial, r, pphi, dpphi_dr, hp, fc) -> Float64

Odd-PA-order closed-form correction to `pr*` at one grid point (Eq. 2.6 of arXiv:2105.06983,
`pr_eqn`, `postadiabatic_C_fast.pyx:33-78`) -- a single direct algebraic evaluation (no
root-finding, no iteration internal to this function), using [`_seobv5_auxderivs`](@ref)'s
`dQ/dprst` and this file's own already-ported [`_seobv5_rr_force`](@ref) for `Fphi`. `pr_trial` is
the current (previous-order) `pr*` value, used only to evaluate the (weakly `pr*`-dependent) RHS
potentials at this grid point -- the returned value is the new, corrected `pr*`, not a residual.
"""
@inline function _seobv5_pr_eqn(pr_trial::Float64, r::Float64, pphi::Float64, dpphi_dr::Float64, hp, fc)
    nu = hp.nu
    H, xi, A, Bnp, Heven, Heff, dQdprst, omega, omega_circ = _seobv5_sym_pr_terms(r, pr_trial, pphi, hp)
    _, Fphi = _seobv5_rr_force(pr_trial, pphi, omega, omega_circ, Heff, H, nu, fc)
    return xi / (2.0 * (1.0 + Bnp)) * (Fphi / dpphi_dr * 2.0 * nu * H * Heven / A - xi * dQdprst)
end

"""
    _seobv5_pphi_eqn(pphi_trial, r, pr, dpr_dr, hp, fc) -> Float64

Even-PA-order closed-form correction to `pphi` at one grid point (Eq. 2.7 of arXiv:2105.06983,
`pphi_eqn`, `postadiabatic_C_fast.pyx:126-231`) -- reduces to a **quadratic equation**
`C_2*pphi^2+C_1*pphi+C_0=0` in `pphi`, solved via the quadratic formula (the `-` root, matching
`pyseobnr`'s own choice `(-C_1-sqrt(D))/(2*C_2)`) -- no root-finding, a single closed-form
evaluation. `pphi_trial` (the previous-order value) enters the coefficients `C_1`/`C_0` themselves
(via `dHodddr/pphi_trial` and `pr/pphi_trial*flux[1]`, confirmed directly from the source -- **not**
a transcription error: the quadratic model is built by linearizing around the current trial value,
so this is a single Newton-like correction step per PA order, not an exact closed-form solve of the
true nonlinear equation** -- exactly why the outer 8-pass loop
([`_seobv5_postadiabatic_solution!`](@ref)) applies 4 such even-order corrections total, not 1).
If the discriminant `D` is negative but `|D|<1e-10`, it is clamped to `0` (matches `pyseobnr`'s own
`pr_eqn`'s numerically-motivated tolerance, `postadiabatic_C_fast.pyx:222-226`); a more negative `D`
raises an error (`pyseobnr` itself does `raise ValueError` there unconditionally).
"""
@inline function _seobv5_pphi_eqn(pphi_trial::Float64, r::Float64, pr::Float64, dpr_dr::Float64, hp, fc)
    C_2, C_1, D = _seobv5_pphi_quadratic(pphi_trial, r, pr, dpr_dr, hp, fc)
    if D < 0.0
        if abs(D) < 1e-10
            D = 0.0
        else
            throw(DomainError(D, "_seobv5_pphi_eqn: negative discriminant beyond numerical tolerance"))
        end
    end
    return (-C_1 - sqrt(D)) / (2.0 * C_2)
end

"""
    _seobv5_pphi_quadratic(pphi_trial, r, pr, dpr_dr, hp, fc) -> (C_2, C_1, D)

The quadratic `C_2 pphi^2 + C_1 pphi + C_0 = 0` of [`_seobv5_pphi_eqn`](@ref) (`pphi_eqn`,
`postadiabatic_C_fast.pyx:126-231`): its leading coefficients and discriminant `D = C_1^2 - 4 C_2 C_0`.
"""
@inline function _seobv5_pphi_quadratic(pphi_trial::Float64, r::Float64, pr::Float64, dpr_dr::Float64, hp, fc)
    nu = hp.nu
    H, xi, A, Bnp, Bnpa, Qq, Heven, Heff, dAdr, dBnpdr, dBnpadr, dxidr, dQdr, dQdprst, dHodddr, omega, omega_circ =
        _seobv5_sym_pphi_terms(r, pr, pphi_trial, hp)

    drdt = A / (2.0 * nu * H * Heven) * (2.0 * pr / xi * (1.0 + Bnp) + xi * dQdprst)
    _, Fphi = _seobv5_rr_force(pr, pphi_trial, omega, omega_circ, Heff, H, nu, fc)

    pr2 = pr * pr
    xi2 = xi * xi
    xi3 = xi2 * xi
    r2 = r * r
    r3 = r2 * r
    ap = hp.X1 * hp.chi1 + hp.X2 * hp.chi2
    ap2 = ap * ap

    C_2 = dAdr * (1.0 / r2 + ap2 / r2 * Bnpa) + A * (-2.0 / r3 - 2.0 * ap2 / r3 * Bnpa + ap2 / r2 * dBnpadr)
    C_1 = dHodddr / pphi_trial * 2.0 * Heven
    C_0_pt1 = dAdr * (1.0 + pr2 / xi2 * Bnp + pr2 / xi2 + Qq) +
              A * (-2.0 * pr2 / xi3 * Bnp * dxidr + pr2 / xi2 * dBnpdr - 2.0 * pr2 / xi3 * dxidr + dQdr)
    tmp = 2.0 * Heven * nu * H / xi
    C_0_pt2 = dpr_dr * drdt * tmp - pr / pphi_trial * tmp * Fphi
    C_0 = C_0_pt1 + C_0_pt2

    return C_2, C_1, C_1 * C_1 - 4.0 * C_2 * C_0
end

"""
    _seobv5_postadiabatic_solution!(r, pphi, hp, fc; order=8) -> (pr, pphi)

The fixed 8-pass (default) alternating `pr*`/`pphi` correction sweep -- verbatim port of
`compute_postadiabatic_solution` (`postadiabatic_C_fast.pyx:277-322`): odd orders (`n=1,3,5,7`)
correct `pr*(r)` via [`_seobv5_pr_eqn`](@ref) using `dpphi/dr = -`[`_seobv5_findiff_derivative9`](@ref)`(r,pphi)`
(the explicit sign flip `pyseobnr`'s own `compute_pr` applies, see
[`_seobv5_findiff_derivative9`](@ref)'s docstring); even orders (`n=2,4,6,8`) correct `pphi(r)` via
[`_seobv5_pphi_eqn`](@ref) using `dpr/dr` the same way. `pr` starts at all-zeros
(`postadiabatic_C.pyx`'s own `pr = np.zeros(r.shape[0])`); `pphi` starts at the adiabatic solution
([`_seobv5_adiabatic_j0`](@ref)'s output). Returns new arrays (not literally mutating in place,
despite the `!` -- named to signal "produces corrected copies of its inputs", matching this
function's own conceptual role in the driver, not a strict Julia in-place-mutation convention).
"""
function _seobv5_postadiabatic_solution!(r::Vector{Float64}, pphi_in::Vector{Float64}, hp, fc; order::Int=8)
    N = length(r)
    pr = zeros(Float64, N)
    pphi = copy(pphi_in)
    for n in 1:order
        if isodd(n)
            dpphi_dr = -_seobv5_findiff_derivative9(r, pphi)
            @inbounds for i in 1:N
                pr[i] = _seobv5_pr_eqn(pr[i], r[i], pphi[i], dpphi_dr[i], hp, fc)
            end
        else
            dpr_dr = -_seobv5_findiff_derivative9(r, pr)
            @inbounds for i in 1:N
                pphi[i] = _seobv5_pphi_eqn(pphi[i], r[i], pr[i], dpr_dr[i], hp, fc)
            end
        end
    end
    return pr, pphi
end

"""
    seobnrv5alignedspin_postadiabatic_dynamics(coeffs; order=8, dr0=0.2) -> NamedTuple

The full PA-only dynamics table (the driver, plus the grid construction/handoff-radius/
validity-check/quadrature machinery), verbatim port of
`compute_postadiabatic_dynamics` (`postadiabatic_C_fast.pyx:325-401`). `coeffs` is this file's own
`_seobv5_coeffs(...)` output (`SEOBNRv5AlignedSpin.jl`) -- reuses its `r0`/`pphi0` (from the
already-ported `IC_cons` two-stage solve) as the PA grid's starting radius, exactly as
`pyseobnr`'s own driver reuses `computeIC`'s `r0` (PA needs no new initial condition). Returns a `NamedTuple`:
- `success::Bool` -- `false` exactly when `pyseobnr`'s own `compute_postadiabatic_dynamics` would
  raise (`r_size<=4` or `r0<=11.5`, `postadiabatic_C_fast.pyx:398-401`) -- callers must fall back to
  full-ODE-only integration in that case (see [`seobnrv5alignedspin_combined_dynamics`](@ref)).
- If `success`: `t, r, phi, pr, pphi` -- five `Vector{Float64}`s of the same runtime-determined
  length (`r` monotonically **decreasing** from `r0` to `r_final`, `t`/`phi` monotonically
  increasing, matching `pyseobnr`'s own sign convention -- see
  [`_seobv5_cumulative_integral7`](@ref)'s docstring).

**Grid spacing `dr0=0.2`** (not `0.3`) -- the "analytic" variant's own default, confirmed directly
from `postadiabatic_C_fast.pyx` itself, not assumed from the "numeric" variant's different default
(see this file's own module docstring for the full discrepancy note).
"""
function seobnrv5alignedspin_postadiabatic_dynamics(coeffs; order::Int=8, dr0::Float64=0.2)
    hp = _seobv5_f64(coeffs.hp)
    nu, X1, X2, chi1, chi2 = hp.nu, hp.X1, hp.X2, hp.chi1, hp.chi2
    fc = _seobv5_f64(coeffs.fc)
    r0 = Float64(coeffs.r0)

    chi_eff = X1 * chi1 + X2 * chi2
    r_final_prefactor = 2.7 + chi_eff * (1.0 - 4.0 * nu)
    r_isco, _ = _seobv5_kerr_isco(chi1, chi2, X1, X2)
    r_final = max(10.0, r_final_prefactor * r_isco)

    r_size = ceil(Int, (r0 - r_final) / dr0)
    if r_size <= 4 || r0 <= 11.5
        return (success = false, r_final = r_final, r_size = r_size)
    elseif r_size < 10
        r_size = 10
    end

    r = collect(range(r0, r_final; length = r_size))
    pphi = _seobv5_adiabatic_j0(r, hp)
    pr, pphi = _seobv5_postadiabatic_solution!(r, pphi, hp, fc; order = order)

    dt_dr = Vector{Float64}(undef, r_size)
    dphi_dr = Vector{Float64}(undef, r_size)
    @inbounds for i in 1:r_size
        xi, dHdpr, dHdpphi = _seobv5_sym_tphi(r[i], pr[i], pphi[i], hp)
        dH_dpr_times_xi = dHdpr * xi
        dt_dr[i] = 1.0 / dH_dpr_times_xi
        dphi_dr[i] = dHdpphi / dH_dpr_times_xi
    end

    t = _seobv5_cumulative_integral7(r, dt_dr)
    phi = _seobv5_cumulative_integral7(r, dphi_dr)

    return (success = true, t = t, r = r, phi = phi, pr = pr, pphi = pphi, r_final = r_final)
end

# =============================================================================================
# Handoff wiring to this file's existing seobnrv5alignedspin_rk4_integrate, plus the
# explicit fallback-to-full-ODE-only path pyseobnr itself uses when PA's own validity precondition
# fails. Verbatim structural port of compute_combined_dynamics (postadiabatic_C_fast.pyx:404-460),
# with this file's own fixed-step RK4 standing in for pyseobnr's adaptive GSL RK8(9) (an existing,
# already-documented fidelity difference -- see SEOBNRv5AlignedSpin.jl's docstring's "handoff/
# matching point" citation trail). =========================
# =============================================================================================

"""
    seobnrv5alignedspin_combined_dynamics(coeffs, dt_hat, nsteps_post; order=8, dr0=0.2) -> NamedTuple

Wires [`seobnrv5alignedspin_postadiabatic_dynamics`](@ref) to this file's existing
`seobnrv5alignedspin_rk4_integrate`: if PA succeeds, the RK4 tail starts from the **PA table's last
state** (`r_final, phi_final, prst_final, pphi_final`), not from the two-stage `IC_cons`/`IC_diss`
solve -- exactly mirroring `compute_combined_dynamics`'s own `ode_y_init =
postadiabatic_dynamics[-1, 1:]` handoff. If PA fails its own validity precondition (`r_size<=4` or
`r0<=11.5`, `pyseobnr`'s own explicit, checked fallback -- not a silent degradation), the RK4 tail
instead starts from the plain two-stage IC, i.e. **falls back to exactly this file's pre-existing,
already-validated pure-full-ODE behavior** (`_seobv5_state_scalar`'s own contract, unchanged).

`dt_hat`/`nsteps_post` are in this file's own reduced time units (matching
`seobnrv5alignedspin_rk4_integrate`'s own signature) -- **not** physical seconds; this is a
lower-level function than `Phi`/`Ampl`, deliberately not wired into their per-scalar-`t`
random-access contract (that would need a monotonic-interpolation/lookup primitive against the PA
table).

Returns a `NamedTuple` with `pa_success::Bool`, the PA table `pa` (from
[`seobnrv5alignedspin_postadiabatic_dynamics`](@ref)), the handoff `SEOBNRv5AlignedSpinState`
(`handoff_state`) the RK4 tail was started from, the PA-relative-time offset at handoff
(`t_handoff`, `0` when `pa_success` is `false`), and the RK4 tail's final state (`tail_state`).
"""
function seobnrv5alignedspin_combined_dynamics(coeffs, dt_hat, nsteps_post::Integer; order::Int=8, dr0::Float64=0.2)
    T = coeffs.T
    pa = seobnrv5alignedspin_postadiabatic_dynamics(coeffs; order = order, dr0 = dr0)
    if pa.success
        handoff_state = SEOBNRv5AlignedSpinState(T(pa.r[end]), T(pa.phi[end]), T(pa.pr[end]), T(pa.pphi[end]))
        t_handoff = pa.t[end]
        tail_state = seobnrv5alignedspin_rk4_integrate(handoff_state, T(dt_hat), nsteps_post, coeffs)
        return (pa_success = true, pa = pa, handoff_state = handoff_state, t_handoff = t_handoff, tail_state = tail_state)
    else
        state0 = SEOBNRv5AlignedSpinState(coeffs.r0, zero(T), coeffs.pr0, coeffs.pphi0)
        tail_state = seobnrv5alignedspin_rk4_integrate(state0, T(dt_hat), nsteps_post, coeffs)
        return (pa_success = false, pa = pa, handoff_state = state0, t_handoff = zero(T), tail_state = tail_state)
    end
end
