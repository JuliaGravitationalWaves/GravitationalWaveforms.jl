using StaticArrays: FieldVector, SVector
using LinearAlgebra: cross

export PrecessingSpinPNHam, PrecessingSpinPNHamState, validate_precessingspinpnham_params,
    precessingspinpnham_rk4_integrate, evolve

"""
    PrecessingSpinPNHam

1.5PN precessing-spin post-Newtonian inspiral dynamics, generalizing `AlignedSpinPNHam` from a
planar `(r,phi,p_r,p_phi)` Hamiltonian system to genuine 3D spin precession. Unlike that model, this
model does **not** invent a Cartesian two-body Hamiltonian -- it directly ports LALSuite's own
**orbit-averaged, reduced-state** precessing-spin ODE system (`XLALSimInspiralSpinTaylorT4DerivativesAvg`/
`XLALSimInspiralSpinDerivativesAvg`, `lalsimulation/lib/LALSimInspiralSpinTaylor.c`), at the
`spinO = LAL_SIM_INSPIRAL_SPIN_ORDER_15PN` (1.5PN) truncation. The reduced `(phase, v, LNhat, S1, S2)` state, not a literal position/momentum
3-vector pair, is what LALSuite's own reference implementation actually uses and validates against.

# State vector

`PrecessingSpinPNHamState{T} <: FieldVector{11,T}`: `phase` (orbital phase), `v` (PN velocity
parameter, `v^3 = omega_hat = M*Omega_orb`), `LNhat` (3, unit orbital angular momentum direction),
`S1`, `S2` (3 each, reduced spins `S_i/M^2 = X_i^2*chi_i`, dimensionless). No orbital separation or
radial momentum anywhere -- confirmed directly against LALSuite's own state (same 11 components,
LALSuite's own 14-component state additionally carries `E1`, an in-plane reference/polarization-
frame vector that is dynamically decoupled from everything else and only needed for waveform
reconstruction; see [`evolve`](@ref)'s docstring for how this port reconstructs an equivalent E1
internally, only where actually needed, rather than carrying it in this exported "hot" state).

# Equations of motion -- ported directly from LALSuite

All at `spinO = LAL_SIM_INSPIRAL_SPIN_ORDER_15PN` (the leading, `v^5`-in-`dS_i`/`v^3`-relative-in-
`domega` truncation) and (for `domega`) `phaseO = 3` (non-spin domega/dt truncated at the *same*
order the leading spin-orbit correction first appears, so the whole model is 1.5PN-complete and
nothing in it silently outruns its own spin truncation -- a deliberate scope choice, see below).

## Spin precession (`XLALSimInspiralSpinDerivativesAvg`, `LALSimInspiralSpinTaylor.c` lines 483-785)

```
dS_i/dt_hat = c_i(X_i) * v^5 * (LNhat x S_i),   c_i(X_i) = 3/2 - X_i - X_i^2/2
```
`c_i(X_i) = XLALSimInspiralSpinDot_3PNCoeff(X_i)` (`LALSimInspiralPNCoefficients.c` line 2328,
comment cites "eq. (8) of gr-qc/0405090" = Buonanno, Chen, Pan & Vallisneri, Phys. Rev. D 70, 104003
(2004) -- confirmed against the arXiv record, NOT the PRD 67, 104025 (2003) paper that could be
mistaken for it). Ported from the `if
((params->spinO>=3)...)` branch, `LALSimInspiralSpinTaylor.c` line 546 (`S1dot3`/`S2dot3` assigned
at lines 437/957-ish -- `params->S1dot3 = XLALSimInspiralSpinDot_3PNCoeff(m1M)`, confirmed at line
437). This is `dS_i/dt_hat = Omega_i x S_i` with `Omega_i = c_i(X_i)*v^5*LNhat`, as expected.

Orbital angular momentum direction (same function, lines 758-776): the *un-normalized* rate
`d(L*LNhat)/dt_hat = -(dS1+dS2)` (conservative exchange, `L = eta/v` the leading-order magnitude),
divided by `L` to get a raw `dLNhat`, then projected into a pure rotation via
```
Om = LNhat x dLNhat_raw     # LALSuite line 767ish: "Om=LNhat x dLNhat"
dLNhat = Om x LNhat         # LALSuite line 771ish: "Take cross product of Om with LNhat"
```
-- a cross product with `LNhat` annihilates any component parallel to `LNhat`, so `|LNhat|=1` is
preserved by construction at the *continuous* level (`d|LNhat|^2/dt = 2*LNhat.dLNhat =
2*LNhat.(Om x LNhat) = 0` identically, `Om x LNhat` always being perpendicular to `LNhat`) -- not
by an explicit per-step renormalization, which LALSuite's own integrator does not perform either
(confirmed: no `normalize`/renormalization of the *state* `LNhat` anywhere in
`XLALSimInspiralSpinTaylorT4DerivativesAvg`/`XLALSimInspiralSpinDerivativesAvg`, only the
*derivative* `dLNhat` gets divided by `LNmag` as part of computing `Om`, which is a different
operation). This port mirrors that faithfully -- see `test/precessingspinpnham_tests.jl`'s "unit
LNhat drift" testset for a direct check of how much floating-point/finite-step drift this leaves
over the model's own valid integration range .

## Orbital frequency evolution (`XLALSimInspiralSpinTaylorT4DerivativesAvg`, lines 2540-2671)

```
domega/dt_hat = wdotnewt * v^11 * (wdotcoeff[0] + v*wdotcoeff[1] + v^2*wdotcoeff[2]
                                    + v^3*(wdotcoeff[3] + wspin3))
wdotnewt = 96*eta/5                                     [XLALSimInspiralTaylorT4wdot_0PNCoeff, line 276 use / definition at its own file]
wdotcoeff[0] = 1, wdotcoeff[1] = 0                       [XLALSimInspiralSpinTaylorT4Setup, lines 884/1301 (=1), 878/1295 (=0)]
wdotcoeff[2] = -(1/336)*(743+924*eta)                    [XLALSimInspiralTaylorT4wdot_2PNCoeff, LALSimInspiralPNCoefficients.c line 1827; assigned line 872]
wdotcoeff[3] = 4*pi                                      [XLALSimInspiralTaylorT4wdot_3PNCoeff, line 1834; assigned line 866]
wspin3 = wdot3S1O*(LNhat.S1) + wdot3S2O*(LNhat.S2)
wdot3S1O = XLALSimInspiralTaylorT4wdot_3PNSOCoeff(X1) = -19/6 - 25/(4*X1)   [line 1841; assigned line 957]
```
`dv/dt_hat = (domega/dt_hat) / (3*v^2)` (chain rule, `omega=v^3`); `dphase/dt_hat = omega` exactly
at this order (the `omegashift` correction, `LALSimInspiralSpinTaylor.c` line ~2663, is a genuine
3PN-order `dphi/dt != omega` effect per its own cited source, arXiv:1507.00406 -- correctly out of
scope at 1.5PN, noted here for any future extension).

# PNFlux_*SOCoeff vs TaylorT4wdot_*SOCoeff

It is not obvious whether `AlignedSpinPNHam`'s
`_asph_flux_coeffs`-based construction (which ports `XLALSimInspiralPNFlux_3PNSOCoeff(x) =
-3/2 - 5/(4x)`, and computes `domega/dt` via the *unexpanded* energy-balance ratio
`dp_phi/dt = -F(v)/(eta*Omega)`, `Omega = dH/dp_phi` from its own exact Hamiltonian) is consistent
with the `TaylorT4wdot_3PNSOCoeff(x) = -19/6 - 25/(4x)` family used directly above. **They are, both analytically and numerically:**

The standard adiabatic relation `dv/dt = -F(v)/(dE/dv)` (energy balance), Taylor-re-expanded and
truncated consistently in `v` (which is literally what "TaylorT4" *means* as a PN-approximant
family -- as opposed to "TaylorT1", the same ratio left *unexpanded*), gives, at the order of the
*first* nonzero spin-orbit correction (`v^3`, 1.5PN -- confirmed by direct series expansion: writing
`E(v) = -(eta/2)v^2(1+e2 v^2+e3 v^3+...)`, `F(v)=F_N v^10(1+f2 v^2+f3 v^3+...)`, `dv/dt` re-expanded
gives a `v^3`-relative coefficient of exactly `f3 - (5/2)*e3`, with **no** cross term against the
lower-order `e2`/`f2` -- those first enter at `v^4`/2PN, one order beyond this model's scope):
```
domega/dt's SO coefficient = f3SO(x) - (5/2)*e3SO(x),   f3SO = PNFlux_3PNSOCoeff, e3SO = PNEnergy_3PNSOCoeff
```
Evaluated symbolically: `f3SO(x) - 2.5*e3SO(x) = (-3/2-5/(4x)) - 2.5*(2/3+2/x) = -19/6 - 25/(4x)`
-- **exactly** `TaylorT4wdot_3PNSOCoeff(x)`, confirmed numerically to machine precision at
`x in (0.3,0.5,0.7,0.9,1.0)` (`test/precessingspinpnham_tests.jl`'s "PNFlux vs
TaylorT4wdot coefficient-family resolution" testset). **Conclusion: the two coefficient families
are NOT in conflict -- they are the same physics, expressed via two different (but, at this
leading spin order, numerically identical) PN-approximant conventions**, related by the standard
T1-to-T4 re-expansion identity. `AlignedSpinPNHam`'s flux-ratio route and this file's direct
`TaylorT4wdot`-series route are both individually correct; this file uses the `TaylorT4wdot` family
directly (not `AlignedSpinPNHam`'s flux-ratio construction) purely because there is no Hamiltonian
in this reduced state to divide by (`v` is primary, see "Initial conditions" below) -- `domega/dt`
must be built as a direct `v`-power series regardless, and that series *is* the `TaylorT4wdot`
family by LALSuite's own construction (confirmed directly, `LALSimInspiralSpinTaylor.c` lines
2648-2664).

Also confirmed by reading `LALSimInspiralSpinTaylor.c` lines
483-785: the `dS_i/dt = Omega_i x S_i` / `dLNhat` construction quoted above IS LALSuite's own
`XLALSimInspiralSpinDerivativesAvg` at `spinO=3` -- every coefficient and the `Om`-construction's two-cross-product structure
match line-for-line. The strongest evidence this port reproduces it correctly is the full
trajectory-level comparison in "LALSuite validation" below, not just this analytic check.

# Initial conditions (no Newton solve is needed)

`LNhat(0) = (0,0,1)`, `S_i(0) = X_i^2 * chi_i_vec` (in the `LNhat`-aligned-at-`f_ref` frame, matching
`IMRPhenomXPrecession.jl`'s `_xp_precession_setup` convention exactly), `v(0) = cbrt(pi*M_sec*f_ref)`,
`phase(0) = 0`.

**Why this needs no analogue of `AlignedSpinPNHam`'s 8-iteration Newton solve** (that model needs
one because `v` there is *derived* from the Hamiltonian's `dH/dp_phi` at a
closed-form-estimated `(r0,p_phi0)`, and the closed form and the model's own exact Hamiltonian
disagree on what `(r0,p_phi0)` gives the requested `v0`): in *this* reduced state, `v` **is** a
primary dynamical variable, not derived from anything else -- there is no second quantity
(`LNhat(0)`, `S_i(0)`) whose value depends on `v0` or needs to be jointly solved for consistency
with it. `LNhat(0)` and `S_i(0)` are frame-*defining* conventions (by definition of the "spins
specified in the L-aligned frame at `f_ref`" input convention), independent of `v0`'s value.
**Confirmed, not just argued**: `v0 = cbrt(pi*M_sec*f_ref)` was checked directly against real
LALSuite's own initial state (`SimInspiralSpinTaylorPNEvolveOrbit`'s first output sample `V[0]`,
`m1=30,m2=20 Msun, fStart=20 Hz`) and matches to every displayed digit
(`0.24919144739758323` both sides) -- i.e. LALSuite's own integrator
literally starts from this exact closed form too, with no internal adjustment. This is also
independently corroborated by "LALSuite validation" below: the full trajectory
comparison would show an immediate, large `t=0` mismatch if `v0`/`LNhat(0)`/`S_i(0)` needed any
correction, and it does not.

# Enzyme/KernelAbstractions readiness

`StaticArrays.jl`'s own `cross`/`LinearAlgebra.cross` for `SVector{3,T}` was directly confirmed
(not assumed) to compile allocation-free and generically over `T` inside this package's `@Tlit T`
style (`@allocated` == 0 for both `Float64` and `Float32` inputs to a `where {T}`-generic function
building local `SVector`s from scalar `FieldVector` components and calling `cross`) -- so this file uses `SVector`+`cross` directly rather than
hand-unrolling every cross product `_xp_twist_up22`-style. A central finite-difference smoothness
check stands in for an Enzyme test of this model
(`test/precessingspinpnham_tests.jl`'s own such testset, mirroring `AlignedSpinPNHam`'s).

# LALSuite validation

Compared against `lalsim.SimInspiralSpinTaylorPNEvolveOrbit` at matched `spinO`/`phaseO` truncation:
`V`/`LNhat`/
`S1`/`S2`/`Phi` all agree with real LALSuite to `~1e-9`-`1e-10` out to `~1s` of inspiral, across 3
configurations including the aligned-spin limit.

# Aligned-spin limit

When `chi1`/`chi2` are both exactly along `(0,0,1)` (`LNhat`-aligned at `t=0`), `S_i(0) ∥ LNhat(0)`
exactly, so `LNhat x S_i = 0` identically at `t=0` -- and this is **preserved for all `t`, exactly,
by construction, not just approximately**: if `S_i ∥ LNhat` then `dS_i/dt = c_i*v^5*(LNhat x S_i) =
0` exactly, so `S_i` never changes; then `dL_raw = -(dS1+dS2) = 0`, so `dLNhat_raw = 0`, `Om =
LNhat x 0 = 0`, `dLNhat = 0 x LNhat = 0` -- **`LNhat(t)` stays exactly `(0,0,1)` for all `t`, an
algebraic identity of this construction, not merely a numerically-small effect.** In this limit,
`wspin3 = wdot3S1O*chi1 + wdot3S2O*chi2` reduces to a constant (since `LNhat.S_i = S_i` exactly,
constant), and `domega/dt`'s structure becomes identical in form to a non-precessing 1.5PN-SO
`TaylorT4`-style series -- see `test/precessingspinpnham_tests.jl`'s "Aligned-spin limit"
testset for the numerical confirmation (`LNhat(t) == (0,0,1)` to machine precision at every sampled
`t`) and the cross-check of `phase(t)`/`v(t)` against `AlignedSpinPNHam`'s own non-spin-plus-1.5PN-SO
truncation (both models share the same PN content at this order: Newtonian+1PN non-spin dynamics
plus 1.5PN spin-orbit `domega/dt`, so their frequency evolutions should track closely, *not*
merely both be "finite and distinct").

# Not implemented

- `dE1/dt` is **not** part of the exported 11-component state (see "State vector" above); an
  internal, `evolve`-only companion computation reconstructs an equivalent quantity only where
  needed for the Euler angle `epsilon(t)` -- see [`evolve`](@ref)'s docstring for the important
  caveat about how well-validated that piece specifically is (less than everything else in this
  file).
- `zeta_polarization(t)` (the polarization-frame rotation `twist_up22`/`IMRPhenomXPrecession.jl`
  applies as a final step) is **not** computed or applied by [`evolve`](@ref) -- it does not affect `|hp|`/`|hc|` (a pure common-phase rotation), same
  as `_xp_apply_zeta_rotation`'s own documented invariant.
- Beyond-1.5PN spin terms (2PN spin-spin, 2.5PN/3PN/3.5PN spin-orbit) available in LALSuite's own
  tables -- deliberately omitted here, addable later
  via the identical direct-port technique demonstrated here.
"""
struct PrecessingSpinPNHam end

"""
    PrecessingSpinPNHamState{T} <: FieldVector{11, T}

The `PrecessingSpinPNHam` ODE state: orbital phase `phase`, PN velocity parameter `v`, unit orbital
angular momentum direction `LNhx,LNhy,LNhz`, and reduced spins `S1x,S1y,S1z`, `S2x,S2y,S2z`. A
`StaticArrays.jl` `FieldVector` (not a plain `Array`), mirroring `AlignedSpinPNHamState`/
`ODEToyState` exactly -- see `PrecessingSpinPNHam`'s docstring for the exact hat-unit convention and
why this reduced (not literal position/momentum) state is the right one to reproduce LALSuite's own
reference dynamics.
"""
struct PrecessingSpinPNHamState{T} <: FieldVector{11, T}
    phase::T
    v::T
    LNhx::T
    LNhy::T
    LNhz::T
    S1x::T
    S1y::T
    S1z::T
    S2x::T
    S2y::T
    S2z::T
end

"""
    validate_precessingspinpnham_params(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z; f_ref = nothing)

Host-side sanity check, mirroring `validate_phenomxp_precession_params`'s rationale/signature
exactly (this file's mass-ordering/spin-input convention matches `IMRPhenomXPrecession.jl`'s):
`m1 >= m2 > 0`, `|chi1| <= 1`, `|chi2| <= 1`, and (if `f_ref` given) `f_ref > 0`. Throws
`ArgumentError` on violation, `nothing` otherwise. Not called by [`evolve`](@ref) itself (hot path).
"""
function validate_precessingspinpnham_params(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z; f_ref = nothing)
    m2 > 0.0 || throw(ArgumentError("Invalid m2 = $m2: must be positive"))
    m1 >= m2 || throw(ArgumentError("Invalid m1 = $m1 < m2 = $m2: this file assumes m1 >= m2 (body 1 is the more massive one); swap bodies/spins before calling"))
    chi1_norm2 = chi1x^2 + chi1y^2 + chi1z^2
    chi2_norm2 = chi2x^2 + chi2y^2 + chi2z^2
    chi1_norm2 <= 1.0 || throw(ArgumentError("Invalid chi1 = ($chi1x, $chi1y, $chi1z): |chi1| must be <= 1"))
    chi2_norm2 <= 1.0 || throw(ArgumentError("Invalid chi2 = ($chi2x, $chi2y, $chi2z): |chi2| must be <= 1"))
    if f_ref !== nothing
        f_ref > 0.0 || throw(ArgumentError("Invalid f_ref = $f_ref: reference frequency must be positive"))
    end
    return nothing
end

"""
    _psph_core(phase, v, LNhx, LNhy, LNhz, S1x, S1y, S1z, S2x, S2y, S2z, coeffs) -> (derivs..., Omx, Omy, Omz)

The shared physics core: computes the RHS of all 11 `PrecessingSpinPNHamState` components (in the
same order as the struct's fields) **and** the precession angular velocity `Om` (needed only by
[`evolve`](@ref)'s internal `E1` companion, not by the core 11-component integrator) -- factored out
so that piece is computed once, not duplicated between [`_psph_rhs`](@ref) and the `evolve`-only
14-component companion RHS. See `PrecessingSpinPNHam`'s docstring for the physics/citations. Pure,
allocation-free (confirmed: `SVector`+`LinearAlgebra.cross` for 3-vectors compiles allocation-free
generically over `T`, see that docstring's "Enzyme/KernelAbstractions readiness" section).
"""
@inline function _psph_core(phase::T, v::T, LNhx::T, LNhy::T, LNhz::T,
        S1x::T, S1y::T, S1z::T, S2x::T, S2y::T, S2z::T, coeffs) where {T}
@Tlit T begin
    LNhat = SVector(LNhx, LNhy, LNhz)
    S1 = SVector(S1x, S1y, S1z)
    S2 = SVector(S2x, S2y, S2z)

    LNhdotS1 = LNhx * S1x + LNhy * S1y + LNhz * S1z
    LNhdotS2 = LNhx * S2x + LNhy * S2y + LNhz * S2z

    v2 = v * v
    v3 = v2 * v
    v5 = v3 * v2
    v10 = v5 * v5
    v11 = v10 * v
    omega = v3

    dS1 = coeffs.S1dot3 * v5 * cross(LNhat, S1)
    dS2 = coeffs.S2dot3 * v5 * cross(LNhat, S2)

    dL_raw = -(dS1 + dS2)
    LNmag = coeffs.eta / v
    dLNhat_raw = dL_raw / LNmag
    Om = cross(LNhat, dLNhat_raw)
    dLNhat = cross(Om, LNhat)

    wspin3 = coeffs.wdot3S1O * LNhdotS1 + coeffs.wdot3S2O * LNhdotS2
    domega = coeffs.wdotnewt * v11 * (1.0 + v2 * coeffs.wdotcoeff2 + v3 * (coeffs.wdotcoeff3 + wspin3))
    dv = domega / (3.0 * v2)

    dphase = omega

    return (dphase, dv, dLNhat[1], dLNhat[2], dLNhat[3],
        dS1[1], dS1[2], dS1[3], dS2[1], dS2[2], dS2[3], Om[1], Om[2], Om[3])
end # @Tlit T
end

"""
    _psph_rhs(state::PrecessingSpinPNHamState{T}, coeffs) -> PrecessingSpinPNHamState{T}

The ODE right-hand side for the exported 11-component state (see [`_psph_core`](@ref) for the
shared physics; this just discards the `Om` output that only `evolve`'s internal `E1` companion
needs). Pure, allocation-free, mirroring `AlignedSpinPNHam`'s `_asph_rhs` precedent.
"""
@inline function _psph_rhs(state::PrecessingSpinPNHamState{T}, coeffs) where {T}
    d = _psph_core(state.phase, state.v, state.LNhx, state.LNhy, state.LNhz,
        state.S1x, state.S1y, state.S1z, state.S2x, state.S2y, state.S2z, coeffs)
    return PrecessingSpinPNHamState(d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11])
end

"""
    _psph_rk4_step(state, dt, coeffs) -> PrecessingSpinPNHamState

One classical RK4 step, mirroring `AlignedSpinPNHam`'s `_asph_rk4_step`/`ODEToy`'s
`_odetoy_rk4_step` exactly.
"""
@inline function _psph_rk4_step(state::PrecessingSpinPNHamState{T}, dt::T, coeffs) where {T}
@Tlit T begin
    k1 = _psph_rhs(state, coeffs)
    k2 = _psph_rhs(state + (dt / 2.0) * k1, coeffs)
    k3 = _psph_rhs(state + (dt / 2.0) * k2, coeffs)
    k4 = _psph_rhs(state + dt * k3, coeffs)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
end # @Tlit T
end

"""
    precessingspinpnham_rk4_integrate(state0, dt, nsteps, coeffs) -> PrecessingSpinPNHamState

Fixed-step RK4 integration of [`_psph_rhs`](@ref) from `state0`, `nsteps` steps of size `dt`
(reduced time units). Exported directly, mirroring `alignedspinpnham_rk4_integrate`'s rationale.
"""
@inline function precessingspinpnham_rk4_integrate(state0::PrecessingSpinPNHamState{T}, dt::T, nsteps::Integer, coeffs) where {T}
    state = state0
    for _ in 1:nsteps
        state = _psph_rk4_step(state, dt, coeffs)
    end
    return state
end

"""
    _psph_coeffs(model, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z; f_ref, GMsun_over_c3)

Time-independent part of the evaluation: mass fractions, spin-precession/`domega`-series
coefficients, and the initial state at `f_ref` (see `PrecessingSpinPNHam`'s docstring's "Initial
conditions" section) -- computed once per waveform, mirroring `AlignedSpinPNHam`'s `_asph_coeffs`.
"""
@inline function _psph_coeffs(model::PrecessingSpinPNHam,
    m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z;
    f_ref = 20.0,
    GMsun_over_c3 = GMsun_over_c3,
)
    T = promote_type(typeof(m1), typeof(m2), typeof(chi1x), typeof(chi1y), typeof(chi1z),
        typeof(chi2x), typeof(chi2y), typeof(chi2z))
    GMsun_over_c3 = T(GMsun_over_c3)
    # f_ref is a keyword-default-derived value, not a literal inside the @Tlit block below --
    # @Tlit's own docstring explains why it must be cast explicitly here (same pattern
    # IMRPhenomD.Phi/AlignedSpinPNHam._asph_coeffs already follow for their own kwargs). Without
    # this cast, `v0 = cbrt(pi*M_sec*f_ref)` silently promotes back to Float64 even when T=Float32
    # (found via this file's own Float32-genericity test).
    f_ref = T(f_ref)
@Tlit T begin
    Mtot = m1 + m2
    X1 = m1 / Mtot
    X2 = m2 / Mtot
    eta = X1 * X2

    S1dot3 = 1.5 - X1 - X1 * X1 / 2.0
    S2dot3 = 1.5 - X2 - X2 * X2 / 2.0

    wdotnewt = 96.0 / 5.0 * eta
    wdotcoeff2 = -(1.0 / 336.0) * (743.0 + 924.0 * eta)
    wdotcoeff3 = 4.0 * pi
    wdot3S1O = -19.0 / 6.0 - 25.0 / (4.0 * X1)
    wdot3S2O = -19.0 / 6.0 - 25.0 / (4.0 * X2)

    M_sec = Mtot * GMsun_over_c3
    v0 = cbrt(pi * M_sec * f_ref)

    LNhx0 = 0.0
    LNhy0 = 0.0
    LNhz0 = 1.0
    S1x0 = X1 * X1 * chi1x
    S1y0 = X1 * X1 * chi1y
    S1z0 = X1 * X1 * chi1z
    S2x0 = X2 * X2 * chi2x
    S2y0 = X2 * X2 * chi2y
    S2z0 = X2 * X2 * chi2z

    return (T = T, Mtot = Mtot, eta = eta, X1 = X1, X2 = X2,
        S1dot3 = S1dot3, S2dot3 = S2dot3,
        wdotnewt = wdotnewt, wdotcoeff2 = wdotcoeff2, wdotcoeff3 = wdotcoeff3,
        wdot3S1O = wdot3S1O, wdot3S2O = wdot3S2O,
        M_sec = M_sec, v0 = v0,
        LNhx0 = LNhx0, LNhy0 = LNhy0, LNhz0 = LNhz0,
        S1x0 = S1x0, S1y0 = S1y0, S1z0 = S1z0,
        S2x0 = S2x0, S2y0 = S2y0, S2z0 = S2z0)
end # @Tlit T
end

"""
    _psph_state_scalar(coeffs, t, dt)

Non-allocating: integrates from the shared initial condition at `f_ref` for `round(Int, t/dt)`
fixed RK4 steps of reduced-time size `dt/coeffs.M_sec`, returning the resulting
`PrecessingSpinPNHamState`. `t`, `dt` in physical seconds (seconds since the `f_ref`-referenced
initial condition, matching `AlignedSpinPNHam`'s own domain convention).
"""
@inline function _psph_state_scalar(coeffs, t, dt)
    T = coeffs.T
    state0 = PrecessingSpinPNHamState(zero(T), coeffs.v0,
        coeffs.LNhx0, coeffs.LNhy0, coeffs.LNhz0,
        coeffs.S1x0, coeffs.S1y0, coeffs.S1z0,
        coeffs.S2x0, coeffs.S2y0, coeffs.S2z0)
    nsteps = round(Int, t / dt)
    dt_hat = T(dt) / coeffs.M_sec
    return precessingspinpnham_rk4_integrate(state0, dt_hat, nsteps, coeffs)
end

# ================================================================================================
# evolve(::PrecessingSpinPNHam, t, ...) -> (hp, hc). Does NOT participate in the Phi/Ampl
# contract (see PrecessingSpinPNHam's docstring; matching IMRPhenomXP/XHM/XPHM's own precedent). Reuses _xp_twist_up22's Wigner-d/spherical-
# harmonic algebra (IMRPhenomXPrecession.jl lines ~556-645) via a new function parameterized
# directly on (alpha, cos_beta, epsilon, thetaJN) rather than a `_xp_precession_setup`-style
# NamedTuple, since this model's angles come from its own live ODE trajectory, not the NNLO closed
# form _xp_precession_setup's fields are built around.
# ================================================================================================

"""
    _psph_twist_up22(alpha, cos_beta, epsilon, thetaJN, hAS)

Genuinely scalar, non-allocating twisting-up of a co-precessing `(2,2)` mode strain `hAS` into
inertial-frame `(hp, hc)`, given the Euler angles `(alpha, beta, epsilon)` and the fixed line-of-
sight polar angle `thetaJN` **directly** (not derived from a `_xp_precession_setup` NamedTuple's
own NNLO closed form) -- this is the *same* algebraic transform `_xp_twist_up22`
(`IMRPhenomXPrecession.jl`, ported from `IMRPhenomXPTwistUp22` lines 2026-2079/2100-2111) implements
(Wigner-d half-angle coefficients, spin-weighted spherical harmonics `Y_{2,m}(thetaJN,phi=0)`, the
`m=-2:2` transfer-function sum), duplicated here (not called via `_xp_twist_up22` itself) because
that function's *inputs* are hardwired to `_xp_cos_beta`/`_xp_pn_euler_alpha_nnlo`/
`_xp_pn_euler_epsilon_nnlo` (the NNLO closed form this model deliberately does not use -- see
`PrecessingSpinPNHam`'s docstring) rather than accepting angles directly. See `_xp_twist_up22`'s own
docstring for the full derivation/citation of every piece of this algebra; nothing here is new
physics, only a different (explicit-angle) calling convention for the identical formula.
"""
@inline function _psph_twist_up22(alpha::T, cos_beta::T, epsilon::T, thetaJN::T, hAS::Complex{T}) where {T}
@Tlit T begin
    cBetah = sqrt(abs(1.0 + cos_beta) / 2.0)
    sBetah = sqrt(abs(1.0 - cos_beta) / 2.0)

    cBetah2 = cBetah * cBetah
    cBetah3 = cBetah * cBetah2
    cBetah4 = cBetah * cBetah3
    sBetah2 = sBetah * sBetah
    sBetah3 = sBetah * sBetah2
    sBetah4 = sBetah * sBetah3

    sqrt6 = sqrt(6.0)

    d22_m2 = sBetah4
    d22_m1 = 2.0 * cBetah * sBetah3
    d22_0  = sqrt6 * sBetah2 * cBetah2
    d22_p1 = 2.0 * cBetah3 * sBetah
    d22_p2 = cBetah4

    d2m2_m2 = d22_p2
    d2m2_m1 = -d22_p1
    d2m2_0  = d22_0
    d2m2_p1 = -d22_m1
    d2m2_p2 = d22_m2

    theta = thetaJN
    ct = cos(theta)
    st = sin(theta)
    Y2m2 = sqrt(5.0 / (64.0 * pi)) * (1.0 - ct) * (1.0 - ct)
    Y2m1 = sqrt(5.0 / (16.0 * pi)) * st * (1.0 - ct)
    Y20  = sqrt(15.0 / (32.0 * pi)) * st * st
    Y2p1 = sqrt(5.0 / (16.0 * pi)) * st * (1.0 + ct)
    Y2p2 = sqrt(5.0 / (64.0 * pi)) * (1.0 + ct) * (1.0 + ct)

    cis_alpha   = cis(alpha)
    cis_2alpha  = cis_alpha * cis_alpha
    cis_malpha  = conj(cis_alpha)
    cis_m2alpha = conj(cis_2alpha)

    A2m2emm_m2 = cis_2alpha  * d2m2_m2 * Y2m2
    A2m2emm_m1 = cis_alpha   * d2m2_m1 * Y2m1
    A2m2emm_0  =                d2m2_0  * Y20
    A2m2emm_p1 = cis_malpha  * d2m2_p1 * Y2p1
    A2m2emm_p2 = cis_m2alpha * d2m2_p2 * Y2p2

    A22emmstar_m2 = cis_m2alpha * d22_m2 * Y2m2
    A22emmstar_m1 = cis_malpha  * d22_m1 * Y2m1
    A22emmstar_0  =                d22_0  * Y20
    A22emmstar_p1 = cis_alpha   * d22_p1 * Y2p1
    A22emmstar_p2 = cis_2alpha  * d22_p2 * Y2p2

    polarizationSymmetry = 1.0

    hp_sum = (A2m2emm_m2 + polarizationSymmetry * A22emmstar_m2) +
             (A2m2emm_m1 + polarizationSymmetry * A22emmstar_m1) +
             (A2m2emm_0  + polarizationSymmetry * A22emmstar_0 ) +
             (A2m2emm_p1 + polarizationSymmetry * A22emmstar_p1) +
             (A2m2emm_p2 + polarizationSymmetry * A22emmstar_p2)

    hc_sum = im * ((A2m2emm_m2 - polarizationSymmetry * A22emmstar_m2) +
                   (A2m2emm_m1 - polarizationSymmetry * A22emmstar_m1) +
                   (A2m2emm_0  - polarizationSymmetry * A22emmstar_0 ) +
                   (A2m2emm_p1 - polarizationSymmetry * A22emmstar_p1) +
                   (A2m2emm_p2 - polarizationSymmetry * A22emmstar_p2))

    eps_phase_hP = cis(-2.0 * epsilon) * hAS / 2.0

    hp = eps_phase_hP * hp_sum
    hc = eps_phase_hP * hc_sum
    return (hp, hc)
end # @Tlit T
end

"""
    _psph_angles(state, coeffs, E1, Nvec) -> (alpha, cos_beta, epsilon, thetaJN)

Reconstructs the twisting-up Euler angles **from a snapshot** of the model's own live state (no
separate angle-ODE for `alpha`/`beta`, see below) plus a companion `E1` (see [`evolve`](@ref)'s
docstring for how `E1` is obtained). `Nvec = (Nx,Ny,Nz)` is the fixed line-of-sight direction in the
`LNhat(0)`-aligned frame (same construction as `_xp_precession_setup`'s `Nx,Ny,Nz`,
`LALSimIMRPhenomX_precession.c` lines 865-867).

`J(t) = L(t)*LNhat(t) + S1(t) + S2(t)`, `L(t) = eta/v(t)` (leading order, matching this model's own
`LNmag` used in [`_psph_core`](@ref) -- NOT `IMRPhenomXPrecession.jl`'s 3PN `L(v)` ansatz, which
would be inconsistent with this model's own 1.5PN truncation). `cos_beta(t) = LNhat(t).Jhat(t)`
directly (dot product, no closed form needed). `alpha(t)`: the azimuthal angle of `LNhat(t)` in the
**live** `Jhat(t)`-aligned frame, reconstructed via the *same* `rotate_z(-phiJ)`/`rotate_y(-thetaJ)`
recipe `_xp_precession_setup` uses once (at `f_ref`) to build `kappa` from the line-of-sight `N` --
here re-evaluated at every dynamical `t` from this model's own live `J(t)`, giving a genuinely
time-dependent precession-cone evolution (as opposed to `IMRPhenomXPrecession.jl`'s single frozen `J0`). `thetaJN(t) =
acos(Jhat(t).Nvec)` (Convention-1-style direct dot product, mirroring `_xp_precession_setup`'s own
`thetaJN` construction, now time-dependent).

`epsilon(t)` is reconstructed from the companion `E1(t)` (which is `perp` to `LNhat(t)` by
construction, see [`evolve`](@ref)'s docstring) via the same `rotate_z`/`rotate_y` recipe, applied
this time to extract the residual in-plane angle of `E1` after undoing the `(alpha,beta)` rotation
-- **this piece is the least independently validated part of this file**, see `evolve`'s docstring.
"""
@inline function _psph_angles(state::PrecessingSpinPNHamState{T}, coeffs, E1x::T, E1y::T, E1z::T, Nx::T, Ny::T, Nz::T) where {T}
@Tlit T begin
    L = coeffs.eta / state.v
    Jx = L * state.LNhx + state.S1x + state.S2x
    Jy = L * state.LNhy + state.S1y + state.S2y
    Jz = L * state.LNhz + state.S1z + state.S2z
    Jnorm = sqrt(Jx * Jx + Jy * Jy + Jz * Jz)
    Jnorm_safe = max(Jnorm, eps(T))
    Jhx, Jhy, Jhz = Jx / Jnorm_safe, Jy / Jnorm_safe, Jz / Jnorm_safe

    phiJ = atan(Jhy, Jhx)
    thetaJ = acos(clamp(Jhz, -1.0, 1.0))

    Lx1, Ly1, Lz1 = _xp_rotate_z(-phiJ, state.LNhx, state.LNhy, state.LNhz)
    Lx2, Ly2, Lz2 = _xp_rotate_y(-thetaJ, Lx1, Ly1, Lz1)
    cos_beta = clamp(Lz2, -1.0, 1.0)
    alpha = atan(Ly2, Lx2)

    Ex1, Ey1, Ez1 = _xp_rotate_z(-phiJ, E1x, E1y, E1z)
    Ex2, Ey2, Ez2 = _xp_rotate_y(-thetaJ, Ex1, Ey1, Ez1)
    Ex3, Ey3, Ez3 = _xp_rotate_z(-alpha, Ex2, Ey2, Ez2)
    beta = acos(cos_beta)
    Ex4, Ey4, Ez4 = _xp_rotate_y(-beta, Ex3, Ey3, Ez3)
    epsilon = atan(Ey4, Ex4)

    JdotN = Jhx * Nx + Jhy * Ny + Jhz * Nz
    thetaJN = acos(clamp(JdotN, -1.0, 1.0))

    return (alpha, cos_beta, epsilon, thetaJN)
end # @Tlit T
end

"""
    _psph_rhs_with_E1(state11, E1x, E1y, E1z, coeffs) -> (state11_deriv..., dE1x, dE1y, dE1z)

`evolve`-internal companion: the same 11-component physics as [`_psph_rhs`](@ref), plus
`dE1/dt = Om x E1` (LALSuite's own passive bookkeeping relation for its 12th-14th state component,
`XLALSimInspiralSpinDerivativesAvg`, the final `dE1x/dE1y/dE1z = -Om[2]*E1y+Om[1]*E1z, ...` lines).
`E1` is proportional to neither `S1`/`S2`/`LNhat` and starts at an arbitrary (but fixed, and
`LNhat`-perpendicular) convention `E1(0)=(1,0,0)` -- since `LNhat(0)=(0,0,1)` always in this model's
own initial-condition convention, `E1(0)=(1,0,0)` is automatically perpendicular, matching
LALSuite's own typical convention.

**Algebraic fact, checked directly** (not just assumed): `E1(t)` stays exactly perpendicular to
`LNhat(t)` for all `t`, since both are driven by the same `Om` (`d(E1.LNhat)/dt =
(Om x E1).LNhat + E1.(Om x LNhat) = Om.(E1 x LNhat) + Om.(LNhat x E1) = 0` by the scalar triple
product identity) -- confirmed numerically in `test/precessingspinpnham_tests.jl`.
"""
@inline function _psph_rhs_with_E1(state::PrecessingSpinPNHamState{T}, E1x::T, E1y::T, E1z::T, coeffs) where {T}
@Tlit T begin
    d = _psph_core(state.phase, state.v, state.LNhx, state.LNhy, state.LNhz,
        state.S1x, state.S1y, state.S1z, state.S2x, state.S2y, state.S2z, coeffs)
    Omx, Omy, Omz = d[12], d[13], d[14]
    dE1x = -Omz * E1y + Omy * E1z
    dE1y = -Omx * E1z + Omz * E1x
    dE1z = -Omy * E1x + Omx * E1y
    return (d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], dE1x, dE1y, dE1z)
end # @Tlit T
end

"""
    _psph_state_with_E1_scalar(coeffs, t, dt) -> (state::PrecessingSpinPNHamState, E1x, E1y, E1z)

`evolve`-internal: fixed-step RK4 integration of the 11-component state **and** the companion `E1`
(see [`_psph_rhs_with_E1`](@ref)) together, using a plain manual RK4 (rather than a second
`FieldVector` type) since `E1` is only 3 extra scalars and this keeps the exported
`PrecessingSpinPNHamState`/`precessingspinpnham_rk4_integrate` surface at exactly 11 components, per
`PrecessingSpinPNHam`'s docstring's "State vector" section.
"""
@inline function _psph_state_with_E1_scalar(coeffs, t, dt)
    T = coeffs.T
    state0 = PrecessingSpinPNHamState(zero(T), coeffs.v0,
        coeffs.LNhx0, coeffs.LNhy0, coeffs.LNhz0,
        coeffs.S1x0, coeffs.S1y0, coeffs.S1z0,
        coeffs.S2x0, coeffs.S2y0, coeffs.S2z0)
    E1x, E1y, E1z = one(T), zero(T), zero(T)
    nsteps = round(Int, t / dt)
    dth = T(dt) / coeffs.M_sec

    state = state0
    for _ in 1:nsteps
        k1 = _psph_rhs_with_E1(state, E1x, E1y, E1z, coeffs)
        s2 = state + (dth / 2) * PrecessingSpinPNHamState(k1[1:11]...)
        e2x, e2y, e2z = E1x + (dth / 2) * k1[12], E1y + (dth / 2) * k1[13], E1z + (dth / 2) * k1[14]
        k2 = _psph_rhs_with_E1(s2, e2x, e2y, e2z, coeffs)
        s3 = state + (dth / 2) * PrecessingSpinPNHamState(k2[1:11]...)
        e3x, e3y, e3z = E1x + (dth / 2) * k2[12], E1y + (dth / 2) * k2[13], E1z + (dth / 2) * k2[14]
        k3 = _psph_rhs_with_E1(s3, e3x, e3y, e3z, coeffs)
        s4 = state + dth * PrecessingSpinPNHamState(k3[1:11]...)
        e4x, e4y, e4z = E1x + dth * k3[12], E1y + dth * k3[13], E1z + dth * k3[14]
        k4 = _psph_rhs_with_E1(s4, e4x, e4y, e4z, coeffs)

        state = state + (dth / 6) * PrecessingSpinPNHamState((k1[1:11] .+ 2 .* k2[1:11] .+ 2 .* k3[1:11] .+ k4[1:11])...)
        E1x = E1x + (dth / 6) * (k1[12] + 2 * k2[12] + 2 * k3[12] + k4[12])
        E1y = E1y + (dth / 6) * (k1[13] + 2 * k2[13] + 2 * k3[13] + k4[13])
        E1z = E1z + (dth / 6) * (k1[14] + 2 * k2[14] + 2 * k3[14] + k4[14])
    end
    return state, E1x, E1y, E1z
end

"""
    evolve(model::PrecessingSpinPNHam, t, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dL,
           inclination, phiRef; f_ref=20.0, dt=1.0/4096.0, GMsun_over_c3=GMsun_over_c3,
           GMsun_over_c2_Gpc=GMsun_over_c2_Gpc) -> (hp, hc)

End-to-end precessing waveform: integrates this model's own 11-component ODE state
(plus an internal `E1` companion, see below) to time(s) `t` (seconds since the `f_ref`-referenced
initial condition, `t >= 0`, matching `AlignedSpinPNHam`'s domain convention), builds the
co-precessing `(2,2)`-mode strain directly from this model's own `(phase, v)` (restricted/Newtonian
amplitude `4*eta*M*GMsun_over_c2_Gpc/dL*v^2`, GW phase `2*phase` -- identical formulas to
`AlignedSpinPNHam.Ampl`/`Phi`, since this model's own `domega/dt` already includes the
`LNhat(t).S_i(t)`-projected 1.5PN spin-orbit term, i.e. it already *is* the
instantaneously aligned-spin-like physics, so no separate Hamiltonian evaluation is needed), and twists it up via [`_psph_twist_up22`](@ref) using Euler angles
derived from this model's own dynamically-integrated `LNhat(t)`/`S1(t)`/`S2(t)` (see
[`_psph_angles`](@ref)) -- **not** `IMRPhenomXPrecession.jl`'s NNLO closed form.

Does **not** participate in the `Phi`/`Ampl` two-function contract (matching `IMRPhenomXP`/
`IMRPhenomXHM`/`IMRPhenomXPHM`'s own precedent -- see `PrecessingSpinPNHam`'s docstring).

# Arguments
`m1 >= m2` in solar masses; `chi1x,y,z`/`chi2x,y,z` dimensionless spin components in the
`LNhat`-aligned frame at `f_ref` (matching `IMRPhenomXPrecession.jl`'s convention exactly); `dL`
luminosity distance in Gpc; `inclination`, `phiRef` define the fixed line-of-sight direction `N`
(same formula as `_xp_precession_setup`'s `Nx,Ny,Nz`, `LALSimIMRPhenomX_precession.c` lines
865-867) in the `LNhat(0)`-aligned frame.

# Important caveat: `epsilon(t)`'s validation status

`alpha(t)`/`beta(t)`/`thetaJN(t)` (see [`_psph_angles`](@ref)) are direct, unambiguous geometric
reconstructions from this model's own live `J(t)`/`LNhat(t)` -- no extra machinery, no ambiguity.
`epsilon(t)` additionally needs a companion in-plane reference vector `E1(t)` (`dE1/dt = Om x E1`,
[`_psph_rhs_with_E1`](@ref)) -- this exactly mirrors *why LALSuite's own state carries `E1` at all*
(it is otherwise completely dynamically decoupled from everything else, see that function's
docstring) -- reconstructed here as an `evolve`-only internal companion rather than a 12th-14th
component of the exported state. **This piece (the specific sign/rotation-order convention used to extract `epsilon`
from `E1` in [`_psph_angles`](@ref)) has no direct LALSuite ground truth to check against** -- unlike
every other formula in this file, which was checked either against a closed-form LALSuite
coefficient or a full LALSuite trajectory, LALSuite itself only ever computes `epsilon`
via the NNLO closed form (a *different*, approximate physical quantity -- fixed-`J0`, not a live
trajectory) or via internal waveform-generation code not separately exposed for this kind of
comparison. Only an internal self-consistency check was performed (`test/precessingspinpnham_tests.jl`:
reconstructing `E1(t)` from `(alpha(t),beta(t),epsilon(t))` via the same rotation composition and
confirming it reproduces the integrated `E1(t)` used to derive them) -- this confirms the formula is
*self-consistent*, not that its sign/convention matches what a from-scratch derivation or a future
LALSuite comparison would independently require. This remains an open
item (e.g. until a direct LALSuite `epsilon`-equivalent trajectory becomes available to check
against, or a from-scratch minimal-rotation-condition derivation, Boyle, Owen & Pfeiffer, Phys. Rev.
D 84, 124011 (2011), arXiv:1110.2965, is independently re-derived and cross-checked). Since
`epsilon` enters [`_psph_twist_up22`](@ref) only as a common overall phase factor
`exp(-2i*epsilon)` applied identically to both `hp` and `hc` (see that function's docstring), this
caveat affects only an overall (time-dependent) phase convention of the output, not `|hp|`/`|hc|`
or the relative `hp`/`hc` mixing the twisting-up sum itself produces -- i.e., it does not affect
this model's core validated content, only this specific optional convenience entry
point's own phase gauge.

Also **not applied**: the `zeta_polarization` post-rotation `twist_up22`/`IMRPhenomXPrecession.jl`
uses (explicitly deferred, see `PrecessingSpinPNHam`'s docstring) -- another pure common-phase
effect, same caveat class as `epsilon` above.
"""
function evolve(model::PrecessingSpinPNHam,
    t,
    m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dL, inclination, phiRef;
    f_ref = 20.0,
    dt = 1.0 / 4096.0,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    coeffs = _psph_coeffs(model, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z;
        f_ref = f_ref, GMsun_over_c3 = GMsun_over_c3)
    T = coeffs.T
    dL = T(dL)
    GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc)
    inclination = T(inclination)
    phiRef = T(phiRef)

    Nx = sin(inclination) * cos(T(pi) / 2 - phiRef)
    Ny = sin(inclination) * sin(T(pi) / 2 - phiRef)
    Nz = cos(inclination)

    function hphc_scalar(tt)
        state, E1x, E1y, E1z = _psph_state_with_E1_scalar(coeffs, tt, dt)
        alpha, cos_beta, epsilon, thetaJN = _psph_angles(state, coeffs, E1x, E1y, E1z, Nx, Ny, Nz)
        ampl = 4 * coeffs.eta * coeffs.Mtot * GMsun_over_c2_Gpc / dL * state.v * state.v
        hAS = Complex{T}(ampl * cis(2 * state.phase))
        return _psph_twist_up22(alpha, cos_beta, epsilon, thetaJN, hAS)
    end

    if t isa AbstractArray
        hp = similar(t, Complex{T})
        hc = similar(t, Complex{T})
        @inbounds for i in eachindex(t)
            hp[i], hc[i] = hphc_scalar(t[i])
        end
        return hp, hc
    else
        return hphc_scalar(t)
    end
end
