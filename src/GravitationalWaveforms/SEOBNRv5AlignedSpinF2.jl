export SEOBNRv5AlignedSpinF2

"""
    SEOBNRv5AlignedSpinF2

**Frequency-domain evaluator for [`SEOBNRv5AlignedSpin`](@ref), built on its post-adiabatic dense
trajectory** (`seobnrv5alignedspin_postadiabatic_dynamics_fixed`, `SEOBNRv5AlignedSpinPAFixed.jl`)
rather than on `SEOBNRv5AlignedSpin.Phi`/`.Ampl`'s own from-scratch-per-query time-domain design.
`SEOBNRv5AlignedSpin.Phi`/`.Ampl` re-integrate from `t=0` on every
call, making a naive per-bin-edge Newton inversion of `Omega(t)=pi*f` cost `O(nbins * Newton_iters)`
full re-integrations -- plausibly *more* expensive than a dense-in-time pass, defeating relative
binning's entire point -- and recommended instead a **single dense-in-time forward integration
that stores `(t, Omega, Phi, Ampl)` once, followed by cheap lookups per bin edge**. That storage
already exists (`SEOBNRv5AlignedSpinPAFixed.jl`'s fixed-buffer post-adiabatic driver, forward-mode
`Enzyme`-validated); this file adds the missing
time-to-frequency reconstruction on top of it.

# Method: numerical stationary-phase-approximation (SPA) reconstruction

The post-adiabatic driver returns, at each of `nreal` grid nodes (built on a fixed-step **radius**
grid, so **non-uniform** in `t`), the exact state `(t_i, r_i, phi_i, pr_i, pphi_i)`. Two facts make
a monotonic-cubic-Hermite-based reconstruction possible without any additional ODE work:

1. **`dphi/dt = Omega` exactly at every node**, `Omega = dH/dpphi` -- this is just Hamilton's
   equation, true regardless of whether the node came from post-adiabatic quasi-circular
   `r`-parametrized integration or a genuine `t`-stepped RK4 (both this file and
   `AlignedSpinPNHam`/`SEOBNRv5AlignedSpin`'s own RK4 satisfy it identically). So each grid
   interval `[t_i, t_{i+1}]` has an exact cubic Hermite interpolant of `phi(t)` (value + slope
   known at both ends) -- no finite-difference derivative estimate needed.
2. **`Omega(t)` is monotonically increasing through the adiabatic inspiral** (confirmed directly on
   real configurations below), so inverting `Omega(t*) = pi*f` per target frequency `f` is a single
   monotonic quadratic solve on that interval's Hermite derivative (closed-form, not iterative) --
   `O(1)` per query after an `O(log nreal)` bracket search, not a re-integration.

Given `t*(f)`, the standard SPA formulas (e.g. Maggiore, *Gravitational Waves* Vol. 1, eq. 4.34,
applied to `h(t)=A(t)cos(Phi(t))`) reconstruct the frequency-domain waveform:
```
Phi_FD(f) = 2*pi*f*t*(f) - Phi_TD(t*(f)) - pi/4
Ampl_FD(f) = (1/2)*Ampl_TD(t*(f)) * sqrt(2*pi / (dOmega_GW/dt)|_{t*(f)})
```
`dOmega_GW/dt` at `t*` (`Omega_GW = 2*Omega_orb`) comes from nodal `d(Omega)/dt` values (degree-6 polynomial through
the 7 nodes around each node, [`_spa_node_wdot`](@ref)), with `1/sqrt(dOmega/dt)` interpolated between the nodes as a
cubic Hermite in `t`. The phase Hermite's own second derivative would be `O(h^2)`, discontinuous at
every node and jumping by ~1e-3 (relative) whenever the PA grid gains or loses a node.

# Validation performed

This exact machinery (implemented as a throwaway prototype before being committed here) was
validated two ways, entirely reusing physics already validated elsewhere in this package:

1. **Exact-to-floating-point amplitude check** against [`AlignedSpinPNHamF2`](@ref) (itself
   validated to `~1e-14` against real `lalsimulation`): built the same reconstruction on top of
   `AlignedSpinPNHam`'s own dense RK4 trajectory (its `dphi/dt=Omega` RHS convention is identical),
   substituted the analytic **Newtonian-order** `dOmega/dt = (96/5)*eta*Omega^(11/3)` (matching
   `AlignedSpinPNHamF2`'s own restricted-PN-order amplitude convention exactly, since a "restricted"
   PN waveform keeps the amplitude at leading order even when the phase is high-order), and got
   `Ampl_FD` agreeing with `AlignedSpinPNHamF2.Ampl` to full `Float64` precision (ratio `1.0` to
   15-16 significant digits) at 5 test frequencies. This pins down the `(1/2)` prefactor (a factor
   easy to drop by mistake -- an earlier attempt without it was off by a smooth, frequency-dependent
   factor of ~1.3-2x, not a constant, which is what motivated deriving it from Maggiore's formula
   directly rather than guessing).
2. **Phase-shape check** using the model's own true (non-Newtonian-restricted) `dOmega/dt`: matched
   `AlignedSpinPNHamF2.Phi` to `~2e-3` rad residual *after* removing a best-fit affine
   (`a + b*f`) term -- expected, since the two models' `t=0`/`phi=0` references differ by
   construction (this file's `t*` is measured from the `f_min` initial condition; `AlignedSpinPNHamF2`
   anchors phase via its own `t0`/`phiRef` convention) and `AlignedSpinPNHamF2` is independently
   known to omit `AlignedSpinPNHam`'s quadrupole spin-spin content (see that file's own docstring),
   which would show up as exactly such a residual at this level.
3. **Self-consistency check on the real `SEOBNRv5AlignedSpin` PA table itself**: interpolated
   `Phi_TD(t*(f))` from the PA-fixed dense table agrees with an *independent from-scratch RK4
   re-integration* (`SEOBNRv5AlignedSpin.Phi` called directly at the same `t*`) to `~0.02-0.04` rad
   out of a phase that has accumulated to `~1000+` rad by `f=100`Hz (config `mc=8.0, eta=0.2479,
   chi1=0.3, chi2=-0.2, f_min=20`) -- a tiny, expected post-adiabatic-vs-full-dynamics residual (a
   known, already-documented approximation elsewhere in this package), not a defect in the
   interpolation itself.

# Enzyme validation

`test/seobnrv5alignedspinf2_enzyme_ka_tests.jl` (25 assertions, all passing) directly answers the
data-dependent-bracket-search risk flagged above: **both** `Enzyme.Reverse` and
`Enzyme.Forward` differentiate `_seobv5f2_phi_scalar`/`_seobv5f2_ampl_scalar` correctly (matching
central finite differences to `rtol=1e-4`) w.r.t. `coeffs.M_sec` -- the one scalar field that feeds
BOTH the bracket-selecting `omega_target = pi*M_sec*f` computation (a genuinely discontinuous,
data-dependent `searchsortedlast` branch) AND the final physical-unit conversions -- and w.r.t. `f`
itself across 8 frequencies spanning different bracket indices. **This is done exactly mirroring
`SEOBNRv5AlignedSpin`'s own established precedent** (`seobnrv5alignedspin_enzyme_ka_tests.jl`):
differentiate the analytic hot path w.r.t. a `coeffs` field, with the dense tables themselves
(built by the non-differentiable, `Float64`-only `_seobv5f2_coeffs`/PA-fixed driver/two-stage IC
Newton solve) held fixed -- **not** a claim that `mc`/`eta`/`chi1`/`chi2` differentiate through the
full `_seobv5f2_coeffs` pipeline end-to-end, exactly as the base model never claimed that for its
own IC solve either. Getting a full physical-parameter gradient for a real `GravitationalWaveBayes.jl` kernel would
need the same treatment the base model already uses in production: differentiate w.r.t. a `coeffs`
field passed in directly (e.g. via `merge`), holding the IC/table construction outside the AD graph.

# What this is NOT (yet)

**Not proven Enzyme-differentiable through the full `_seobv5f2_coeffs` pipeline** (the two-stage IC
Newton solve and the PA-fixed driver itself) -- only the reconstruction hot path on top of an
already-built table, per the validation above. **No `KernelAbstractions.jl` `@kernel` test yet** for
this model (the base model's own `@kernel` composition was validated separately; this file's
`@kernel` composition is untested). **Not valid past the post-adiabatic handoff radius** (`~r=10-11.5M`,
see `SEOBNRv5AlignedSpinPAFixed.jl`) -- this covers the adiabatic inspiral only, which is exactly
where the standard SPA's own slowly-varying-amplitude assumption holds; frequencies above
`coeffs.fmax` (the frequency reached at the PA table's last real node) return `0`, the same
hard-cutoff convention `AlignedSpinPNHamF2`/`TaylorF2` already use for their own ISCO cutoff. Reaching
merger/ringdown needs the plunge-region physics `SEOBNRv5AlignedSpinPA.jl`'s `combined_dynamics`
tail already carries, stitched in separately -- out of scope here.
"""
struct SEOBNRv5AlignedSpinF2 end

"""
    _seobv5f2_searchsortedlast(v, x)

`searchsortedlast(v, x)` for a sorted `v`, written with a COMPILE-TIME-CONSTANT trip count (9
power-of-two strides, no early exit) for `length(v) <= 511`; longer vectors fall back to
`Base.searchsortedlast` (data-dependent trip count -- fine on CPU / forward mode, and only reached
for vectors well beyond the fixed `NMAX = 256` table this is used with).

Why not just `Base.searchsortedlast`: its loop has a data-dependent trip count, and Enzyme reverse mode
caches such a loop's `lo`/`hi` with its "exponential allocation" scheme. On GPU (no `realloc`) that
scheme `malloc`s a bigger buffer, `memcpy`s, and NEVER frees the old one, leaking ~0.5MB of device heap
per 16-walker gradient call until the 8MB default device heap is exhausted after ~11 calls (CUDA
`illegal memory access`, code 700). A constant trip count gets a fixed-size cache instead. Same result
as `searchsortedlast` for sorted `v` (0 if `x < v[1]`).
"""
@inline function _seobv5f2_searchsortedlast(v::AbstractVector{Float64}, x)
    n = length(v)
    n > 511 && return searchsortedlast(v, x)
    pos = 0
    @inbounds for k in 8:-1:0
        cand = pos + (1 << k)
        if cand <= n && v[cand] <= x
            pos = cand
        end
    end
    return pos
end

"""
    _spa_root(ts, phis, omegas, i, omega_target) -> (s, t_star, phi_star)

Stationary point of the SPA on node interval `i` of a `(t, phi, omega)` table with `dphi/dt = omega` at the nodes:
the root `s in [0, 1]` (`t = ts[i] + s h`) of the cubic Hermite interpolant's derivative `phi'(t) = omega_target`
(closed-form quadratic), and the interpolated phase there. Shared by the F2, `EOBIMRTable` and SEOBNRv5HM tables;
the caller does the bracket search (each table has its own fixed-trip-count one) and clamps `i` to `1:n-1`.
"""
@inline function _spa_root(ts, phis, omegas, i, omega_target)
    h = ts[i + 1] - ts[i]
    phi_i, phi_ip1 = phis[i], phis[i + 1]
    om_i, om_ip1 = omegas[i], omegas[i + 1]
    # Hermite derivative phi'(s) = A*s^2 + B*s + C on s in [0,1] (t = ts[i] + s*h);
    # solve A*s^2 + B*s + (C - h*omega_target) = 0 for the physical root in [0,1].
    A = 6.0 * (phi_i - phi_ip1) + 3.0 * h * (om_i + om_ip1)
    B = 6.0 * (phi_ip1 - phi_i) - h * (4.0 * om_i + 2.0 * om_ip1)
    Cc = h * om_i - h * omega_target
    # Root in [0,1]; at the interval ends rounding can push the physical root just outside (e.g. 1 + 1e-12 when
    # f is clamped to the table top, omega_target == omegas[nreal]), so take the root CLOSER to [0,1] and clamp it.
    # (The other root is spurious -- ~0.4 s off in the f_ref time shift for
    # walkers whose f_ref clamps to fmax, and backend-dependent through FMA rounding.) Identical whenever a root
    # lies inside [0,1].
    s = if abs(A) < 1e-14
        clamp(-Cc / B, 0.0, 1.0)
    else
        disc = B * B - 4.0 * A * Cc
        sqrt_disc = sqrt(max(disc, 0.0))
        s1 = (-B + sqrt_disc) / (2.0 * A)
        s2 = (-B - sqrt_disc) / (2.0 * A)
        d1 = max(0.0, -s1, s1 - 1.0)
        d2 = max(0.0, -s2, s2 - 1.0)
        clamp(d1 <= d2 ? s1 : s2, 0.0, 1.0)
    end
    t_star = ts[i] + s * h
    h00 = 2.0 * s^3 - 3.0 * s^2 + 1.0
    h10 = s^3 - 2.0 * s^2 + s
    h01 = -2.0 * s^3 + 3.0 * s^2
    h11 = s^3 - s^2
    phi_star = h00 * phi_i + h10 * h * om_i + h01 * phi_ip1 + h11 * h * om_ip1
    return (s = s, t_star = t_star, phi_star = phi_star)
end

# One divided-difference level k of `_spa_node_wdot` (entries j > k replaced, the rest kept).
@inline _spa_ddiff(d, x, ::Val{k}) where {k} = ntuple(j -> j <= k ? d[j] : (d[j] - d[j - 1]) / (x[j] - x[j - k]), Val(7))

"""
    _spa_node_wdot(ts, omegas, i, n) -> (wdot, wddot)

`d(omega)/dt` and `d^2(omega)/dt^2` at node `i` of an SPA table with `n` real nodes, from the degree-6 polynomial
through the 7 nodes around it (window shifted inside `1:n` at the table ends). Works on any node spacing: the PA nodes
are uniform in `r`, the `EOBIMRTable`/SEOBNRv5HM tables append nodes uniform in `t`.

Why not the second derivative of the phase Hermite: that one is only `O(h^2)`, jumps at every
node, and moves when the PA grid gains or loses a node (`nreal = ceil((r0 - r_final) / dr0)` as `r0(mc)` moves) --
a ~1e-3 jump in the amplitude, so in `logL`, at every such step. The nodal value is ~80x more stable there and makes
the amplitude C1 in `f`. Unrolled (no loops), so in-kernel Enzyme reverse mode tapes nothing per iteration. Fewer
than 7 nodes (only a table that is never evaluated: `EOBIMRTable` without an SPA part): the chord slope and 0.
"""
@inline function _spa_node_wdot(ts, omegas, i, n)
    if n < 7
        j = clamp(i, 1, n - 1)
        return (omegas[j + 1] - omegas[j]) / (ts[j + 1] - ts[j]), 0.0
    end
    j0 = clamp(i - 3, 1, n - 6)
    tc = ts[i]
    x = ntuple(m -> ts[j0 + m - 1] - tc, Val(7))
    d = ntuple(m -> omegas[j0 + m - 1], Val(7))
    d = _spa_ddiff(d, x, Val(1)); d = _spa_ddiff(d, x, Val(2)); d = _spa_ddiff(d, x, Val(3))
    d = _spa_ddiff(d, x, Val(4)); d = _spa_ddiff(d, x, Val(5)); d = _spa_ddiff(d, x, Val(6))
    # Newton form omega(t) = d1 + d2 (t - x1) + d3 (t - x1)(t - x2) + ...: first and second derivative at t = 0
    # (node i) of the products P_k = prod_{m <= k} (t - x_m), accumulated with their coefficients d[k + 1].
    P, P1, P2 = 1.0, 0.0, 0.0
    w1, w2 = 0.0, 0.0
    Base.Cartesian.@nexprs 6 k -> begin
        u = -x[k]
        P2 = P2 * u + 2.0 * P1
        P1 = P1 * u + P
        P = P * u
        w1 += d[k + 1] * P1
        w2 += d[k + 1] * P2
    end
    return w1, w2
end

"""
    _spa_node_q(ts, omegas, R, dR, i, n) -> (Q, dQ)

The SPA amplitude factor `Q = R / sqrt(d omega/dt)` and its time derivative at node `i` ([`_spa_node_wdot`](@ref)), for a
mode amplitude `R` with time derivative `dR` there. Interpolated in `t` (cubic Hermite) between nodes, it replaces
`R(t*) / sqrt(d omega/dt (t*))` in the SPA amplitude. NaN where `omega` is not increasing (no stationary point; the
sampler rejects the point).
"""
@inline function _spa_node_q(ts, omegas, R, dR, i, n)
    wd, wdd = _spa_node_wdot(ts, omegas, i, n)
    iv = wd > 0.0 ? 1.0 / sqrt(wd) : NaN
    return R * iv, dR * iv - 0.5 * R * wdd * iv^3
end

"""
    _seobv5f2_dense_to_freq(ts, phis, omegas, nreal, M_sec, f) -> (i, s, t_star, phi_star)

Generic (model-agnostic) numeric SPA time-to-frequency inversion on a dense, possibly
non-uniformly-spaced, monotonic-in-`omega` trajectory satisfying `dphi/dt = omega` exactly at each
node (`ts`, `phis`, `omegas` reduced/dimensionless, same convention as this package's other
ODE-track models). `f` is the target **physical** GW frequency (Hz); `omega_target =
pi*M_sec*f` (this package's standard reduced-angular-frequency convention, `omega_hat =
pi*M_sec*f` at the `f_min` initial condition -- see `AlignedSpinPNHam`/`SEOBNRv5AlignedSpin`'s own
`_asph_coeffs`/`_seobv5_coeffs`). Returns the bracketing interval `i`, the position `s in [0, 1]` in it and the cubic
Hermite interpolant's exact root ([`_spa_root`](@ref)) with the interpolated orbital phase there.

Caller's responsibility: `f` within `[omegas[1], omegas[nreal]]/(pi*M_sec)` (checked by callers via
`coeffs.fmin`/`coeffs.fmax`, not here -- this is the shared `O(1)`-after-bracket-search primitive).
`nreal` bounds the search to the real (non-padding) region of a fixed-size `coeffs.omegas` buffer
(see [`_seobv5f2_coeffs`](@ref)'s own docstring) -- the padding region (`i>nreal`) is constructed to
be strictly greater than `omegas[nreal]`, so `searchsortedlast` over the FULL buffer never needs
`nreal` to land correctly, but the final `clamp` does, since `length(omegas)` is `NMAX`, not `nreal`.
"""
@inline function _seobv5f2_dense_to_freq(ts::AbstractVector{Float64}, phis::AbstractVector{Float64},
    omegas::AbstractVector{Float64}, nreal::Int, M_sec::Float64, f)
    omega_target = pi * M_sec * f
    i = clamp(_seobv5f2_searchsortedlast(omegas, omega_target), 1, nreal - 1)
    r = _spa_root(ts, phis, omegas, i, omega_target)
    return (i = i, s = r.s, t_star = r.t_star, phi_star = r.phi_star)
end

"""
    _seobv5f2_coeffs_base_pa(model, mc, eta, chi1, chi2; f_min=20.0, NMAX=Val(256),
        GMsun_over_c3=GMsun_over_c3) -> (base, pa)

Stage 1 of [`_seobv5f2_coeffs`](@ref): the genuinely-sequential part (`_seobv5_coeffs`'s scalar
setup, then the post-adiabatic driver's own pass-dependent sweep). Split out so a GPU caller can run
this stage with one thread per walker (its own internal structure has real cross-node dependencies,
unlike stage 2 below) while stage 2 -- independent per grid node -- gets its own, much wider launch.
Pure refactor: `_seobv5f2_coeffs` itself is unchanged in its actual arithmetic, just now expressed as
stage1 -> (per-node stage2) -> stage3.
"""
@inline function _seobv5f2_coeffs_base_pa(model::SEOBNRv5AlignedSpinF2, mc, eta, chi1, chi2;
    f_min = 20.0, NMAX::Val{N} = Val(256), GMsun_over_c3 = GMsun_over_c3) where {N}
    base = _seobv5_coeffs(SEOBNRv5AlignedSpin(), Float64(mc), Float64(eta), Float64(chi1), Float64(chi2);
        f_min = Float64(f_min), GMsun_over_c3 = Float64(GMsun_over_c3))
    pa = seobnrv5alignedspin_postadiabatic_dynamics_fixed(base, NMAX)
    return base, pa
end

"""
    _seobv5f2_omega_at_node(r_i, pr_i, pphi_i, hp) -> Float64

Stage 2 of [`_seobv5f2_coeffs`](@ref), ONE grid node: `Omega = dH/dpphi` at a single post-adiabatic
node. Fully independent across nodes (a pure function of that node's own `(r,pr,pphi)` plus the
walker-level scalars from stage 1) -- the embarrassingly-parallel ~43% (measured on CPU; more on GPU) of `_seobv5f2_coeffs`'s total cost that stage 1's serial
per-walker loop used to pay one node at a time. Extracted verbatim from that loop body, not
re-derived.
"""
@inline function _seobv5f2_omega_at_node(r_i, pr_i, pphi_i, hp)
    _, _, om = _seobv5_sym_gradient(r_i, pr_i, pphi_i, hp)
    return om
end

"""
    _seobv5f2_coeffs_finish(base, pa, omegas_raw::SVector{N,Float64}) -> NamedTuple

Stage 3 of [`_seobv5f2_coeffs`](@ref): given stage 1's `(base, pa)` and stage 2's per-node
`omegas_raw` (however it was computed -- serially or via a separate wide GPU launch), applies the
padding-region masking and assembles the same NamedTuple `_seobv5f2_coeffs` has always returned.
Cheap (`O(N)`, no transcendental calls) -- stays single-threaded-per-walker even in the GPU-split
design below, unlike stage 2.
"""
@inline function _seobv5f2_coeffs_finish(base, pa, omegas_raw::SVector{N,Float64}) where {N}
    n = pa.nreal
    om_n = omegas_raw[clamp(n, 1, N)]
    omegas = zero(MVector{N,Float64})
    @inbounds for i in 1:N
        omegas[i] = ifelse(i <= n, omegas_raw[i], om_n + Float64(i - n))
    end
    omegas = SVector(omegas)
    fmin_reached = n >= 1 ? omegas[1] / (pi * base.M_sec) : NaN
    fmax_reached = n >= 2 ? omegas[clamp(n, 1, N)] / (pi * base.M_sec) : NaN
    return (success = pa.success && n >= 2, M_sec = base.M_sec, eta = base.eta, M = base.M,
        nreal = n, ts = pa.t, phis = pa.phi, omegas = omegas, fmin = fmin_reached, fmax = fmax_reached)
end

"""
    _seobv5f2_coeffs(model::SEOBNRv5AlignedSpinF2, mc, eta, chi1, chi2; f_min=20.0, NMAX=Val(256),
        GMsun_over_c3=GMsun_over_c3) -> NamedTuple

Time-independent (per-walker, once) setup: runs [`_seobv5_coeffs`](@ref) then
[`seobnrv5alignedspin_postadiabatic_dynamics_fixed`](@ref) once, computes `Omega = dH/dpphi` at
EVERY one of the fixed `NMAX` grid nodes (real `1:nreal` and padding `nreal+1:NMAX` alike, uniform
cost across the whole buffer -- same convention as `seobnrv5alignedspin_postadiabatic_dynamics_fixed`
itself) via [`_seobv5_ham_xi`](@ref)/[`_seobv5_gradient`](@ref), and stores the resulting dense
`(t, phi, omega)` tables (`ts`/`phis` reused directly from the PA-fixed driver's own
`SVector{NMAX,Float64}` output, `omegas` built the same way here) plus the frequency band the PA
table actually reaches (`fmin`/`fmax`, physical Hz, from the REAL region `1:nreal` only).

**CUDA-capable fixed-buffer design** (see
`SEOBNRv5AlignedSpinPAFixed.jl`'s PA driver): `ts`/`phis`/`omegas` are
`SVector{NMAX,Float64}`, not runtime-sized `Vector{Float64}` -- the prior `Vector{Float64}(undef, n)`
allocation here was the confirmed, pinpointed CUDA blocker for this model (binned-kernel NUTS itself
has no CUDA blocker; this allocation did). The padding region of `omegas` (`i>nreal`) is NOT the raw
physics value (which can be non-monotonic relative to the real trajectory, since every padding index
shares the identical physically-inert `r=_SEOBV5_PA_R_PAD` state) but is instead forced to
`omegas[nreal] + (i-nreal)`, a strictly increasing continuation -- this lets
[`_seobv5f2_dense_to_freq`](@ref)'s `searchsortedlast` run over the FULL `NMAX`-length buffer
unconditionally (uniform cost, no dynamic-length array, `KernelAbstractions.jl`/CUDA-safe) while
still landing on a real-region bracket for any `f` within `[fmin,fmax]`, exactly as the padding
conventions in `SEOBNRv5AlignedSpinPAFixed.jl` mask their own downstream consumers. `Float64`-only
(mirrors `seobnrv5alignedspin_postadiabatic_dynamics_fixed`'s own `Float64`-only fixed-buffer design
-- see this file's own module docstring's "What this is NOT (yet)"). It is
implemented as stage1 (`_seobv5f2_coeffs_base_pa`) -> stage2 per-node (`_seobv5f2_omega_at_node`) ->
stage3 (`_seobv5f2_coeffs_finish`), split out so a GPU caller can give the embarrassingly-parallel
stage 2 a much wider launch than stages 1/3 need -- pure refactor, same arithmetic.
"""
function _seobv5f2_coeffs(model::SEOBNRv5AlignedSpinF2,
    mc,
    eta,
    chi1,
    chi2;
    f_min = 20.0,
    NMAX::Val{N} = Val(256),
    GMsun_over_c3 = GMsun_over_c3,
) where {N}
    base, pa = _seobv5f2_coeffs_base_pa(model, mc, eta, chi1, chi2; f_min, NMAX, GMsun_over_c3)
    # Padding entries (i > nreal) are overwritten by `_seobv5f2_coeffs_finish`, so skip them
    # (constant trip count kept; see the PA driver's `skip_padding`).
    omegas_raw = zero(MVector{N,Float64})
    @inbounds for i in 1:N
        i > pa.nreal && continue
        omegas_raw[i] = _seobv5f2_omega_at_node(pa.r[i], pa.pr[i], pa.pphi[i], base.hp)
    end
    return _seobv5f2_coeffs_finish(base, pa, SVector(omegas_raw))
end

"""
    _seobv5f2_phi_scalar(coeffs, f) -> Float64

Non-allocating (given `coeffs`) frequency-domain SPA phase at a single frequency `f` (Hz), `0`
outside `[coeffs.fmin, coeffs.fmax]` (same hard-cutoff convention as `AlignedSpinPNHamF2`'s `fCut`).
"""
@inline function _seobv5f2_phi_scalar(coeffs, f)
    !coeffs.success && return 0.0
    (f < coeffs.fmin || f > coeffs.fmax) && return 0.0
    r = _seobv5f2_dense_to_freq(coeffs.ts, coeffs.phis, coeffs.omegas, coeffs.nreal, coeffs.M_sec, f)
    t_star_phys = r.t_star * coeffs.M_sec
    Phi_TD = 2.0 * r.phi_star
    return 2.0 * pi * f * t_star_phys - Phi_TD - pi / 4.0
end

"""
    _seobv5f2_ampl_scalar(coeffs, f, dL, GMsun_over_c2_Gpc) -> Float64

Non-allocating restricted (Newtonian-order envelope, full-dynamics Jacobian) frequency-domain SPA
amplitude at a single frequency `f` (Hz), `0` outside `[coeffs.fmin, coeffs.fmax]`. Unlike
`AlignedSpinPNHamF2`/`TaylorF2`'s amplitude (which uses the analytic *Newtonian-order*
`dOmega/dt` baked into their closed-form derivation), this uses the model's own true
post-adiabatic-dynamics `dOmega/dt` (nodal values, [`_spa_node_q`](@ref), interpolated to the
stationary point of [`_seobv5f2_dense_to_freq`](@ref)) -- a more accurate SPA Jacobian than the restricted-PN convention, validated to
reproduce it exactly when the same substitution is made (see this file's module docstring).
"""
@inline function _seobv5f2_ampl_scalar(coeffs, f, dL, GMsun_over_c2_Gpc)
    !coeffs.success && return 0.0
    (f < coeffs.fmin || f > coeffs.fmax) && return 0.0
    ts, oms, n = coeffs.ts, coeffs.omegas, coeffs.nreal
    r = _seobv5f2_dense_to_freq(ts, coeffs.phis, oms, n, coeffs.M_sec, f)
    i = r.i
    # 1/sqrt(d omega/dt) from the nodal values at both interval ends (computed here, not stored: the F2 table and its
    # hand-written JVP/VJP stay (ts, phis, omegas)); C1 in f and stable under PA regridding.
    q0, dq0 = _spa_node_q(ts, oms, 1.0, 0.0, i, n)
    q1, dq1 = _spa_node_q(ts, oms, 1.0, 0.0, i + 1, n)
    q, _ = _eobimr_hermite(q0, dq0, q1, dq1, ts[i + 1] - ts[i], r.s)
    v = cbrt(pi * coeffs.M_sec * f)
    Ampl_TD = 4.0 * coeffs.eta * coeffs.M * GMsun_over_c2_Gpc / dL * v * v
    # 0.5 Ampl_TD sqrt(2 pi / (d Omega_GW/dt)_phys) with (d Omega_GW/dt)_phys = 2 (d omega/dt) / M_sec^2
    return 0.5 * Ampl_TD * coeffs.M_sec * sqrt(pi) * q
end

"""
    _seobv5f2_amp_phi(coeffs, f, dL, GMsun_over_c2_Gpc) -> (amplitude, phase)

Both scalars at once, for likelihood kernels. For the PA-F2 tables this is just the two scalar calls;
table types whose evaluation is shared between amplitude and phase (e.g. `EOBIMRTable`) add a
method that evaluates once.
"""
@inline _seobv5f2_amp_phi(coeffs, f, dL, GMsun_over_c2_Gpc) =
    (_seobv5f2_ampl_scalar(coeffs, f, dL, GMsun_over_c2_Gpc), _seobv5f2_phi_scalar(coeffs, f))

"""
    Phi(model::SEOBNRv5AlignedSpinF2, f, mc, eta, chi1, chi2; f_min=20.0, NMAX=Val(256),
        t0=0.0, phiRef=0.0, GMsun_over_c3=GMsun_over_c3)

Frequency-domain SPA phase, `array-in/array-out` (scalar `f` -> scalar), same `t0`/`phiRef`
external-alignment convention as [`Phi(::AlignedSpinPNHamF2, ...)`](@ref)/[`Phi(::TaylorF2,
...)`](@ref) (`phis_out = phis - t0*f - phiRef`). See this file's module docstring for the
reconstruction method, validation, and current (CPU-only, non-Enzyme, inspiral-only) scope.
"""
function Phi(model::SEOBNRv5AlignedSpinF2,
    f,
    mc,
    eta,
    chi1,
    chi2;
    f_min = 20.0,
    NMAX::Val = Val(256),
    t0 = 0.0,
    phiRef = 0.0,
    GMsun_over_c3 = GMsun_over_c3,
)
    coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2; f_min = f_min, NMAX = NMAX,
        GMsun_over_c3 = GMsun_over_c3)
    t0 = Float64(t0)
    phiRef = Float64(phiRef)
    if f isa AbstractArray
        out = similar(f, Float64)
        @inbounds for i in eachindex(f)
            out[i] = _seobv5f2_phi_scalar(coeffs, f[i]) - t0 * f[i] - phiRef
        end
        return out
    else
        return _seobv5f2_phi_scalar(coeffs, f) - t0 * f - phiRef
    end
end

"""
    Ampl(model::SEOBNRv5AlignedSpinF2, f, mc, eta, chi1, chi2, dL; f_min=20.0, NMAX=Val(256),
        GMsun_over_c3=GMsun_over_c3, GMsun_over_c2_Gpc=GMsun_over_c2_Gpc)

Frequency-domain SPA amplitude, same `array-in/array-out` contract as [`Phi`](@ref). See this
file's module docstring for the reconstruction method, validation, and current scope.
"""
function Ampl(model::SEOBNRv5AlignedSpinF2,
    f,
    mc,
    eta,
    chi1,
    chi2,
    dL;
    f_min = 20.0,
    NMAX::Val = Val(256),
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    coeffs = _seobv5f2_coeffs(model, mc, eta, chi1, chi2; f_min = f_min, NMAX = NMAX,
        GMsun_over_c3 = GMsun_over_c3)
    dL = Float64(dL)
    GMsun_over_c2_Gpc = Float64(GMsun_over_c2_Gpc)
    if f isa AbstractArray
        out = similar(f, Float64)
        @inbounds for i in eachindex(f)
            out[i] = _seobv5f2_ampl_scalar(coeffs, f[i], dL, GMsun_over_c2_Gpc)
        end
        return out
    else
        return _seobv5f2_ampl_scalar(coeffs, f, dL, GMsun_over_c2_Gpc)
    end
end
