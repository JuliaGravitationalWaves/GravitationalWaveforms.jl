using StaticArrays: FieldVector

export AlignedSpinPNHam, AlignedSpinPNHamState, validate_alignedspinpnham_params,
    alignedspinpnham_rk4_integrate

"""
    AlignedSpinPNHam

The first
*physically useful* ODE-integrated model in this package: a spinning, non-precessing
(aligned-spin) post-Newtonian inspiral, evolved as a genuine 4-variable Hamiltonian dynamical
system `(r, phi, p_r, p_phi)` (separation, orbital phase, radial momentum, orbital angular
momentum), rather than `TaylorT3`'s closed-form `theta`-parametrized phasing or `ODEToy`'s
2-variable `(phase, omega)` toy. Architecture mirrors `ODEToy.jl` exactly (`StaticArrays`
`FieldVector` state, allocation-free analytic RHS, fixed-step RK4, `_coeffs`/`_scalar` split) --
see that file's docstring for the rationale (Enzyme/KernelAbstractions readiness).

# Physics and provenance -- what was ported from where, and why

No single LALSuite source file gives an aligned-spin PN system in exactly this `(r,p_r,phi,p_phi)`
Hamiltonian form: LALSuite's own `TaylorT4`-family aligned-spin dynamics
(`lalsimulation/lib/LALSimInspiralSpinTaylor.c`, `XLALSimInspiralSpinTaylorT4DerivativesAvg`) is a
2-variable `(phase, omega)` energy-balance ODE (like this package's `ODEToy`, just PN-complete and
spinning), not a phase-space Hamiltonian -- confirmed directly by reading that function (see below).
The **conservative** dynamics here is
therefore built from a standard ADM-like PN Hamiltonian cited directly from the literature, and the
**dissipative** (radiation-reaction) dynamics is ported verbatim from LALSuite's own aligned-spin PN
energy-flux coefficient tables -- each piece's provenance is documented separately below, since they
come from different places and were cross-validated against each other (not just asserted
consistent).

## Conservative Hamiltonian `H(r, p_r, p_phi)`

Reduced/dimensionless ("hat") units throughout, standard ADM/ EOB-literature convention:
`r_hat = R/M`, `p_r_hat = P_R/mu`, `p_phi_hat = P_phi/(mu*M)`, `t_hat = t/M` (`M = m1+m2` total
mass, `mu = m1*m2/M` reduced mass, `eta = mu/M` symmetric mass ratio, both in seconds via
`GMsun_over_c3`). `H_hat = H/mu` is what this file calls `H` below.

```
H = H_N + H_1PN + H_SO + H_SS
H_N   = p_r^2/2 + p_phi^2/(2 r^2) - 1/r
H_1PN = (1/8)(3 eta - 1) p^4 - (1/2)[(3+eta) p^2 + eta p_r^2]/r + 1/(2 r^2),   p^2 = p_r^2 + p_phi^2/r^2
H_SO  = -(1/5) (p_phi/r^3) * [ E3S1O(X1) X1^2 chi1 + E3S1O(X2) X2^2 chi2 ]
H_SS  = (1/(3 eta r^3)) * (X1^2 chi1)(X2^2 chi2)
```
with `X1 = m1/M`, `X2 = m2/M` (`m1 >= m2` by this package's usual convention), `chi1`/`chi2` the
dimensionless spins along the (fixed, aligned) orbital angular momentum direction, and
`E3S1O(x) = 2/3 + 2/x`.

`H_N + H_1PN` is the textbook Newtonian+1PN two-body ADM Hamiltonian (e.g. Damour, Jaranowski &
Schafer, Phys. Rev. D 62, 084011 (2000); Buonanno & Damour, Phys. Rev. D 59, 084006 (1999) eq. 8) --
**independently re-derived and confirmed here** (not just recalled) by computing its own
circular-orbit binding energy `E(v)` in closed form (perturbative solve of the circular-orbit
condition `dH/dr_hat = 0`, `p_r_hat = 0`) and checking it reproduces `TaylorT3.jl`'s own
already-LALSuite-validated 1PN energy coefficient `dETa1 = -(1.5 + eta/6)` *exactly*
(`E(v) = -eta*v^2/2*(1 + (dETa1/2)*v^2 + ...)`, confirmed to machine precision for several `eta`
values -- see `test/alignedspinpnham_tests.jl`'s `"1PN Hamiltonian reproduces TaylorT3's dETa1"`
testset).

`H_SO`/`H_SS`'s **functional form** (`p_phi/r^3` radial falloff for spin-orbit, `1/r^3` for
spin-spin) is the standard leading-order structure common to every PN spin-coupling reference
(Barker & O'Connell 1975; Kidder, Phys. Rev. D 52, 821 (1995), eqs. 2.2/3.3). Rather than trust a
recalled numeric prefactor for these terms (exactly the failure mode this codebase's whole
development discipline exists to catch), **the numeric coefficients above were derived, not
recalled**: they are fixed by requiring this Hamiltonian's own circular-orbit energy reproduce, at
leading PN order, LALSuite's own `LALSimInspiralPNCoefficients.c` aligned-spin energy coefficients
`XLALSimInspiralPNEnergy_3PNSOCoeff` (`= E3S1O` above, cited there to eq. 4.6 of Bohe, Marsat, Faye &
Blanchet, arXiv:1212.5520) and `XLALSimInspiralPNEnergy_4PNS1S2Coeff` (`= -2/eta`, cited to
astro-ph/0504538v2 eq. 6) via the identical perturbative circular-orbit technique used for the 1PN
check above (worked derivation: `eta*H_circ(v)` must equal LALSuite's `E(v)` series' `v^5`/`v^6`
spin terms; solving that equation for the overall `H_SO`/`H_SS` prefactor gives exactly `-1/5` and
`1/(3 eta)` respectively -- see this module's own derivation notes and
`test/alignedspinpnham_tests.jl` for the numerical confirmation). LALSuite's own internal
`SpinTaylorT4`/`SpinTaylorT1` energy-array bookkeeping (`Ecoeff`/`Espin3`/`Espin4` in
`XLALSimInspiralSetEnergyPNTermsAvg`) was independently confirmed to use the *identical* raw
coefficient functions for its non-spin terms (`Ecoeff[2] === XLALSimInspiralPNEnergy_2PNCoeff(eta)`,
confirmed by direct comparison against `TaylorT3.jl`'s own `dETa1`), which is what justifies
plugging `E3S1O`/`4PNS1S2Coeff` into the *same* PN-order slot in this file's circular-orbit energy
series as `TaylorT3.jl`'s `dETa1`/`dETa2`.

**Explicit scope limit**: the conservative Hamiltonian carries spin only at *leading* PN order
(1.5PN spin-orbit, 2PN spin-spin) -- not the next-to-leading/next-to-next-to-leading spin terms
LALSuite's own `SpinTaylorT4` energy tables also have available (`XLALSimInspiralPNEnergy_5PNSOCoeff`
[2.5PN], `_7PNSOCoeff` [3.5PN], `_6PNS1S2Coeff` [3PN]). This is a deliberate, documented scope
decision, not an oversight -- a later version could
extend `H_SO`/`H_SS` with those terms using the exact same derivation method demonstrated here.

## Dissipative flux `F(v)` (radiation reaction)

Ported verbatim from `LALSimInspiralPNCoefficients.c`, extending `TaylorT3.jl`'s own already-
LALSuite-validated non-spin flux (`FTaN, FTa2..FTa7, FTl6`, identical coefficients, same citation:
Buonanno, Iyer, Ochsner, Pan & Sathyaprakash, Phys. Rev. D 80, 084043 (2009), arXiv:0907.0700 eq.
3.2) with the aligned-spin flux corrections from the same file's `XLALSimInspiralPNFlux_*SOCoeff`
(spin-orbit, cited there to eq. 3.13 of Bohe, Faye, Marsat & Porter, arXiv:1303.7412 [1.5PN term]
and eq. 4.9 of Marsat, Bohe, Blanchet & Buonanno, Class. Quant. Grav. 31, 025023 (2014),
arXiv:1307.6793 [2.5PN, 3PN-tail, 3.5PN terms]) and `XLALSimInspiralPNFlux_4PNS1S2Coeff`/
`_6PNS1S2Coeff` (spin-spin, cited to astro-ph/0504538v2 and eq. 4.12 of arXiv:1501.01529
respectively). Full PN order carried through **3.5PN spin-orbit + 3PN spin-spin** -- deliberately
*higher* spin order than the conservative Hamiltonian above, exactly mirroring how
`TaylorT4`/`TaylorT1`-family energy-balance models in LALSuite itself routinely carry the flux to
higher PN order than a simplified conservative sector, and how `TaylorT3.jl` itself already treats
its (non-spin) energy/flux asymmetrically (3PN energy feeding a 3.5PN flux, per that file's own
docstring).

For general (non-aligned) spins, LALSuite's flux/energy coefficient tables have *three* independent
spin-spin structures per PN order (`S1.S2`, `(S1.n)(S2.n)`, `(S1.v)(S2.v)`, `n`/`v` an instantaneous
orbital-plane triad) -- for **exactly aligned** spins (`S1`, `S2` both parallel to the fixed orbital
angular momentum direction, hence exactly perpendicular to `n`/`v` at every instant, not just on
orbit average), `(S1.n) = (S2.n) = (S1.v) = (S2.v) = 0` identically, leaving only the plain `S1.S2`
term in each case -- confirmed directly against the LALSuite source structure, not assumed; see
`_asph_flux`'s implementation comments for exactly which `XLALSimInspiralPNFlux_*` function is used
at each order.

## Radiation-reaction equation of motion -- dimensional derivation

Standard adiabatic energy/angular-momentum balance (`dE/dt = -F`, `dL/dt = -F/Omega`, both physical,
`F >= 0` the GW luminosity) reduces, in the hat-unit convention above, to (worked in this module's
own derivation notes, re-derived independently rather than assumed since the two momenta have
*different* mass-normalization conventions -- `p_phi_hat` is reduced by `mu*M`, `E_hat` by `M` alone
-- so the naive guess `dp_phi_hat/dt_hat = -F/Omega_hat` is dimensionally wrong by a factor of
`eta`):

```
dp_phi_hat/dt_hat = -F(v) / (eta * Omega_hat),   Omega_hat = dH/dp_phi_hat,   v = Omega_hat^(1/3)
```

No radial radiation-reaction term is included (`dp_r_hat/dt_hat = -dH/dr_hat` only, purely
conservative) -- the standard adiabatic/quasi-circular approximation used throughout the EOB/PN
literature (e.g. Buonanno & Damour 1999 sec. II explicitly sets the radial RR force to zero), valid
because the dominant angular-momentum-loss torque circularizes the orbit far faster than any radial
RR correction would matter at the PN order considered here.

# Initial conditions

`Phi`/`Ampl` (below) take a reference frequency `f_min`/`f_ref`, converted to the standard PN
velocity parameter `v0 = cbrt(pi * M_sec * f_min)` (matching `TaylorT3.jl`'s own `v` convention
exactly). The initial `(r_hat, p_phi_hat)` (`p_r_hat,0 = 0`) is the **exact** circular-orbit
solution of *this Hamiltonian itself* (`dH/dr_hat = 0` at `p_r_hat = 0`, `dH/dp_phi_hat = v0^3`),
found by a **fixed** (8-iteration, no early-exit) 2D Newton solve seeded from the non-spin 1PN
closed-form estimate (`x0 = v0^2*(1+(3-eta)*v0^2/3)`, `r_hat = 1/x0`,
`p_phi_hat = x0^(-1/2)*(1+2*x0)`) -- fixed iteration count and no dynamic branching keeps this
Enzyme/`@kernel`-style-compatible (see `AlignedSpinPNHam`'s architecture note), unlike a
convergence-tolerance-based solve.

**Why a solve, not just the closed form**: an earlier version of this code used the non-spin 1PN
closed-form estimate directly as the initial condition. Checked directly against a fully-converged
circular solve, that estimate has a **non-negligible, confirmed error that grows quickly with
`v0`** -- e.g. at `v0 ~ 0.25` (`f_min = 20` Hz for a `~55 Msun` total-mass binary, an entirely
ordinary configuration for this ecosystem's own baseline parameters), `r_hat`/`p_phi_hat` are off
by `~3%`/`~4-6%`, shrinking rapidly for smaller `v0` (confirmed to scale down by ~16x from `v0=0.1`
to `v0=0.05`, consistent with the expected next-PN-order suppression) -- large enough to
meaningfully bias the very first several orbits of a "circular-orbit" trajectory that's supposed
to start exactly circular. The Newton solve removes this source of error at the one-time cost of a
handful of extra flops before integration begins (not inside the per-step RK4 loop).

The solve is for the **non-spin-plus-leading-spin** Hamiltonian in full (using the actual
`chi1`/`chi2` for this waveform, not the non-spin closed form's own approximation) -- so, unlike
the earlier closed-form-only approach, spinning configurations now also start on their own true
circular orbit for *this* Hamiltonian (to the fixed solve's own convergence, not exactly the
physical spin-corrected equilibrium of the real two-body problem, which this Hamiltonian only
approximates at leading PN spin order to begin with).

# Spin-direction finding: resolved -- a genuine, verified feature of this truncation, not a bug

One result here looks surprising at first: at fixed elapsed time since `f_min`, this model's
aligned-spin configuration was often found to reach a *higher* frequency than anti-aligned --
opposite the commonly-quoted qualitative "aligned spin delays merger"/"hang-up" picture (Campanelli
et al., Phys. Rev. Lett. 96, 111101 (2006)). Chased down to an actual resolution (not just
re-asserted) by directly instrumenting the running `_asph_hamiltonian_partials`/`_asph_flux`/
`alignedspinpnham_rk4_integrate` code -- see `test/alignedspinpnham_tests.jl`'s `"Spin-direction
mechanism"` testset for the checks this section describes, re-run as a locked-in regression.

**Both underlying physical mechanisms are individually correct**, confirmed by direct evaluation of
the running code (not just re-derived on paper):
- At *fixed* `(r, p_r=0, p_phi)`, `H_SO`'s contribution to `Omega = dH/dp_phi` is strictly *more
  negative* for aligned spin than anti-aligned (`E3S1O(x) = 2/3+2/x > 0` for `x in (0,1)`, so
  `SOcoef` has the same sign as `chi1`/`chi2`, and `dH_SO/dp_phi = -(1/5)*SOcoef/r^3` is then more
  negative for aligned) -- e.g. `eta=0.248`, `r=20`, `p_phi=sqrt(20)`: `chi1=chi2=+0.5` gives
  `Omega=9.264e-3` vs. `chi1=chi2=-0.5`'s `9.322e-3` -- the textbook hang-up direction.
- At fixed `v`, `F(v)`'s spin-orbit correction is strictly *higher* for anti-aligned than aligned
  (`f3SO(x) = -3/2-5/(4x) < 0` always, so anti-aligned's negative `S1`/`S2` make `FSO3` positive) --
  e.g. `eta=0.248`, `v=0.2`: `chi1=chi2=+0.5` gives `F=3.648e-8` vs. `chi1=chi2=-0.5`'s `3.715e-8` --
  anti-aligned loses angular momentum faster, again the textbook direction.

**So where does the reversal come from?** Not from either coefficient's sign -- from how this
model's "start every spin configuration on its own true circular orbit at the same `f_min`"
initial-condition procedure (above) interacts with a genuinely leading-PN-order-only spin sector.
Matching a *common target orbital frequency* `Omega_hat = v0^3` across spin configurations is
standard practice (frequency, not radius, is the physically meaningful/observable matching
quantity for a waveform template), but it forces aligned and anti-aligned configurations onto
*different* circular orbits (`r0`, `p_phi,0`) to begin with -- confirmed directly, e.g. baseline
system (`mc=23.82 Msun`, `eta=0.248`, `f_min=20Hz`, `chi1=chi2=+/-0.5`): `r0=14.237` (aligned) vs.
`14.166` (anti-aligned). Since `Omega` is identically `Omega_(Newtonian+1PN) + Omega_direct-spin`,
and both configurations are *constructed* to share the exact same total `Omega` at `t=0`, the
"orbit-shape" piece (`Omega_(Newtonian+1PN)`, evaluated on each configuration's own, already-
different, `(r0, p_phi,0)`) and the "direct-spin" piece exactly cancel at `t=0` -- not a
coincidence, an algebraic identity of the matching procedure, confirmed numerically to the Newton
solve's own convergence precision (~11 significant figures).

This cancellation is **not dynamically preserved** as the trajectory integrates forward -- confirmed
by decomposing `Omega(t)` along the *actual* integrated `(r(t), p_phi(t))` trajectory into these
same two pieces at each `t` (evaluating `_asph_hamiltonian_partials` with `SOcoef=SScoef=0` on that
same trajectory to isolate the "orbit-shape" piece). The "orbit-shape" piece grows *faster* in time
than the "direct-spin" piece (baseline system, `chi1=chi2=+/-0.5`, by `t=0.4s`: orbit-shape
difference `~3.86e-3` vs. direct-spin difference `~3.7e-4`, an order of magnitude apart -- the
direct-spin piece stays hang-up-consistent in sign the whole time, exactly as the fixed-point
argument above predicts). The larger, faster-growing orbit-shape effect dominates the sum, so the
*net*, observable frequency difference flips away from the textbook direction once the initial
cancellation unwinds.

**Confirmed via direct experimentation, not just this one case**:
- The reversal is a genuine *time-dependent, non-monotonic* effect, not single-signed: for the
  baseline system, `Omega_aligned - Omega_antialigned` is positive (opposite hang-up) for
  `t < ~0.015s`, briefly *negative* (textbook-consistent) for `t ~ 0.02s-0.08s`, then positive
  (opposite hang-up) again and *growing* for the remainder of the model's valid integration range
  (`t ~ 0.1s` onward, up to the PN-Hamiltonian breakdown -- found directly at `t ~ 0.44-0.5s` for
  this spin magnitude, later than the `t ~ 0.07-0.075s` figure quoted in
  `test/alignedspinpnham_tests.jl`'s non-spin/mild-spin baseline, since larger aligned spin here
  hastens its own breakdown).
- Scales close to linearly with spin magnitude, as expected for a leading-order-in-`chi` effect:
  `chi = 0.1, 0.5, 0.9` all show the identical three-regime qualitative pattern, with the late-time
  magnitude scaling within ~1% of linear in `chi`.
- Not an artifact of the equal, symmetric `chi1=chi2` choice above: confirmed with the asymmetric
  `(chi1,chi2)=(0.3,-0.2)` vs. its mirror `(-0.3,0.2)` too.
- Every coefficient used (`E3S1O`, `4PNS1S2Coeff`, the flux `*SOCoeff`/`*S1S2Coeff` functions) was
  re-checked directly against the actual `LALSimInspiralPNCoefficients.c` source (not just
  recalled) and matches this file's own formulas exactly, sign included -- ruling out a simple
  transcription bug as the cause.

**Bottom line**: this is a genuine, checked feature of combining a *leading-PN-order-only*
conservative spin-orbit/spin-spin Hamiltonian with a common-target-frequency initial condition --
not a sign or derivation bug in either the conservative or dissipative sector. Adding
the next-order (2.5PN/3PN) conservative spin terms LALSuite's own tables already have available
(see "Explicit scope limit" above) could plausibly shift this balance back toward the textbook
direction.

# Domain convention: time since the reference frequency, not time-to-coalescence

Unlike `TaylorT3` (`t <= 0`, coalescence at `t = 0`, matching LALSuite's `choose_td_waveform`
convention), this ODE-integrated model has no closed-form coalescence reference to align against
(and, being a truncated PN model, may become unphysical well before any literal `r -> 0`). `Phi`/
`Ampl` here instead take `t >= 0`, seconds **since** the `f_min`-referenced initial condition,
matching `ODEToy`'s own "step count since the initial condition" convention, just in physical time
units instead of step count. Each requested `t` is integrated independently from the shared initial
condition (`t=0` state), exactly mirroring `ODEToy`'s and `TaylorT3`'s own per-sample-independent
evaluation design.

**Step-quantization note**: each requested `t` is rounded to the nearest multiple of `dt`
(`nsteps = round(Int, t/dt)`) before integrating, so two requested times closer together than a
few `dt` can land on unevenly-spaced actual step counts -- harmless to `Phi`/`Ampl`'s own accuracy
at any single `t`, but a naive finite-difference frequency estimate (`diff(Phi.(t))./diff(t)`)
across *closely*-spaced samples can show a small spurious artifact from this rounding. Request `t`
values spaced several `dt` apart (or larger) to avoid it -- found directly during testing (see
`test/alignedspinpnham_tests.jl`).

# Phase convention

GW phase is `2x` orbital phase (`Phi = 2*phi`), matching every other model in this package.

# Frequency-domain SPA sibling: [`AlignedSpinPNHamF2`](@ref)

`Phi`/`Ampl` here only take **time** and re-integrate the ODE from scratch on every call -- not
usable for `GravitationalWaveBayes.jl`'s relative-binning likelihood, which needs an `O(1)`-per-frequency-query
closed form. [`AlignedSpinPNHamF2`](@ref) is the closed-form, frequency-domain SPA sibling of
*this* model -- the same relationship [`TaylorF2`](@ref) already has to [`TaylorT3`](@ref) --
built from this model's own `H_SO`/`H_SS`/flux content via the standard energy-balance
construction; see that file's docstring for the full derivation, including a genuine, documented
discrepancy found (and not silently resolved) between a from-scratch reconstruction of the
leading spin-orbit/spin-spin phasing from this file's raw Hamiltonian/flux and the published,
LALSuite-validated `TaylorF2` spin phasing coefficients that model ultimately reuses.
"""
struct AlignedSpinPNHam end

"""
    AlignedSpinPNHamState{T} <: FieldVector{4, T}

The `AlignedSpinPNHam` ODE state: reduced (dimensionless) orbital separation `r`, orbital phase
`phi`, reduced radial momentum `pr`, and reduced orbital angular momentum `pphi` -- see
`AlignedSpinPNHam`'s docstring for the exact hat-unit convention. A `StaticArrays.jl` `FieldVector`
(not a plain `Array`), mirroring `ODEToyState` exactly -- see that struct's docstring for why this
keeps the RK4 loop allocation-free.
"""
struct AlignedSpinPNHamState{T} <: FieldVector{4, T}
    r::T
    phi::T
    pr::T
    pphi::T
end

"""
    validate_alignedspinpnham_params(eta, chi1, chi2; dL = nothing)

Host-side sanity check: `0 < eta <= 0.25`, `-1 <= chi1,chi2 <= 1` (dimensionless spins), and (if
`dL` given) `dL > 0`. Throws `ArgumentError` on violation, `nothing` otherwise. Not called by
`Phi`/`Ampl` themselves, matching `validate_taylort3_params`'s documented rationale (hot-path;
one-time domain checks belong at the sampler/prior level).
"""
function validate_alignedspinpnham_params(eta, chi1, chi2; dL = nothing)
    0.0 < eta <= 0.25 || throw(ArgumentError("Invalid eta = $eta: must be in (0, 0.25]"))
    -1.0 <= chi1 <= 1.0 || throw(ArgumentError("Invalid chi1 = $chi1: spin must be in [-1, 1]"))
    -1.0 <= chi2 <= 1.0 || throw(ArgumentError("Invalid chi2 = $chi2: spin must be in [-1, 1]"))
    if dL !== nothing
        dL > 0.0 || throw(ArgumentError("Invalid dL = $dL: luminosity distance must be positive (and must be in Gpc, see Ampl's docstring)"))
    end
    return nothing
end

# --- Flux F(v): verbatim non-spin coefficients from TaylorT3.jl (same LALSuite source/citation),
# extended with aligned-spin corrections -- see AlignedSpinPNHam's docstring for full citations. ---

"""
    _asph_flux_coeffs(eta, X1, X2, chi1, chi2, LAL_GAMMA) -> NamedTuple

Precomputes the (eta, X1, X2, chi1, chi2)-dependent flux series coefficients once per waveform
(mirroring `TaylorT3.jl`'s `_taylort3_coeffs` pattern) -- see [`_asph_flux`](@ref) for how they
combine into `F(v)`.
"""
@inline function _asph_flux_coeffs(eta::Float64, X1::Float64, X2::Float64, chi1::Float64, chi2::Float64, LAL_GAMMA::Float64)
    S1 = X1 * X1 * chi1   # reduced spin S1/M^2 = (m1/M)^2 * chi1
    S2 = X2 * X2 * chi2
    S1S2 = S1 * S2

    # Non-spin, verbatim from TaylorT3.jl's _taylort3_flux (same LALSuite source/citation).
    FTaN = 32.0 * eta * eta / 5.0
    FTa2 = -(12.47 / 3.36 + 3.5 / 1.2 * eta)
    FTa3 = 4.0 * pi
    FTa4 = -(44.711 / 9.072 - 92.71 / 5.04 * eta - 6.5 / 1.8 * eta * eta)
    FTa5 = -(81.91 / 6.72 + 58.3 / 2.4 * eta) * pi
    FTa6 = 664.3739519 / 6.9854400 + 16.0 / 3.0 * pi * pi - 17.12 / 1.05 * LAL_GAMMA -
           17.12 / 1.05 * log(4.0) + (4.1 / 4.8 * pi * pi - 134.543 / 7.776) * eta -
           94.403 / 3.024 * eta * eta - 7.75 / 3.24 * eta * eta * eta
    FTl6 = -17.12 / 1.05
    FTa7 = -(162.85 / 5.04 - 214.745 / 1.728 * eta - 193.385 / 3.024 * eta * eta) * pi

    # Spin-orbit: XLALSimInspiralPNFlux_{3,5,6,7}PNSOCoeff(mByM), aligned so LNhdotS_i = S_i exactly.
    f3SO(x) = -3.0 / 2.0 - 5.0 / (4.0 * x)
    f5SO(x) = 63.0 / 8.0 - 13.0 / (16.0 * x) - 73.0 * x / 36.0 - 157.0 * x * x / 18.0
    f6SO(x) = pi * (-17.0 / 3.0 - 31.0 / (6.0 * x))
    f7SO(x) = 380.647 / 13.608 + 95.35 / (3.36 * x) - 401.15 * x / 7.56 +
              3742.0 * x * x / 63.0 - 35.0 * x * x * x / 108.0 - 1117.0 * x * x * x * x / 54.0

    FSO3 = f3SO(X1) * S1 + f3SO(X2) * S2      # 1.5PN
    FSO5 = f5SO(X1) * S1 + f5SO(X2) * S2      # 2.5PN
    FSO6 = f6SO(X1) * S1 + f6SO(X2) * S2      # 3PN (tail)
    FSO7 = f7SO(X1) * S1 + f7SO(X2) * S2      # 3.5PN

    # Spin-spin (aligned: only the plain S1.S2 term survives -- (S1.n)=(S2.n)=(S1.v)=(S2.v)=0
    # identically for spins exactly parallel to the fixed orbital angular momentum direction).
    FSS4 = (31.0 / 8.0 / eta) * S1S2                        # 2PN: XLALSimInspiralPNFlux_4PNS1S2Coeff
    FSS6 = (-2.9 / 16.8 / eta - 25.9 / 1.2) * S1S2           # 3PN: XLALSimInspiralPNFlux_6PNS1S2Coeff

    return (FTaN = FTaN, FTa2 = FTa2, FTa3 = FTa3 + FSO3, FTa4 = FTa4 + FSS4, FTa5 = FTa5 + FSO5,
        FTa6 = FTa6 + FSO6 + FSS6, FTl6 = FTl6, FTa7 = FTa7 + FSO7)
end

"""
    _asph_flux(v, fc)

GW luminosity `F(v)` (dimensionless, geometrized units -- see `AlignedSpinPNHam`'s docstring's
"radiation-reaction equation of motion" section for why no extra mass factors are needed), given
the precomputed coefficients `fc` from [`_asph_flux_coeffs`](@ref). Same series structure as
`TaylorT3.jl`'s `_taylort3_flux`, extended with the spin terms already folded into `fc` above.
"""
@inline function _asph_flux(v::T, fc) where {T}
@Tlit T begin
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v4 * v2
    v7 = v6 * v
    v10 = v6 * v4
    return fc.FTaN * v10 * (1.0 + fc.FTa2 * v2 + fc.FTa3 * v3 + fc.FTa4 * v4 + fc.FTa5 * v5 +
                             (fc.FTa6 + fc.FTl6 * log(v)) * v6 + fc.FTa7 * v7)
end # @Tlit T
end

"""
    _asph_hamiltonian_partials(r, pr, pphi, eta, SOcoef, SScoef) -> (dHdr, dHdpr, dHdpphi)

Analytic partial derivatives of the conservative Hamiltonian `H = H_N + H_1PN + H_SO + H_SS` (see
`AlignedSpinPNHam`'s docstring for the closed form and its derivation/provenance) -- computed by
hand differentiation, not autodiff, so that this function itself has no dependency on Enzyme/
ForwardDiff and stays a plain, allocation-free scalar function (matching `ODEToy`'s
`_odetoy_rhs` precedent). `SOcoef = E3S1O(X1)*X1^2*chi1 + E3S1O(X2)*X2^2*chi2`,
`SScoef = (X1^2*chi1)*(X2^2*chi2)`, both precomputed once per waveform in [`_asph_coeffs`](@ref).
"""
@inline function _asph_hamiltonian_partials(r::T, pr::T, pphi::T, eta::T, SOcoef::T, SScoef::T) where {T}
@Tlit T begin
    r2 = r * r
    r3 = r2 * r
    r4 = r3 * r
    p2 = pr * pr + pphi * pphi / r2
    dp2dr = -2.0 * pphi * pphi / r3

    # H_N
    dHNdr = -pphi * pphi / r3 + 1.0 / r2
    dHNdpr = pr
    dHNdpphi = pphi / r2

    # H_1PN = (1/8)(3eta-1) p2^2 - (1/2)[(3+eta) p2 + eta pr^2]/r + 1/(2 r^2)
    dH1PNdr = 0.25 * (3.0 * eta - 1.0) * p2 * dp2dr -
              0.5 * (3.0 + eta) * dp2dr / r + 0.5 * (3.0 + eta) * p2 / r2 +
              0.5 * eta * pr * pr / r2 - 1.0 / r3
    dH1PNdpr = 0.5 * (3.0 * eta - 1.0) * p2 * pr - (3.0 + 2.0 * eta) * pr / r
    dH1PNdpphi = 0.5 * (3.0 * eta - 1.0) * p2 * pphi / r2 - (3.0 + eta) * pphi / r3

    # H_SO = -(1/5)*(pphi/r^3)*SOcoef
    dHSOdr = 0.6 * SOcoef * pphi / r4
    dHSOdpphi = -0.2 * SOcoef / r3

    # H_SS = (1/(3 eta))*(1/r^3)*SScoef
    dHSSdr = -(1.0 / eta) * SScoef / r4

    dHdr = dHNdr + dH1PNdr + dHSOdr + dHSSdr
    dHdpr = dHNdpr + dH1PNdpr
    dHdpphi = dHNdpphi + dH1PNdpphi + dHSOdpphi
    return (dHdr, dHdpr, dHdpphi)
end # @Tlit T
end

"""
    _asph_rhs(state::AlignedSpinPNHamState{T}, coeffs) -> AlignedSpinPNHamState{T}

The ODE right-hand side: pure, allocation-free (mirroring `ODEToy`'s `_odetoy_rhs` precedent).
Conservative `(r, phi, pr)` evolution via Hamilton's equations; `pphi` evolution via the
radiation-reaction relation derived in `AlignedSpinPNHam`'s docstring
(`dpphi/dt_hat = -F(v)/(eta*Omega_hat)`).
"""
@inline function _asph_rhs(state::AlignedSpinPNHamState{T}, coeffs) where {T}
@Tlit T begin
    r, phi, pr, pphi = state.r, state.phi, state.pr, state.pphi
    dHdr, dHdpr, dHdpphi = _asph_hamiltonian_partials(r, pr, pphi, coeffs.eta, coeffs.SOcoef, coeffs.SScoef)
    omega = dHdpphi
    omega_safe = max(omega, eps(T))     # guards a pathological (e.g. post-ISCO) omega <= 0
    v = cbrt(omega_safe)
    flux = _asph_flux(v, coeffs.fluxcoeffs)
    dpphi = -flux / (coeffs.eta * omega_safe)
    return AlignedSpinPNHamState(dHdpr, omega, -dHdr, dpphi)
end # @Tlit T
end

"""
    _asph_rk4_step(state, dt, coeffs) -> AlignedSpinPNHamState

One classical RK4 step, mirroring `ODEToy`'s `_odetoy_rk4_step` exactly.
"""
@inline function _asph_rk4_step(state::AlignedSpinPNHamState{T}, dt::T, coeffs) where {T}
@Tlit T begin
    k1 = _asph_rhs(state, coeffs)
    k2 = _asph_rhs(state + (dt / 2.0) * k1, coeffs)
    k3 = _asph_rhs(state + (dt / 2.0) * k2, coeffs)
    k4 = _asph_rhs(state + dt * k3, coeffs)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
end # @Tlit T
end

"""
    alignedspinpnham_rk4_integrate(state0, dt, nsteps, coeffs) -> AlignedSpinPNHamState

Fixed-step RK4 integration of [`_asph_rhs`](@ref) from `state0`, `nsteps` steps of size `dt`
(reduced time units -- see `AlignedSpinPNHam`'s docstring). Exported directly, mirroring
`odetoy_rk4_integrate`'s rationale (the function a later Enzyme/KernelAbstractions validation
pass will differentiate/kernelize).
"""
@inline function alignedspinpnham_rk4_integrate(state0::AlignedSpinPNHamState{T}, dt::T, nsteps::Integer, coeffs) where {T}
    state = state0
    for _ in 1:nsteps
        state = _asph_rk4_step(state, dt, coeffs)
    end
    return state
end

"""
    _asph_circular_ic(v0, eta, SOcoef, SScoef) -> (r0, pphi0)

Fixed (8-iteration, no early-exit) 2D Newton solve for the exact circular-orbit `(r_hat, p_phi_hat)`
of the conservative Hamiltonian (`chi1`/`chi2` folded into `SOcoef`/`SScoef`) at PN velocity `v0`
(`dH/dr_hat = 0`, `dH/dp_phi_hat = v0^3`, `p_r_hat = 0`) -- see `AlignedSpinPNHam`'s docstring's
"Initial conditions" section for why a solve (rather than the cheaper non-spin 1PN closed form
alone) is needed. Plain `Float64` throughout (one-time-per-waveform cost, like `TaylorT3.jl`'s
`theta_lso`/`t0_shift`), seeded from that same closed-form estimate. The Jacobian is a central
finite difference scaled to each variable's own `O(1)`-relative magnitude (`r_hat`, `p_phi_hat`
themselves, not a tiny spin-induced correction to them) -- this is *not* the precision trap noted
elsewhere in this module (that one arose from trying to resolve a `~1e-10`-relative correction via
finite differences on an `O(1e-4)`-scale quantity; here the finite difference resolves the
`O(1)`-scale `(r_hat, p_phi_hat)` solution itself, many orders of magnitude above float64 noise).
"""
function _asph_circular_ic(v0::Float64, eta::Float64, SOcoef::Float64, SScoef::Float64)
    x0 = v0 * v0 * (1.0 + (3.0 - eta) * v0 * v0 / 3.0)
    rhat = 1.0 / x0
    pphi = x0^(-0.5) * (1.0 + 2.0 * x0)
    target_Omega = v0^3
    for _ in 1:8
        dHdr, _, Om = _asph_hamiltonian_partials(rhat, 0.0, pphi, eta, SOcoef, SScoef)
        F1 = dHdr
        F2 = Om - target_Omega
        hr = 1e-6 * rhat
        hp = 1e-6 * pphi
        dHdr_rp, _, Om_rp = _asph_hamiltonian_partials(rhat + hr, 0.0, pphi, eta, SOcoef, SScoef)
        dHdr_rm, _, Om_rm = _asph_hamiltonian_partials(rhat - hr, 0.0, pphi, eta, SOcoef, SScoef)
        dHdr_pp, _, Om_pp = _asph_hamiltonian_partials(rhat, 0.0, pphi + hp, eta, SOcoef, SScoef)
        dHdr_pm, _, Om_pm = _asph_hamiltonian_partials(rhat, 0.0, pphi - hp, eta, SOcoef, SScoef)
        J11 = (dHdr_rp - dHdr_rm) / (2hr)
        J12 = (dHdr_pp - dHdr_pm) / (2hp)
        J21 = (Om_rp - Om_rm) / (2hr)
        J22 = (Om_pp - Om_pm) / (2hp)
        det = J11 * J22 - J12 * J21
        delta_r = (-F1 * J22 + F2 * J12) / det
        delta_p = (-J11 * F2 + J21 * F1) / det
        rhat += delta_r
        pphi += delta_p
    end
    return rhat, pphi
end

"""
    _asph_coeffs(model, mc, eta, chi1, chi2; f_min, GMsun_over_c3, LAL_GAMMA)

Time-independent part of the evaluation: mass fractions, spin-orbit/spin-spin Hamiltonian
coefficients, flux series coefficients, and the initial `(r, pphi)` at `f_min` (see
`AlignedSpinPNHam`'s docstring's "Initial conditions" section) -- computed once per waveform,
mirroring `TaylorT3.jl`'s `_taylort3_coeffs`/`IMRPhenomD`'s `_phi_coeffs` split.
"""
@inline function _asph_coeffs(model::AlignedSpinPNHam,
    mc,
    eta,
    chi1,
    chi2;
    f_min = 20.0,
    LAL_GAMMA = 0.577215664901532860606512090082402431,
    GMsun_over_c3 = GMsun_over_c3,
)
    T = promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))

    # eta, X1, X2 (mass fractions, m1 >= m2), spin-orbit/spin-spin coefficients, and the flux
    # series are all computed in plain Float64 first (cheap, one-time-per-waveform, and the flux
    # coefficient formulas below have no natural @Tlit-friendly form with their many literal
    # divisions) then cast to T -- mirroring TaylorT3.jl's _taylort3_coeffs pattern for theta_lso.
    eta_f64 = Float64(eta)
    delta_f64 = sqrt(max(1.0 - 4.0 * eta_f64, 0.0))
    X1_f64 = 0.5 * (1.0 + delta_f64)
    X2_f64 = 0.5 * (1.0 - delta_f64)
    chi1_f64 = Float64(chi1)
    chi2_f64 = Float64(chi2)

    E3S1O(x) = 2.0 / 3.0 + 2.0 / x
    SOcoef_f64 = E3S1O(X1_f64) * X1_f64^2 * chi1_f64 + E3S1O(X2_f64) * X2_f64^2 * chi2_f64
    SScoef_f64 = (X1_f64^2 * chi1_f64) * (X2_f64^2 * chi2_f64)

    fluxcoeffs_f64 = _asph_flux_coeffs(eta_f64, X1_f64, X2_f64, chi1_f64, chi2_f64, Float64(LAL_GAMMA))
    fluxcoeffs = map(T, fluxcoeffs_f64)

    # v0/circular initial condition: also plain Float64, one-time-per-waveform (see
    # _asph_circular_ic's docstring and AlignedSpinPNHam's "Initial conditions" section).
    M_sec_f64 = (Float64(mc) / Float64(eta)^0.6) * Float64(GMsun_over_c3)
    v0_f64 = cbrt(pi * M_sec_f64 * Float64(f_min))
    r0_f64, pphi0_f64 = _asph_circular_ic(v0_f64, eta_f64, SOcoef_f64, SScoef_f64)

    GMsun_over_c3 = T(GMsun_over_c3)
    eta = T(eta)
    X1 = T(X1_f64)
    X2 = T(X2_f64)
    chi1 = T(chi1)
    chi2 = T(chi2)
    SOcoef = T(SOcoef_f64)
    SScoef = T(SScoef_f64)

    r0 = T(r0_f64)
    pphi0 = T(pphi0_f64)
@Tlit T begin
    M = mc / (eta^(3.0 / 5.0))
    M_sec = M * GMsun_over_c3
    return (T = T, M = M, M_sec = M_sec, eta = eta, X1 = X1, X2 = X2, chi1 = chi1, chi2 = chi2,
        SOcoef = SOcoef, SScoef = SScoef, fluxcoeffs = fluxcoeffs, r0 = r0, pphi0 = pphi0)
end # @Tlit T
end

"""
    _asph_state_scalar(coeffs, t, dt)

Non-allocating: integrates from the shared circular-orbit initial condition at `f_min` (`r =
coeffs.r0`, `phi = 0`, `pr = 0`, `pphi = coeffs.pphi0`) for `round(Int, t/dt)` fixed RK4 steps of
reduced-time size `dt/coeffs.M_sec`, returning the resulting `AlignedSpinPNHamState`. `t`, `dt` in
physical seconds.
"""
@inline function _asph_state_scalar(coeffs, t, dt)
    T = coeffs.T
    state0 = AlignedSpinPNHamState(coeffs.r0, zero(T), zero(T), coeffs.pphi0)
    nsteps = round(Int, t / dt)
    dt_hat = T(dt) / coeffs.M_sec
    return alignedspinpnham_rk4_integrate(state0, dt_hat, nsteps, coeffs)
end

"""
    Phi(model::AlignedSpinPNHam, t, mc, eta, chi1, chi2; f_min=20.0, dt=1.0/4096.0, GMsun_over_c3=GMsun_over_c3, phiRef=0)

`AlignedSpinPNHam` GW phase (`2x` orbital phase) at time(s) `t` **seconds since the `f_min`-
referenced initial condition** (`t >= 0` -- see `AlignedSpinPNHam`'s docstring's "Domain
convention" section, this differs from `TaylorT3`'s `t <= 0`-to-coalescence convention). `t` may
be a scalar or `AbstractVector`.

# Arguments
- `mc`: chirp mass in solar masses. `eta`: symmetric mass ratio, `0 < eta <= 0.25`.
- `chi1`, `chi2`: dimensionless aligned spins (along the orbital angular momentum direction) of
  the more/less massive body respectively (`m1 >= m2` convention, matching
  `IMRPhenomXPrecession.jl`).

# Keywords
- `f_min`: reference GW frequency (Hz) defining the initial condition, default `20.0`.
- `dt`: fixed RK4 step size in seconds, default `1/4096` (matches this ecosystem's baseline).
- `GMsun_over_c3`: physical constant, overridable (mirroring `TaylorT3.Phi`).
- `phiRef`: additive constant phase offset (gauge freedom), default `0`.

No parameter-domain assertions run here -- see [`validate_alignedspinpnham_params`](@ref).
"""
function Phi(model::AlignedSpinPNHam,
    t,
    mc,
    eta,
    chi1,
    chi2;
    f_min = 20.0,
    dt = 1.0 / 4096.0,
    GMsun_over_c3 = GMsun_over_c3,
    phiRef = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
)
    coeffs = _asph_coeffs(model, mc, eta, chi1, chi2; f_min = f_min, GMsun_over_c3 = GMsun_over_c3)
    T = coeffs.T
    phiRef = T(phiRef)
    if t isa AbstractArray
        out = similar(t, T)
        @inbounds for i in eachindex(t)
            state = _asph_state_scalar(coeffs, t[i], dt)
            out[i] = 2 * state.phi + phiRef
        end
        return out
    else
        state = _asph_state_scalar(coeffs, t, dt)
        return 2 * state.phi + phiRef
    end
end

"""
    Ampl(model::AlignedSpinPNHam, t, mc, eta, chi1, chi2, dL; f_min=20.0, dt=1.0/4096.0, GMsun_over_c3=GMsun_over_c3, GMsun_over_c2_Gpc=GMsun_over_c2_Gpc)

Restricted (Newtonian-order) time-domain amplitude, same "(2,2)-mode-only, face-on" convention as
`TaylorT3.Ampl`/`IMRPhenomD.Ampl`: `h+ = -Ampl(t)*cos(Phi(t))`, `hx = -Ampl(t)*sin(Phi(t))` at zero
inclination. `Ampl(t) = 4*eta*M*GMsun_over_c2_Gpc/dL * v(t)^2`, identical formula to
`TaylorT3.Ampl` (see that docstring for the derivation), with `v(t) = Omega_hat(t)^(1/3)` sourced
from this model's own integrated `(r,pr,pphi)` state (via the same analytic `dH/dpphi` used in the
ODE's RHS) rather than a closed-form frequency series.

`t`, `mc`, `eta`, `chi1`, `chi2`, keywords follow [`Phi`](@ref) exactly; `dL` is the luminosity
distance in Gpc (same convention as `IMRPhenomD.Ampl`).
"""
function Ampl(model::AlignedSpinPNHam,
    t,
    mc,
    eta,
    chi1,
    chi2,
    dL;
    f_min = 20.0,
    dt = 1.0 / 4096.0,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    coeffs = _asph_coeffs(model, mc, eta, chi1, chi2; f_min = f_min, GMsun_over_c3 = GMsun_over_c3)
    T = coeffs.T
    GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc)
    dL = T(dL)
    if t isa AbstractArray
        out = similar(t, T)
        @inbounds for i in eachindex(t)
            state = _asph_state_scalar(coeffs, t[i], dt)
            _, _, omega = _asph_hamiltonian_partials(state.r, state.pr, state.pphi, coeffs.eta, coeffs.SOcoef, coeffs.SScoef)
            v = cbrt(max(omega, eps(T)))
            out[i] = 4 * coeffs.eta * coeffs.M * GMsun_over_c2_Gpc / dL * v * v
        end
        return out
    else
        state = _asph_state_scalar(coeffs, t, dt)
        _, _, omega = _asph_hamiltonian_partials(state.r, state.pr, state.pphi, coeffs.eta, coeffs.SOcoef, coeffs.SScoef)
        v = cbrt(max(omega, eps(T)))
        return 4 * coeffs.eta * coeffs.M * GMsun_over_c2_Gpc / dL * v * v
    end
end
