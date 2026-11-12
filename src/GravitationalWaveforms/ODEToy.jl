using StaticArrays: FieldVector

export ODEToy, ODEToyState, validate_odetoy_params, odetoy_rk4_integrate, odetoy_omega_analytic

"""
    ODEToy

**Scaffolding model --
NOT a physically complete or LALSuite-validated waveform model.** Its only purpose is to
validate, end-to-end, the architecture pattern the later (physically useful) ODE-integrated
models in this package will reuse: a `StaticArrays`-based state type, a fixed-step,
allocation-free RK4 integrator, an `Enzyme.gradient` call through that integrator, and a
successful compile inside a `KernelAbstractions.jl` `@kernel`. See
`test/odetoy_enzyme_ka_tests.jl` for that validation.

# Physics
0PN (leading, "Newtonian") order circular-orbit inspiral with radiation reaction -- the
simplest non-trivial chirping system, 2 state variables (orbital phase `phi`, orbital angular
frequency `omega`):

```
domega/dt = (96/5) * Mc^(5/3) * omega^(11/3)
dphi/dt   = omega
```

where `Mc` is the chirp mass **in seconds** (geometric units, `G = c = 1`) -- i.e.
`Mc_sec = mc_Msun * GMsun_over_c3`, mirroring `TaylorT3`'s `M_sec` convention. This is the
standard leading-order PN frequency-evolution ODE (e.g. Maggiore, *Gravitational Waves* Vol. 1,
eq. 4.29-4.37) truncated at 0PN, with no higher-PN terms, no spin, and no attempt at the
sub-leading amplitude corrections `IMRPhenomD`/`TaylorT3` include. Because the RHS is simple enough
to have a closed-form solution (see [`odetoy_omega_analytic`](@ref)), it doubles as a built-in
correctness check for the RK4 integrator itself, independent of any Enzyme/KernelAbstractions
concern.

# Phase convention
As with `TaylorT3` (see that struct's docstring), the GW phase is twice the orbital phase:
`Phi = 2*phi`, so that `Phi`/`Ampl` stay mutually consistent with this package's
`h ~ Ampl(x)*exp(i*Phi(x))` convention.

# Entry point: step count, not continuous time
Unlike `IMRPhenomD`/`IMRPhenomXAS` (continuous frequency domain) or `TaylorT3` (continuous time
domain via a closed-form formula), a fixed-step RK4 integrator fundamentally only produces
states at multiples of a chosen step size `dt` -- there is no closed-form way to ask for an
arbitrary intermediate time without either interpolating (which this scaffolding step
deliberately avoids -- it would reintroduce exactly the kind of dynamic, allocating control
flow the whole point of this exercise is to rule out) or re-deriving a continuous solution
(which would defeat the purpose of testing the *ODE-integration* pattern). So `Phi`/`Ampl`
here take an integer step count `n` (or an `AbstractVector` of them) as their domain variable,
not a time or frequency: `Phi(model, n, mc, omega0)` is the GW phase reached after `n` fixed
RK4 steps of size `dt` from the initial condition `phi=0, omega=omega0` at step `0`. The
corresponding time is `t = n*dt`, available to callers that want it.

Each requested `n` is integrated independently from the shared initial condition (mirroring
`TaylorT3`'s independent-per-sample evaluation) rather than reusing partial trajectories across
an array of `n`s -- appropriate for a validation scaffold evaluated at a handful of points, not
optimized for dense trajectory output.
"""
struct ODEToy end

"""
    ODEToyState{T} <: FieldVector{2, T}

The `ODEToy` ODE state: orbital phase `phase` and orbital angular frequency `omega`, both `T`.
A `StaticArrays.jl` `FieldVector` (not a plain `Vector`/`Array`) -- fixed-size, stack-allocated,
and immutable, which is exactly what keeps the RK4 loop in
[`odetoy_rk4_integrate`](@ref) allocation-free (a hard requirement for Enzyme to differentiate
through code running inside a `KernelAbstractions.jl` `@kernel` -- see
`GravitationalWaveBayes.jl/src/GravitationalWaveBayes/Waveforms/IMRPhenomD.jl`'s `_phi_scalar`/`_ampl_scalar` precedent, which
this file's `_odetoy_*` split directly mirrors).

Subtyping `FieldVector{2, T}` (rather than hand-writing `+`/`*` methods) gets elementwise `+`
and scalar `*` "for free" from `StaticArrays.jl`'s generic `StaticArray` arithmetic -- confirmed
directly (not assumed): `ODEToyState(1.0,2.0) + ODEToyState(0.5,0.25)` and
`2*ODEToyState(1.0,2.0)` both return a fresh, correctly-typed `ODEToyState`, which is exactly
the algebra an RK4 step needs (`state + (dt/2)*k1`, etc.) with no custom operator definitions
required.
"""
struct ODEToyState{T} <: FieldVector{2, T}
    phase::T
    omega::T
end

"""
    validate_odetoy_params(mc, omega0; dt = nothing)

Host-side sanity check: `mc > 0` (solar masses), `omega0 > 0` (rad/s), and (if `dt` given)
`dt > 0`. Throws `ArgumentError` on violation, `nothing` otherwise. Not called by `Phi`/`Ampl`
themselves, matching `validate_phenomd_params`/`validate_taylort3_params`'s documented rationale
(hot-path; one-time domain checks belong at the sampler/prior level).
"""
function validate_odetoy_params(mc, omega0; dt = nothing)
    mc > 0.0 || throw(ArgumentError("Invalid mc = $mc: chirp mass must be positive (solar masses)"))
    omega0 > 0.0 || throw(ArgumentError("Invalid omega0 = $omega0: initial orbital angular frequency must be positive (rad/s)"))
    if dt !== nothing
        dt > 0.0 || throw(ArgumentError("Invalid dt = $dt: integrator step size must be positive (seconds)"))
    end
    return nothing
end

"""
    _odetoy_rhs(state::ODEToyState{T}, Mc_sec::T) -> ODEToyState{T}

The ODE right-hand side (see `ODEToy`'s docstring for the physics): pure, allocation-free --
takes and returns only `ODEToyState`/`T` values, no arrays, no mutation. This is the function
Enzyme ultimately differentiates through (indirectly, via [`odetoy_rk4_integrate`](@ref)); its
purity/non-allocation is what keeps that differentiable inside a `KernelAbstractions.jl`
`@kernel` body, matching `GravitationalWaveBayes.jl`'s `_phi_scalar`/`_ampl_scalar` precedent.
"""
@inline function _odetoy_rhs(state::ODEToyState{T}, Mc_sec::T) where {T}
    @Tlit T begin
        omega = state.omega
        domega = (96.0 / 5.0) * Mc_sec^(5.0 / 3.0) * omega^(11.0 / 3.0)
        dphase = omega
        return ODEToyState(dphase, domega)
    end # @Tlit T
end

"""
    _odetoy_rk4_step(state::ODEToyState{T}, dt::T, Mc_sec::T) -> ODEToyState{T}

One classical (4th-order, 4-stage) Runge-Kutta step of fixed size `dt`, built from
[`_odetoy_rhs`](@ref) and `ODEToyState`'s `+`/scalar-`*` (see that struct's docstring) -- no
adaptive step-size control, no array allocation.
"""
@inline function _odetoy_rk4_step(state::ODEToyState{T}, dt::T, Mc_sec::T) where {T}
    @Tlit T begin
        k1 = _odetoy_rhs(state, Mc_sec)
        k2 = _odetoy_rhs(state + (dt / 2.0) * k1, Mc_sec)
        k3 = _odetoy_rhs(state + (dt / 2.0) * k2, Mc_sec)
        k4 = _odetoy_rhs(state + dt * k3, Mc_sec)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    end # @Tlit T
end

"""
    odetoy_rk4_integrate(state0::ODEToyState{T}, dt::T, nsteps::Integer, Mc_sec::T) -> ODEToyState{T}

Fixed-step RK4 integration of [`_odetoy_rhs`](@ref) from `state0`, `nsteps` steps of size `dt`
(a plain `for` loop reassigning an immutable `ODEToyState`, matching this package's/`GravitationalWaveBayes.jl`'s
"no heap allocation in the differentiated hot path" requirement -- no array resizing, no
adaptive control flow, statically-sized state throughout). Exported directly (not just an
internal `_`-prefixed helper) since this is the function the Enzyme/KernelAbstractions
validation tests in `test/odetoy_enzyme_ka_tests.jl` differentiate/kernelize -- the actual point
of this scaffolding step.
"""
@inline function odetoy_rk4_integrate(state0::ODEToyState{T}, dt::T, nsteps::Integer, Mc_sec::T) where {T}
    state = state0
    for _ in 1:nsteps
        state = _odetoy_rk4_step(state, dt, Mc_sec)
    end
    return state
end

"""
    odetoy_omega_analytic(t, omega0, Mc_sec) -> closed-form omega(t)

Closed-form solution of `domega/dt = (96/5)*Mc_sec^(5/3)*omega^(11/3)`, `omega(0) = omega0`,
used only as an independent correctness check on [`odetoy_rk4_integrate`](@ref) in
`test/odetoy_tests.jl` (not called from `Phi`/`Ampl`). Derivation: substituting
`u = omega^(-8/3)` linearizes the ODE to `du/dt = -(8/3)*(96/5)*Mc_sec^(5/3) =
-(256/5)*Mc_sec^(5/3)`, a constant, so `u(t) = omega0^(-8/3) - (256/5)*Mc_sec^(5/3)*t` and
`omega(t) = u(t)^(-3/8)`. (An earlier draft mis-simplified `(8/3)*(96/5) = 768/15` down to
`256/15` instead of the correct `256/5` -- caught by comparing against the RK4 integrator, which
diverged from the [initially buggy] closed form by an *increasing* relative amount as `t` grew,
rather than the flat, tiny (RK4-truncation-only) discrepancy expected of two correct
implementations of the same ODE; recomputing the fraction by hand found the arithmetic slip.)
"""
function odetoy_omega_analytic(t, omega0, Mc_sec)
    k = (256.0 / 5.0) * Mc_sec^(5.0 / 3.0)
    u = omega0^(-8.0 / 3.0) - k * t
    return u^(-3.0 / 8.0)
end

"""
    _odetoy_coeffs(model::ODEToy, mc, omega0; GMsun_over_c3 = GMsun_over_c3)

Time-independent part of the `ODEToy` evaluation (chirp-mass unit conversion, working-type
resolution), computed once per `Phi`/`Ampl` call -- mirroring `IMRPhenomD`'s/`TaylorT3`'s own
`_coeffs`/`_scalar` split.
"""
@inline function _odetoy_coeffs(model::ODEToy, mc, omega0; GMsun_over_c3 = GMsun_over_c3)
    T = promote_type(typeof(mc), typeof(omega0))
    GMsun_over_c3 = T(GMsun_over_c3)
    @Tlit T begin
        mc = mc
        omega0 = omega0
        Mc_sec = mc * GMsun_over_c3
        return (T = T, Mc_sec = Mc_sec, omega0 = omega0)
    end # @Tlit T
end

"""
    _odetoy_state_scalar(coeffs, n::Integer, dt)

Non-allocating: integrates from the shared initial condition (`phase = 0`, `omega = omega0`)
for `n` fixed RK4 steps of size `dt`, returning the resulting `ODEToyState`. See
[`odetoy_rk4_integrate`](@ref).
"""
@inline function _odetoy_state_scalar(coeffs, n::Integer, dt)
    T = coeffs.T
    state0 = ODEToyState(zero(T), coeffs.omega0)
    return odetoy_rk4_integrate(state0, dt, n, coeffs.Mc_sec)
end

"""
    Phi(model::ODEToy, n, mc, omega0; dt = 1.0/4096.0, GMsun_over_c3 = GMsun_over_c3, phiRef = 0)

`ODEToy` GW phase (`2x` orbital phase, see `ODEToy`'s docstring) after `n` fixed RK4 steps of
size `dt` from the initial condition `phase = 0, omega = omega0` at step `0`. `n` may be a
scalar `Integer` or an `AbstractVector{<:Integer}` (see `ODEToy`'s docstring for why the domain
variable here is a step count, not a continuous time/frequency, unlike `IMRPhenomD`/`TaylorT3`).

# Arguments
- `mc`: chirp mass in solar masses.
- `omega0`: initial orbital angular frequency in rad/s.

# Keywords
- `dt`: fixed RK4 step size in seconds, default `1/4096` (matches this ecosystem's established
  `TaylorT3`/data-conditioning baseline `deltaT`).
- `GMsun_over_c3`: physical constant, overridable (mirroring `IMRPhenomD.Phi`/`TaylorT3.Phi`).
- `phiRef`: additive constant phase offset (gauge freedom), default `0`.

No parameter-domain assertions run here -- see [`validate_odetoy_params`](@ref) for an opt-in
host-side check.
"""
function Phi(model::ODEToy,
    n,
    mc,
    omega0;
    dt = 1.0 / 4096.0,
    GMsun_over_c3 = GMsun_over_c3,
    phiRef = zero(promote_type(typeof(mc), typeof(omega0))),
)
    coeffs = _odetoy_coeffs(model, mc, omega0; GMsun_over_c3 = GMsun_over_c3)
    T = coeffs.T
    phiRef = T(phiRef)
    dtT = T(dt)
    if n isa AbstractArray
        out = similar(n, T)
        @inbounds for i in eachindex(n)
            state = _odetoy_state_scalar(coeffs, n[i], dtT)
            out[i] = 2 * state.phase + phiRef
        end
        return out
    else
        state = _odetoy_state_scalar(coeffs, n, dtT)
        return 2 * state.phase + phiRef
    end
end

"""
    Ampl(model::ODEToy, n, mc, omega0, dL; dt = 1.0/4096.0, GMsun_over_c3 = GMsun_over_c3)

**Not a physically validated PN amplitude.** `ODEToy`'s ODE (see that struct's docstring) only
specifies the leading-order *phasing*, and this scaffolding model has no mass-ratio/`eta`
decomposition to build the usual restricted-Newtonian amplitude prefactor
(`eta*M*(pi*M*f)^(2/3)`, as `TaylorT3.Ampl`/`IMRPhenomD.Ampl` do) from -- only a chirp mass. This
is a smooth, monotonically-growing-with-frequency placeholder (`Mc_sec * omega^(2/3) / dL`)
included purely so `Ampl` exists at all, completing the `Phi`/`Ampl` dispatch pattern this
package's other models follow, for architecture-pattern purposes (a second differentiable/
kernelizable function alongside `Phi`). Do not use for real waveform generation; the physically
useful ODE-integrated models (`AlignedSpinPNHam` onwards) replace this with
actual PN amplitude physics.

`n`, `mc`, `omega0`, `dt`, `GMsun_over_c3` follow [`Phi`](@ref) exactly; `dL` is the luminosity
distance (any consistent unit -- this placeholder does no unit conversion, unlike `IMRPhenomD`/
`TaylorT3`'s `GMsun_over_c2_Gpc`-based `Ampl`, precisely because it isn't meant to be a
physically meaningful amplitude in the first place).
"""
function Ampl(model::ODEToy,
    n,
    mc,
    omega0,
    dL;
    dt = 1.0 / 4096.0,
    GMsun_over_c3 = GMsun_over_c3,
)
    coeffs = _odetoy_coeffs(model, mc, omega0; GMsun_over_c3 = GMsun_over_c3)
    T = coeffs.T
    dtT = T(dt)
    dL = T(dL)
    if n isa AbstractArray
        out = similar(n, T)
        @inbounds for i in eachindex(n)
            state = _odetoy_state_scalar(coeffs, n[i], dtT)
            out[i] = coeffs.Mc_sec * state.omega^(2 * one(T) / 3) / dL
        end
        return out
    else
        state = _odetoy_state_scalar(coeffs, n, dtT)
        return coeffs.Mc_sec * state.omega^(2 * one(T) / 3) / dL
    end
end
