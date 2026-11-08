export frequency_series, time_series

using GravitationalWaveData: FrequencySeries, TimeSeries

"""
    frequency_series(model, f0, df, n, mc, eta, chi1, chi2, dL; name=nothing,
                      ampl_kwargs=NamedTuple(), phi_kwargs=NamedTuple()) -> GravitationalWaveData.FrequencySeries

Convenience wrapper around [`Ampl`](@ref)/[`Phi`](@ref) for frequency-domain models with the
`Ampl(model, f, mc, eta, chi1, chi2, dL; ...)`/`Phi(model, f, mc, eta, chi1, chi2; ...)`
dominant-mode contract (`IMRPhenomD`, `IMRPhenomXAS`) -- evaluates both on the uniform grid
`f0, f0+df, ..., f0+(n-1)*df` and packages the result as a `GravitationalWaveData.jl` `FrequencySeries`
(`h = Ampl .* cis.(Phi)`, the same SPA complex-strain convention used for the LALSuite comparisons).

This is a deliberately thin first use of `GravitationalWaveData` as a dependency: the raw `Ampl`/`Phi`
scalar/vector API remains this package's primary interface (in particular, the allocation-free
scalar path a GPU `@kernel`/Enzyme call needs bypasses this wrapper entirely) -- reach for this
only when a `GravitationalWaveData.jl`-typed result is actually wanted, e.g. to hand a waveform to that
package's I/O/PSD/plotting code.

`ampl_kwargs`/`phi_kwargs` forward to the respective `Ampl`/`Phi` call (e.g.
`phi_kwargs = (; t0 = ..., phiRef = ...)`); `name` becomes the `FrequencySeries`'s free-text label.
"""
function frequency_series(model, f0, df, n::Integer, mc, eta, chi1, chi2, dL;
    name::Union{AbstractString,Nothing} = nothing,
    ampl_kwargs::NamedTuple = NamedTuple(),
    phi_kwargs::NamedTuple = NamedTuple())
    f = collect(f0 .+ df .* (0:(n - 1)))
    amp = Ampl(model, f, mc, eta, chi1, chi2, dL; ampl_kwargs...)
    phi = Phi(model, f, mc, eta, chi1, chi2; phi_kwargs...)
    h = amp .* cis.(phi)
    return FrequencySeries(h, Float64(f0), Float64(df); name = name)
end

"""
    time_series(model, t0, dt, n, mc, eta, dL; name=nothing,
                ampl_kwargs=NamedTuple(), phi_kwargs=NamedTuple()) -> GravitationalWaveData.TimeSeries

Convenience wrapper around [`Ampl`](@ref)/[`Phi`](@ref) for time-domain models with the
`Ampl(model, t, mc, eta, dL; ...)`/`Phi(model, t, mc, eta; ...)` non-spinning contract
(`TaylorT3`) -- evaluates both on the uniform grid `t0, t0+dt, ..., t0+(n-1)*dt` and packages the
result as a `GravitationalWaveData.jl` `TimeSeries`. Uses the same `hp = -Ampl*cos(Phi)`, `hc = -Ampl*sin(Phi)`
convention documented in `TaylorT3.jl`, combined into
the standard complex strain `h = hp - i*hc = -Ampl*cis(-Phi)`.

See [`frequency_series`](@ref)'s docstring for the general rationale (thin, optional, `Ampl`/`Phi`
stay primary).
"""
function time_series(model, t0, dt, n::Integer, mc, eta, dL;
    name::Union{AbstractString,Nothing} = nothing,
    ampl_kwargs::NamedTuple = NamedTuple(),
    phi_kwargs::NamedTuple = NamedTuple())
    t = collect(t0 .+ dt .* (0:(n - 1)))
    amp = Ampl(model, t, mc, eta, dL; ampl_kwargs...)
    phi = Phi(model, t, mc, eta; phi_kwargs...)
    h = -amp .* cis.(-phi)
    return TimeSeries(h, Float64(t0), Float64(dt); name = name)
end
