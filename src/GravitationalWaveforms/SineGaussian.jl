export SineGaussian, hphc_sine_gaussian

"""
    SineGaussian

Time-domain sine-Gaussian burst waveform (a windowed sinusoid, not a compact-binary-coalescence
model) -- the standard burst-search/glitch-injection template used e.g. by `lalinference`'s
Bayeswave and by `BayesWave`/coherent-WaveBurst-style analyses, and ripple's `SineGaussian.py`
(`gen_SineGaussian_hphc`). Included alongside this package's CBC models purely because ripple/Jim
expose it; it shares no machinery (no `Phi`/`Ampl`, no chirp mass, no PN physics) with the rest of
this package -- see [`hphc_sine_gaussian`](@ref) for its actual (very different) call contract.

Ported from `lalinference/lib/LALInferenceBurstRoutines.c`'s sine-Gaussian generator via ripple's
`SineGaussian.py`, which documents itself as directly following that LAL source.
"""
struct SineGaussian end

@inline function _sine_gaussian_semi_axes(eccentricity)
    T = typeof(eccentricity)
@Tlit T begin
    a = 1.0 / sqrt(2.0 - eccentricity * eccentricity)
    b = a * sqrt(1.0 - eccentricity * eccentricity)
    return a, b
end
end

"""
    hphc_sine_gaussian(model::SineGaussian, t, quality, frequency, hrss, phase, eccentricity)

Generate the sine-Gaussian burst waveform's `(h+, hx)` polarizations at times `t` (a scalar or
`AbstractVector`, seconds, centered so the envelope peaks at `t == 0` -- e.g.
`range(-duration/2, duration/2; step = 1/fs)`).

# Arguments
- `quality`: quality factor of the sine-Gaussian envelope (dimensionless).
- `frequency`: central frequency (Hz).
- `hrss`: root-sum-squared strain amplitude, sets the overall normalization.
- `phase`: carrier phase (rad).
- `eccentricity` (`0` to `1`): controls the relative `h+`/`hx` amplitudes -- `0` gives circularly
  polarized (equal `h+`/`hx` amplitude), `1` gives purely linearly polarized (`hx == 0`).

Returns a `(hp, hc)` tuple, each the same shape as `t` (scalar `t` -> scalar result, matching
this package's `Phi`/`Ampl` scalar/vector convention elsewhere, though this model has no
`Phi`/`Ampl` split of its own -- the envelope and carrier are inseparable in this
parametrization).

No parameter-domain assertions run here (same hot-path rationale as [`validate_phenomd_params`](@ref)
documents for the CBC models) -- `quality > 0`, `frequency > 0`, `hrss > 0`, `0 <= eccentricity <= 1`
are the caller's responsibility.
"""
@inline function hphc_sine_gaussian(model::SineGaussian, t, quality, frequency, hrss, phase, eccentricity)
    T = promote_type(typeof(quality), typeof(frequency), typeof(hrss), typeof(phase), typeof(eccentricity),
        (t isa AbstractArray ? eltype(t) : typeof(t)))
    quality = T(quality)
    frequency = T(frequency)
    hrss = T(hrss)
    phase = T(phase)
    eccentricity = T(eccentricity)

    a, b = _sine_gaussian_semi_axes(eccentricity)
@Tlit T begin
    norm_prefactor = quality / (4.0 * frequency * sqrt(pi))
    cosine_norm = norm_prefactor * (1.0 + exp(-quality * quality))
    sine_norm = norm_prefactor * (1.0 - exp(-quality * quality))
    cos_phase = cos(phase)
    sin_phase = sin(phase)
    h0_plus = hrss * a / sqrt(cosine_norm * cos_phase * cos_phase + sine_norm * sin_phase * sin_phase)
    h0_cross = hrss * b / sqrt(cosine_norm * sin_phase * sin_phase + sine_norm * cos_phase * cos_phase)

    if t isa AbstractArray
        hp = similar(t, T)
        hc = similar(t, T)
        @inbounds for i in eachindex(t)
            hp[i], hc[i] = _sine_gaussian_scalar(t[i], quality, frequency, phase, h0_plus, h0_cross)
        end
        return hp, hc
    else
        return _sine_gaussian_scalar(t, quality, frequency, phase, h0_plus, h0_cross)
    end
end # @Tlit T
end

@inline function _sine_gaussian_scalar(t, quality, frequency, phase, h0_plus, h0_cross)
    T = typeof(t)
@Tlit T begin
    phi = 2.0 * pi * frequency * t
    envelope = exp(phi * phi / (-2.0 * quality * quality))
    carrier_re = cos(phi - phase)
    carrier_im = sin(phi - phase)
    return envelope * carrier_re * h0_plus, envelope * carrier_im * h0_cross
end
end
