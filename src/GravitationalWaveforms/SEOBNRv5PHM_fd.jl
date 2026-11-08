export SEOBNRv5PHM

"""
    SEOBNRv5PHM

**SEOBNRv5PHM in the frequency domain**: the Fourier transform of the time-domain model ([`seobnrv5phm_td`](@ref),
validated against pyseobnr 0.3.7 including the anti-symmetric modes) sampled on the frequency grid's own time grid,
`h(f_k) = dt sum_n h(t_n) e^{-2 pi i f_k (t_n - t_peak)}` by one FFT of `h+ - i hx`, with a Planck turn-on over the
first four (2,2) cycles (the start of the waveform; pyseobnr's FD output is likewise a conditioned TD transform).
Time origin: pyseobnr's `t = 0` (the frame-invariant amplitude peak). No stationary-phase approximation is involved.

Why not an SPA "twist" (as the inspiral tables of [`SEOBNRv5HM`](@ref) and IMRPhenomXPHM): with strong precession
the twist's phase (`m alpha + 2 gamma` of the rotation to the observer) changes over one SPA width by up to ~1 rad
(q = 8, chi_perp ~ 0.7 at 20 Hz), and where the precession cone passes near the line of sight the observer-frame
angles swing so fast that the twisted signal has several stationary points per frequency; both SPA-twist variants
tried (twist at the stationary time; twist folded into each sign's phase) missed the transform of the same TD
model by up to 1e-2 in mismatch there. Sampling the TD model costs one dense-output evaluation per sample (no
dynamics), which is sample-parallel on a GPU, and the FFT is batched: the device path (GravitationalWaveBayes) evaluates the
waveform on the data's time grid directly.

Conventions (pyseobnr `GenerateWaveform`, LAL): `h+ = -K Re(hpc)`, `hx = K Im(hpc)`, `K = M / d_L`; masses swapped
with `phiRef + pi` when `m1 < m2`; `f22_start = f_min`; model `phiref = pi/2 - phiRef`.
"""
struct SEOBNRv5PHM end

# The FFT length (power of two) for a grid of spacing df (Hz) up to fmax and total mass Ms (s): sampling rate
# 1/dt >= fmax + 0.3/M. h+ - i hx has content up to |f| ~ 0.25/M (the (5,5) ringdown at high remnant spin, both signs
# of m), so with that rate nothing aliases into |f| <= fmax.
function _seobv5phm_nfft(df, fmax, Ms)
    N = 1
    while N * df < fmax + 0.3 / Ms
        N <<= 1
    end
    return N
end

# h+ - i hx at dynamics time tq (M), zero outside [0, t_end] (the ringdown is evaluated for 15 (2,2) damping times).
@inline function _seobv5phm_hpc_or0(S, tq, incl, phiref, t_end)
    (tq < 0 || tq > t_end) && return zero(ComplexF64)
    return _seobv5phm_hpc(S, tq, incl, phiref)
end

"""
    _seobv5phm_fd_fft(S, inclination, phiref, df, N) -> (Hplus, Hminus)

`H(+-f_j)`, `f_j = j df` (`j = 0 .. N/2`; `df` in units of 1/M), of `hpc(t)` sampled at `dt = 1/(N df)` from the start of
the dynamics (turn-on over the first four (2,2) cycles), phases referred to `S.t_peak`. `N` a power of two with
`N dt` at least the waveform's length (the caller's segment).
"""
function _seobv5phm_fd_fft(S, inclination, phiref, df, N)
    dt = 1 / (N * df)
    t_end = S.t_a + _EOBIMR_NTAU * S.mr[1].tau
    Lt = 4 * pi / S.dyn.dphi[1]                      # four (2,2) cycles
    re = zeros(N); imv = zeros(N)
    for n in 1:N
        t = (n - 1) * dt
        # FFT+ of conj(h): conj(result) = sum h e^{-2 pi i j n / N}
        h = _seobv5phm_hpc_or0(S, t, inclination, phiref, t_end) * _eobimr_turn_on(t / Lt)
        re[n] = real(h); imv[n] = -imag(h)
    end
    _eobimr_fft!(re, imv)
    NF = N ÷ 2 + 1
    Hp = Vector{ComplexF64}(undef, NF); Hm = Vector{ComplexF64}(undef, NF)
    for j in 0:N÷2
        sh = cis(2pi * j * df * S.t_peak)            # e^{-2 pi i f (0 - t_peak)}
        Hp[j+1] = dt * complex(re[j+1], -imv[j+1]) * sh
        jm = mod(-j, N)
        Hm[j+1] = dt * complex(re[jm+1], -imv[jm+1]) * conj(sh)
    end
    return Hp, Hm
end

"""
    hphc(model::SEOBNRv5PHM, f, m1, m2, chi1, chi2, dL, inclination, phiRef; f_min = 20.0, f_ref = f_min, ...)

Frequency-domain `(h+, hx)` of SEOBNRv5PHM at the frequencies `f` (Hz): a uniform grid `f = (k0 .. k1) df`
(as a data segment's; `1/df` must cover the waveform). Masses in solar masses, spin vectors `chi1`, `chi2` (frame
with `L_N` along z at `f_ref`), `dL` in Gpc; see [`SEOBNRv5PHM`](@ref) for the conventions. `nfft` (0: automatic,
[`_seobv5phm_nfft`](@ref)) sets the FFT length, i.e. the sampling `dt = 1/(nfft df)`.
"""
function hphc(model::SEOBNRv5PHM, f::AbstractVector, m1, m2, chi1, chi2, dL, inclination, phiRef;
        f_min = 20.0, f_ref = f_min, nfft::Int = 0, GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc)
    if m1 < m2
        m1, m2, chi1, chi2 = m2, m1, chi2, chi1
        phiRef += pi
    end
    M = Float64(m1 + m2)
    Ms = M * Float64(GMsun_over_c3)
    df = f[2] - f[1]
    k = round.(Int, f ./ df)
    maximum(abs.(k .* df .- f)) < 1e-9 * df || throw(ArgumentError("SEOBNRv5PHM hphc: f must be a uniform grid of multiples of its spacing"))
    S = _seobv5phm_build(m1 / M, m2 / M, collect(Float64, chi1), collect(Float64, chi2), pi * f_ref * Ms, pi * f_min * Ms)
    # sampling: dt <= 0.25 M and <= 1/(2 f_max); N dt = 1/df must cover the waveform
    t_end = S.t_a + _EOBIMR_NTAU * S.mr[1].tau
    dfM = df * Ms
    t_end <= 1 / dfM || throw(ArgumentError("SEOBNRv5PHM hphc: waveform ($(t_end * Ms) s) longer than 1/df"))
    N = nfft > 0 ? nfft : _seobv5phm_nfft(df, maximum(f), Ms)
    Hp, Hm = _seobv5phm_fd_fft(S, Float64(inclination), pi / 2 - Float64(phiRef), dfM, N)
    Kd = M * Float64(GMsun_over_c2_Gpc) / Float64(dL) * Ms
    hp = similar(f, ComplexF64); hc = similar(f, ComplexF64)
    for (i, kk) in enumerate(k)
        a, b = Hp[kk+1], Hm[kk+1]
        hp[i] = -Kd * (a + conj(b)) / 2
        hc[i] = Kd * (a - conj(b)) / (2im)
    end
    return hp, hc
end
