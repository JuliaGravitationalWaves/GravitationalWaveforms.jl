using StaticArrays: SVector, MVector

export EOBIMRTable

# ===================================================================================================
# Shared aligned-spin EOB inspiral-merger-ringdown machinery ((2,2) mode, frequency domain), used by
# SEOBNRv5IMRAlignedSpinF2 (this file's pipeline with SEOBNRv5 hooks), SEOBNRv5HM and SEOBNRv5PHM (helpers,
# fits, constants), and GravitationalWaveBayes' device tables. The model supplies the dynamics through the `_imr_*` hooks
# below; everything else is model-independent.
#
# Pipeline (the SEOBNRv5HM aligned-spin structure, `pyseobnr` 0.3.7 `models/SEOBNRv5HM.py`):
# 1. Post-adiabatic inspiral from the model's PA table (`_imr_pa`); when the PA grid is unusable the
#    inspiral starts from the model's short-inspiral state (`_imr_short_start`).
# 2. Plunge: fixed-step-count RK4 of the full equations of motion (`_imr_rhs4`) from the PA endpoint,
#    adaptive step `dt = clamp(min(k/Omega, kr r/|dr/dt|), hmin, hmax)` (`_imr_plunge_h`), stopped
#    (later steps masked) at the orbital-frequency peak, `dr/dt >= 0` or `r < 1.4` -- `pyseobnr`'s ODE
#    stopping conditions; with a finite `hend` a step that ends past the frequency peak or at `dr/dt >= 0`
#    is first retried with half the step (and all later steps capped there) until the step is below
#    `hend`. `_EOBIMR_NRK` loop iterations, constant trip count (Enzyme- and GPU-friendly).
# 3. Attachment time (Eqs. 41-43 of arXiv:2303.18039): `t_ISCO` where `r` crosses the Kerr ISCO of the
#    remnant (`_seobv5_kerr_isco`, HBR2016 final spin); `t_attach = t_ISCO - DeltaT_22` with the SEOBNRv5
#    fit `NR_deltaT_NS(nu) + NR_deltaT(nu, a+, a-)` (`fits_Hamiltonian.py`); attached at the end of the
#    dynamics if that is earlier (as `pyseobnr` does).
# 4. Inspiral (2,2) mode: `|h22|` from the model (`_imr_amp`), phase `2 phi - _imr_phase_corr`.
# 5. NQC corrections (`EOBCalculateNQCCoefficients_freeattach`, `EOB_fits.py`):
#    `|h22| *= 1 + pr^2/(r Omega)^2 (a1 + a2/r + a3/r^(3/2))`, `phase += b1 pr/(r Omega) + b2 pr^3/(r Omega)`,
#    with `(a1, a2, a3)` making the amplitude and its first two time derivatives, and `(b1, b2)` making the
#    frequency and its derivative, equal the SEOBNRv5 input-value fits (`IV_fits.py`) at `t_attach`.
#    Optional smooth shrinkage of `(b1, b2)` (`_imr_nqc_phase_scale`; `Inf` = none, SEOBNRv5's choice).
# 6. Merger-ringdown (`compute_MR_mode_free`, `compute_MR.py`, Eqs. 56-61 of the SEOBNRv5HM notes):
#    `h22 = nu (c1c tanh(c1f dt + c2f) + c2c) exp(i phi_match - i d1c log((1 + d2f e^{-d1f dt})/(1 + d2f)))
#    exp(-i omega_220 dt - dt/tau_220)`, `c1f, c2f, d1f, d2f` from `MR_fits.py`, `c1c, c2c, d1c` from
#    continuity of amplitude, its derivative and the frequency at `t_attach`. Remnant: UIB2016 final mass,
#    HBR2016 (M3J4) final spin; (2,2,0) QNM from the IMRPhenomX fits (`_xas_fRD_fdamp`).
# 7. Frequency domain: below a blend band the stationary-phase approximation on the (NQC-corrected)
#    mode's time/phase/frequency table (the same Hermite machinery as `_seobv5f2_dense_to_freq`, plus a
#    table of the SPA amplitude factor amplitude-correction / sqrt(d omega/dt)); above it a direct Fourier transform of the late mode (from 1500 M before
#    `t_attach`, 600 M Planck-type turn-on) through the ringdown, tabulated at `_EOBIMR_NF` frequencies
#    together with its frequency derivative and interpolated with cubic Hermite polynomials in amplitude
#    and phase. The two are blended with a C1 weight across `f(t_attach - 150 M) .. f(t_attach - 90 M)`.
#    The last 20% of the table band (up to `2.2 f_220`) carries a C1 amplitude taper. The Fourier table
#    is the bin grid of two radix-2 FFTs (of the mode and of `t` times the mode, for the frequency
#    derivatives), `_EOBIMR_NF` consecutive bins from just below the blend band.
# ===================================================================================================

const _EOBIMR_NRK = 512                               # RK4 plunge steps (fixed trip count)
const _EOBIMR_NDYN = _SEOBV5_PA_NMAX + _EOBIMR_NRK     # dynamics node buffer (PA nodes + RK nodes)
const _EOBIMR_NSPA_RK = 128                           # SPA-table nodes placed on the post-PA segment
const _EOBIMR_NT = _SEOBV5_PA_NMAX + _EOBIMR_NSPA_RK   # SPA table length
const _EOBIMR_NSEG = 2048                             # Fourier-transformed segment samples
const _EOBIMR_NF = 512                                # Fourier table: consecutive FFT bins kept
const _EOBIMR_TSEG_PRE = 1500.0                       # segment start, M before t_attach
const _EOBIMR_TTAPER = 600.0                          # Planck-taper length at the segment start, M
const _EOBIMR_TBLEND = (150.0, 90.0)                  # blend band: f(t_attach - 150) .. f(t_attach - 90)
const _EOBIMR_NTAU = 15.0                             # ringdown length, damping times
const _EOBIMR_FHI_QNM = 2.2                           # table top: this many f_220
const _EOBIMR_R_START = 13.5                          # dynamics start no later than this radius, M

# ---------------------------------------------------------------------------------------------------
# SEOBNRv5 fits (pyseobnr 0.3.7), (2,2) mode only. Verbatim coefficients.
# ---------------------------------------------------------------------------------------------------

"""
    _eobimr_nr_deltaT(nu, ap, am)

`NR_deltaT_NS(nu) + NR_deltaT(nu, ap, am)` (`fits_Hamiltonian.py`), `ap = X1 chi1 + X2 chi2`,
`am = X1 chi1 - X2 chi2`. Positive = attachment before `t_ISCO`.
"""
@inline function _eobimr_nr_deltaT(nu, ap, am)
    ns = nu^(-1.0 / 5.0 + 1.00513217e01 * nu) *
         (-5.96231800e01 - 1.05687385e03 * nu - 9.79317619e03 * nu^2 + 5.55652392e04 * nu^3)
    s = nu^(-1.0 / 5.0) * (
        8.39238879807543 * am^2 * ap - 16.9056858928167 * am^2 * nu + 7.23410583477034 * am^2 +
        6.38975598319936 * am * ap^2 + 179.569824846781 * am * ap * nu - 40.6063653476775 * am * ap +
        144.253395844761 * am * nu^2 - 90.1929138487509 * am * nu + 14.2203101910927 * am -
        6.78913884987037 * ap^4 + 5.39962303470497 * ap^3 - 132.224950777226 * ap^2 * nu +
        49.8016443361381 * ap^2 + 384.201018794943 * ap * nu^2 - 141.253181790353 * ap * nu +
        17.5710132409988 * ap)
    return ns + s
end

# `chi` of the input-value and merger-ringdown fits: chiS + chiA dm/(1 - 2nu), dm = X1 - X2.
@inline function _eobimr_fit_chi(nu, X1, X2, chi1, chi2)
    chiS = 0.5 * (chi1 + chi2)
    chiA = 0.5 * (chi1 - chi2)
    return chiS + chiA * (X1 - X2) / (1.0 - 2.0 * nu)
end

"""
    _eobimr_iv22(nu, chi) -> (habs, hdotdot, omega, omegadot)

(2,2) input-value fits at the attachment point (`IV_fits.py`: `habs`, `hdotdot`, `omega`, `omegadot`;
`hdot` is 0 for (2,2)). Amplitudes are `|h22|/nu`; `omega` < 0 in `pyseobnr`'s phase convention.
"""
@inline function _eobimr_iv22(nu, chi)
    habs = abs(71.97969776036882194603 * nu^4 - 13.35761402231352157344 * nu^3 * chi -
               46.87585958426210908101 * nu^3 + 0.61988944517825661507 * nu^2 * chi^2 +
               7.19426416189229733789 * nu^2 * chi + 12.44040490932310127903 * nu^2 +
               0.43014673069078152023 * nu * chi^3 - 1.74313546783413597652 * nu * chi -
               0.86828935763242798274 * nu - 0.08493901280736430859 * chi^3 -
               0.02082621429567295401 * chi^2 + 0.18693991146784910695 * chi + 1.46709663479911811557)
    hdd = -0.00335300225882774906 * nu^2 + 0.00358851415965951012 * nu * chi -
          0.00561520957901851664 * nu + 0.00038615328462788281 * chi^2 +
          0.00132564277249644174 * chi - 0.00245697909115198589
    om = 5.89352329617707670906 * nu^4 + 3.75145580491965446868 * nu^3 * chi -
         3.34930536209472551334 * nu^3 - 0.97140932625194231775 * nu^2 * chi^2 -
         1.69734302394369973577 * nu^2 * chi + 0.28539204856044564362 * nu^2 +
         0.2419483723662931296 * nu * chi^3 + 0.51801427018052081941 * nu * chi^2 +
         0.25096450064948544467 * nu * chi - 0.31709602351033533418 * nu -
         0.01525897668158244028 * chi^4 - 0.06692658483513916345 * chi^3 -
         0.08715176045684569495 * chi^2 - 0.09133931944098934441 * chi - 0.2685414392185025978
    omd = -0.23712612963269574795 * nu^3 + 0.07799016321986182443 * nu^2 * chi +
          0.09221479145462828375 * nu^2 - 0.00839285104015016943 * nu * chi^2 -
          0.02877175649350346628 * nu * chi - 0.03103970973029888253 * nu +
          0.00061394267373741083 * chi^3 + 0.0019481328417233967 * chi^2 +
          0.0017051416772119448 * chi - 0.0054839528158373476
    return habs, hdd, om, omd
end

"""
    _eobimr_mr22(nu, chi) -> (c1f, c2f, d1f, d2f)

(2,2) merger-ringdown ansatz fits (`MR_fits.py`).
"""
@inline function _eobimr_mr22(nu, chi)
    c1f = -4.23824640099272276217 * nu^4 + 1.86797765630645606905 * nu^3 * chi +
          2.04371156181773017124 * nu^3 + 0.01316051994812337048 * nu^2 * chi^2 -
          0.70248812998796750229 * nu^2 * chi - 0.40699224653253718298 * nu^2 +
          0.06284152952186422558 * nu * chi^3 + 0.04938791239730106614 * nu * chi^2 +
          0.03388527758147390212 * nu * chi + 0.05358902726316702725 * nu -
          0.00177674004113037185 * chi^4 - 0.01890815113487190682 * chi^3 -
          0.01931426101231131093 * chi^2 - 0.01161150126773277842 * chi + 0.08625435880606084627
    c2f = -63.28645899089733006804 * nu^4 + 2.00294725303467924249 * nu^3 * chi +
          44.33138899436394098075 * nu^3 - 3.55617293922388588712 * nu^2 * chi^2 -
          5.58585057654383287939 * nu^2 * chi - 9.5295732728313318205 * nu^2 +
          1.02187518454288950309 * nu * chi^3 + 1.97008188121834493245 * nu * chi^2 +
          1.83772448389004638969 * nu * chi + 1.15569522525235401922 * nu -
          0.20348032514327910047 * chi^3 - 0.2642970192733161694 * chi^2 -
          0.27076037187561419195 * chi - 0.52876279548305116229
    d1f = -28.42370101139921700906 * nu^4 + 4.11346289839689127632 * nu^3 * chi +
          20.71987362022024470321 * nu^3 + 1.03335215030655280799 * nu^2 * chi^2 -
          1.65292430358775521704 * nu^2 * chi - 6.07567868511363951001 * nu^2 +
          0.04730524488221983515 * nu * chi^3 - 0.254350860993373451 * nu * chi^2 +
          0.09083410987717309426 * nu * chi + 0.78009259453928059269 * nu -
          0.01332056979451640664 * chi^4 - 0.0242033556149483034 * chi^3 -
          0.00784682245346276369 * chi^2 + 0.1357578010277912528
    d2f = exp(-352.24938296898454836992 * nu^4 + 9.05730635731021394008 * nu^3 * chi +
              275.84349920209979245556 * nu^3 + 23.975132253988164166 * nu^2 * chi^2 -
              5.26829618908132601973 * nu^2 * chi - 81.48331396357356481985 * nu^2 -
              3.39885766491276442025 * nu * chi^3 - 10.06495407151063048445 * nu * chi^2 +
              0.46455322692280037744 * nu * chi + 11.18457585889310479388 * nu -
              0.1631129108825159213 * chi^4 + 0.728816370357556087 * chi^3 +
              1.2114999080794128794 * chi^2 + 0.56269034372727599891 * chi + 0.03570970180918431325)
    return c1f, c2f, d1f, d2f
end

"""
    _eobimr_final_mass_uib2016(X1, X2, chi1, chi2) -> Mf/M

UIB2016 (v2) aligned-spin final-mass fit, Jimenez-Forteza et al. arXiv:1611.00332 (`nr_utils.py`
`bbh_final_mass_non_precessing_UIB2016`, verbatim, `m1 + m2 = 1`, `m1 >= m2`).
"""
@inline function _eobimr_final_mass_uib2016(X1, X2, chi1, chi2)
    eta = min(X1 * X2, 0.25)
    eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta2 * eta2
    Shat = (chi1 * X1^2 + chi2 * X2^2) / (X1^2 + X2^2)
    Shat2 = Shat * Shat; Shat3 = Shat2 * Shat
    chidiff = chi1 - chi2
    sqrt2 = sqrt(2.0)
    sqrt1m4eta = X1 - X2          # = sqrt(1 - 4 eta) for X1 >= X2, exact and smooth at X1 = X2
    b10, b20, b30, b50 = 0.346, 0.211, 0.128, -0.212
    a2, a3, a4 = 0.5609904135313374, -0.84667563764404, 3.145145224278187
    b1, b2, b3, b5 = -0.2091189048177395, -0.19709136361080587, -0.1588185739358418, 2.9852925538232014
    f20, f30, f50, f10, f21 = 4.271313308472851, 31.08987570280556, 1.5673498395263061, 1.8083565298668276, 0.0
    d10, d11, d20, d30, d31 = -0.09803730445895877, -3.2283713377939134, 0.01118530335431078,
                              -0.01978238971523653, -4.91667749015812
    f11, f31, f51 = 15.738082204419655, -243.6299258830685, -0.5808669012986468
    Erad = ((1.0 - 2.0 / 3.0 * sqrt2) * eta + a2 * eta2 + a3 * eta3 + a4 * eta4) *
           (1.0 + b10 * b1 * Shat * (f10 + f11 * eta + (16.0 - 16.0 * f10 - 4.0 * f11) * eta2) +
                  b20 * b2 * Shat2 * (f20 + f21 * eta + (16.0 - 16.0 * f20 - 4.0 * f21) * eta2) +
                  b30 * b3 * Shat3 * (f30 + f31 * eta + (16.0 - 16.0 * f30 - 4.0 * f31) * eta2)) /
           (1.0 + b50 * b5 * Shat * (f50 + f51 * eta + (16.0 - 16.0 * f50 - 4.0 * f51) * eta2)) +
           d10 * sqrt1m4eta * eta2 * (1.0 + d11 * eta) * chidiff +
           d30 * Shat * sqrt1m4eta * eta * (1.0 + d31 * eta) * chidiff +
           d20 * eta3 * chidiff * chidiff
    return 1.0 - Erad
end

"""
    _eobimr_qnm220(X1, X2, chi1, chi2) -> (omega_R, tau, Mf, af)

(2,2,0) ringdown angular frequency `omega_R` and damping time `tau` in units of the total mass `M`,
from the UIB2016 final mass and HBR2016 final spin, via the IMRPhenomX QNM fits (`_xas_fRD_fdamp`,
which already divide by `Mf`: `fRD = omega_R/(2 pi)`, `fdamp = 1/(2 pi tau)`).
"""
@inline function _eobimr_qnm220(X1, X2, chi1, chi2)
    af = _seobv5_final_spin_hbr2016(X1, X2, chi1, chi2)
    Mf = _eobimr_final_mass_uib2016(X1, X2, chi1, chi2)
    fRD, fdamp = _xas_fRD_fdamp(af, 1.0 - Mf)
    return 2.0 * pi * fRD, 1.0 / (2.0 * pi * fdamp), Mf, af
end

# ---------------------------------------------------------------------------------------------------
# Small numerical helpers (constant trip counts throughout).
# ---------------------------------------------------------------------------------------------------

# searchsortedlast over a sorted vector of length <= 4095 in 12 unrolled strides (SEOBNRv5PHM's node buffer has 1536
# entries). Unrolled, not a loop: Enzyme's in-kernel reverse tapes every loop iteration on the device malloc heap,
# serialized across threads (the SEOBNRv5PHM per-sample gradient ran ~1 s per walker on a V100 with the loop).
@inline function _eobimr_searchsortedlast(v::AbstractVector{Float64}, x)
    n = length(v)
    pos = 0
    @inbounds Base.Cartesian.@nexprs 12 j -> begin
        cand = pos + (1 << (12 - j))
        pos = ifelse((cand <= n) && (v[min(cand, n)] <= x), cand, pos)
    end
    return pos
end

# Cubic Hermite value/derivative on [t0, t1] (h = t1 - t0) at s in [0, 1].
@inline function _eobimr_hermite(y0, dy0, y1, dy1, h, s)
    s2 = s * s; s3 = s2 * s
    val = (2s3 - 3s2 + 1) * y0 + (s3 - 2s2 + s) * h * dy0 + (-2s3 + 3s2) * y1 + (s3 - s2) * h * dy1
    der = ((6s2 - 6s) * y0 + (3s2 - 4s + 1) * h * dy0 + (-6s2 + 6s) * y1 + (3s2 - 2s) * h * dy1) / h
    return val, der
end

# C1 smoothstep weight on [0, 1].
@inline _eobimr_smoothstep(x) = (y = clamp(x, 0.0, 1.0); y * y * (3.0 - 2.0 * y))

# Planck-taper-like C-infinity turn-on over [0, 1] (0 at x <= 0, 1 at x >= 1).
@inline function _eobimr_turn_on(x)
    x <= 0.0 && return 0.0
    x >= 1.0 && return 1.0
    z = 1.0 / x - 1.0 / (1.0 - x)
    # 1 / (1 + e^z) as (1 - tanh(z/2)) / 2: no overflow anywhere, so no 0 * Inf = NaN in derivatives (the former
    # exp form, guarded at |z| > 700, still gave NaN in Enzyme's in-kernel reverse on CUDA for z ~ 450..660, where
    # (1 + e^z)^2 overflows; scripts/experiments_turnon_cuda_grad.jl in GravitationalWaveBayes).
    return (1.0 - tanh(z / 2)) / 2
end

"""
    _eobimr_fft!(re, im)

In-place iterative radix-2 FFT with the `+i` sign, `X_j = sum_k x_k e^{+2 pi i j k / N}`, for a
power-of-two `N`. Plain loops over `MVector`s (constant trip counts; Enzyme differentiates it).
"""
function _eobimr_fft!(re::AbstractVector{Float64}, im::AbstractVector{Float64})
    N = length(re)              # a constant for MVectors; heap vectors (SEOBNRv5HM's 4096-sample buffers) work too
    nbits = trailing_zeros(N)
    @inbounds for i in 0:N-1
        j = bitreverse(i) >>> (8 * sizeof(Int) - nbits)
        if j > i
            re[i + 1], re[j + 1] = re[j + 1], re[i + 1]
            im[i + 1], im[j + 1] = im[j + 1], im[i + 1]
        end
    end
    m = 2
    @inbounds while m <= N
        half = m >>> 1
        wr0, wi0 = cos(2pi / m), sin(2pi / m)
        wr, wi = 1.0, 0.0
        for k in 0:half-1
            for g in 0:m:N-1
                a = g + k + 1
                b = a + half
                tr = wr * re[b] - wi * im[b]
                ti = wr * im[b] + wi * re[b]
                re[b] = re[a] - tr; im[b] = im[a] - ti
                re[a] += tr; im[a] += ti
            end
            wr, wi = wr * wr0 - wi * wi0, wr * wi0 + wi * wr0
        end
        m <<= 1
    end
    return nothing
end

# ---------------------------------------------------------------------------------------------------
# Model hooks. The IMR pipeline below is shared by every EOB dynamics that provides these methods
# (`m` is the model instance): per-binary scalars `_imr_base` (must carry `eta, X1, X2, chi1, chi2, M,
# M_sec, r0`), the PA table `_imr_pa`, the equations of motion `_imr_rhs4 -> (rdot, phidot, prdot,
# pphidot)`, the pre-NQC inspiral |h22| (units M/dL) `_imr_amp`, the short-inspiral start state
# `_imr_short_start -> (r, pr, pphi)` used when the PA grid is too short, the plunge step rule
# `_imr_plunge_steps -> (kph, kphl, om1, om2, kr, hmin, hmax, hend)` (see `_imr_plunge_h`), the inspiral
# mode's own phase beyond -2 phi `_imr_phase_corr(m, r, pr, pphi, om, base)` (e.g. SEOBNRv5's tail phase +
# delta22; h22 = |h22| e^{-i (2 phi - phasecorr)}), and the phase-NQC shrinkage scale `_imr_nqc_phase_scale`
# (see `_eobimr_nqc`; `Inf` disables it).
# ---------------------------------------------------------------------------------------------------

const _EOBIMR_H22N = 8.0 * sqrt(pi / 5.0)

# The plunge step at orbital frequency `om`, radius `r`, `rdot`: at most `k / Omega` radians of orbital phase, with `k`
# going linearly from `kph` to `kphl` as Omega goes from `om1` to `om2`, and at most a fraction `kr` of r, clamped to
# [hmin, hmax] and capped by `hcap`. A step that ends past the frequency peak or at dr/dt >= 0 (with a finite state at
# r > 1.4) is retried with half the step (`hcap`, kept for all later steps) while the step is above `hend` (`hend = Inf`:
# stop at the first such step). Shared by
# `_eobimr_dynamics` and the GravitationalWaveBayes device plunge.
const _EOBIMR_R_STOP = 1.4    # plunge stopping radius (pyseobnr's r < 1.4 condition)
# An RK stage's radius: below the stopping radius the current (valid) radius instead; the step is then rejected.
@inline _imr_stage_r(rs, r) = ifelse(rs > _EOBIMR_R_STOP, rs, r)

@inline function _imr_plunge_h(sp, om, r, rdot, hcap)
    k = sp.kph + (sp.kphl - sp.kph) * clamp((om - sp.om1) / (sp.om2 - sp.om1), 0.0, 1.0)
    return min(clamp(min(k / om, sp.kr * r / abs(rdot)), sp.hmin, sp.hmax), hcap)
end

# ---------------------------------------------------------------------------------------------------
# Dynamics: PA nodes + RK4 plunge nodes in one fixed buffer with dense (cubic Hermite) output. The
# pre-NQC |h22| is evaluated once per node (and interpolated), so nothing downstream calls the
# Hamiltonian again.
# ---------------------------------------------------------------------------------------------------

"""
    _eobimr_dynamics(m, base, pa) -> NamedTuple

Joins the PA table (if usable) and a fixed-`_EOBIMR_NRK` adaptive RK4 plunge into one time-ordered node
buffer `(t, r, phi, pr, pphi)` with their time derivatives, plus the pre-NQC `|h22|` and its time
derivative per node (for dense output); `n` valid nodes.
"""
function _eobimr_dynamics(m, base, pa)
    ND = _EOBIMR_NDYN
    t = zero(MVector{ND,Float64}); r = zero(MVector{ND,Float64}); ph = zero(MVector{ND,Float64})
    pr = zero(MVector{ND,Float64}); pp = zero(MVector{ND,Float64})
    dr = zero(MVector{ND,Float64}); dph = zero(MVector{ND,Float64})
    dpr = zero(MVector{ND,Float64}); dpp = zero(MVector{ND,Float64})
    amp = zero(MVector{ND,Float64}); damp = zero(MVector{ND,Float64})
    phc = zero(MVector{ND,Float64}); dphc = zero(MVector{ND,Float64})

    use_pa = pa.success
    npa = use_pa ? pa.nreal : 1
    r0, pr0, pphi0 = _imr_short_start(m, base)
    @inbounds for i in 1:_SEOBV5_PA_NMAX
        if i <= npa
            ri = use_pa ? pa.r[i] : r0
            pri = use_pa ? pa.pr[i] : pr0
            ppi = use_pa ? pa.pphi[i] : pphi0
            t[i] = use_pa ? pa.t[i] : 0.0
            ph[i] = use_pa ? pa.phi[i] : 0.0
            r[i] = ri; pr[i] = pri; pp[i] = ppi
            a, b, c, d = _imr_rhs4(m, ri, pri, ppi, base)
            dr[i] = a; dph[i] = b; dpr[i] = c; dpp[i] = d
        end
    end

    # RK4 from node `npa`. Steps after the stop condition are masked (constant trip count).
    sp = _imr_plunge_steps(m)
    sr, sph, spr, spp, st = r[npa], ph[npa], pr[npa], pp[npa], t[npa]
    om_prev = dph[npa]
    stopped = false
    hcap = sp.hmax
    n = npa
    @inbounds for k in 1:_EOBIMR_NRK
        if !stopped
            k1r, k1f, k1p, k1j = _imr_rhs4(m, sr, spr, spp, base)
            h = _imr_plunge_h(sp, k1f, sr, k1r, hcap)
            # stages below the stopping radius are evaluated at the current state and invalidate the step (the
            # Hamiltonian is singular not far inside r = 1.4; the device kernel gives the same, finite, result)
            r2 = sr + 0.5h * k1r
            k2r, k2f, k2p, k2j = _imr_rhs4(m, _imr_stage_r(r2, sr), spr + 0.5h * k1p, spp + 0.5h * k1j, base)
            r3 = sr + 0.5h * k2r
            k3r, k3f, k3p, k3j = _imr_rhs4(m, _imr_stage_r(r3, sr), spr + 0.5h * k2p, spp + 0.5h * k2j, base)
            r4 = sr + h * k3r
            k4r, k4f, k4p, k4j = _imr_rhs4(m, _imr_stage_r(r4, sr), spr + h * k3p, spp + h * k3j, base)
            nr = sr + h / 6.0 * (k1r + 2k2r + 2k3r + k4r)
            nph = sph + h / 6.0 * (k1f + 2k2f + 2k3f + k4f)
            npr = spr + h / 6.0 * (k1p + 2k2p + 2k3p + k4p)
            npp = spp + h / 6.0 * (k1j + 2k2j + 2k3j + k4j)
            okr = isfinite(nr) & isfinite(npr) & isfinite(npp) & (nr > _EOBIMR_R_STOP) & (npp > 0.0) &
                  (min(r2, r3, r4) > _EOBIMR_R_STOP)
            ok = okr
            if ok
                a, b, c, d = _imr_rhs4(m, nr, npr, npp, base)
                ok = (b > om_prev) & (a < 0.0)
                if ok
                    n += 1
                    st += h
                    sr, sph, spr, spp = nr, nph, npr, npp
                    t[n] = st; r[n] = nr; ph[n] = nph; pr[n] = npr; pp[n] = npp
                    dr[n] = a; dph[n] = b; dpr[n] = c; dpp[n] = d
                    om_prev = b
                end
            end
            # halving only before the orbital-frequency peak / dr/dt >= 0 stop, not at r < 1.4 or a non-finite step
            retry = okr & !ok & (h > sp.hend)
            hcap = ifelse(retry, 0.5h, hcap)
            stopped = !ok & !retry
        end
    end
    @inbounds for i in 1:ND
        if i <= n
            amp[i] = _imr_amp(m, r[i], pr[i], pp[i], dph[i], base)
            phc[i] = _imr_phase_corr(m, r[i], pr[i], pp[i], dph[i], base)
        else
            t[i] = t[n] + Float64(i - n)     # strictly increasing padding for the bracket search
        end
    end
    # d|h22|/dt and d(phasecorr)/dt per node: 3-point non-uniform-grid derivative (one-sided at the ends).
    @inbounds for i in 1:ND
        if i <= n
            if i == 1
                damp[i] = (amp[2] - amp[1]) / (t[2] - t[1])
                dphc[i] = (phc[2] - phc[1]) / (t[2] - t[1])
            elseif i == n
                damp[i] = (amp[n] - amp[n - 1]) / (t[n] - t[n - 1])
                dphc[i] = (phc[n] - phc[n - 1]) / (t[n] - t[n - 1])
            else
                h1 = t[i] - t[i - 1]; h2 = t[i + 1] - t[i]
                c1 = -h2 / (h1 * (h1 + h2)); c2 = (h2 - h1) / (h1 * h2); c3 = h1 / (h2 * (h1 + h2))
                damp[i] = c1 * amp[i - 1] + c2 * amp[i] + c3 * amp[i + 1]
                dphc[i] = c1 * phc[i - 1] + c2 * phc[i] + c3 * phc[i + 1]
            end
        end
    end
    return (t = SVector(t), r = SVector(r), phi = SVector(ph), pr = SVector(pr), pphi = SVector(pp),
        dr = SVector(dr), dphi = SVector(dph), dpr = SVector(dpr), dpphi = SVector(dpp),
        amp = SVector(amp), damp = SVector(damp), phc = SVector(phc), dphc = SVector(dphc),
        nqc_scale = _imr_nqc_phase_scale(m), n = n, npa = npa, stopped = stopped)
end

# Dense output at time `tq` (clamped to the valid node range): (r, phi, pr, pphi, Omega, |h22|, dr/dt).
# Omega is the derivative of the phase interpolant, so d(phase)/dt = Omega holds exactly.
@inline function _eobimr_dyn_at(dyn, tq)
    n = dyn.n
    tq = clamp(tq, dyn.t[1], dyn.t[n])
    i = clamp(_eobimr_searchsortedlast(dyn.t, tq), 1, n - 1)
    h = dyn.t[i + 1] - dyn.t[i]
    s = (tq - dyn.t[i]) / h
    r, rdot = _eobimr_hermite(dyn.r[i], dyn.dr[i], dyn.r[i + 1], dyn.dr[i + 1], h, s)
    ph, om = _eobimr_hermite(dyn.phi[i], dyn.dphi[i], dyn.phi[i + 1], dyn.dphi[i + 1], h, s)
    pr, _ = _eobimr_hermite(dyn.pr[i], dyn.dpr[i], dyn.pr[i + 1], dyn.dpr[i + 1], h, s)
    pp, _ = _eobimr_hermite(dyn.pphi[i], dyn.dpphi[i], dyn.pphi[i + 1], dyn.dpphi[i + 1], h, s)
    A, _ = _eobimr_hermite(dyn.amp[i], dyn.damp[i], dyn.amp[i + 1], dyn.damp[i + 1], h, s)
    pc, dpc = _eobimr_hermite(dyn.phc[i], dyn.dphc[i], dyn.phc[i + 1], dyn.dphc[i + 1], h, s)
    return r, ph, pr, pp, om, A, rdot, pc, dpc
end

# ---------------------------------------------------------------------------------------------------
# (2,2) mode: NQC factors and the full inspiral-plunge mode.
# ---------------------------------------------------------------------------------------------------

# NQC building blocks (pyseobnr's q1, q2, q3 amplitude factors and p1, p2 phase factors).
@inline function _eobimr_nqc_q(r, pr, om)
    rO = r * om
    q1 = pr * pr / (rO * rO)
    return q1, q1 / r, q1 / (r * sqrt(r))
end
@inline _eobimr_nqc_p(r, pr, om) = (pr / (r * om), pr^3 / (r * om))

# Full inspiral-plunge mode at time tq: (|h22|, phase varphi), h22 = |h22| e^{-i varphi}.
@inline function _eobimr_mode_at(dyn, nqc, tq)
    r, ph, pr, _, om, A, _, pc, _ = _eobimr_dyn_at(dyn, tq)
    q1, q2, q3 = _eobimr_nqc_q(r, pr, om)
    p1, p2 = _eobimr_nqc_p(r, pr, om)
    mag = 1.0 + q1 * nqc.a1 + q2 * nqc.a2 + q3 * nqc.a3
    return A * mag, 2.0 * ph - pc - (nqc.b1 * p1 + nqc.b2 * p2)
end

# Central finite-difference value/first/second derivative from samples at t0 - h, t0, t0 + h.
@inline _eobimr_d012(ym, y0, yp, h) = (y0, (yp - ym) / (2h), (yp - 2.0 * y0 + ym) / (h * h))

# NQC basis at one time: (A, A q1, A q2, A q3, p1, p2, 2 phi), all from the pre-NQC inspiral mode.
@inline function _eobimr_nqc_basis(dyn, tq)
    r, ph, pr, _, om, A, _, pc, _ = _eobimr_dyn_at(dyn, tq)
    q1, q2, q3 = _eobimr_nqc_q(r, pr, om)
    p1, p2 = _eobimr_nqc_p(r, pr, om)
    return A, A * q1, A * q2, A * q3, p1, p2, 2.0 * ph - pc
end

"""
    _eobimr_nqc(dyn, t_a, nu, chi) -> (a1, a2, a3, b1, b2, shrink)

Solve the SEOBNRv5 NQC conditions at `t_a` (3x3 amplitude and 2x2 frequency systems by Cramer's rule).
Time derivatives by central differences of the dense output (step 0.5 M). No closures: Enzyme reverse
mode hit an argument-classification assertion with closure-capturing finite differences.
"""
function _eobimr_nqc(dyn, t_a, nu, chi)
    habs, hdd, om_fit, omd_fit = _eobimr_iv22(nu, chi)
    h = 0.5
    m = _eobimr_nqc_basis(dyn, t_a - h)
    z = _eobimr_nqc_basis(dyn, t_a)
    p = _eobimr_nqc_basis(dyn, t_a + h)
    A0, A1, A2 = _eobimr_d012(m[1], z[1], p[1], h)
    Q11, Q21, Q31 = _eobimr_d012(m[2], z[2], p[2], h)
    Q12, Q22, Q32 = _eobimr_d012(m[3], z[3], p[3], h)
    Q13, Q23, Q33 = _eobimr_d012(m[4], z[4], p[4], h)
    R1, R2, R3 = nu * habs - A0, 0.0 - A1, nu * hdd - A2
    det = Q11 * (Q22 * Q33 - Q23 * Q32) - Q12 * (Q21 * Q33 - Q23 * Q31) + Q13 * (Q21 * Q32 - Q22 * Q31)
    a1 = (R1 * (Q22 * Q33 - Q23 * Q32) - Q12 * (R2 * Q33 - Q23 * R3) + Q13 * (R2 * Q32 - Q22 * R3)) / det
    a2 = (Q11 * (R2 * Q33 - Q23 * R3) - R1 * (Q21 * Q33 - Q23 * Q31) + Q13 * (Q21 * R3 - R2 * Q31)) / det
    a3 = (Q11 * (Q22 * R3 - R2 * Q32) - Q12 * (Q21 * R3 - R2 * Q31) + R1 * (Q21 * Q32 - Q22 * Q31)) / det
    _, w0, wd0 = _eobimr_d012(m[7], z[7], p[7], h)         # inspiral frequency 2 Omega and its derivative
    _, P11, P21 = _eobimr_d012(m[5], z[5], p[5], h)
    _, P12, P22 = _eobimr_d012(m[6], z[6], p[6], h)
    P11, P21, P12, P22 = -P11, -P21, -P12, -P22
    S1 = abs(om_fit) - w0
    S2 = abs(omd_fit) - wd0
    detp = P11 * P22 - P12 * P21
    b1 = (S1 * P22 - P12 * S2) / detp
    b2 = (P11 * S2 - S1 * P21) / detp
    # Optional smooth shrinkage (`_imr_nqc_phase_scale`): where the dynamics' plunge differs a lot from NR
    # (small pr near t_a) the frequency conditions need |b| >> 1, and the correction b1 pr/(r Omega) then
    # dominates the phase long before merger. Scale (b1, b2) down smoothly once the correction 150 M
    # before t_a exceeds ~dyn.nqc_scale rad; the merger-ringdown attaches continuously to whatever
    # frequency results. With an infinite scale (SEOBNRv5) shrink == 1 exactly.
    x = _eobimr_dyn_at(dyn, t_a - _EOBIMR_TBLEND[1])
    p1x, p2x = _eobimr_nqc_p(x[1], x[3], x[5])
    # `ifelse`, not `x / Inf`: same value (0 for SEOBNRv5's Inf scale), but forward-mode AD of `x / Inf` is
    # (dx Inf - x 0) / Inf^2 = NaN (seen in GravitationalWaveBayes' forward-mode device table).
    delta = ifelse(isinf(dyn.nqc_scale), 0.0, (b1 * p1x + b2 * p2x) / dyn.nqc_scale)
    shrink = 1.0 / sqrt(1.0 + delta * delta)
    return (a1 = a1, a2 = a2, a3 = a3, b1 = shrink * b1, b2 = shrink * b2, shrink = shrink)
end

# Merger-ringdown mode at dt = t - t_a >= 0: (|h22|, varphi, dvarphi/dt), h22 = |h22| e^{-i varphi}.
@inline function _eobimr_mr_at(mr, dt)
    A = mr.nu * (mr.c1c * tanh(mr.c1f * dt + mr.c2f) + mr.c2c) * exp(-dt / mr.tau)
    e = mr.d2f * exp(-mr.d1f * dt)
    varphi = mr.phi_a + mr.omR * dt + mr.d1c * log((1.0 + e) / (1.0 + mr.d2f))
    w = mr.omR - mr.d1c * mr.d1f * e / (1.0 + e)
    return A, varphi, w
end

# ---------------------------------------------------------------------------------------------------
# Frequency-domain table.
# ---------------------------------------------------------------------------------------------------

"""
    EOBIMRTable

Per-binary frequency-domain representation built by [`_eobimr_coeffs`](@ref). Fields shared with the
inspiral-only PA-F2 `coeffs` (`success, M_sec, eta, M, nreal, ts, phis, omegas, fmin, fmax`) keep their
meaning, except that `ts/phis/omegas` (orbital-equivalent: `phis = varphi/2`, `omegas = (dvarphi/dt)/2`
of the NQC-corrected (2,2) mode) end at the blend band (`fspa_max`), while `fmax` is the top of the
Fourier table. `Qs/dQs`: the SPA amplitude factor `Q = R / sqrt(d omega/dt)` ([`_spa_node_q`](@ref)), `R` the
amplitude relative to the Newtonian `8 sqrt(pi/5) nu x` at the node's own frequency, and its time derivative. `ft_*`: the Fourier table (frequencies Hz, amplitude and phase of
`U(f) = e^{-2 pi i f t_a} htilde(f) dL/(M K)` in seconds, and their frequency derivatives).
"""
# Time origin: the phase is referred to t_attach (~ the (2,2) amplitude peak), so a likelihood's t_c is the
# merger time, as for other IMR models. (`ts` stay measured from the start of the dynamics; subtract
# `t_a` for peak-relative times.) The start of the dynamics would be
# self-consistent for injections but puts real-data t_c priors (around the trigger) seconds away from
# the signal.
struct EOBIMRTable{NT,NF}
    success::Bool
    M_sec::Float64
    eta::Float64
    M::Float64
    nreal::Int
    ts::SVector{NT,Float64}
    phis::SVector{NT,Float64}
    omegas::SVector{NT,Float64}
    Qs::SVector{NT,Float64}
    dQs::SVector{NT,Float64}
    fmin::Float64
    fmax::Float64
    fspa_max::Float64
    f_b1::Float64
    f_b2::Float64
    t_a::Float64
    ft_shift::Float64
    ft_f::SVector{NF,Float64}
    ft_amp::SVector{NF,Float64}
    ft_damp::SVector{NF,Float64}
    ft_arg::SVector{NF,Float64}
    ft_darg::SVector{NF,Float64}
end

# Full (2,2) mode (inspiral-plunge with NQC, then merger-ringdown): (|h22|, varphi).
@inline function _eobimr_full_mode(dyn, nqc, mr, t_a, tq)
    if tq <= t_a
        return _eobimr_mode_at(dyn, nqc, tq)
    else
        A, vp, _ = _eobimr_mr_at(mr, tq - t_a)
        return A, vp
    end
end

# NQC phase correction at time tq.
@inline function _eobimr_nqc_phase(dyn, nqc, tq)
    x = _eobimr_dyn_at(dyn, tq)
    p = _eobimr_nqc_p(x[1], x[3], x[5])
    return nqc.b1 * p[1] + nqc.b2 * p[2]
end

# GW frequency dvarphi/dt of the NQC-corrected inspiral mode: 2 Omega minus the NQC phase rate (FD).
@inline function _eobimr_freq_insp(dyn, nqc, tq)
    x = _eobimr_dyn_at(dyn, tq)
    om = x[5] - 0.5 * x[9]          # (2 Omega - d phasecorr/dt)/2
    return 2.0 * om - (_eobimr_nqc_phase(dyn, nqc, tq + 0.25) - _eobimr_nqc_phase(dyn, nqc, tq - 0.25)) / 0.5
end

# Amplitude relative to the Newtonian (2,2) amplitude at the mode's own frequency.
@inline function _eobimr_R(dyn, nqc, nu, tq)
    A = _eobimr_mode_at(dyn, nqc, tq)[1]
    w = _eobimr_freq_insp(dyn, nqc, tq)
    return A / (_EOBIMR_H22N * nu * cbrt(0.5 * w)^2)
end

"""
    _eobimr_mr_params(dyn, base, nqc, t_a, nu, chi) -> NamedTuple

Merger-ringdown ansatz parameters (`compute_MR_mode_free`), attached continuously in amplitude, its
derivative, phase and frequency to the NQC-corrected inspiral mode at `t_a`.
"""
function _eobimr_mr_params(dyn, base, nqc, t_a, nu, chi)
    Am = _eobimr_mode_at(dyn, nqc, t_a - 0.5)[1]
    A_a, phi_a = _eobimr_mode_at(dyn, nqc, t_a)
    Ap = _eobimr_mode_at(dyn, nqc, t_a + 0.5)[1]
    dA_a = (Ap - Am) / 1.0
    w_a = _eobimr_freq_insp(dyn, nqc, t_a)
    omR, tau, Mf, af = _eobimr_qnm220(base.X1, base.X2, base.chi1, base.chi2)
    c1f, c2f, d1f, d2f = _eobimr_mr22(nu, chi)
    sigR = -1.0 / tau
    c1f = sigR > 2.0 * c1f * tanh(c2f) ? sigR / (2.0 * tanh(c2f)) : c1f
    c1c = (dA_a - sigR * A_a) * cosh(c2f)^2 / (c1f * nu)
    c2c = A_a / nu - (dA_a - sigR * A_a) * cosh(c2f) * sinh(c2f) / (c1f * nu)
    d1c = (omR - w_a) * (1.0 + d2f) / (d1f * d2f)
    return (nu = nu, c1c = c1c, c1f = c1f, c2f = c2f, c2c = c2c, tau = tau, d1f = d1f, d2f = d2f,
        d1c = d1c, omR = omR, phi_a = phi_a, w_a = w_a, A_a = A_a, Mf = Mf, af = af)
end

"""
    _eobimr_build(m, mc, eta, chi1, chi2; f_min, GMsun_over_c3) -> NamedTuple

Everything up to the time-domain mode: dynamics, attachment time, NQC and merger-ringdown parameters.
"""
function _eobimr_build(m, mc, eta, chi1, chi2; f_min = 20.0, GMsun_over_c3 = GMsun_over_c3)
    # Start no later than r = _EOBIMR_R_START, so the Fourier segment's pre-roll (_EOBIMR_TSEG_PRE before
    # t_attach) always lies inside the dynamics: heavy binaries start below `f_min` (bins below f_min
    # are never evaluated by the likelihood, so this only removes the truncation artifact).
    Msec_est = Float64(mc) / Float64(eta)^0.6 * Float64(GMsun_over_c3)
    f_start = min(Float64(f_min), _EOBIMR_R_START^(-1.5) / (pi * Msec_est))
    base = _imr_base(m, mc, eta, chi1, chi2, f_start, Float64(GMsun_over_c3))
    in_domain = (0.0 < base.eta <= 0.25) & (abs(base.chi1) <= 1.0) & (abs(base.chi2) <= 1.0) &
                isfinite(base.r0)
    # Out-of-domain inputs: evaluate everything at a harmless placeholder binary, report failure.
    b = in_domain ? base : _imr_base(m, 20.0, 0.24, 0.0, 0.0, 20.0, Float64(GMsun_over_c3))
    nu, X1, X2, c1, c2 = b.eta, b.X1, b.X2, b.chi1, b.chi2
    pa = _imr_pa(m, b)
    dyn = _eobimr_dynamics(m, b, pa)
    n = dyn.n
    t_start, t_end = dyn.t[1], dyn.t[n]

    # Attachment time: t_ISCO - DeltaT_22, clamped into the dynamics.
    rI, _ = _seobv5_kerr_isco(c1, c2, X1, X2)
    i_is = 1
    @inbounds for i in 1:_EOBIMR_NDYN
        if i <= n && dyn.r[i] >= rI
            i_is = i
        end
    end
    t_isco = t_end
    if i_is < n
        # Bisection on the dense output inside [t_i, t_{i+1}] (r decreasing there).
        lo, hi = dyn.t[i_is], dyn.t[i_is + 1]
        for _ in 1:40
            mid = 0.5 * (lo + hi)
            if _eobimr_dyn_at(dyn, mid)[1] >= rI
                lo = mid
            else
                hi = mid
            end
        end
        tb = 0.5 * (lo + hi)
        # One Newton step at the converged root. Numerically a no-op, but it gives reverse/forward AD the
        # implicit-function derivative -(dr/dtheta)/(dr/dt): differentiating the bisection alone only
        # follows the bracket endpoints (d/deta would be 7% off finite differences).
        x = _eobimr_dyn_at(dyn, tb)
        rb, rdot = x[1], x[7]
        t_isco = tb - (rb - rI) / rdot
    end
    ap = X1 * c1 + X2 * c2
    am = X1 * c1 - X2 * c2
    t_a = clamp(t_isco - _eobimr_nr_deltaT(nu, ap, am), t_start + 1.0, t_end - 1.0)
    chi = _eobimr_fit_chi(nu, X1, X2, c1, c2)
    nqc = _eobimr_nqc(dyn, t_a, nu, chi)
    mr = _eobimr_mr_params(dyn, b, nqc, t_a, nu, chi)
    success = in_domain & (b.r0 >= 6.0) & dyn.stopped & (n >= dyn.npa + 8) &
              (t_a > t_start + 20.0) & isfinite(nqc.a1 + nqc.a2 + nqc.a3 + nqc.b1 + nqc.b2) &
              isfinite(mr.c1c + mr.c2c + mr.d1c) & (mr.w_a < mr.omR)
    return (base = b, dyn = dyn, t_a = t_a, t_isco = t_isco, nqc = nqc, mr = mr, success = success,
        t_start = t_start, t_end = t_end)
end

"""
    _eobimr_coeffs(model, mc, eta, chi1, chi2; f_min=20.0, GMsun_over_c3=GMsun_over_c3) -> EOBIMRTable

Build the whole IMR frequency-domain table for one binary of `model` (an `_imr_*` hook provider, e.g.
[`SEOBNRv5IMRAlignedSpinF2`](@ref)); see this file's header for the pipeline.
"""
function _eobimr_coeffs(model, mc, eta, chi1, chi2; f_min = 20.0,
        GMsun_over_c3 = GMsun_over_c3, kwargs...)
    NT, NF, NS = _EOBIMR_NT, _EOBIMR_NF, _EOBIMR_NSEG
    B = _eobimr_build(model, mc, eta, chi1, chi2; f_min = f_min, GMsun_over_c3 = GMsun_over_c3)
    b, dyn, t_a, nqc, mr = B.base, B.dyn, B.t_a, B.nqc, B.mr
    nu = b.eta
    Ms = b.M_sec
    t_start = B.t_start

    # --- SPA table: PA nodes, then _EOBIMR_NSPA_RK nodes uniform in time up to the blend band's top.
    t_spa_end = t_a - _EOBIMR_TBLEND[2]
    use_spa = t_spa_end > t_start + 10.0
    npa = dyn.npa
    npa_use = 1
    # PA nodes past t_spa_end (only possible for very short tables) are dropped.
    @inbounds for i in 1:_SEOBV5_PA_NMAX
        npa_use = (use_spa & (i <= npa) && (dyn.t[i] <= t_spa_end - 1.0)) ? i : npa_use
    end
    t_seg0 = dyn.t[npa_use]
    ts = zero(MVector{NT,Float64}); phis = zero(MVector{NT,Float64}); oms = zero(MVector{NT,Float64})
    Qs = zero(MVector{NT,Float64}); dQs = zero(MVector{NT,Float64})
    nspa = use_spa ? npa_use + _EOBIMR_NSPA_RK : 2
    t_top = use_spa ? t_spa_end : t_start + 2.0
    @inbounds for i in 1:NT
        if i <= nspa
            tq = i <= npa_use ? dyn.t[i] :
                 t_seg0 + (t_top - t_seg0) * Float64(i - npa_use) / Float64(nspa - npa_use)
            ts[i] = tq
            phis[i] = 0.5 * _eobimr_mode_at(dyn, nqc, tq)[2]
            oms[i] = 0.5 * _eobimr_freq_insp(dyn, nqc, tq)
            Qs[i] = _eobimr_R(dyn, nqc, nu, tq)            # R and dR/dt for now, Q below
            dQs[i] = _eobimr_R(dyn, nqc, nu, tq + 0.5) - _eobimr_R(dyn, nqc, nu, tq - 0.5)
        end
    end
    # R -> Q = R / sqrt(d omega/dt) (needs every node's omega first: the nodal d omega/dt reads its neighbours)
    @inbounds for i in 1:NT
        if i <= nspa
            Qs[i], dQs[i] = _spa_node_q(ts, oms, Qs[i], dQs[i], i, nspa)
        end
    end
    @inbounds for i in 1:NT
        if i > nspa
            ts[i] = ts[nspa] + Float64(i - nspa)
            oms[i] = oms[nspa] + Float64(i - nspa)
        end
    end
    fmin = oms[1] / (pi * Ms)
    fspa_max = oms[nspa] / (pi * Ms)

    # --- Fourier table of the last segment.
    t_w = max(t_start, t_a - _EOBIMR_TSEG_PRE)
    Lt = min(_EOBIMR_TTAPER, 0.4 * (t_a - t_w))
    t_e = t_a + _EOBIMR_NTAU * mr.tau
    dts = (t_e - t_w) / Float64(NS - 1)
    # Mode samples (complex, e^{-i varphi}) tapered at the start, and the same times (t - t_a).
    sre = zero(MVector{NS,Float64}); sim = zero(MVector{NS,Float64})
    tre = zero(MVector{NS,Float64}); tim = zero(MVector{NS,Float64})
    norm = Ms * dts / (4.0 * sqrt(pi / 5.0))
    @inbounds for k in 1:NS
        tk = t_w + dts * Float64(k - 1)
        A, vp = _eobimr_full_mode(dyn, nqc, mr, t_a, tk)
        wgt = _eobimr_turn_on((tk - t_w) / Lt) * norm * A
        sre[k] = wgt * cos(vp)
        sim[k] = -wgt * sin(vp)
        tre[k] = (tk - t_a) * sre[k]
        tim[k] = (tk - t_a) * sim[k]
    end
    # Two FFTs: U(f_j) = e^{2 pi i f_j Ms (t_w - t_a)} X_j and dU/df = 2 pi i Ms (FFT of (t - t_a) s),
    # f_j = j/(NS dts Ms) Hz. The table keeps NF consecutive bins starting just below the blend band.
    _eobimr_fft!(sre, sim)
    _eobimr_fft!(tre, tim)
    df = 1.0 / (Float64(NS) * dts * Ms)
    f_b1_est = use_spa ? _eobimr_freq_insp(dyn, nqc, t_a - _EOBIMR_TBLEND[1]) / (2pi * Ms) : fmin
    j0 = clamp(floor(Int, 0.8 * f_b1_est / df), 1, NS ÷ 2 - NF)
    # The table top must not depend on the integer window start j0 (that would make fmax jump with the
    # parameters), so it is set by the QNM alone; `covered` flags the (untested-for) case where the
    # window would not reach it.
    fhi = _EOBIMR_FHI_QNM * mr.omR / (2pi * Ms)
    covered = (Float64(j0 + NF - 1) * df >= fhi) & (fhi < 0.45 / (dts * Ms))
    ftf = zero(MVector{NF,Float64})
    amp = zero(MVector{NF,Float64}); damp = zero(MVector{NF,Float64})
    arg = zero(MVector{NF,Float64}); darg = zero(MVector{NF,Float64})
    @inbounds for j in 1:NF
        jj = j0 + j - 1
        f = Float64(jj) * df
        ftf[j] = f
        th = 2pi * f * Ms * (t_w - t_a)
        cth, sth = cos(th), sin(th)
        Xr, Xi = sre[jj + 1], sim[jj + 1]
        Ur = Xr * cth - Xi * sth
        Ui = Xr * sth + Xi * cth
        Vr = tre[jj + 1] * cth - tim[jj + 1] * sth
        Vi = tre[jj + 1] * sth + tim[jj + 1] * cth
        dUr = -2pi * Ms * Vi
        dUi = 2pi * Ms * Vr
        a2 = Ur * Ur + Ui * Ui
        amp[j] = sqrt(a2)
        damp[j] = (Ur * dUr + Ui * dUi) / amp[j]
        arg[j] = atan(Ui, Ur)
        darg[j] = (Ur * dUi - Ui * dUr) / a2
    end
    # Unwrap guided by the analytic phase derivative (trapezoid prediction).
    @inbounds for j in 2:NF
        pred = arg[j - 1] + 0.5 * (darg[j - 1] + darg[j]) * (ftf[j] - ftf[j - 1])
        arg[j] += 2pi * round((pred - arg[j]) / (2pi))
    end

    f_b1 = f_b1_est
    f_b2 = use_spa ? fspa_max : fmin
    success = B.success & covered
    tab0 = EOBIMRTable{NT,NF}(success, Ms, nu, b.M, nspa, SVector(ts), SVector(phis), SVector(oms),
        SVector(Qs), SVector(dQs), fmin, fhi, fspa_max, f_b1, f_b2, t_a, 0.0, SVector(ftf),
        SVector(amp), SVector(damp), SVector(arg), SVector(darg))
    # Align the Fourier-table phase to the SPA phase (multiple of 2 pi) in the middle of the blend band.
    fm = 0.5 * (f_b1 + f_b2)
    shift = use_spa ? 2pi * round((_eobimr_spa_eval(tab0, fm)[2] - _eobimr_ft_eval(tab0, fm)[2]) / (2pi)) : 0.0
    fmin_out = use_spa ? fmin : ftf[1]
    return EOBIMRTable{NT,NF}(success, Ms, nu, b.M, nspa, SVector(ts), SVector(phis), SVector(oms),
        SVector(Qs), SVector(dQs), fmin_out, fhi, fspa_max, f_b1, f_b2, t_a, shift, SVector(ftf),
        SVector(amp), SVector(damp), SVector(arg), SVector(darg))
end

# SPA part: (reduced amplitude, phase); physical amplitude = reduced * M * K / dL. The stationary point is
# `_spa_root` (one bracket search for phase and amplitude).
# `_eobimr_spa_eval`, `_eobimr_ft_eval` and `_eobimr_eval` are duck-typed: any `c` with the `EOBIMRTable`
# field names works (GravitationalWaveBayes' device tables pass NamedTuples of array views).
@inline function _eobimr_spa_eval(c, f)
    Ms = c.M_sec
    f = clamp(f, c.fmin, c.fspa_max)       # the SPA table ends at the blend band's top
    omega_target = pi * Ms * f
    i = clamp(_eobimr_searchsortedlast(c.omegas, omega_target), 1, c.nreal - 1)
    r = _spa_root(c.ts, c.phis, c.omegas, i, omega_target)
    Q, _ = _eobimr_hermite(c.Qs[i], c.dQs[i], c.Qs[i + 1], c.dQs[i + 1], c.ts[i + 1] - c.ts[i], r.s)
    v2 = cbrt(omega_target)^2
    # 2 nu v^2 R sqrt(2 pi / (d Omega_GW/dt)_phys) with (d Omega_GW/dt)_phys = 2 (d omega/dt) / Ms^2
    amp = 2.0 * c.eta * v2 * Ms * sqrt(pi) * Q
    psi = 2pi * f * (r.t_star - c.t_a) * Ms - 2.0 * r.phi_star - pi / 4.0      # time origin: t_attach
    return amp, psi
end

# Fourier-table part: (reduced amplitude, phase), with the C1 top-of-band taper.
@inline function _eobimr_ft_eval(c, f)
    j = clamp(_eobimr_searchsortedlast(c.ft_f, f), 1, length(c.ft_f) - 1)
    h = c.ft_f[j + 1] - c.ft_f[j]
    s = (f - c.ft_f[j]) / h
    a, _ = _eobimr_hermite(c.ft_amp[j], c.ft_damp[j], c.ft_amp[j + 1], c.ft_damp[j + 1], h, s)
    g, _ = _eobimr_hermite(c.ft_arg[j], c.ft_darg[j], c.ft_arg[j + 1], c.ft_darg[j + 1], h, s)
    tap = 1.0 - _eobimr_smoothstep((f - 0.8 * c.fmax) / (0.2 * c.fmax))
    return max(a, 0.0) * tap, g + c.ft_shift          # U already carries the t_attach time origin
end

@inline function _eobimr_eval(c, f)
    if f <= c.f_b1
        return _eobimr_spa_eval(c, f)
    elseif f >= c.f_b2
        return _eobimr_ft_eval(c, f)
    else
        a1, p1 = _eobimr_spa_eval(c, f)
        a2, p2 = _eobimr_ft_eval(c, f)
        w = _eobimr_smoothstep((f - c.f_b1) / (c.f_b2 - c.f_b1))
        return (1.0 - w) * a1 + w * a2, (1.0 - w) * p1 + w * p2
    end
end

# The PA-F2 scalar interface every GravitationalWaveBayes PAF2 kernel calls (same conventions: 0 outside the band).
@inline function _seobv5f2_phi_scalar(c::EOBIMRTable, f)
    !c.success && return 0.0
    (f < c.fmin || f > c.fmax) && return 0.0
    return _eobimr_eval(c, f)[2]
end

@inline function _seobv5f2_ampl_scalar(c::EOBIMRTable, f, dL, GMsun_over_c2_Gpc)
    !c.success && return 0.0
    (f < c.fmin || f > c.fmax) && return 0.0
    return _eobimr_eval(c, f)[1] * c.M * GMsun_over_c2_Gpc / dL
end

# Amplitude and phase from one table evaluation (the likelihood kernels' per-bin call).
@inline function _seobv5f2_amp_phi(c::EOBIMRTable, f, dL, GMsun_over_c2_Gpc)
    (!c.success || f < c.fmin || f > c.fmax) && return 0.0, 0.0
    a, p = _eobimr_eval(c, f)
    return a * c.M * GMsun_over_c2_Gpc / dL, p
end
