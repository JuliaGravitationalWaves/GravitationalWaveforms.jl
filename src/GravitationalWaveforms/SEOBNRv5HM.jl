export seobnrv5hm_td_modes

"""
SEOBNRv5HM (aligned spin, all default modes) -- time-domain construction.

The pyseobnr 0.3.7 `SEOBNRv5HM_opt._evaluate_model` pipeline for the six default modes
`(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)` ((5,5) is excluded by default in pyseobnr too), on top of the
dynamics, attachment time and machinery of [`SEOBNRv5IMRAlignedSpinF2`](@ref) (PA inspiral + RK4 plunge,
`t_attach = t_ISCO - DeltaT_22`):

1. **Inspiral modes** (`compute_mode`, `waveform.pyx`): `h_lm = T_lm e^{i delta_lm} S_lm rho_lm hNewton_lm`,
   stored per dynamics node as the slowly varying complex *reduced* mode `H_lm = h_lm e^{i m phi}` (real and
   imaginary parts with their time derivatives, cubic-Hermite dense output). A complex representation, not
   amplitude + phase: the odd-`m` modes' amplitudes can pass through zero (the reason for step 2).
2. **Special calibration coefficients** of `(2,1)`, `(4,3)` (`compute_special_coeffs`): the `v^7` term of
   `f_lm` set so that `|h_lm|(t_attach)` equals the signed input-value fit `hsign` (thresholded).
3. **NQC** per mode (`NQC_correction`, `EOBCalculateNQCCoefficients_freeattach`) against the input-value fits
   of that mode (`IV_fits.py`), all at `t_attach`.
4. **Merger-ringdown** per mode (`compute_IMR_modes`, `compute_MR_mode_free`), with the MR fits of that mode
   (`MR_fits.py`), the `(l,m,0)` QNM ([`_seobv5hm_qnm`](@ref): Chebyshev fits of the `qnm` interpolant
   pyseobnr uses), the input-value-fit frequency for odd `m`, and spherical-spheroidal **mode mixing** for
   `(3,2)` and `(4,3)` (`compute_mixed_mode`, Berti-Klein 2014 mixing coefficients).

Conventions: complex modes are pyseobnr's (`h = sum h_lm -2Y_lm`, `h_lm = |h_lm| e^{i phi_lm}` with `phi_lm`
decreasing); the shared (2,2) IMR pipeline (`EOBIMR.jl`) writes `h = A e^{-i varphi}`, `varphi = -phi_lm`.
The fit files `SEOBNRv5HM_fits.jl` and `SEOBNRv5HM_qnm.jl` are generated (`scripts/translate_pyseobnr_fits.py`,
`scripts/fit_qnm_chebyshev.py`).

Model features reproduced on purpose although they are not smooth in the parameters (flagged for the
sampler, design notes Section 3.6): the `hsign` threshold/sign of the special coefficients; the
`|habs|`, `|hdot33|`-type absolute values in the input-value fits; the NQC skip for odd `m` at
(nearly) equal masses with equal spins; the equal-mass branches of the odd-`m` modes.
"""

const _SEOBV5HM_L = (2, 2, 3, 3, 4, 4)
const _SEOBV5HM_M = (2, 1, 3, 2, 4, 3)
const _SEOBV5HM_NMODES = 6

# ---------------------------------------------------------------------------------------------------
# Fit variables, QNMs, mode-mixing coefficients.
# ---------------------------------------------------------------------------------------------------

"""
    _seobv5hm_fitvars(X1, X2, chi1, chi2) -> NamedTuple

The variables of pyseobnr's `InputValueFits` / `MergerRingdownFits` (`q = m1/m2 >= 1`).
"""
@inline function _seobv5hm_fitvars(X1, X2, chi1, chi2)
    q = X1 / X2
    nu = q / (1.0 + q)^2
    dm = (q - 1.0) / (q + 1.0)
    chiAS = 0.5 * (chi1 - chi2)
    chiS = 0.5 * (chi1 + chi2)
    chi = chiS + chiAS * dm / (1.0 - 2.0 * nu)
    chi33 = chiS * dm + chiAS
    chi21A = chiS * dm / (1.0 - 1.3 * nu) + chiAS
    chi44A = (1.0 - 5.0 * nu) * chiS + chiAS * dm
    chi21D = chiS * dm / (1.0 - 2.0 * nu) + chiAS
    chi44D = (1.0 - 7.0 * nu) * chiS + chiAS * dm
    return (nu = nu, dm = dm, chiS = chiS, chiAS = chiAS, chi = chi, chi33 = chi33, chi21A = chi21A,
        chi44A = chi44A, chi21D = chi21D, chi44D = chi44D, nuprime = 0.25 - nu, chi33abs = abs(chi33),
        chi21Aabs = abs(chi21A), chi44Aabs = abs(chi44A))
end

# Mode k of a fit NamedTuple (keys h22, h33, h21, h44, h55, h32, h43).
@inline function _seobv5hm_pick(t, k::Int)
    k == 1 && return t.h22
    k == 2 && return t.h21
    k == 3 && return t.h33
    k == 4 && return t.h32
    k == 5 && return t.h44
    return t.h43
end

# |z| without Base's `hypot` / complex scaling paths (Enzyme forward mode on CUDA: see `_seobv5_arg_gamma`).
@inline _seobv5hm_cabs(z) = sqrt(abs2(z))

# Chebyshev series (Clenshaw).
# Unrolled by recursion on the (static) index: a loop here is a device-malloc'd cache in Enzyme's in-kernel reverse
# on GPUs.
@inline _seobv5hm_clenshaw(c::NTuple{N,Float64}, x) where {N} = _seobv5hm_clenshaw_step(c, x, Val(N), 0.0, 0.0)
@inline _seobv5hm_clenshaw_step(c, x, ::Val{1}, b1, b2) = x * b1 - b2 + c[1]
@inline _seobv5hm_clenshaw_step(c, x, ::Val{j}, b1, b2) where {j} =
    _seobv5hm_clenshaw_step(c, x, Val(j - 1), 2.0 * x * b1 - b2 + c[j], b1)

"""
    _seobv5hm_qnm(k, af) -> (omega_R, omega_I)

Fundamental `(l, m, 0)` QNM of mode `k` (1..6 in `_SEOBV5HM_L/_M` order, 7 = (5,5)) for a remnant of unit mass
and spin `af`: `omega = omega_R + i omega_I`, `omega_R > 0`, `omega_I < 0` (pyseobnr `compute_QNM` before the
division by `M_f`; for `af < 0` the `-m` branch at `|af|`, as pyseobnr).
"""
@inline function _seobv5hm_qnm(k::Int, af)
    a = min(abs(af), _SEOBV5HM_QNM_AMAX)
    br = af >= 0.0 ? 0 : 1
    lower = a <= _SEOBV5HM_QNM_ASPLIT
    lo, hi = lower ? (0.0, _SEOBV5HM_QNM_ASPLIT) : (_SEOBV5HM_QNM_ASPLIT, _SEOBV5HM_QNM_AMAX)
    ylo, yhi = sqrt(1.0 - hi), sqrt(1.0 - lo)
    x = 2.0 * (sqrt(1.0 - a) - ylo) / (yhi - ylo) - 1.0
    base = (((k - 1) * 2 + br) * 2 + (lower ? 0 : 1)) * 2
    return _seobv5hm_clenshaw(_SEOBV5HM_QNM_CHEB[base + 1], x), _seobv5hm_clenshaw(_SEOBV5HM_QNM_CHEB[base + 2], x)
end

@inline _seobv5hm_mu_fit(j, p1, p2, p3, p4) = p1 * abs(j)^p2 + p3 * abs(j)^p4

"""
    _seobv5hm_mu(m, l, lp, j) -> ComplexF64

Spherical-spheroidal mixing coefficient `mu_{m l lp 0}(j)` (pyseobnr `auxiliary_functions_modemixing.mu`,
Berti & Klein 2014 Eq. 11 fits), for the cases the (3,2) and (4,3) modes need.
"""
@inline function _seobv5hm_mu(m::Int, l::Int, lp::Int, j)
    if m == 2 && l == 2 && lp == 2
        return complex(1.0 + _seobv5hm_mu_fit(j, -7.39789e-03, 2.88933e00, -6.61369e-03, 1.71287e01),
            _seobv5hm_mu_fit(j, 1.53046e-02, 1.21928e00, -9.34293e-03, 2.49915e01))
    elseif m == 2 && l == 3 && lp == 2
        return -complex(_seobv5hm_mu_fit(j, -1.03512e-01, 1.22285e00, -5.74989e-02, 8.70536e00),
            _seobv5hm_mu_fit(j, -1.60040e-02, 9.53385e-01, 1.00344e-02, 1.47550e01))
    elseif m == 2 && l == 3 && lp == 3
        return complex(1.0 + _seobv5hm_mu_fit(j, -2.51626e-02, 2.43323e00, -1.30029e-02, 1.09648e01),
            _seobv5hm_mu_fit(j, -3.23812e-02, 9.21248e-01, 1.88165e-02, 1.06988e01))
    elseif m == 3 && l == 3 && lp == 3
        return complex(1.0 + _seobv5hm_mu_fit(j, -1.20858e-02, 2.94266e00, -1.12402e-02, 1.72866e01),
            _seobv5hm_mu_fit(j, -1.72988e-02, 8.51080e-01, 1.43925e-02, 5.98677e00))
    elseif m == 3 && l == 4 && lp == 3
        return -complex(_seobv5hm_mu_fit(j, -1.32724e-01, 1.24145e00, -7.34932e-02, 8.51670e00),
            _seobv5hm_mu_fit(j, -1.15656e-02, 9.01101e-01, 7.32291e-03, 1.17007e01))
    else # m == 3 && l == 4 && lp == 4
        return complex(1.0 + _seobv5hm_mu_fit(j, -3.42425e-02, 2.59986e00, -2.12509e-02, 1.25368e01),
            _seobv5hm_mu_fit(j, -4.54823e-02, 9.25823e-01, 2.84355e-02, 1.14204e01))
    end
end

# ---------------------------------------------------------------------------------------------------
# Inspiral modes.
# ---------------------------------------------------------------------------------------------------

"""
    _seobv5hm_prefix(X1, X2, l, m) -> ComplexF64

Newtonian multipole prefactor `n eta c` (`calculate_multipole_prefix`, `waveform.pyx`, Eqs. 25-27 of the
SEOBNRv5HM notes), including the exactly-equal-mass odd-`m` constants `c0`, times the signed
`Y_l^{-m}(pi/2, 0)` of `EOBFluxCalculateNewtonianMultipole` (positive for all six modes).
"""
@inline function _seobv5hm_prefix(X1, X2, l::Int, m::Int)
    eps = (l + m) % 2
    sgn = iseven(m) ? 1.0 : -1.0
    eta = X1 * X2
    c = if X1 != X2 || sgn == 1.0
        X2^(l + eps - 1) + sgn * X1^(l + eps - 1)
    else
        l <= 3 ? -1.0 : -0.5
    end
    if eps == 0
        n = (im * m)^l * (8.0 * pi / _seobv5_dfact_odd(2l + 1)) * sqrt((l + 1.0) * (l + 2.0) / (l * (l - 1.0)))
    else
        n = -(im * m)^l * im * (16.0 * pi / _seobv5_dfact_odd(2l + 1)) *
            sqrt((2l + 1.0) * (l + 2.0) * (l * l - m * m) / ((2l - 1.0) * (l + 1.0) * l * (l - 1.0)))
    end
    ylm = l == 2 ? (m == 2 ? 0.3862742020231896 : 0.3454941494713355) :
          l == 3 ? (m == 3 ? 0.4172238236327842 : 0.3862742020231896) :
          (m == 4 ? 0.4425326924449826 : 0.4172238236327842)
    return n * eta * c * ylm
end

# Tail factor T_lm (`compute_mode`): magnitude and phase, k = m Omega, hk = m nu H Omega.
@inline function _seobv5hm_tail(l::Int, m::Int, om, nuHomega)
    hk = m * nuHomega
    hksq4 = 4.0 * hk * hk
    # prod_{j=1}^{l} (hksq4 + j^2), l = 2..4 (written out: a loop of runtime length is an exponentially reallocated,
    # never freed device-malloc cache in Enzyme's in-kernel reverse on GPUs)
    prod = (hksq4 + 1.0) * (hksq4 + 4.0) * ifelse(l >= 3, hksq4 + 9.0, 1.0) * ifelse(l >= 4, hksq4 + 16.0, 1.0)
    fact = l == 2 ? 2.0 : (l == 3 ? 6.0 : 24.0)
    mag = sqrt(4.0 * pi * hk / (1.0 - exp(-4.0 * pi * hk))) * sqrt(prod) / fact
    k = m * om
    ph = 2.0 * hk * log(4.0 * k / sqrt(exp(1.0))) + _seobv5_arg_gamma(l + 1.0, -2.0 * hk)
    return mag, ph
end

# delta_lm (`compute_delta_coeffs`; aDelta = 0): v = Omega^(1/3), vh3 = nu H Omega.
@inline function _seobv5hm_deltalm(k::Int, v, vh3, nu, delta, chiS, chiA)
    vh6 = vh3 * vh3
    vh9 = vh6 * vh3
    v5 = v^5
    if k == 1
        return 7.0 / 3.0 * vh3 + (-4.0 / 3.0 * (delta * chiA + chiS * (1.0 - 2.0 * nu)) + 428.0 * pi / 105.0) * vh6 +
               (-2203.0 / 81.0 + 1712.0 * pi * pi / 315.0) * vh9 - 24.0 * nu * v5
    elseif k == 2
        return 2.0 / 3.0 * vh3 + 107.0 * pi / 105.0 * vh6 + (-272.0 / 81.0 + 214.0 * pi * pi / 315.0) * vh9 -
               25.0 * nu / 2.0 * v5
    elseif k == 3
        return 13.0 / 10.0 * vh3 + 39.0 * pi / 7.0 * vh6 + (-227827.0 / 3000.0 + 78.0 * pi * pi / 7.0) * vh9 -
               80897.0 * nu / 2430.0 * v5
    elseif k == 4
        return (10.0 + 33.0 * nu) / (-15.0 * (-1.0 + 3.0 * nu)) * vh3 + 52.0 * pi / 21.0 * vh6 +
               (-9112.0 / 405.0 + 208.0 * pi * pi / 63.0) * vh9
    elseif k == 5
        return (112.0 + 219.0 * nu) / (-120.0 * (-1.0 + 3.0 * nu)) * vh3 + 25136.0 * pi / 3465.0 * vh6 +
               (-55144.0 / 375.0 + 201088.0 * pi * pi / 10395.0) * vh9
    else
        return (486.0 + 4961.0 * nu) / (810.0 * (1.0 - 2.0 * nu)) * vh3 + 1571.0 * pi / 385.0 * vh6
    end
end

"""
    _seobv5hm_mode_ap(k, om, om_circ, H, Heff, pphi, b, pref, c21, c43) -> (A, theta)

Reduced inspiral mode `H_lm = h_lm e^{i m phi} = A e^{i theta}` of mode `k` (pyseobnr `compute_mode`, units
`M/r`), from the orbital frequency, the circular-orbit frequency at the same `(r, pphi)`, `H` (the SEOBNRv5
Hamiltonian over `nu`, as returned by `_seobv5_ham_xi`) and `Heff`. `pref` = [`_seobv5hm_prefix`](@ref) of the
mode, `c21`, `c43` the special calibration coefficients.

`A` is a *signed* real amplitude and `theta` a smooth phase: `rho_lm` of the odd-`m` modes is real and can
change sign before merger (the special coefficients set its sign at the attachment), and for `(3,3)` the
`atan(Im/Re)` of the complex `rho_33` jumps by `pi` exactly where `A` changes sign, so `A e^{i theta}` and both
factors stay continuous. Interpolating `A` and `theta` (not `|h|` and `arg h`) keeps the dense output and the
SPA tables smooth through such zeros.
"""
@inline function _seobv5hm_mode_ap(k::Int, om, om_circ, H, Heff, pphi, b, pref, c21, c43)
    l, m = _SEOBV5HM_L[k], _SEOBV5HM_M[k]
    nu = b.nu
    v = cbrt(om)
    vphi = om / om_circ^(2.0 / 3.0)
    nuHomega = nu * H * om
    rf = _seobv5_rho_f(v, log(v), nuHomega, b.fc.rho)
    eqm = abs(nu - 0.25) < 1e-14            # compute_rholm_single: odd m keep f only
    v7 = v^7
    rho_re, rho_im = if k == 1
        rf.rho22^2, 0.0
    elseif k == 2
        f = rf.f21 + c21 * v7
        (eqm ? f : rf.rho21^2 + f), 0.0
    elseif k == 3
        (eqm ? rf.f33 : rf.rho33^3 + rf.f33), rf.f33_im
    elseif k == 4
        rf.rho32^3, 0.0
    elseif k == 5
        rf.rho44^4, 0.0
    else
        f = rf.f43 + c43 * v7
        (eqm ? f : rf.rho43^4 + f), 0.0
    end
    S = iseven(l + m) ? Heff : v * pphi
    Tm, Tph = _seobv5hm_tail(l, m, om, nuHomega)
    dl = _seobv5hm_deltalm(k, v, nuHomega, nu, b.delta, b.chiS, b.chiA)
    eps = (l + m) % 2
    # signed |rho| with the phase atan(Im/Re): continuous through Re rho = 0 (0 when both vanish)
    rho_s = rho_im == 0.0 ? rho_re : sign(rho_re) * sqrt(rho_re^2 + rho_im^2)
    rho_ph = ifelse(rho_re == 0.0, 0.0, atan(rho_im / ifelse(rho_re == 0.0, 1.0, rho_re)))
    A = Tm * S * vphi^(l + eps) * _seobv5hm_cabs(pref) * rho_s
    return A, Tph + dl + angle(pref) + rho_ph
end

@inline function _seobv5hm_reduced_mode(k::Int, om, om_circ, H, Heff, pphi, b, pref, c21, c43)
    A, th = _seobv5hm_mode_ap(k, om, om_circ, H, Heff, pphi, b, pref, c21, c43)
    return A * complex(cos(th), sin(th))
end

# The state-dependent inputs of `_seobv5hm_reduced_mode` at one dynamics point.
@inline _seobv5hm_state(r, prst, pphi, b) = _seobv5_sym_state(r, prst, pphi, b.hp)

"""
    _seobv5hm_special_coeffs(b, st, pref) -> (c21, c43)

`compute_special_coeffs` (`waveform.pyx`): the `v^7` coefficient of `f_21`, `f_43` such that the inspiral
mode's amplitude at `t_attach` (state `st`) equals `nu hsign_lm` (sign kept; pushed to `+|.|` when `|hsign| < 1e-4`;
magnitude floored at `hsign_22 / threshold`, thresholds 300 and `200 nu (1 - 0.8 chiA)`). Zero for an exactly
equal-mass, equal-spin binary (pyseobnr skips it there).
"""
function _seobv5hm_special_coeffs(b, st, pref)
    nu = b.nu
    r, _, pr, pp, om = st
    H, Heff, om_circ = _seobv5hm_state(r, pr, pp, b)
    hs = _seobv5hm_iv_hsign(_seobv5hm_fitvars(b.X1, b.X2, b.chi1, b.chi2))
    rf = _seobv5_rho_f(cbrt(om), log(cbrt(om)), nu * H * om, b.fc.rho)
    skip = (abs(nu - 0.25) < 1e-14) & (abs(b.chiA) < 1e-14)
    c21 = _seobv5hm_special_k(2, om, om_circ, H, Heff, pp, b, pref[2], rf, hs.h21, hs.h22, 300.0)
    c43 = _seobv5hm_special_k(6, om, om_circ, H, Heff, pp, b, pref[6], rf, hs.h43, hs.h22, 200.0 * nu * (1.0 - 0.8 * b.chiA))
    return ifelse(skip, 0.0, c21), ifelse(skip, 0.0, c43)
end

# One special coefficient (scalar branches as `ifelse`: no loops, buffers or reassigned locals for Enzyme).
@inline function _seobv5hm_special_k(k, om, om_circ, H, Heff, pp, b, pref, rf, hsign, amp22, min_amp)
    nu = b.nu
    # |h_lm| and rho_lm with the coefficient set to 0, K = |h_lm| / |rho_lm|
    h0 = _seobv5hm_reduced_mode(k, om, om_circ, H, Heff, pp, b, pref, 0.0, 0.0)
    eqm = abs(nu - 0.25) < 1e-14
    rho = k == 2 ? ifelse(eqm, rf.f21, rf.rho21^2 + rf.f21) : ifelse(eqm, rf.f43, rf.rho43^4 + rf.f43)
    K = _seobv5hm_cabs(h0) / abs(rho)
    a1 = ifelse(abs(hsign) < 1e-4, abs(hsign), hsign)
    a2 = ifelse(abs(a1) < amp22 / min_amp, sign(a1) * amp22 / min_amp, a1)
    return (nu * a2 / K - rho) / om^(7.0 / 3.0)
end

# ---------------------------------------------------------------------------------------------------
# Attachment-point states: a local fine re-integration around t_attach.
# ---------------------------------------------------------------------------------------------------

# SEOBNRv5HM: RK4 steps from the last node before t_a - 1 to t_a - 1, and per 0.5 M between the five states. One step
# each: the states move by < 1e-7 relative, the waveform by ||dh||^2/||h||^2 < 1e-13.
const _SEOBV5HM_ATT_N0 = 1
const _SEOBV5HM_ATT_NF = 1

# One RK4 step of the SEOBNRv5 equations of motion, plus Omega at its start (k1's phi component). The four right-hand
# sides come from ONE call site (a stage loop): the device forward kernels inline every call site of the Hamiltonian
# + 35-mode flux, and the unrolled step (x 25 in `_seobv5hm_attach_states`) made a 59 MB PTX that ptxas needed 32 GB
# for. Same operations as the unrolled step.
@inline function _seobv5hm_rk4(m, b, r, ph, pr, pp, h)
    k1 = k2 = k3 = k4 = (0.0, 0.0, 0.0, 0.0)
    for s in 1:4
        kp = ifelse(s == 2, k1, ifelse(s == 3, k2, k3))
        c = ifelse(s == 4, h, 0.5h)
        k = _imr_rhs4(m, ifelse(s == 1, r, r + c * kp[1]), ifelse(s == 1, pr, pr + c * kp[3]),
            ifelse(s == 1, pp, pp + c * kp[4]), b)
        k1 = ifelse(s == 1, k, k1); k2 = ifelse(s == 2, k, k2); k3 = ifelse(s == 3, k, k3); k4 = ifelse(s == 4, k, k4)
    end
    k1r, k1f, k1p, k1j = k1; k2r, k2f, k2p, k2j = k2; k3r, k3f, k3p, k3j = k3; k4r, k4f, k4p, k4j = k4
    return (r + h / 6.0 * (k1r + 2k2r + 2k3r + k4r), ph + h / 6.0 * (k1f + 2k2f + 2k3f + k4f),
        pr + h / 6.0 * (k1p + 2k2p + 2k3p + k4p), pp + h / 6.0 * (k1j + 2k2j + 2k3j + k4j), k1f)
end

"""
    _seobv5hm_attach_states(B) -> NTuple{5, NTuple{5,Float64}}

The dynamics state `(r, phi, pr, pphi, Omega)` at `t_attach + (-1, -0.5, 0, 0.5, 1)`, by re-integrating the
equations of motion from the last dynamics node before `t_attach - 1` with fixed-count RK4 steps.
The plunge nodes are ~1.5 M apart near the attachment, and the second time derivatives the NQC and
special-coefficient conditions need are not accurate enough from their cubic-Hermite dense output
(the (4,3) mode's frequency derivative changed by 20% with the difference step); pyseobnr reads them
from a 0.1 M fine dynamics.
"""
function _seobv5hm_attach_states(B)
    m, b, dyn, t_a = SEOBNRv5IMRAlignedSpinF2(), B.base, B.dyn, B.t_a
    n = dyn.n
    i0 = clamp(_eobimr_searchsortedlast(dyn.t, t_a - 1.0), 1, n)
    r, ph, pr, pp, t0 = dyn.r[i0], dyn.phi[i0], dyn.pr[i0], dyn.pphi[i0], dyn.t[i0]
    N0, NF = _SEOBV5HM_ATT_N0, _SEOBV5HM_ATT_NF
    h0 = (t_a - 1.0 - t0) / N0
    # N0 steps of h0, then 4 x NF steps of 0.5 / NF, recording the state (Omega from the next step's k1) after the h0
    # steps and after every NF steps: one loop, one RK4 site (see `_seobv5hm_rk4`); the last iteration's step is only
    # for that k1.
    z = (0.0, 0.0, 0.0, 0.0, 0.0)
    s1 = s2 = s3 = s4 = s5 = z
    for j in 0:(N0 + 4NF)
        rn, phn, prn, ppn, om = _seobv5hm_rk4(m, b, r, ph, pr, pp, ifelse(j < N0, h0, 0.5 / NF))
        cur = (r, ph, pr, pp, om)
        s1 = ifelse(j == N0, cur, s1)
        s2 = ifelse(j == N0 + NF, cur, s2)
        s3 = ifelse(j == N0 + 2NF, cur, s3)
        s4 = ifelse(j == N0 + 3NF, cur, s4)
        s5 = ifelse(j == N0 + 4NF, cur, s5)
        r, ph, pr, pp = rn, phn, prn, ppn
    end
    return (s1, s2, s3, s4, s5)
end

@inline _seobv5hm_pick5(t, i) = ifelse(i == 1, t[1], ifelse(i == 2, t[2], ifelse(i == 3, t[3], ifelse(i == 4, t[4], t[5]))))
@inline _seobv5hm_set5(t, i, v) = (ifelse(i == 1, v, t[1]), ifelse(i == 2, v, t[2]), ifelse(i == 3, v, t[3]),
    ifelse(i == 4, v, t[4]), ifelse(i == 5, v, t[5]))

# Reduced mode k at an exact state `st` with `hst = _seobv5hm_state` there.
@inline function _seobv5hm_mode_at_state(k, st, hst, b, pref, c21, c43)
    _, _, _, pp, om = st
    H, Heff, om_circ = hst
    return _seobv5hm_reduced_mode(k, om, om_circ, H, Heff, pp, b, pref[k], c21, c43)
end

# The six reduced modes at the five attach states, `Hs[k][i]`: a loop over the states (one Hamiltonian site, one site
# per mode; the unrolled 6 x 5 was 30 inlined copies of the state + rho_f in the device kernel).
function _seobv5hm_Hs(sts, b, pref, c21, c43)
    z = (0.0im, 0.0im, 0.0im, 0.0im, 0.0im)
    H1 = H2 = H3 = H4 = H5 = H6 = z
    for i in 1:5
        st = _seobv5hm_pick5(sts, i)
        hst = _seobv5hm_state(st[1], st[3], st[4], b)
        H1 = _seobv5hm_set5(H1, i, _seobv5hm_mode_at_state(1, st, hst, b, pref, c21, c43))
        H2 = _seobv5hm_set5(H2, i, _seobv5hm_mode_at_state(2, st, hst, b, pref, c21, c43))
        H3 = _seobv5hm_set5(H3, i, _seobv5hm_mode_at_state(3, st, hst, b, pref, c21, c43))
        H4 = _seobv5hm_set5(H4, i, _seobv5hm_mode_at_state(4, st, hst, b, pref, c21, c43))
        H5 = _seobv5hm_set5(H5, i, _seobv5hm_mode_at_state(5, st, hst, b, pref, c21, c43))
        H6 = _seobv5hm_set5(H6, i, _seobv5hm_mode_at_state(6, st, hst, b, pref, c21, c43))
    end
    return (H1, H2, H3, H4, H5, H6)
end

# ---------------------------------------------------------------------------------------------------
# Dense output of the reduced modes, NQC, merger-ringdown.
# ---------------------------------------------------------------------------------------------------

# 3-point non-uniform-grid time derivative of node values y (one-sided at the ends).
@inline function _seobv5hm_node_deriv(t, y, i, n)
    if i == 1
        return (y[2] - y[1]) / (t[2] - t[1])
    elseif i == n
        return (y[n] - y[n - 1]) / (t[n] - t[n - 1])
    end
    h1 = t[i] - t[i - 1]; h2 = t[i + 1] - t[i]
    return -h2 / (h1 * (h1 + h2)) * y[i - 1] + (h2 - h1) / (h1 * h2) * y[i] + h1 / (h2 * (h1 + h2)) * y[i + 1]
end

"""
    _seobv5hm_node_modes!(D, B, pref, c21, c43) -> nothing

Signed amplitude `A` and smooth phase `theta` of every reduced mode ([`_seobv5hm_mode_ap`](@ref)) at the
dynamics nodes, with their time derivatives (3-point FD), into the `(node, mode)` matrices of `D`
([`_seobv5hm_alloc_modes`](@ref)) for cubic-Hermite dense output. Caller-owned buffers: a function returning
heap arrays inside a struct takes Enzyme's sret+returnRoots path, which asserts on Julia 1.13.
"""
function _seobv5hm_node_modes!(D, B, pref, c21, c43)
    dyn, b = B.dyn, B.base
    n = dyn.n
    A, dA, th, dth = D.A, D.dA, D.th, D.dth
    fill!(A, 0.0); fill!(dA, 0.0); fill!(th, 0.0); fill!(dth, 0.0)
    @inbounds for i in 1:n
        H, Heff, om_circ = _seobv5hm_state(dyn.r[i], dyn.pr[i], dyn.pphi[i], b)
        for k in 1:_SEOBV5HM_NMODES
            A[i, k], th[i, k] = _seobv5hm_mode_ap(k, dyn.dphi[i], om_circ, H, Heff, dyn.pphi[i], b, pref[k], c21, c43)
        end
    end
    @inbounds for k in 1:_SEOBV5HM_NMODES, i in 1:n
        dA[i, k] = _seobv5hm_node_deriv(dyn.t, view(A, :, k), i, n)
        dth[i, k] = _seobv5hm_node_deriv(dyn.t, view(th, :, k), i, n)
    end
    return nothing
end

# Dense-output buffers for `_seobv5hm_node_modes!`.
@inline _seobv5hm_alloc_modes() = (A = zeros(_EOBIMR_NDYN, _SEOBV5HM_NMODES), dA = zeros(_EOBIMR_NDYN, _SEOBV5HM_NMODES),
    th = zeros(_EOBIMR_NDYN, _SEOBV5HM_NMODES), dth = zeros(_EOBIMR_NDYN, _SEOBV5HM_NMODES))

# Dynamics + reduced mode k at time tq: (r, phi, pr, Omega, A, dA/dt, theta, dtheta/dt).
@inline function _seobv5hm_at(dyn, D, k, tq)
    n = dyn.n
    tq = clamp(tq, dyn.t[1], dyn.t[n])
    i = clamp(_eobimr_searchsortedlast(dyn.t, tq), 1, n - 1)
    h = dyn.t[i + 1] - dyn.t[i]
    s = (tq - dyn.t[i]) / h
    r, _ = _eobimr_hermite(dyn.r[i], dyn.dr[i], dyn.r[i + 1], dyn.dr[i + 1], h, s)
    ph, om = _eobimr_hermite(dyn.phi[i], dyn.dphi[i], dyn.phi[i + 1], dyn.dphi[i + 1], h, s)
    pr, _ = _eobimr_hermite(dyn.pr[i], dyn.dpr[i], dyn.pr[i + 1], dyn.dpr[i + 1], h, s)
    A, dA = _eobimr_hermite(D.A[i, k], D.dA[i, k], D.A[i + 1, k], D.dA[i + 1, k], h, s)
    th, dth = _eobimr_hermite(D.th[i, k], D.dth[i, k], D.th[i + 1, k], D.dth[i + 1, k], h, s)
    return r, ph, pr, om, A, dA, th, dth
end

# NQC-corrected inspiral mode k at tq as (signed amplitude, varphi): h_lm = A e^{-i varphi} (varphi increasing;
# pyseobnr's phase is -varphi).
@inline function _seobv5hm_insp_ap(dyn, D, nqc, k, tq)
    r, ph, pr, om, A, _, th, _ = _seobv5hm_at(dyn, D, k, tq)
    m = _SEOBV5HM_M[k]
    q1, q2, q3 = _eobimr_nqc_q(r, pr, om)
    p1, p2 = _eobimr_nqc_p(r, pr, om)
    a1, a2, a3, b1, b2 = nqc[k]
    return A * (1.0 + q1 * a1 + q2 * a2 + q3 * a3), m * ph - th - (b1 * p1 + b2 * p2)
end

# NQC-corrected inspiral mode k (pyseobnr complex convention) at tq.
@inline function _seobv5hm_insp(dyn, D, nqc, k, tq)
    A, vp = _seobv5hm_insp_ap(dyn, D, nqc, k, tq)
    return A * complex(cos(vp), -sin(vp))
end

# d varphi / dt of the NQC-corrected inspiral mode: m Omega - dtheta/dt minus the NQC phase rate (FD, +-0.25 M).
@inline function _seobv5hm_insp_freq(dyn, D, nqc, k, tq)
    _, _, _, om, _, _, _, dth = _seobv5hm_at(dyn, D, k, tq)
    a1, a2, a3, b1, b2 = nqc[k]
    xm = _seobv5hm_at(dyn, D, k, tq - 0.25); xp = _seobv5hm_at(dyn, D, k, tq + 0.25)
    pm = _eobimr_nqc_p(xm[1], xm[3], xm[4]); pp = _eobimr_nqc_p(xp[1], xp[3], xp[4])
    return _SEOBV5HM_M[k] * om - dth - (b1 * (pp[1] - pm[1]) + b2 * (pp[2] - pm[2])) / 0.5
end

# Value, first and second derivative from 5 equally spaced samples (step h), 5-point stencils.
@inline _seobv5hm_d5(f, h) = (f[3], (f[1] - 8.0 * f[2] + 8.0 * f[4] - f[5]) / (12.0 * h),
    (-f[1] + 16.0 * f[2] - 30.0 * f[3] + 16.0 * f[4] - f[5]) / (12.0 * h * h))

# Phases of the 5 samples of a complex mode, unwrapped from the middle one.
@inline function _seobv5hm_unwrap5(h)
    p3 = angle(h[3])
    p2 = p3 + angle(h[2] * conj(h[3])); p1 = p2 + angle(h[1] * conj(h[2]))
    p4 = p3 + angle(h[4] * conj(h[3])); p5 = p4 + angle(h[5] * conj(h[4]))
    return (p1, p2, p3, p4, p5)
end

"""
    _seobv5hm_nqc(sts, Hs, k, nu, iv) -> (a1, a2, a3, b1, b2)

`EOBCalculateNQCCoefficients_freeattach` for mode `k` at `t_attach`: amplitude NQCs from `|h_lm|` and its first
two derivatives vs `nu (habs, hdot, hdotdot)`, frequency NQCs from the mode frequency and its derivative vs
`(|omega|, |omegadot|)` of the input-value fits `iv`. `sts`: the states at `t_attach + (-1, -0.5, 0, 0.5, 1)`
([`_seobv5hm_attach_states`](@ref)), `Hs` the reduced mode there; derivatives by 5-point stencils.
"""
function _seobv5hm_nqc(sts, Hs, k, nu, iv)
    habs, hdot, hdd, om_fit, omd_fit = iv
    m = _SEOBV5HM_M[k]
    h = 0.5
    # (explicit 5-tuples, no closures: Enzyme's argument classification asserted on closures here)
    A = (_seobv5hm_cabs(Hs[1]), _seobv5hm_cabs(Hs[2]), _seobv5hm_cabs(Hs[3]), _seobv5hm_cabs(Hs[4]), _seobv5hm_cabs(Hs[5]))
    q1 = _eobimr_nqc_q(sts[1][1], sts[1][3], sts[1][5]); q2 = _eobimr_nqc_q(sts[2][1], sts[2][3], sts[2][5])
    q3 = _eobimr_nqc_q(sts[3][1], sts[3][3], sts[3][5]); q4 = _eobimr_nqc_q(sts[4][1], sts[4][3], sts[4][5])
    q5 = _eobimr_nqc_q(sts[5][1], sts[5][3], sts[5][5])
    p1 = _eobimr_nqc_p(sts[1][1], sts[1][3], sts[1][5]); p2 = _eobimr_nqc_p(sts[2][1], sts[2][3], sts[2][5])
    p3 = _eobimr_nqc_p(sts[3][1], sts[3][3], sts[3][5]); p4 = _eobimr_nqc_p(sts[4][1], sts[4][3], sts[4][5])
    p5 = _eobimr_nqc_p(sts[5][1], sts[5][3], sts[5][5])
    # varphi = m phi - arg H (pyseobnr phase = -varphi), unwrapped from the middle sample
    ah = _seobv5hm_unwrap5(Hs)
    vp = (m * sts[1][2] - ah[1], m * sts[2][2] - ah[2], m * sts[3][2] - ah[3], m * sts[4][2] - ah[4], m * sts[5][2] - ah[5])
    A0, A1, A2 = _seobv5hm_d5(A, h)
    Q11, Q21, Q31 = _seobv5hm_d5((A[1] * q1[1], A[2] * q2[1], A[3] * q3[1], A[4] * q4[1], A[5] * q5[1]), h)
    Q12, Q22, Q32 = _seobv5hm_d5((A[1] * q1[2], A[2] * q2[2], A[3] * q3[2], A[4] * q4[2], A[5] * q5[2]), h)
    Q13, Q23, Q33 = _seobv5hm_d5((A[1] * q1[3], A[2] * q2[3], A[3] * q3[3], A[4] * q4[3], A[5] * q5[3]), h)
    R1, R2, R3 = nu * habs - A0, nu * hdot - A1, nu * hdd - A2
    det = Q11 * (Q22 * Q33 - Q23 * Q32) - Q12 * (Q21 * Q33 - Q23 * Q31) + Q13 * (Q21 * Q32 - Q22 * Q31)
    a1 = (R1 * (Q22 * Q33 - Q23 * Q32) - Q12 * (R2 * Q33 - Q23 * R3) + Q13 * (R2 * Q32 - Q22 * R3)) / det
    a2 = (Q11 * (R2 * Q33 - Q23 * R3) - R1 * (Q21 * Q33 - Q23 * Q31) + Q13 * (Q21 * R3 - R2 * Q31)) / det
    a3 = (Q11 * (Q22 * R3 - R2 * Q32) - Q12 * (Q21 * R3 - R2 * Q31) + R1 * (Q21 * Q32 - Q22 * Q31)) / det
    _, w0, wd0 = _seobv5hm_d5(vp, h)
    # pyseobnr: omega = |omega|, omegaDot signed relative to it
    wd0 = w0 * wd0 > 0.0 ? abs(wd0) : -abs(wd0)
    w0 = abs(w0)
    _, P11, P21 = _seobv5hm_d5((p1[1], p2[1], p3[1], p4[1], p5[1]), h)
    _, P12, P22 = _seobv5hm_d5((p1[2], p2[2], p3[2], p4[2], p5[2]), h)
    P11, P21, P12, P22 = -P11, -P21, -P12, -P22
    S1 = abs(om_fit) - w0
    S2 = abs(omd_fit) - wd0
    detp = P11 * P22 - P12 * P21
    b1 = (S1 * P22 - P12 * S2) / detp
    b2 = (P11 * S2 - S1 * P21) / detp
    return (a1, a2, a3, b1, b2)
end

# NQC-corrected inspiral mode (pyseobnr convention) from an exact state and its reduced mode.
@inline function _seobv5hm_insp_state(st, Hk, nqc_k, m)
    r, ph, pr, _, om = st
    q1, q2, q3 = _eobimr_nqc_q(r, pr, om)
    p1, p2 = _eobimr_nqc_p(r, pr, om)
    a1, a2, a3, b1, b2 = nqc_k
    bp = b1 * p1 + b2 * p2
    return Hk * (1.0 + q1 * a1 + q2 * a2 + q3 * a3) * complex(cos(bp - m * ph), sin(bp - m * ph))
end

"""
    _seobv5hm_mr(nu, k, A, dA, phi, om, Mf, af, fits, rot = 0) -> NamedTuple

`compute_MR_mode_free` for mode `k` (pyseobnr convention: `phi`, `om` the phase and its rate of
`h = |h| e^{i phi}` at the attachment, `om < 0`). Returns the ansatz parameters; evaluate with
[`_seobv5hm_mr_eval`](@ref). `rot`: SEOBNRv5PHM's co-precessing-frame QNM shift (`compute_omegalm_P_frame`,
`omega_lm -> omega_lm - m rot`); 0 for the aligned-spin model.
"""
@inline function _seobv5hm_mr(nu, k, A, dA, phi, om, Mf, af, fits, rot = 0.0)
    wr, wi = _seobv5hm_qnm(k, af)
    wr = wr / Mf - _SEOBV5HM_M[k] * rot
    tau = -Mf / wi
    sigR = -1.0 / tau
    sigI = -wr
    c1f, c2f, d1f, d2f = fits
    if k == 1 && sigR > 2.0 * c1f * tanh(c2f)
        c1f = sigR / (2.0 * tanh(c2f))
    end
    c1c = 1.0 / (c1f * nu) * (dA - sigR * A) * cosh(c2f)^2
    c2c = A / nu - 1.0 / (c1f * nu) * (dA - sigR * A) * cosh(c2f) * sinh(c2f)
    d1c = (om - sigI) * (1.0 + d2f) / (d1f * d2f)
    return (nu = nu, c1c = c1c, c2c = c2c, c1f = c1f, c2f = c2f, d1f = d1f, d2f = d2f, d1c = d1c,
        phi = phi, wr = wr, tau = tau)
end

# Merger-ringdown value at dt = t - t_attach >= 0 (pyseobnr complex convention).
@inline function _seobv5hm_mr_eval(p, dt)
    A = p.nu * (p.c1c * tanh(p.c1f * dt + p.c2f) + p.c2c) * exp(-dt / p.tau)
    ph = p.phi - p.d1c * log((1.0 + p.d2f * exp(-p.d1f * dt)) / (1.0 + p.d2f)) - p.wr * dt
    return A * complex(cos(ph), sin(ph))
end

# Spheroidal (l,m,0) attachment values from the spherical (l,m) and (m,m) modes (pyseobnr
# auxiliary_functions_modemixing: h_ellm0_nu, phi_ellm0, hdot_ellm0_nu, omega_ellm0).
@inline function _seobv5hm_spheroidal_iv(l, m, j, h, hmm, hd, hdmm, om, ommm, ph, phmm)
    mu_lm = _seobv5hm_mu(m, l, m, j)
    mu_mm = _seobv5hm_mu(m, m, m, j)
    mu_ll = _seobv5hm_mu(m, l, l, j)
    rho = _seobv5hm_cabs(mu_lm) * hmm / (h * _seobv5hm_cabs(mu_mm))
    dphi = phmm - ph - angle(mu_lm) + angle(mu_mm)
    F = sqrt((1.0 - rho * cos(dphi))^2 + rho^2 * sin(dphi)^2)
    alpha = atan(-rho * sin(dphi), 1.0 - rho * cos(dphi))
    rho_dot = _seobv5hm_cabs(mu_lm) * (hdmm / h - hmm * hd / h^2)
    dphi_dot = ommm - om
    F_dot = (rho * rho_dot + rho * sin(dphi) * dphi_dot - rho_dot * cos(dphi)) / F
    alpha_dot = (rho^2 * dphi_dot - rho * cos(dphi) * dphi_dot - rho_dot * sin(dphi)) / F^2
    return h * F / _seobv5hm_cabs(mu_ll), ph + angle(mu_ll) + alpha, (hd * F + h * F_dot) / _seobv5hm_cabs(mu_ll), om + alpha_dot
end

# Per-mode helpers of `_seobv5hm_build`: top-level functions applied to k = 1..6 (closures in the differentiated
# path made Enzyme's argument classification assert, as noted at `_eobimr_nqc`).
@inline _seobv5hm_tup6(f, args...) = (f(1, args...), f(2, args...), f(3, args...), f(4, args...), f(5, args...), f(6, args...))
@inline _seobv5hm_pref_k(k, X1, X2) = _seobv5hm_prefix(X1, X2, _SEOBV5HM_L[k], _SEOBV5HM_M[k])
@inline _seobv5hm_ivk(k, ivs) = (_seobv5hm_pick(ivs[1], k), _seobv5hm_pick(ivs[2], k), _seobv5hm_pick(ivs[3], k),
    _seobv5hm_pick(ivs[4], k), _seobv5hm_pick(ivs[5], k))
@inline _seobv5hm_nqc_k(k, sts, Hs, nu, ivs, skip_odd) = (isodd(_SEOBV5HM_M[k]) && skip_odd) ? (0.0, 0.0, 0.0, 0.0, 0.0) :
    _seobv5hm_nqc(sts, Hs[k], k, nu, _seobv5hm_ivk(k, ivs))
@inline _seobv5hm_fits_k(k, mrf) = (_seobv5hm_pick(mrf[1], k), _seobv5hm_pick(mrf[2], k), _seobv5hm_pick(mrf[3], k),
    _seobv5hm_pick(mrf[4], k))
@inline function _seobv5hm_att_k(k, sts, Hs, nqc, om_iv)
    m = _SEOBV5HM_M[k]
    H = Hs[k]
    hk = (_seobv5hm_insp_state(sts[1], H[1], nqc[k], m), _seobv5hm_insp_state(sts[2], H[2], nqc[k], m),
        _seobv5hm_insp_state(sts[3], H[3], nqc[k], m), _seobv5hm_insp_state(sts[4], H[4], nqc[k], m),
        _seobv5hm_insp_state(sts[5], H[5], nqc[k], m))
    A, dA, _ = _seobv5hm_d5((_seobv5hm_cabs(hk[1]), _seobv5hm_cabs(hk[2]), _seobv5hm_cabs(hk[3]), _seobv5hm_cabs(hk[4]), _seobv5hm_cabs(hk[5])), 0.5)
    ph = _seobv5hm_unwrap5(hk)
    om = isodd(m) ? _seobv5hm_pick(om_iv, k) : _seobv5hm_d5(ph, 0.5)[2]
    return (A, dA, ph[3], om)
end
@inline _seobv5hm_mr_k(k, nu, att, Mf, af, mrf) =
    _seobv5hm_mr(nu, k, att[k][1], att[k][2], att[k][3], att[k][4], Mf, af, _seobv5hm_fits_k(k, mrf))
@inline function _seobv5hm_mix_k(k, kmm, nu, att, Mf, af, mrf)
    l, m = _SEOBV5HM_L[k], _SEOBV5HM_M[k]
    h, hd_, ph, om = att[k]
    hmm, hdmm, phmm, ommm = att[kmm]
    A0, ph0, dA0, om0 = _seobv5hm_spheroidal_iv(l, m, af, h, hmm, hd_, hdmm, om, ommm, ph, phmm)
    sph = _seobv5hm_mr(nu, k, A0, dA0, ph0, om0, Mf, af, _seobv5hm_fits_k(k, mrf))
    return (sph = sph, mu_mm = _seobv5hm_mu(m, m, m, af), mu_lm = _seobv5hm_mu(m, l, m, af),
        mu_ll = _seobv5hm_mu(m, l, l, af), zero = !(h >= 1e-8))
end

"""
    _seobv5hm_build(D, mc, eta, chi1, chi2; f_min, GMsun_over_c3) -> NamedTuple

Dynamics, attachment (from [`_eobimr_build`](@ref) with the SEOBNRv5 hooks), reduced inspiral modes (into `D`),
special coefficients, per-mode NQC and merger-ringdown parameters. The returned NamedTuple holds no arrays;
evaluators take `merge(S, (D = D,))`.
"""
function _seobv5hm_build(D, mc, eta, chi1, chi2; f_min = 20.0, GMsun_over_c3 = GMsun_over_c3)
    B = _eobimr_build(SEOBNRv5IMRAlignedSpinF2(), mc, eta, chi1, chi2; f_min = f_min, GMsun_over_c3 = GMsun_over_c3)
    a = _seobv5hm_attach(B)
    _seobv5hm_node_modes!(D, B, a.pref, a.c21, a.c43)
    return merge((B = B, base = B.base, dyn = B.dyn, t_a = B.t_a), a)
end

"""
    _seobv5hm_attach(B) -> NamedTuple

Everything of [`_seobv5hm_build`](@ref) that depends only on the dynamics `B = (base, dyn, t_a)` and not on the
node modes: prefactors, attachment states, special coefficients, per-mode NQC, merger-ringdown and mixing. `B` is
duck-typed (`dyn` read through `t, r, phi, pr, pphi, n`): the device table calls it per walker on views.
"""
@inline function _seobv5hm_attach(B)
    b = B.base
    nu, X1, X2, c1, c2 = b.nu, b.X1, b.X2, b.chi1, b.chi2
    pref = _seobv5hm_tup6(_seobv5hm_pref_k, X1, X2)
    sts = _seobv5hm_attach_states(B)
    c21, c43 = _seobv5hm_special_coeffs(b, sts[3], pref)
    Hs = _seobv5hm_Hs(sts, b, pref, c21, c43)

    fv = _seobv5hm_fitvars(X1, X2, c1, c2)
    ivs = (_seobv5hm_iv_habs(fv), _seobv5hm_iv_hdot(fv), _seobv5hm_iv_hdotdot(fv), _seobv5hm_iv_omega(fv),
        _seobv5hm_iv_omegadot(fv))
    # NQC_correction: odd-m modes are skipped at (nearly) equal masses with equal (or zero) spins
    skip_odd = (b.delta < 1e-4) & ((abs(c1) < 1e-4) & (abs(c2) < 1e-4) | (abs(c1 - c2) < 1e-4))
    nqc = _seobv5hm_tup6(_seobv5hm_nqc_k, sts, Hs, nu, ivs, skip_odd)

    # Attachment values of the NQC-corrected inspiral modes (pyseobnr convention).
    Mf = _eobimr_final_mass_uib2016(X1, X2, c1, c2)
    af = _seobv5_final_spin_hbr2016(X1, X2, c1, c2)
    mrf = (_seobv5hm_mr_c1f(fv), _seobv5hm_mr_c2f(fv), _seobv5hm_mr_d1f(fv), _seobv5hm_mr_d2f(fv))
    att = _seobv5hm_tup6(_seobv5hm_att_k, sts, Hs, nqc, ivs[4])
    mr = _seobv5hm_tup6(_seobv5hm_mr_k, nu, att, Mf, af, mrf)
    # Mixed modes (3,2) <- (2,2), (4,3) <- (3,3): spheroidal (l,m,0) ringdown plus the (m,m) mode's leakage.
    mix = (_seobv5hm_mix_k(4, 1, nu, att, Mf, af, mrf), _seobv5hm_mix_k(6, 3, nu, att, Mf, af, mrf))
    return (pref = pref, c21 = c21, c43 = c43, nqc = nqc, mr = mr, mix = mix, Mf = Mf, af = af, att = att)
end

# Full IMR mode k (pyseobnr convention) at time tq (dynamics time).
@inline function _seobv5hm_full(S, k, tq)
    dt = tq - S.t_a
    dt <= 0.0 && return _seobv5hm_insp(S.dyn, S.D, S.nqc, k, tq)
    return _seobv5hm_full_mr(S, k, dt)
end
# its merger-ringdown part (dt = tq - t_attach > 0): reads only S.mr and S.mix
@inline function _seobv5hm_full_mr(S, k, dt)
    if k == 4 || k == 6
        x = S.mix[k == 4 ? 1 : 2]
        x.zero && return zero(ComplexF64)
        kmm = k == 4 ? 1 : 3
        return _seobv5hm_mr_eval(S.mr[kmm], dt) * x.mu_mm / abs2(x.mu_mm) * conj(x.mu_lm) +    # / conj(mu_mm)
               _seobv5hm_mr_eval(x.sph, dt) * conj(x.mu_ll)
    end
    return _seobv5hm_mr_eval(S.mr[k], dt)
end

"""
    seobnrv5hm_td_modes(mc, eta, chi1, chi2; f_min=20.0, dt=0.5) -> (t, modes, info)

Time-domain SEOBNRv5HM modes on a uniform grid (units of M, `t = 0` at `t_attach`): `modes[k]` is the complex
`h_lm` (units `M/dL`, pyseobnr convention) of `(l, m) = _SEOBV5HM_L[k], _SEOBV5HM_M[k]`, k = 1..6 for
(2,2), (2,1), (3,3), (3,2), (4,4), (4,3). For validation and plotting.
"""
function seobnrv5hm_td_modes(mc, eta, chi1, chi2; f_min = 20.0, dt = 0.5, GMsun_over_c3 = GMsun_over_c3)
    D = _seobv5hm_alloc_modes()
    S = merge(_seobv5hm_build(D, mc, eta, chi1, chi2; f_min = f_min, GMsun_over_c3 = GMsun_over_c3), (D = D,))
    ts = collect(S.B.t_start:dt:(S.t_a + _EOBIMR_NTAU * S.mr[1].tau))
    modes = ntuple(k -> [_seobv5hm_full(S, k, tq) for tq in ts], _SEOBV5HM_NMODES)
    info = (t_attach = S.t_a, t_isco = S.B.t_isco, nqc = S.nqc, c21 = S.c21, c43 = S.c43, Mf = S.Mf, af = S.af,
        success = S.B.success, modes = tuple(zip(_SEOBV5HM_L, _SEOBV5HM_M)...))
    return ts .- S.t_a, modes, info
end

# ---------------------------------------------------------------------------------------------------
# Frequency domain: per-mode SPA table (inspiral) + Fourier table (merger-ringdown), blended.
# ---------------------------------------------------------------------------------------------------

export SEOBNRv5HM

"""
    SEOBNRv5HM

**SEOBNRv5HM in the frequency domain** (aligned spins; modes (2,2), (2,1), (3,3), (3,2), (4,4), (4,3), the
pyseobnr 0.3.7 default): the time-domain model of [`seobnrv5hm_td_modes`](@ref) turned into one table per mode
by [`_seobv5hm_coeffs`](@ref), the same way [`SEOBNRv5IMRAlignedSpinF2`](@ref) does it for the (2,2) mode:

- below a blend band `f(t_a - 150 M) .. f(t_a - 90 M)` (each mode's own frequency), the stationary-phase
  approximation on the mode's NQC-corrected phase and signed amplitude (cubic-Hermite root for the stationary
  time, as [`_eobimr_spa_eval`](@ref));
- above it, a direct Fourier transform of the mode from `t_a - 1500 M` (Planck-type turn-on) through 15
  (2,2) damping times, tabulated on the FFT bins as `U(f) = e^{2 pi i f t_a} h(f)` and `dU/df` (real and
  imaginary parts, cubic Hermite: no trigonometric functions per frequency bin);
- a C1 blend in between, and a C1 amplitude taper over the top 20% of each mode's band `[.., 2.2 f_QNM]`.

Per-mode output: `h_{l,-m}(f)` for positive `m`, LAL's convention (as [`IMRPhenomXHM`](@ref)'s `h_lm`): the
Fourier transform of `h_{l,-m}(t) = (-1)^l conj(h_{l,m}(t))` with the LAL-convention modes, which are
pyseobnr's internal modes with odd `m` negated (pyseobnr `GenerateWaveform.generate_td_modes`). Time origin:
`t_attach` (the (2,2) peak to < 0.1 M), so a likelihood's `t_c` is the merger time.

Accuracy vs a direct Fourier transform of the time-domain modes: per-mode noise-weighted mismatch
~1e-6 (aLIGO design PSD), <= 2e-5 for (4,3) at q ~ 1. The SPA region omits the merger's broadband leakage, which
shows as a few-percent oscillating ratio in modes that are weak in the inspiral (the (4,3) at q ~ 1).
The (2,2) table equals the (2,2)-only [`SEOBNRv5IMRAlignedSpinF2`](@ref) one (`amp e^{-i psi}`) up to its folded
face-on factor `2/sqrt(5/(4 pi))` and a constant phase `pi`, to ~3e-4.
"""
struct SEOBNRv5HM end

const _SEOBV5HM_NSEG = 4096                  # Fourier segment samples (the (4,4) band reaches ~0.6/M at high spin)
const _SEOBV5HM_NFT = _SEOBV5HM_NSEG ÷ 2     # positive-frequency bins kept

"""
    SEOBNRv5HMTable

Per-binary frequency-domain tables built by [`_seobv5hm_coeffs`](@ref); `(node or bin, mode)` matrices.
SPA part: `ts` (node times, M), `phis`/`omegas` (mode phase and frequency divided by `m`, i.e. the
orbital-equivalent `varphi_lm / m`), `Qs`/`dQs` (the SPA amplitude factor `Q = R / sqrt(d omega/dt)`,
[`_spa_node_q`](@ref), `R` the signed amplitude over `omega^((l+eps)/3)`, and its time derivative). Fourier part: `ft_re/ft_im/ft_dre/ft_dim` on bins `f_j = (j - 1) df` (Hz), in seconds (times
`M_phys/dL` for strain). Per-mode band edges `fmin, fspa_max, f_b1, f_b2, fmax` (Hz).
"""
struct SEOBNRv5HMTable{V<:AbstractVector{Float64},Mx<:AbstractMatrix{Float64}}
    success::Bool
    M_sec::Float64
    M::Float64
    eta::Float64
    t_a::Float64
    nreal::Int
    ts::V
    phis::Mx
    omegas::Mx
    Qs::Mx
    dQs::Mx
    fmin::NTuple{6,Float64}
    fspa_max::NTuple{6,Float64}
    f_b1::NTuple{6,Float64}
    f_b2::NTuple{6,Float64}
    fmax::NTuple{6,Float64}
    df::Float64
    ft_re::Mx
    ft_im::Mx
    ft_dre::Mx
    ft_dim::Mx
end

# LAL-convention sign of the reduced FD mode h_{l,-m} = (-1)^l conj(h^LAL_lm), h^LAL_lm = (-1)^m h^int_lm.
@inline _seobv5hm_fdsign(k) = iseven(_SEOBV5HM_L[k] + _SEOBV5HM_M[k]) ? 1.0 : -1.0

# SPA amplitude ratio R = A / (varphi'/m)^((l+eps)/3) of mode k at tq.
@inline function _seobv5hm_R(dyn, D, nqc, k, tq)
    m, ex = _SEOBV5HM_M[k], (_SEOBV5HM_L[k] + _SEOBV5HM_M[k]) % 2
    A, _ = _seobv5hm_insp_ap(dyn, D, nqc, k, tq)
    return A / (_seobv5hm_insp_freq(dyn, D, nqc, k, tq) / m)^((_SEOBV5HM_L[k] + ex) / 3.0)
end

# Buffers of one binary's tables (`_seobv5hm_coeffs!`).
@inline function _seobv5hm_alloc()
    NT, NF, NM = _EOBIMR_NT, _SEOBV5HM_NFT, _SEOBV5HM_NMODES
    return (D = _seobv5hm_alloc_modes(), ts = zeros(NT), phis = zeros(NT, NM), omegas = zeros(NT, NM),
        Qs = zeros(NT, NM), dQs = zeros(NT, NM), ft_re = zeros(NF, NM), ft_im = zeros(NF, NM),
        ft_dre = zeros(NF, NM), ft_dim = zeros(NF, NM), zr = zeros(_SEOBV5HM_NSEG), zi = zeros(_SEOBV5HM_NSEG),
        wr = zeros(_SEOBV5HM_NSEG), wi = zeros(_SEOBV5HM_NSEG))
end

# The table from the buffers and the scalars `_seobv5hm_coeffs!` returns.
@inline _seobv5hm_table(sc, T) = SEOBNRv5HMTable(sc.success, sc.M_sec, sc.M, sc.eta, sc.t_a, sc.nreal, T.ts, T.phis,
    T.omegas, T.Qs, T.dQs, sc.fmin, sc.fspa_max, sc.f_b1, sc.f_b2, sc.fmax, sc.df, T.ft_re, T.ft_im, T.ft_dre, T.ft_dim)

# Band edges per mode (Hz).
@inline _seobv5hm_fnode_k(k, oms, i, Ms) = _SEOBV5HM_M[k] * oms[i, k] / (2pi * Ms)
@inline _seobv5hm_fb1_k(k, dyn, D, nqc, t_a, Ms) = _seobv5hm_insp_freq(dyn, D, nqc, k, t_a - _EOBIMR_TBLEND[1]) / (2pi * Ms)
@inline _seobv5hm_fmax_k(k, mr, Ms) = _EOBIMR_FHI_QNM * mr[k].wr / (2pi * Ms)

"""
    _seobv5hm_coeffs(mc, eta, chi1, chi2; f_min=20.0, GMsun_over_c3) -> SEOBNRv5HMTable
    _seobv5hm_coeffs!(T, mc, eta, chi1, chi2; ...) -> scalars

Build the frequency-domain tables of all six modes for one binary (see [`SEOBNRv5HM`](@ref)). The `!` form fills
the buffers `T` ([`_seobv5hm_alloc`](@ref)) and returns only the scalar part (isbits; what differentiated code
calls, then `_seobv5hm_table(sc, T)` inline).
"""
_seobv5hm_coeffs(mc, eta, chi1, chi2; kwargs...) =
    (T = _seobv5hm_alloc(); _seobv5hm_table(_seobv5hm_coeffs!(T, mc, eta, chi1, chi2; kwargs...), T))

function _seobv5hm_coeffs!(T, mc, eta, chi1, chi2; f_min = 20.0, GMsun_over_c3 = GMsun_over_c3)
    NT, NS, NF, NM = _EOBIMR_NT, _SEOBV5HM_NSEG, _SEOBV5HM_NFT, _SEOBV5HM_NMODES
    S = merge(_seobv5hm_build(T.D, mc, eta, chi1, chi2; f_min = f_min, GMsun_over_c3 = GMsun_over_c3), (D = T.D,))
    B, dyn, D, nqc, t_a = S.B, S.dyn, S.D, S.nqc, S.t_a
    b = S.base
    Ms = b.M_sec
    t_start = B.t_start

    # --- SPA nodes: PA nodes, then _EOBIMR_NSPA_RK nodes uniform in time up to the blend band's top (as the (2,2) table)
    t_spa_end = t_a - _EOBIMR_TBLEND[2]
    use_spa = t_spa_end > t_start + 10.0
    npa_use = 1
    for i in 1:_SEOBV5_PA_NMAX
        npa_use = (use_spa & (i <= dyn.npa) && (dyn.t[i] <= t_spa_end - 1.0)) ? i : npa_use
    end
    t_seg0 = dyn.t[npa_use]
    nspa = use_spa ? npa_use + _EOBIMR_NSPA_RK : 2
    t_top = use_spa ? t_spa_end : t_start + 2.0
    ts = T.ts
    for i in 1:NT
        ts[i] = i <= nspa ? (i <= npa_use ? dyn.t[i] : t_seg0 + (t_top - t_seg0) * Float64(i - npa_use) / Float64(nspa - npa_use)) :
                ts[nspa] + Float64(i - nspa)
    end
    phis, oms, Qs, dQs = T.phis, T.omegas, T.Qs, T.dQs
    fill!(phis, 0.0); fill!(oms, 0.0); fill!(Qs, 0.0); fill!(dQs, 0.0)
    for k in 1:NM
        m = _SEOBV5HM_M[k]
        for i in 1:NT
            if i <= nspa
                _, vp = _seobv5hm_insp_ap(dyn, D, nqc, k, ts[i])
                phis[i, k] = vp / m
                oms[i, k] = _seobv5hm_insp_freq(dyn, D, nqc, k, ts[i]) / m
                Qs[i, k] = _seobv5hm_R(dyn, D, nqc, k, ts[i])            # R and dR/dt for now, Q below
                dQs[i, k] = _seobv5hm_R(dyn, D, nqc, k, ts[i] + 0.5) - _seobv5hm_R(dyn, D, nqc, k, ts[i] - 0.5)
            else
                oms[i, k] = oms[nspa, k] + Float64(i - nspa)
            end
        end
        # R -> Q = R / sqrt(d omega/dt) (the nodal d omega/dt reads the neighbouring nodes' omega)
        omk = view(oms, :, k)
        for i in 1:NT
            if i <= nspa
                Qs[i, k], dQs[i, k] = _spa_node_q(ts, omk, Qs[i, k], dQs[i, k], i, nspa)
            end
        end
    end
    fmin = _seobv5hm_tup6(_seobv5hm_fnode_k, oms, 1, Ms)
    fspa_max = _seobv5hm_tup6(_seobv5hm_fnode_k, oms, nspa, Ms)
    f_b1 = use_spa ? _seobv5hm_tup6(_seobv5hm_fb1_k, dyn, D, nqc, t_a, Ms) : fmin
    f_b2 = use_spa ? fspa_max : fmin

    # --- Fourier tables: U(f_j) = sum_n z(t_n) e^{-2 pi i f_j Ms (t_n - t_a)} Ms dts, z = h_{l,-m}(t) (LAL),
    # dU/df = -2 pi i Ms (same sum with (t_n - t_a) z); one +i radix-2 FFT of conj(z) gives the -i sums.
    t_w = max(t_start, t_a - _EOBIMR_TSEG_PRE)
    Lt = min(_EOBIMR_TTAPER, 0.4 * (t_a - t_w))
    t_e = t_a + _EOBIMR_NTAU * S.mr[1].tau
    dts = (t_e - t_w) / Float64(NS - 1)
    df = 1.0 / (Float64(NS) * dts * Ms)
    ft_re, ft_im, ft_dre, ft_dim = T.ft_re, T.ft_im, T.ft_dre, T.ft_dim
    # heap scratch (4096-element static vectors made Enzyme's type analysis crawl)
    zr, zi, wr, wi = T.zr, T.zi, T.wr, T.wi
    for k in 1:NM
        sg = _seobv5hm_fdsign(k)
        for n in 1:NS
            tn = t_w + dts * Float64(n - 1)
            h = _seobv5hm_full(S, k, tn)                     # internal h_lm; z = sg conj(h), so conj(z) = sg h
            w = _eobimr_turn_on((tn - t_w) / Lt) * sg * Ms * dts
            zr[n] = w * real(h); zi[n] = w * imag(h)
            wr[n] = (tn - t_a) * zr[n]; wi[n] = (tn - t_a) * zi[n]
        end
        _eobimr_fft!(zr, zi)
        _eobimr_fft!(wr, wi)
        for j in 1:NF
            f = Float64(j - 1) * df
            th = -2pi * f * Ms * (t_w - t_a)                   # e^{-2 pi i f Ms (t_w - t_a)}
            c, s_ = cos(th), sin(th)
            Xr, Xi = zr[j], -zi[j]                            # conj(FFT+(conj z)) = sum z e^{-2 pi i j n / NS}
            Vr, Vi = wr[j], -wi[j]
            ft_re[j, k] = Xr * c - Xi * s_; ft_im[j, k] = Xr * s_ + Xi * c
            Yr = Vr * c - Vi * s_; Yi = Vr * s_ + Vi * c
            ft_dre[j, k] = 2pi * Ms * Yi; ft_dim[j, k] = -2pi * Ms * Yr      # -2 pi i Ms (Yr + i Yi)
        end
    end
    fmax = _seobv5hm_tup6(_seobv5hm_fmax_k, S.mr, Ms)
    covered = true
    for k in 1:NM
        covered &= fmax[k] < 0.45 / (dts * Ms)
    end
    success = B.success & covered
    return (success = success, M_sec = Ms, M = b.M, eta = b.nu, t_a = t_a, nreal = nspa, fmin = fmin,
        fspa_max = fspa_max, f_b1 = f_b1, f_b2 = f_b2, fmax = fmax, df = df)
end

# SPA part of mode k: reduced h_{l,-m}(f) (seconds). Stationary point from `_spa_root`, as `_eobimr_spa_eval`.
@inline function _seobv5hm_spa_eval(c, k, f)
    l, m = _SEOBV5HM_L[k], _SEOBV5HM_M[k]
    ex = (l + m) % 2
    Ms = c.M_sec
    f = clamp(f, c.fmin[k], c.fspa_max[k])
    omega_target = 2pi * Ms * f / m
    omk = view(c.omegas, :, k)
    i = clamp(_eobimr_searchsortedlast(omk, omega_target), 1, c.nreal - 1)
    r = _spa_root(c.ts, view(c.phis, :, k), omk, i, omega_target)
    # Q = R / sqrt(d omega/dt): NaN for a mode whose inspiral frequency is not increasing here (seen for an odd-m mode
    # near q = 1 with unequal spins, e.g. mc 29.6, q 0.991, chi 0.01 / 0.37): no stationary point, so the device kernels
    # and the sampler reject the point (`_spa_node_q`)
    Q, _ = _eobimr_hermite(c.Qs[i, k], c.dQs[i, k], c.Qs[i + 1, k], c.dQs[i + 1, k], c.ts[i + 1] - c.ts[i], r.s)
    # R omega^((l+eps)/3) Ms sqrt(2 pi / (m d omega/dt))
    amp = Q * omega_target^((l + ex) / 3.0) * Ms * sqrt(2pi / m)
    psi = m * r.phi_star - 2pi * f * Ms * (r.t_star - c.t_a) + pi / 4.0
    return _seobv5hm_fdsign(k) * amp * complex(cos(psi), sin(psi))
end

# Fourier-table part of mode k (uniform bins: no search), with the C1 top-of-band taper.
@inline function _seobv5hm_ft_eval(c, k, f)
    x = f / c.df
    j = clamp(floor(Int, x), 0, size(c.ft_re, 1) - 2)
    s = x - j
    jj = j + 1
    re, _ = _eobimr_hermite(c.ft_re[jj, k], c.ft_dre[jj, k], c.ft_re[jj + 1, k], c.ft_dre[jj + 1, k], c.df, s)
    im_, _ = _eobimr_hermite(c.ft_im[jj, k], c.ft_dim[jj, k], c.ft_im[jj + 1, k], c.ft_dim[jj + 1, k], c.df, s)
    fx = c.fmax[k]
    tap = 1.0 - _eobimr_smoothstep((f - 0.8 * fx) / (0.2 * fx))
    return tap * complex(re, im_)
end

"""
    _seobv5hm_fd_eval(c, k, f) -> ComplexF64

Reduced `h_{l,-m}(f)` of mode `k` from a [`SEOBNRv5HMTable`](@ref) (seconds; times `M GMsun_over_c2_Gpc / dL` for
strain per Hz), zero outside `[fmin, fmax]`. Duck-typed on the table's field names.
"""
@inline function _seobv5hm_fd_eval(c, k, f)
    (!c.success || f < c.fmin[k] || f > c.fmax[k]) && return zero(ComplexF64)
    if f <= c.f_b1[k]
        return _seobv5hm_spa_eval(c, k, f)
    elseif f >= c.f_b2[k]
        return _seobv5hm_ft_eval(c, k, f)
    end
    w = _eobimr_smoothstep((f - c.f_b1[k]) / (c.f_b2[k] - c.f_b1[k]))
    return (1.0 - w) * _seobv5hm_spa_eval(c, k, f) + w * _seobv5hm_ft_eval(c, k, f)
end

@inline function _seobv5hm_mode_index(l::Integer, m::Integer)
    for k in 1:_SEOBV5HM_NMODES
        (_SEOBV5HM_L[k] == l && _SEOBV5HM_M[k] == m) && return k
    end
    throw(ArgumentError("SEOBNRv5HM mode (l, m) = ($l, $m) not modelled; available: (2,2), (2,1), (3,3), (3,2), (4,4), (4,3)"))
end

"""
    h_lm(model::SEOBNRv5HM, l, m, f, m1, m2, chi1, chi2, dL; f_min=20.0, GMsun_over_c3, GMsun_over_c2_Gpc)

Single-mode frequency-domain strain `h_{l,-m}(f)` for positive `m` (LAL convention, as
`h_lm(::IMRPhenomXHM, ...)`), time origin at the merger (`t_attach`). `m1 >= m2` in solar masses, `chi1` the
spin of `m1`, `dL` in Gpc. `f` scalar or vector.
"""
function h_lm(model::SEOBNRv5HM, l::Integer, m::Integer, f, m1, m2, chi1, chi2, dL;
        f_min = 20.0, GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc)
    k = _seobv5hm_mode_index(l, m)
    mc, eta = _xhm_mc_eta(Float64(m1), Float64(m2))
    c = _seobv5hm_coeffs(mc, eta, Float64(chi1), Float64(chi2); f_min = f_min, GMsun_over_c3 = GMsun_over_c3)
    scale = c.M * Float64(GMsun_over_c2_Gpc) / Float64(dL)
    g(x) = scale * _seobv5hm_fd_eval(c, k, Float64(x))
    return f isa AbstractArray ? map(g, f) : g(f)
end

"""
    hphc(model::SEOBNRv5HM, f, m1, m2, chi1, chi2, dL, inclination, phiRef; f_min=20.0, ...)

Frequency-domain polarizations `(hp, hc)` from the six modes and their `-m` partners: `h+ - i hx = sum h_lm
-2Y_lm(inclination, pi/2 - phiRef)` (pyseobnr/LAL's azimuth convention), summed with
[`_xhm_add_mode!`](@ref)'s formula.
"""
function hphc(model::SEOBNRv5HM, f, m1, m2, chi1, chi2, dL, inclination, phiRef;
        f_min = 20.0, GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc)
    mc, eta = _xhm_mc_eta(Float64(m1), Float64(m2))
    c = _seobv5hm_coeffs(mc, eta, Float64(chi1), Float64(chi2); f_min = f_min, GMsun_over_c3 = GMsun_over_c3)
    scale = c.M * Float64(GMsun_over_c2_Gpc) / Float64(dL)
    az = pi / 2 - Float64(phiRef)
    function g(x)
        hp = zero(ComplexF64); hc = zero(ComplexF64)
        for k in 1:_SEOBV5HM_NMODES
            hp, hc = _xhm_add_mode!(hp, hc, _SEOBV5HM_L[k], _SEOBV5HM_M[k], scale * _seobv5hm_fd_eval(c, k, Float64(x)),
                Float64(inclination), az)
        end
        return hp, hc
    end
    if f isa AbstractArray
        r = map(g, f)
        return first.(r), last.(r)
    end
    return g(f)
end
