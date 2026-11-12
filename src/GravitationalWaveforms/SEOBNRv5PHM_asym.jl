using StaticArrays: SVector
using LinearAlgebra: dot, norm, cross

include("SEOBNRv5PHM_asym_fits_generated.jl")

"""
SEOBNRv5PHM anti-symmetric modes (pyseobnr 0.3.7 `eob/waveform/compute_antisymmetric.py`, Estelles et al. 2025;
on by default in pyseobnr): the anti-symmetric parts of the co-precessing (2,2), (3,3), (4,4) modes, which
precessing binaries radiate through the in-plane spins.

1. PN inspiral (`compute_asymmetric_PN`, Boyle et al. 2014: (2,2) to next-to-leading order, (3,3), (4,4) leading)
   from the projections of `S = S1 + S2` and `Sigma = m2 chi2 - m1 chi1` on the orbital triad: `n_hat` = the
   co-precessing frame's x axis turned by the orbital phase about `LN_hat` (`get_all_dynamics`), `lambda_hat =
   LN_hat x n_hat`. Evaluated directly at any time here (pyseobnr resamples on 32 points per orbit and splines).
2. Fits at `t_attach` (`fits_iv_mrd_antisymmetric`; generated `SEOBNRv5PHM_asym_fits_generated.jl`): the (2,2)
   amplitude and frequency, (3,3), (4,4) frequencies, the (2,2) merger-ringdown `c1f, c2f`; the other MRD
   coefficients from the aligned fits of the projections on `ell = LN_hat`. Clipped to pyseobnr's bounds.
3. Amplitude factor `1 + b6 v^6` on (2,2) (`apply_antisym_factorized_correction`, `b6` matching the amplitude fit)
   and a phase NQC `exp(i b1 pr / (r Omega))` (`apply_nqc_phase_antisymmetric`) on every mode whose mean amplitude
   over the last 10% of pyseobnr's resampling grid before `t_attach` is >= 1e-8 (the grid is rebuilt here for
   that test only).
4. Merger-ringdown (`compute_IMR_modes` with these fits), the same remnant and co-precessing QNM shift as the
   symmetric modes; the (3,3) attachment frequency is the *symmetric* (3,3) input-value fit (pyseobnr).
5. Negative-m partners `h_{l,-m} = -(-1)^l conj(h_lm)`, added to the symmetric modes.
"""
struct SEOBNRv5PHMAntisymmetric end

const _SEOBV5PHM_ASYM_K = (1, 3, 5)      # symmetric-mode indices of (2,2), (3,3), (4,4)

# Inspiral J -> P quaternion (LN_hat-following, also after t_attach), I -> P = I2J * J2P.
@inline function _seobv5phm_qJ2P_insp(S, om)
    return _qmul(_seobv5phm_qmin(S.b.tab, S.G, om, S.e1, S.e2, S.e3), S.C)
end

# Orbital-triad projections at dynamics time tq <= t_attach (`get_all_dynamics`).
@inline function _seobv5phm_asym_dyn(S, tq)
    r, ph, pr, pp, om, _ = _seobv5phm_dyn_at(S.dyn, tq)
    return _seobv5phm_asym_dyn_from(S, r, ph, pr, om, _seobv5phm_qJ2P_insp(S, om))
end
# ... at the attachment's sample times t_attach + 0.5 (i - 3), i = 1 .. 5: the attachment states `sts` (r, phi, pr,
# pphi, Omega) of the local RK4 re-integration (`_seobv5phm_attach_states`, the pre-record's `sts`), which continues
# the trajectory past the last node (t_attach within 1 M of the end, e.g. at the Omega peak when DeltaT_NR < 0);
# between them the quartic through all five (`_seobv5phm_sts_at`, x in units of 0.5 M from t_attach). (An RK4
# step from the last node per sample would need, in the GravitationalWaveBayes device reverse of the build,
# 74 KB of local memory per thread, 11 GiB at full V100 occupancy.)
@inline _seobv5phm_asym_dyn_st(S, st) = _seobv5phm_asym_dyn_from(S, st[1], st[2], st[3], st[5], _seobv5phm_qJ2P_insp(S, st[5]))
@inline function _seobv5phm_sts_at(sts, x)
    a, b, c, d, e = x + 2, x + 1, x, x - 1, x - 2
    w = (b * c * d * e / 24, -a * c * d * e / 6, a * b * d * e / 4, -a * b * c * e / 6, a * b * c * d / 24)
    return ntuple(k -> w[1] * sts[1][k] + w[2] * sts[2][k] + w[3] * sts[3][k] + w[4] * sts[4][k] + w[5] * sts[5][k], Val(5))
end
# ... from the dynamics at tq and the J -> P rotation there (shared with the polarizations in `_seobv5phm_hpc_insp`);
# the spins and LN_hat from the PN state directly (`seobnrv5phm_pn_everything`'s values, without its unused L vector)
@inline function _seobv5phm_asym_dyn_from(S, r, ph, pr, om, qJ2P)
    b = S.b
    tab = b.tab
    Lh, S1v, S2v = seobnrv5phm_pn_state(tab, om)
    c1 = S1v / tab.m1^2
    c2 = S2v / tab.m2^2
    LN = Lh / norm(Lh)
    qc = _qmul(S.qI2J, qJ2P)
    q2 = (cos(ph / 2), LN[1] * sin(ph / 2), LN[2] * sin(ph / 2), LN[3] * sin(ph / 2))
    Q = _qmul(q2, qc)
    xq = _qmul(_qmul(Q, (0.0, 1.0, 0.0, 0.0)), _qconj(Q))
    x = SVector(xq[2], xq[3], xq[4])
    x = x - dot(x, LN) * LN
    x = x / norm(x)
    lam = cross(LN, x)
    mA, mB = b.X1, b.X2
    Sig = mB * c2 - mA * c1
    S1 = mA^2 * c1
    S2 = mB^2 * c2
    St = S1 + S2
    return (v = cbrt(om), om = om, phi = ph, r = r, pr = pr, Sigma_n = dot(Sig, x), Sigma_lambda = dot(Sig, lam),
        S_n = dot(St, x), S_lambda = dot(St, lam), S1_l = dot(S1, LN), S2_l = dot(S2, LN))
end

# PN anti-symmetric mode j (1: (2,2), 2: (3,3), 3: (4,4)) from the triad projections (`compute_asymmetric_PN`).
@inline function _seobv5phm_asym_pn(j, a, nu, delta)
    v = a.v
    prefac = -8 * nu * v^2 * sqrt(pi / 5)
    Sn, Sl, Gn, Gl = a.S_n, a.S_lambda, a.Sigma_n, a.Sigma_lambda
    if j == 1
        lo = -v^2 * complex(Gl, Gn) / 2
        nlo = v^4 * complex(19 * delta * Sl + (5 - 43 * nu) * Gl, 182 * delta * Sn + 14 * (7 - 20 * nu) * Gn) / 84
        return prefac * (lo + nlo) * complex(cos(2 * a.phi), -sin(2 * a.phi))
    elseif j == 2
        lo = -sqrt(10 / 21) * v^3 * complex(Sn + delta * Gn, -(Sl + delta * Gl))
        return prefac * lo * complex(cos(3 * a.phi), -sin(3 * a.phi))
    else
        lo = 9 * sqrt(5 / 7) * v^4 * complex(delta * Sl + (1 - 3 * nu) * Gl, delta * Sn + (1 - 3 * nu) * Gn) / 8
        return prefac * lo * complex(cos(4 * a.phi), -sin(4 * a.phi))
    end
end

# p1 = pr / (r Omega) of a state (the NQC phase variable)
@inline _seobv5phm_asym_p1(st) = st[3] / (st[1] * st[5])

# Corrected inspiral anti-symmetric mode j at tq (from the triad projections `a` at tq). Inlined, as
# `seobnrv5phm_pn_state`: a non-inlined call taking a model record by reference made Enzyme's GPU reverse emit
# 1-byte-aligned atomic shadow adds (misaligned address / NVPTX "unsupported atomicrmw").
_seobv5phm_asym_insp(S, A, j, tq) = _seobv5phm_asym_insp_a(S, A, j, _seobv5phm_asym_dyn(S, tq))
@inline function _seobv5phm_asym_insp_a(S, A, j, a)
    h = _seobv5phm_asym_pn(j, a, S.b.nu, S.b.delta)
    if A.flags[j]
        if j == 1
            h *= 1 + A.b6 * a.v^6
        end
        x = A.b1[j] * a.pr / (a.r * a.om)
        h *= complex(cos(x), sin(x))
    end
    return h
end

# NQC flag of mode j: Omega-weighted mean |h_j| over [tw0, t_a] (64 samples) >= 1e-8.
function _seobv5phm_asym_flag(S, j, tw0, t_a)
    acc = 0.0; wsum = 0.0
    for i in 1:64
        tq = tw0 + (t_a - tw0) * (i - 0.5) / 64
        a = _seobv5phm_asym_dyn(S, tq)
        acc += abs(_seobv5phm_asym_pn(j, a, S.b.nu, S.b.delta)) * a.om
        wsum += a.om
    end
    return acc / wsum >= 1e-8
end
# Phase-NQC coefficient of mode j (0 if its flag is off).
function _seobv5phm_asym_b1(S, A0, j, sts, he, om_fit, P)
    A0.flags[j] || return 0.0
    hp = _seobv5phm_asym_insp_a(S, A0, j, _seobv5phm_asym_dyn_st(S, _seobv5phm_sts_at(sts, he / 0.5)))
    hm = _seobv5phm_asym_insp_a(S, A0, j, _seobv5phm_asym_dyn_st(S, _seobv5phm_sts_at(sts, -he / 0.5)))
    w = abs(angle(hp * conj(hm)) / (2he))
    return (om_fit[j] - w) / P
end
# The same from the triad projections at t_a -+ he (`ahe`).
@inline function _seobv5phm_asym_b1_a(S, A0, j, he, om_fit, P, ahe)
    A0.flags[j] || return 0.0
    hp = _seobv5phm_asym_insp_a(S, A0, j, ahe[2])
    hm = _seobv5phm_asym_insp_a(S, A0, j, ahe[1])
    w = abs(angle(hp * conj(hm)) / (2he))
    return (om_fit[j] - w) / P
end
# Merger-ringdown of mode j from the corrected inspiral's triad projections at t_a + (-1, -0.5, 0, 0.5, 1)
# (`a5`) and the MRD fits `fits`.
@inline function _seobv5phm_asym_mr_a(S, A1, j, a5, om_iv_sym, fits)
    k = _SEOBV5PHM_ASYM_K[j]
    m = _SEOBV5HM_M[k]
    hs0 = (_seobv5phm_asym_insp_a(S, A1, j, a5[1]), _seobv5phm_asym_insp_a(S, A1, j, a5[2]),
        _seobv5phm_asym_insp_a(S, A1, j, a5[3]), _seobv5phm_asym_insp_a(S, A1, j, a5[4]),
        _seobv5phm_asym_insp_a(S, A1, j, a5[5]))
    # a mode whose flag is off (mean |h| < 1e-8, e.g. spins along L, where it is exactly 0) gets a zero-amplitude
    # merger-ringdown; abs / the phase unwrap see a unit dummy instead of its (vanishing) amplitudes, whose derivatives
    # are 0/0 there (NaN in a reverse pass)
    f = A1.flags[j]
    hs = map(h -> ifelse(f, h, one(h)), hs0)
    Am, dAm, _ = _seobv5hm_d5(abs.(hs), 0.5)
    Am, dAm = ifelse(f, Am, zero(Am)), ifelse(f, dAm, zero(dAm))
    ph = _seobv5hm_unwrap5(hs)
    om = isodd(m) ? _seobv5hm_pick(om_iv_sym, k) : _seobv5hm_d5(ph, 0.5)[2]
    return _seobv5hm_mr(S.b.nu, k, Am, dAm, ph[3], om, S.Mf, S.af, fits, S.rot)
end

"""
    _seobv5phm_asym_build(S, sts[, flags]) -> NamedTuple

Flags, fits, `b6`, `b1` and merger-ringdown parameters of the anti-symmetric modes for a built model `S`
([`_seobv5phm_build`](@ref)) with the attachment states `sts` of its pre-record.
"""
# NQC flags of the three modes (Booleans: no derivative; the device reverse passes the primal ones in).
function _seobv5phm_asym_flags(S)
    d = S.dyn
    t_a = S.t_a
    # NQC flags (`apply_antisym_factorized_correction`): the mean |h_j| over the last 10% of pyseobnr's resampling grid
    # before t_attach. That grid has 32 points per orbit (`compute_time_for_asym`), so the window is the last 10% of the
    # total orbital phase before t_attach and the mean is over orbital phase: here 64 samples uniform in time weighted
    # by Omega (a threshold test at 1e-8, i.e. whether the in-plane spins vanish; no data-dependent grid).
    t1, tn = d.t[1], d.t[d.n]
    phspan = _seobv5phm_dyn_at(d, tn)[2] - _seobv5phm_dyn_at(d, t1)[2]
    tw0 = max(_seobv5phm_t_at_phase(d, _seobv5phm_dyn_at(d, t_a)[2] - phspan / 10), t1)
    # (explicit tuples of helper calls, here and below: closures capturing S defeat inference on GPUs)
    return (_seobv5phm_asym_flag(S, 1, tw0, t_a), _seobv5phm_asym_flag(S, 2, tw0, t_a), _seobv5phm_asym_flag(S, 3, tw0, t_a))
end

@inline function _seobv5phm_asym_build(S, sts, flags = _seobv5phm_asym_flags(S))
    b = S.b
    nu, delta, X1, X2 = b.nu, b.delta, b.X1, b.X2
    t_a = S.t_a

    # fits at t_attach (get_params_for_fit, fits_iv_mrd_antisymmetric)
    a = _seobv5phm_asym_dyn(S, t_a)
    q = X1 / X2
    chi1L = a.S1_l / X1^2
    chi2L = a.S2_l / X2^2
    chi_eff = (q * chi1L + chi2L) / (1 + q)
    chi_a = (chi1L - chi2L) / 2
    # in-plane magnitudes and the double angles of their directions, without sqrt / atan at zero in-plane spin (spins
    # along L: singular derivatives, NaN in a reverse pass even where the anti-symmetric part is switched off)
    Sig2 = a.Sigma_n^2 + a.Sigma_lambda^2
    S2 = a.S_n^2 + a.S_lambda^2
    Sig_in = _seobv5phm_ssqrt(Sig2)
    S_in = _seobv5phm_ssqrt(S2)
    # (the quotients' denominators are offset, not switched: LLVM folds `ifelse(t, x / ifelse(t, y, 1), 1)` back to
    # `ifelse(t, x / y, 1)`, and the reverse of the dead branch, with a zero cotangent, is 0/0 at y = 0: NaN gradients at
    # exactly aligned spins. The offset 1e-150 is below rounding for every y > 1e-130 (where the quotient is used), and
    # its square stays a normal number.)
    tS = S2 > 1e-130; tG = Sig2 > 1e-130
    dS = S2 + 1e-150; dG = Sig2 + 1e-150
    c2S = ifelse(tS, (a.S_n^2 - a.S_lambda^2) / dS, one(S2)); s2S = ifelse(tS, 2 * a.S_n * a.S_lambda / dS, zero(S2))
    c2G = ifelse(tG, (a.Sigma_n^2 - a.Sigma_lambda^2) / dG, one(S2)); s2G = ifelse(tG, 2 * a.Sigma_n * a.Sigma_lambda / dG, zero(S2))
    habs22 = min(abs(_seobv5phm_asymfit_habs_22(nu, S_in, Sig_in, chi_eff, chi_a, c2G, s2G)) * Sig_in, 0.6)
    om_pk = abs(_seobv5phm_asymfit_omega_peak_22(nu, S_in, Sig_in, chi_eff, chi_a, c2G, s2G))
    om22 = clamp(_seobv5phm_asymfit_omegalm_22(nu, S_in, Sig_in, chi_eff, chi_a, c2G, s2G) + om_pk, 0.05, 0.5)
    c1f22 = clamp(_seobv5phm_asymfit_c1f_22(nu, chi_eff, chi_a, c2S, s2S), 0.05, 0.15)
    c2f22 = clamp(_seobv5phm_asymfit_c2f_22(nu, c1f22, chi_eff, chi_a, c2S, s2S), -1.4, -0.4)
    om33 = clamp(_seobv5phm_asymfit_omegalm_33(nu, S_in, Sig_in, chi_eff, chi_a, c2G, s2G), 0.03, 0.8)
    om44 = clamp(_seobv5phm_asymfit_omegalm_44(nu, S_in, Sig_in, chi_eff, chi_a, c2G, s2G), 0.03, 1.42)
    om_fit = (abs(om22), abs(om33), abs(om44))

    # (2,2) amplitude factor: b6 = (nu A_fit - |h|(t_a)) / (|h|(t_a) v(t_a)^6)
    h22a = abs(ifelse(flags[1], _seobv5phm_asym_pn(1, a, nu, delta), one(ComplexF64)))   # (abs' is 0/0 at 0: see above)
    b6 = flags[1] ? (nu * habs22 - h22a) / (h22a * a.v^6) : 0.0
    # phase NQC: b1 = (|omega_fit| - |omega_mode|(t_a)) / P, P = -dp1/dt(t_a)
    he = 0.05
    sm, sp = _seobv5phm_sts_at(sts, -he / 0.5), _seobv5phm_sts_at(sts, he / 0.5)
    P = -(_seobv5phm_asym_p1(sp) - _seobv5phm_asym_p1(sm)) / (2he)
    A0 = (flags = flags, b6 = b6, b1 = (0.0, 0.0, 0.0))
    # the triad projections at the 7 times the rest reads (each evaluated once: a device reverse differentiates them)
    ahe = (_seobv5phm_asym_dyn_st(S, sm), _seobv5phm_asym_dyn_st(S, sp))
    a5 = (_seobv5phm_asym_dyn_st(S, sts[1]), _seobv5phm_asym_dyn_st(S, sts[2]), _seobv5phm_asym_dyn_st(S, sts[3]),
        _seobv5phm_asym_dyn_st(S, sts[4]), _seobv5phm_asym_dyn_st(S, sts[5]))
    b1 = (_seobv5phm_asym_b1_a(S, A0, 1, he, om_fit, P, ahe), _seobv5phm_asym_b1_a(S, A0, 2, he, om_fit, P, ahe),
        _seobv5phm_asym_b1_a(S, A0, 3, he, om_fit, P, ahe))
    A1 = (flags = flags, b6 = b6, b1 = b1)

    # merger-ringdown: attachment values of the corrected inspiral modes, aligned MRD fits of the ell projections
    # with the (2,2) c1f, c2f replaced
    fvL = _seobv5hm_fitvars(X1, X2, chi1L, chi2L)
    mrf = (_seobv5hm_mr_c1f(fvL), _seobv5hm_mr_c2f(fvL), _seobv5hm_mr_d1f(fvL), _seobv5hm_mr_d2f(fvL))
    fv_sym = _seobv5hm_fitvars(X1, X2, S.chi1LN_a, S.chi2LN_a)
    om_iv_sym = _seobv5hm_iv_omega(fv_sym)
    fits22 = (c1f22, c2f22, _seobv5hm_pick(mrf[3], 1), _seobv5hm_pick(mrf[4], 1))
    mr = (_seobv5phm_asym_mr_a(S, A1, 1, a5, om_iv_sym, fits22), _seobv5phm_asym_mr_a(S, A1, 2, a5, om_iv_sym, _seobv5hm_fits_k(3, mrf)),
        _seobv5phm_asym_mr_a(S, A1, 3, a5, om_iv_sym, _seobv5hm_fits_k(5, mrf)))
    return (flags = flags, b6 = b6, b1 = b1, mr = mr, habs22 = habs22, om_fit = om_fit)
end

# The three anti-symmetric modes ((2,2), (3,3), (4,4)) at tq, sharing one evaluation of the triad projections.
function _seobv5phm_asym3(S, tq)
    A = S.asym
    dt = tq - S.t_a
    if dt <= 0
        a = _seobv5phm_asym_dyn(S, tq)
        return (_seobv5phm_asym_insp_a(S, A, 1, a), _seobv5phm_asym_insp_a(S, A, 2, a), _seobv5phm_asym_insp_a(S, A, 3, a))
    end
    return (_seobv5hm_mr_eval(A.mr[1], dt), _seobv5hm_mr_eval(A.mr[2], dt), _seobv5hm_mr_eval(A.mr[3], dt))
end

# Anti-symmetric mode j (positive m) at dynamics time tq: corrected PN inspiral, then merger-ringdown.
function _seobv5phm_asym(S, j, tq)
    A = S.asym
    dt = tq - S.t_a
    dt <= 0 && return _seobv5phm_asym_insp(S, A, j, tq)
    return _seobv5hm_mr_eval(A.mr[j], dt)
end
