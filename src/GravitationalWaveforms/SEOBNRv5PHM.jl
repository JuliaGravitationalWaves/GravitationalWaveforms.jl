using StaticArrays: SVector
using LinearAlgebra: dot, norm, cross

export seobnrv5phm_td

"""
SEOBNRv5PHM waveform in the time domain (milestone P3 of GravitationalWaveBayes' `SEOBNRv5PHM_DESIGN.md`): pyseobnr 0.3.7's
`SEOBNRv5PHM_opt._evaluate_model` after the dynamics ([`seobnrv5phm_dynamics`](@ref)).

1. **Attachment** from the dynamics' dense output: the orbital frequency at `r = 10 M` gives the spins for the
   remnant's Kerr ISCO (aligned HBR2016 of the `LN_hat` projections) and the final state; `t_ISCO` where
   `r = r_ISCO` (pyseobnr: argmin on a 0.001 M grid; here a root); `t_attach = t_ISCO - DeltaT_NR(nu, a+, a-)`
   with the projections at `t_ISCO`, both clamped to the end of the dynamics: pyseobnr's last node, here the
   interpolated time of the stopping event (`_seobv5phm_t_stop`, e.g. the Omega peak; the last node moves
   with the step rule). States at `t_attach + (-1 .. 1)` by a local re-integration, as in
   [`_seobv5hm_attach_states`](@ref), which continue the trajectory past the last node; the anti-symmetric modes'
   samples there are read from these states (`_seobv5phm_sts_at`).
2. **Co-precessing modes** (2,2), (2,1), (3,3), (3,2), (4,4), (4,3): the SEOBNRv5HM inspiral modes with the
   rho/delta coefficients of each node's own `LN_hat` projections (pyseobnr `compute_hlms` with `aligned = False`),
   the special `(2,1)`, `(4,3)` coefficients, input-value and merger-ringdown fits of the projections at
   `t_attach`; NQC as SEOBNRv5HM. Remnant: UIB2016 mass and **precessing HBR2016 spin** of the spins at
   `r = 10 M`, signed by the aligned fit (`precessing_final_spin`, including its `phi12 = arccos(perp1 . perp2 /
   (a1 a2))`); merger-ringdown QNMs shifted to the co-precessing frame, `omega_lm -> omega_lm - m (1 - |cos
   beta_attach|) (omega_220 - omega_210) sign(af)`.
3. **Frames**: J frame from `J = S1 + S2 + L` at `t_attach` (`SEOBBuildJframeVectors`), the co-precessing frame
   following `LN_hat` with the minimal-rotation condition `dgamma = -cos(beta) dalpha` (here integrated along the
   PN table in orbital frequency; pyseobnr differentiates splined quaternions), fixed so that the co-precessing
   frame equals the inertial frame at the reference time; after `t_attach` simple precession about `J` at the QNM
   difference rate (`rd_approx`, no smoothing, constant beta).
4. **Polarizations** `h+ - i hx = sum_lm h_lm(t) -2Y_lm(beta_T, alpha_T) e^{2 i gamma_T}` from the Euler angles of
   the total rotation (`compute_hpc_from_coprec_modes`), negative-m modes `h_{l,-m} = (-1)^l conj(h_lm)`.

Anti-symmetric modes (pyseobnr's default, `antisym = true`): see [`SEOBNRv5PHMAntisymmetric`](@ref).

Times are the dynamics' (from its start); pyseobnr's `t = 0` (the frame-invariant amplitude's peak) is `S.t_peak`.
"""
struct SEOBNRv5PHMTD end

# ---------------------------------------------------------------------------------------------------
# Dense output of the dynamics and crossings
# ---------------------------------------------------------------------------------------------------

# (r, phi, pr, pphi, Omega, dr/dt) at time tq (clamped to the dynamics).
@inline function _seobv5phm_dyn_at(d, tq)
    n = d.n
    tq = clamp(tq, d.t[1], d.t[n])
    i = clamp(_eobimr_searchsortedlast(d.t, tq), 1, n - 1)
    h = d.t[i+1] - d.t[i]
    s = (tq - d.t[i]) / h
    r, rd = _eobimr_hermite(d.r[i], d.dr[i], d.r[i+1], d.dr[i+1], h, s)
    ph, om = _eobimr_hermite(d.phi[i], d.dphi[i], d.phi[i+1], d.dphi[i+1], h, s)
    pr, _ = _eobimr_hermite(d.pr[i], d.dpr[i], d.pr[i+1], d.dpr[i+1], h, s)
    pp, _ = _eobimr_hermite(d.pphi[i], d.dpphi[i], d.pphi[i+1], d.dpphi[i+1], h, s)
    return r, ph, pr, pp, om, rd
end

"""
    _seobv5phm_crossing(d, y, dy, yq, sgn) -> t

First time the Hermite interpolant of node values `y` (time derivatives `dy`) crosses `yq` (`sgn = -1`: from
above, as `r`; `+1`: from below, as `Omega`); the first/last node time if it starts beyond / never reaches it.
Bisection on the bracketing interval (fixed 60 halvings), then one Newton step: it leaves the converged root
unchanged and gives automatic differentiation the implicit-function derivative (bisection alone has none).
"""
function _seobv5phm_crossing(d, y, dy, yq, sgn)
    n = d.n
    sgn * (y[1] - yq) >= 0 && return d.t[1]
    i = 0
    @inbounds for j in 1:n-1
        if i == 0 && sgn * (y[j+1] - yq) >= 0
            i = j
        end
    end
    i == 0 && return d.t[n]
    h = d.t[i+1] - d.t[i]
    lo, hi = zero(h), one(h)
    for _ in 1:60
        mid = (lo + hi) / 2
        v, _ = _eobimr_hermite(y[i], dy[i], y[i+1], dy[i+1], h, mid)
        if sgn * (v - yq) >= 0
            hi = mid
        else
            lo = mid
        end
    end
    sm = (lo + hi) / 2
    v, dv = _eobimr_hermite(y[i], dy[i], y[i+1], dy[i+1], h, sm)
    return d.t[i] + sm * h - (v - yq) / dv
end

# Second time derivative of the cubic Hermite interpolant of `_eobimr_hermite` at s in [0, 1].
@inline function _seobv5phm_hermite_dd(y0, dy0, y1, dy1, h, s)
    return ((12s - 6) * y0 + (6s - 4) * h * dy0 + (6 - 12s) * y1 + (6s - 2) * h * dy1) / (h * h)
end

# First time the orbital phase (Hermite interpolant) reaches ph (bisection).
function _seobv5phm_t_at_phase(d, ph)
    n = d.n
    d.phi[1] >= ph && return d.t[1]
    i = 0
    @inbounds for j in 1:n-1
        if i == 0 && d.phi[j+1] >= ph
            i = j
        end
    end
    i == 0 && return d.t[n]
    h = d.t[i+1] - d.t[i]
    lo, hi = zero(h), one(h)
    for _ in 1:60
        mid = (lo + hi) / 2
        v, _ = _eobimr_hermite(d.phi[i], d.dphi[i], d.phi[i+1], d.dphi[i+1], h, mid)
        if v >= ph
            hi = mid
        else
            lo = mid
        end
    end
    return d.t[i] + (lo + hi) / 2 * h
end

# First time the orbital frequency (derivative of the phase's Hermite interpolant) reaches w (bisection).
function _seobv5phm_t_at_omega(d, w)
    n = d.n
    d.dphi[1] >= w && return d.t[1]
    i = 0
    @inbounds for j in 1:n-1
        if i == 0 && d.dphi[j+1] >= w
            i = j
        end
    end
    i == 0 && return d.t[n]
    h = d.t[i+1] - d.t[i]
    lo, hi = zero(h), one(h)
    for _ in 1:60
        mid = (lo + hi) / 2
        _, om = _eobimr_hermite(d.phi[i], d.dphi[i], d.phi[i+1], d.dphi[i+1], h, mid)
        if om >= w
            hi = mid
        else
            lo = mid
        end
    end
    sm = (lo + hi) / 2           # + one Newton step (see `_seobv5phm_crossing`)
    _, om = _eobimr_hermite(d.phi[i], d.dphi[i], d.phi[i+1], d.dphi[i+1], h, sm)
    return d.t[i] + sm * h - (om - w) / _seobv5phm_hermite_dd(d.phi[i], d.dphi[i], d.phi[i+1], d.dphi[i+1], h, sm)
end

# ---------------------------------------------------------------------------------------------------
# Spins -> mode coefficients; Hamiltonian quantities at a state
# ---------------------------------------------------------------------------------------------------

# The fields of `b` the SEOBNRv5HM mode functions read, with the LN_hat projections of spins `s`.
@inline function _seobv5phm_modeb(b, s)
    chiS = (s.chi1LN + s.chi2LN) / 2
    chiA = (s.chi1LN - s.chi2LN) / 2
    return (nu = b.nu, fc = s.fc, delta = b.delta, chiS = chiS, chiA = chiA)
end

# (H, Heff, omega_circ) at (r, pr, pphi) with spins s.
@inline function _seobv5phm_hstate(r, pr, pp, s, b)
    return _seobv5phm_sym_state(_seobv5phm_hargs(r, pr, pp, s, b)...)
end

# The six reduced modes at a state st = (r, phi, pr, pphi, Omega), spins read at the state's frequency.
# Mode k's part of `_seobv5phm_post_nqc`: its reduced mode at the five attachment states and its NQC coefficients
# (the six are independent: a device reverse runs one per mode).
@inline function _seobv5phm_post_nqc_k(b, R, k::Int)
    sts = R.sts
    _, ivs = _seobv5phm_post_fits(b, R)
    Hk = (_seobv5phm_state_mode(sts[1], b, R, k), _seobv5phm_state_mode(sts[2], b, R, k),
        _seobv5phm_state_mode(sts[3], b, R, k), _seobv5phm_state_mode(sts[4], b, R, k), _seobv5phm_state_mode(sts[5], b, R, k))
    skip_odd = (b.delta < 1e-4) & (((abs(R.chi1a) < 1e-4) & (abs(R.chi2a) < 1e-4)) | (abs(R.chi1a - R.chi2a) < 1e-4))
    nk = (isodd(_SEOBV5HM_M[k]) && skip_odd) ? (0.0, 0.0, 0.0, 0.0, 0.0) :
        _seobv5hm_nqc(sts, Hk, k, b.nu, _seobv5hm_ivk(k, ivs))
    return Hk, nk
end
@inline function _seobv5phm_state_mode(st, b, R, k::Int)
    r, _, pr, pp, om = st
    s = _seobv5phm_spins(b, om)
    bm = _seobv5phm_modeb(b, s)
    H, Heff, om_c = _seobv5phm_hstate(r, pr, pp, s, b)
    return _seobv5hm_reduced_mode(k, om, om_c, H, Heff, pp, bm, R.pref[k], R.c21, R.c43)
end
@inline function _seobv5phm_state_modes(st, b, pref, c21, c43)
    r, _, pr, pp, om = st
    s = _seobv5phm_spins(b, om)
    bm = _seobv5phm_modeb(b, s)
    H, Heff, om_c = _seobv5phm_hstate(r, pr, pp, s, b)
    return ntuple(k -> _seobv5hm_reduced_mode(k, om, om_c, H, Heff, pp, bm, pref[k], c21, c43), 6)
end

# One RK4 step of the precessing EOB equations of motion with the spins of the plunge (read at om_s, then at each
# stage's frequency: `_seobv5phm_rk4_inc`), plus Omega at its start.
@inline function _seobv5phm_rk4(r, ph, pr, pp, h, om_s, b)
    s = _seobv5phm_spins(b, om_s)
    k1 = _seobv5phm_rhs(r, pr, pp, s, b)
    Dr, Dph, Dpr, Dpp = _seobv5phm_rk4_inc(r, pr, pp, h, k1, s, _seobv5phm_pn_interval(b.tab, om_s), b)
    return (r + Dr, ph + Dph, pr + Dpr, pp + Dpp, k1[2])
end

# RK4 steps from the last node before t_a - 1 to t_a - 1, and per 0.5 M between the five states (one each,
# as SEOBNRv5HM's `_SEOBV5HM_ATT_N0` / `_NF`; the GravitationalWaveBayes device reverse reads these too)
const _SEOBV5PHM_ATT_N0 = 1
const _SEOBV5PHM_ATT_NF = 1

"""
    _seobv5phm_attach_states(d, b, t_a) -> NTuple{5}

States `(r, phi, pr, pphi, Omega)` at `t_a + (-1, -0.5, 0, 0.5, 1)` by re-integrating from the last node before
`t_a - 1` (`_SEOBV5PHM_ATT_N0` steps, then `_SEOBV5PHM_ATT_NF` steps per 0.5 M), the plunge's RK4 steps
(`_seobv5phm_rk4`). See [`_seobv5hm_attach_states`](@ref).
"""
function _seobv5phm_attach_states(d, b, t_a)
    n = d.n
    i0 = clamp(_eobimr_searchsortedlast(d.t, t_a - 1), 1, n)
    r, ph, pr, pp, t0 = d.r[i0], d.phi[i0], d.pr[i0], d.pphi[i0], d.t[i0]
    om_s = d.dphi[i0]      # as the plunge's step from node i0 (`omega_s[i0]` is the frequency of the step into it)
    N0, NF = _SEOBV5PHM_ATT_N0, _SEOBV5PHM_ATT_NF
    h0 = (t_a - 1 - t0) / N0
    z = (zero(r), zero(r), zero(r), zero(r), zero(r))
    s1 = s2 = s3 = s4 = s5 = z
    for j in 0:(N0 + 4NF)
        rn, phn, prn, ppn, om = _seobv5phm_rk4(r, ph, pr, pp, ifelse(j < N0, h0, oftype(h0, 0.5 / NF)), om_s, b)
        cur = (r, ph, pr, pp, om)
        s1 = ifelse(j == N0, cur, s1)
        s2 = ifelse(j == N0 + NF, cur, s2)
        s3 = ifelse(j == N0 + 2NF, cur, s3)
        s4 = ifelse(j == N0 + 3NF, cur, s4)
        s5 = ifelse(j == N0 + 4NF, cur, s5)
        r, ph, pr, pp = rn, phn, prn, ppn
        om_s = om
    end
    return (s1, s2, s3, s4, s5)
end

# ---------------------------------------------------------------------------------------------------
# Remnant: precessing final spin (precessing_final_spin, bbh_final_spin_precessing_HBR2016 "M3J4")
# ---------------------------------------------------------------------------------------------------

"""
    _seobv5phm_final_spin(m1, m2, chi1LN, chi2LN, chi1v, chi2v, LNhat) -> (af, sign)

pyseobnr `precessing_final_spin`: magnitude from the precessing HBR2016 fit of the spin magnitudes, tilts w.r.t.
`LN_hat` and `phi12 = arccos(clip(perp1 . perp2 / (a1 a2)))` (pyseobnr normalizes the in-plane components by the
full spin magnitudes), sign from the aligned HBR2016 fit of the projections.

Written without the angles (their derivatives are singular at zero tilt, i.e. spins along L): with x = cos(tilt),
cos(tilt + e sin(tilt)) = x cos(e s) - s sin(e s), s^2 = 1 - x^2, both terms even in s (series in y^2 = e^2 s^2 <=
5.8e-4, four terms: 1e-17 relative), and cos(phi12) = clip(perp1 . perp2 / (a1 a2)).
"""
@inline function _seobv5phm_final_spin(m1, m2, chi1LN, chi2LN, chi1v, chi2v, LNhat)
    a1 = norm(chi1v)
    a2 = norm(chi2v)
    cb = a1 != 0 ? clamp(chi1LN / a1, -one(a1), one(a1)) : one(a1)
    cg = a2 != 0 ? clamp(chi2LN / a2, -one(a2), one(a2)) : one(a2)
    p1 = chi1v - chi1LN * LNhat
    p2 = chi2v - chi2LN * LNhat
    cphi12 = (a1 == 0 || a2 == 0) ? zero(a1) : clamp(dot(p1 / a1, p2 / a2), -one(a1), one(a1))
    af_np = _seobv5_final_spin_hbr2016(m1, m2, chi1LN, chi2LN)
    sgn = (af_np != 0 && af_np < 0) ? -1 : 1
    q = m2 / m1
    eps = 0.024
    cbs = _seobv5phm_cos_tilt_shift(cb, eps)
    cgs = _seobv5phm_cos_tilt_shift(cg, eps)
    ca = _seobv5phm_ssqrt((1 - cb * cb) * (1 - cg * cg)) * cphi12 + cb * cg
    q2 = q * q
    ell = _seobv5_hbr2016_ell(m1, m2, a1 * cbs, a2 * cgs)
    arg = a1 * a1 + a2 * a2 * q2 * q2 + 2 * a1 * a2 * q2 * ca + 2 * (a1 * cbs + a2 * q2 * cgs) * ell * q + ell * ell * q2
    return sgn * sqrt(max(arg, zero(arg))) / ((1 + q) * (1 + q)), sgn
end

# cos(t + e sin(t)) from x = cos(t) (t in [0, pi]), smooth at t = 0 (see `_seobv5phm_final_spin`)
@inline function _seobv5phm_cos_tilt_shift(x, e)
    s2 = max(1 - x * x, zero(x))
    y2 = e * e * s2
    c = 1 - y2 / 2 * (1 - y2 / 12 * (1 - y2 / 30 * (1 - y2 / 56)))           # cos(e s)
    sc = 1 - y2 / 6 * (1 - y2 / 20 * (1 - y2 / 42 * (1 - y2 / 72)))          # sin(e s) / (e s)
    return x * c - e * s2 * sc
end

# ---------------------------------------------------------------------------------------------------
# Quaternions (w, x, y, z) as numpy-quaternion
# ---------------------------------------------------------------------------------------------------

@inline _qmul(a, b) = (a[1] * b[1] - a[2] * b[2] - a[3] * b[3] - a[4] * b[4],
    a[1] * b[2] + a[2] * b[1] + a[3] * b[4] - a[4] * b[3],
    a[1] * b[3] - a[2] * b[4] + a[3] * b[1] + a[4] * b[2],
    a[1] * b[4] + a[2] * b[3] - a[3] * b[2] + a[4] * b[1])
@inline _qconj(a) = (a[1], -a[2], -a[3], -a[4])

# R = exp(alpha z/2) exp(beta y/2) exp(gamma z/2) (quaternion.from_euler_angles)
@inline function _qeuler(al, be, ga)
    cb, sb = cos(be / 2), sin(be / 2)
    return (cb * cos((al + ga) / 2), -sb * sin((al - ga) / 2), sb * cos((al - ga) / 2), cb * sin((al + ga) / 2))
end

# quaternion.as_euler_angles
@inline function _qangles(q)
    n = q[1]^2 + q[2]^2 + q[3]^2 + q[4]^2
    a1 = atan(q[4], q[1])
    a2 = atan(-q[2], q[3])
    return a1 + a2, 2 * acos(clamp(sqrt((q[1]^2 + q[4]^2) / n), -one(n), one(n))), a1 - a2
end

# quaternion.from_rotation_matrix for a proper rotation with columns c1, c2, c3 (Shepperd's method)
@inline function _qfrom_rotmat(c1, c2, c3)
    m11, m21, m31 = c1
    m12, m22, m32 = c2
    m13, m23, m33 = c3
    tr = m11 + m22 + m33
    if tr > 0
        s = 2 * sqrt(1 + tr)
        q = (s / 4, (m32 - m23) / s, (m13 - m31) / s, (m21 - m12) / s)
    elseif m11 > m22 && m11 > m33
        s = 2 * sqrt(1 + m11 - m22 - m33)
        q = ((m32 - m23) / s, s / 4, (m12 + m21) / s, (m13 + m31) / s)
    elseif m22 > m33
        s = 2 * sqrt(1 + m22 - m11 - m33)
        q = ((m13 - m31) / s, (m12 + m21) / s, s / 4, (m23 + m32) / s)
    else
        s = 2 * sqrt(1 + m33 - m11 - m22)
        q = ((m21 - m12) / s, (m13 + m31) / s, (m23 + m32) / s, s / 4)
    end
    return q
end

"""
    _seobv5phm_jframe(Jhat) -> (e1J, e2J, e3J)

`SEOBBuildJframeVectors`: `e3J = Jhat`, `e1J` the normalized projection of `x` orthogonal to it (blended
continuously towards `y` when `1 - |x . Jhat| < 1e-4`), `e2J = e3J x e1J`.
"""
@inline function _seobv5phm_jframe(Jhat)
    ex = SVector(1.0, 0.0, 0.0)
    ey = SVector(0.0, 1.0, 0.0)
    e3 = Jhat
    xd = dot(ex, e3)
    yd = dot(ey, e3)
    lam = 1 - abs(xd)
    e1 = if lam > 1e-4
        (ex - xd * e3) * sqrt(1 - xd * xd)
    elseif lam < 1e-5
        (ey - yd * e3) * sqrt(1 - yd * yd)
    else
        wx = (lam - 1e-5) / (1e-4 - 1e-5)
        v = wx * (ex - xd * e3) * sqrt(1 - xd * xd) + (1 - wx) * (ey - yd * e3) * sqrt(1 - yd * yd)
        v / norm(v)
    end
    e2 = cross(e3, e1)
    return e1 / norm(e1), e2 / norm(e2), e3 / norm(e3)
end

# ---------------------------------------------------------------------------------------------------
# The co-precessing frame: Euler angles of LN_hat in the J frame along the PN table, minimal rotation
# ---------------------------------------------------------------------------------------------------

# LN_hat at omega in the J frame, and its derivative along the table variable x (dLN_J/dx).
@inline function _seobv5phm_lnj(tab, omega, e1, e2, e3)
    xq = _seobv5phm_pn_x(omega, tab.p)
    h = tab.x[2] - tab.x[1]
    y = clamp((xq - tab.x[1]) / h, zero(xq), oftype(xq, tab.n - 1))
    i = min(floor(Int, y), tab.n - 2) + 1
    s = y - (i - 1)
    zi, dzi, zj, dzj = tab.z[i], tab.dz[i], tab.z[i+1], tab.dz[i+1]
    v1, d1 = _eobimr_hermite(zi[1], dzi[1], zj[1], dzj[1], h, s)
    v2, d2 = _eobimr_hermite(zi[2], dzi[2], zj[2], dzj[2], h, s)
    v3, d3 = _eobimr_hermite(zi[3], dzi[3], zj[3], dzj[3], h, s)
    L = SVector(v1, v2, v3)
    dL = SVector(d1, d2, d3)
    nL = norm(L)
    Lh = L / nL
    dLh = (dL - dot(Lh, dL) * Lh) / nL
    return SVector(dot(Lh, e1), dot(Lh, e2), dot(Lh, e3)), SVector(dot(dLh, e1), dot(dLh, e2), dot(dLh, e3)), xq
end

# The minimal-rotation frame without Euler angles. With LN_J = (sin b cos a, sin b sin a, cos b) the minimal rotation is
# R = exp(a z/2) exp(b y/2) exp(g z/2), g' = -cos(b) a', i.e. R = qn(LN_J) exp(d z/2): qn the rotation taking z to LN_J
# about z x LN_J, and d = a + g with d' = (1 - cos b) a' = (n1 n2' - n2 n1') / (1 + n3). Unlike (a, b, g) this is smooth
# at b = 0 (spins along L: a = atan(n2, n1) and b = acos(n3) have singular derivatives there, which gave NaN / 1/eps
# gradients near aligned spins); it is singular at b = pi only (L_N antiparallel to J).
@inline _qz(t) = (cos(t / 2), zero(t), zero(t), sin(t / 2))
@inline function _qminrot(n)
    s = sqrt(2 * (1 + n[3]))
    return ((1 + n[3]) / s, -n[2] / s, n[1] / s, zero(s))
end
# dd/dx = (1 - cos(beta)) dalpha/dx for LN_J and its x-derivative
@inline _seobv5phm_ddelta(LJ, dLJ) = (LJ[1] * dLJ[2] - LJ[2] * dLJ[1]) / (1 + LJ[3])

"""
    _seobv5phm_delta_table!(G, tab, e1, e2, e3) -> G

`d(x_i) = int_{x_1}^{x_i} (1 - cos(beta)) dalpha` (= alpha + gamma of the minimal rotation, up to a constant) at the
PN table nodes (3-point Gauss-Legendre per interval on the Hermite interpolant). The absolute offset is irrelevant:
the frame is fixed at the reference time.
"""
function _seobv5phm_delta_table!(G, tab, e1, e2, e3)
    n = tab.n
    for i in 1:n-1
        G[i+1] = G[i] + _seobv5phm_delta_inc(tab, i, e1, e2, e3)
    end
    for i in n+1:length(G)
        G[i] = G[n]
    end
    return G
end
# The table's increment over interval i, G[i+1] - G[i] (a device reverse differentiates it per interval).
@inline function _seobv5phm_delta_inc(tab, i, e1, e2, e3)
    h = tab.x[2] - tab.x[1]
    xm = tab.x[i] + h / 2
    d = h * sqrt(0.6) / 2
    LJa, dLJa, _ = _seobv5phm_lnj(tab, _seobv5phm_pn_om(xm - d, tab.p), e1, e2, e3)
    LJb, dLJb, _ = _seobv5phm_lnj(tab, _seobv5phm_pn_om(xm, tab.p), e1, e2, e3)
    LJc, dLJc, _ = _seobv5phm_lnj(tab, _seobv5phm_pn_om(xm + d, tab.p), e1, e2, e3)
    acc = 5 / 9 * _seobv5phm_ddelta(LJa, dLJa) + 8 / 9 * _seobv5phm_ddelta(LJb, dLJb) + 5 / 9 * _seobv5phm_ddelta(LJc, dLJc)
    return acc * h / 2
end

# d at omega: Hermite on the d table with the exact node derivatives.
@inline function _seobv5phm_delta_at(tab, G, omega, e1, e2, e3)
    xq = _seobv5phm_pn_x(omega, tab.p)
    h = tab.x[2] - tab.x[1]
    y = clamp((xq - tab.x[1]) / h, zero(xq), oftype(xq, tab.n - 1))
    i = min(floor(Int, y), tab.n - 2) + 1
    s = y - (i - 1)
    LJi, dLJi, _ = _seobv5phm_lnj(tab, _seobv5phm_pn_om(tab.x[i], tab.p), e1, e2, e3)
    LJj, dLJj, _ = _seobv5phm_lnj(tab, _seobv5phm_pn_om(tab.x[i+1], tab.p), e1, e2, e3)
    g, _ = _eobimr_hermite(G[i], _seobv5phm_ddelta(LJi, dLJi), G[i+1], _seobv5phm_ddelta(LJj, dLJj), h, s)
    return g
end

# Minimal-rotation J -> P quaternion (up to a constant z rotation on the right) at orbital frequency omega.
@inline function _seobv5phm_qmin(tab, G, omega, e1, e2, e3)
    LJ, _, _ = _seobv5phm_lnj(tab, omega, e1, e2, e3)
    return _qmul(_qminrot(LJ), _qz(_seobv5phm_delta_at(tab, G, omega, e1, e2, e3)))
end

# ---------------------------------------------------------------------------------------------------
# Build: attachment, modes, NQC, merger-ringdown, frames
# ---------------------------------------------------------------------------------------------------

@inline function _seobv5phm_mix_k(k, kmm, nu, att, Mf, af, mrf, rot)
    l, m = _SEOBV5HM_L[k], _SEOBV5HM_M[k]
    h, hd_, ph, om = att[k]
    hmm, hdmm, phmm, ommm = att[kmm]
    A0, ph0, dA0, om0 = _seobv5hm_spheroidal_iv(l, m, af, h, hmm, hd_, hdmm, om, ommm, ph, phmm)
    sph = _seobv5hm_mr(nu, k, A0, dA0, ph0, om0, Mf, af, _seobv5hm_fits_k(k, mrf), rot)
    return (sph = sph, mu_mm = _seobv5hm_mu(m, m, m, af), mu_lm = _seobv5hm_mu(m, l, m, af),
        mu_ll = _seobv5hm_mu(m, l, l, af), zero = !(h >= 1e-8))
end

"""
    _seobv5phm_build(m1, m2, chi1, chi2, omega_ref, omega_start; dynamics kwargs...) -> NamedTuple

Dynamics, attachment, the co-precessing modes' node tables (`D`), special coefficients, NQC, remnant,
merger-ringdown (with the co-precessing QNM shift) and the frame data. Evaluate with
[`_seobv5phm_coprec`](@ref) and [`_seobv5phm_hpc`](@ref).

Frequencies as pyseobnr's `SEOBNRv5PHM_opt.__init__`: `omega_start` is lowered to `10.5^(-3/2)` (Newtonian
`r0 >= 10.5 M`); a reference frequency below the start becomes the start; otherwise, if it differs from the start,
the reference time `t_ref` is where the orbital frequency reaches `omega_ref`: the orbital phase is shifted to 0
there and the co-precessing frame equals the inertial one there (else both at the start of the dynamics).
"""
function _seobv5phm_build(m1, m2, chi1, chi2, omega_ref, omega_start; antisym::Bool = true, kwargs...)
    om_s = omega_start^(-2 / 3) < 10.5 ? 10.5^(-1.5) : omega_start
    use_tref = abs(omega_ref - omega_start) > 1e-12 * omega_start
    if use_tref && omega_ref < omega_start
        om_s = omega_ref
        use_tref = false
    end
    d = seobnrv5phm_dynamics(m1, m2, chi1, chi2, omega_ref, om_s; kwargs...)
    ND = length(d.t)
    D = (A = zeros(ND, 6), dA = zeros(ND, 6), th = zeros(ND, 6), dth = zeros(ND, 6))
    G = zeros(typeof(d.b.tab.omega_end), size(d.b.tab.z.A, 2))
    return _seobv5phm_build_dyn!(D, G, d, d.b, use_tref, omega_ref, chi1[3], chi2[3]; antisym = antisym)
end

"""
    _seobv5phm_build_dyn!(D, G, d, b, use_tref, omega_ref, chi1z, chi2z; antisym = true) -> NamedTuple

Everything of [`_seobv5phm_build`](@ref) after the dynamics `d` (fields of [`seobnrv5phm_dynamics`](@ref)), with
caller-owned buffers: `D` (`A, dA, th, dth`, node x 6 mode tables) and `G` (the gamma table, length of the PN table);
`chi1z`, `chi2z` the input spins' z components at `omega_ref` (pyseobnr's `chi_A` of the (4,3) special-coefficient
threshold). The returned NamedTuple holds `d`, `D`, `G` themselves (e.g. a device walker's views).
"""
function _seobv5phm_build_dyn!(D, G, d, b, use_tref, omega_ref, chi1z, chi2z; antisym::Bool = true)
    use_tref && _seobv5phm_phase_shift!(d, omega_ref)
    R = _seobv5phm_build_pre(d, b, chi1z, chi2z)
    for i in 1:d.n
        _seobv5phm_node_modes!(D, d, b, R, i)
    end
    for i in 1:d.n
        _seobv5phm_node_derivs!(D, d, i)
    end
    S = _seobv5phm_build_post(D, G, d, b, R, use_tref, omega_ref, antisym)
    return merge(S, (t_peak = _seobv5phm_tpeak_newton(S, _seobv5phm_tpeak_search(S)),))
end

# The build in pieces with clean reverse-mode boundaries (the device gradient differentiates each on its own):
# phase shift, `_seobv5phm_build_pre` (attachment, special coefficients), node modes and their node derivatives
# (node-parallel), `_seobv5phm_build_post` (NQC, remnant, frames, gamma table, merger-ringdown, anti-symmetric fits),
# and t_peak (a search with no derivative + one Newton step).

# Phase origin at omega_ref (`use_tref`): phi -= phi(t_ref) in place.
function _seobv5phm_phase_shift!(d, omega_ref)
    ommax = d.dphi[1]
    for i in 2:d.n
        ommax = max(ommax, d.dphi[i])
    end
    omega_ref > ommax && throw(DomainError(omega_ref, "f_ref is larger than the highest frequency in the inspiral"))
    ph_ref = _seobv5phm_ref_phase(d, omega_ref)
    @inbounds for i in 1:d.n
        d.phi[i] -= ph_ref
    end
    return nothing
end
@inline _seobv5phm_ref_phase(d, omega_ref) = _seobv5phm_dyn_at(d, _seobv5phm_t_at_omega(d, omega_ref))[2]

# Attachment (Step 2 of _evaluate_model) and the special coefficients: an isbits record.
function _seobv5phm_build_pre(d, b, chi1z, chi2z)
    A = _seobv5phm_pre_att(d, b)
    sts = _seobv5phm_attach_states(d, b, A.t_a)
    c21, c43 = _seobv5phm_pre_spec(b, A, sts, chi1z, chi2z)
    return (t_r10 = A.t_r10, r_isco = A.r_isco, t_isco = A.t_isco, dT = A.dT, t_a = A.t_a, om_a = A.om_a, Lvec = A.Lvec,
        Jf = A.Jf, chi1a = A.chi1a, chi2a = A.chi2a, pref = A.pref, sts = sts, c21 = c21, c43 = c43, chi1LN10 = A.chi1LN10,
        chi2LN10 = A.chi2LN10, c1_10 = A.c1_10, c2_10 = A.c2_10, LNhat10 = A.LNhat10)
end
# (the pieces of the attachment record, reversed separately on a device: the attachment times and spins; the states
# around t_attach (`_seobv5phm_attach_states`); the special coefficients)
# dOmega/dt at node k: central difference of dH/dpphi along the node's rates (+-1e-3 M; spins as the node's)
@inline function _seobv5phm_omdot(d, k)
    b = d.b
    s = _seobv5phm_spins(b, d.omega_s[k])
    e = 1e-3
    r, pr, pp, dr, dpr, dpp = d.r[k], d.pr[k], d.pphi[k], d.dr[k], d.dpr[k], d.dpphi[k]
    op = _seobv5phm_ham_omega(_seobv5phm_hargs(r + e * dr, pr + e * dpr, pp + e * dpp, s, b)...)
    om = _seobv5phm_ham_omega(_seobv5phm_hargs(r - e * dr, pr - e * dpr, pp - e * dpp, s, b)...)
    return (op - om) / (2e)
end
# The maximum of the cubic Hermite of Omega (values and derivatives w0, w1 at the ends) on [t0, t0 + h], w0 > 0 > w1:
# bisection on the sign of its derivative, then a Newton step (implicit-function derivatives for AD).
@inline function _seobv5phm_hermite_peak(t0, h, y0, w0, y1, w1)
    a2 = 6y0 + 3h * w0 - 6y1 + 3h * w1
    a1 = -6y0 - 4h * w0 + 6y1 - 2h * w1
    a0 = h * w0                      # h dOmega/dt = a2 s^2 + a1 s + a0
    lo, hi = zero(h), one(h)
    for _ in 1:40
        mid = (lo + hi) / 2
        pos = (a2 * mid + a1) * mid + a0 > 0
        lo, hi = ifelse(pos, mid, lo), ifelse(pos, hi, mid)
    end
    sm = (lo + hi) / 2
    sm -= ((a2 * sm + a1) * sm + a0) / (2a2 * sm + a1)
    return t0 + h * sm
end
# The time of the plunge's stopping event (`d.stop`, see `_seobv5phm_plunge!`), interpolated between the last nodes:
# the Omega peak (maximum of Omega's cubic Hermite, dOmega/dt at the nodes from `_seobv5phm_omdot`; the peak can lie
# before node n - 1), r = 1.4 (Hermite root), the zero of dr/dt,
# dpr/dt or omega_circ - 1, Omega = omega_PN_end; the last node for the others (non-finite state, step budget, spin
# guard). pyseobnr uses the last node, the step after the event: where t_ISCO or t_attach falls past the end (r_ISCO
# below the last node's r; DeltaT_NR < 0 at high q), that moved them by up to a step with the step rule.
@inline _seobv5phm_tzero(t1, t2, y1, y2) = ifelse(y1 == y2, t2, t1 + (t2 - t1) * y1 / (y1 - y2))
@inline function _seobv5phm_t_stop(d)
    n = d.n
    t = d.t
    st = d.stop
    n < 3 && return t[n]
    te = if st == 3
        wm = _seobv5phm_omdot(d, n - 1)      # > 0: the peak is in (t[n-1], t[n]), else in (t[n-2], t[n-1])
        k = ifelse(wm > 0, n - 1, n - 2)
        w0 = ifelse(wm > 0, wm, _seobv5phm_omdot(d, n - 2))
        w1 = ifelse(wm > 0, _seobv5phm_omdot(d, n), wm)
        ifelse((w0 > 0) & (w1 < 0), _seobv5phm_hermite_peak(t[k], t[k+1] - t[k], d.dphi[k], w0, d.dphi[k+1], w1), t[n])
    elseif st == 1
        h = t[n] - t[n-1]
        lo, hi = zero(h), one(h)
        for _ in 1:40
            mid = (lo + hi) / 2
            v, _ = _eobimr_hermite(d.r[n-1], d.dr[n-1], d.r[n], d.dr[n], h, mid)
            lo, hi = ifelse(v > 1.4, mid, lo), ifelse(v > 1.4, hi, mid)
        end
        sm = (lo + hi) / 2
        v, dv = _eobimr_hermite(d.r[n-1], d.dr[n-1], d.r[n], d.dr[n], h, sm)
        t[n-1] + sm * h - (v - 1.4) / dv
    elseif st == 5 || st == 8
        _seobv5phm_tzero(t[n-1], t[n], d.dr[n-1], d.dr[n])
    elseif st == 6
        _seobv5phm_tzero(t[n-1], t[n], d.dpr[n-1], d.dpr[n])
    elseif st == 7
        _seobv5phm_tzero(t[n-1], t[n], d.omega_circ[n-1] - 1, d.omega_circ[n] - 1)
    elseif st == 9
        _seobv5phm_tzero(t[n-1], t[n], d.dphi[n-1] - d.b.tab.omega_end, d.dphi[n] - d.b.tab.omega_end)
    else
        t[n]
    end
    return clamp(te, ifelse(st == 3, t[n-2], t[n-1]), t[n])
end
const _SEOBV5PHM_PRE_ATT = (:t_r10, :r_isco, :t_isco, :dT, :t_a, :om_a, :Lvec, :Jf, :chi1a, :chi2a, :pref, :chi1LN10,
    :chi2LN10, :c1_10, :c2_10, :LNhat10)
@inline function _seobv5phm_pre_att(d, b)
    tab = b.tab
    nu, X1, X2 = b.nu, b.X1, b.X2
    n = d.n
    t_r10 = _seobv5phm_crossing(d, d.r, d.dr, 10.0, -1)
    om_r10 = _seobv5phm_dyn_at(d, t_r10)[5]
    s10 = _seobv5phm_spins(b, om_r10)
    r_isco, _ = _seobv5_kerr_isco(s10.chi1LN, s10.chi2LN, X1, X2)
    t_end = _seobv5phm_t_stop(d)
    t_isco = r_isco < d.r[n] ? t_end : _seobv5phm_crossing(d, d.r, d.dr, r_isco, -1)
    s_isco = _seobv5phm_spins(b, _seobv5phm_dyn_at(d, t_isco)[5])
    dT = _eobimr_nr_deltaT(nu, s_isco.ap, s_isco.am)
    t_a = min(t_isco - dT, t_end)
    om_a = _seobv5phm_dyn_at(d, t_a)[5]
    s_a = _seobv5phm_spins(b, om_a)
    e_a = seobnrv5phm_pn_everything(tab, om_a)
    Lvec = SVector(e_a[14], e_a[15], e_a[16])
    Jf = s_a.c1 * X1^2 + s_a.c2 * X2^2 + Lvec
    return (t_r10 = t_r10, r_isco = r_isco, t_isco = t_isco, dT = dT, t_a = t_a, om_a = om_a, Lvec = Lvec, Jf = Jf,
        chi1a = s_a.chi1LN, chi2a = s_a.chi2LN, pref = _seobv5hm_tup6(_seobv5hm_pref_k, X1, X2), chi1LN10 = s10.chi1LN,
        chi2LN10 = s10.chi2LN, c1_10 = s10.c1, c2_10 = s10.c2, LNhat10 = s10.LNhat)
end
# The special coefficients c21, c43 at the attachment (from the middle attachment state).
@inline function _seobv5phm_pre_spec(b, A, sts, chi1z, chi2z)
    nu, X1, X2 = b.nu, b.X1, b.X2
    s_a = _seobv5phm_spins(b, A.om_a)
    ba = _seobv5phm_modeb(b, s_a)
    r3, _, pr3, pp3, om3 = sts[3]
    Ha, Heffa, omca = _seobv5phm_hstate(r3, pr3, pp3, s_a, b)
    fv = _seobv5hm_fitvars(X1, X2, A.chi1a, A.chi2a)
    hs = _seobv5hm_iv_hsign(fv)
    rf = _seobv5_rho_f(cbrt(om3), log(cbrt(om3)), nu * Ha * om3, ba.fc.rho)
    chiA_in = (chi1z - chi2z) / 2                   # the z components at omega_ref (pyseobnr chi_A)
    skip = (abs(nu - 0.25) < 1e-14) & (abs(chiA_in) < 1e-14)
    pref = A.pref
    c21 = skip ? 0.0 : _seobv5hm_special_k(2, om3, omca, Ha, Heffa, pp3, ba, pref[2], rf, hs.h21, hs.h22, 300.0)
    c43 = skip ? 0.0 : _seobv5hm_special_k(6, om3, omca, Ha, Heffa, pp3, ba, pref[6], rf, hs.h43, hs.h22,
        200.0 * nu * (1.0 - 0.8 * chiA_in))
    return c21, c43
end

# Signed amplitude and smooth phase of the six reduced modes at dynamics node i (spins at the node omega_s).
@inline function _seobv5phm_node_modes!(D, d, b, R, i)
    s = _seobv5phm_spins(b, d.omega_s[i])
    bm = _seobv5phm_modeb(b, s)
    H, Heff, _ = _seobv5phm_hstate(d.r[i], d.pr[i], d.pphi[i], s, b)
    # (the six modes unrolled: a loop here is a device-malloc'd cache in Enzyme's in-kernel reverse on GPUs)
    Base.Cartesian.@nexprs 6 k -> begin
        D.A[i, k], D.th[i, k] = _seobv5hm_mode_ap(k, d.dphi[i], d.omega_circ[i], H, Heff, d.pphi[i], bm, R.pref[k], R.c21,
            R.c43)
    end
    return nothing
end
# Their time derivatives at node i (3-point finite differences on the node times).
@inline function _seobv5phm_node_derivs!(D, d, i)
    Base.Cartesian.@nexprs 6 k -> begin              # (unrolled, as `_seobv5phm_node_modes!`)
        D.dA[i, k] = _seobv5hm_node_deriv(d.t, view(D.A, :, k), i, d.n)
        D.dth[i, k] = _seobv5hm_node_deriv(d.t, view(D.th, :, k), i, d.n)
    end
    return nothing
end

# NQC, remnant (spins at r = 10 M), frames + gamma table (into G), merger-ringdown, anti-symmetric fits.
@inline function _seobv5phm_build_post(D, G, d, b, R, use_tref, omega_ref, antisym::Bool, asym_flags = nothing)   # (inlined: Enzyme cannot type the returned record of views on GPUs)
    rec0 = _seobv5phm_post_rest!(G, d, b, R, _seobv5phm_post_nqc(b, R), use_tref, omega_ref)
    S = merge((dyn = d, D = D, b = b, G = G), rec0)
    asym = asym_flags === nothing ? _seobv5phm_asym_build(S, R.sts) : _seobv5phm_asym_build(S, R.sts, asym_flags)
    return merge(S, (asym = asym, use_asym = antisym))   # (always built: a type-stable record)
end

# The fit variables and input values of the attachment (spins along LN_hat at t_attach).
@inline function _seobv5phm_post_fits(b, R)
    fv = _seobv5hm_fitvars(b.X1, b.X2, R.chi1a, R.chi2a)
    ivs = (_seobv5hm_iv_habs(fv), _seobv5hm_iv_hdot(fv), _seobv5hm_iv_hdotdot(fv), _seobv5hm_iv_omega(fv),
        _seobv5hm_iv_omegadot(fv))
    return fv, ivs
end

# NQC (Steps 3-6): the six reduced modes at the five attachment states and their NQC coefficients (isbits).
@inline function _seobv5phm_post_nqc(b, R)
    pref, c21, c43, sts = R.pref, R.c21, R.c43, R.sts
    chi1a, chi2a = R.chi1a, R.chi2a
    _, ivs = _seobv5phm_post_fits(b, R)
    Hs1 = map(st -> _seobv5phm_state_modes(st, b, pref, c21, c43), sts)
    Hs = ntuple(k -> ntuple(i -> Hs1[i][k], 5), 6)
    skip_odd = (b.delta < 1e-4) & (((abs(chi1a) < 1e-4) & (abs(chi2a) < 1e-4)) | (abs(chi1a - chi2a) < 1e-4))
    nqc = _seobv5hm_tup6(_seobv5hm_nqc_k, sts, Hs, b.nu, ivs, skip_odd)
    return (Hs = Hs, nqc = nqc)
end

# Remnant (spins at r = 10 M), frames + gamma table (into G), merger-ringdown with the co-precessing QNM shift:
# every record field of the build except the anti-symmetric part and t_peak (isbits, no views).
function _seobv5phm_post_rest!(G, d, b, R, Q, use_tref, omega_ref)
    _seobv5phm_delta_table!(G, b.tab, _seobv5phm_jframe(R.Jf / norm(R.Jf))...)
    return _seobv5phm_post_rest(G, d, b, R, Q, use_tref, omega_ref)
end
# The same with the gamma table `G` already filled (read only: no loop over the table, see `_eobimr_searchsortedlast`).
@inline function _seobv5phm_post_rest(G, d, b, R, Q, use_tref, omega_ref)
    tab = b.tab
    nu, X1, X2 = b.nu, b.X1, b.X2
    fv, ivs = _seobv5phm_post_fits(b, R)
    sts, Hs, nqc = R.sts, Q.Hs, Q.nqc

    # --- remnant (spins at r = 10 M)
    Mf = _eobimr_final_mass_uib2016(X1, X2, R.chi1LN10, R.chi2LN10)
    af, sgn_af = _seobv5phm_final_spin(X1, X2, R.chi1LN10, R.chi2LN10, R.c1_10, R.c2_10, R.LNhat10)

    # --- frames (Step 8.9)
    Jf = R.Jf
    e1, e2, e3 = _seobv5phm_jframe(Jf / norm(Jf))
    qI2J = _qfrom_rotmat(e1, e2, e3)
    qJ2I = _qconj(qI2J)
    # reference time (see the docstring): J2P(t_ref) = J2I
    qref = _seobv5phm_qmin(tab, G, use_tref ? omega_ref : d.dphi[1], e1, e2, e3)
    C = _qmul(_qconj(qref), qJ2I)
    qJ2P_a = _qmul(_seobv5phm_qmin(tab, G, R.om_a, e1, e2, e3), C)
    cb_a = qJ2P_a[1]^2 + qJ2P_a[4]^2 - qJ2P_a[2]^2 - qJ2P_a[3]^2      # cos(beta) of the attachment frame

    # --- merger-ringdown with the co-precessing QNM shift
    w220 = _seobv5hm_qnm(1, af)[1] / Mf
    w210 = _seobv5hm_qnm(2, af)[1] / Mf
    precRate = (w220 - w210) * sgn_af
    rot = (1 - abs(cb_a)) * precRate
    mrf = (_seobv5hm_mr_c1f(fv), _seobv5hm_mr_c2f(fv), _seobv5hm_mr_d1f(fv), _seobv5hm_mr_d2f(fv))
    att = _seobv5hm_tup6(_seobv5hm_att_k, sts, Hs, nqc, ivs[4])
    mr = ntuple(k -> _seobv5hm_mr(nu, k, att[k][1], att[k][2], att[k][3], att[k][4], Mf, af, _seobv5hm_fits_k(k, mrf), rot), 6)
    mix = (_seobv5phm_mix_k(4, 1, nu, att, Mf, af, mrf, rot), _seobv5phm_mix_k(6, 3, nu, att, Mf, af, mrf, rot))

    Lvec = R.Lvec
    return (t_a = R.t_a, t_isco = R.t_isco, t_r10 = R.t_r10, r_isco = R.r_isco, dT = R.dT, om_a = R.om_a, pref = R.pref,
        c21 = R.c21, c43 = R.c43, nqc = nqc, mr = mr, mix = mix, Mf = Mf, af = af, sgn_af = sgn_af, att = att, e1 = e1,
        e2 = e2, e3 = e3, qI2J = qI2J, C = C, frame_a = (q = qJ2P_a, cb = cb_a), precRate = precRate, rot = rot,
        chi1LN_a = R.chi1a, chi2LN_a = R.chi2a, Jhat = Jf / norm(Jf), Lhat_a = Lvec / norm(Lvec))
end

# Frame-invariant amplitude sqrt(sum_lm |h_lm|^2) over the co-precessing modes (both signs of m, anti-symmetric parts
# included), pyseobnr `frame_inv_amp`.
@inline function _seobv5phm_inv_amp(S, tq)
    ha3 = S.use_asym ? _seobv5phm_asym3(S, tq) : (zero(ComplexF64), zero(ComplexF64), zero(ComplexF64))
    # (unrolled as in `_seobv5phm_hpc`)
    acc = _seobv5phm_inv_amp_k(S, Val(1), tq, ha3) + _seobv5phm_inv_amp_k(S, Val(2), tq, ha3) +
          _seobv5phm_inv_amp_k(S, Val(3), tq, ha3) + _seobv5phm_inv_amp_k(S, Val(4), tq, ha3) +
          _seobv5phm_inv_amp_k(S, Val(5), tq, ha3) + _seobv5phm_inv_amp_k(S, Val(6), tq, ha3)
    return sqrt(acc)
end
@inline function _seobv5phm_inv_amp_k(S, ::Val{k}, tq, ha3) where {k}
    h = _seobv5phm_coprec(S, k, tq)
    hn = (-1)^_SEOBV5HM_L[k] * conj(h)
    if S.use_asym && isodd(k)
        ha = ha3[(k + 1) >> 1]
        h += ha
        hn -= (-1)^_SEOBV5HM_L[k] * conj(ha)
    end
    return abs2(h) + abs2(hn)
end

"""
    _seobv5phm_tpeak_search(S) -> t

Time of the frame-invariant amplitude's maximum (pyseobnr's `t = 0`, `estimate_time_max_amplitude` of
`frame_inv_amp`): a 0.1 M scan of `t_attach +- 50 M`, then golden-section refinement. No derivative: the model's
`t_peak` is `_seobv5phm_tpeak_newton(S, _seobv5phm_tpeak_search(S))`.
"""
function _seobv5phm_tpeak_search(S)
    tb, ab = S.t_a, -Inf
    for j in 0:1000
        t = S.t_a - 50 + 0.1 * j
        a = _seobv5phm_inv_amp(S, t)
        if a > ab
            tb, ab = t, a
        end
    end
    lo, hi = tb - 0.1, tb + 0.1
    g = (sqrt(5) - 1) / 2
    for _ in 1:40
        x1 = hi - g * (hi - lo)
        x2 = lo + g * (hi - lo)
        if _seobv5phm_inv_amp(S, x1) > _seobv5phm_inv_amp(S, x2)
            hi = x2
        else
            lo = x1
        end
    end
    return (lo + hi) / 2
end

# One Newton step on dA/dt from the search's tm (central differences, step 1e-3 M): leaves the maximum where it is
# (to ~1e-7 M) and gives automatic differentiation the implicit-function derivative of the argmax.
@inline function _seobv5phm_tpeak_newton(S, tm)
    dl = 1e-3
    am, a0, ap = _seobv5phm_inv_amp(S, tm - dl), _seobv5phm_inv_amp(S, tm), _seobv5phm_inv_amp(S, tm + dl)
    return tm - dl * (ap - am) / (2 * (ap - 2a0 + am))
end

# Co-precessing mode k (1..6, m > 0; pyseobnr convention) at dynamics time tq.
@inline _seobv5phm_coprec(S, k, tq) = _seobv5hm_full(S, k, tq)

# Simple precession about J after the attachment, by x = precRate (t - t_a): the attachment frame's Euler angles
# (a + x, b, g - cos(b) x), written without the angles (singular at b = 0) as exp(x z/2) q_a exp(-cos(b) x z/2).
# (one record field `frame_a = (q, cb)`: the record has 31 fields, and `map` over a tuple of 32 or more falls back to
# dynamic code that does not compile for GPUs -- GravitationalWaveBayes maps over the record's fields)
@inline _seobv5phm_qprec(S, x) = _qmul(_qz(x), _qmul(S.frame_a.q, _qz(-S.frame_a.cb * x)))
# J -> P quaternion at dynamics time tq: minimal rotation along LN_hat before t_attach, simple precession about J after.
@inline function _seobv5phm_qJ2P(S, tq)
    if tq < S.t_a
        om = _seobv5phm_dyn_at(S.dyn, tq)[5]
        return _qmul(_seobv5phm_qmin(S.b.tab, S.G, om, S.e1, S.e2, S.e3), S.C)
    end
    x = (tq - S.t_a) * S.precRate
    return _seobv5phm_qprec(S, x)
end

# -2Y_lm(beta, alpha) of pyseobnr's custom_swsh (alpha enters as e^{i m alpha}), (l, m) and (l, -m). The Euler-angle form of
# `_seobv5phm_swsh_q` (which the model uses); kept as the reference the tests compare it with.
@inline function _seobv5phm_swsh(k, be, al)
    cB, sB = cos(be / 2), sin(be / 2)
    l, m = _SEOBV5HM_L[k], _SEOBV5HM_M[k]
    e = complex(cos(m * al), sin(m * al))
    if k == 1
        p = 0.5 * sqrt(5 / pi) * cB^4; q = 0.5 * sqrt(5 / pi) * sB^4
    elseif k == 2
        p = sqrt(5 / pi) * cB^3 * sB; q = sqrt(5 / pi) * sB^3 * cB
    elseif k == 3
        p = -sqrt(10.5 / pi) * cB^5 * sB; q = sqrt(10.5 / pi) * sB^5 * cB
    elseif k == 4
        p = 0.25 * sqrt(7 / pi) * cB^4 * (6 * (cB^2 - sB^2) - 4); q = 0.25 * sqrt(7 / pi) * sB^4 * (6 * (cB^2 - sB^2) + 4)
    elseif k == 5
        p = 3 * sqrt(7 / pi) * cB^6 * sB^2; q = 3 * sqrt(7 / pi) * sB^6 * cB^2
    else
        p = -0.75 * sqrt(3.5 / pi) * cB^5 * sB * (8 * (cB^2 - sB^2) - 4); q = 0.75 * sqrt(3.5 / pi) * sB^5 * cB * (8 * (cB^2 - sB^2) + 4)
    end
    return p * e, q * conj(e)
end

# The observer rotation without Euler angles: with the Cayley-Klein parameters of q = exp(a z/2) exp(b y/2) exp(g z/2),
# Ra = w + i z = cos(b/2) e^{i(a+g)/2}, Rb = y - i x = sin(b/2) e^{i(a-g)/2}, each term -2Y_lm(b, a) e^{2ig} (and its
# m -> -m partner) of the sum below is a monomial in Ra, Rb and their conjugates (times a polynomial in
# cos b = |Ra|^2 - |Rb|^2): smooth for every rotation, unlike (a, b, g) (singular derivatives at b = 0, pi: the line
# of sight along the co-precessing z axis, face-on / face-off). Both have even degree, so q and -q agree.
@inline _qcayley(q) = (complex(q[1], q[4]), complex(q[3], -q[2]))
@inline function _seobv5phm_swsh_q(k, Ra, Rb)
    # (powers as explicit products: `Complex ^ Int` is a power-by-squaring loop, which an in-kernel Enzyme reverse
    # tapes on the device heap -- 100x slower sample reverse)
    cRa, cRb = conj(Ra), conj(Rb)
    cb = abs2(Ra) - abs2(Rb)
    Ra2 = Ra * Ra; Ra4 = Ra2 * Ra2
    cRb2 = cRb * cRb; cRb4 = cRb2 * cRb2
    if k == 1          # (2, 2)
        c = 0.5 * sqrt(5 / pi)
        return c * Ra4, c * cRb4
    elseif k == 2      # (2, 1)
        c = sqrt(5 / pi)
        return c * (Ra2 * Ra) * cRb, c * Ra * (cRb2 * cRb)
    elseif k == 3      # (3, 3)
        c = sqrt(10.5 / pi)
        return -c * (Ra4 * Ra) * Rb, c * cRa * (cRb4 * cRb)
    elseif k == 4      # (3, 2)
        c = 0.25 * sqrt(7 / pi)
        return c * Ra4 * (6 * cb - 4), c * cRb4 * (6 * cb + 4)
    elseif k == 5      # (4, 4)
        c = 3 * sqrt(7 / pi)
        return c * (Ra4 * Ra2) * (Rb * Rb), c * (cRa * cRa) * (cRb4 * cRb2)
    else               # (4, 3)
        c = 0.75 * sqrt(3.5 / pi)
        return -c * (Ra4 * Ra) * Rb * (8 * cb - 4), c * cRa * (cRb4 * cRb) * (8 * cb + 4)
    end
end

"""
    _seobv5phm_hpc(S, tq, inclination, phiref) -> ComplexF64

`h+ - i hx` (geometric units, pyseobnr `compute_hpc_from_coprec_modes`; `phiref` is the model's setting, which
`GenerateWaveform` sets to `pi/2 - phi_ref`).
"""
@inline function _seobv5phm_hpc(S, tq, inclination, phiref)
    qJ2P = _seobv5phm_qJ2P(S, tq)
    qtot = _qmul(_qconj(qJ2P), _qmul(_qconj(S.qI2J), _qeuler(phiref, inclination, 0.0)))
    Ra, Rb = _qcayley(qtot)
    ha3 = S.use_asym ? _seobv5phm_asym3(S, tq) : (zero(ComplexF64), zero(ComplexF64), zero(ComplexF64))
    # (the six modes unrolled: no loop for Enzyme's device reverse to tape, see `_eobimr_searchsortedlast`)
    return _seobv5phm_hpc_k(S, Val(1), tq, ha3, Ra, Rb) + _seobv5phm_hpc_k(S, Val(2), tq, ha3, Ra, Rb) +
           _seobv5phm_hpc_k(S, Val(3), tq, ha3, Ra, Rb) + _seobv5phm_hpc_k(S, Val(4), tq, ha3, Ra, Rb) +
           _seobv5phm_hpc_k(S, Val(5), tq, ha3, Ra, Rb) + _seobv5phm_hpc_k(S, Val(6), tq, ha3, Ra, Rb)
end
# mode k's term (both signs of m) of the sum above
@inline _seobv5phm_hpc_k(S, ::Val{k}, tq, ha3, Ra, Rb) where {k} =
    _seobv5phm_hpc_kh(Val(k), _seobv5phm_coprec(S, k, tq), S.use_asym, ha3, Ra, Rb)
# ... given the co-precessing mode h = h_lm (m > 0)
@inline function _seobv5phm_hpc_kh(::Val{k}, h, use_asym, ha3, Ra, Rb) where {k}
    sl = (-1)^_SEOBV5HM_L[k]
    hn = sl * conj(h)
    if use_asym && isodd(k) && k <= 5       # (2,2), (3,3), (4,4): + anti-symmetric part
        ha = ha3[(k + 1) >> 1]
        h += ha
        hn -= sl * conj(ha)
    end
    yp, ym = _seobv5phm_swsh_q(k, Ra, Rb)
    return yp * h + ym * hn
end

# `_seobv5phm_hpc` split at t_attach, for a device reverse that differentiates each part with only the record fields
# it reads (same values; at tq == t_a exactly the inspiral part takes the J -> P rotation from the inspiral branch, the
# same rotation to rounding):
# - `_seobv5phm_hpc_insp` (tq <= t_a): the dynamics, mode, PN and gamma tables, nqc, C, qI2J, e1..e3 and the
#   anti-symmetric flags, b6, b1;
# - `_seobv5phm_hpc_mr` (tq > t_a): t_a, mr, mix, frame_a, precRate, qI2J and the anti-symmetric mr.
# (the dynamics interpolation, the J -> P rotation and the NQC basis evaluated once and shared by the frame, the six
# modes and the anti-symmetric part: the per-call helpers' values to rounding)
@inline function _seobv5phm_hpc_insp(S, tq, inclination, phiref)
    d, D, nqc = S.dyn, S.D, S.nqc
    n = d.n
    tc = clamp(tq, d.t[1], d.t[n])
    i = clamp(_eobimr_searchsortedlast(d.t, tc), 1, n - 1)
    h = d.t[i+1] - d.t[i]
    s = (tc - d.t[i]) / h
    r, _ = _eobimr_hermite(d.r[i], d.dr[i], d.r[i+1], d.dr[i+1], h, s)
    ph, om = _eobimr_hermite(d.phi[i], d.dphi[i], d.phi[i+1], d.dphi[i+1], h, s)
    pr, _ = _eobimr_hermite(d.pr[i], d.dpr[i], d.pr[i+1], d.dpr[i+1], h, s)
    qJ2P = _qmul(_seobv5phm_qmin(S.b.tab, S.G, om, S.e1, S.e2, S.e3), S.C)
    qtot = _qmul(_qconj(qJ2P), _qmul(_qconj(S.qI2J), _qeuler(phiref, inclination, 0.0)))
    Ra, Rb = _qcayley(qtot)
    A = S.asym
    a = _seobv5phm_asym_dyn_from(S, r, ph, pr, om, qJ2P)
    ha3 = (_seobv5phm_asym_insp_a(S, A, 1, a), _seobv5phm_asym_insp_a(S, A, 2, a), _seobv5phm_asym_insp_a(S, A, 3, a))
    qn = _eobimr_nqc_q(r, pr, om)
    pn = _eobimr_nqc_p(r, pr, om)
    return _seobv5phm_hpc_kh(Val(1), _seobv5phm_insp_k(D, nqc, 1, i, h, s, ph, qn, pn), true, ha3, Ra, Rb) +
          _seobv5phm_hpc_kh(Val(2), _seobv5phm_insp_k(D, nqc, 2, i, h, s, ph, qn, pn), true, ha3, Ra, Rb) +
          _seobv5phm_hpc_kh(Val(3), _seobv5phm_insp_k(D, nqc, 3, i, h, s, ph, qn, pn), true, ha3, Ra, Rb) +
          _seobv5phm_hpc_kh(Val(4), _seobv5phm_insp_k(D, nqc, 4, i, h, s, ph, qn, pn), true, ha3, Ra, Rb) +
          _seobv5phm_hpc_kh(Val(5), _seobv5phm_insp_k(D, nqc, 5, i, h, s, ph, qn, pn), true, ha3, Ra, Rb) +
          _seobv5phm_hpc_kh(Val(6), _seobv5phm_insp_k(D, nqc, 6, i, h, s, ph, qn, pn), true, ha3, Ra, Rb)
end
# `_seobv5hm_insp` of mode k given the interval (i, h, s), the orbital phase and the NQC basis (qn, pn)
@inline function _seobv5phm_insp_k(D, nqc, k, i, h, s, ph, qn, pn)
    A, _ = _eobimr_hermite(D.A[i, k], D.dA[i, k], D.A[i+1, k], D.dA[i+1, k], h, s)
    th, _ = _eobimr_hermite(D.th[i, k], D.dth[i, k], D.th[i+1, k], D.dth[i+1, k], h, s)
    a1, a2, a3, b1, b2 = nqc[k]
    Ac = A * (1.0 + qn[1] * a1 + qn[2] * a2 + qn[3] * a3)
    vp = _SEOBV5HM_M[k] * ph - th - (b1 * pn[1] + b2 * pn[2])
    return Ac * complex(cos(vp), -sin(vp))
end
@inline function _seobv5phm_hpc_mr(S, tq, inclination, phiref)
    x = (tq - S.t_a) * S.precRate
    qJ2P = _seobv5phm_qprec(S, x)
    qtot = _qmul(_qconj(qJ2P), _qmul(_qconj(S.qI2J), _qeuler(phiref, inclination, 0.0)))
    Ra, Rb = _qcayley(qtot)
    dt = tq - S.t_a
    A = S.asym
    ha3 = (_seobv5hm_mr_eval(A.mr[1], dt), _seobv5hm_mr_eval(A.mr[2], dt), _seobv5hm_mr_eval(A.mr[3], dt))
    return _seobv5phm_hpc_kh(Val(1), _seobv5hm_full_mr(S, 1, dt), true, ha3, Ra, Rb) +
          _seobv5phm_hpc_kh(Val(2), _seobv5hm_full_mr(S, 2, dt), true, ha3, Ra, Rb) +
          _seobv5phm_hpc_kh(Val(3), _seobv5hm_full_mr(S, 3, dt), true, ha3, Ra, Rb) +
          _seobv5phm_hpc_kh(Val(4), _seobv5hm_full_mr(S, 4, dt), true, ha3, Ra, Rb) +
          _seobv5phm_hpc_kh(Val(5), _seobv5hm_full_mr(S, 5, dt), true, ha3, Ra, Rb) +
          _seobv5phm_hpc_kh(Val(6), _seobv5hm_full_mr(S, 6, dt), true, ha3, Ra, Rb)
end

"""
    seobnrv5phm_td(m1, m2, chi1, chi2, omega_ref, omega_start, ts; inclination, phiref, antisym = true)
        -> (hpc, modes, S)

Time-domain SEOBNRv5PHM (see [`SEOBNRv5PHMTD`](@ref)) at dynamics times `ts` (units of M from the start of the
dynamics; pyseobnr's `t = 0` is `S.t_peak`): `hpc = h+ - i hx` (geometric units, pyseobnr's model-level
`phiref` and `inclination`) and the six co-precessing modes' symmetric parts (`m > 0`). `m1 + m2 = 1`; spin
vectors at `omega_ref` in the frame where `LN_hat = z` there.
"""
function seobnrv5phm_td(m1, m2, chi1, chi2, omega_ref, omega_start, ts; inclination = 0.0, phiref = 0.0, kwargs...)
    S = _seobv5phm_build(m1, m2, chi1, chi2, omega_ref, omega_start; kwargs...)
    hpc = [_seobv5phm_hpc(S, t, inclination, phiref) for t in ts]
    modes = ntuple(k -> [_seobv5phm_coprec(S, k, t) for t in ts], 6)     # symmetric parts
    return hpc, modes, S
end

# The scalar (isbits) part of a built model `S` (without the dynamics, mode tables, base and gamma table, which a
# device keeps as arrays), and the model back from it with those arrays (e.g. a device walker's views).
@inline _seobv5phm_rec(S) = Base.structdiff(S, NamedTuple{(:dyn, :D, :b, :G)})
@inline _seobv5phm_from_rec(rec, d, D, b, G) = merge(rec, (dyn = d, D = D, b = b, G = G))
