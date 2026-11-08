# ================================================================================================
# IMRPhenomXP/XPHM multiple-scale-analysis (MSA) precession angles, `IMRPhenomXPrecVersion = 223`
# (Chatziioannou et al., PRD 95, 104004 (2017), arXiv:1703.03967, "as implemented in
# LALSimInspiralFDPrecAngles"): the LALSuite default for IMRPhenomXP/XPHM from lalsimulation 2.4.0
# (GWTC-2.1) through 6.2.0 (`PrecVersion` default 300 -> 223). Ported from
# `lalsimulation/lib/LALSimIMRPhenomX_precession.c`:
#   IMRPhenomX_Initialize_MSA_System (lines 2297-3097), IMRPhenomX_Return_phi_zeta_costhetaL_MSA
#   (2229-2294), _Roots_MSA (3112-3225), _JNorm_MSA, _SNorm_MSA, _Spin_Evolution_Coefficients_MSA,
#   _Constants_c_MSA / _d_MSA, _Psi(_dot)_MSA, _phiz_MSA, _zeta_MSA, _MSA_Corrections_MSA
#   (3230-3693), IMRPhenomX_L_norm_3PN_of_v (2222), IMRPhenomX_Get_PN_beta/sigma/tau (3700-3720),
#   and the 222/223 L(v) coefficients of IMRPhenomXGetAndSetPrecessionVariables (lines 716-728).
# Only the 223 branches are ported (where 220/221/222/224 differ, the 223 one is taken). The
# expansion order is LAL's default 5, which zeroes the 5th-order Omega coefficients (lines 2920-2968).
#
# GSL replacements: `gsl_sf_ellint_F(phi, k)` (modulus k) by Carlson's R_F, and
# `gsl_sf_elljac_e(u, m)` (parameter m) by the AGM / descending-Landen method -- both with a fixed
# iteration cap and an early freeze, so they are GPU-compilable and forward-differentiable.
# Branches are `ifelse` on values that are always finite where it matters; the NaN-to-zero guards
# LAL applies to phi_z, zeta and the MSA corrections are reproduced.
# ================================================================================================

# Carlson's symmetric elliptic integral R_F(x, y, z) (duplication theorem, Carlson 1995; Numerical
# Recipes `rf`): 12 duplication steps (|delta| shrinks 4x per step) then the 5th-order series.
@inline function _carlson_rf(x, y, z)
    T = typeof(x)
    for _ in 1:12
        sx = sqrt(x); sy = sqrt(y); sz = sqrt(z)
        lam = sx * (sy + sz) + sy * sz
        x = (x + lam) / 4; y = (y + lam) / 4; z = (z + lam) / 4
    end
    ave = (x + y + z) / 3
    dx = (ave - x) / ave; dy = (ave - y) / ave; dz = -(dx + dy)
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    return (1 + (e2 / 24 - T(0.1) - T(3) / 44 * e3) * e2 + e3 / 14) / sqrt(ave)
end

# Incomplete elliptic integral of the first kind F(phi, k), modulus k (GSL's `gsl_sf_ellint_F` convention), for
# |phi| <= pi/2 (all MSA uses: phi = asin(...)).
@inline _ellint_F(phi, k) = sin(phi) * _carlson_rf(cos(phi)^2, 1 - (k * sin(phi))^2, one(phi))

# Jacobi elliptic sn(u | m), parameter m in [0, 1] (GSL's `gsl_sf_elljac_e` convention), by the arithmetic-geometric
# mean (Abramowitz & Stegun 16.4): descend while c_n / a_n is significant (at most 16 levels), then ascend. Unrolled
# scalars (no loops, no tuple updates): Enzyme reverse mode cannot type a tuple rebuilt inside a loop (EnzymeNoTypeError),
# and in-kernel reverses must be loop-free anyway. Levels past the last significant one are skipped by
# branches (as selects, all 16 asin / sin / sqrt were evaluated with their tangents -- the bulk of the XPHM
# 223 device angle table; the AGM converges in ~5 levels, and once |c| <= eps a it stays there, c -> c^2 / 4a).
@inline function _jacobi_sn(u, m)
    T = typeof(u)
    a = one(T); b = sqrt(max(one(T) - m, zero(T))); n = 0; an = a; go = true
    Base.Cartesian.@nexprs 16 i -> (r_i = zero(T))
    Base.Cartesian.@nexprs 16 i -> begin
        if go
            ai = (a + b) / 2
            ci = (a - b) / 2
            b = sqrt(a * b)
            a = ai
            if abs(ci) > eps(T) * abs(ai)
                r_i = ci / ai; n = i; an = ai
            else
                go = false
            end
        end
    end
    # phi_n = 2^n a_n u, then down to phi_0
    phi = an * T(1 << n) * u                        # (not ldexp: Enzyme GPU forward cannot differentiate its exponent-bit ops)
    Base.Cartesian.@nexprs 16 k -> begin
        if 17 - k <= n
            phi = (phi + asin(r_{17 - k} * sin(phi))) / 2
        end
    end
    phi_0 = phi
    sn = sin(phi_0)
    # GSL's limits: m ~ 0 -> sin(u), m ~ 1 -> tanh(u)
    sn = ifelse(abs(m) < 2 * eps(T), sin(u), sn)
    sn = ifelse(abs(m - 1) < 2 * eps(T), tanh(u), sn)
    return sn
end

# The 222/223 L(v) coefficients (L0..L8L) of IMRPhenomXGetAndSetPrecessionVariables (lines 716-728): used for L(v_ref) in
# J0 (the frame), as the 102 coefficients are by `_xp_L3PN_coeffs`.
@inline function _xp_L3PN_coeffs_223(eta, chi1L, chi2L, delta)
    T = typeof(eta)
@Tlit T begin
    eta2 = eta * eta
    L0 = 1.0
    L1 = 0.0
    L2 = 3.0 / 2.0 + eta / 6.0
    L3 = (-7.0 * (chi1L + chi2L + chi1L * delta - chi2L * delta) + 5.0 * (chi1L + chi2L) * eta) / 6.0
    L4 = (81.0 + (-57.0 + eta) * eta) / 24.0
    L5 = (-1650.0 * (chi1L + chi2L + chi1L * delta - chi2L * delta) + 1336.0 * (chi1L + chi2L) * eta +
          511.0 * (chi1L - chi2L) * delta * eta + 28.0 * (chi1L + chi2L) * eta2) / 600.0
    L6 = (10935.0 + eta * (-62001.0 + 1674.0 * eta + 7.0 * eta2 + 2214.0 * pi * pi)) / 1296.0
    return (L0, L1, L2, L3, L4, L5, L6, 0.0, 0.0, 0.0)
end
end

# Roots of the S^2 cubic (Eq. 21; `IMRPhenomX_Return_Roots_MSA`): (S3^2, S-^2, S+^2) at (|L|, |J|).
@inline function _xp_msa_roots(LNorm, JNorm, P)
    T = typeof(LNorm)
@Tlit T begin
    q = P.qq; eta = P.eta; delta = P.delta_qq; Seff = P.Seff
    S1N2 = P.S1_norm_2; S2N2 = P.S2_norm_2
    JNorm2 = JNorm * JNorm; LNorm2 = LNorm * LNorm
    J2mL2 = JNorm2 - LNorm2; J2mL2Sq = J2mL2 * J2mL2
    omqq = (1.0 - q) / q
    B = (LNorm2 + S1N2) * q + 2.0 * LNorm * Seff - 2.0 * JNorm2 - S1N2 - S2N2 + (LNorm2 + S2N2) / q
    C = J2mL2Sq - 2.0 * LNorm * Seff * J2mL2 - 2.0 * omqq * LNorm2 * (S1N2 - q * S2N2) +
        4.0 * eta * LNorm2 * Seff * Seff - 2.0 * delta * (S1N2 - S2N2) * Seff * LNorm +
        2.0 * omqq * (q * S1N2 - S2N2) * JNorm2
    D = omqq * (S2N2 - q * S1N2) * J2mL2Sq + delta * delta * (S1N2 - S2N2) * (S1N2 - S2N2) * LNorm2 / eta +
        2.0 * delta * LNorm * Seff * (S1N2 - S2N2) * J2mL2
    p = C - B * B / 3.0
    qc = (2.0 / 27.0) * B * B * B - B * C / 3.0 + D
    mp3 = -p / 3.0
    sqrtarg_nan = mp3 < 0.0                            # NaN in C when -p/3 < 0 -> degenerate branch below
    pos = mp3 > 0.0
    sqrtarg = ifelse(pos, sqrt(ifelse(pos, mp3, 1.0)), 0.0)
    acosarg_nan = isnan(clamp(1.5 * qc / p / sqrtarg, -1.0, 1.0))
    # Safe inputs (AD only): p, sqrtarg are 1 where they vanish (theta is then multiplied by sqrtarg = 0 or the branch is
    # degenerate), and the argument is frozen just inside +-1. At a numerically double root (|S-^2 - S+^2| -> 0: q -> 1,
    # or a vanishing spin) it reaches +-1, where acos' = Inf gives 0 * Inf = NaN gradients in both modes; the frozen
    # theta (<= 1.5e-8) moves the roots by < 1e-7 sqrtarg, far below the 1e-5 nutation threshold.
    acosarg = clamp(1.5 * qc / ifelse(pos, p, -3.0) / ifelse(pos, sqrtarg, 1.0), -1.0 + 1.0e-15, 1.0 - 1.0e-15)
    acosarg = ifelse(acosarg_nan, 0.0, acosarg)
    theta = acos(acosarg) / 3.0
    t1 = 2.0 * sqrtarg * cos(theta - 2.0 * (2.0 * pi) / 3.0) - B / 3.0
    t2 = 2.0 * sqrtarg * cos(theta - (2.0 * pi) / 3.0) - B / 3.0
    t3 = 2.0 * sqrtarg * cos(theta) - B / 3.0
    t4 = max(max(t1, t2), t3)
    t5 = min(min(t1, t2), t3)
    t6 = ifelse((t4 - t3) > 0.0 && (t5 - t3) < 0.0, t3, ifelse((t4 - t1) > 0.0 && (t5 - t1) < 0.0, t1, t2))
    degenerate = acosarg_nan | sqrtarg_nan | (P.dotS1Ln == 1.0) | (P.dotS2Ln == 1.0) | (P.dotS1Ln == -1.0) |
                 (P.dotS2Ln == -1.0) | (S1N2 == 0.0) | (S2N2 == 0.0)
    S32 = ifelse(degenerate, 0.0, t5)
    Smi2 = ifelse(degenerate, P.S_0_norm_2, abs(t6))
    Spl2 = ifelse(degenerate, P.S_0_norm_2 + 1.0e-9, abs(t4))
    return S32, Smi2, Spl2
end
end

# Safe-input roots for the branches LAL switches off when |S-^2 - S+^2| <= 1e-5 (SNorm oscillation, MSA corrections,
# psi0): there S-^2 ~ S+^2 (exactly at q = 1) and those branches divide by S-^2 - S+^2, so their discarded values are
# Inf / NaN and their AD partials poison the gradient (0 * Inf). Values with `big` are unchanged.
@inline _xp_msa_safe_roots(big, S32, Smi2, Spl2) =
    (ifelse(big, S32, Spl2 - oftype(Spl2, 0.02)), ifelse(big, Smi2, Spl2 - oftype(Spl2, 0.01)), Spl2)

@inline _xp_msa_JNorm(LNorm, P) = sqrt(LNorm * LNorm + 2 * LNorm * P.c1_over_eta + P.SAv2)

# Eq. 51: psi(v) (`IMRPhenomX_psiofv` with the stored psi1, psi2)
@inline _xp_msa_psiofv(v, psi0, P) = psi0 - (3 * P.g0 * P.delta_qq / 4) * (1 + P.psi1 * v + P.psi2 * v * v) / (v * v * v)

# |S|(v) (Eq. 23, `IMRPhenomX_Return_SNorm_MSA`), for the current roots
@inline function _xp_msa_SNorm(v, S32, Smi2, Spl2, P)
    T = typeof(v)
    small = abs(Smi2 - Spl2) < T(1.0e-5)
    S32s, Smi2s, _ = _xp_msa_safe_roots(!small, S32, Smi2, Spl2)
    m = ifelse(small, zero(T), (Smi2s - Spl2) / (S32s - Spl2))
    sn = ifelse(small, zero(T), _jacobi_sn(ifelse(small, zero(T), _xp_msa_psiofv(v, P.psi0, P)), m))
    return sqrt(Spl2 + (Smi2 - Spl2) * sn * sn)
end

# phi_z (Eq. 66, `IMRPhenomX_Return_phiz_MSA`) with the integration constant phiz_0
@inline function _xp_msa_phiz(v, JNorm, phiz_0, P)
    T = typeof(v)
@Tlit T begin
    eta = P.eta; invv = 1.0 / v; invv2 = invv * invv
    LNewt = eta / v
    c1 = P.c1; c12 = c1 * c1
    SAv2 = P.SAv2; SAv = P.SAv; invSAv = 1.0 / SAv; invSAv2 = 1.0 / SAv2
    inveta = 1.0 / eta; inveta2 = inveta * inveta; inveta3 = inveta2 * inveta; inveta4 = inveta2 * inveta2
    eta2 = eta * eta; eta4 = eta2 * eta2
    log1 = log(abs(c1 + JNorm * eta + eta * LNewt))
    log2 = log(abs(c1 + JNorm * SAv * v + SAv2 * v))
    p0 = (JNorm * inveta4) * (0.5 * c12 - c1 * eta2 * invv / 6.0 - SAv2 * eta2 / 3.0 - eta4 * invv2 / 3.0) -
         (c1 * 0.5 * inveta) * (c12 * inveta4 - SAv2 * inveta2) * log1
    p1 = -0.5 * JNorm * inveta2 * (c1 + eta * LNewt) + 0.5 * inveta3 * (c12 - eta2 * SAv2) * log1
    p2 = -JNorm + SAv * log2 - c1 * log1 * inveta
    p3 = JNorm * v - eta * log1 + c1 * log2 * invSAv
    p4 = (0.5 * JNorm * invSAv2 * v) * (c1 + v * SAv2) - (0.5 * invSAv2 * invSAv) * (c12 - eta2 * SAv2) * log2
    p5 = -JNorm * v * ((0.5 * c12 * invSAv2 * invSAv2) - (c1 * v * invSAv2 / 6.0) - v * v / 3.0 - eta2 * invSAv2 / 3.0) +
         (0.5 * c1 * invSAv2 * invSAv2 * invSAv) * (c12 - eta2 * SAv2) * log2
    O = P.Omegaz_coeff
    out = p0 * O[1] + p1 * O[2] + p2 * O[3] + p3 * O[4] + p4 * O[5] + p5 * O[6] + phiz_0
    return ifelse(isnan(out), 0.0, out)
end
end

# zeta (Eq. F5, `IMRPhenomX_Return_zeta_MSA`) with the integration constant zeta_0
@inline function _xp_msa_zeta(v, zeta_0, P)
    T = typeof(v)
    invv = 1 / v
    O = P.Omegazeta_coeff
    out = P.eta * (O[1] * invv * invv * invv + O[2] * invv * invv + O[3] * invv + O[4] * log(v) + O[5] * v + O[6] * v * v) +
          zeta_0
    return ifelse(isnan(out), zero(T), out)
end

# MSA corrections (Eq. 67 / F19 in the 223 form, `IMRPhenomX_Return_MSA_Corrections_MSA`) for the current roots.
@inline function _xp_msa_corrections(v, LNorm, JNorm, S32, Smi2, Spl2, P)
    T = typeof(v)
@Tlit T begin
    eta = P.eta; inveta = 1.0 / eta; Seff = P.Seff
    v2 = v * v; v3 = v2 * v; v4 = v2 * v2; v6 = v3 * v3
    JNorm2 = JNorm * JNorm; LNorm2 = LNorm * LNorm
    Spl = sqrt(Spl2)
    dS = Spl2 - Smi2
    JmS = JNorm2 - Spl2
    # B6-B8 (version != 220)
    c0 = JNorm * (0.75 * (1.0 - Seff * v) * v2 * (eta * eta * eta + 4.0 * eta * eta * eta * Seff * v -
                  2.0 * eta * (JmS + 2.0 * (P.S1_norm_2 - P.S2_norm_2) * P.delta_qq) * v2 -
                  4.0 * eta * Seff * JmS * v3 + JmS * JmS * v4 * inveta))
    c2 = JNorm * (-1.5 * eta * dS * (1.0 + 2.0 * Seff * v - JmS * v2 * inveta * inveta) * (1.0 - Seff * v) * v4)
    c4 = JNorm * (0.75 * inveta * dS * dS * (1.0 - Seff * v) * v6)
    # B9-B11
    d0 = -(JNorm2 - (LNorm + Spl) * (LNorm + Spl)) * (JNorm2 - (LNorm - Spl) * (LNorm - Spl))
    d2 = -2.0 * dS * (JNorm2 + LNorm2 - Spl2)
    d4 = -dS * dS
    two_d0 = 2.0 * d0
    sd = sqrt(abs(d2 * d2 - 4.0 * d0 * d4))
    A_theta_L = 0.5 * ((JNorm / LNorm) + (LNorm / JNorm) - (Spl2 / (JNorm * LNorm)))
    B_theta_L = 0.5 * dS / (JNorm * LNorm)
    nc_num = 2.0 * (d0 + d2 + d4)
    nc_denom = two_d0 + d2 + sd
    nc = nc_num / nc_denom
    nd = nc_denom / two_d0
    sqrt_nc = sqrt(abs(nc)); sqrt_nd = sqrt(abs(nd))
    psi = -0.75 * P.g0 * P.delta_qq * (1.0 + P.psi1 * v + P.psi2 * v2) / (v2 * v) + P.psi0
    psi_dot = 0.5 * (-1.5 * v6 * (1.0 - v * Seff) / sqrt(eta)) * sqrt(Spl2 - S32)
    tan_psi = tan(psi)
    atan_psi = atan(tan_psi)
    C2den = 2.0 * d0 * sd * (d0 + d2 + d4)
    Cphi = abs((c4 * d0 * ((2.0 * d0 + d2) + sd) - c2 * d0 * ((d2 + 2.0 * d4) - sd) -
                c0 * ((2.0 * d0 * d4) - (d2 + d4) * (d2 - sd))) / C2den) *
           (sqrt_nc / (nc - 1.0) * (atan_psi - atan(sqrt_nc * tan_psi))) / psi_dot
    Cphi = ifelse(nc == 1.0, 0.0, Cphi)
    Dphi = abs((-c4 * d0 * ((2.0 * d0 + d2) - sd) + c2 * d0 * ((d2 + 2.0 * d4) + sd) -
                c0 * (-(2.0 * d0 * d4) + (d2 + d4) * (d2 + sd)))) / C2den *
           (sqrt_nd / (nd - 1.0) * (atan_psi - atan(sqrt_nd * tan_psi))) / psi_dot
    Dphi = ifelse(nd == 1.0, 0.0, Dphi)
    x = Cphi + Dphi
    y = A_theta_L * x + 2.0 * B_theta_L * d0 * (Cphi / (sd - d2) - Dphi / (sd + d2))
    return ifelse(isnan(x), 0.0, x), ifelse(isnan(y), 0.0, y)
end
end

# (phi_z, zeta, cos theta_L) at v (`IMRPhenomX_Return_phi_zeta_costhetaL_MSA`), with given integration constants.
@inline function _xp_msa_phiz_zeta_costhL(v, phiz_0, zeta_0, P)
    T = typeof(v)
@Tlit T begin
    L_norm = P.eta / v
    J_norm = _xp_msa_JNorm(L_norm, P)
    v2 = v * v
    cL = P.constants_L
    L_norm3PN = L_norm * (1.0 + v2 * (cL[1] + v * cL[2] + v2 * (cL[3] + v * cL[4] + v2 * cL[5])))
    J_norm3PN = _xp_msa_JNorm(L_norm3PN, P)
    S32, Smi2, Spl2 = _xp_msa_roots(L_norm, J_norm, P)
    SNorm = _xp_msa_SNorm(v, S32, Smi2, Spl2, P)
    big = abs(Smi2 - Spl2) > 1.0e-5
    cx, cy = _xp_msa_corrections(v, L_norm, J_norm, _xp_msa_safe_roots(big, S32, Smi2, Spl2)..., P)
    cx = ifelse(big, cx, 0.0); cy = ifelse(big, cy, 0.0)
    phiz = _xp_msa_phiz(v, J_norm, phiz_0, P)
    zeta = _xp_msa_zeta(v, zeta_0, P)
    costhL = clamp(0.5 * (J_norm3PN * J_norm3PN + L_norm3PN * L_norm3PN - SNorm * SNorm) / (L_norm3PN * J_norm3PN), -1.0, 1.0)
    return phiz + cx, zeta + cy, costhL
end
end

"""
    _xp_msa_setup(m1n, m2n, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, v_0) -> NamedTuple

`IMRPhenomX_Initialize_MSA_System` for `IMRPhenomXPrecVersion = 223` (expansion order 5), with normalized masses
`m1n >= m2n` (`m1n + m2n = 1`) and `v_0 = (pi M f_ref)^(1/3)`. `ok = false` is LAL's `MSA_ERROR` (|Omega_z5| > 1000),
on which LAL falls back to the NNLO (102) angles. The integration constants `phiz_0`, `zeta_0` make phi_z + its MSA
correction and zeta + its correction vanish at `v_0`.
"""
@inline function _xp_msa_setup(m1n, m2n, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, v_0)
    T = promote_type(typeof(m1n), typeof(chi1x), typeof(chi2x), typeof(v_0))
@Tlit T begin
    eta = m1n * m2n
    eta2 = eta * eta; eta3 = eta2 * eta; eta4 = eta2 * eta2
    q = m2n / m1n
    delta_qq = (1.0 - q) / (1.0 + q)
    m1_2 = m1n * m1n; m2_2 = m2n * m2n
    # dimensionful spins (eta / q = m1^2, eta * q = m2^2)
    S1x = chi1x * m1_2; S1y = chi1y * m1_2; S1z = chi1z * m1_2
    S2x = chi2x * m2_2; S2y = chi2y * m2_2; S2z = chi2z * m2_2
    S1_norm = sqrt(chi1x * chi1x + chi1y * chi1y + chi1z * chi1z) * m1_2
    S2_norm = sqrt(chi2x * chi2x + chi2y * chi2y + chi2z * chi2z) * m2_2
    S1_norm_2 = S1_norm * S1_norm; S2_norm_2 = S2_norm * S2_norm
    S1_0_norm = sqrt(S1x * S1x + S1y * S1y + S1z * S1z)
    S2_0_norm = sqrt(S2x * S2x + S2y * S2y + S2z * S2z)
    v_0_2 = v_0 * v_0
    L0z = eta / v_0                                       # Newtonian L along z
    dotS1L = S1z; dotS2L = S2z
    dotS1S2 = S1x * S2x + S1y * S2y + S1z * S2z
    dotS1Ln = dotS1L / S1_0_norm
    dotS2Ln = dotS2L / S2_0_norm
    beta(a, b) = dotS1L * (a + b * q) + dotS2L * (a + b / q)
    sigma(a, b) = (a * dotS1S2 - b * dotS1L * dotS2L) / eta
    tau(a, b) = (q * (S1_norm_2 * a - b * dotS1L * dotS1L) + (a * S2_norm_2 - b * dotS2L * dotS2L) / q) / eta
    constants_L = (1.5 + eta / 6.0,
                   beta(-14.0 / 6.0, -1.5),
                   27.0 / 8.0 - 19.0 / 8.0 * eta + eta * eta / 24.0,
                   beta(-11.0 / 2.0 + 133.0 / 72.0 * eta, -33.0 / 8.0 + 7.0 / 4.0 * eta),
                   135.0 / 16.0 + (-6889.0 / 144.0 + 41.0 / 24.0 * pi * pi) * eta + 31.0 / 24.0 * eta * eta +
                   7.0 / 1296.0 * eta * eta * eta)
    Seff = (1.0 + q) * dotS1L + (1.0 + 1.0 / q) * dotS2L
    S0x = S1x + S2x; S0y = S1y + S2y; S0z = S1z + S2z
    S_0_norm_2 = S0x * S0x + S0y * S0y + S0z * S0z
    J0z = L0z + S0z
    L_0_norm = abs(L0z)
    J_0_norm = sqrt(S0x * S0x + S0y * S0y + J0z * J0z)

    # roots at (L_0, J_0); c1 and SAv2 come from these, so the roots need a partial struct
    P0 = (qq = q, eta = eta, delta_qq = delta_qq, Seff = Seff, S1_norm_2 = S1_norm_2, S2_norm_2 = S2_norm_2,
          dotS1Ln = dotS1Ln, dotS2Ln = dotS2Ln, S_0_norm_2 = S_0_norm_2)
    S32_0, Smi2_0, Spl2_0 = _xp_msa_roots(L_0_norm, J_0_norm, P0)
    SAv2 = 0.5 * (Spl2_0 + Smi2_0)
    SAv = sqrt(SAv2)
    c1 = 0.5 * (J_0_norm * J_0_norm - L_0_norm * L_0_norm - SAv2) / L_0_norm * eta
    c12 = c1 * c1
    c1_over_eta = c1 / eta

    # precession-averaged couplings (A9-A14): only S1L_pav + S2L_pav is used (final spin version 3), which is exactly
    # c1 / eta. LAL sums the two terms over (1 - q^2) + 1e-16, which agrees to 1e-16 / (1 - q^2) but is rounding noise
    # / 1e-16 at q = 1 (a ~0.1 jump in the final spin, and 1e16 derivatives).
    SL_pav = c1_over_eta

    # PN coefficients of d omega / dt from the initial spin couplings (222/223, lines 2628-2635)
    a0 = eta * (96.0 / 5.0)
    a2 = eta * (-1486.0 / 35.0 + eta * (-264.0 / 5.0))
    a3 = eta * (384.0 * pi / 5.0 + beta(-904.0 / 5.0, -120.0))
    a4 = eta * (34103.0 / 945.0 + eta * (13661.0 / 105.0 + eta * (944.0 / 15.0)) + sigma(-494.0 / 5.0, -1442.0 / 5.0) +
                tau(-233.0 / 5.0, -719.0 / 5.0))
    a5 = eta * (pi * (-4159.0 / 35.0) + eta * (pi * (-2268.0 / 5.0)) +
                beta(-62638.0 / 105.0 + eta * (4636.0 / 5.0), -6472.0 / 35.0 + eta * (3372.0 / 5.0)))
    a0_2 = a0 * a0; a0_3 = a0_2 * a0
    g0 = 1.0 / a0
    g2 = -(a2 / a0_2)
    g3 = -(a3 / a0_2)
    g4 = -(a4 * a0 - a2 * a2) / a0_3
    g5 = -(a5 * a0 - 2.0 * a3 * a2) / a0_3

    # psi1, Delta, psi2 (222/223 forms, lines 2690, 2706-2715, 2740-2752)
    # psi1 ~ 1 / delta^2, psi2 ~ 1 / (1 - q)^4: +-Inf at q = 1, where psi = delta (...) is then NaN. psi only reaches
    # branches switched off at q = 1 (|S-^2 - S+^2| = 0), so safe denominators there leave the values as LAL's.
    delta = delta_qq; delta2 = delta * delta
    psi1 = 3.0 * (2.0 * eta2 * Seff - c1) / (eta * ifelse(delta2 > 0.0, delta2, 1.0))
    c1n = c1_over_eta; c1n2 = c1n * c1n
    opq2 = (1.0 + q) * (1.0 + q); q_2 = q * q
    omq2_2 = (1.0 - q_2) * (1.0 - q_2)
    omq_4 = ((1.0 - q) * (1.0 - q)) * ((1.0 - q) * (1.0 - q))
    omq_4 = ifelse(omq_4 > 0.0, omq_4, 1.0)
    Seff_2 = Seff * Seff
    Del1 = 4.0 * c1n2 * opq2
    Del2 = 8.0 * c1n * q * (1.0 + q) * Seff
    Del3 = 4.0 * (omq2_2 * S1_norm_2 - q_2 * Seff_2)
    Del4 = 4.0 * c1n2 * q_2 * opq2
    Del5 = 8.0 * c1n * q_2 * (1.0 + q) * Seff
    Del6 = 4.0 * (omq2_2 * S2_norm_2 - q_2 * Seff_2)
    Delta = sqrt(abs((Del1 - Del2 - Del3) * (Del4 - Del5 - Del6)))
    u1 = 3.0 * g2 / g0
    u2 = 0.75 * opq2 / omq_4
    u3 = -20.0 * c1n2 * q_2 * opq2
    u4 = 2.0 * omq2_2 * (q * (2.0 + q) * S1_norm_2 + (1.0 + 2.0 * q) * S2_norm_2 - 2.0 * q * SAv2)
    u5 = 2.0 * q_2 * (7.0 + 6.0 * q + 7.0 * q_2) * 2.0 * c1n * Seff
    u6 = 2.0 * q_2 * (3.0 + 4.0 * q + 3.0 * q_2) * Seff_2
    u7 = q * Delta
    psi2 = u1 + u2 * (u3 + u4 + u5 - u6 + u7)

    # Omega_z, Omega_zeta (Appendix D, E; lines 2777-2917)
    Rm = Spl2_0 - Smi2_0; Rm_2 = Rm * Rm
    cp = Spl2_0 * eta2 - c12
    cm = Smi2_0 * eta2 - c12
    sqrt_cpcm = sqrt(abs(cp * cm))
    a1dD = 0.5 + 0.75 / eta
    a2dD = -0.75 * Seff / eta
    D2RmSq = (cp - sqrt_cpcm) / eta2
    D4RmSq = -0.5 * Rm * sqrt_cpcm / eta2 - cp / eta4 * (sqrt_cpcm - cp)
    S0m = S1_norm_2 - S2_norm_2
    aw = (-3.0 * (1.0 + q) / q * (2.0 * (1.0 + q) * eta2 * Seff * c1 - (1.0 + q) * c12 + (1.0 - q) * eta2 * S0m))
    cw = 3.0 / 32.0 / eta * Rm_2
    dw = 4.0 * cp - 4.0 * D2RmSq * eta2
    hw = -2.0 * (2.0 * D2RmSq - Rm) * c1
    fw = Rm * D2RmSq - D4RmSq - 0.25 * Rm_2
    adD = aw / dw; hdD = hw / dw; cdD = cw / dw; fdD = fw / dw
    gw = 3.0 / 16.0 / eta2 / eta * Rm_2 * (c1 - eta2 * Seff)
    gdD = gw / dw
    hdD_2 = hdD * hdD; adDfdD = adD * fdD; adDfdDhdD = adDfdD * hdD; adDhdD_2 = adD * hdD_2
    Oz0 = a1dD + adD
    Oz1 = a2dD - adD * Seff - adD * hdD
    Oz2 = adD * hdD * Seff + cdD - adD * fdD + adD * hdD_2
    Oz3 = (adDfdD - cdD - adDhdD_2) * (Seff + hdD) + adDfdDhdD
    Oz4 = (cdD + adDhdD_2 - 2.0 * adDfdD) * (hdD * Seff + hdD_2 - fdD) - adD * fdD * fdD
    Oz5 = (cdD - adDfdD + adDhdD_2) * fdD * (Seff + 2.0 * hdD) - (cdD + adDhdD_2 - 2.0 * adDfdD) * hdD_2 * (Seff + hdD) -
          adDfdD * fdD * hdD
    ok = !(abs(Oz5) > 1000.0)
    Omegaz_coeff = (3.0 * g0 * Oz0,
                    3.0 * g0 * Oz1,
                    3.0 * (g0 * Oz2 + g2 * Oz0),
                    3.0 * (g0 * Oz3 + g2 * Oz1 + g3 * Oz0),
                    3.0 * (g0 * Oz4 + g2 * Oz2 + g3 * Oz1 + g4 * Oz0),
                    0.0)                                   # expansion order 5: order 5 dropped
    c1oveta2 = c1 / eta2
    Oze0 = Oz0
    Oze1 = Oz1 + Oz0 * c1oveta2
    Oze2 = Oz2 + Oz1 * c1oveta2
    Oze3 = Oz3 + Oz2 * c1oveta2 + gdD
    Oze4 = Oz4 + Oz3 * c1oveta2 - gdD * Seff - gdD * hdD
    Omegazeta_coeff = (-g0 * Oze0,
                       -1.5 * g0 * Oze1,
                       -3.0 * (g0 * Oze2 + g2 * Oze0),
                       3.0 * (g0 * Oze3 + g2 * Oze1 + g3 * Oze0),
                       3.0 * (g0 * Oze4 + g2 * Oze2 + g3 * Oze1 + g4 * Oze0),
                       0.0)

    # psi0 (lines 3004-3034): F(asin(sign(V) sqrt(B)), k) - psi(v_0), V = (L_0 x S1) . S2
    P1 = (P0..., g0 = g0, psi1 = psi1, psi2 = psi2)
    small = abs(Smi2_0 - Spl2_0) < 1.0e-5
    S32_0s, Smi2_0s, _ = _xp_msa_safe_roots(!small, S32_0, Smi2_0, Spl2_0)
    mm = sqrt(abs((Smi2_0s - Spl2_0) / (S32_0s - Spl2_0)))
    tmpB = (S_0_norm_2 - Spl2_0) / (Smi2_0s - Spl2_0)
    vol = L0z * (S1x * S2y - S1y * S2x)                  # (L_0 x S1) . S2 with L_0 = (0, 0, L0z)
    vol_sign = T((vol > 0.0) - (vol < 0.0))
    psi_of_v0 = _xp_msa_psiofv(v_0, 0.0, P1)
    Bc = ifelse(tmpB > 1.0 && (tmpB - 1.0) < 0.00001, 1.0, ifelse(tmpB < 0.0 && tmpB > -0.00001, 0.0, tmpB))
    inrange = (0.0 <= tmpB <= 1.0) | (tmpB > 1.0 && (tmpB - 1.0) < 0.00001) | (tmpB < 0.0 && tmpB > -0.00001)
    Bc = clamp(ifelse(small | !inrange, 0.5, Bc), 0.0, 1.0)  # 0.5: safe input (sqrt' = Inf at 0) for the discarded branch
    psi0 = ifelse(small | !inrange, 0.0, _ellint_F(asin(vol_sign * sqrt(Bc)), mm) - psi_of_v0)

    P = (P1..., psi0 = psi0, c1 = c1, c1_over_eta = c1_over_eta, SAv2 = SAv2, SAv = SAv,
         Omegaz_coeff = Omegaz_coeff, Omegazeta_coeff = Omegazeta_coeff, constants_L = constants_L)

    # integration constants: phi_z + correction and zeta + correction vanish at v_0 (lines 3042-3068)
    bigc = abs(Spl2_0 - Smi2_0) > 1.0e-5
    cx, cy = _xp_msa_corrections(v_0, L_0_norm, J_0_norm, _xp_msa_safe_roots(bigc, S32_0, Smi2_0, Spl2_0)..., P)
    cx = ifelse(bigc, cx, 0.0); cy = ifelse(bigc, cy, 0.0)
    phiz_0 = -_xp_msa_phiz(v_0, J_0_norm, 0.0, P) - cx
    zeta_0 = -_xp_msa_zeta(v_0, 0.0, P) - cy

    return (P..., phiz_0 = phiz_0, zeta_0 = zeta_0, ok = ok, SL_pav = SL_pav, v_0 = v_0)
end
end

"""
    _xp_msa_angles(v, msa) -> (phi_z, zeta, cos theta_L)

The MSA angles at PN velocity `v` (LAL's `IMRPhenomX_Return_phi_zeta_costhetaL_MSA` with the setup's integration
constants): alpha = phi_z - alpha_offset, epsilon = zeta - epsilon_offset, cos(beta) = cos theta_L.
"""
@inline _xp_msa_angles(v, msa) = _xp_msa_phiz_zeta_costhL(v, msa.phiz_0, msa.zeta_0, msa)

# ================================================================================================
# Version-aware precession setup and Euler angles (102 NNLO / 223 MSA), as LAL's
# IMRPhenomXGetAndSetPrecessionVariables + IMRPhenomXPTwistUp22 / IMRPhenomXPHM per-mode twist:
#   - 223: J0 frame from L(v_ref) with the 222/223 L coefficients; angles phi_z, zeta, cos theta_L (MSA) with
#     alpha = phi_z - alpha_offset, epsilon = zeta - epsilon_offset, offsets from the angles at v_ref (m' = 2, reused
#     for every mode, `Get_alphaepsilon_atfref`); final spin = FinalSpinMod 3 (LAL default 4 -> 3 for these versions):
#     sign(a_par) sqrt(S_av^2 + L_f^2 + 2 L_f SL_pav), SL_pav = S1L_pav + S2L_pav = c1 / eta, L_f = a_par - m1^2 chi1z - m2^2 chi2z.
#   - MSA failure (|Omega_z5| > 1000, `ok = false`): LAL switches the whole setup to 102 BEFORE the L coefficients and
#     the final spin are chosen (lines 616-640 precede 686-787 and 1399-1458), so the fallback is the complete 102
#     setup: NNLO angles, 102 frame, chi_p final spin (FinalSpinMod 3 -> 0 on MSA_ERROR). Both are computed and
#     selected with `ifelse` (GPU / forward-AD friendly).
#   - 102: as before; final spin = FinalSpinMod 0 (chi_p), what LAL uses for 102 (3 needs MSA -> 0).
# ================================================================================================

# eta as the waveform struct computes it (m1 m2 / M^2, as `_xhm_mc_eta`; the setup's m1n m2n differs by an ulp, which the
# ringdown frequency amplifies to ~1e-11 in the waveform): the final spin uses this one.
@inline _xp_eta_w(T, m1, m2) = (T(m1) * T(m2)) / ((T(m1) + T(m2)) * (T(m1) + T(m2)))

# a_parallel: `XLALSimIMRPhenomXFinalSpin2017(eta, chi1L, chi2L)` (the aligned-spin fit inside `_xas_precessing_final_spin`)
@inline function _xas_aligned_final_spin(eta, chi1, chi2)
    T = promote_type(typeof(eta), typeof(chi1), typeof(chi2))
    @Tlit T begin
        Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
        mm1 = 0.5 * (1.0 + Seta)
        mm2 = 0.5 * (1.0 - Seta)
        StotR = (mm1 * mm1 * chi1 + mm2 * mm2 * chi2) / (mm1 * mm1 + mm2 * mm2)
        a_parallel, _ = _xas_a_Erad(eta, Seta, chi1 - chi2, StotR)
        return a_parallel
    end
end

"""
    _xp_precession_setup(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination, phiRef;
                         GMsun_over_c3, prec_version = Val(102))

The precession setup for LAL's `IMRPhenomXPrecVersion` 102 (NNLO) or 223 (MSA): the frame and NNLO fields of
[`_xp_precession_setup_core`](@ref) plus `final_spin` (the remnant spin for the co-precessing ringdown), and for 223
the MSA block `msa`, `msa_ok` and the MSA offsets. Evaluate the angles with [`_xp_euler_angles`](@ref).
"""
@inline function _xp_precession_setup(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination, phiRef;
                                      GMsun_over_c3 = GMsun_over_c3, prec_version::Val = Val(102))
    return _xp_precession_setup(prec_version, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination,
                                phiRef; GMsun_over_c3 = GMsun_over_c3)
end

@inline function _xp_precession_setup(::Val{102}, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination,
                                      phiRef; GMsun_over_c3 = GMsun_over_c3)
    s = _xp_precession_setup_core(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination, phiRef, Val(102);
                                  GMsun_over_c3 = GMsun_over_c3)
    return merge(s, (final_spin = _xas_precessing_final_spin(_xp_eta_w(s.T, m1, m2), s.T(chi1z), s.T(chi2z), s.chi_p),))
end

@inline function _xp_precession_setup(::Val{223}, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination,
                                      phiRef; GMsun_over_c3 = GMsun_over_c3)
    s102 = _xp_precession_setup_core(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination, phiRef,
                                     Val(102); GMsun_over_c3 = GMsun_over_c3)
    s223 = _xp_precession_setup_core(m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_ref, inclination, phiRef,
                                     Val(223); GMsun_over_c3 = GMsun_over_c3)
    T = s102.T
    v_ref = cbrt(T(pi) * (T(m1) + T(m2)) * T(GMsun_over_c3) * T(f_ref))
    msa = _xp_msa_setup(s102.m1n, s102.m2n, T(chi1x), T(chi1y), T(chi1z), T(chi2x), T(chi2y), T(chi2z), v_ref)
    ok = msa.ok
    # MSA offsets (Get_alphaepsilon_atfref, m' = 2): the angles at v_ref (zero by construction of phiz_0, zeta_0, up to
    # rounding) minus alpha0, epsilon0 of the 223 frame
    phiz_ref, zeta_ref, _ = _xp_msa_angles(v_ref, msa)
    sel(a, b) = ifelse(ok, a, b)
    eta_w = _xp_eta_w(T, m1, m2)
    a_par = _xas_aligned_final_spin(eta_w, T(chi1z), T(chi2z))
    Lf = a_par - s102.m1n * s102.m1n * T(chi1z) - s102.m2n * s102.m2n * T(chi2z)
    fs_msa = copysign(one(T), a_par) * sqrt(abs(msa.SAv2 + Lf * Lf + 2 * Lf * msa.SL_pav))
    fs_102 = _xas_precessing_final_spin(eta_w, T(chi1z), T(chi2z), s102.chi_p)
    return merge(s102, (
        J0x = sel(s223.J0x, s102.J0x), J0y = sel(s223.J0y, s102.J0y), J0z = sel(s223.J0z, s102.J0z),
        J0 = sel(s223.J0, s102.J0), thetaJ = sel(s223.thetaJ, s102.thetaJ), phiJ = sel(s223.phiJ, s102.phiJ),
        thetaJN = sel(s223.thetaJN, s102.thetaJN), kappa = sel(s223.kappa, s102.kappa),
        alpha0 = sel(s223.alpha0, s102.alpha0), epsilon0 = sel(s223.epsilon0, s102.epsilon0),
        zeta_polarization = sel(s223.zeta_polarization, s102.zeta_polarization),
        msa = msa, msa_ok = ok,
        alpha_offset_msa = phiz_ref - s223.alpha0, epsilon_offset_msa = zeta_ref - s223.epsilon0,
        final_spin = sel(fs_msa, fs_102),
    ))
end

"""
    _xp_euler_angles(v, setup) -> (cos_beta, alpha, epsilon)

The precession Euler angles at PN velocity `v` (for mode m', `v = (pi M f 2 / m')^(1/3)`): NNLO for a 102 setup; MSA
for a 223 setup, or NNLO where the MSA system failed (`setup.msa_ok == false`, LAL's fallback to 102).
"""
@inline function _xp_euler_angles(v, setup)
    cb_n = _xp_cos_beta(v, setup)
    al_n = _xp_pn_euler_alpha_nnlo(v, setup)
    ep_n = _xp_pn_euler_epsilon_nnlo(v, setup)
    hasfield(typeof(setup), :msa) || return cb_n, al_n, ep_n
    phiz, zeta, cb_m = _xp_msa_angles(v, setup.msa)
    ok = setup.msa_ok
    return ifelse(ok, cb_m, cb_n), ifelse(ok, phiz - setup.alpha_offset_msa, al_n),
           ifelse(ok, zeta - setup.epsilon_offset_msa, ep_n)
end
