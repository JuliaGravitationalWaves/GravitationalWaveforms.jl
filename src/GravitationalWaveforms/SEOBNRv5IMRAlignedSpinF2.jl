export SEOBNRv5IMRAlignedSpinF2

"""
    SEOBNRv5IMRAlignedSpinF2

**SEOBNRv5 (2,2) inspiral-merger-ringdown in the frequency domain**: the real SEOBNRv5 Hamiltonian and
7-mode factorized radiation reaction ([`SEOBNRv5AlignedSpin`](@ref)), the fixed post-adiabatic inspiral
([`seobnrv5alignedspin_postadiabatic_dynamics_fixed`](@ref)), then the shared IMR pipeline of
`EOBIMR.jl` (adaptive RK4 plunge to the orbital-frequency peak, `t_attach =
t_ISCO - DeltaT_22`, NQC against the SEOBNRv5 input-value fits, SEOBNRv5 merger-ringdown ansatz, SPA +
FFT frequency domain), with the dynamics those fits were calibrated for.

Inspiral mode: the factorized `|h22| = T22 Heff hNewton22 rho22^2` (pyseobnr `compute_hlms`'s (2,2)
magnitude, the same expressions `_seobv5_rr_force` sums into the flux, see [`_seobv5_h22_abs`](@ref)),
with non-circular `v_phi`, and its phase including the tail phase and `delta22`
([`_imr_phase_corr`](@ref)). Full NQC (no shrinkage).
"""
struct SEOBNRv5IMRAlignedSpinF2 end

# The SEOBNRv5 coefficient NamedTuple minus its `T` (DataType) field: a returned struct holding a
# GC-tracked field takes Enzyme's sret+returnRoots path, which asserts on Julia 1.13 (Enzyme
# `fixup_1p12_sret!`, "TODO for 1.13"). Only the time-domain helpers read `coeffs.T`.
function _imr_base(::SEOBNRv5IMRAlignedSpinF2, mc, eta, chi1, chi2, f_start, GMsun_over_c3)
    b = _seobv5_coeffs(SEOBNRv5AlignedSpin(), Float64(mc), Float64(eta), Float64(chi1), Float64(chi2);
        f_min = f_start, GMsun_over_c3 = GMsun_over_c3)
    return Base.structdiff(b, NamedTuple{(:T,)})
end
_imr_pa(::SEOBNRv5IMRAlignedSpinF2, b) = seobnrv5alignedspin_postadiabatic_dynamics_fixed(b)

@inline function _imr_rhs4(::SEOBNRv5IMRAlignedSpinF2, r, prst, pphi, b)
    d = _seobv5_rhs(SEOBNRv5AlignedSpinState(r, 0.0, prst, pphi), b)
    return d.r, d.phi, d.prst, d.pphi
end

_imr_short_start(::SEOBNRv5IMRAlignedSpinF2, b) = (b.r0, b.pr0, b.pphi0)

# Plunge steps: up to 2 rad of orbital phase or 60 M per step, 3% of r per step near
# merger, and the stop located to 0.05 M by step halving. Over M_c 26-33, q 0.6-1, |chi| <= 0.4: at most 67 steps (the
# previous rule, 0.3 rad / 4% of r / 10 M, no halving: 134); waveform error vs an exactly integrated reference,
# PSD-weighted ||dh||^2/||h||^2 over 20-896 Hz, <= 7e-7 (previous rule: 2e-9); (2,2) dephasing over the last 1500 M
# <= 1e-4 rad. The dynamics' r, pr, pphi have no orbital oscillation (radiation-reaction time scale) and phi is a
# quadrature of Omega, so the step is accuracy-limited by the late plunge (kr), not by the orbital phase per step;
# higher-order tableaus (Tsitouras 5, Verner 6-8) gave no lower error per rhs evaluation. kr also sets the node spacing
# at merger, which the peak-aligned comparisons with pyseobnr are sensitive to (|h22| peak time median 0.02 M from the
# dense-node limit; previous rule: 0.04 M).
@inline _imr_plunge_steps(::SEOBNRv5IMRAlignedSpinF2) = (kph = 2.0, kphl = 2.0, om1 = 1.0, om2 = 2.0, kr = 0.03, hmin = 0.02,
    hmax = 60.0, hend = 0.05)

"""
    _seobv5_h22_abs(omega, omega_circ, Heff, H, nu, fc) -> Float64

Factorized `|h22|` (units M/dL): `T22 * Heff * hNewton22 * rho22^2`, the (2,2) block of
[`_seobv5_rr_force`](@ref) (rho22 from the shared [`_seobv5_rho_f`](@ref), tail factor magnitude,
`hNewton22 = absN22 * v_phi^2 * Y22` with `absN22 * Y22 = 8 sqrt(pi/5) nu`).
"""
@inline function _seobv5_h22_abs(omega, omega_circ, Heff, H, nu, fc)
    omega_safe = max(omega, eps(Float64))
    v = cbrt(omega_safe)
    v_phi = omega_safe / max(omega_circ, eps(Float64))^(2.0 / 3.0)
    rho22 = _seobv5_rho_f(v, log(v), nu * H * omega_safe, fc.rho).rho22
    hathatk2 = 2.0 * nu * H * omega_safe
    hathatksq4_2 = 4.0 * hathatk2 * hathatk2
    tlmprefac2 = sqrt(4.0 * pi * hathatk2 / (1.0 - exp(-4.0 * pi * hathatk2)))
    T22 = tlmprefac2 * sqrt((hathatksq4_2 + 1.0) * (hathatksq4_2 + 4.0)) / 2.0
    hNewton22 = (32.0 * pi * sqrt(6.0) / 15.0) * nu * v_phi * v_phi * 0.3862742020231896
    return T22 * Heff * hNewton22 * rho22 * rho22
end

@inline function _imr_amp(::SEOBNRv5IMRAlignedSpinF2, r, prst, pphi, om, b)
    H, Heff, om_circ = _seobv5_sym_state(r, prst, pphi, b.hp)
    return _seobv5_h22_abs(om, om_circ, Heff, H, b.nu, b.fc)
end

# Im log Gamma(x + i y) for x >= 3 (Stirling series after shifting by 6; |error| < 1e-12 here), in real arithmetic:
# Base's complex division / log made Enzyme's forward-mode type analysis fail on CUDA (IllegalTypeAnalysisException in
# `ssqs`), the reason the GravitationalWaveBayes device table once had its own copy.
@inline function _seobv5_arg_gamma(x, y)
    s = 0.0
    for k in 0:5
        s += atan(y, x + k)                    # Im log(x + k + i y)
    end
    wr = x + 6.0; wi = y
    m2 = wr * wr + wi * wi
    lnr = 0.5 * log(m2); th = atan(wi, wr)
    ir = wr / m2; ii = -wi / m2               # 1 / w
    i2r = ir * ir - ii * ii; i2i = 2.0 * ir * ii
    inr = 1.0 / 360.0 - i2r / 1260.0; ini = -i2i / 1260.0
    tr = i2r * inr - i2i * ini; ti = i2r * ini + i2i * inr
    ur = 1.0 / 12.0 - tr; ui = -ti
    lg_im = (wr - 0.5) * th + wi * lnr - wi + (ir * ui + ii * ur)
    return lg_im - s
end

"""
    _imr_phase_corr(::SEOBNRv5IMRAlignedSpinF2, r, prst, pphi, om, b)

The factorized (2,2) mode's phase beyond `-2 phi` (pyseobnr `compute_mode`): the tail phase
`2 hk log(4 k / sqrt(e)) + arg Gamma(3 - 2 i hk)` (`k = 2 Omega`, `hk = 2 Omega nu H`) plus `delta22`
(`waveform.pyx` `delta_coeffs[2,2]`: `7/3 vh^3 + (428 pi/105 - 4/3 (dm chiA + (1 - 2 nu) chiS)) vh^6 +
(-2203/81 + 1712 pi^2/315) vh^9 - 24 nu v^5`, `v = Omega^(1/3)`, `vh = (nu H Omega)^(1/3)`).
"""
@inline function _imr_phase_corr(::SEOBNRv5IMRAlignedSpinF2, r, prst, pphi, om, b)
    nu = b.nu
    H, _, _ = _seobv5_ham_xi(r, prst, pphi, b.hp)
    k = 2.0 * om
    hk = k * nu * H
    tail = 2.0 * hk * log(4.0 * k / sqrt(exp(1.0))) + _seobv5_arg_gamma(3.0, -2.0 * hk)
    v = cbrt(om)
    vh3 = nu * H * om
    d22 = 7.0 / 3.0 * vh3 + (428.0 * pi / 105.0 - 4.0 / 3.0 * (b.delta * b.chiA + b.chiS * (1.0 - 2.0 * nu))) * vh3^2 +
          (-2203.0 / 81.0 + 1712.0 * pi^2 / 315.0) * vh3^3 - 24.0 * nu * v^5
    return tail + d22
end

# The SEOBNRv5 dynamics are what the NQC fits were calibrated against: no phase-NQC shrinkage.
@inline _imr_nqc_phase_scale(::SEOBNRv5IMRAlignedSpinF2) = Inf

_seobv5f2_coeffs(model::SEOBNRv5IMRAlignedSpinF2, mc, eta, chi1, chi2; kwargs...) =
    _eobimr_coeffs(model, mc, eta, chi1, chi2; kwargs...)

function Phi(model::SEOBNRv5IMRAlignedSpinF2, f, mc, eta, chi1, chi2;
    f_min = 20.0, t0 = 0.0, phiRef = 0.0, GMsun_over_c3 = GMsun_over_c3, kwargs...)
    c = _eobimr_coeffs(model, mc, eta, chi1, chi2; f_min = f_min, GMsun_over_c3 = GMsun_over_c3)
    g(x) = _seobv5f2_phi_scalar(c, x) - Float64(t0) * x - Float64(phiRef)
    return f isa AbstractArray ? map(g, f) : g(f)
end

function Ampl(model::SEOBNRv5IMRAlignedSpinF2, f, mc, eta, chi1, chi2, dL;
    f_min = 20.0, GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc, kwargs...)
    c = _eobimr_coeffs(model, mc, eta, chi1, chi2; f_min = f_min, GMsun_over_c3 = GMsun_over_c3)
    g(x) = _seobv5f2_ampl_scalar(c, x, Float64(dL), Float64(GMsun_over_c2_Gpc))
    return f isa AbstractArray ? map(g, f) : g(f)
end
