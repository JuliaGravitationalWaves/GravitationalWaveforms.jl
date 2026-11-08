"""
Reference data for the SEOBNRv5PHM quasi-precessing EOB dynamics (milestone P2) from pyseobnr 0.3.7.

    python scripts/pyseobnr_phm_dyn_reference.py OUTDIR [--production]

Files (m1 + m2 = 1):
  - pyseobnr037_phmham.csv: the v5PHM Hamiltonian (`Ham_AvgS2precess_simple_cython_PA_AD`) at random phase-space
    points. Columns: 19 inputs (r, prst, pphi, chi1_v (3), chi2_v (3), chi1_LN, chi2_LN, chi1_L, chi2_L, nu, X1,
    X2, delta, a6, dSO), then _call (9), grad (4), dynamics (6), omega (1), auxderivs (9), csi (1), hessian (16).
  - pyseobnr037_phmdyn_<tag>_ic.csv: the circular initial conditions at omega_start (computeIC_augm): r0, pphi0,
    pr0 and the spins used.
  - pyseobnr037_phmdyn_<tag>_pa.csv: the postadiabatic dynamics (compute_postadiabatic_dynamics, analytic, order 8)
    t, r, phi, pr, pphi, H, omega, omega_circ, chi1_LN, chi2_LN, and omega_pa (11 columns).
  - pyseobnr037_phmdyn_<tag>_ode.csv: the ODE continuation (compute_dynamics_prec_opt from the PA end, times shifted
    to the PA clock) dynamics_low and dynamics_fine (every 20th row of pyseobnr's 0.1 M resampling, plus its last row) stacked,
    10 columns as above, plus a last column 0/1 = low/fine.
  - the header line of every per-binary file stores the inputs.

The postadiabatic pieces are called exactly as `compute_combined_dynamics_exp_v1` calls them (the precessing PA
default of `GenerateWaveform`). The ODE runs at rtol = atol = 1e-11 (converged: pyseobnr's production 1e-8, with its
spins updated once per accepted step, leaves the merger phase 1e-3 .. 1.5e-2 rad away; GravitationalWaveforms' RK4 plunge
converges to the 1e-11 result); --production uses pyseobnr's 1e-8.
"""

import os
import re
import sys

import numpy as np
from pyseobnr.eob.dynamics.initial_conditions_aligned_precessing import computeIC_augm
from pyseobnr.eob.dynamics.integrate_ode_prec import compute_dynamics_prec_opt
from pyseobnr.eob.dynamics.pn_evolution_opt import build_splines_PN, compute_quasiprecessing_PNdynamics_opt
from pyseobnr.eob.dynamics.postadiabatic_C_prec import compute_postadiabatic_dynamics
from pyseobnr.eob.fits.fits_Hamiltonian import dSO as dSO_poly_fit
from pyseobnr.eob.fits.GSF_fits import GSF_amplitude_fits
from pyseobnr.eob.hamiltonian.Ham_AvgS2precess_simple_cython_PA_AD import Ham_AvgS2precess_simple_cython_PA_AD
from pyseobnr.eob.waveform.waveform import SEOBNRv5RRForce
from pyseobnr.models.SEOBNRv5HM import SEOBNRv5PHM_opt

CONFIGS = [
    # tag, q, chi1, chi2, omega_ref, omega_start  (GenerateWaveform: omega = pi f M MTSUN; f22_start 0.9 f_ref)
    ("q1.5_a", 1.5, (0.5, 0.1, 0.3), (-0.3, 0.2, 0.1), 0.02, 0.018),
    ("q3_b", 3.0, (0.8, -0.3, -0.4), (0.0, 0.5, 0.2), 0.025, 0.015),
    ("q1_c", 1.0, (0.99 * np.sin(0.5), 0.0, 0.99 * np.cos(0.5)), (0.99 * np.sin(2.5), 0.0, 0.99 * np.cos(2.5)),
     0.018, 0.016),
    ("q8_d", 8.0, (0.0, 0.7, -0.5), (0.4, -0.4, 0.4), 0.03, 0.012),
    ("q2_e", 2.0, (0.1, 0.0, 0.85), (0.0, 0.1, 0.6), 0.012, 0.011),
    ("q4_f", 4.0, (-0.6, 0.5, 0.3), (0.2, -0.7, -0.5), 0.0075, 0.0065),
]


def setup_model(q, c1, c2, wref, wstart):
    model = SEOBNRv5PHM_opt(q, *c1, *c2, wstart, Ham_AvgS2precess_simple_cython_PA_AD, SEOBNRv5RRForce(),
                            omega_ref=wref, settings={"postadiabatic": True, "polarizations_from_coprec": True})
    model._initialize_params(phys_pars=model.phys_pars)
    model._set_H_coeffs()
    gsf = GSF_amplitude_fits(model.nu)
    for key, val in gsf.items():
        t = re.findall(r"h(\d)(\d)_v(\d+)", key)
        if t:
            l, m, v = [int(x) for x in t[0]]
            model.eob_pars.flux_params.extra_coeffs[l, m, v] = val
        else:
            t = re.findall(r"h(\d)(\d)_vlog(\d+)", key)
            if t:
                l, m, v = [int(x) for x in t[0]]
                model.eob_pars.flux_params.extra_coeffs_log[l, m, v] = val
    return model


def ham_rows(rng, n=60):
    rows = []
    for _ in range(n):
        q = rng.uniform(1, 10)
        c1 = rng.normal(size=3); c1 *= rng.uniform(0, 0.99) / np.linalg.norm(c1)
        c2 = rng.normal(size=3); c2 *= rng.uniform(0, 0.99) / np.linalg.norm(c2)
        model = setup_model(q, c1, c2, 0.02, 0.02)
        H = model.H
        pp = model.eob_pars.p_params
        nu, X1, X2, delta = pp.nu, pp.X_1, pp.X_2, pp.delta
        lN = rng.normal(size=3); lN /= np.linalg.norm(lN)
        lL = lN + 0.05 * rng.normal(size=3); lL /= np.linalg.norm(lL)
        chi1LN, chi2LN, chi1L, chi2L = c1 @ lN, c2 @ lN, c1 @ lL, c2 @ lL
        dso = dSO_poly_fit(nu, X1 * chi1LN + X2 * chi2LN, X1 * chi1LN - X2 * chi2LN) + 0.3 * rng.normal()
        H.calibration_coeffs.dSO = dso
        a6 = H.calibration_coeffs.a6
        r = rng.uniform(1.8, 40.0)
        prst = rng.uniform(-0.6, 0.05) if r < 6 else rng.uniform(-0.05, 0.01)
        pphi = np.sqrt(r) * rng.uniform(0.8, 1.2) if r > 4 else rng.uniform(2.0, 4.5)
        qv = np.array([r, 0.0])
        pv = np.array([prst, pphi])
        args = (qv, pv, tuple(c1), tuple(c2), pp.m_1, pp.m_2, chi1LN, chi2LN, chi1L, chi2L)
        out = list(H._call(*args)) + list(H.grad(*args)) + list(H.dynamics(*args)) + [H.omega(*args)] + \
            list(H.auxderivs(*args)) + [H.csi(*args)] + list(np.asarray(H.hessian(*args)).ravel())
        rows.append([r, prst, pphi, *c1, *c2, chi1LN, chi2LN, chi1L, chi2L, nu, X1, X2, delta, a6, dso] + out)
    return np.array(rows)


def dyn_files(out, tag, q, c1, c2, wref, wstart, tol):
    c1 = np.array(c1); c2 = np.array(c2)
    model = setup_model(q, c1, c2, wref, wstart)
    H, RR, params = model.H, model.RR, model.eob_pars
    m1, m2 = model.m_1, model.m_2
    hdr = (f"q={q!r} m1={m1!r} m2={m2!r} chi1={[float(x) for x in c1]!r} chi2={[float(x) for x in c2]!r} "
           f"omega_ref={wref!r} omega_start={wstart!r} a6={H.calibration_coeffs.a6!r} ode_tol={tol!r}")
    # == compute_combined_dynamics_exp_v1 ==
    combined_t, combined_y = compute_quasiprecessing_PNdynamics_opt(wref, 0.9 * wstart, m1, m2, c1, c2)
    omegaPN_f = combined_y[:, -1][-1]
    splines = build_splines_PN(combined_t, combined_y, m1, m2, wstart)
    # initial conditions as compute_postadiabatic_dynamics computes them (spins at omega_start)
    tmp = splines["everything"](wstart)
    pp = params.p_params
    pp.omega = wstart
    pp.chi1_v, pp.chi2_v = tmp[4:7], tmp[7:10]
    pp.chi_1, pp.chi_2 = tmp[0], tmp[1]
    pp.chi1_L, pp.chi2_L = tmp[2], tmp[3]
    pp.update_spins(tmp[0], tmp[1])
    X1, X2 = pp.X_1, pp.X_2
    H.calibration_coeffs.dSO = dSO_poly_fit(pp.nu, tmp[0] * X1 + tmp[1] * X2, tmp[0] * X1 - tmp[1] * X2)
    r0, pphi0, pr0 = computeIC_augm(wstart, H, RR, tmp[4:7], tmp[7:10], m1, m2, params=params)
    np.savetxt(os.path.join(out, f"pyseobnr037_phmdyn_{tag}_ic.csv"), np.array([[r0, pphi0, pr0, *tmp[:10]]]),
               delimiter=",", header=hdr + f" omegaPN_f={omegaPN_f!r}", fmt="%.17g")
    pa, omega_pa = compute_postadiabatic_dynamics(wref, wstart, H, RR, c1, c2, m1, m2, splines, combined_t, combined_y,
                                                  tol=1e-12, params=params, order=8, postadiabatic_type="analytic",
                                                  window_length=10)
    np.savetxt(os.path.join(out, f"pyseobnr037_phmdyn_{tag}_pa.csv"), np.c_[pa, omega_pa], delimiter=",",
               header=hdr, fmt="%.17g")
    ode_low, ode_high, _, _ = compute_dynamics_prec_opt(wref, omega_pa[-1], omegaPN_f, H, RR, m1, m2, splines,
                                                        combined_t, combined_y, params, rtol=tol, atol=tol,
                                                        step_back=250.0, y_init=pa[-1, 1:5])
    ode_low[:, 0] += pa[-1, 0]
    ode_high[:, 0] += pa[-1, 0]
    # the fine part is a 0.1 M resampling of the last ~250 M; keep every 20th row and its last row
    keep = np.r_[np.arange(0, len(ode_high), 20), len(ode_high) - 1]
    ode = np.r_[np.c_[ode_low, np.zeros(len(ode_low))], np.c_[ode_high[np.unique(keep)], np.ones(len(np.unique(keep)))]]
    np.savetxt(os.path.join(out, f"pyseobnr037_phmdyn_{tag}_ode.csv"), ode, delimiter=",", header=hdr, fmt="%.17g")
    print(tag, "r0", r0, "PA nodes", len(pa), "r", pa[0, 1], "->", pa[-1, 1], "ODE low/fine", len(ode_low),
          len(ode_high), "t_end", ode[-1, 0], "omegaPN_f", omegaPN_f)


def main(out, tol):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261007)
    np.savetxt(os.path.join(out, "pyseobnr037_phmham.csv"), ham_rows(rng), delimiter=",", fmt="%.17g",
               header="r,prst,pphi,chi1v(3),chi2v(3),chi1LN,chi2LN,chi1L,chi2L,nu,X1,X2,delta,a6,dSO | call(9) "
                      "grad(4) dynamics(6) omega auxderivs(9) csi hessian(16)")
    for cfg in CONFIGS:
        dyn_files(out, *cfg, tol)


if __name__ == "__main__":
    main(sys.argv[1], 1e-8 if "--production" in sys.argv else 1e-11)
