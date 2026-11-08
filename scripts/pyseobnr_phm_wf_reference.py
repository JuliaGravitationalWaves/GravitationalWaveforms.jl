"""
Reference data for the SEOBNRv5PHM waveform (milestone P3) from pyseobnr 0.3.7.

    python scripts/pyseobnr_phm_wf_reference.py OUTDIR [--antisym] [--full | --test] [--production]

Per configuration (m1 + m2 = 1, spins at omega_ref; the precessing PA dynamics as GenerateWaveform),
the model class SEOBNRv5PHM_opt with polarizations_from_coprec = True at the given inclination and phiref (the
model's setting; GenerateWaveform passes pi/2 - phi_ref), 0.1 M sampling:
  - pyseobnr037_phmwf[_as]_<tag>_info.csv: one row of scalars (see the header): t_attach, t_ISCO, NR_deltaT,
    final mass and spin, sign, Euler angles at attachment, J_hat, L_hat at attachment, t0 (model.t[0]).
  - pyseobnr037_phmwf[_as]_<tag>_wf.csv: dynamics time (model.t - model.t[0]), Re/Im of hpc, Re/Im of the six
    co-precessing modes (m > 0, order (2,2), (2,1), (3,3), (3,2), (4,4), (4,3); the symmetric parts with --antisym,
    followed by Re/Im of the anti-symmetric (2,2), (3,3), (4,4)); every 20th sample (2 M) up to 400 M before the
    attachment, every sample after (or every sample with --full).
--antisym enables the anti-symmetric modes (pyseobnr's default); files get the _as tag. --test writes the
compact test data (test/data/phm_wf: three configurations, every 100th sample up to 400 M before the attachment,
every 10th after, 10 significant digits). The precessing ODE runs at rtol = atol = 1e-11 (converged, see
pyseobnr_ode_tol.py); --production keeps pyseobnr's 1e-8.
"""

import os
import sys

import numpy as np
import pyseobnr_ode_tol
from pyseobnr.eob.hamiltonian.Ham_AvgS2precess_simple_cython_PA_AD import Ham_AvgS2precess_simple_cython_PA_AD
from pyseobnr.eob.waveform.waveform import SEOBNRv5RRForce
from pyseobnr.models.SEOBNRv5HM import SEOBNRv5PHM_opt

CONFIGS = [
    # tag, q, chi1, chi2, omega_ref, omega_start, inclination, phiref
    ("q1.5_a", 1.5, (0.5, 0.1, 0.3), (-0.3, 0.2, 0.1), 0.018, 0.018, 0.7, 0.4),
    ("q3_b", 3.0, (0.8, -0.3, -0.4), (0.0, 0.5, 0.2), 0.015, 0.015, 1.2, 2.0),
    ("q8_d", 8.0, (0.0, 0.7, -0.5), (0.4, -0.4, 0.4), 0.02, 0.02, 0.3, -1.0),
    ("q2_e", 2.0, (0.1, 0.0, 0.85), (0.0, 0.1, 0.6), 0.011, 0.011, 2.5, 0.0),
    ("q1.2_g", 1.2, (-0.6, -0.5, -0.4), (0.3, 0.6, -0.2), 0.016, 0.016, 1.0, 1.0),
    ("q3_fref", 3.0, (0.8, -0.3, -0.4), (0.0, 0.5, 0.2), 0.022, 0.015, 1.2, 2.0),
    ("q2_lowref", 2.0, (0.4, 0.4, 0.2), (-0.5, 0.0, 0.3), 0.009, 0.011, 0.9, 0.3),
]
MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
ODE_TOL = pyseobnr_ode_tol.TOL = 1e-8 if "--production" in sys.argv else 1e-11


def run(tag, q, c1, c2, wref, wstart, incl, phiref, antisym):
    model = SEOBNRv5PHM_opt(q, *c1, *c2, wstart, Ham_AvgS2precess_simple_cython_PA_AD, SEOBNRv5RRForce(),
                            omega_ref=wref,
                            settings={"postadiabatic": True, "polarizations_from_coprec": True,
                                      "enable_antisymmetric_modes": antisym, "inclination": incl, "phiref": phiref,
                                      "M": 50.0})
    n0 = pyseobnr_ode_tol.calls()
    model()
    pyseobnr_ode_tol.check(n0)
    return model


def main(out, antisym, full, test):
    os.makedirs(out, exist_ok=True)
    pre = "pyseobnr037_phmwf_as" if antisym else "pyseobnr037_phmwf"
    s_insp, s_mr, fmt = (100, 10, "%.10g") if test else (20, 1, "%.12g")
    for tag, q, c1, c2, wref, wstart, incl, phiref in CONFIGS:
        if test and tag not in ("q1.5_a", "q8_d", "q3_fref"):
            continue
        model = run(tag, q, c1, c2, wref, wstart, incl, phiref, antisym)
        m1, m2 = q / (1 + q), 1 / (1 + q)
        hdr = (f"q={q!r} m1={m1!r} m2={m2!r} chi1={list(c1)!r} chi2={list(c2)!r} omega_ref={wref!r} "
               f"omega_start={wstart!r} inclination={incl!r} phiref={phiref!r} ode_tol={ODE_TOL!r}")
        t = model.t - model.t[0]
        ea = model.euler_angles_attach
        info = [model.t_attach, model.t_ISCO, model.NR_deltaT, model.final_mass, model.final_spin,
                model._sign_final_spin, float(ea[0]), float(ea[1]), float(ea[2]), *model.Jfhat_attach,
                *model.Lvec_hat_attach, model.t[0], model.omega_attach]
        if antisym:
            iv, mrd = model.ivs_asym, model.mrd_ivs
            info += [iv["amp"][(2, 2)], iv["omega"][(2, 2)], iv["omega"][(3, 3)], iv["omega"][(4, 4)],
                     mrd["c1f"][(2, 2)], mrd["c2f"][(2, 2)]]
        np.savetxt(os.path.join(out, f"{pre}_{tag}_info.csv"), np.array([info]), delimiter=",", fmt="%.17g",
                   header=hdr + " | t_attach,t_ISCO,NR_deltaT,Mf,af,sign,alpha_a,beta_a,gamma_a,Jhat(3),Lhat(3),"
                                "t0,omega_attach[,asym: amp22,omega22,omega33,omega44,c1f22,c2f22]")
        if full:
            keep = np.arange(len(t))
        else:
            i400 = np.searchsorted(t, model.t_attach - 400.0)
            keep = np.r_[np.arange(0, i400, s_insp), np.arange(i400, len(t), s_mr)]
        cols = [t[keep], model.hpc.real[keep], model.hpc.imag[keep]]
        sym = model.symmetric_modes_full if antisym else model.imr_full
        for lm in MODES:
            cols += [sym[lm].real[keep], sym[lm].imag[keep]]
        if antisym:
            for lm in [(2, 2), (3, 3), (4, 4)]:
                a = model.antisymmetric_modes_full[lm]
                cols += [a.real[keep], a.imag[keep]]
        np.savetxt(os.path.join(out, f"{pre}_{tag}_wf.csv"), np.column_stack(cols), delimiter=",", fmt=fmt,
                   header=hdr)
        print(tag, "samples", len(t), "kept", len(keep), "t_attach", model.t_attach, "af", model.final_spin,
              "Mf", model.final_mass)


if __name__ == "__main__":
    main(sys.argv[1], "--antisym" in sys.argv, "--full" in sys.argv, "--test" in sys.argv)
