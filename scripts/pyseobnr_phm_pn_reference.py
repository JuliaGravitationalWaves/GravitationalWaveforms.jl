"""
Reference data for the SEOBNRv5PHM post-Newtonian spin evolution (milestone P1) from pyseobnr 0.3.7.

    python scripts/pyseobnr_phm_pn_reference.py OUTDIR

Per configuration (m1 + m2 = 1, spins at omega_ref, start frequency omega_start; pyseobnr integrates from
0.9 * omega_start as SEOBNRv5PHM does with the postadiabatic inspiral):
  - pyseobnr037_phmpn_<tag>_rhs.csv: z (10) and prec_eqns_20102022(z) (10) at states along the solution (raw
    rows, including un-normalized LNhat from the solver).
  - pyseobnr037_phmpn_<tag>_sol.csv: combined_t, combined_y (10 columns) -- the DOP853 output nodes.
  - pyseobnr037_phmpn_<tag>_spl.csv: omega and the 20 spline columns of build_splines_PN ("everything":
    chi1_LN, chi2_LN, chi1_L, chi2_L, chi1_v(3), chi2_v(3), LN_hat(3), L_vec(3)) at 200 frequencies inside the
    PN range, plus the stored inputs in the header comment.
"""

import os
import sys

import numpy as np
from pyseobnr.eob.dynamics.pn_evolution_opt import (
    build_splines_PN,
    compute_quasiprecessing_PNdynamics_opt,
    prec_eqns_20102022,
)

CONFIGS = [
    # tag, q, chi1, chi2, omega_ref, omega_start
    ("q1.5_a", 1.5, (0.5, 0.1, 0.3), (-0.3, 0.2, 0.1), 0.02, 0.02),
    ("q3_b", 3.0, (0.8, -0.3, -0.4), (0.0, 0.5, 0.2), 0.025, 0.015),
    ("q1_c", 1.0, (0.99 * np.sin(0.5), 0.0, 0.99 * np.cos(0.5)), (0.99 * np.sin(2.5), 0.0, 0.99 * np.cos(2.5)),
     0.018, 0.018),
    ("q8_d", 8.0, (0.0, 0.7, -0.5), (0.4, -0.4, 0.4), 0.03, 0.03),
]


def main(out):
    os.makedirs(out, exist_ok=True)
    for tag, q, c1, c2, wref, wstart in CONFIGS:
        m1 = q / (1 + q)
        m2 = 1 / (1 + q)
        c1 = np.array(c1)
        c2 = np.array(c2)
        t, y = compute_quasiprecessing_PNdynamics_opt(wref, 0.9 * wstart, m1, m2, c1, c2)
        hdr = f"q={q!r} m1={m1!r} m2={m2!r} chi1={[float(x) for x in c1]!r} chi2={[float(x) for x in c2]!r} omega_ref={wref!r} omega_start={wstart!r}"
        np.savetxt(os.path.join(out, f"pyseobnr037_phmpn_{tag}_sol.csv"), np.c_[t, y], delimiter=",",
                   header=hdr, fmt="%.17g")
        nu = m1 * m2
        rows = []
        for i in np.linspace(0, len(t) - 1, 12).astype(int):
            z = y[i].copy()
            d = prec_eqns_20102022(t[i], z.copy(), nu, m1, m2, m1, m2)
            rows.append(np.r_[y[i], d])
        np.savetxt(os.path.join(out, f"pyseobnr037_phmpn_{tag}_rhs.csv"), np.array(rows), delimiter=",",
                   header=hdr, fmt="%.17g")
        spl = build_splines_PN(t, y, m1, m2, wstart)["everything"]
        om = y[:, -1]
        w = np.linspace(om[0], om[-1], 202)[1:-1]
        np.savetxt(os.path.join(out, f"pyseobnr037_phmpn_{tag}_spl.csv"), np.c_[w, spl(w)], delimiter=",",
                   header=hdr, fmt="%.17g")
        print(tag, "nodes", len(t), "omega", om[0], "->", om[-1], "t_end", t[-1])


if __name__ == "__main__":
    main(sys.argv[1])
