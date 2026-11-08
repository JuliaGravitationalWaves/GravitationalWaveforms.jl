"""
Reference data for SEOBNRv5HM (aligned spin, all default modes) from pyseobnr 0.3.7.

Run in an environment with pyseobnr 0.3.7 (conda-forge), e.g.
    python scripts/pyseobnr_hm_reference.py OUTDIR [--dense] [--pa]

`--pa`: pyseobnr's post-adiabatic inspiral (postadiabatic=True, the "analytic" variant GravitationalWaveforms ports) instead
of its default ODE inspiral; files get a `_pa` suffix.

Writes, per configuration (q, chi1, chi2), starting at orbital frequency omega_start = 0.02 (orbital phase 0
there, f_ref = f_start), dt = 0.1 M:
  - pyseobnr037_hm_q{q}_chi{c1}_{c2}.csv: t (M, 0 = peak of |h22|), then for each mode (2,2), (2,1), (3,3), (3,2),
    (4,4), (4,3): |h_lm| (M/r) and varphi_lm = -unwrap(angle(h_lm)) (h_lm = |h_lm| e^{-i varphi_lm}; unwrapped from
    the first sample, branch (-pi, pi]) at 43 fixed peak-relative times (the (2,2) test grid).
  - pyseobnr037_hm_q{q}_chi{c1}_{c2}_internals.txt: attachment, NQC and special coefficients, input values, QNMs.
  - (with --dense) the full time series of every mode, for development plots.
Plus qnm_table.csv: the complex QNM frequencies (l, m, n=0) of pyseobnr's `compute_QNM` (the `qnm` package's
interpolant, interp_only=True) on a dense spin grid, for fitting.
"""

import os
import sys

import numpy as np
import pyseobnr
from pyseobnr.eob.fits.EOB_fits import compute_QNM
from pyseobnr.eob.fits.IV_fits import InputValueFits
from pyseobnr.eob.fits.MR_fits import MergerRingdownFits
from pyseobnr.eob.utils.nr_utils import (
    bbh_final_mass_non_precessing_UIB2016,
    bbh_final_spin_non_precessing_HBR2016,
)
from pyseobnr.generate_waveform import generate_modes_opt

MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
TGRID = [-1500.0 + 100.0 * k for k in range(13)] + [-250.0 + 25.0 * k for k in range(9)] + \
        [-40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0] + [5.0 * k for k in range(13)]
CONFIGS = [
    (1.25, 0.0, 0.0),
    (1.0, -0.6, -0.6),
    (4.0, 0.3, 0.0),
    (3.0, 0.5, -0.4),
    (2.0, -0.8, 0.5),
    (8.0, 0.8, 0.2),
    (1.5, 0.9, 0.9),
    (6.0, -0.5, -0.7),
    (1.1, 0.3, -0.3),
]
MTSUN_SI = 4.925490947641267e-06


def tag(q, c1, c2):
    return f"q{q:.2f}_chi{c1}_{c2}"


def run(q, c1, c2, outdir, dense, pa=False):
    M = 60.0
    settings = dict(M=M, dt=0.1 * M * MTSUN_SI, return_modes=MODES)
    if pa:
        settings.update(postadiabatic=True, postadiabatic_type="analytic")
    sfx = "_pa" if pa else ""
    t, modes, model = generate_modes_opt(q, c1, c2, 0.02, approximant="SEOBNRv5HM", settings=settings, debug=True)
    h22 = modes["2,2"]
    ipk = int(np.argmax(np.abs(h22)))
    # peak by parabola through the 3 samples around the max (sub-sample alignment)
    a = np.abs(h22[ipk - 1:ipk + 2])
    dpk = 0.5 * (a[0] - a[2]) / (a[0] - 2 * a[1] + a[2])
    tpk = t[ipk] + dpk * (t[1] - t[0])
    tt = t - tpk
    cols = []
    for (l, m) in MODES:
        h = modes[f"{l},{m}"]
        vp = -np.unwrap(np.angle(h))
        vp0 = -np.angle(h[0])
        vp = vp - vp[0] + vp0
        cols.append((np.abs(h), vp))
    fn = os.path.join(outdir, f"pyseobnr037_hm_{tag(q, c1, c2)}{sfx}.csv")
    with open(fn, "w") as f:
        f.write(f"# pyseobnr {pyseobnr.__version__} SEOBNRv5HM{' (postadiabatic=True, analytic)' if pa else ''} q={q} chi1={c1} chi2={c2}: omega_start=0.02 (orbital phase 0), "
                f"dt=0.1M; columns t (M, 0 = |h22| peak), then |h_lm| (M/r), varphi_lm (h_lm = |h_lm| e^(-i varphi_lm)) for "
                f"modes {MODES}; duration from start to peak = {-tt[0]:.3f}\n")
        for tq in TGRID:
            row = [tq]
            for amp, vp in cols:
                row += [np.interp(tq, tt, amp), np.interp(tq, tt, vp)]
            f.write(",".join(repr(float(x)) for x in row) + "\n")
    if dense:
        arr = np.column_stack([tt] + [c for pair in cols for c in pair])
        np.savetxt(os.path.join(outdir, f"pyseobnr037_hm_{tag(q, c1, c2)}{sfx}_dense.csv"), arr[::5], delimiter=",")

    m1, m2 = q / (1 + q), 1 / (1 + q)
    iv = InputValueFits(m1, m2, [0.0, 0.0, c1], [0.0, 0.0, c2])
    mr = MergerRingdownFits(m1, m2, [0.0, 0.0, c1], [0.0, 0.0, c2])
    Mf = bbh_final_mass_non_precessing_UIB2016(m1, m2, c1, c2)
    af = bbh_final_spin_non_precessing_HBR2016(m1, m2, c1, c2, version="M3J4")
    fp = model.eob_pars.flux_params
    with open(os.path.join(outdir, f"pyseobnr037_hm_{tag(q, c1, c2)}{sfx}_internals.txt"), "w") as f:
        f.write(f"t_ISCO {float(model.t_ISCO)!r}\nt_attach {float(model.t_attach)!r}\nNR_deltaT {float(model.NR_deltaT)!r}\n")
        # returned t is uniform with dt = 0.1 M from the start of the dynamics
        t_pk_dyn = model.dynamics[0, 0] + (ipk + dpk) * 0.1
        f.write(f"r_ISCO {float(model.r_ISCO)!r}\nMf {float(Mf)!r}\naf {float(af)!r}\nt_peak22_dyn {float(t_pk_dyn)!r}\n")
        f.write(f"duration_start_to_peak {float(-tt[0])!r}\ndynamics_t_end {float(model.dynamics[-1, 0])!r}\n")
        for (l, m), pw in (((2, 1), 7), ((4, 3), 7), ((5, 5), 5)):
            f.write(f"special_f_coeff {l} {m} {pw} {float(fp.f_coeffs[l, m, pw])!r}\n")
        for k, v in model.nqc_coeffs.items():
            f.write(f"nqc {k[0]} {k[1]} " + " ".join(f"{n}={float(v[n])!r}" for n in ("a1", "a2", "a3", "b1", "b2")) + "\n")
        for name, d in (("habs", iv.habs()), ("hdot", iv.hdot()), ("hdotdot", iv.hdotdot()), ("omega", iv.omega()),
                        ("omegadot", iv.omegadot()), ("hsign", iv.hsign())):
            f.write(f"iv {name} " + " ".join(f"{k[0]}{k[1]}={float(d[k])!r}" for k in d) + "\n")
        for name in ("c1f", "c2f", "d1f", "d2f"):
            d = getattr(mr, name)()
            f.write(f"mr {name} " + " ".join(f"{k[0]}{k[1]}={float(d[k])!r}" for k in d) + "\n")
        for (l, m) in MODES + [(5, 5)]:
            w = compute_QNM(l, m, 0, af, Mf)
            f.write(f"qnm {l} {m} {float(w.real)!r} {float(w.imag)!r}\n")
    print("done", tag(q, c1, c2), "duration", -tt[0])


def qnm_table(outdir):
    import qnm
    a = np.concatenate([np.linspace(0.0, 0.99, 991), np.linspace(0.9901, 0.9999, 99)])
    with open(os.path.join(outdir, "qnm_table.csv"), "w") as f:
        f.write("# l, m (negative = the prograde-negative branch pyseobnr uses for af < 0), a, Re omega, Im omega "
                "(qnm package interp_only=True, M_f = 1)\n")
        for (l, m) in MODES + [(5, 5)]:
            for mm in (m, -m):
                c = qnm.modes_cache(s=-2, l=l, m=mm, n=0)
                for ai in a:
                    w, _, _ = c(a=ai, interp_only=True)
                    f.write(f"{l},{mm},{float(ai)!r},{float(w.real)!r},{float(w.imag)!r}\n")
    print("qnm table done")


if __name__ == "__main__":
    outdir = sys.argv[1]
    dense = "--dense" in sys.argv
    pa = "--pa" in sys.argv
    os.makedirs(outdir, exist_ok=True)
    for cfg in CONFIGS:
        run(*cfg, outdir, dense, pa)
    if not pa:
        qnm_table(outdir)
