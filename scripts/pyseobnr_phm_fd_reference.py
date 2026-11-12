"""
Reference frequency-domain SEOBNRv5PHM polarizations from pyseobnr 0.3.7 (GenerateWaveform, physical units, LAL
conventions) for test/seobnrv5phm_fd_tests.jl.

    python scripts/pyseobnr_phm_fd_reference.py OUTDIR [--production]

Per configuration: GenerateWaveform(...).generate_td_polarizations() at 8192 Hz from f22_start = f_ref = 20 Hz, a
Planck turn-on over the first four (2,2) cycles at 20 Hz (GravitationalWaveforms' `_eobimr_turn_on`, the same as the FD model's),
then the transform h(f) = dt sum_n h(t_n) e^{-2 pi i f t_n} (t = 0 at pyseobnr's epoch reference, the peak) at
f = 20 .. 512 Hz, df = 0.25 Hz. Columns: f, Re/Im h+, Re/Im hx. The precessing ODE at rtol = atol = 1e-11
(converged, see pyseobnr_ode_tol.py); --production keeps pyseobnr's 1e-8.
"""

import os
import sys

import numpy as np
import pyseobnr_ode_tol
from pyseobnr.generate_waveform import GenerateWaveform

CASES = {
    "a": (36.0, 29.0, (0.3, 0.2, 0.4), (-0.4, 0.1, 0.2), 1000.0, 0.7, 0.4),
    "b": (60.0, 10.0, (0.0, 0.7, -0.5), (0.4, -0.4, 0.4), 800.0, 0.3, 1.1),
}


def turn_on(x):
    # GravitationalWaveforms _eobimr_turn_on (Planck-type, C-infinity, 0 at x <= 0, 1 at x >= 1)
    x = np.clip(x, 0.0, 1.0)
    out = np.zeros_like(x)
    inside = (x > 0) & (x < 1)
    xi = x[inside]
    out[inside] = 1.0 / (1.0 + np.exp(1.0 / xi - 1.0 / (1.0 - xi)))
    out[x >= 1] = 1.0
    return out


def main(out):
    os.makedirs(out, exist_ok=True)
    f = np.arange(20.0, 512.0 + 1e-9, 0.25)
    for tag, (m1, m2, s1, s2, d, incl, ph) in CASES.items():
        p = dict(mass1=m1, mass2=m2, spin1x=s1[0], spin1y=s1[1], spin1z=s1[2], spin2x=s2[0], spin2y=s2[1],
                 spin2z=s2[2], distance=d, inclination=incl, phi_ref=ph, f22_start=20.0, f_ref=20.0, deltaT=1 / 8192,
                 approximant="SEOBNRv5PHM")
        n0 = pyseobnr_ode_tol.calls()
        hp, hc = GenerateWaveform(p).generate_td_polarizations()
        pyseobnr_ode_tol.check(n0)
        t = float(hp.epoch) + np.arange(len(hp.data.data)) * hp.deltaT
        w = turn_on((t - t[0]) / (4 / (2 * 20.0)))
        E = np.exp(-2j * np.pi * np.outer(f, t))
        Hp = E @ (hp.data.data * w) * hp.deltaT
        Hc = E @ (hc.data.data * w) * hp.deltaT
        hdr = f"m1={m1} m2={m2} chi1={list(s1)} chi2={list(s2)} dist={d} incl={incl} phi_ref={ph} ode_tol={pyseobnr_ode_tol.TOL}"
        np.savetxt(os.path.join(out, f"pyseobnr037_phmfd_{tag}.csv"), np.c_[f, Hp.real, Hp.imag, Hc.real, Hc.imag],
                   delimiter=",", fmt="%.10g", header=hdr)
        print(tag, len(t))


if __name__ == "__main__":
    pyseobnr_ode_tol.TOL = 1e-8 if "--production" in sys.argv else 1e-11
    main(sys.argv[1])
