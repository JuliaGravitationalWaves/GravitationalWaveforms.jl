"""
Absolute-convention reference for `hphc(SEOBNRv5HM(), ...)`: pyseobnr 0.3.7's own frequency-domain polarizations
(`GenerateWaveform(...).generate_fd_polarizations()`, the LAL convention: TD modes -> -2Y_lm(iota, pi/2 - phi_ref)
-> conditioned FFT) at two configurations with strong odd-m modes.

Run in an environment with pyseobnr 0.3.7 (conda-forge), e.g.
    python scripts/pyseobnr_hm_polarizations.py OUTDIR

Writes pyseobnr037_hm_fdpol.csv: one header line per configuration
    # tag m1 m2 chi1 chi2 inclination phi_ref
then rows `tag, f, Re hp, Im hp, Re hc, Im hc` (distance 1 Gpc, f_22 start = f_ref = 15 Hz, deltaF = 1/8 s, kept every
4th bin over 20-512 Hz). pyseobnr's FFT has its own time origin (conditioning), so the test maximises over a time
shift only, never over phase: the phase at the peak tests the mode-sign / phi_ref convention.
"""

import os
import sys

import numpy as np
from pyseobnr.generate_waveform import GenerateWaveform

CONFIGS = [  # tag, m1, m2, chi1, chi2, inclination, phi_ref
    ("a", 48.0, 12.0, 0.3, -0.1, np.pi / 3, 0.7),
    ("b", 60.0, 10.0, -0.4, 0.6, 0.9, 2.3),
]

out = sys.argv[1]
lines = []
for tag, m1, m2, c1, c2, inc, phi in CONFIGS:
    p = {"mass1": m1, "mass2": m2, "spin1z": c1, "spin2z": c2, "deltaT": 1 / 4096, "deltaF": 0.125, "f_max": 1024.0,
         "f22_start": 15.0, "f_ref": 15.0, "phi_ref": phi, "distance": 1000.0, "inclination": inc,
         "approximant": "SEOBNRv5HM"}
    hp, hc = GenerateWaveform(p).generate_fd_polarizations()[:2]
    f = hp.deltaF * np.arange(hp.data.length)
    keep = np.nonzero((f >= 20.0) & (f <= 512.0))[0][::4]
    lines.append(f"# {tag} {m1!r} {m2!r} {c1!r} {c2!r} {inc!r} {phi!r}")
    for i in keep:
        a, b = hp.data.data[i], hc.data.data[i]
        lines.append(",".join([tag] + [f"{x:.17g}" for x in (f[i], a.real, a.imag, b.real, b.imag)]))
with open(os.path.join(out, "pyseobnr037_hm_fdpol.csv"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
