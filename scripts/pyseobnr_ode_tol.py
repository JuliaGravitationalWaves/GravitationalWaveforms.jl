"""
pyseobnr's precessing ODE tolerance for the reference scripts: SEOBNRv5PHM_opt (and so GenerateWaveform) hard-codes
rtol = atol = 1e-8 for `compute_dynamics_prec_opt` (with 1e-11 / 1e-12 commented out). At 1e-8, with the spins
updated once per accepted step, its merger phase is 1e-3 .. 1.5e-2 rad from its converged result; GravitationalWaveforms' plunge
converges to the latter. Importing this module rebinds `compute_dynamics_prec_opt` to a wrapper that sets the
tolerance, in the integrator module and in the compiled PA module (Cython looks the name up in its module dict at call
time). `check(n0)` asserts the wrapper ran since `calls()` was n0.

    import pyseobnr_ode_tol
    pyseobnr_ode_tol.TOL = 1e-8     # pyseobnr's production value (default: 1e-11, converged)
"""

import pyseobnr.eob.dynamics.integrate_ode_prec as _iop
import pyseobnr.eob.dynamics.postadiabatic_C_prec as _pa

TOL = 1e-11
_orig = _iop.compute_dynamics_prec_opt
_calls = [0]


def _with_tol(*a, **k):
    _calls[0] += 1
    k["rtol"], k["atol"] = TOL, TOL
    return _orig(*a, **k)


_iop.compute_dynamics_prec_opt = _with_tol
_pa.compute_dynamics_prec_opt = _with_tol


def calls():
    return _calls[0]


def check(n0):
    assert _calls[0] > n0, "the ODE tolerance wrapper was not used"
