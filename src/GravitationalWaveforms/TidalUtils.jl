"""
    _tidal_quadparam_octparam(lambda) -> (quadparam, octparam)

Spin-induced quadrupole/octupole parameters as a function of the dimensionless tidal
deformability `lambda`, following `LALSimUniversalRelations.c`'s universal-relation fits
(Yagi 2014 / Yagi & Yunes 2017), transcribed via ripple's `IMRPhenom_tidal_utils.py`
(`get_quadparam_octparam`). `quadparam == 1` exactly at `lambda == 0` (the black-hole limit,
`C_Q == 1` for a Kerr BH) and departs from `1` for a neutron star; `octparam` is the analogous
octupole coefficient.

Used by [`TaylorF2`](@ref)'s spin-squared phasing terms (evaluated even for `lambda == 0`,
since the point-particle "self-spin"/"quadrupole-monopole" PN terms need `quadparam == 1` for a
BH). A separate low-`lambda`
(`< 1`) polynomial fit is smoothly joined to the main high-`lambda` universal relation so the
`lambda -> 0` limit is exactly `quadparam -> 1` (the raw high-`lambda` fit, a fit to `log(lambda)`,
is singular at `lambda == 0`) -- same two-branch structure LALSuite/ripple use, selected with
`ifelse` (not a real branch) for GPU/Enzyme safety, matching this package's established
zero-allocation, branchless-control-flow convention.
"""
@inline function _tidal_quadparam_octparam(lambda)
    T = typeof(lambda)
@Tlit T begin
    lam = max(lambda, zero(T))
    is_low = lam < 1.0

    quadparam_low = 1.0 + lam * (0.427688866723244 + lam * (-0.324336526985068 + lam * 0.1107439432180572))
    log_quadparam_low = log(quadparam_low)
    log_octparam_low = 0.003131 + 2.071 * log_quadparam_low - 0.7152 * log_quadparam_low^2 +
                        0.2458 * log_quadparam_low^3 - 0.03309 * log_quadparam_low^4
    octparam_low = exp(log_octparam_low)

    log_lam = log(max(lam, eps(T)))
    log_quadparam_high = 0.1940 + 0.09163 * log_lam + 0.04812 * log_lam^2 -
                          4.283e-3 * log_lam^3 + 1.245e-4 * log_lam^4
    quadparam_high = exp(log_quadparam_high)
    log_octparam_high = 0.003131 + 2.071 * log_quadparam_high - 0.7152 * log_quadparam_high^2 +
                         0.2458 * log_quadparam_high^3 - 0.03309 * log_quadparam_high^4
    octparam_high = exp(log_octparam_high)

    quadparam = ifelse(is_low, quadparam_low, quadparam_high)
    octparam = ifelse(is_low, octparam_low, octparam_high)
    return quadparam, octparam
end
end
