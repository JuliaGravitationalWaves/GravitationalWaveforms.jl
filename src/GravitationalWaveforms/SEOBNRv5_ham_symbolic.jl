# The SEOBNRv5 / SEOBNRv5PHM Hamiltonian derivatives in the layouts the models use (pyseobnr's method layouts), from the
# generated SEOBNRv5_ham_symbolic_generated.jl (gen/symbolic/generate.jl, source gen/symbolic/hamiltonians/v5phm.jl).
# phi is cyclic: its derivatives are zero.

"""
    _seobv5phm_ham_grad(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)

pyseobnr `grad` layout: (dH/dr, dH/dphi = 0, dH/dprst, dH/dpphi).
"""
@inline function _seobv5phm_ham_grad(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)
    g = _seobv5phm_sym_grad(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)
    return (g[1], zero(g[1]), g[2], g[3])
end

"""
    _seobv5phm_ham_hessian(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)

pyseobnr `hessian` layout: the 16 entries of the 4x4 Hessian in (r, phi, prst, pphi), row-major.
"""
@inline function _seobv5phm_ham_hessian(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)
    h = _seobv5phm_sym_hessian(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)
    z = zero(h[1])
    return (h[1], z, h[2], h[3], z, z, z, z, h[4], z, h[5], h[6], h[7], z, h[8], h[9])
end

"""
    _seobv5phm_ham_dynamics(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)

pyseobnr `dynamics` layout: (dH/dr, dH/dphi = 0, dH/dprst, dH/dpphi, H, xi).
"""
@inline function _seobv5phm_ham_dynamics(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)
    d = _seobv5phm_sym_dynamics(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)
    return (d[1], zero(d[1]), d[2], d[3], d[4], d[5])
end

"""
    _seobv5phm_ham_omega(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)

pyseobnr `omega`: dH/dpphi.
"""
@inline _seobv5phm_ham_omega(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO) = _seobv5phm_sym_omega(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)

"""
    _seobv5phm_ham_auxderivs(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)

pyseobnr `auxderivs`: (dA/dr, dBnp/dr, dBnpa/dr, dxi/dr, dQ/dr, dQ/dprst, dHodd/dr, dBp/dr, dHeven/dr).
"""
@inline _seobv5phm_ham_auxderivs(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO) = _seobv5phm_sym_auxderivs(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu, X_1, X_2, delta, a6, dSO)
