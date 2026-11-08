# The binary parameters the aligned-spin SEOBNRv5 Hamiltonian reads. The models pass these to the generated call
# patterns (`_seobv5_sym_rhs(r, prst, pphi, hp)` etc., SEOBNRv5_ham_symbolic_generated.jl).

"""
    SEOBNRv5HamParams{T}(nu, X1, X2, chi1, chi2, a6, dSO)

SEOBNRv5HM (binary black hole): symmetric mass ratio, mass fractions, aligned dimensionless spins, the NR calibration
parameters a6 and dSO.
"""
struct SEOBNRv5HamParams{T}
    nu::T
    X1::T
    X2::T
    chi1::T
    chi2::T
    a6::T
    dSO::T
end
