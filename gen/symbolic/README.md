# gen/symbolic: symbolic EOB code generation (dev-only)

Symbolics.jl is a dependency of this environment only, never of GravitationalWaveforms. The generator writes plain Julia source
into `src/`, which is committed; the package never loads Symbolics.

    julia +1.13.2 --project=gen/symbolic -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
    julia +1.13.2 --project=gen/symbolic gen/symbolic/generate.jl         # regenerate the files below
    julia +1.13.2 --project=gen/symbolic gen/symbolic/generate.jl check   # fail if a committed file is stale

- `hamiltonians/v5phm.jl`: **the** SEOBNRv5 / SEOBNRv5PHM Hamiltonian (pyseobnr 0.3.7 `Hamiltonian_v5PHM._call`, which
  reduces to the aligned-spin `evaluate_H` for spins along L). Straight-line Julia, named potentials; edit here, then regenerate. It is also valid Julia on its own (`include` it to evaluate it).
- `generate.jl` writes `src/GravitationalWaveforms/SEOBNRv5_ham_symbolic_generated.jl`: the precessing `_seobv5phm_ham_call`,
  the aligned-spin `_seobv5_potentials` (the source with zero in-plane spins) and their derivatives: gradients,
  Hessians, the pyseobnr method layouts, and one fused function per call pattern of the models (equations of motion
  with omega_circ, post-adiabatic equations, initial-condition Newton steps, ...).
  `src/GravitationalWaveforms/SEOBNRv5_ham_symbolic.jl` maps them onto the layouts the models call. The aligned-spin call
  patterns come from one table (`PATTERNS`), followed by dispatch methods on `SEOBNRv5HamParams`: the models call
  `_seobv5_sym_rhs(r, prst, pphi, hp)` etc.
- `symeob.jl`: Julia function body -> Symbolics DAG -> memoised derivatives -> straight-line Julia (one local per distinct
  subterm, tuple return). Handles scalar and small-vector (`SVector`, `cross`, `dot`, `norm`) straight-line code.

Tests: `test/seobnrv5_ham_symbolic_tests.jl` (derivatives vs 256-bit central differences, aligned = precessing at zero
in-plane spin, fused sets); the values against pyseobnr in the SEOBNRv5 / SEOBNRv5PHM test files.
