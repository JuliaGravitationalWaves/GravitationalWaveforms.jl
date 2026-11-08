using StaticArrays: FieldVector

export SEOBNRv5AlignedSpin, SEOBNRv5AlignedSpinState, validate_seobnrv5alignedspin_params,
    seobnrv5alignedspin_rk4_integrate

"""
    SEOBNRv5AlignedSpin

A port of the aligned-spin
SEOBNRv5HM effective-one-body Hamiltonian, radiation-reaction force, and two-stage initial-condition
solve from `pyseobnr` (v0.3.6), architecturally mirroring `AlignedSpinPNHam.jl` (`(r,phi,pr,pphi)`
`FieldVector` state, allocation-free hand-differentiated RHS, `_coeffs`/`_scalar` split, fixed-step
RK4) but replacing that file's bare-PN Hamiltonian with pyseobnr's real, calibrated, resummed
EOB Hamiltonian.

# Provenance -- what was ported from where

`pyseobnr`'s binary distribution ships the Hamiltonian only as compiled Cython `.so` files; the
actual source was read from a shallow clone of `https://github.com/AEI-ACR/pyseobnr` (tag/commit
matching the installed v0.3.6), specifically:

- `pyseobnr/eob/hamiltonian/Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C.pyx` -- the Hamiltonian value
  (`evaluate_H`, lines 23-168) and its **already-analytically-differentiated** gradient (the `grad`
  cpdef method, lines 340-591) and tortoise factor (`csi`, lines 1285-1325; `dynamics`, lines
  1327-1580, which just bundles `grad`+`H`+`xi` into one 6-tuple).
- `pyseobnr/eob/dynamics/rhs_aligned.pyx`'s `get_rhs` (lines 17-60) -- the tortoise-factored RHS
  assembly.
- `pyseobnr/eob/waveform/waveform.pyx` -- the `RadiationReactionForce`/`RR_force` class (lines
  2246-2308) and the factorized-resummed multipolar flux machinery it calls (`compute_flux` lines
  1515-1591, `compute_tail` lines 345-377, `calculate_multipole_prefix` lines 209-291,
  `EOBFluxCalculateNewtonianMultipoleAbs` lines 1473-1489, `compute_rholm_single` lines 1390-1444,
  `compute_extra_flm_terms` lines 1371-1381, and -- as of the 7-mode `_seobv5_rr_force` extension,
  superseding the original port's `(2,2)`-mode-only branch of `compute_rho_coeffs` -- all seven
  modes' branches of `compute_rho_coeffs` (lines 386-1156): `(2,2)` lines 432-480, `(2,1)` lines
  489-600, `(3,3)` lines 607-663, `(3,2)` lines 669-693 (base) + 1101-1130 (`extra_PN_terms`),
  `(4,4)` lines 724-761, `(4,3)` lines 765-778 (base) + 1132-1148 (`extra_PN_terms`), `(5,5)` lines
  822-873).
- `pyseobnr/eob/fits/fits_Hamiltonian.py` -- the `a6_NS(nu)` (lines 172-188) and `dSO(nu,ap,am)`
  (lines 90-127) calibration-coefficient fits, both plain closed-form polynomials with an
  auditable machine-generation provenance header (`polynomial_fit_v4.py`, 2022-09-01, cited git
  revision and posterior-data file).
- `pyseobnr/eob/dynamics/initial_conditions_aligned_opt.py` -- the two-stage `IC_cons`/`IC_diss`
  initial-condition solve (lines 14-83), cited there to Eqs(60)/(68) of Khalil, Buonanno, Steinhoff
  & Vines, Phys. Rev. D 104, 024046 (2021) [`Khalil2021`].
- `pyseobnr/eob/utils/containers.pyx` lines 43-49 -- the derived-spin-quantity definitions
  (`chi_S`, `chi_A`, the "test-particle spin" `a`, `ap`, `am`) needed by the flux's `rho22` series.
- `pyseobnr/eob/fits/GSF_fits.py` (93 lines) -- `GSF_amplitude_fits(nu)`, the 2nd-order
  gravitational-self-force amplitude-fit corrections to every mode's `rho_lm` (added during Task
  4i', not the original port -- see "Radiation-reaction force" below for the discovery story).

The two public LIGO DCC PDFs the `.pyx` docstrings themselves cite for the closed-form equations
(`"The Hamiltonian is given by Eq(14) in [SEOBNRv5HM] doc..."`, `rhs_aligned.pyx`'s own docstring)
were fetched directly (`https://dcc.ligo.org/public/0186/T2300060/002/SEOBNRv5HM.pdf`, LIGO
Document T2300060-v2, "Note on the SEOBNRv5HM waveform model", Pompili et al. 2023) and read in
full for this port -- **every equation number cited below (Eq. 7, 14, 16-19, 43, 55) is from that
PDF, confirmed directly, not guessed from the arXiv summary paper** (arXiv:2303.18039, which does
not itself contain these closed-form equations). The companion `SEOBNRv5_theory.pdf` was also
fetched but not needed in the end -- the notes PDF alone contained every equation this port needed.

## The Hamiltonian (Eq. 7, 14, 16-19)

Reduced ("hat") units throughout, identical convention to `AlignedSpinPNHam.jl` (`r_hat=R/M`,
`t_hat=T/M`) with an important difference in the momentum variable -- see "Tortoise coordinates"
below. `nu` = symmetric mass ratio, `X1=m1/M >= X2=m2/M` (`m1>=m2`, matching this package's usual
convention and the DCC note's own §I "Notation").

The EOB energy map (Eq. 7, `M=1` in this reduced-unit convention so `HEOB=H` below):
```
H = sqrt(1 + 2*nu*(Heff - 1)) / nu,   Heff = Heven + Hodd
```
`Heven` (the point-mass-plus-spin-spin-plus-even-in-momentum sector, Eq. 14's square-root term)
and `Hodd` (the spin-orbit, "gyro-gravitomagnetic" sector, Eq. 14's `pphi*(ga+*a+ + ga-*am) +
SOcalib + Ga3` term, divided by a `2*(ap^2+r^2)+r*(ap^2+r^2-2r)` denominator) are both transcribed
**verbatim** from `evaluate_H`'s body (function [`_seobv5_ham_xi`](@ref) below) -- every
intermediate potential (`Dbpm`, `Apm`, `Qpm`/`QSalign2`, `Bnpa`/`BnpSalign2`/`Bnp`, `ASalign2`/`A`,
`Ga3`, `SOcalib`, `gam`/`gap`) is a direct, term-for-term port of the source's own `cdef double`
assignments, using the source's own variable names, not re-derived or simplified by hand -- these
are large (degree ~5, in some cases degree ~13) rational functions of `r` with `log(r)` dependence
(Padé-resummed 5PN potentials, per Eq. 9-11 of the DCC note), and re-deriving them from a
literature closed form would reintroduce exactly the transcription risk this port's whole
discipline exists to avoid.

## Analytic partials -- ported, not re-derived by hand

**A deliberate, explicitly-flagged departure from `AlignedSpinPNHam.jl`'s own precedent of
hand-differentiating the Hamiltonian on paper** (`_asph_hamiltonian_partials`'s docstring): this
Hamiltonian's `Dbpm`/`Apm` potentials alone are large enough (each a ratio of a ~13-term numerator
over an ~8-term denominator, both polynomial in `r` up to degree 5 and `log(r)` up to degree 2,
with additional `nu`-polynomial dependence up to `nu^4`) that hand-differentiating them on paper
would reintroduce real, non-trivial transcription risk. Since `pyseobnr`'s own `grad` cpdef method is **itself already an
analytic (not autodiff, not finite-difference) partial derivative** of `evaluate_H` -- evidently
produced by a symbolic-differentiation/CSE tool upstream in `pyseobnr`'s own development (the
`x0`...`x195` common-subexpression-elimination variable names are the tell) -- this port instead
**transcribes `grad`'s body verbatim**, term-for-term, keeping the source's own `x0`...`x195`
intermediate-variable names so the Julia code can be visually diffed against the `.pyx` source
line-for-line. This is functionally equivalent to (and arguably lower-risk than) "hand
differentiation": no autodiff/finite-difference machinery is used at runtime (matching
`_asph_hamiltonian_partials`'s own architectural requirement), and no term was independently
re-derived by this port -- only transcribed. [`_seobv5_gradient`](@ref) is this transcription;
correctness of the transcription itself is what the validation below checks.

Only `dHdr`, `dHdpr*`, `dHdpphi` are needed (`dHdphi` is identically `0` -- `phi` is cyclic, exactly
as in `AlignedSpinPNHam`); the Hessian (`pyseobnr`'s separate `hessian` cpdef, needed only by the
dissipative-initial-condition solve, itself another multi-hundred-line CSE tree) was **not**
transcribed -- see "Initial conditions" below for how this port gets the one Hessian component it
actually needs (a central finite difference of the already-analytic gradient, not a separate
from-scratch analytic port).

## Tortoise coordinates (Eq. 16-19) -- the state's momentum is `pr*`, not raw `pr`

Confirmed directly (not assumed) by reading `rhs_aligned.pyx`'s `get_rhs`: the integrated
momentum is the **tortoise-coordinate** radial momentum `pr* = pr*xi(r)` (Eq. 17), not the raw
`pr` `AlignedSpinPNHam.jl` uses -- named `prst` throughout `pyseobnr`'s own source and in this
file's state (`SEOBNRv5AlignedSpinState`'s `prst` field, deliberately *not* called `pr`, so a
future reader doesn't confuse it with `AlignedSpinPNHam`'s raw canonical momentum). The tortoise
factor (Eq. 19)
```
xi(r) = sqrt(Dbpm) * (Apm + ap^2/r^2) / (1 + ap^2/r^2)
```
(`ap = X1*chi1+X2*chi2`) is computed directly inside `evaluate_H`/[`_seobv5_ham_xi`](@ref) (it
needs `Dbpm`/`Apm`, already computed there for `H` itself) and enters Hamilton's equations
explicitly (Eq. 55, confirmed identical in structure to `rhs_aligned.pyx`'s `get_rhs`):
```
rdot     = xi * dH/dpr*  |_r
phidot   = dH/dpphi
pr*dot   = -xi * dH/dr |_pr*  + Fr
pphidot  = Fphi
```
where `(Fr,Fphi)` is the radiation-reaction force (below) -- **not** `AlignedSpinPNHam`'s
`dp_r_hat/dt_hat = -dH/dr_hat` alone; this Hamiltonian's dissipative sector has a genuine radial
component, confirmed directly from `RR_force`'s two-component return (see next section).

## Radiation-reaction force (Eq. 43) -- full 7-mode factorized-resummed flux

`pyseobnr`'s real `RR_force` (`waveform.pyx` lines 2246-2268) computes the *full* factorized-
resummed multipolar flux (Eq. 43: `F = Omega/(16*pi) * (p/|L|) * sum_{l=2}^{...} sum_{m=-l}^{l}
m^2 |D_L h^F_lm|^2`), summing over the **(2,2), (3,3), (2,1), (4,4), (3,2), (5,5), (4,3)** modes
(the DCC note's own §II list of modes SEOBNRv5HM actually models) -- each with its own
resummed-amplitude `rho_lm` PN series (`compute_rho_coeffs`, `waveform.pyx` lines 386-1156, a
~770-line function covering all seven modes). **This is now implemented in full** --
[`_seobv5_rr_force`](@ref) sums all seven modes, superseding this file's original, explicitly-
documented `(2,2)`-mode-only scope reduction (kept below as history, since the improvement it
produced is itself the evidence the original reduction's own attribution was correct).
[`_seobv5_rr_force`](@ref)'s own docstring gives the full per-mode/per-piece citation (Newtonian
prefactor magnitude, tail factor, `rho_lm`/`f_lm` PN coefficients, source term) and the structural
findings (real-valued `Slm` always; complex `rholmPwrl` for `(3,3)` only; `atest` used by `(2,2)`
only) that let this extension avoid genuine `Complex{T}` arithmetic everywhere except one mode.

**Validation (against a real `pyseobnr` trajectory,
`q=1.5,chi1=0.3,chi2=-0.2,omega_start=0.01`) confirms the expected, dramatic error-shrinkage,
not just "more code":**

| checkpoint (`t`, reduced) | `r` (M)  | old `(2,2)`-only resid | 7-mode resid | 7-mode + GSF-fit resid |
|---------------------------|----------|------------------------|--------------------------|-------------------------------------|
| `t=0` (IC, `pr0`)         | `21.59`  | `~0.3%`                | `~1e-5` (`~9.5e-6`)      | `~1.1e-7` |
| `t=125.66`                | `21.55`  | `r:~5e-6`, `om:~7.6e-6`| `r:~2e-7`, `om:~3e-7`    | `r:~1.9e-7`, `om:~2.9e-7` |
| `t=10628.49`              | `17.16`  | `r:~0.12%`, `om:~0.17%`| `r:~1.1e-6`, `om:~1.7e-6`| `r:~3.0e-7`, `om:~4.6e-7` |
| `t=14982.68`              | `13.36`  | (not checked)          | `r:~5.5e-5`, `om:~8.3e-5`| `r:~2.3e-6`, `om:~3.5e-6` |
| `t=16496.78`              | `10.45`  | (not checked)          | `r:~4.6e-4`, `om:~6.8e-4`| `r:~6.7e-6`, `om:~9.9e-6` |
| `t=17006.69`              | `8.29`   | (not checked)          | `r:~2.5e-3`, `om:~3.6e-3`| `r:~1.4e-5`, `om:~2.1e-5` |
| `t=17199.95`              | `6.45`   | (not checked, deep plunge)| `r:~1.2e-2`, `om:~1.7e-2`| `r:~1.3e-5`, `om:~1.8e-5` |

**A second, major correction**: cross-validating a new post-adiabatic driver's `pr_eqn`/`pphi_eqn` against
real `pyseobnr` at an arbitrary off-trajectory point surfaced a `~0.1%`-level discrepancy in
[`_seobv5_rr_force`](@ref) itself, traced to a **missing 2nd-order gravitational-self-force (GSF)
amplitude-fit correction** (`pyseobnr/eob/fits/GSF_fits.py`'s `GSF_amplitude_fits(nu)`, a small,
non-spinning-only closed-form fit wired into `eob_pars.flux_params.extra_coeffs`/
`extra_coeffs_log` for all seven modes at PN orders `v^4`-`v^10` -- `h22_v8,v10`, `h33_v8,v10`,
`h21_v6,v8,v10`, `h44_v6,v8,v10`, `h55_v4,v6,v8`, `h32_v6,v8,v10,vlog10`, `h43_v4,v6,v8,vlog8`,
confirmed directly via `compute_rholm_single`'s own loop, `waveform.pyx:1412-1429`, which adds
`extra_coeffs`/`extra_coeffs_log` into every mode's `rho_lm` polynomial at the matching power of
`v`, for both even- and odd-`m` modes alike). The 7-mode checkpoint validation (the middle
column above) does not isolate it because its residual is dominated first by the missing 6
modes, then by this file's genuinely-still-missing NQC/ringdown-attachment physics, masking a
smaller, systematic `~0.1%`-level gap in the flux itself. Porting `GSF_amplitude_fits` (added
directly into [`_seobv5_rr_force`](@ref), not a separate function -- see that function's own
updated docstring) and re-running the exact same checkpoints gives the right-hand column above:
**every checkpoint improved by 1-4 further orders of magnitude, including deep in the plunge
(`r~6.4M`) where the residual was previously this file's largest documented value (`~1.2%`) and is
now `~1.3e-5`** -- strongly suggesting that essentially all of the near-merger error this file
previously attributed to "no NQC corrections or merger-ringdown attachment" was actually this
missing GSF term, not primarily missing merger physics. This is a genuine correction to that prior
framing, not a re-derivation from first principles -- what fraction (if any) of the residual beyond
`r~6.4M` (this file still has no checkpoint validated closer to merger) is NQC/attachment-related
remains open, since the GSF fix's own effect could not be cleanly separated from merger-adjacent
physics at exactly the point this file's validation currently stops. All six checkpoints are locked
into `test/seobnrv5alignedspin_tests.jl` as a tightened regression (now checking the residual STAYS
small at each checkpoint, not merely that it grows monotonically -- the residual is now small enough
that checkpoint-to-checkpoint ordering is no longer a meaningful discriminator, see that test's own
comment for why the strict-monotonic-growth assertion was relaxed).


## Initial conditions -- two-stage IC_cons/IC_diss, cited to Khalil et al. 2021 Eq(60)/(68)

Mirrors `pyseobnr`'s own two-stage solve (`initial_conditions_aligned_opt.py`) rather than
`AlignedSpinPNHam._asph_circular_ic`'s single-stage circular solve, since this Hamiltonian's true
initial condition is genuinely **not** exactly circular (`pr*,0 != 0`, reflecting the real
inspiral's slow radial drift) -- unlike `AlignedSpinPNHam`'s simpler initial condition:
1. **`IC_cons`** ([`_seobv5_circular_ic`](@ref), Eq(60) of Khalil2021): 2D **fixed-iteration**
   Newton solve for `(r0,pphi0)` at `pr*=0` such that `dH/dpphi=Omega0` and `dH/dr=0`, seeded from
   `pyseobnr`'s own Newtonian guess (`r_guess=Omega0^(-2/3)`, `pphi_guess=sqrt(r_guess)`,
   `initial_conditions_aligned_opt.py` line 102-103) -- structurally the *same* two equations
   `AlignedSpinPNHam._asph_circular_ic` already solves, with the same central-finite-difference
   Jacobian technique (mirrors that file's own docstring's justification: the FD here resolves the
   `O(1)`-scale `(r,pphi)` solution itself, not a tiny correction to it, so is not the precision
   trap that file's docstring warns about elsewhere).
2. **`IC_diss`** ([`_seobv5_dissipative_ic`](@ref), Eq(68) of Khalil2021): 1D **fixed-iteration**
   Newton solve for `pr*,0` (seeded at the bracket midpoint `pr*=0`, since `pyseobnr`'s own
   `scipy.optimize.root_scalar` uses bracket `[-3e-2,0]`) of `rdot(pr*) - dH/dpr*(pr*) = 0`, where
   `rdot = (1/xi) * Fphi / dL/dr` and `dL/dr = -d2H/dr2 / d2H/dr dpphi`. **The Hessian components
   `d2H/dr2`, `d2H/dr dpphi` are themselves obtained by a central finite difference of this port's
   own already-analytic gradient** (not an analytic Hessian port).

**Open question this port explicitly investigated, per the plan's own instruction, rather than
assuming an answer**: does a fixed-iteration Newton scheme (this package's house style) converge
adequately for both stages, given this Hamiltonian's much greater nonlinearity (rational functions
of `r`, not `AlignedSpinPNHam`'s bare polynomial) than the precedent this pattern was first
validated on? **Answer, confirmed empirically against `pyseobnr`'s own real initial state: yes for the
configurations tested**, at the iteration counts this file actually uses. Extreme mass ratio and
high spin were not checked here (where the Hamiltonian's
rational-function structure could plausibly need more iterations or a better-conditioned seed).

## What this port does NOT include

- **Exactly-equal-mass (`delta==0`) is not handled for the four odd-`m` modes**
  (`(2,1),(3,3),(5,5),(4,3)`) -- see [`_seobv5_rr_force`](@ref)'s and
  [`_seobv5_multipole_prefix_abs`](@ref)'s own docstrings for exactly which terms divide by
  `delta` and would need `pyseobnr`'s own `dm2==0`-special-cased formulas (not ported) to be safe
  there. (Higher harmonics in the radiation-reaction force themselves -- the item this bullet
  point used to be -- are now implemented in full; see "Radiation-reaction force" above.)
- **Precession** -- see `SEOBNRv5PHM`.
- **NQC (non-quasi-circular) corrections, merger-ringdown attachment, and the higher-mode waveform
  machinery** -- this port targets the *dynamics* (`(r,phi,pr*,pphi)` trajectory through/near
  merger), the actual validation target `pyseobnr.generate_waveform.generate_modes_opt(...,
  debug=True).dynamics` and `XLALSimIMRSpinAlignedEOBModes`'s `dynamics_out` both expose directly,
  not the final calibrated `(2,2)`-mode waveform (which needs NQC/ringdown machinery this port does
  not implement).

# Domain convention

Same as `AlignedSpinPNHam`: `t >= 0`, seconds since the `f_min`-referenced initial condition (not
`TaylorT3`'s `t<=0`-to-coalescence convention). GW phase is `2*phi` (dominant-mode convention).
"""
struct SEOBNRv5AlignedSpin end

"""
    SEOBNRv5AlignedSpinState{T} <: FieldVector{4, T}

The `SEOBNRv5AlignedSpin` ODE state: reduced (dimensionless) orbital separation `r`, orbital phase
`phi`, **tortoise-coordinate** reduced radial momentum `prst` (`pr* = pr*xi(r)`, *not*
`AlignedSpinPNHamState`'s raw `pr`), and reduced orbital angular momentum `pphi` -- see
`SEOBNRv5AlignedSpin`'s docstring's "Tortoise coordinates" section for why this is a genuinely
different momentum variable, not just a naming choice. A `StaticArrays.jl` `FieldVector`, mirroring
`AlignedSpinPNHamState` exactly (same allocation-free-RK4 rationale).
"""
struct SEOBNRv5AlignedSpinState{T} <: FieldVector{4, T}
    r::T
    phi::T
    prst::T
    pphi::T
end

"""
    validate_seobnrv5alignedspin_params(eta, chi1, chi2; dL = nothing)

Host-side sanity check, identical contract to `validate_alignedspinpnham_params`: `0 < eta <=
0.25`, `-1 <= chi1,chi2 <= 1`, and (if `dL` given) `dL > 0`. Not called by `Phi`/`Ampl` themselves.
"""
function validate_seobnrv5alignedspin_params(eta, chi1, chi2; dL = nothing)
    0.0 < eta <= 0.25 || throw(ArgumentError("Invalid eta = $eta: must be in (0, 0.25]"))
    -1.0 <= chi1 <= 1.0 || throw(ArgumentError("Invalid chi1 = $chi1: spin must be in [-1, 1]"))
    -1.0 <= chi2 <= 1.0 || throw(ArgumentError("Invalid chi2 = $chi2: spin must be in [-1, 1]"))
    if dL !== nothing
        dL > 0.0 || throw(ArgumentError("Invalid dL = $dL: luminosity distance must be positive (and must be in Gpc, see Ampl's docstring)"))
    end
    return nothing
end

# --- Calibration-coefficient fits: verbatim from pyseobnr/eob/fits/fits_Hamiltonian.py, a
# machine-generated file with its own auditable provenance header (polynomial_fit_v4.py,
# 2022-09-01, git revision 159a400f4d78ddb6f88aa7bd43c2d31289a3a4ae, posterior data
# spinning_pst_handcut.h5). Cited there to Eq(88) ([SEOBNRv5HM-notes]) for a6_NS, Eq(91) for dSO. ---

"""
    _seobv5_a6_NS(nu) -> Float64

Non-spinning 5PN calibration coefficient in the `Apm` potential (DCC note Eq. 9's `a6`), a plain
quartic-in-`nu` polynomial. Verbatim from `pyseobnr/eob/fits/fits_Hamiltonian.py`'s `a6_NS`
(lines 172-188), cited there to Eq(88) of `[SEOBNRv5HM-notes]`. One-time-per-waveform, plain
`Float64` (mirrors `AlignedSpinPNHam._asph_circular_ic`'s own non-hot-path `Float64` precedent).
"""
@inline function _seobv5_a6_NS(nu)
    p0 = 4.17877875e01
    p1 = -3.02193382e03
    p2 = 3.34144394e04
    p3 = -1.69019140e05
    p4 = 3.29523262e05
    return p0 + p1 * nu + p2 * nu^2 + p3 * nu^3 + p4 * nu^4
end

"""
    _seobv5_dSO(nu, ap, am) -> Float64

Spin-orbit 4.5PN calibration coefficient (DCC note Eq. 15's `dSO`), a cubic polynomial in
`(nu,ap,am)` (`ap=X1*chi1+X2*chi2`, `am=X1*chi1-X2*chi2`). Verbatim from
`pyseobnr/eob/fits/fits_Hamiltonian.py`'s `dSO` (lines 90-127), cited there to Eq(91) of
`[SEOBNRv5HM-notes]`.
"""
@inline function _seobv5_dSO(nu, ap, am)
    return (
        -7.71251231383957 * am^3
        - 17.2294679794015 * am^2 * ap
        - 238.430383378296 * am^2 * nu
        + 69.5461667822545 * am^2
        - 10.5225438990315 * am * ap^2
        + 362.767393298729 * am * ap * nu
        - 85.8036338010274 * am * ap
        - 1254.66845939312 * am * nu^2
        + 472.431937787377 * am * nu
        - 39.742317057316 * am
        - 7.58458103577458 * ap^3
        - 42.7601129678844 * ap^2 * nu
        + 18.1783435552183 * ap^2
        - 201.905934468847 * ap * nu^2
        - 90.5790079104259 * ap * nu
        + 49.6299175121658 * ap
        + 478.546231305475 * nu^3
        + 679.521769948995 * nu^2
        - 177.334831768076 * nu
        - 37.6897780220529
    )
end

# --- Newtonian multipole-prefactor magnitudes: verbatim port of calculate_multipole_prefix's
# |n*eta*c| (waveform.pyx lines 209-291, Eq. 25-27 of [SEOBNRv5HM-notes]), needed for the 6
# non-(2,2) modes added to _seobv5_rr_force below. See _seobv5_multipole_prefix_abs's own
# docstring for exactly what's ported vs. deliberately not (the equal-mass odd-m special case). ---

"""
    _seobv5_dfact_odd(n) -> Float64

Odd double factorial `n!! = n*(n-2)*(n-4)*...*1` for odd `n`, needed by
[`_seobv5_multipole_prefix_abs`](@ref)'s `mult1 = 8*pi/(2l+1)!!` (`epsilon=0`) /
`16*pi/(2l+1)!!` (`epsilon=1`) terms (`calculate_multipole_prefix`, `waveform.pyx` lines
209-291). One-time-per-waveform, plain Float64.
"""
@inline function _seobv5_dfact_odd(n::Int)
    result = 1.0
    k = n
    while k > 1
        result *= k
        k -= 2
    end
    return result
end

"""
    _seobv5_multipole_prefix_abs(nu, X1, X2, l, m) -> Float64

`|calculate_multipole_prefix(m1,m2,l,m)|` (`waveform.pyx` lines 209-291, Eq. 25-27 of
`[SEOBNRv5HM-notes]`) -- the magnitude of the complex Newtonian-multipole prefactor `n*eta*c`.
Only the *magnitude* is ever needed anywhere downstream (confirmed directly: `compute_flux`,
`waveform.pyx` lines 1515-1591, calls `EOBFluxCalculateNewtonianMultipoleAbs` exclusively, built
from `prefixes_abs = np.abs(prefixes)`, `pyseobnr/models/SEOBNRv5Base.py` line 128 -- the complex
`prefixes` array itself is never dereferenced by the aligned-spin flux path), so this port never
needs the complex `n`/`prefix` itself, only its magnitude, computed here with plain real
arithmetic (`n`'s own phase is always a fixed power of `i`/`-i`/`1`/`-1` depending on parity, per
`calculate_multipole_prefix`'s own `epsilon`-branch structure -- confirmed by direct inspection,
not assumed -- so `|n|` is all that survives once every other factor multiplying it
(`Tlm`, `Slm`, `hNewton`) is confirmed real, see [`_seobv5_rr_force`](@ref)'s docstring). One-time-
per-waveform, plain Float64 (mirrors `_seobv5_a6_NS`/`_seobv5_dSO`).

**Scope limit, deliberately not ported**: `calculate_multipole_prefix`'s `m1==m2 and sign==-1`
branch (the special finite `c0` lookup for odd-`m` modes at *exactly* equal mass -- `l=2,3: c0=-1`,
`l=4,5: c0=-0.5` -- used because the general formula's `c=X2^(l+eps-1)+sign*X1^(l+eps-1)` is
identically zero at exactly `X1==X2` for odd `m`) is not implemented here -- this function always
evaluates the general (`m1!=m2`) formula. This is a real, deliberate scope limit, not worse than
(but also not independently fixed relative to) [`_seobv5_rr_force`](@ref)'s own hot-path
coefficient formulas below, which separately divide by `delta=X1-X2` for the same four odd-`m`
modes' `f_lm` terms (`compute_rho_coeffs`'s `dm2`-nonzero branch) -- both would need
`pyseobnr`'s own `dm2==0`-special-cased formulas to be exactly-equal-mass-safe, and neither is
here. Flagged once, in both places, not silently patched over.
"""
@inline function _seobv5_multipole_prefix_abs(nu, X1, X2, l::Int, m::Int)
    epsilon = (l + m) % 2
    sgn = iseven(m) ? 1.0 : -1.0
    c = X2^(l + epsilon - 1) + sgn * X1^(l + epsilon - 1)
    n_mag = Float64(m)^l
    if epsilon == 0
        mult1 = 8.0 * pi / _seobv5_dfact_odd(2l + 1)
        mult2 = sqrt(((l + 1.0) * (l + 2.0)) / (l * (l - 1.0)))
    else
        mult1 = 16.0 * pi / _seobv5_dfact_odd(2l + 1)
        mult2 = sqrt(((2l + 1.0) * (l + 2.0) * (l * l - m * m)) / ((2l - 1.0) * (l + 1.0) * l * (l - 1.0)))
    end
    n_mag *= mult1 * mult2
    return abs(n_mag * nu * c)
end

"""
    _seobv5_ham_xi(r, prst, pphi, hp) -> (H, xi, Heff)

The SEOBNRv5HM aligned-spin Hamiltonian value, tortoise factor, and effective Hamiltonian
(`H = sqrt(1+2*nu*(Heff-1))/nu`, DCC note Eq. 7/14; `xi`, Eq. 19) of pyseobnr's `evaluate_H`
(`Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C.pyx`), from the generated [`_seobv5_potentials`](@ref)
(source: `gen/symbolic/hamiltonians/v5phm.jl`). `M=1` throughout (pyseobnr's own
`EOBParams.p_params.M=1` for its dimensionless `(r,phi,pr*,pphi)` state). `hp`: the binary's Hamiltonian
parameters ([`SEOBNRv5HamParams`](@ref)).
"""
@inline function _seobv5_ham_xi(r, prst, pphi, hp)
    H, xi, _, _, _, _, _, Heff = _seobv5_potentials(r, prst, pphi, hp)
    return (H, xi, Heff)
end

# --- Radiation-reaction force: full factorized-resummed flux (all l <= 8 modes) -- see
# SEOBNRv5AlignedSpin's docstring's "Radiation-reaction force" section for the citation trail
# (this superseded the file's original (2,2)-mode-only scope reduction). Split into a per-binary part
# (`_seobv5_flux_coeffs`: every coefficient that depends only on nu, delta, chiS, chiA, atest, computed
# once per waveform and stored in the coefficient NamedTuple as `fc`) and a per-call part
# (`_seobv5_rr_force`): the coefficient algebra (hundreds of divisions) dominated the flux. ---

"""
    _seobv5_rho_coeffs(nu, delta, chiS, chiA, atest) -> NamedTuple

The binary-dependent coefficients of the seven SEOBNRv5HM modes' `rho_lm` and (odd `m`) `f_lm` series
(`compute_rho_coeffs`, `waveform.pyx`), GSF amplitude-fit terms (`GSF_amplitude_fits`) added to the `rho`
coefficient of the same power; evaluated by [`_seobv5_rho_f`](@ref). Coefficients that are the same for every
binary are literals there. See [`_seobv5_rr_force`](@ref)'s docstring for the per-term citations.
"""
@inline function _seobv5_rho_coeffs(nu::T, delta::T, chiS::T, chiA::T, atest::T) where {T}
@Tlit T begin
    nu2 = nu * nu
    nu3 = nu2 * nu
    nu4 = nu3 * nu
    nu5 = nu4 * nu

    # ---- GSF (2nd-order gravitational-self-force) amplitude fit corrections -- verbatim port of
    # pyseobnr/eob/fits/GSF_fits.py's GSF_amplitude_fits(nu) (93 lines, non-spinning-only,
    # "multiplied by nu" per its own docstring). Each entry here is
    # `extra_coeffs[l,m,v]`/`extra_coeffs_log[l,m,v]` in pyseobnr's own naming (the GSF fit's own
    # `hLM_vN`/`hLM_vlogN` dict keys), added into the corresponding mode's `rho_lm` polynomial at
    # the same power of `v` (never into `f_lm` -- confirmed directly:
    # `compute_rholm_single`'s own loop adds `extra_coeffs`/`extra_coeffs_log` to `rho` for both
    # even- and odd-`m` modes alike, `waveform.pyx:1412-1429`). The `nu`-independent `vlog` terms
    # (`h32_vlog10`, `h43_vlog8`) are literals in `_seobv5_rho_f`. ----
    h22_v8 = 21.2 * nu
    h22_v10 = -411.0 * nu
    h33_v8 = 12.0 * nu
    h33_v10 = -215.0 * nu
    h21_v6 = 1.65 * nu
    h21_v8 = 26.5 * nu
    h21_v10 = 80.0 * nu
    h44_v6 = -3.56 * nu
    h44_v8 = 15.6 * nu
    h44_v10 = -216.0 * nu
    h55_v4 = -2.61 * nu
    h55_v6 = 1.25 * nu
    h55_v8 = -35.7 * nu
    h32_v6 = 0.333 * nu
    h32_v8 = -6.5 * nu
    h32_v10 = 98.0 * nu - 1312549797426453052.0 / 176264081083715625.0
    h43_v4 = -0.654 * nu
    h43_v6 = -3.69 * nu
    h43_v8 = 18.5 * nu - 2465107182496333.0 / 460490801971200.0

    delta2 = delta * delta

    chiS2 = chiS * chiS
    chiA2 = chiA * chiA
    chiS3 = chiS2 * chiS
    chiA3 = chiA2 * chiA

    m1p3nu = -1.0 + 3.0 * nu
    m1p3nu2 = m1p3nu * m1p3nu
    m1p3nu3 = m1p3nu2 * m1p3nu
    m1p3nu4 = m1p3nu2 * m1p3nu2

    m1p2nu = -1.0 + 2.0 * nu

    # ---- (2,2) ----
    a2 = atest * atest
    a4 = a2 * a2

    rho22_2 = -43.0 / 42.0 + 55.0 * nu / 84.0
    rho22_3 = -2.0 * (chiS + chiA * delta - chiS * nu) / 3.0
    rho22_4 = -20555.0 / 10584.0 + 0.5 * (chiS + chiA * delta)^2 - 33025.0 * nu / 21168.0 + 19583.0 * nu2 / 42336.0
    rho22_5 = (-34.0 / 21.0 + 49.0 * nu / 18.0 + 209.0 * nu2 / 126.0) * chiS + (-34.0 / 21.0 - 19.0 * nu / 42.0) * delta * chiA
    # extra_PN_terms (the SEOBNRv5HM default, `compute_rho_coeffs` `if extra_PN_terms:`): the test-spin terms
    # 89 a^2/252 (v^6) and a^3/3 (v^7) are replaced by the NLO spin-squared and LO spin-cubed terms. 
    rho22_6 = 1556919113.0 / 122245200.0 - 48993925.0 * nu / 9779616.0 - 6292061.0 * nu2 / 3259872.0 + 10620745.0 * nu3 / 39118464.0 + 41.0 * nu * pi^2 / 192.0 +
              ((178.0 - 457.0 * nu - 972.0 * nu2) * chiA2) / 504.0 + (delta * (178.0 - 781.0 * nu) * chiA * chiS) / 252.0 +
              ((178.0 - 1817.0 * nu + 560.0 * nu2) * chiS2) / 504.0
    rho22_7 = chiA * delta * (18733.0 / 15876.0 + 50140.0 * nu / 3969.0 + 97865.0 * nu2 / 63504.0) + chiS * (18733.0 / 15876.0 + 74749.0 * nu / 5292.0 - 245717.0 * nu2 / 63504.0 + 50803.0 * nu3 / 63504.0) +
              ((delta - 4.0 * delta * nu) * chiA3) / 3.0 + (1.0 - 3.0 * nu - 4.0 * nu2) * chiA2 * chiS +
              (delta + 2.0 * delta * nu) * chiA * chiS2 + (1.0 / 3.0 + nu) * chiS3
    rho22_8 = -387216563023.0 / 160190110080.0 + 18353.0 * a2 / 21168.0 - a4 / 8.0
    rho22_10 = -16094530514677.0 / 533967033600.0

    # ---- (3,3) ----
    rho33_2 = -7.0 / 6.0 + 2.0 * nu / 3.0
    rho33_4 = -6719.0 / 3960.0 - 1861.0 * nu / 990.0 + 149.0 * nu2 / 330.0
    rho33_6 = 3203101567.0 / 227026800.0 + (-129509.0 / 25740.0 + 41.0 / 192.0 * pi^2) * nu - 274621.0 / 154440.0 * nu2 + 12011.0 / 46332.0 * nu3
    rho33_8 = -57566572157.0 / 8562153600.0
    rho33_10 = -903823148417327.0 / 30566888352000.0

    # Equal-mass (delta==0) is a SEPARATE closed-form branch in pyseobnr (waveform.pyx:655-665,
    # `if dm2: ... else: ...`), not the delta->0 limit of the general formula below -- several
    # terms (chiS2/chiA2 pieces) drop out that would not vanish under a naive limit, since the
    # general expression's /delta terms and its delta-free terms were fit/derived together as one
    # rational function, not independently continuous at delta=0. Ported verbatim from both of
    # pyseobnr's own branches, not re-derived -- fixes a real NaN (0*NaN, since hNewton33=absN33=0
    # here is a genuine, correctly-computed zero, but the /delta divisions upstream of it produced
    # 0/0 = NaN which then survived multiplication by that legitimate zero) for any exactly-
    # equal-mass binary, independent of spin.
    if delta == 0.0
        f33_3 = chiA * 3.0 / 8.0
        f33_4 = (3.0 - 12.0 * nu) * chiA * chiS
        f33_5 = (407.0 / 30.0 * nu2 - 593.0 / 60.0 * nu + 2.0 / 3.0) * chiA
        f33_6 = (44.0 * nu2 - 1.0 * nu - 7.0 / 2.0) * chiA * chiS
        f33_6vh = (7339.0 / 540.0 * nu - 81.0 / 20.0) * chiA
    else
        f33_3 = (chiS * delta * (-4.0 + 5.0 * nu) + chiA * (-4.0 + 19.0 * nu)) / (2.0 * delta)
        f33_4 = (1.5 * chiS2 * delta + (3.0 - 12.0 * nu) * chiA * chiS + delta * (1.5 - 6.0 * nu) * chiA2) / delta
        f33_5 = (delta * (241.0 / 30.0 * nu2 + 11.0 / 20.0 * nu + 2.0 / 3.0) * chiS + (407.0 / 30.0 * nu2 - 593.0 / 60.0 * nu + 2.0 / 3.0) * chiA) / delta
        f33_6 = (delta * (6.0 * nu2 - 27.0 / 2.0 * nu - 7.0 / 4.0) * chiS2 + (44.0 * nu2 - 1.0 * nu - 7.0 / 2.0) * chiA * chiS + delta * (-12.0 * nu2 + 11.0 / 2.0 * nu - 7.0 / 4.0) * chiA2) / delta
        f33_6vh = (delta * (593.0 / 108.0 * nu - 81.0 / 20.0) * chiS + (7339.0 / 540.0 * nu - 81.0 / 20.0) * chiA) / delta
    end

    # ---- (2,1) ----
    rho21_2 = -59.0 / 56.0 + 23.0 * nu / 84.0
    rho21_4 = -47009.0 / 56448.0 - 10993.0 * nu / 14112.0 + 617.0 * nu2 / 4704.0
    rho21_6 = 7613184941.0 / 2607897600.0
    rho21_8 = -1168617463883.0 / 911303737344.0
    rho21_10 = -63735873771463.0 / 16569158860800.0

    # Equal-mass (delta==0) branch: see the (3,3) mode's own comment above for why this is a
    # separate ported formula, not a delta->0 limit -- waveform.pyx:583-600, `if dm2: ... else:`.
    if delta == 0.0
        f21_1 = -3.0 * chiA / 2.0
        f21_3 = chiA * (147.0 + 1251.0 * nu) / 84.0
        f21_4 = (-6.0 + 21.0 * nu / 2.0) * chiS * chiA
        f21_5 = (0.75 - 3.0 * nu) * chiA3 +
                (-81.0 / 16.0 - 703.0 * nu2 / 112.0 + 8797.0 * nu / 1008.0 + (2.25 - 6.0 * nu) * chiS2) * chiA
        f21_6 = ((16652.0 - 39264.0 * nu + 9487.0 * nu2) * chiA * chiS) / 504.0
    else
        f21_1 = -3.0 * (chiS + chiA / delta) / 2.0
        f21_3 = (chiS * delta * (427.0 + 79.0 * nu) + chiA * (147.0 + 280.0 * delta2 + 1251.0 * nu)) / 84.0 / delta
        f21_4 = (-3.0 - 2.0 * nu) * chiA2 + (-3.0 + nu / 2.0) * chiS2 + (-6.0 + 21.0 * nu / 2.0) * chiS * chiA / delta
        f21_5 = (0.75 - 3.0 * nu) * chiA3 / delta +
                (-81.0 / 16.0 + 1709.0 * nu / 1008.0 + 613.0 * nu2 / 1008.0 + (2.25 - 3.0 * nu) * chiA2) * chiS +
                0.75 * chiS3 +
                (-81.0 / 16.0 - 703.0 * nu2 / 112.0 + 8797.0 * nu / 1008.0 + (2.25 - 6.0 * nu) * chiS2) * chiA / delta
        f21_6 = ((16652.0 - 9287.0 * nu + 720.0 * nu2) * chiA2) / 1008.0 +
                ((16652.0 - 39264.0 * nu + 9487.0 * nu2) * chiA * chiS) / (504.0 * delta) +
                ((16652.0 - 2633.0 * nu + 1946.0 * nu2) * chiS2) / 1008.0
    end

    # ---- (4,4) ----
    rho44_2 = (1614.0 - 5870.0 * nu + 2625.0 * nu2) / (1320.0 * m1p3nu)
    rho44_3 = (chiA * (10.0 - 39.0 * nu) * delta + chiS * (10.0 - 41.0 * nu + 42.0 * nu2)) / (15.0 * m1p3nu)
    rho44_4 = (-511573572.0 + 2338945704.0 * nu - 313857376.0 * nu2 - 6733146000.0 * nu3 + 1252563795.0 * nu2 * nu2) / (317116800.0 * m1p3nu2) +
              chiS2 / 2.0 + delta * chiS * chiA + delta2 * chiA2 / 2.0
    rho44_5 = chiA * delta * (-8280.0 + 42716.0 * nu - 57990.0 * nu2 + 8955.0 * nu3) / (6600.0 * m1p3nu2) +
              chiS * (-8280.0 + 66284.0 * nu - 176418.0 * nu2 + 128085.0 * nu3 + 88650.0 * nu2 * nu2) / (6600.0 * m1p3nu2)
    rho44_6 = 16600939332793.0 / 1098809712000.0
    rho44_8 = -172066910136202271.0 / 19426955708160000.0
    rho44_10 = -17154485653213713419357.0 / 568432724020761600000.0

    # ---- (3,2) ----
    rho32_1 = (4.0 * chiS * nu) / (-3.0 * m1p3nu)
    rho32_2 = (328.0 - 1115.0 * nu + 320.0 * nu2) / (270.0 * m1p3nu) +
              (-(16.0 * nu2 * chiS2) / (9.0 * m1p3nu2))
    rho32_3 = (delta * (2.0 + 13.0 * nu) * chiA) / (9.0 - 27.0 * nu) +
              ((90.0 - 1478.0 * nu + 2515.0 * nu2 + 3035.0 * nu3) * chiS) / (405.0 * m1p3nu2) -
              (320.0 * nu3 * chiS3) / (81.0 * m1p3nu3)
    rho32_4 = (-1444528.0 + 8050045.0 * nu - 4725605.0 * nu2 - 20338960.0 * nu3 + 3085640.0 * nu2 * nu2) / (1603800.0 * m1p3nu2) +
              ((1.0 - 9.0 * nu + 12.0 * nu2) * chiA2) / (3.0 - 9.0 * nu) -
              (2.0 * delta * (-9.0 + 44.0 * nu + 25.0 * nu2) * chiA * chiS) / (27.0 * m1p3nu2) +
              ((-81.0 + 387.0 * nu - 1435.0 * nu2 + 1997.0 * nu3 + 2452.0 * nu2 * nu2) * chiS2) / (243.0 * m1p3nu3)
    rho32_5 = (delta * (-245344.0 + 1128531.0 * nu - 1514740.0 * nu2 + 889673.0 * nu3) * chiA) / (106920.0 * m1p3nu2) +
              ((2208096.0 - 20471053.0 * nu + 70519165.0 * nu2 - 101706029.0 * nu3 + 40204523.0 * nu4 + 11842250.0 * nu5) * chiS) / (962280.0 * m1p3nu3) -
              (8.0 * nu * (1.0 - 9.0 * nu + 12.0 * nu2) * chiA2 * chiS) / (9.0 * m1p3nu2) -
              (16.0 * delta * nu * (-9.0 + 46.0 * nu + 38.0 * nu2) * chiA * chiS2) / (81.0 * m1p3nu3) +
              (8.0 * nu * (-243.0 + 1269.0 * nu - 5029.0 * nu2 + 5441.0 * nu3 + 12022.0 * nu4) * chiS3) / (2187.0 * m1p3nu4)
    rho32_6 = 5849948554.0 / 940355325.0
    rho32_8 = -10607269449358.0 / 3072140846775.0

    # ---- (5,5) ----
    rho55_2 = (487.0 - 1298.0 * nu + 512.0 * nu2) / (390.0 * m1p2nu)
    rho55_4 = -3353747.0 / 2129400.0
    rho55_6 = 190606537999247.0 / 11957879934000.0
    rho55_8 = -1213641959949291437.0 / 118143853747920000.0

    # Equal-mass (delta==0) branch: see the (3,3) mode's own comment above for why this is a
    # separate ported formula, not a delta->0 limit -- waveform.pyx:837-873, `if dm2: ... else:`.
    if delta == 0.0
        f55_3 = chiA * (10.0 / (3.0 * m1p2nu) - 70.0 * nu / (3.0 * m1p2nu) + 110.0 * nu2 / (3.0 * m1p2nu))
        f55_4 = chiA * chiS * (-5.0 / m1p2nu + 30.0 * nu / m1p2nu - 40.0 * nu2 / m1p2nu)
    else
        f55_3 = chiA / delta * (10.0 / (3.0 * m1p2nu) - 70.0 * nu / (3.0 * m1p2nu) + 110.0 * nu2 / (3.0 * m1p2nu)) +
                chiS * (10.0 / (3.0 * m1p2nu) - 10.0 * nu / m1p2nu + 10.0 * nu2 / m1p2nu)
        f55_4 = chiS2 * (-5.0 / (2.0 * m1p2nu) + 5.0 * nu / m1p2nu) +
                chiA * chiS / delta * (-5.0 / m1p2nu + 30.0 * nu / m1p2nu - 40.0 * nu2 / m1p2nu) +
                chiA2 * (-5.0 / (2.0 * m1p2nu) + 15.0 * nu / m1p2nu - 20.0 * nu2 / m1p2nu)
    end

    # ---- (4,3) ----
    rho43_2 = (222.0 - 547.0 * nu + 160.0 * nu2) / (176.0 * m1p2nu)
    rho43_4 = -6894273.0 / 7047040.0
    rho43_6 = 1664224207351.0 / 195343948800.0

    # Equal-mass (delta==0) branch: see the (3,3) mode's own comment above for why this is a
    # separate ported formula, not a delta->0 limit -- waveform.pyx:773-778 (f43_1) and :1132-1148
    # (f43_3/f43_4, `extra_PN_terms`'s own `if dm2: ... else:`).
    if delta == 0.0
        f43_1 = -5.0 * chiA / 4.0
        f43_3 = (nu * (-2661.0 + 3143.0 * nu) * chiA) / (132.0 * m1p2nu)
        f43_4 = ((18.0 - 108.0 * nu + 137.0 * nu2) * chiA * chiS) / (6.0 - 12.0 * nu)
    else
        f43_1 = 5.0 * (chiA - chiS * delta) * nu / (2.0 * delta * m1p2nu)
        f43_3 = (nu * (-2661.0 + 3143.0 * nu) * chiA) / (132.0 * delta * m1p2nu) +
                (23.0 * nu * (87.0 + 23.0 * nu) * chiS) / (132.0 * m1p2nu)
        f43_4 = ((9.0 - 74.0 * nu + 72.0 * nu2) * chiA2) / (6.0 - 12.0 * nu) +
                ((18.0 - 108.0 * nu + 137.0 * nu2) * chiA * chiS) / (6.0 * delta - 12.0 * delta * nu) +
                ((9.0 + 2.0 * nu + 35.0 * nu2) * chiS2) / (6.0 - 12.0 * nu)
    end

    # GSF terms summed into the same power's coefficient in the order the series used to add them
    return (rho22_2 = rho22_2, rho22_3 = rho22_3, rho22_4 = rho22_4, rho22_5 = rho22_5, rho22_6 = rho22_6,
        rho22_7 = rho22_7, rho22_8 = rho22_8 + h22_v8, rho22_10 = rho22_10 + h22_v10,
        rho33_2 = rho33_2, rho33_4 = rho33_4, rho33_6 = rho33_6, rho33_8 = rho33_8 + h33_v8, rho33_10 = rho33_10 + h33_v10,
        f33_3 = f33_3, f33_4 = f33_4, f33_5 = f33_5, f33_6 = f33_6, f33_6vh = f33_6vh,
        rho21_2 = rho21_2, rho21_4 = rho21_4, rho21_6 = rho21_6 + h21_v6, rho21_8 = rho21_8 + h21_v8,
        rho21_10 = rho21_10 + h21_v10, f21_1 = f21_1, f21_3 = f21_3, f21_4 = f21_4, f21_5 = f21_5, f21_6 = f21_6,
        rho44_2 = rho44_2, rho44_3 = rho44_3, rho44_4 = rho44_4, rho44_5 = rho44_5, rho44_6 = rho44_6 + h44_v6,
        rho44_8 = rho44_8 + h44_v8, rho44_10 = rho44_10 + h44_v10,
        rho32_1 = rho32_1, rho32_2 = rho32_2, rho32_3 = rho32_3, rho32_4 = rho32_4, rho32_5 = rho32_5,
        rho32_6 = rho32_6 + h32_v6, rho32_8 = rho32_8 + h32_v8, rho32_10 = h32_v10,
        rho55_2 = rho55_2, rho55_4 = rho55_4 + h55_v4, rho55_6 = rho55_6 + h55_v6, rho55_8 = rho55_8 + h55_v8,
        f55_3 = f55_3, f55_4 = f55_4,
        rho43_2 = rho43_2, rho43_4 = rho43_4 + h43_v4, rho43_6 = rho43_6 + h43_v6, rho43_8 = h43_v8,
        f43_1 = f43_1, f43_3 = f43_3, f43_4 = f43_4)
end # @Tlit T
end

"""
    _seobv5_rho_f(v, lnv, nuHomega, rc) -> NamedTuple

The resummed PN amplitude factors of the seven SEOBNRv5HM modes, shared by the radiation-reaction flux
([`_seobv5_rr_force`](@ref)) and the SEOBNRv5HM inspiral modes: `rho_lm` (before its `l`-th power, GSF terms
included) and, for odd `m`, `f_lm` (`compute_rho_coeffs` / `compute_rholm_single`, `waveform.pyx`), at
`v = Omega^(1/3)`, `lnv = log(v)`, from the coefficients `rc` of [`_seobv5_rho_coeffs`](@ref). `f33_im` is the
(3,3) mode's imaginary `f` term `f_coeffs_vh[3,3][6] vh^6` (`compute_extra_flm_terms`). The odd-`m` modes'
special calibration terms (`f_coeffs[l,m][7]`, set after the dynamics by `compute_special_coeffs`) are not
included: they are zero in the flux, and the waveform adds them. `eulerlog_m = gamma + log(2 m v)` is
`lnv` plus a constant.
"""
@inline function _seobv5_rho_f(v::T, lnv::T, nuHomega::T, rc) where {T}
@Tlit T begin
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v3 * v3
    v7 = v6 * v
    v8 = v4 * v4
    v10 = v5 * v5

    eulerlog1 = lnv + 1.2703628454614782     # gamma + log(2 m), m = 1..5
    eulerlog2 = lnv + 1.9635100260214235
    eulerlog3 = lnv + 2.3689751341295877
    eulerlog4 = lnv + 2.6566572065813685
    eulerlog5 = lnv + 2.8798007578955787

    rho22 = 1.0 + rc.rho22_2 * v2 + rc.rho22_3 * v3 + rc.rho22_4 * v4 + rc.rho22_5 * v5 +
            (rc.rho22_6 + (-428.0 / 105.0) * eulerlog2) * v6 + rc.rho22_7 * v7 +
            (rc.rho22_8 + (9202.0 / 2205.0) * eulerlog2) * v8 +
            (rc.rho22_10 + (439877.0 / 55566.0) * eulerlog2) * v10

    rho33 = 1.0 + rc.rho33_2 * v2 + rc.rho33_4 * v4 +
            (rc.rho33_6 + (-26.0 / 7.0) * eulerlog3) * v6 +
            (rc.rho33_8 + (13.0 / 3.0) * eulerlog3) * v8 +
            (rc.rho33_10 + (87347.0 / 13860.0) * eulerlog3) * v10
    f33 = rc.f33_3 * v3 + rc.f33_4 * v4 + rc.f33_5 * v5 + rc.f33_6 * v6
    f33_im = rc.f33_6vh * nuHomega * nuHomega

    rho21 = 1.0 + rc.rho21_2 * v2 + rc.rho21_4 * v4 +
            (rc.rho21_6 + (-107.0 / 105.0) * eulerlog1) * v6 +
            (rc.rho21_8 + (6313.0 / 5880.0) * eulerlog1) * v8 +
            (rc.rho21_10 + (5029963.0 / 5927040.0) * eulerlog1) * v10
    f21 = rc.f21_1 * v + rc.f21_3 * v3 + rc.f21_4 * v4 + rc.f21_5 * v5 + rc.f21_6 * v6

    rho44 = 1.0 + rc.rho44_2 * v2 + rc.rho44_3 * v3 + rc.rho44_4 * v4 + rc.rho44_5 * v5 +
            (rc.rho44_6 + (-12568.0 / 3465.0) * eulerlog4) * v6 +
            (rc.rho44_8 + (845198.0 / 190575.0) * eulerlog4) * v8 +
            (rc.rho44_10 + (22324502267.0 / 3815311500.0) * eulerlog4) * v10

    rho32 = 1.0 + rc.rho32_1 * v + rc.rho32_2 * v2 + rc.rho32_3 * v3 + rc.rho32_4 * v4 + rc.rho32_5 * v5 +
            (rc.rho32_6 + (-104.0 / 63.0) * eulerlog2) * v6 +
            (rc.rho32_8 + (17056.0 / 8505.0) * eulerlog2) * v8 +
            (rc.rho32_10 + (18778864.0 / 12629925.0) * eulerlog2) * v10

    rho55 = 1.0 + rc.rho55_2 * v2 + rc.rho55_4 * v4 +
            (rc.rho55_6 + (-1546.0 / 429.0) * eulerlog5) * v6 +
            (rc.rho55_8 + (376451.0 / 83655.0) * eulerlog5) * v8 +
            (-150082616449726042201261.0 / 4837990810977324000000.0 + (2592446431.0 / 456756300.0) * eulerlog5) * v10
    f55 = rc.f55_3 * v3 + rc.f55_4 * v4

    rho43 = 1.0 + rc.rho43_2 * v2 + rc.rho43_4 * v4 + (rc.rho43_6 + (-1571.0 / 770.0) * eulerlog3) * v6 +
            (rc.rho43_8 + (174381.0 / 67760.0) * eulerlog3) * v8
    f43 = rc.f43_1 * v + rc.f43_3 * v3 + rc.f43_4 * v4

    return (rho22 = rho22, rho33 = rho33, f33 = f33, f33_im = f33_im, rho21 = rho21, f21 = f21,
        rho44 = rho44, rho32 = rho32, rho55 = rho55, f55 = f55, rho43 = rho43, f43 = f43)
end # @Tlit T
end

"""
    _seobv5_flux_coeffs(nu, delta, chiS, chiA, atest, absN33, absN21, absN44, absN32, absN55, absN43) -> NamedTuple

The per-binary part of [`_seobv5_rr_force`](@ref): `rho` ([`_seobv5_rho_coeffs`](@ref), the seven hand-ported
modes), `C` (their prefactors `m^2 |n_lm Y_lm|^2 / (l!)^2`, order (2,2), (3,3), (2,1), (4,4), (3,2), (5,5), (4,3);
`absN_lm` from [`_seobv5_multipole_prefix_abs`](@ref), `|n_22| = 32 pi sqrt(6) nu / 15`) and `extra`
([`_seobv5_flux_extra_coeffs`](@ref), the other `l <= 8` modes). Stored once per waveform (`coeffs.fc`).
"""
@inline function _seobv5_flux_coeffs(nu::T, delta::T, chiS::T, chiA::T, atest::T,
    absN33::T, absN21::T, absN44::T, absN32::T, absN55::T, absN43::T) where {T}
@Tlit T begin
    n22 = (32.0 * pi * sqrt(6.0) / 15.0) * nu * 0.3862742020231896
    n33 = absN33 * 0.4172238236327842
    n21 = absN21 * 0.3454941494713355
    n44 = absN44 * 0.4425326924449826
    n32 = absN32 * 0.3862742020231896
    n55 = absN55 * 0.46413220344085826
    n43 = absN43 * 0.4172238236327842
    C = (n22 * n22, 9.0 * n33 * n33 / 36.0, n21 * n21 / 4.0, 16.0 * n44 * n44 / 576.0, 4.0 * n32 * n32 / 36.0,
        25.0 * n55 * n55 / 14400.0, 9.0 * n43 * n43 / 576.0)
    return (rho = _seobv5_rho_coeffs(nu, delta, chiS, chiA, atest), C = C,
        extra = _seobv5_flux_extra_coeffs(nu, delta, chiS, chiA, atest))
end # @Tlit T
end

# `fc` (nested tuples) as Float64, for the plain-Float64 PA solves. Unrolled: Base `map` over a tuple longer than 32
# (`fc.extra` has 70 entries) falls back to a dynamic path that does not compile in a GPU kernel ("unsupported dynamic
# function invocation (call to kwcall)" in the PA-fixed / F2 KA CUDA tests).
_seobv5_f64(x::Real) = Float64(x)
@generated _seobv5_f64(x::Tuple) =
    Expr(:block, Expr(:meta, :inline), Expr(:tuple, (:(_seobv5_f64(x[$i])) for i in 1:fieldcount(x))...))
@generated _seobv5_f64(x::NamedTuple{N}) where {N} =
    Expr(:block, Expr(:meta, :inline), :(NamedTuple{$N}($(Expr(:tuple, (:(_seobv5_f64(x[$i])) for i in 1:length(N))...)))))
_seobv5_f64(hp::SEOBNRv5HamParams) = SEOBNRv5HamParams(ntuple(i -> _seobv5_f64(getfield(hp, i)), Val(7))...)

"""
    _seobv5_rr_force(prst, pphi, omega, omega_circ, Heff, H, nu, fc) -> (Fr, Fphi)

The full aligned-spin radiation-reaction force, cited to Eq(43) of the `[SEOBNRv5HM]` DCC note
and ported from `waveform.pyx`'s `RR_force`/`compute_flux` (lines 2246-2268/1515-1591): the seven
waveform modes `(2,2),(3,3),(2,1),(4,4),(3,2),(5,5),(4,3)` hand-ported below, plus every other mode
with `l <= 8` (`compute_flux` sums `l = 2..ell_max`, `ell_max = 8` in `eob_parameters.h`), generated
from `compute_rho_coeffs` into [`_seobv5_flux_extra`](@ref) (`scripts/translate_pyseobnr_flux.py`).
`fc` holds every per-binary coefficient ([`_seobv5_flux_coeffs`](@ref), `coeffs.fc`). Equals pyseobnr 0.3.7's
`RR_force` to <= 7e-15 relative from the inspiral through the plunge (six binaries). The
extra modes carry ~1e-5 of the flux at r ~ 6.7 and up to ~4% in the late plunge.

The flux needs only `|h_lm|^2 = |T_lm|^2 S_lm^2 |hNewton_lm|^2 |rholmPwrl|^2`, so the tail factor enters squared,
`|T_lm|^2 = tp2_m prod_(j=1..l) (hs_m + j^2) / (l!)^2` with `tp2_m = 4 pi k / (1 - exp(-4 pi k))`, `hs_m = 4 k^2`,
`k = m nu H Omega`: no square roots, and one `expm1` for all `m` (`1 - E^m = (1 - E)(1 + E + ... + E^(m-1))`,
`E = exp(-4 pi nu H Omega)`).

# Structural findings that shaped this implementation

1. **`Slm` (the source term) is real for every mode, always** -- `compute_flux`'s own local
   `Slm` is a plain `cdef double`, set to `source1 = Heff` (even `l+m`) or `source2 = v*pphi`
   (odd `l+m`), never complex (`waveform.pyx` lines ~1546-1551). No mode needs a complex source
   term.
2. **`rholmPwrl` (`rho_lm^l + f_lm`) is complex for exactly one mode: `(3,3)`.** `f_lm`
   (`compute_rholm_single`, `waveform.pyx` lines 1390-1444) is populated only for odd `m`
   (`(2,1),(3,3),(5,5),(4,3)` here), but `compute_extra_flm_terms` (lines 1371-1381) -- the one
   place an imaginary correction (`* I`) is ever added to `f_lm` -- fires *only* for
   `l==3 and m==3`; every other odd-`m` mode's `f_lm` stays real. **Consequence: this port never
   needs genuine `Complex{T}` arithmetic** -- only `(3,3)` needs a real/imaginary pair (two plain
   `T`s), combined via `re^2+im^2` at the point `|hlm|^2` is taken (since `Tlm`, `Slm`, `hNewton`
   are all real scalars multiplying that one complex `rholmPwrl`, `|hlm|^2 =
   (Tlm*Slm*hNewton)^2*(re^2+im^2)` exactly).
3. **The test-particle spin `atest` is used by `(2,2)` only.** `compute_rho_coeffs` explicitly
   zeroes `a=a2=a3=0` immediately after the `(2,2)` mode's block ("We set test-spin terms to 0
   (as in SEOBNRv4HM) as no major improvement was found when trying to include them",
   `waveform.pyx` ~line 478), and never restores it -- every `a`-dependent term in every mode's
   coefficients below `(2,2)` is therefore transcribed as the constant it reduces to (the `a`
   term simply dropped), not carried as a live variable.
4. **`extra_PN_terms=True` (`pyseobnr/models/SEOBNRv5Base.py` line 131-132, confirmed the
   `SEOBNRv5HM` default)** adds NLO spin corrections to `(2,2)` (replacing its test-spin `a^2`, `a^3`
   terms at `v^6`, `v^7`, see [`_seobv5_rho_coeffs`](@ref)), `(3,2)` (`rho_coeffs[3,2][2..5] +=`, unconditional, no `dm2` gate), and
   `(4,3)` (`f_coeffs[4,3][3],[4]`, `dm2`-gated) -- all included.
5. **Newtonian-prefactor magnitude, not the complex prefactor itself, is all any mode needs**
   (`EOBFluxCalculateNewtonianMultipoleAbs`/`prefixes_abs`, see
   [`_seobv5_multipole_prefix_abs`](@ref)'s own docstring) -- `absN33`,`absN21`,`absN44`,`absN32`,
   `absN55`,`absN43` are computed once per waveform by that function and folded into `fc.C`;
   `absN22`'s own closed form (`c=X1+X2=1` always, since the `(2,2)` Newtonian-prefactor exponent
   happens to be `1`) is used directly.
6. **Tail factor `T_lm` uses `hathatk_m = nu*H*m*omega` (the mode's own `m`, not `l`) and
   `z2=l!`** (`compute_tail`, `waveform.pyx` lines 345-377: the outer loop variable is `m`, the
   inner accumulates `prod_{j=1}^{l}(hathatksq4_m+j^2)`, stored divided by `LOOKUP_TABLE[l]=l!`)
   -- confirmed directly from the loop's own store condition (`if m>j: continue`, i.e. store only
   once the inner index reaches `l`), not assumed from the `(2,2)` special case alone. Several
   modes share an `m` (`(2,2)`&`(3,2)` both `m=2`; `(3,3)`&`(4,3)` both `m=3`), so the per-`m`
   factors are computed once and shared, with the generated modes too.
7. **A 2nd-order gravitational-self-force (GSF) amplitude-fit correction applies to every mode**
   (see this file's module docstring's "Radiation-reaction force" section for the
   before/after residual table). Verbatim port of `pyseobnr/eob/fits/
   GSF_fits.py`'s `GSF_amplitude_fits(nu)` (93 lines, non-spinning-only, "multiplied by nu" per its
   own docstring): `h22_v8,v10`, `h33_v8,v10`, `h21_v6,v8,v10`, `h44_v6,v8,v10`, `h55_v4,v6,v8`,
   `h32_v6,v8,v10,vlog10`, `h43_v4,v6,v8,vlog8` (see [`_seobv5_rho_coeffs`](@ref)) are added into the
   corresponding mode's `rho_lm` polynomial at the matching
   power of `v` -- confirmed directly that `compute_rholm_single`'s own loop adds
   `extra_coeffs`/`extra_coeffs_log` into `rho` (never `f`) for every mode regardless of parity
   (`waveform.pyx:1412-1429`), and that `pyseobnr.models.SEOBNRv5HM`'s own model setup populates
   `eob_pars.flux_params.extra_coeffs`/`extra_coeffs_log` from exactly this function's output
   (`models/SEOBNRv5HM.py` lines ~285-300, regex-parsing `GSF_amplitude_fits`'s own `"h{l}{m}_v{n}"`/
   `"h{l}{m}_vlog{n}"`-keyed dict). `(3,2)` and `(4,3)` each gain a genuinely new `v^10`/`v^8` term
   that did not exist in this port's (or `pyseobnr`'s own base `rho_coeffs`') PN series at all before
   this fix (`h32_v10`/`h32_vlog10`, `h43_v8`/`h43_vlog8`) -- not merely a correction to an existing
   coefficient for those two modes.

# Deliberate, explicitly-flagged scope limits (see also [`_seobv5_multipole_prefix_abs`](@ref))

- **Exactly-equal-mass odd-`m` prefactors** follow the general formula (zero at `X1 == X2`; the `c0`
  constants of `calculate_multipole_prefix` are not used), for the seven modes and the generated ones.
"""
@inline function _seobv5_rr_force(prst::T, pphi::T, omega::T, omega_circ::T, Heff::T, H::T, nu::T, fc) where {T}
@Tlit T begin
    omega_safe = max(omega, eps(T))
    omega_circ_safe = max(omega_circ, eps(T))

    v = cbrt(omega_safe)
    lnv = log(v)
    v_phi = omega_safe / omega_circ_safe^(2.0 / 3.0)
    w = v_phi * v_phi
    w2 = w * w
    w3 = w2 * w
    w4 = w2 * w2
    w5 = w4 * w

    nuHomega = nu * H * omega_safe
    rf = _seobv5_rho_f(v, lnv, nuHomega, fc.rho)

    # ---- squared tail prefactors and hathatksq4 per m = 1..8 (compute_tail) ----
    x4pi = 4.0 * pi * nuHomega
    d = -expm1(-x4pi)
    E = 1.0 - d
    g2 = 1.0 + E
    g3 = 1.0 + E * g2
    g4 = 1.0 + E * g3
    g5 = 1.0 + E * g4
    g6 = 1.0 + E * g5
    g7 = 1.0 + E * g6
    g8 = 1.0 + E * g7
    r = x4pi / d
    tp2 = (r, 2.0 * r / g2, 3.0 * r / g3, 4.0 * r / g4, 5.0 * r / g5, 6.0 * r / g6, 7.0 * r / g7, 8.0 * r / g8)
    x2 = 4.0 * nuHomega * nuHomega
    hs = (x2, 4.0 * x2, 9.0 * x2, 16.0 * x2, 25.0 * x2, 36.0 * x2, 49.0 * x2, 64.0 * x2)

    Heff2 = Heff * Heff
    L = v * pphi
    L2 = L * L
    C = fc.C

    # ---- (2,2): Slm = Heff ----
    rholm22 = rf.rho22 * rf.rho22
    flux_acc = C[1] * tp2[2] * ((hs[2] + 1.0) * (hs[2] + 4.0)) * Heff2 * w2 * rholm22 * rholm22

    # ---- (3,3): rho waveform.pyx 607-628, f 631-663 (dm2 branch), extra_flm (imaginary correction) via
    # compute_extra_flm_terms lines 1371-1381 / f_coeffs_vh[3,3][6]; Slm = Heff ----
    rholm33_re = rf.rho33 * rf.rho33 * rf.rho33 + rf.f33
    rholm33_im = rf.f33_im
    flux_acc += C[2] * tp2[3] * ((hs[3] + 1.0) * (hs[3] + 4.0) * (hs[3] + 9.0)) * Heff2 * w3 *
                (rholm33_re * rholm33_re + rholm33_im * rholm33_im)

    # ---- (2,1): rho waveform.pyx 489-521, f 524-600 (dm2 branch); Slm = v pphi ----
    rholm21 = rf.rho21 * rf.rho21 + rf.f21
    flux_acc += C[3] * tp2[1] * ((hs[1] + 1.0) * (hs[1] + 4.0)) * L2 * w3 * rholm21 * rholm21

    # ---- (4,4): rho waveform.pyx 724-761; Slm = Heff ----
    rholm44 = rf.rho44 * rf.rho44 * rf.rho44 * rf.rho44
    flux_acc += C[4] * tp2[4] * ((hs[4] + 1.0) * (hs[4] + 4.0) * (hs[4] + 9.0) * (hs[4] + 16.0)) * Heff2 * w4 *
                rholm44 * rholm44

    # ---- (3,2): rho waveform.pyx 669-693 (base) + 1101-1130 (extra_PN_terms, unconditional -- no dm2 gate);
    # Slm = v pphi ----
    rholm32 = rf.rho32 * rf.rho32 * rf.rho32
    flux_acc += C[5] * tp2[2] * ((hs[2] + 1.0) * (hs[2] + 4.0) * (hs[2] + 9.0)) * L2 * w4 * rholm32 * rholm32

    # ---- (5,5): rho waveform.pyx 822-834, f 837-873 (dm2 branch); Slm = Heff ----
    rholm55 = rf.rho55 * rf.rho55 * rf.rho55 * rf.rho55 * rf.rho55 + rf.f55
    flux_acc += C[6] * tp2[5] * ((hs[5] + 1.0) * (hs[5] + 4.0) * (hs[5] + 9.0) * (hs[5] + 16.0) * (hs[5] + 25.0)) *
                Heff2 * w5 * rholm55 * rholm55

    # ---- (4,3): rho waveform.pyx 765-771, f 773-778 (base) + 1132-1148 (extra_PN_terms, dm2 branch);
    # Slm = v pphi ----
    rholm43 = rf.rho43 * rf.rho43 * rf.rho43 * rf.rho43 + rf.f43
    flux_acc += C[7] * tp2[3] * ((hs[3] + 1.0) * (hs[3] + 4.0) * (hs[3] + 9.0) * (hs[3] + 16.0)) * L2 * w5 *
                rholm43 * rholm43

    # ---- The other l <= 8 modes pyseobnr's compute_flux sums (ell_max = 8; generated) ----
    flux_acc += _seobv5_flux_extra(v, lnv, w, Heff2, L2, tp2, hs, fc.extra)

    # ---- Final assembly: RR_force, waveform.pyx lines 2246-2268 ----
    flux = omega_safe * omega_safe * flux_acc / (8.0 * pi)
    flux = flux / nu
    f_over_om = flux / omega_safe
    Fr = -prst / pphi * f_over_om
    Fphi = -f_over_om
    return (Fr, Fphi)
end # @Tlit T
end

# --- RHS assembly: get_rhs, rhs_aligned.pyx lines 17-60. omega_circ is obtained by re-evaluating
# this same Hamiltonian's dHdpphi at prst=0 (same r,pphi) -- mathematically identical to
# pyseobnr's separate H.omega() method (both encode the same partial derivative of the same H),
# just reusing this file's own gradient function rather than porting a second, separately-CSE'd
# code path for the same quantity. ---

"""
    _seobv5_rhs(state::SEOBNRv5AlignedSpinState{T}, coeffs) -> SEOBNRv5AlignedSpinState{T}

The ODE right-hand side, DCC note Eq. 55 / `rhs_aligned.pyx`'s `get_rhs`: `rdot = xi*dHdpr`,
`phidot = dHdpphi`, `prdot = -xi*dHdr + Fr`, `pphidot = Fphi` (`dHdphi=0` identically, `phi`
cyclic). Allocation-free (mirrors `AlignedSpinPNHam._asph_rhs`).
"""
@inline function _seobv5_rhs(state::SEOBNRv5AlignedSpinState{T}, coeffs) where {T}
@Tlit T begin
    r, phi, prst, pphi = state.r, state.phi, state.prst, state.pphi

    H, xi, Heff, dHdr, dHdpr, omega, omega_circ = _seobv5_sym_rhs(r, prst, pphi, coeffs.hp)

    Fr, Fphi = _seobv5_rr_force(prst, pphi, omega, omega_circ, Heff, H, coeffs.nu, coeffs.fc)

    rdot = xi * dHdpr
    phidot = omega
    prdot = -xi * dHdr + Fr
    pphidot = Fphi
    return SEOBNRv5AlignedSpinState(rdot, phidot, prdot, pphidot)
end # @Tlit T
end

"""
    _seobv5_rk4_step(state, dt, coeffs) -> SEOBNRv5AlignedSpinState

One classical RK4 step, mirroring `AlignedSpinPNHam._asph_rk4_step` exactly.
"""
@inline function _seobv5_rk4_step(state::SEOBNRv5AlignedSpinState{T}, dt::T, coeffs) where {T}
@Tlit T begin
    k1 = _seobv5_rhs(state, coeffs)
    k2 = _seobv5_rhs(state + (dt / 2.0) * k1, coeffs)
    k3 = _seobv5_rhs(state + (dt / 2.0) * k2, coeffs)
    k4 = _seobv5_rhs(state + dt * k3, coeffs)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
end # @Tlit T
end

"""
    seobnrv5alignedspin_rk4_integrate(state0, dt, nsteps, coeffs) -> SEOBNRv5AlignedSpinState

Fixed-step RK4 integration of [`_seobv5_rhs`](@ref) from `state0`, `nsteps` steps of size `dt`
(reduced time units). Exported directly, mirroring `alignedspinpnham_rk4_integrate`'s rationale.
"""
@inline function seobnrv5alignedspin_rk4_integrate(state0::SEOBNRv5AlignedSpinState{T}, dt::T, nsteps::Integer, coeffs) where {T}
    state = state0
    for _ in 1:nsteps
        state = _seobv5_rk4_step(state, dt, coeffs)
    end
    return state
end

# --- Two-stage initial condition: IC_cons (2D Newton, circular) then IC_diss (1D Newton,
# dissipative pr* correction). See SEOBNRv5AlignedSpin's docstring's "Initial conditions" section.
# Plain Float64, one-time-per-waveform (mirrors AlignedSpinPNHam._asph_circular_ic). ---

"""
    _seobv5_circular_ic(omega0, hp) -> (r0, pphi0)

Stage 1 (`IC_cons`, DCC-note-cited Eq(60) of Khalil2021): fixed (8-iteration, no early-exit) 2D
Newton solve for `(r,pphi)` at `pr*=0` such that `dH/dpphi=omega0`, `dH/dr=0`. Exact Jacobian (the
Hessian at `pr*=0`, [`_seobv5_sym_circ_hessian`](@ref); not a finite difference). Seeded from `pyseobnr`'s own Newtonian guess (`initial_conditions_aligned_opt.py` line
102-103): `r_guess=omega0^(-2/3)`, `pphi_guess=sqrt(r_guess)`.

8 iterations (24 with the finite-difference Jacobian): with the exact one the iterates reach their round-off
floor (~1e-13 relative; beyond it they wander at that level) after 5 over q 1-20, |chi| < 0.99,
omega0 0.002-0.06.
"""
function _seobv5_circular_ic(omega0::Float64, hp)
    rhat = omega0^(-2.0 / 3.0)
    pphi = sqrt(rhat)
    for _ in 1:8
        dHdr, dHdpphi, Hrr, Hrp, Hpp = _seobv5_sym_circ_hessian(rhat, pphi, hp)
        F1 = dHdpphi - omega0
        F2 = dHdr
        J11 = Hrp   # dF1/dr
        J12 = Hpp   # dF1/dpphi
        J21 = Hrr   # dF2/dr
        J22 = Hrp   # dF2/dpphi
        det = J11 * J22 - J12 * J21
        delta_r = (-F1 * J22 + F2 * J12) / det
        delta_p = (-J11 * F2 + J21 * F1) / det
        rhat += delta_r
        pphi += delta_p
    end
    return rhat, pphi
end

"""
    _seobv5_dissipative_ic(r0, pphi0, hp, fc) -> pr0

Stage 2 (`IC_diss`, DCC-note-cited Eq(68) of Khalil2021): fixed (5-step, no early-exit) secant solve
for `pr*,0` such that `rdot(pr*) - dHdpr*(pr*) = 0`, `rdot = (1/xi)*Fphi/dLdr`, `dLdr = -d2Hdr2/d2HdrdL`,
with the exact Hessian entries of [`_seobv5_sym_diss_terms`](@ref) (not finite differences). Started from `pr* = 0` (in `pyseobnr`'s own `scipy.optimize.root_scalar` bracket
`[-3e-2,0]`, `initial_conditions_aligned_opt.py` line 118) and `-1e-4`, as the SEOBNRv5PHM IC.

The secant reaches round-off after 3 steps wherever a root exists (r0 > 11.5, q 1-20, |chi| < 0.99,
omega0 0.002-0.06): 7 residuals, each with the full flux.
"""
function _seobv5_dissipative_ic(r0::Float64, pphi0::Float64, hp, fc)
    function residual(prtrial::Float64)
        H0, xi0, Heff0, dHdpr0, omega, omega_circ, d2Hdr2, d2HdrdL = _seobv5_sym_diss_terms(r0, prtrial, pphi0, hp)
        dLdr = -d2Hdr2 / d2HdrdL
        _, Fphi = _seobv5_rr_force(prtrial, pphi0, omega, omega_circ, Heff0, H0, hp.nu, fc)

        rdot = (1.0 / xi0) * Fphi / dLdr
        return rdot - dHdpr0
    end

    p0, p1 = 0.0, -1e-4
    g0, g1 = residual(p0), residual(p1)
    for _ in 1:5
        den = g1 - g0
        p2 = ifelse(den == 0, p1, p1 - g1 * (p1 - p0) / ifelse(den == 0, one(den), den))
        p0, g0 = p1, g1
        p1 = p2
        g1 = residual(p1)
    end
    return p1
end

"""
    _seobv5_coeffs(model, mc, eta, chi1, chi2; f_min, GMsun_over_c3)

Time-independent part of the evaluation: mass fractions, spin combinations, calibration
coefficients, and the two-stage initial `(r,prst,pphi)` at `f_min` -- mirrors
`AlignedSpinPNHam._asph_coeffs`.
"""
@inline function _seobv5_coeffs(model::SEOBNRv5AlignedSpin,
    mc,
    eta,
    chi1,
    chi2;
    f_min = 20.0,
    GMsun_over_c3 = GMsun_over_c3,
)
    T = promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))

    eta_f64 = Float64(eta)
    delta_f64 = sqrt(max(1.0 - 4.0 * eta_f64, 0.0))
    X1_f64 = 0.5 * (1.0 + delta_f64)
    X2_f64 = 0.5 * (1.0 - delta_f64)
    chi1_f64 = Float64(chi1)
    chi2_f64 = Float64(chi2)

    ap_f64 = X1_f64 * chi1_f64 + X2_f64 * chi2_f64
    am_f64 = X1_f64 * chi1_f64 - X2_f64 * chi2_f64
    chiS_f64 = 0.5 * (chi1_f64 + chi2_f64)
    chiA_f64 = 0.5 * (chi1_f64 - chi2_f64)
    atest_f64 = (1.0 - 2.0 * eta_f64) * chiS_f64 + delta_f64 * chiA_f64

    a6_f64 = _seobv5_a6_NS(eta_f64)
    dSO_f64 = _seobv5_dSO(eta_f64, ap_f64, am_f64)

    absN33_f64 = _seobv5_multipole_prefix_abs(eta_f64, X1_f64, X2_f64, 3, 3)
    absN21_f64 = _seobv5_multipole_prefix_abs(eta_f64, X1_f64, X2_f64, 2, 1)
    absN44_f64 = _seobv5_multipole_prefix_abs(eta_f64, X1_f64, X2_f64, 4, 4)
    absN32_f64 = _seobv5_multipole_prefix_abs(eta_f64, X1_f64, X2_f64, 3, 2)
    absN55_f64 = _seobv5_multipole_prefix_abs(eta_f64, X1_f64, X2_f64, 5, 5)
    absN43_f64 = _seobv5_multipole_prefix_abs(eta_f64, X1_f64, X2_f64, 4, 3)

    M_sec_f64 = (Float64(mc) / Float64(eta)^0.6) * Float64(GMsun_over_c3)
    v0_f64 = cbrt(pi * M_sec_f64 * Float64(f_min))
    omega0_f64 = v0_f64^3

    hp_f64 = SEOBNRv5HamParams(eta_f64, X1_f64, X2_f64, chi1_f64, chi2_f64, a6_f64, dSO_f64)
    r0_f64, pphi0_f64 = _seobv5_circular_ic(omega0_f64, hp_f64)
    fc_f64 = _seobv5_flux_coeffs(eta_f64, delta_f64, chiS_f64, chiA_f64, atest_f64,
        absN33_f64, absN21_f64, absN44_f64, absN32_f64, absN55_f64, absN43_f64)
    pr0_f64 = _seobv5_dissipative_ic(r0_f64, pphi0_f64, hp_f64, fc_f64)

    GMsun_over_c3 = T(GMsun_over_c3)
    eta = T(eta)
    X1 = T(X1_f64)
    X2 = T(X2_f64)
    chi1 = T(chi1)
    chi2 = T(chi2)
    delta = T(delta_f64)
    chiS = T(chiS_f64)
    chiA = T(chiA_f64)
    atest = T(atest_f64)
    a6 = T(a6_f64)
    dSO = T(dSO_f64)

    fc = _seobv5_flux_coeffs(eta, delta, chiS, chiA, atest, T(absN33_f64), T(absN21_f64), T(absN44_f64),
        T(absN32_f64), T(absN55_f64), T(absN43_f64))
    hp = SEOBNRv5HamParams(eta, X1, X2, chi1, chi2, a6, dSO)

    r0 = T(r0_f64)
    pphi0 = T(pphi0_f64)
    pr0 = T(pr0_f64)
@Tlit T begin
    M = mc / (eta^(3.0 / 5.0))
    M_sec = M * GMsun_over_c3
    return (T = T, M = M, M_sec = M_sec, eta = eta, X1 = X1, X2 = X2, chi1 = chi1, chi2 = chi2,
        nu = eta, delta = delta, chiS = chiS, chiA = chiA, atest = atest,
        hp = hp, fc = fc, r0 = r0, pphi0 = pphi0, pr0 = pr0)
end # @Tlit T
end

"""
    _seobv5_state_scalar(coeffs, t, dt)

Non-allocating: integrates from the shared two-stage initial condition at `f_min` for
`round(Int, t/dt)` fixed RK4 steps of reduced-time size `dt/coeffs.M_sec`. `t`, `dt` in physical
seconds.
"""
@inline function _seobv5_state_scalar(coeffs, t, dt)
    T = coeffs.T
    state0 = SEOBNRv5AlignedSpinState(coeffs.r0, zero(T), coeffs.pr0, coeffs.pphi0)
    nsteps = round(Int, t / dt)
    dt_hat = T(dt) / coeffs.M_sec
    return seobnrv5alignedspin_rk4_integrate(state0, dt_hat, nsteps, coeffs)
end

"""
    Phi(model::SEOBNRv5AlignedSpin, t, mc, eta, chi1, chi2; f_min=20.0, dt=1.0/4096.0, GMsun_over_c3=GMsun_over_c3, phiRef=0)

`SEOBNRv5AlignedSpin` GW phase (`2x` orbital phase) at time(s) `t` seconds since the `f_min`-
referenced initial condition (`t>=0`) -- same domain convention as `AlignedSpinPNHam.Phi`.
"""
function Phi(model::SEOBNRv5AlignedSpin,
    t,
    mc,
    eta,
    chi1,
    chi2;
    f_min = 20.0,
    dt = 1.0 / 4096.0,
    GMsun_over_c3 = GMsun_over_c3,
    phiRef = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
)
    coeffs = _seobv5_coeffs(model, mc, eta, chi1, chi2; f_min = f_min, GMsun_over_c3 = GMsun_over_c3)
    T = coeffs.T
    phiRef = T(phiRef)
    if t isa AbstractArray
        out = similar(t, T)
        @inbounds for i in eachindex(t)
            state = _seobv5_state_scalar(coeffs, t[i], dt)
            out[i] = 2 * state.phi + phiRef
        end
        return out
    else
        state = _seobv5_state_scalar(coeffs, t, dt)
        return 2 * state.phi + phiRef
    end
end

"""
    Ampl(model::SEOBNRv5AlignedSpin, t, mc, eta, chi1, chi2, dL; f_min=20.0, dt=1.0/4096.0, GMsun_over_c3=GMsun_over_c3, GMsun_over_c2_Gpc=GMsun_over_c2_Gpc)

Restricted (Newtonian-order) time-domain amplitude, same convention as `AlignedSpinPNHam.Ampl`:
`h+ = -Ampl(t)*cos(Phi(t))`, `hx = -Ampl(t)*sin(Phi(t))` at zero inclination, `Ampl(t) =
4*eta*M*GMsun_over_c2_Gpc/dL * v(t)^2`, `v(t)=Omega_hat(t)^(1/3)` sourced from this model's own
integrated state via `dH/dpphi`.
"""
function Ampl(model::SEOBNRv5AlignedSpin,
    t,
    mc,
    eta,
    chi1,
    chi2,
    dL;
    f_min = 20.0,
    dt = 1.0 / 4096.0,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    coeffs = _seobv5_coeffs(model, mc, eta, chi1, chi2; f_min = f_min, GMsun_over_c3 = GMsun_over_c3)
    T = coeffs.T
    GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc)
    dL = T(dL)
    if t isa AbstractArray
        out = similar(t, T)
        @inbounds for i in eachindex(t)
            state = _seobv5_state_scalar(coeffs, t[i], dt)
            _, _, omega = _seobv5_sym_gradient(state.r, state.prst, state.pphi, coeffs.hp)
            v = cbrt(max(omega, eps(T)))
            out[i] = 4 * coeffs.eta * coeffs.M * GMsun_over_c2_Gpc / dL * v * v
        end
        return out
    else
        state = _seobv5_state_scalar(coeffs, t, dt)
        _, _, omega = _seobv5_sym_gradient(state.r, state.prst, state.pphi, coeffs.hp)
        v = cbrt(max(omega, eps(T)))
        return 4 * coeffs.eta * coeffs.M * GMsun_over_c2_Gpc / dL * v * v
    end
end
