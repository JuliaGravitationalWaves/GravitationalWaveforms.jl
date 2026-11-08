export TaylorT3, validate_taylort3_params

"""
    TaylorT3

Non-spinning, restricted-amplitude, closed-form time-domain post-Newtonian inspiral model.

Ported verbatim from `lalsimulation`'s legacy `TaylorT3` generator:
`lalsimulation/lib/LALSimInspiralTaylorT3.c` (phasing/frequency evolution equations, restricted
to `amplitudeO = 0`) and `lalsimulation/lib/LALSimInspiralPNCoefficients.c` (the numeric PN
coefficients themselves), fetched from the canonical repository
`https://git.ligo.org/lscsoft/lalsuite` (`master` branch). Physics reference: Alessandra
Buonanno, Bala R. Iyer, Evan Ochsner, Yi Pan, B. S. Sathyaprakash, "Comparison of post-Newtonian
templates for compact binary inspiral signals in gravitational-wave detectors", Phys. Rev. D 80,
084043 (2009), arXiv:0907.0700 -- the source code comments cite eq. (3.10a) (phasing) and
(3.10b) (frequency) of that paper for the coefficient values.

# PN order
The legacy generator's phasing/frequency coefficient arrays (`pta0..pta7`, `fta0..fta7`, plus
the 6PN log terms `ptl6`/`ftl6`) stop at `pta7`/`fta7` -- i.e. **3.5PN** order (LAL's own
"twice-PN-order" convention: `O=7` <=> 3.5PN) -- confirmed directly from
`XLALSimInspiralTaylorT3Setup`'s `switch (O)` (case `7`/`-1` selects
`XLALSimInspiralPhasing3_7PN`/`XLALSimInspiralFrequency3_7PN`, the highest available). This
implementation hardcodes that same (highest-available) 3.5PN order -- LAL exposes phase-order
truncation as a run-time option, but this package's "closed-form PN phasing" design goal doesn't
call for exposing that knob.

# Parametrization: `theta`
Following the source exactly, define the dimensionless "reduced time" variable
`theta = (eta*td/(5*M))^(-1/8)`, where `M` is the total mass in seconds
(`= (mc/eta^(3/5)) * GMsun_over_c3`) and `td` is (nominally) "time until coalescence". **This
package's `t` argument to `Phi`/`Ampl` matches LALSuite's `choose_td_waveform`/
`SimInspiralChooseTDWaveform` output convention exactly** (`t <= 0`, coalescence nominally at
`t = 0`), via
`td = t0_shift - t`, where `t0_shift` corrects for a genuine subtlety below -- **it is *not*
simply `-t`.**

## `t0_shift`: LAL's `t = 0` is not the formula's own coalescence singularity
Naively, one might assume `td = -t` directly (`t = 0` <=> the formal PN coalescence, where
`theta` diverges) -- **this is wrong for TaylorT3**, as cross-validation
against real `lalsimulation` shows: comparing
against real output showed a real, systematically *growing* discrepancy, present from the very
first sample, that persisted even at pure-Newtonian (0PN) phasing (ruling out any higher-PN
coefficient bug). The root cause: `XLALSimInspiralTaylorT3PNEvolveOrbit`'s integration loop
terminates the moment `v >= v_isco = 1/sqrt(6)` **or** the frequency series stops increasing
(`f <= fOld` -- this truncated 3.5PN series is well known to turn over well before `v_isco` for
many mass ratios, confirmed directly for this ecosystem's own baseline masses), *not* when
`theta` reaches its own formal singularity. LAL's output-array epoch-shift convention
nonetheless always treats "one sample past the last generated sample" as `t = 0`, for
bookkeeping -- even though that point is really wherever the loop happened to be cut off, which
is generically *before* the formula's own theta-diverges point.

`t0_shift` (computed once per waveform by [`_taylort3_t0_shift`](@ref), a closed-form
root-find -- no ODE integration) is exactly that gap: `theta_stop`, the `theta` value at which
LAL's own termination condition would trigger, mapped back to a time interval via
`t0_shift = (5*M/eta) * theta_stop^(-8)`. With this correction, `t = 0` (LAL's own
"coalescence") maps to `theta = theta_stop` (finite, physically meaningful -- not `theta = ∞`),
and `t <= 0` never reaches the formula's actual singularity at all (that would require
`t = t0_shift > 0`, outside this model's expected input domain) -- a strictly better-behaved
domain than the naive `-t` parametrization.

# Phase convention: GW phase (2x orbital), not orbital phase
The source's `phasing3` functions return **orbital** phase (the array is even named
`"ORBITAL_PHASE"` in `XLALSimInspiralTaylorT3PNEvolveOrbit`), and the restricted-amplitude
polarizations use `cos(2*phi_orb)`/`sin(2*phi_orb)` directly (see
`XLALSimInspiralPNPolarizationWaveforms`'s `ampO=0` case: `hp0 = -(1+ci^2)*cos(2*phi)`,
`hc0 = -2*ci*sin(2*phi)`). To match this package's own `IMRPhenomD`/`IMRPhenomXAS` convention -- where
`h ~ Ampl(f)*exp(i*Phi(f))`, i.e. `Phi` IS the phase that appears directly inside the
trig/exponential, not something that needs an extra factor of 2 at the call site -- **`Phi`
here returns `2*phi_orb`, the GW phase**, not the bare orbital phase. This also makes
`Phi`/`Ampl` mutually consistent for the "(2,2)-mode-only, face-on" restricted-amplitude
convention documented on [`Ampl`](@ref): `h+ = -Ampl(t)*cos(Phi(t))`,
`hx = -Ampl(t)*sin(Phi(t))` at zero inclination (matching `IMRPhenomD.Ampl`'s own documented
face-on convention).

# The `theta_lso` reference scale (5PN+ log term)
The 5PN-and-higher phasing/frequency terms contain a term `pta5*log(theta/theta_lso)*theta^5`
(phasing) / an analogous 6PN `log(2*theta)` term (frequency) — `theta_lso` is a *reference*
value of `theta` (not a physical cutoff enforced here), computed once per waveform from
`XLALSimInspiralTaylorLength`: `theta_lso = tn^(-1/8)` where
`tn = -M*sign*integral(dE/dv / F(v), v0, vlso) - deltaT`, `v0` is the PN velocity parameter at
`f_min`, and `vlso = min(1/sqrt(6), (pi*M/(2*deltaT))^(1/3))` (almost always `1/sqrt(6)`, the
Schwarzschild ISCO velocity, for realistic `deltaT`/masses — the Nyquist-velocity cap only binds
for pathologically large `deltaT`). This requires numerically integrating the same 3PN
energy/3.5PN flux functions LAL itself uses there (`XLALSimInspiraldEt6`/`XLALSimInspiralFt7`),
which this file replicates with a from-scratch adaptive Simpson quadrature (`_taylort3_*`
helpers below) rather than porting LAL's GSL-based adaptive Gauss-Kronrod call -- the two
should agree far beyond the precision this sub-leading, logarithmic-scale reference term needs
(GSL's own default tolerance there is only `1e-7} relative). One genuine, documented LAL source
quirk: `XLALSimInspiralTaylorLength`'s local `expnCoeffsdEnergyFlux akEF` struct never assigns
its `dETa5`/`dETa6` fields (5PN/6PN spin-related energy terms that don't exist in the
non-spinning 3PN energy expansion `XLALSimInspiraldEt6` is actually evaluating) before they're
read inside `XLALSimInspiraldEt6` -- technically reading uninitialized stack memory in C. This
implementation takes the physically-intended value, `0.0`, for both (consistent with them being
5PN/6PN spin-coupling coefficients that simply aren't part of any non-spinning PN energy
expansion at this order) rather than attempting to reproduce whatever incidental garbage a
particular compiler/stack state produced -- see `_taylort3_theta_lso`'s docstring. `f_min` and
`deltaT` are therefore real (if usually tiny-effect) keyword inputs to `Phi`/`Ampl` here, not
just LAL-side data-generation settings; defaults (`f_min=20.0`, `deltaT=1/4096`) match the
other LALSuite comparisons.
"""
struct TaylorT3 end

"""
    validate_taylort3_params(eta; dL = nothing)

Host-side sanity check: `0 < eta <= 0.25`, and (if `dL` given) `dL > 0`. Throws `ArgumentError`
on violation, `nothing` otherwise. Not spin-checked (TaylorT3 is non-spinning by construction --
LAL's own legacy generator rejects non-zero spins outright for this approximant).

Not called by `Phi`/`Ampl` themselves, matching `validate_phenomd_params`'s documented rationale
(hot-path; one-time domain checks belong at the sampler/prior level).
"""
function validate_taylort3_params(eta; dL = nothing)
    0.0 < eta <= 0.25 || throw(ArgumentError("Invalid eta = $eta: must be in (0, 0.25]"))
    if dL !== nothing
        dL > 0.0 || throw(ArgumentError("Invalid dL = $dL: luminosity distance must be positive (and must be in Gpc, see Ampl's docstring)"))
    end
    return nothing
end

# --- theta_lso: a from-scratch adaptive Simpson quadrature (Float64 only -- this feeds a small,
# logarithmic-scale reference constant computed once per waveform, not a hot per-sample path, so
# there's no genericity/@Tlit requirement here; doing it in Float64 unconditionally also avoids
# any Float32 quadrature-robustness surprises for a term whose contribution is already tiny). ---

@inline function _taylort3_adaptive_simpson_step(f, a, b, tol, whole, fa, fb, fm, depth)
    m = 0.5 * (a + b)
    lm = 0.5 * (a + m)
    rm = 0.5 * (m + b)
    flm = f(lm)
    frm = f(rm)
    left = (m - a) / 6.0 * (fa + 4.0 * flm + fm)
    right = (b - m) / 6.0 * (fm + 4.0 * frm + fb)
    delta = left + right - whole
    # The `depth <= 0` cutoff is not just a safety backstop: the classic "halve the tolerance
    # every level" recursive Simpson scheme has no floor for floating-point roundoff noise, so
    # once the interval width approaches the local `eps` of `a`/`b` (here, widths around
    # 1e-16-relative near v ~ 0.2-0.4), `delta` stops shrinking with further subdivision
    # (dominated by roundoff, not truncation error) while the required `15*tol` threshold keeps
    # shrinking geometrically without bound -- so an over-generous `maxdepth` (e.g. the first
    # version of this function, `maxdepth=60`: 60 halvings from a ~0.15-wide interval reaches
    # sub-`eps` widths around level ~25 and then never converges) makes this recurse toward
    # `2^maxdepth` calls, an effective infinite loop (confirmed directly: reproduced a hang that
    # burned 600+ CPU-seconds before being killed, then isolated it to exactly this with a
    # call-count guard in a standalone script). `maxdepth=24` here is comfortably above the
    # handful of levels (~10-15) this smooth, non-singular integrand actually needs to satisfy
    # `tol`, while its worst realistic case (`2^24` calls, should the tolerance check never
    # fire) still finishes in well under a second -- unlike `2^60`.
    if depth <= 0 || abs(delta) <= 15.0 * tol
        return left + right + delta / 15.0
    end
    return _taylort3_adaptive_simpson_step(f, a, m, tol / 2.0, left, fa, fm, flm, depth - 1) +
           _taylort3_adaptive_simpson_step(f, m, b, tol / 2.0, right, fm, fb, frm, depth - 1)
end

"""
    _taylort3_adaptive_simpson(f, a, b; tol=1e-13, maxdepth=24)

Standard recursive adaptive Simpson quadrature (Float64), used only for the `theta_lso`
integral (see `TaylorT3`'s docstring) -- a generic, well-known numerical algorithm, not
something ported from LAL (LAL uses GSL's adaptive Gauss-Kronrod, `gsl_integration_qags`, at a
`1e-7` relative tolerance; this reaches `tol` far tighter than that on the smooth,
non-singular integrand encountered here). See `_taylort3_adaptive_simpson_step`'s comment for
why `maxdepth` is capped well below floating-point-precision-limited recursion depths.
"""
function _taylort3_adaptive_simpson(f, a::Float64, b::Float64; tol::Float64 = 1e-13, maxdepth::Int = 24)
    fa = f(a)
    fb = f(b)
    m = 0.5 * (a + b)
    fm = f(m)
    whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb)
    return _taylort3_adaptive_simpson_step(f, a, b, tol, whole, fa, fb, fm, maxdepth)
end

# dE/dv at 3PN ("Et6" in LAL's v^6-labeled naming), verbatim from LALSimInspiraldEnergyFlux.c's
# XLALSimInspiraldEt6 combined with LALSimInspiralPNCoefficients.c's XLALSimInspiralPNEnergy_*
# (dETaN = 2*Energy_0PN, dETa1 = 2*Energy_2PN, dETa2 = 3*Energy_4PN, dETa3 = 4*Energy_6PN, per
# XLALSimInspiralTaylorLength). dETa5/dETa6 (5PN/6PN spin terms, not part of the non-spinning
# energy expansion at this order) are taken as 0.0 -- see TaylorT3's docstring.
@inline function _taylort3_dEnergy_dv(v::Float64, eta::Float64)
    dETaN = -eta
    dETa1 = -(1.5 + eta / 6.0)
    dETa2 = -(81.0 / 8.0 - 57.0 / 8.0 * eta + eta * eta / 8.0)
    dETa3 = -4.0 * (67.5 / 6.4 - (344.45 / 5.76 - 20.5 / 9.6 * pi * pi) * eta +
                     15.5 / 9.6 * eta * eta + 3.5 / 518.4 * eta * eta * eta)
    x = v * v
    x2 = x * x
    x3 = x2 * x
    return dETaN * v * (1.0 + dETa1 * x + dETa2 * x2 + dETa3 * x3)
end

# GW flux at 3.5PN ("Ft7"), verbatim from LALSimInspiraldEnergyFlux.c's XLALSimInspiralFt7
# combined with LALSimInspiralPNCoefficients.c's XLALSimInspiralPNFlux_*PNCoeff functions.
@inline function _taylort3_flux(v::Float64, eta::Float64, LAL_GAMMA::Float64)
    FTaN = 32.0 * eta * eta / 5.0
    FTa2 = -(12.47 / 3.36 + 3.5 / 1.2 * eta)
    FTa3 = 4.0 * pi
    FTa4 = -(44.711 / 9.072 - 92.71 / 5.04 * eta - 6.5 / 1.8 * eta * eta)
    FTa5 = -(81.91 / 6.72 + 58.3 / 2.4 * eta) * pi
    FTa6 = 664.3739519 / 6.9854400 + 16.0 / 3.0 * pi * pi - 17.12 / 1.05 * LAL_GAMMA -
           17.12 / 1.05 * log(4.0) + (4.1 / 4.8 * pi * pi - 134.543 / 7.776) * eta -
           94.403 / 3.024 * eta * eta - 7.75 / 3.24 * eta * eta * eta
    FTl6 = -17.12 / 1.05
    FTa7 = -(162.85 / 5.04 - 214.745 / 1.728 * eta - 193.385 / 3.024 * eta * eta) * pi
    v2 = v * v
    v3 = v2 * v
    v4 = v2 * v2
    v5 = v4 * v
    v6 = v4 * v2
    v7 = v6 * v
    v10 = v6 * v4
    return FTaN * v10 * (1.0 + FTa2 * v2 + FTa3 * v3 + FTa4 * v4 + FTa5 * v5 +
                          (FTa6 + FTl6 * log(v)) * v6 + FTa7 * v7)
end

"""
    _taylort3_theta_lso(M_sec, eta, f_min, deltaT, LAL_GAMMA) -> Float64

Reference `theta` scale for the 5PN+ logarithmic phasing/frequency terms, replicating
`XLALSimInspiralTaylorLength` (see `TaylorT3`'s docstring for the full derivation and the
`dETa5`/`dETa6 = 0.0` note). All arguments and the return value are plain `Float64`; the caller
casts the result to the working type `T`.
"""
function _taylort3_theta_lso(M_sec::Float64, eta::Float64, f_min::Float64, deltaT::Float64, LAL_GAMMA::Float64)
    v0 = cbrt(pi * M_sec * f_min)
    vlso = sqrt(1.0 / 6.0)
    vn_nyquist = cbrt(pi * M_sec / (2.0 * deltaT))
    vn = min(vn_nyquist, vlso)
    integrand(v) = _taylort3_dEnergy_dv(v, eta) / _taylort3_flux(v, eta, LAL_GAMMA)
    if v0 > vn
        integral = _taylort3_adaptive_simpson(integrand, vn, v0)
        sgn = -1.0
    else
        integral = _taylort3_adaptive_simpson(integrand, v0, vn)
        sgn = 1.0
    end
    tofv = M_sec * integral * sgn
    tn = -tofv - deltaT
    return tn^(-0.125)
end

# --- t0_shift: correcting for LAL's actual `theta=t=0` convention. ---
#
# Assuming this package's `t = 0`
# means "the literal, formal PN coalescence" (where `theta -> infinity`) is WRONG for TaylorT3.
# `XLALSimInspiralTaylorT3PNEvolveOrbit`'s integration loop terminates the moment EITHER
# `v >= v_isco = 1/sqrt(6)` OR the frequency series stops increasing (`f <= fOld`, a well-known
# tendency of this truncated 3.5PN series to turn over well before reaching v_isco for many mass
# ratios) -- NOT when `theta` actually reaches its formal singularity. LAL's own output-array
# convention nonetheless treats "one sample past the last generated sample" as `t = 0`
# ("coalescence", for bookkeeping), even though that point is really wherever the loop was cut
# off, not the theta-formula's own singular point. Reproduced and confirmed directly: comparing
# against real `lalsimulation` output (even at pure Newtonian/0PN phasing, to rule out a
# higher-PN-coefficient bug) showed a real, systematically growing discrepancy -- present even
# at the very first sample -- that vanished once this shift was included.
#
# Fix: `theta(t)` uses `(theta_stop_shift - t)` instead of bare `(-t)`, where `theta_stop_shift`
# is derived from `theta_stop`, the value of `theta` at which LAL's own termination condition
# would trigger (computed once per waveform via closed-form root-finding on the frequency
# series itself -- no ODE integration needed, matching this model's overall design):
# `theta_stop = min(theta at the frequency series' own local maximum, theta at v = v_isco)`,
# whichever occurs first (i.e. at smaller `theta`). Then, since `theta = td^(-1/8)` and
# `td = 5*M/eta * theta^(-8)` represents (in LAL's own loop) the remaining time to the *formal*
# coalescence, `theta_stop_shift = (5*M_sec/eta) * theta_stop^(-8)` is exactly the gap between
# that formal coalescence and LAL's own `t = 0` reference point.

@inline function _taylort3_freq_f64_coeffs(eta::Float64, LAL_GAMMA::Float64)
    eta2 = eta * eta
    eta3 = eta2 * eta
    fta2 = 7.43 / 26.88 + 1.1 / 3.2 * eta
    fta3 = -3.0 / 10.0 * pi
    fta4 = 1.855099 / 14.450688 + 5.6975 / 25.8048 * eta + 3.71 / 20.48 * eta2
    fta5 = (-7.729 / 21.504 + 1.3 / 25.6 * eta) * pi
    fta6 = -7.20817631400877 / 2.88412611379200 + 5.3 / 20.0 * pi * pi + 1.07 / 2.80 * LAL_GAMMA +
           (25.302017977 / 4.161798144 - 4.51 / 20.48 * pi * pi) * eta -
           3.0913 / 183.5008 * eta2 + 2.35925 / 17.69472 * eta3
    ftl6 = 1.07 / 2.80
    fta7 = (-1.88516689 / 4.33520640 - 9.7765 / 25.8048 * eta + 1.41769 / 12.90240 * eta2) * pi
    return (fta2 = fta2, fta3 = fta3, fta4 = fta4, fta5 = fta5, fta6 = fta6, ftl6 = ftl6, fta7 = fta7)
end

@inline function _taylort3_freq_of_theta_f64(theta::Float64, ftaN::Float64, c)
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta3 * theta
    theta5 = theta4 * theta
    theta6 = theta5 * theta
    theta7 = theta6 * theta
    return ftaN * theta3 * (1.0 + c.fta2 * theta2 + c.fta3 * theta3 + c.fta4 * theta4 + c.fta5 * theta5 +
                             (c.fta6 + c.ftl6 * log(2.0 * theta)) * theta6 + c.fta7 * theta7)
end

"""
    _taylort3_golden_section_max(f, a, b; tol=1e-13, maxiter=200) -> Float64

Standard golden-section search for the location of a maximum of a unimodal `f` on `[a, b]`
(Float64) -- a generic numerical algorithm, used to find `theta` at the frequency series' own
local maximum (see the `t0_shift` block's comment above).
"""
function _taylort3_golden_section_max(f, a::Float64, b::Float64; tol::Float64 = 1e-13, maxiter::Int = 200)
    gr = (sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in 1:maxiter
        abs(b - a) < tol && break
        if fc > fd
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = f(c)
        else
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = f(d)
        end
    end
    return 0.5 * (a + b)
end

"""
    _taylort3_theta_stop(M_sec, eta, LAL_GAMMA) -> Float64

`theta` at which LAL's own `XLALSimInspiralTaylorT3PNEvolveOrbit` integration loop would
terminate: either where the (3.5PN-truncated) frequency series reaches its own local maximum
(a coarse grid scan over a generous bracket, refined via golden section, since the series can
turn over well before reaching the ISCO velocity for many mass ratios -- confirmed directly for
this ecosystem's own baseline masses), or where `v = 1/sqrt(6)` (Schwarzschild ISCO) is reached
first, whichever corresponds to the smaller `theta`. See the block comment above for the full
derivation and why this matters.
"""
function _taylort3_theta_stop(M_sec::Float64, eta::Float64, LAL_GAMMA::Float64)
    ftaN = 1.0 / (8.0 * pi * M_sec)
    c = _taylort3_freq_f64_coeffs(eta, LAL_GAMMA)
    freq(theta) = _taylort3_freq_of_theta_f64(theta, ftaN, c)

    # Coarse scan for the frequency series' own local maximum, over a generously wide bracket
    # (theta this small/large is never physically reached, but bounds the search safely).
    lo, hi, ngrid = 0.02, 3.0, 600
    dtheta = (hi - lo) / (ngrid - 1)
    best_i = 1
    best_f = freq(lo)
    prev_f = best_f
    for i in 2:ngrid
        th = lo + (i - 1) * dtheta
        f = freq(th)
        if f > best_f
            best_f = f
            best_i = i
        end
        prev_f = f
    end
    bracket_a = max(lo, lo + (best_i - 2) * dtheta)
    bracket_b = min(hi, lo + best_i * dtheta)
    theta_max = _taylort3_golden_section_max(freq, bracket_a, bracket_b)
    f_peak = freq(theta_max)

    visco = sqrt(1.0 / 6.0)
    f_isco_target = visco^3 / (pi * M_sec)

    if f_peak <= f_isco_target
        # The truncated series turns over before ever reaching the ISCO velocity -- the
        # "frequency stalled" condition wins, matching `XLALSimInspiralTaylorT3PNEvolveOrbit`'s
        # `if (f <= fOld) break;`.
        return theta_max
    else
        # ISCO is reached first (`if (v >= visco) break;`) -- bisect for theta on the
        # monotonically-increasing branch below theta_max.
        a, b = lo, theta_max
        ga = freq(a) - f_isco_target
        for _ in 1:200
            m = 0.5 * (a + b)
            gm = freq(m) - f_isco_target
            if sign(gm) == sign(ga)
                a = m
                ga = gm
            else
                b = m
            end
        end
        return 0.5 * (a + b)
    end
end

"""
    _taylort3_t0_shift(M_sec, eta, LAL_GAMMA) -> Float64

The gap between the theta-formula's own formal (unreached) coalescence and LAL's own `t = 0`
output-array convention (see the block comment above `_taylort3_freq_f64_coeffs`). All
arguments and the return value are plain `Float64`; the caller casts the result to the working
type `T`.
"""
function _taylort3_t0_shift(M_sec::Float64, eta::Float64, LAL_GAMMA::Float64)
    theta_stop = _taylort3_theta_stop(M_sec, eta, LAL_GAMMA)
    return (5.0 * M_sec / eta) * theta_stop^(-8.0)
end

"""
    _taylort3_coeffs(model::TaylorT3, mc, eta; kwargs...)

Time-independent part of the TaylorT3 phasing/frequency/amplitude calculation -- computed once
per waveform (mirroring `IMRPhenomD`'s `_phi_coeffs`/`_ampl_coeffs` split), not per time sample.
"""
@inline function _taylort3_coeffs(model::TaylorT3,
    mc,
    eta;
    f_min = 20.0,
    deltaT = 1.0 / 4096.0,
    theta_lso = nothing,
    t0_shift = nothing,
    LAL_GAMMA = 0.577215664901532860606512090082402431,
    GMsun_over_c3 = GMsun_over_c3,
)
    T = promote_type(typeof(mc), typeof(eta))

    # theta_lso's quadrature and t0_shift's root-finds (see docstrings above) run in plain
    # Float64, independent of T -- compute them from Float64 versions of the (possibly-T,
    # possibly-kwarg-overridden) inputs before doing any T-casting/@Tlit-wrapped arithmetic below.
    M_sec_f64 = Float64(mc) / (Float64(eta)^0.6) * Float64(GMsun_over_c3)
    thlso_f64 = theta_lso === nothing ?
        _taylort3_theta_lso(M_sec_f64, Float64(eta), Float64(f_min), Float64(deltaT), Float64(LAL_GAMMA)) :
        Float64(theta_lso)
    t0shift_f64 = t0_shift === nothing ?
        _taylort3_t0_shift(M_sec_f64, Float64(eta), Float64(LAL_GAMMA)) :
        Float64(t0_shift)

    # Keyword-derived scalars aren't reached by the `@Tlit` rewrite below (see TypeGenericity.jl's
    # docstring) -- cast them explicitly, matching IMRPhenomD.jl's established pattern.
    LAL_GAMMA = T(LAL_GAMMA)
    GMsun_over_c3 = T(GMsun_over_c3)
    thlso = T(thlso_f64)
    t0shift = T(t0shift_f64)

@Tlit T begin
    M = mc / (eta^(3.0 / 5.0))
    M_sec = M * GMsun_over_c3
    eta2 = eta * eta
    eta3 = eta2 * eta

    ptaN = -1.0 / eta
    pta2 = 3.715 / 8.064 + 5.5 / 9.6 * eta
    pta3 = -3.0 / 4.0 * pi
    pta4 = 9.275495 / 14.450688 + 2.84875 / 2.58048 * eta + 1.855 / 2.048 * eta2
    pta5 = (3.8645 / 2.1504 - 6.5 / 25.6 * eta) * pi
    pta6 = 83.1032450749357 / 5.7682522275840 - 5.3 / 4.0 * pi * pi - 10.7 / 5.6 * LAL_GAMMA +
           (-126.510089885 / 4.161798144 + 2.255 / 2.048 * pi * pi) * eta +
           1.54565 / 18.35008 * eta2 - 1.179625 / 1.769472 * eta3
    ptl6 = -10.7 / 5.6
    pta7 = (1.88516689 / 1.73408256 + 4.88825 / 5.16096 * eta - 1.41769 / 5.16096 * eta2) * pi

    ftaN = 1.0 / (8.0 * pi * M_sec)
    fta2 = 7.43 / 26.88 + 1.1 / 3.2 * eta
    fta3 = -3.0 / 10.0 * pi
    fta4 = 1.855099 / 14.450688 + 5.6975 / 25.8048 * eta + 3.71 / 20.48 * eta2
    fta5 = (-7.729 / 21.504 + 1.3 / 25.6 * eta) * pi
    fta6 = -7.20817631400877 / 2.88412611379200 + 5.3 / 20.0 * pi * pi + 1.07 / 2.80 * LAL_GAMMA +
           (25.302017977 / 4.161798144 - 4.51 / 20.48 * pi * pi) * eta -
           3.0913 / 183.5008 * eta2 + 2.35925 / 17.69472 * eta3
    ftl6 = 1.07 / 2.80
    fta7 = (-1.88516689 / 4.33520640 - 9.7765 / 25.8048 * eta + 1.41769 / 12.90240 * eta2) * pi

    return (
        T = T, M = M, M_sec = M_sec, eta = eta,
        ptaN = ptaN, pta2 = pta2, pta3 = pta3, pta4 = pta4, pta5 = pta5, pta6 = pta6, ptl6 = ptl6, pta7 = pta7,
        ftaN = ftaN, fta2 = fta2, fta3 = fta3, fta4 = fta4, fta5 = fta5, fta6 = fta6, ftl6 = ftl6, fta7 = fta7,
        theta_lso = thlso, t0_shift = t0shift,
    )
end # @Tlit T
end

"""
    _taylort3_phi_orb_scalar(coeffs, t)

Non-allocating orbital-phase evaluation at a single time `t` (seconds relative to LAL's own
`t = 0` convention, `t <= 0` -- see `TaylorT3`'s docstring's `t0_shift` section for why this is
*not* the literal formal coalescence), verbatim from `XLALSimInspiralPhasing3_7PN`. See
`TaylorT3`'s docstring for the phase convention (`Phi` doubles this to the GW phase).
"""
@inline function _taylort3_phi_orb_scalar(coeffs, t)
    T = coeffs.T
@Tlit T begin
    td_arg = max(coeffs.t0_shift - t, eps(T))
    theta = (coeffs.eta * td_arg / (5.0 * coeffs.M_sec))^(-1.0 / 8.0)
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta3 * theta
    theta5 = theta4 * theta
    theta6 = theta5 * theta
    theta7 = theta6 * theta

    return (coeffs.ptaN / theta5) * (1.0 +
        coeffs.pta2 * theta2 +
        coeffs.pta3 * theta3 +
        coeffs.pta4 * theta4 +
        coeffs.pta5 * log(theta / coeffs.theta_lso) * theta5 +
        (coeffs.ptl6 * log(2.0 * theta) + coeffs.pta6) * theta6 +
        coeffs.pta7 * theta7)
end # @Tlit T
end

"""
    _taylort3_freq_scalar(coeffs, t)

Non-allocating instantaneous GW-frequency evaluation at a single time `t` (Hz; `f_GW`, already
twice the orbital frequency -- see `TaylorT3`'s docstring), verbatim from
`XLALSimInspiralFrequency3_7PN`.
"""
@inline function _taylort3_freq_scalar(coeffs, t)
    T = coeffs.T
@Tlit T begin
    td_arg = max(coeffs.t0_shift - t, eps(T))
    theta = (coeffs.eta * td_arg / (5.0 * coeffs.M_sec))^(-1.0 / 8.0)
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta3 * theta
    theta5 = theta4 * theta
    theta6 = theta5 * theta
    theta7 = theta6 * theta

    return coeffs.ftaN * theta3 * (1.0 +
        coeffs.fta2 * theta2 +
        coeffs.fta3 * theta3 +
        coeffs.fta4 * theta4 +
        coeffs.fta5 * theta5 +
        (coeffs.fta6 + coeffs.ftl6 * log(2.0 * theta)) * theta6 +
        coeffs.fta7 * theta7)
end # @Tlit T
end

"""
    _taylort3_ampl_scalar(coeffs, t, GMsun_over_c2_Gpc, dL)

Non-allocating restricted (Newtonian/leading-order) amplitude evaluation at a single time `t`,
verbatim (prefactor included) from `XLALSimInspiralPNPolarizationWaveforms`'s `ampO = 0` case
-- see `Ampl`'s docstring for the derivation of the `4*eta*M*GMsun_over_c2_Gpc/dL` prefactor
from LAL's `ampfac = 2*M*G/c^3*eta/dist` and `hp0 = -(1+ci^2)*cos(2*phi)` at face-on (`ci = 1`).

`v` is computed via `cbrt` (real cube root, sign-preserving), not `(pi*M*f)^(1/3)`/`^(2/3)`
directly, matching LAL's own computational graph exactly: `XLALSimInspiralTaylorT3PNEvolveOrbit`
computes `v = cbrt(f * LAL_PI * m)` (C's `cbrt` is defined and real-valued for negative `f`) and
it's *that* `v` time series `XLALSimInspiralPNPolarizationWaveforms` later squares for the
amplitude -- not a fresh `pow(pi*M*f, 2.0/3.0)` recomputed from `f`. This distinction is not
just a Julia-vs-C technicality: `f_gw` legitimately, transiently dips slightly
negative for a handful of samples very close to coalescence for some mass ratios (the truncated
3.5PN frequency series' well-documented tendency to misbehave right at the edge of its validity
range), and `(negative)^(2/3)` via generic exponentiation is complex-valued (Julia throws;
C's `pow` would give `NaN`), whereas `cbrt(negative)^2` is a well-defined positive real number,
exactly reproducing what LAL's actual C code computes there.
"""
@inline function _taylort3_ampl_scalar(coeffs, t, GMsun_over_c2_Gpc, dL)
    T = coeffs.T
@Tlit T begin
    f_gw = _taylort3_freq_scalar(coeffs, t)
    v = cbrt(pi * coeffs.M_sec * f_gw)
    v2 = v * v
    return 4.0 * coeffs.eta * coeffs.M * GMsun_over_c2_Gpc / dL * v2
end # @Tlit T
end

"""
    Phi(model::TaylorT3, t, mc, eta; kwargs...)

TaylorT3 time-domain GW phase (`2x` orbital phase -- see `TaylorT3`'s docstring for why). `t`
may be a scalar or an `AbstractVector` of times **in seconds relative to LAL's own `t = 0`
convention** (`t <= 0`, nominally "coalescence" at `t = 0` -- see `TaylorT3`'s docstring's
`t0_shift` section for why this is *not* the formula's own literal coalescence singularity),
matching LALSuite's `SimInspiralChooseTDWaveform` output convention directly.

# Keywords
- `f_min`, `deltaT`: feed the `theta_lso` reference-scale computation and (`f_min` only) the
  `t0_shift` computation (see `TaylorT3`'s docstring); defaults (`20.0` Hz, `1/4096` s) match
  this ecosystem's established baseline. Pass `theta_lso`/`t0_shift` directly (bypassing those
  computations) if they're already known.
- `phiRef`: additive constant phase offset (gauge freedom), default `0`.
- `GMsun_over_c3`, `LAL_GAMMA`: physical constants, overridable for exact cross-validation
  against a specific `lalsimulation` build's own constant values (mirroring `IMRPhenomD.Phi`'s
  `GMsun_over_c3` keyword).

No parameter-domain assertions run here -- see `validate_taylort3_params` for an opt-in
host-side check.
"""
function Phi(model::TaylorT3,
    t,
    mc,
    eta;
    f_min = 20.0,
    deltaT = 1.0 / 4096.0,
    theta_lso = nothing,
    t0_shift = nothing,
    LAL_GAMMA = 0.577215664901532860606512090082402431,
    GMsun_over_c3 = GMsun_over_c3,
    phiRef = zero(promote_type(typeof(mc), typeof(eta))),
)
    coeffs = _taylort3_coeffs(model, mc, eta; f_min = f_min, deltaT = deltaT,
        theta_lso = theta_lso, t0_shift = t0_shift, LAL_GAMMA = LAL_GAMMA, GMsun_over_c3 = GMsun_over_c3)
    phiRef = coeffs.T(phiRef)
    if t isa AbstractArray
        out = similar(t, coeffs.T)
        @inbounds for i in eachindex(t)
            out[i] = 2 * _taylort3_phi_orb_scalar(coeffs, t[i]) + phiRef
        end
        return out
    else
        return 2 * _taylort3_phi_orb_scalar(coeffs, t) + phiRef
    end
end

"""
    Ampl(model::TaylorT3, t, mc, eta, dL; kwargs...)

TaylorT3 restricted (Newtonian-order) time-domain amplitude, in the same "(2,2)-mode-only,
face-on" convention `IMRPhenomD.Ampl`/`IMRPhenomXAS.Ampl` document: `h+ = -Ampl(t)*cos(Phi(t))`,
`hx = -Ampl(t)*sin(Phi(t))` at zero inclination. `t` follows the same convention as `Phi`
(seconds relative to coalescence, `t <= 0`).

# Derivation of the prefactor
From `XLALSimInspiralPNPolarizationWaveforms`: `ampfac = 2*M*(G/c^3)*eta/dist` (with `dist =
r/c` in seconds) and, at `ampO = 0`, `hp0 = -(1+cos^2(iota))*cos(2*phi_orb)`. At `iota = 0`
(face-on), `hp0 = hc0`'s prefactor `= -2`, giving this package's single-amplitude convention
`Ampl = ampfac*v^2*2`. Rewriting `dist` in Gpc via this package's existing `GMsun_over_c2_Gpc`
constant (`= G*Msun/c^2` expressed in Gpc, exactly as `IMRPhenomD.Ampl` already uses it) gives
`Ampl(t) = 4*eta*M*GMsun_over_c2_Gpc/dL * v(t)^2`, with `M = mc/eta^(3/5)` in solar masses,
`v(t)^2 = (pi*M_sec*f_GW(t))^(2/3)`, and `f_GW` from [`Phi`](@ref)'s underlying frequency
evolution. Algebraically this is the standard chirp-mass-scaled restricted PN amplitude
(`eta*M*v^2 = eta*M^(5/3)*(pi*f)^(2/3) = Mc^(5/3)*(pi*f)^(2/3)`, since `Mc = M*eta^(3/5)`), just
expressed directly in the `(M, eta)` parametrization this package's `Phi`/`Ampl` already use
elsewhere, rather than re-introducing chirp mass as a separate symbol.

# Units
- `mc`: chirp mass in solar masses; `dL`: luminosity distance **in Gpc** (same convention as
  `IMRPhenomD.Ampl` -- see that docstring's units note).

# Keywords
Same as [`Phi`](@ref) (`f_min`, `deltaT`, `theta_lso`, `t0_shift`, `LAL_GAMMA`,
`GMsun_over_c3`), plus `GMsun_over_c2_Gpc` (overridable, mirroring `IMRPhenomD.Ampl`).

No parameter-domain assertions run here -- see `validate_taylort3_params` for an opt-in
host-side check.
"""
function Ampl(model::TaylorT3,
    t,
    mc,
    eta,
    dL;
    f_min = 20.0,
    deltaT = 1.0 / 4096.0,
    theta_lso = nothing,
    t0_shift = nothing,
    LAL_GAMMA = 0.577215664901532860606512090082402431,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    coeffs = _taylort3_coeffs(model, mc, eta; f_min = f_min, deltaT = deltaT,
        theta_lso = theta_lso, t0_shift = t0_shift, LAL_GAMMA = LAL_GAMMA, GMsun_over_c3 = GMsun_over_c3)
    T = coeffs.T
    GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc)
    dL = T(dL)
    if t isa AbstractArray
        out = similar(t, T)
        @inbounds for i in eachindex(t)
            out[i] = _taylort3_ampl_scalar(coeffs, t[i], GMsun_over_c2_Gpc, dL)
        end
        return out
    else
        return _taylort3_ampl_scalar(coeffs, t, GMsun_over_c2_Gpc, dL)
    end
end
