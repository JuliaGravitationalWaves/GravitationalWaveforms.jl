"""
    IMRPhenomXHM QNM fits (this file)

Quasi-normal-mode ringdown-frequency (`fring`) and damping-frequency (`fdamp`) fits for the
higher-harmonic `(l,m) = (2,1), (3,3), (3,2), (4,4)` modes that `IMRPhenomXHM` adds on top of
`IMRPhenomXAS`'s existing dominant `(2,2)` mode (`_xas_fRD_fdamp` in `IMRPhenomXAS.jl`). These are
purely numerical, closed-form-in-final-spin fit
functions, independent of the inspiral/intermediate/ringdown
amplitude-phase ansatze that consume these values, and of the mode-mixing/full-waveform assembly.

Ported directly from LALSuite's `lalsimulation/lib/LALSimIMRPhenomXHM_qnm.c`, specifically the eight
`evaluate_QNMfit_f{ring,damp}{21,33,32,44}` functions (lines 30-167). These fits are rational
functions of the final remnant's dimensionless spin alone (Kerr QNM frequencies computed from
https://pages.jh.edu/~eberti2/ringdown/, arXiv:gr-qc/0512160 and arXiv:1408.1860, then fit as
described in arXiv:2001.10914). Reused verbatim from the paired `evaluate_QNMfit_re/im_l*m2lp*`
mixing-coefficient functions in the same C file, which are in `IMRPhenomXHM.jl`
(`LALSimIMRPhenomXHM_qnm.h` groups them separately as "mixing coefficients (only 32
mode)").

# Units / convention (important, does not match a naive reading of the C source in isolation)
Each raw fit function here returns a value in the SAME convention as LALSuite's raw
`evaluate_QNMfit_*` return value -- **not yet divided by the remnant's dimensionless final mass**.
Confirmed by reading the only call site, `LALSimIMRPhenomXHM_internals.c` line 393-394:
```
wf->fRING = (qnms->fring_lm[wf->modeInt](wf22->afinal))/wf22->Mfinal;
wf->fDAMP = (qnms->fdamp_lm[wf->modeInt](wf22->afinal))/wf22->Mfinal;
```
i.e. LALSuite applies the `/Mfinal` (`Mfinal = 1 - Erad`, the SAME `(1-Erad)` normalization
`IMRPhenomXAS`'s own `_xas_fRD_fdamp` divides by, using the SAME `(2,2)`-mode `afinal`/`Erad` --
`IMRPhenomXHM` does NOT refit final spin/mass per higher mode) uniformly at the call site, not
inside the fit function itself. This port mirrors that split exactly: `_xhm_qnmfit_f{ring,damp}NN`
below are byte-for-byte transcriptions of the raw C fit bodies (final-spin -> raw value, no `Mfinal`
division), and `_xhm_fring_fdamp` is the convenience wrapper applying that division, taking the same
`(a, Erad)` pair `_xas_a_Erad` computes. This matches this package's existing convention of keeping
each ported piece narrowly faithful to its LALSuite source rather than pre-combining steps that
LALSuite itself keeps separate (see e.g. `IMRPhenomXPrecession.jl`'s `_xp_twist_up22`/
`_xp_apply_zeta_rotation` split).

# A numerical sanity check worth flagging (not a bug, but a useful cross-reference during porting)
The `fring21` fit's leading (zero-spin) coefficient, `0.059471695665734674`, is identical (to all
17 given digits) to `IMRPhenomXAS`'s own `_xas_fRD_fdamp` numerator's leading coefficient,
`0.05947169566573468`. This is NOT a coincidence or a copy-paste error in either source: the `(2,1)`
mode's ringdown frequency and the `(2,2)` mode's both converge to the same Schwarzschild
(zero-spin) `l=2` QNM frequency in the test-particle limit that these Kerr-QNM fits are built
from -- `(l,m)` only distinguishes the azimuthal number `m`, and the leading zero-spin term of a
QNM frequency fit depends on `l` (and the overtone), not `m`. Confirmed as an expected, correct degeneracy (Schwarzschild -- zero-spin -- QNM frequencies depend
only on `l`, not `m`, since a non-rotating black hole has no preferred axis) rather than
investigated further as an anomaly, and it recurs exactly as expected for the `l=3` pair: `fring33`
and `fring32`'s zero-spin numerator leading terms are both `0.09540436245212061`, and `fdamp33` and
`fdamp32`'s are both `0.014754148319335946` -- an internal LALSuite self-consistency check that
lines up with this same-`l`-degeneracy expectation, not a copy-paste bug, and independently
confirms this port transcribed all four `l=3` constants correctly (a copy-paste slip between the
`32`/`33` fits would most likely have broken this exact coincidence).
"""

@inline function _xhm_qnmfit_fring21(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a
        x3 = x2 * a
        x4 = x2 * x2
        x5 = x3 * x2
        return (0.059471695665734674 - 0.07585416297991414 * a + 0.021967909664591865 * x2 -
                0.0018964744613388146 * x3 + 0.001164879406179587 * x4 - 0.0003387374454044957 * x5) /
               (1.0 - 1.4437415542456158 * a + 0.49246920313191234 * x2)
    end
end

@inline function _xhm_qnmfit_fdamp21(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a
        x3 = x2 * a
        x4 = x2 * x2
        x5 = x3 * x2
        return (2.0696914454467294 - 3.1358071947583093 * a + 0.14456081596393977 * x2 +
                1.2194717985037946 * x3 - 0.2947372598589144 * x4 + 0.002943057145913646 * x5) /
               (146.1779212636481 - 219.81790388304876 * a + 17.7141194900164 * x2 +
                75.90115083917898 * x3 - 18.975287709794745 * x4)
    end
end

@inline function _xhm_qnmfit_fring33(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a
        x3 = x2 * a
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.09540436245212061 - 0.22799517865876945 * a + 0.13402916709362475 * x2 +
                0.03343753057911253 * x3 - 0.030848060170259615 * x4 - 0.006756504382964637 * x5 +
                0.0027301732074159835 * x6) /
               (1.0 - 2.7265947806178334 * a + 2.144070539525238 * x2 -
                0.4706873667569393 * x4 + 0.05321818246993958 * x6)
    end
end

@inline function _xhm_qnmfit_fdamp33(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a
        x3 = x2 * a
        x4 = x2 * x2
        x5 = x3 * x2
        return (0.014754148319335946 - 0.03124423610028678 * a + 0.017192623913708124 * x2 +
                0.001034954865629645 * x3 - 0.0015925124814622795 * x4 - 0.0001414350555699256 * x5) /
               (1.0 - 2.0963684630756894 * a + 1.196809702382645 * x2 - 0.09874113387889819 * x4)
    end
end

@inline function _xhm_qnmfit_fring32(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a
        x3 = x2 * a
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.09540436245212061 - 0.13628306966373951 * a + 0.030099881830507727 * x2 -
                0.000673589757007597 * x3 + 0.0118277880067919 * x4 + 0.0020533816327907334 * x5 -
                0.0015206141948469621 * x6) /
               (1.0 - 1.6531854335715193 * a + 0.5634705514193629 * x2 +
                0.12256204148002939 * x4 - 0.027297817699401976 * x6)
    end
end

@inline function _xhm_qnmfit_fdamp32(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a
        x3 = x2 * a
        x4 = x2 * x2
        return (0.014754148319335946 - 0.03445752346074498 * a + 0.02168855041940869 * x2 +
                0.0014945908223317514 * x3 - 0.0034761714223258693 * x4) /
               (1.0 - 2.320722660848874 * a + 1.5096146036915865 * x2 - 0.18791187563554512 * x4)
    end
end

@inline function _xhm_qnmfit_fring44(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a
        x3 = x2 * a
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.1287821193485683 - 0.21224284094693793 * a + 0.0710926778043916 * x2 +
                0.015487322972031054 * x3 - 0.002795401084713644 * x4 + 0.000045483523029172406 * x5 +
                0.00034775290179000503 * x6) /
               (1.0 - 1.9931645124693607 * a + 1.0593147376898773 * x2 - 0.06378640753152783 * x4)
    end
end

@inline function _xhm_qnmfit_fdamp44(a)
    T = typeof(a)
    @Tlit T begin
        x2 = a * a
        x3 = x2 * a
        x4 = x2 * x2
        x5 = x3 * x2
        x6 = x3 * x3
        return (0.014986847152355699 - 0.01722587715950451 * a - 0.0016734788189065538 * x2 +
                0.0002837322846047305 * x3 + 0.002510528746148588 * x4 + 0.00031983835498725354 * x5 +
                0.000812185411753066 * x6) /
               (1.0 - 1.1350205970682399 * a - 0.0500827971270845 * x2 +
                0.13983808071522857 * x4 + 0.051876225199833995 * x6)
    end
end

"""
    _xhm_fring_fdamp(l, m, a, Erad)

Convenience wrapper: `(fring, fdamp)` for the `(l,m)` harmonic in units of the *total* mass (`Mf`,
the same convention `IMRPhenomXAS._xas_fRD_fdamp` returns for the `(2,2)` mode), given the `(2,2)`
mode's own final dimensionless spin `a` and radiated-energy fraction `Erad` (from `_xas_a_Erad`) --
`IMRPhenomXHM` reuses that single final-spin/mass fit for every higher mode rather than refitting
per `(l,m)`, see this file's module docstring. `l == 2 && m == 2` is intentionally NOT handled here
(use `_xas_fRD_fdamp` directly for the dominant mode, which has its own richer fit, not this file's
higher-mode-only rational functions).

`(l,m)` dispatch is a plain `if`/`elseif` chain over `Int` arguments (not a generated function or
`Val`-based dispatch) -- `l`/`m` are always small compile-time-known literals at call sites in this
package, and branching on `Int` values does not introduce any type instability as long as (as here)
every branch returns the same `promote_type(typeof(a), typeof(Erad))` pair.
"""
@inline function _xhm_fring_fdamp(l::Integer, m::Integer, a, Erad)
    T = promote_type(typeof(a), typeof(Erad))
    Mfinal = one(T) - Erad
    if l == 2 && m == 1
        return _xhm_qnmfit_fring21(a) / Mfinal, _xhm_qnmfit_fdamp21(a) / Mfinal
    elseif l == 3 && m == 3
        return _xhm_qnmfit_fring33(a) / Mfinal, _xhm_qnmfit_fdamp33(a) / Mfinal
    elseif l == 3 && m == 2
        return _xhm_qnmfit_fring32(a) / Mfinal, _xhm_qnmfit_fdamp32(a) / Mfinal
    elseif l == 4 && m == 4
        return _xhm_qnmfit_fring44(a) / Mfinal, _xhm_qnmfit_fdamp44(a) / Mfinal
    else
        throw(ArgumentError("_xhm_fring_fdamp: unsupported (l,m) = ($l,$m); only (2,1),(3,3),(3,2),(4,4) are implemented"))
    end
end
