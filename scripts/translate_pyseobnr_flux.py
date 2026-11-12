"""
Mechanical translation of pyseobnr's `compute_rho_coeffs` (`eob/waveform/waveform.pyx`) into the part of the
SEOBNRv5 radiation-reaction flux that `_seobv5_rr_force` does not hand-port: every mode with l <= ell_max = 8
other than the seven SEOBNRv5HM waveform modes (pyseobnr's `compute_flux` sums all l = 2..8, m = 1..l).

    python scripts/translate_pyseobnr_flux.py PYSEOBNR_SRC_DIR > src/GravitationalWaveforms/SEOBNRv5_flux_extra.jl

Two functions: `_seobv5_flux_extra_coeffs` (per binary) runs the coefficient code, emitted as scalar locals
(`rho_coeffs[l, m][j]` -> `rc_l_m_j`, `rho_coeffs_log` -> `rcl_`, `f_coeffs` -> `fc_`, `f_coeffs_vh` -> `fvh_`), fully
parenthesized, with `extra_PN_terms = True` (the SEOBNRv5HM default), and returns the binary-dependent ones plus the
Newtonian prefactors; `_seobv5_flux_extra` (per flux call) sums the modes. Which coefficients are zero, constant or
binary-dependent is found by running pyseobnr's code at sample binaries. None of the extra modes has GSF
(`extra_coeffs`) or `f_coeffs_vh` terms.
"""

import ast
import collections
import math
import os
import random
import re
import sys

SEVEN = {(2, 2), (3, 3), (2, 1), (4, 4), (3, 2), (5, 5), (4, 3)}
ARR = {"rho_coeffs": "rc", "rho_coeffs_log": "rcl", "f_coeffs": "fc", "f_coeffs_vh": "fvh"}
ELL_MAX = 8


def expr(n):
    if isinstance(n, ast.Constant):
        if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
            raise ValueError(f"constant {n.value!r}")
        return repr(float(n.value))
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return f"(-{expr(n.operand)})"
    if isinstance(n, ast.BinOp):
        ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}
        op = ops[type(n.op)]
        if op == "^" and isinstance(n.right, ast.Constant) and isinstance(n.right.value, int):
            return f"({expr(n.left)}^{n.right.value})"
        return f"({expr(n.left)} {op} {expr(n.right)})"
    if isinstance(n, ast.Subscript):
        return target(n)
    raise ValueError(ast.dump(n))


def target(t):
    # arr[l, m][j]
    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Subscript) and isinstance(t.value.value, ast.Name):
        arr = ARR[t.value.value.id]
        l, m = (e.value for e in t.value.slice.elts)
        j = t.slice.value
        return f"{arr}_{l}_{m}_{j}"
    raise ValueError(ast.dump(t))


def _seven(t):
    # coefficient targets of the seven hand-ported modes are not emitted (unused here; they only bloat the
    # function Enzyme differentiates)
    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Subscript):
        l, m = (e.value for e in t.value.slice.elts)
        return (l, m) in SEVEN
    return False


def stmts(body, out, ind, assigned):
    for st in body:
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue
        if isinstance(st, ast.Assign):
            tg = st.targets[0]
            if _seven(tg):
                continue
            name = tg.id if isinstance(tg, ast.Name) else target(tg)
            if not isinstance(tg, ast.Name):
                assigned.add(name)
            out.append(f"{ind}{name} = {expr(st.value)}")
        elif isinstance(st, ast.AugAssign):
            if _seven(st.target):
                continue
            name = target(st.target)
            assigned.add(name)
            op = {ast.Add: "+", ast.Sub: "-"}[type(st.op)]
            out.append(f"{ind}{name} {op}= {expr(st.value)}")
        elif isinstance(st, ast.If):
            test = st.test.id
            if test == "extra_PN_terms":
                stmts(st.body, out, ind, assigned)
                assert not st.orelse
            elif test == "dm2":
                body_out, else_out = [], []
                stmts(st.body, body_out, ind + "    ", assigned)
                if st.orelse:
                    stmts(st.orelse, else_out, ind + "    ", assigned)
                if body_out or else_out:
                    out.append(f"{ind}if dm2 != 0.0")
                    out.extend(body_out)
                    if else_out:
                        out.append(f"{ind}else")
                        out.extend(else_out)
                    out.append(f"{ind}end")
            else:
                raise ValueError(test)
        else:
            raise ValueError(ast.dump(st))


def main():
    src = open(os.path.join(sys.argv[1], "eob/waveform/waveform.pyx")).read()
    i = src.index("cpdef void compute_rho_coeffs(")
    i = src.index("):", i) + 2
    j = src.index("@cython.cpow(True)", i)
    body = src[i:j]
    lines = []
    for ln in body.split("\n"):
        if re.match(r"^\s*cdef (double|bint|int) [A-Za-z_0-9, ]*$", ln):   # declarations without a value
            continue
        lines.append(re.sub(r"^(\s*)cdef (double|bint|int) ", r"\1", ln))
    code = "\n".join(l[4:] if l.startswith("    ") else l for l in lines)
    tree = ast.parse(code)
    out, assigned = [], set()
    stmts(tree.body, out, "    ", assigned)


    modes = [(l, m) for l in range(2, ELL_MAX + 1) for m in range(1, l + 1) if (l, m) not in SEVEN]
    # |Y_l^{-m}(pi/2, 0)| (pyseobnr `ylms[m][l]`, indexed at l - eps)
    ylms_src = src[src.index("ylms[:] = ["):src.index("]\n\n", src.index("ylms[:] = ["))]
    ylms = ast.literal_eval(ylms_src.split("=", 1)[1].strip() + "]")

    # Classify the coefficients by running pyseobnr's own code at sample binaries: identically zero (dropped),
    # constant (a literal in the per-call function) or binary-dependent (stored by the per-binary function).
    kind = classify(code, assigned)
    stored = sorted(n for n in assigned if kind[n][0] == "var")

    def coef(name):
        k = kind.get(name, ("zero",))
        if k[0] == "zero":
            return None
        if k[0] == "const":
            return repr(k[1])
        return f"c[{stored.index(name) + 1}]"

    print("# GENERATED by scripts/translate_pyseobnr_flux.py from pyseobnr 0.3.7 `waveform.pyx` (`compute_rho_coeffs`,")
    print("# `compute_flux`). Do not edit by hand: regenerate.")
    print(f'''
"""
    _seobv5_flux_extra_coeffs(nu, delta, chiS, chiA, atest) -> NTuple{{{len(stored) + len(modes)},T}}

Per-binary part of [`_seobv5_flux_extra`](@ref): the {len(stored)} binary-dependent `rho_lm` / `f_lm` coefficients of
the modes with `l <= 8` besides the seven hand-ported in [`_seobv5_rr_force`](@ref) (pyseobnr `compute_rho_coeffs`;
coefficients that are constant or zero for every binary are literals in `_seobv5_flux_extra`), then one prefactor
per mode, `m^2 |n_lm Y_lm|^2 / (l!)^2` (Newtonian prefactor magnitude, general formula like the seven, not the
exactly-equal-mass odd-m constants). Test-spin terms are zero for these modes (pyseobnr sets `a = 0` after the
(2,2) block).
"""
@inline function _seobv5_flux_extra_coeffs(nu::T, delta::T, chiS::T, chiA::T, atest::T) where {{T}}
@Tlit T begin
    dm = delta
    a = atest
    dm2 = dm * dm''')
    for name in sorted(assigned):
        print(f"    {name} = 0.0")
    print("\n".join(out))
    print("    X1 = 0.5 * (1.0 + delta)")
    print("    X2 = 0.5 * (1.0 - delta)")
    print("    X1_1 = X1; X2_1 = X2")
    for k in range(2, ELL_MAX + 1):
        print(f"    X1_{k} = X1_{k - 1} * X1")
        print(f"    X2_{k} = X2_{k - 1} * X2")
    pref = []
    for (l, m) in modes:
        eps = (l + m) % 2
        k = l + eps - 1
        c = f"(X2_{k} + X1_{k})" if m % 2 == 0 else f"(X2_{k} - X1_{k})"
        K = nabs(l, m) * abs(ylms[m][l - eps])
        pref.append(f"{m * m * K * K / math.factorial(l) ** 2!r} * (nu * {c})^2")
    print("    return (" + ", ".join(stored + pref) + ",)")
    print("end # @Tlit T")
    print("end")

    print(f'''
"""
    _seobv5_flux_extra(v, lnv, w, Heff2, L2, tp2, hs, c) -> sum m^2 |h_lm|^2

The modes with `l <= 8` that pyseobnr's flux (`compute_flux`, `ell_max = 8`) sums besides the seven hand-ported
in [`_seobv5_rr_force`](@ref): `(3,1), (4,2), (4,1), (5,4), ..., (8,1)`, as `sum m^2 |h_lm|^2` in the units of
that function's `flux_acc`. Per call: `v = Omega^(1/3)`, `lnv = log(v)`, `w = v_phi^2`, `Heff2 = Heff^2`,
`L2 = (v pphi)^2` (the source terms squared), and per `m = 1..8` the squared tail prefactor
`tp2[m] = 4 pi k / (1 - exp(-4 pi k))` and `hs[m] = 4 k^2` (`k = m nu H Omega`); `c` from
[`_seobv5_flux_extra_coeffs`](@ref). `|T_lm|^2 = tp2 prod_(j=1..l) (hs + j^2) / (l!)^2` (`compute_tail`), so no square
roots. pyseobnr's `rholm = f` for odd `m` within 1e-14 of `nu = 1/4` is not reproduced: those modes' prefactors
vanish as `delta^2` there.
"""
@inline function _seobv5_flux_extra(v::T, lnv::T, w::T, Heff2::T, L2::T, tp2::NTuple{{8,T}}, hs::NTuple{{8,T}}, c) where {{T}}
@Tlit T begin''')
    print("    v_1 = v")
    for k in range(2, 11):
        print(f"    v_{k} = v_{k - 1} * v")
    print("    w_1 = w")
    for k in range(2, ELL_MAX + 2):
        print(f"    w_{k} = w_{k - 1} * w")
    ms = sorted({m for (_, m) in modes})
    for m in ms:
        lmax = max(l for (l, mm) in modes if mm == m)
        print(f"    P_{m}_1 = hs[{m}] + 1.0")
        for jj in range(2, lmax + 1):
            print(f"    P_{m}_{jj} = P_{m}_{jj - 1} * (hs[{m}] + {float(jj * jj)!r})")
    print("    acc = 0.0")
    for i, (l, m) in enumerate(modes):
        eps = (l + m) % 2
        terms = []
        uselog = False
        for jj in range(1, 11):
            c = coef(f"rc_{l}_{m}_{jj}")
            d = coef(f"rcl_{l}_{m}_{jj}")
            if d is not None:
                c = f"({c or '0.0'} + {d} * el)"
                uselog = True
            if c is not None:
                terms.append(f"{c} * v_{jj}")
        fterms = [f"{coef(f'fc_{l}_{m}_{jj}')} * v_{jj}" for jj in range(1, 11) if coef(f"fc_{l}_{m}_{jj}") is not None]
        print(f"    # ({l},{m})")
        print("    let")
        if uselog:
            print(f"        el = lnv + {0.5772156649015329 + math.log(2.0 * m)!r}")     # gamma + log(2 m v)
        print(f"        rho = 1.0" + "".join(f" + {t}" for t in terms))
        if m % 2:
            print(f"        rholm = rho^{l}" + "".join(f" + {t}" for t in fterms))
        else:
            assert not fterms
            print(f"        rholm = rho^{l}")
        S2 = "Heff2" if eps == 0 else "L2"
        print(f"        acc += c[{len(stored) + i + 1}] * tp2[{m}] * P_{m}_{l} * {S2} * w_{l + eps} * rholm * rholm")
        print("    end")
    print("    return acc")
    print("end # @Tlit T")
    print("end")


def dfact_odd(n):
    r = 1
    for k in range(n, 0, -2):
        r *= k
    return r


def nabs(l, m):
    # |n| of calculate_multipole_prefix (waveform.pyx)
    eps = (l + m) % 2
    if eps == 0:
        return m ** l * 8.0 * math.pi / dfact_odd(2 * l + 1) * math.sqrt((l + 1) * (l + 2) / (l * (l - 1)))
    return (m ** l * 16.0 * math.pi / dfact_odd(2 * l + 1) *
            math.sqrt((2 * l + 1) * (l + 2) * (l * l - m * m) / ((2 * l - 1) * (l + 1) * l * (l - 1))))


class _Coeffs:
    # stands in for a `double[:, :, :]` coefficient array: `arr[l, m][j]`
    def __init__(self):
        self.rows = {}

    def __getitem__(self, lm):
        return self.rows.setdefault(tuple(lm), collections.defaultdict(float))


def classify(code, names):
    """("zero",), ("const", value) or ("var",) for each coefficient name, from pyseobnr's code run at unequal- and
    equal-mass sample binaries."""
    rng = random.Random(12345)
    vals = collections.defaultdict(list)
    for s in range(8):
        nu = 0.25 if s >= 6 else rng.uniform(0.05, 0.245)
        dm = math.sqrt(max(1.0 - 4.0 * nu, 0.0))
        chiS, chiA = rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9)
        ns = dict(nu=nu, dm=dm, chiS=chiS, chiA=chiA, a=(1.0 - 2.0 * nu) * chiS + dm * chiA, extra_PN_terms=True,
                  pi=math.pi, **{arr: _Coeffs() for arr in ARR})
        exec(code, ns)
        for arr, pre in ARR.items():
            for (l, m), row in ns[arr].rows.items():
                for j, v in row.items():
                    vals[f"{pre}_{l}_{m}_{j}"].append(v)
    kind = {}
    for n in names:
        vs = vals.get(n, [0.0])
        vs = vs + [0.0] * (8 - len(vs))            # never assigned at some samples: zero there
        assert all(complex(v).imag == 0.0 for v in vs), n
        vs = [float(complex(v).real) for v in vs]
        if all(v == 0.0 for v in vs):
            kind[n] = ("zero",)
        elif all(v == vs[0] for v in vs):
            kind[n] = ("const", vs[0])
        else:
            kind[n] = ("var",)
    return kind


if __name__ == "__main__":
    main()
