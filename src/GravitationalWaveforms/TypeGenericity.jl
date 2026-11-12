"""
    @Tlit(T, expr)

Rewrite every bare `Float64` numeric literal appearing in `expr` (e.g. `0.006927402739328343`,
`2.0`, `-16.0`) into `T(<literal>)`, leaving everything else — variables, integer literals,
`pi`, function/macro calls — untouched.

Why this exists: ordinary Julia numeric promotion means `x::Float32 * 2.0 isa Float64` — a
single bare `Float64` literal anywhere in an expression silently promotes the *whole*
expression back to `Float64`, even when every variable involved is `Float32`. The PN/
phenomenological coefficient expressions in this package's waveform models (`gamma1`,
`sigma1`, `TF2coeffs`, `Acoeffs`, `delta0..4`, the IMRPhenomXAS collocation-fit and
merger-ringdown ansatz coefficients, ...) contain hundreds of such literals, so without this
rewrite a nominally-`Float32` call to `Phi`/`Ampl` would actually compute almost entirely in
`Float64` under the hood — defeating a `Float32` GPU kernel while silently truncating back to
`Float32` only in the final result, which hides the problem rather than erroring.

`pi` (an `Irrational`) IS rewritten too (to `T(pi)`), alongside the `Float64` literals: Julia
special-cases `Irrational` promotion so `pi * x::Float32 isa Float32`, but only when `pi`
directly participates in an operation *with* a same-call `T`-typed operand — `pi^2` alone
(e.g. inside `640.0/3.0*(pi^2)`) evaluates as plain `Float64` before it ever meets a `T`
value, then contaminates the product it's multiplied into. Wrapping `pi` itself sidesteps the
ambiguity entirely. Keyword-argument-derived values (e.g. a kwarg default like `fcutPar = 0.2`)
are NOT literals *inside this expression* — they're a runtime value flowing in through a
variable — so this macro cannot fix those; callers must cast those explicitly
(`fcutPar = T(fcutPar)`) before entering an `@Tlit` block. See the casts at the top of
`IMRPhenomD.Phi`/`IMRPhenomD.Ampl` for that pattern.

This is a syntactic, macro-expansion-time rewrite (a compile-time AST walk, not a runtime
one); the resulting `T(literal)` calls are ordinary conversions that Julia's compiler
constant-folds for a concrete `T`, so there is no hot-path runtime overhead beyond what
writing `T(...)` out by hand at each literal would have cost.

Usage: `@Tlit T begin ... end` around a block of arithmetic that should be computed entirely
in the working type `T`. `begin...end` introduces no new scope, so `return`/assignments
inside behave exactly as if the macro weren't there.

# `@.` interaction (important)

Coefficient-computation blocks in this package's waveform files may contain `@.`
(`@__dot__`) broadcast expressions (e.g. `f_safe = @. max(f, eps(T))`, `ifelse` trees over a
frequency array). `@Tlit` walks the *raw, unexpanded* AST — it runs before `@.` gets a chance
to expand — so when a `@Tlit` region contains a `@.` block, the rewrite descends into it too
and would naively turn a literal into a plain `T(literal)` call sitting inside that block.
`@.`'s own expansion then broadcasts *every* call it finds, including that one: `T(0.0)`
becomes `T.(0.0)`, i.e. a type constructor broadcast over a scalar, which builds a lazy
`Broadcasted{...,Type{T},...}` object. That object is harmless under plain execution but
Enzyme's reverse-mode compiler cannot differentiate through it (it hits an internal
ghost-element/shadow-allocation assertion) — confirmed by running `test_enzyme_gradient.jl`.

The fix: `_tlit_rewrite` tracks whether it is currently inside a `@.`/`@__dot__` macrocall
(flips an `indot` flag on when it recurses into one) and, while `indot` is true, wraps each
emitted `T(literal)` in a `\$(...)` interpolation escape instead of leaving it as a plain call.
`\$(...)` is `@.`'s own documented mechanism for exempting a subexpression from being
dot-broadcast — the escaped literal is evaluated once, as an ordinary scalar, before
broadcasting proceeds, so it never becomes a `Broadcasted` object and Enzyme differentiates
through it exactly as if it were an untouched `Float64` literal (just with the correct
working-type `T` instead).

Consequently, helper functions that are called *from within* another function's `@.` block
(rather than containing their own `@.` blocks) should do their own `@Tlit T begin...end`
wrapping around a plain (non-broadcast) scalar body, and be invoked via broadcasting at the
call site (`helper.(fgrid, args...)`) rather than having their formulas inlined textually into
the caller's `@.` expression. A broadcast *call* to an ordinary function is not a
`Broadcasted{...,Type{T},...}` shadow-allocation site — only a literal *type constructor*
sitting directly inside an expanding `@.` is — so this pattern sidesteps the bug entirely
while keeping each formula written and rewritten exactly once.
"""
macro Tlit(T, expr)
    return esc(_tlit_rewrite(T, expr, false))
end

const _DOT_MACRO_NAMES = (Symbol("@__dot__"), Symbol("@."))

_tlit_cast(T, x, indot) = indot ? Expr(:$, Expr(:call, T, x)) : Expr(:call, T, x)

_tlit_rewrite(T, x::Float64, indot) = _tlit_cast(T, x, indot)
_tlit_rewrite(T, x::Symbol, indot) = (x === :pi || x === :π) ? _tlit_cast(T, x, indot) : x
function _tlit_rewrite(T, x::Expr, indot)
    newindot = indot || (x.head === :macrocall && x.args[1] isa Symbol && x.args[1] in _DOT_MACRO_NAMES)
    return Expr(x.head, (_tlit_rewrite(T, a, newindot) for a in x.args)...)
end
_tlit_rewrite(T, x, indot) = x
