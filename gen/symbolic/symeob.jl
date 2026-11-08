# Julia function body -> Symbolics DAG -> memoised derivatives -> straight-line CSE'd Julia source.

using Symbolics, SymbolicUtils
using SymbolicUtils: iscall, operation, arguments

function find_function(ex, name)
    ex isa Expr || return nothing
    if ex.head === :function
        sig = ex.args[1]
        sig.head === :where && (sig = sig.args[1])
        sig.args[1] === name && return ex
    end
    for a in ex.args
        f = find_function(a, name)
        f === nothing || return f
    end
    return nothing
end

argname(a) = a isa Symbol ? a : a.args[1]           # `r::T` -> :r
function signature_args(fn)
    sig = fn.args[1]
    sig.head === :where && (sig = sig.args[1])
    return Tuple(argname.(sig.args[2:end]))
end

cval(u) = SymbolicUtils.isconst(u) ? SymbolicUtils.unwrap_const(u) : u

# evaluate one expression of the source on symbolic values; domain guards (max(x, floatmin(T)), the _n* NaN wrappers)
# are dropped: they are no-ops on the physical domain and re-added by the emitter's choice of functions
# Vectors (the PN spin equations: `SVector`, `cross`, `dot`, `norm`, `v[i]`) are plain `Vector`s of symbolic scalars.
function symeval(e, env)
    e isa Number && return e
    e isa Symbol && return env[e]
    e.head === :ref && return symeval(e.args[1], env)[e.args[2]]
    e.head === :tuple && return Tuple(symeval(x, env) for x in e.args)
    e.head === :call || error("unexpected $(e.head): $e")
    f = e.args[1]
    f === :zero && return 0
    f === :floatmin && return 0
    a = [symeval(x, env) for x in e.args[2:end]]
    # broadcast: Symbolics makes `Vector{Num} / Num` a symbolic-array term instead of a vector
    f === :+ && return reduce((p, q) -> p .+ q, a)
    f === :- && return length(a) == 1 ? .-a[1] : a[1] .- a[2]
    f === :* && return reduce((p, q) -> p .* q, a)
    f === :/ && return a[1] ./ a[2]
    f === :^ && return a[1]^a[2]
    f === :SVector && return collect(Any, a)
    f === :cross && return [a[1][2] * a[2][3] - a[1][3] * a[2][2], a[1][3] * a[2][1] - a[1][1] * a[2][3],
                            a[1][1] * a[2][2] - a[1][2] * a[2][1]]
    f === :dot && return sum(a[1][i] * a[2][i] for i in eachindex(a[1]))
    f === :norm && return sqrt(sum(x^2 for x in a[1]))
    f === :max && return a[1]
    f in (:log, :_seobv5phm_nlog) && return log(a[1])
    f in (:sqrt, :_seobv5phm_nsqrt) && return sqrt(a[1])
    f === :_seobv5phm_npow && return a[2] == 0.5 ? sqrt(a[1]) : a[1]^a[2]
    f === :sin && return sin(a[1])
    f === :cos && return cos(a[1])
    f === :cbrt && return cbrt(a[1])
    f === :evalpoly && return foldr((c, acc) -> c + a[1] * acc, a[2][1:end-1]; init = a[2][end])
    error("unknown function $f")
end

# run the straight-line body (assignments, possibly inside `@Tlit T begin ... end`) and return the environment
# `fixed`: arguments replaced by constants (e.g. `Dict(:prst => 0)` for the circular-orbit terms), so the expressions
# simplify before differentiation instead of being evaluated at that value
# `globals`: values of the module constants the body reads (e.g. `_PI`). `override`: values that replace those of the
# named assignments (a specialization the simplifier cannot see, e.g. in-plane spin parts that are zero)
function symbolic_body(fn; fixed = Dict{Symbol,Any}(), globals = Dict{Symbol,Any}(), override = Dict{Symbol,Any}())
    args = signature_args(fn)
    vars = Dict{Symbol,Any}(s => (haskey(fixed, s) ? fixed[s] : only(@variables $s)) for s in args)
    env = merge(Dict{Symbol,Any}(globals), vars)
    function run!(blk)
        for st in blk.args
            st isa LineNumberNode && continue
            if st.head === :macrocall
                run!(st.args[end])
            elseif st.head === :block
                run!(st)
            elseif st.head === :(=) && st.args[1] isa Expr && st.args[1].head === :tuple
                for (k, v) in zip(st.args[1].args, symeval(st.args[2], env))
                    env[k] = v
                end
            elseif st.head === :(=)
                k = st.args[1]
                env[k] = haskey(override, k) ? override[k] : symeval(st.args[2], env)
            elseif st.head === :return
                env[:__return] = symeval(st.args[1], env)
                return
            else
                error("unexpected statement $(st.head)")
            end
        end
    end
    run!(fn.args[2])
    return args, vars, env
end

# opaque stand-ins for subterms inside derivatives (symbol => subterm); the emitter substitutes the subterm's local
const OPAQUE = Dict{Any,Any}()
const OPAQUE_OF = IdDict{Any,Any}()
function opaque(u)
    haskey(OPAQUE_OF, u) && return OPAQUE_OF[u]
    s = only(@variables $(Symbol("__q", length(OPAQUE) + 1)))
    OPAQUE[Symbolics.unwrap(s)] = u
    OPAQUE_OF[u] = s
    return s
end

# chain rule for argument symbols that stand for functions of another: CHAIN[(k, r)] = dk/dr (e.g. the tidal
# enhancement k_eff(r) passed to the Hamiltonian as k with its derivatives k_r, k_rr as further arguments)
const CHAIN = Dict{Tuple{Any,Any},Any}()
chain!(k, x, dk) = (CHAIN[(Symbolics.unwrap(k), Symbolics.unwrap(x))] = dk)

# memoised forward derivative on the expression DAG (each distinct subterm differentiated once)
function dag_derivative(ex, x, memo = IdDict{Any,Any}())
    haskey(memo, ex) && return memo[ex]
    d = _dd(ex, x, memo)
    memo[ex] = d
    return d
end
_iszero(d) = d isa Number ? iszero(d) : SymbolicUtils._iszero(Symbolics.unwrap(d))
function _dd(ex, x, memo)
    u = Symbolics.unwrap(ex)
    haskey(OPAQUE, u) && return dag_derivative(OPAQUE[u], x, memo)
    iscall(u) || return isequal(u, Symbolics.unwrap(x)) ? 1 : get(CHAIN, (u, Symbolics.unwrap(x)), 0)
    op, as = operation(u), arguments(u)
    D(a) = dag_derivative(a, x, memo)
    if op === (+)
        return sum(D(a) for a in as)
    elseif op === (*)
        s = 0
        for i in eachindex(as)
            di = D(as[i])
            _iszero(di) && continue
            s += prod((j == i ? di : as[j] for j in eachindex(as)))
        end
        return s
    elseif op === (/)
        # (n' - q m') / m with q = n/m an opaque symbol (emitted as the local holding n/m): Symbolics would otherwise
        # merge q m' back into n m' / m, forming products of large numerators before the division (Float32
        # overflow in Apm / Dbpm, whose denominators are ~1e15) and losing the source's grouping
        n, m = as
        return (D(n) - opaque(u) * D(m)) / m
    elseif op === (^)
        b, p = as[1], cval(as[2])
        p isa Number || error("non-constant exponent")
        return p * b^(p - 1) * D(b)
    elseif op === log
        return D(as[1]) / as[1]
    elseif op === sqrt
        return D(as[1]) / (2 * sqrt(as[1]))
    elseif op === sin
        return cos(as[1]) * D(as[1])
    elseif op === cos
        return -sin(as[1]) * D(as[1])
    elseif op === cbrt
        return D(as[1]) / (3 * cbrt(as[1])^2)
    end
    error("no rule for $op")
end

# Structural key of a subterm, independent of the process: the operands of + and * come out of Symbolics in an order
# that varies between runs, so the emitter sorts them by this key (deterministic output for `generate.jl check`).
const SKEY = IdDict{Any,UInt}()
function skey(ex)
    u = cval(Symbolics.unwrap(ex))
    u isa Number && return hash(Float64(u))
    haskey(OPAQUE, u) && return skey(OPAQUE[u])
    iscall(u) || return hash(string(u))
    return get!(SKEY, u) do
        op = operation(u)
        ks = [skey(a) for a in arguments(u)]
        (op === (+) || op === (*)) && sort!(ks)
        hash((string(nameof(op)), ks))
    end
end

# straight-line source: every distinct non-leaf subterm becomes one local; returns (source, number of locals).
# `typed = true`: `(args::T...) where {T}` with the body in `@Tlit T begin ... end`, as the hand-written aligned code.
function emit(outs, name, args; logf = "_seobv5phm_nlog", sqrtf = "_seobv5phm_nsqrt", powf = "_seobv5phm_npow",
              typed = false, doc = "")
    names = Dict{Any,Symbol}()
    lines = String[]
    function go(ex)
        u = cval(Symbolics.unwrap(ex))
        u isa Number && return string(Float64(u))
        haskey(OPAQUE, u) && return go(OPAQUE[u])
        iscall(u) || return string(u)
        haskey(names, u) && return string(names[u])
        op, as = operation(u), arguments(u)
        (op === (+) || op === (*)) && (as = sort(collect(as); by = skey))
        s = [go(a) for a in as]
        rhs = if op === (^)
            p = cval(as[2])
            p isa Integer || (p isa Real && isinteger(p)) ? "$(s[1])^$(Int(p))" :
                p == 0.5 ? "$sqrtf($(s[1]))" :
                p == -0.5 ? "1.0 / $sqrtf($(s[1]))" : "$powf($(s[1]), $(s[2]))"
        elseif op === (+) || op === (*)
            join(s, " $op ")
        elseif op === (/)
            "$(s[1]) / $(s[2])"
        elseif op === log
            "$logf($(s[1]))"
        elseif op === sqrt
            "$sqrtf($(s[1]))"
        elseif op in (sin, cos, cbrt)
            "$(nameof(op))($(s[1]))"
        else
            error("no emitter for $op")
        end
        n = Symbol("t", length(names) + 1)
        names[u] = n
        push!(lines, "    $n = $rhs")
        return string(n)
    end
    res = [go(o) for o in outs]
    ret = length(res) == 1 ? res[1] : "($(join(res, ", ")),)"
    body = join(lines, "\n")
    src = if typed
        "$doc@inline function $name($(join(string.(args) .* "::T", ", "))) where {T}\n@Tlit T begin\n$body\n    return $ret\nend # @Tlit T\nend\n"
    else
        "$doc@inline function $name($(join(args, ", ")))\n$body\n    return $ret\nend\n"
    end
    return src, length(lines)
end
