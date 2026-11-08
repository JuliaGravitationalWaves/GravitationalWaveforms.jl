# SEOBNRv5PHM attachment-state re-integration step counts (`_SEOBV5PHM_ATT_N0` / `_NF`):
# frequency-domain polarizations of this checkout vs a stored set from another (run twice: save=, then ref=)
# over random precessing binaries: ||dh||^2 / ||h||^2 with a flat-ish PSD weight, no maximization.
#   julia -t 8 --project=<env> scripts/check_phm_attach_steps.jl [n=30] [save=<file>] [ref=<file>]
using GravitationalWaveforms, Random, Printf
const GWW = GravitationalWaveforms
kv = Dict(split(a, "=")[1] => split(a, "=")[2] for a in ARGS if occursin("=", a))
n = parse(Int, get(kv, "n", "30"))
freqs = collect(20.0:0.125:896.0)
psd(f) = (x = f / 215; 1e-49 * (x^(-4.14) - 5 * x^(-2) + 111 * (1 - x^2 + 0.5 * x^4) / (1 + 0.5 * x^2)))
w = 1 ./ psd.(freqs)
rng = Xoshiro(15)
cfgs = map(1:n) do _
    m1 = 25 + 40rand(rng); q = 0.3 + 0.7rand(rng); m2 = m1 * q
    sp() = (a = 0.99rand(rng); u = 2rand(rng) - 1; p = 2pi * rand(rng); a .* [sqrt(1 - u^2) * cos(p), sqrt(1 - u^2) * sin(p), u])
    (m1, m2, sp(), sp(), 0.5, pi * rand(rng), 2pi * rand(rng))
end
wf(c) = GWW.hphc(GWW.SEOBNRv5PHM(), freqs, c...)
# run once per code version: `save=<file>` stores this checkout's waveforms, `ref=<file>` compares against a stored set
using Serialization
N0 = isdefined(GWW, :_SEOBV5PHM_ATT_N0) ? GWW._SEOBV5PHM_ATT_N0 : 20
println("attach steps N0 = ", N0)
t = @elapsed hs = [wf(c) for c in cfgs]
@printf("time %.1f s\n", t)
haskey(kv, "save") && serialize(kv["save"], hs)
if haskey(kv, "ref")
    old = deserialize(kv["ref"])
    ip(a, b) = real(sum(conj.(a) .* b .* w))
    r = map(i -> maximum(((A, B),) -> ip(A - B, A - B) / ip(A, A), zip(old[i], hs[i])), 1:n)
    for (i, c) in enumerate(cfgs)
        @printf("%2d m1 %.1f m2 %.1f |chi1| %.2f |chi2| %.2f  ||dh||^2/||h||^2 %.2e\n", i, c[1], c[2], sqrt(sum(abs2, c[3])),
                sqrt(sum(abs2, c[4])), r[i])
    end
    @printf("median %.2e  max %.2e\n", sort(r)[cld(n, 2)], maximum(r))
end
