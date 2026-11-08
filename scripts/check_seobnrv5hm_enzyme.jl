# Enzyme reverse gradient through the whole SEOBNRv5HM frequency-domain build (coefficients + table + six-mode
# evaluation) vs central finite differences with a step scan (the model's output carries ~4e-9 relative numerical
# noise, so a single small step is not a reference). Too slow for the test suite: the first reverse call compiles for
# ~35 min on a 14 GB machine; run no other Enzyme compile beside it.
#   julia --project=<env with GravitationalWaveforms + Enzyme> scripts/check_seobnrv5hm_enzyme.jl
# Reference run: compile 2084 s, call 0.07 s; best-step agreement mc 4.7e-8, eta 7.5e-7,
# chi1 1.6e-7, chi2 1.5e-7.
using GravitationalWaveforms, Enzyme, Printf
const G = GravitationalWaveforms
Enzyme.API.maxtypeoffset!(65536)

const FS = (30.0, 60.0, 90.0, 140.0, 220.0, 350.0)

function hm_loss(mc, eta, chi1, chi2)
    T = G._seobv5hm_alloc()
    c = G._seobv5hm_table(G._seobv5hm_coeffs!(T, mc, eta, chi1, chi2; f_min = 15.0), T)
    s = 0.0
    for k in 1:6, (j, f) in enumerate(FS)
        h = G._seobv5hm_fd_eval(c, k, f)
        s += 1e3 * ((1.0 + 0.1k) * real(h) + (0.5 + 0.07j) * imag(h))
    end
    return s
end

x = (17.0, 0.21, 0.4, -0.3)      # mc ~ 17 (M ~ 40), q ~ 2.3
t0 = time()
g = Enzyme.autodiff(Enzyme.Reverse, hm_loss, Enzyme.Active, map(Enzyme.Active, x)...)[1]
@printf("first reverse call (compile) %.1f s\n", time() - t0); flush(stdout)
t0 = time()
g = Enzyme.autodiff(Enzyme.Reverse, hm_loss, Enzyme.Active, map(Enzyme.Active, x)...)[1]
@printf("second reverse call %.2f s\n", time() - t0); flush(stdout)
for k in 1:4
    best = Inf
    for e in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7)
        h = e * max(abs(x[k]), 0.1)
        xp = ntuple(j -> j == k ? x[j] + h : x[j], 4); xm = ntuple(j -> j == k ? x[j] - h : x[j], 4)
        best = min(best, abs((hm_loss(xp...) - hm_loss(xm...)) / (2h) / g[k] - 1))
    end
    @printf("d/dx%d  enzyme %+.10e  best-step fd rel diff %.2e\n", k, g[k], best); flush(stdout)
end
