using Test, GravitationalWaveforms, Random

# `_seobv5f2_dense_to_freq` at the table's top frequency (omega_target == omegas[nreal], what callers get when they clamp
# f to fmax, e.g. GravitationalWaveBayes' f_ref time shift): the root must be the last real node (s = 1). Round-off
# could put the right root at 1 + eps and the other, spurious root was taken (t_star ~0.4 s off for some binaries).
@testset "SEOBNRv5AlignedSpinF2: dense_to_freq at the table top returns the last node" begin
    m = GravitationalWaveforms.SEOBNRv5AlignedSpinF2()
    rng = Xoshiro(1)
    nbad = 0
    for _ in 1:40
        m1 = 35 + 4rand(rng); m2 = 30 + 4rand(rng)
        mc = (m1 * m2)^0.6 / (m1 + m2)^0.2; eta = m1 * m2 / (m1 + m2)^2
        c = GravitationalWaveforms._seobv5f2_coeffs(m, mc, eta, 0.3randn(rng), 0.3randn(rng))
        c.success || continue
        r = GravitationalWaveforms._seobv5f2_dense_to_freq(c.ts, c.phis, c.omegas, c.nreal, c.M_sec, c.fmax)
        nbad += isapprox(r.t_star, c.ts[c.nreal]; rtol = 1e-9) ? 0 : 1
        # just inside the top: continuous with the edge value
        r2 = GravitationalWaveforms._seobv5f2_dense_to_freq(c.ts, c.phis, c.omegas, c.nreal, c.M_sec, c.fmax * (1 - 1e-10))
        nbad += isapprox(r2.t_star, c.ts[c.nreal]; rtol = 1e-6) ? 0 : 1
    end
    @test nbad == 0
end
