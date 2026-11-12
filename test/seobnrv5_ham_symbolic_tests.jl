using Test, GravitationalWaveforms, Random

# The generated SEOBNRv5 / SEOBNRv5PHM Hamiltonians and derivatives (gen/symbolic/generate.jl, source
# gen/symbolic/hamiltonians/v5phm.jl): derivatives against 256-bit central differences of the generated values, the
# fused sets against the single-purpose functions, aligned against precessing at zero in-plane spin. The values
# themselves are checked against pyseobnr in seobnrv5alignedspin_*_tests.jl and seobnrv5phm_dyn_tests.jl. Errors are
# normwise (max |error| / max |entry|): single entries such as d2H/dr dprst at small prst are cancellation-dominated.

const _SG = GravitationalWaveforms
_nw(a, b) = maximum(abs.(collect(a) .- collect(b))) / maximum(abs, collect(b))
_unit(v) = v ./ sqrt(sum(abs2, v))

# d f(p) / d p[j] for a tuple-valued f, central difference in BigFloat (h = 1e-30 relative: exact to ~1e-50)
function _cfd(f, p, j)
    h = big(1e-30) * max(abs(p[j]), 1.0)
    pp = collect(big.(p)); pm = copy(pp); pp[j] += h; pm[j] -= h
    return Float64.((collect(f(pp...)) .- collect(f(pm...))) ./ (2h))
end

function _phm_point(rng)
    q = 1 + 19 * rand(rng)
    X1, X2 = q / (1 + q), 1 / (1 + q)
    nu = X1 * X2
    c1 = 0.99 * rand(rng) .* _unit(randn(rng, 3))
    c2 = 0.99 * rand(rng) .* _unit(randn(rng, 3))
    r = 2.2 + 28 * rand(rng)
    ap, am = X1 * c1[3] + X2 * c2[3], X1 * c1[3] - X2 * c2[3]
    return (r, -0.3 * rand(rng) / sqrt(r), sqrt(r) * (0.9 + 0.3 * rand(rng)), c1..., c2..., c1[3], c2[3],
            0.98 * c1[3], 1.01 * c2[3], nu, X1, X2, X1 - X2, _SG._seobv5_a6_NS(nu), _SG._seobv5_dSO(nu, ap, am))
end

function _aligned_point(rng)
    q = 1 + 19 * rand(rng)
    X1, X2 = q / (1 + q), 1 / (1 + q)
    nu = X1 * X2
    c1, c2 = 0.99 * (2rand(rng) - 1), 0.99 * (2rand(rng) - 1)
    r = 2.2 + 28 * rand(rng)
    return (r, -0.3 * rand(rng) / sqrt(r), sqrt(r) * (0.9 + 0.3 * rand(rng)), nu, X1, X2, c1, c2,
            _SG._seobv5_a6_NS(nu), _SG._seobv5_dSO(nu, X1 * c1 + X2 * c2, X1 * c1 - X2 * c2))
end

@testset "SEOBNRv5PHM generated derivatives vs BigFloat central differences" begin
    rng = Xoshiro(11)
    Hc(a...) = (_SG._seobv5phm_ham_call(a...)[1],)
    gr(a...) = _SG._seobv5phm_sym_grad(a...)
    pot(a...) = _SG._seobv5phm_ham_call(a...)
    worst = zeros(5)
    for _ in 1:100
        p = _phm_point(rng)
        all(isfinite, _SG._seobv5phm_ham_call(p...)) || continue
        g = _SG._seobv5phm_sym_grad(p...)
        worst[1] = max(worst[1], _nw(g, [_cfd(Hc, p, j)[1] for j in 1:3]))
        h = _SG._seobv5phm_sym_hessian(p...)
        worst[2] = max(worst[2], _nw(h, vcat((_cfd(gr, p, j) for j in 1:3)...)))   # symmetric: row-major == column order
        worst[3] = max(worst[3], _nw(_SG._seobv5phm_ham_dynamics(p...),
            (g[1], 0.0, g[2], g[3], _SG._seobv5phm_ham_call(p...)[1], _SG._seobv5phm_ham_csi(p...))))
        worst[4] = max(worst[4], abs(_SG._seobv5phm_ham_omega(p...) - g[3]) / abs(g[3]))
        dr, dp = _cfd(pot, p, 1), _cfd(pot, p, 2)                                   # (H, xi, A, Bnp, Bnpa, Q, Heven, Hodd, Bp)
        worst[5] = max(worst[5], _nw(_SG._seobv5phm_ham_auxderivs(p...), (dr[3], dr[4], dr[5], dr[2], dr[6], dp[6], dr[8], dr[9], dr[7])))
    end
    @test all(<(1e-11), worst)
    all(<(1e-11), worst) || @info "PHM normwise errors (grad, Hessian, dynamics, omega, auxderivs)" worst
end

@testset "SEOBNRv5 aligned spin: generated = SEOBNRv5PHM at zero in-plane spin; gradient vs BigFloat" begin
    rng = Xoshiro(12)
    Hb(a...) = (_SG._seobv5_potentials(a...)[1],)
    epot = egr = 0.0
    for _ in 1:300
        p = _aligned_point(rng)
        r, prst, pphi, nu, X1, X2, c1, c2, a6, dSO = p
        al = _SG._seobv5_potentials(p...)
        ph = _SG._seobv5phm_ham_call(r, prst, pphi, 0.0, 0.0, c1, 0.0, 0.0, c2, c1, c2, c1, c2, nu, X1, X2, X1 - X2, a6, dSO)
        epot = max(epot, maximum(abs.(collect(al) .- (ph[1], ph[2], ph[3], ph[4], ph[5], ph[6], ph[7], ph[7] + ph[8])) ./
                                 abs.(collect(al))))
        g = _SG._seobv5_sym_gradient(p...)
        egr = max(egr, _nw(g, [_cfd(Hb, p, j)[1] for j in 1:3]))
    end
    @test epot < 1e-13
    @test egr < 1e-12
    p32 = Float32.((10.0, -0.01, 3.3, 0.2, 0.7236, 0.2764, 0.3, 0.1, -45.0, -30.0))
    @test eltype(collect(_SG._seobv5_sym_gradient(p32...))) === Float32
    @test eltype(collect(_SG._seobv5_potentials(p32...))) === Float32
end

@testset "SEOBNRv5 aligned-spin fused generated terms vs potentials, gradient, BigFloat derivatives" begin
    rng = Xoshiro(13)
    gb(a...) = _SG._seobv5_sym_gradient(a...)
    pot(a...) = _SG._seobv5_potentials(a...)                                       # (H, xi, A, Bnp, Bnpa, Q, Heven, Heff)
    worst = Dict{Symbol,Float64}()
    upd!(k, a, b) = (worst[k] = max(get(worst, k, 0.0), _nw(a, b)))
    for _ in 1:100
        p = _aligned_point(rng)
        e = (p[4:end]...,)
        c = (p[1], 0.0, p[3:end]...)                     # the same point at prst = 0
        H, xi, A, Bnp, Bnpa, Qq, Heven, Heff = _SG._seobv5_potentials(p...)
        dHdr, dHdpr, om = _SG._seobv5_sym_gradient(p...)
        cdHdr, _, omc = _SG._seobv5_sym_gradient(c...)
        dr, dp = _cfd(pot, p, 1), _cfd(pot, p, 2)
        dHodd = dr[8] - dr[7]                            # Hodd = Heff - Heven
        aux = (dr[3], dr[4], dr[5], dr[2], dr[6], dp[6], dHodd)
        upd!(:auxderivs, _SG._seobv5_sym_auxderivs(p...), aux)
        upd!(:rhs, _SG._seobv5_sym_rhs(p...), (H, xi, Heff, dHdr, dHdpr, om, omc))
        upd!(:state, _SG._seobv5_sym_state(p...), (H, Heff, omc))
        upd!(:tphi, _SG._seobv5_sym_tphi(p...), (xi, dHdpr, om))
        upd!(:pr_terms, _SG._seobv5_sym_pr_terms(p...), (H, xi, A, Bnp, Heven, Heff, aux[6], om, omc))
        upd!(:pphi_terms, _SG._seobv5_sym_pphi_terms(p...), (H, xi, A, Bnp, Bnpa, Qq, Heven, Heff, aux..., om, omc))
        upd!(:omega_circ, (_SG._seobv5_sym_omega_circ(p[1], p[3], e...),), (omc,))
        hp1, hp3 = _cfd(gb, p, 1), _cfd(gb, p, 3)
        hc1, hc3 = _cfd(gb, c, 1), _cfd(gb, c, 3)
        upd!(:diss_terms, _SG._seobv5_sym_diss_terms(p...), (H, xi, Heff, dHdpr, om, omc, hp1[1], hp3[1]))
        upd!(:j0, _SG._seobv5_sym_j0(p[1], p[3], e...), (cdHdr, hc3[1]))
        upd!(:circ_hessian, _SG._seobv5_sym_circ_hessian(p[1], p[3], e...), (cdHdr, omc, hc1[1], hc3[1], hc3[3]))
    end
    for (k, v) in sort(collect(worst))
        @test v < 1e-11
        v < 1e-11 || @info "aligned $k normwise error $v"
    end
end

@testset "SEOBNRv5PHM fused generated terms vs the single-purpose generated functions" begin
    rng = Xoshiro(15)
    worst = Dict{Symbol,Float64}()
    upd!(k, a, b) = (worst[k] = max(get(worst, k, 0.0), _nw(a, b)))
    for _ in 1:300
        p = _phm_point(rng)
        c = (p[1], 0.0, p[3:end]...)                     # the same point at prst = 0
        H, xi, A, Bnp, Bnpa, Q, Heven, Hodd, Bp = _SG._seobv5phm_ham_call(p...)
        isfinite(H) || continue
        dHdr, dHdpr, om = _SG._seobv5phm_sym_grad(p...)
        h = _SG._seobv5phm_sym_hessian(p...)
        omc = _SG._seobv5phm_sym_omega(c...)
        gc = _SG._seobv5phm_sym_grad(c...)
        hc = _SG._seobv5phm_sym_hessian(c...)
        aux = _SG._seobv5phm_sym_auxderivs(p...)
        upd!(:rhs, _SG._seobv5phm_sym_rhs(p...), (dHdr, dHdpr, om, H, xi, omc))
        upd!(:tphi, _SG._seobv5phm_sym_tphi(p...), (xi, dHdpr, om))
        upd!(:state, _SG._seobv5phm_sym_state(p...), (H, Heven + Hodd, omc))
        upd!(:pr_terms, _SG._seobv5phm_sym_pr_terms(p...), (H, xi, A, Bnp, Heven, Hodd, aux[6], om, omc))
        upd!(:pphi_terms, _SG._seobv5phm_sym_pphi_terms(p...), (H, xi, A, Bnp, Bnpa, Q, Heven, Hodd, Bp, aux[1:8]..., om, omc))
        upd!(:diss_terms, _SG._seobv5phm_sym_diss_terms(p...), (H, xi, dHdpr, om, omc, h[1], h[3]))
        upd!(:omega_circ, (_SG._seobv5phm_sym_omega_circ(p...),), (omc,))
        upd!(:j0, _SG._seobv5phm_sym_j0(p...), (gc[1], hc[3]))
        upd!(:circ_hessian, _SG._seobv5phm_sym_circ_hessian(p...), (gc[1], gc[3], hc[1], hc[3], hc[9]))
    end
    for (k, v) in sort(collect(worst))
        @test v < 1e-12
        v < 1e-12 || @info "PHM $k normwise error $v"
    end
end
