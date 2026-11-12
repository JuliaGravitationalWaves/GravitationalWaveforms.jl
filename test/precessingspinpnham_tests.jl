# Unit + physics-consistency tests for GravitationalWaveforms.PrecessingSpinPNHam. See PrecessingSpinPNHam.jl's own module docstring for the full physics
# derivation/citations each check below refers to. Validation approach:
#
#  1. Basic construction/evolve sanity (finite outputs, scalar vs vector, Float32 genericity).
#  2. validate_precessingspinpnham_params boundary behavior.
#  3. the PNFlux_*SOCoeff vs TaylorT4wdot_*SOCoeff coefficient-family resolution, checked
#     numerically (does the resulting series match, termwise).
#  4. cross-product allocation-free confirmation (StaticArrays' cross/LinearAlgebra.cross
#     for SVector{3,T} compiles cleanly, generically over T, inside this package's style).
#  5. aligned-spin limit -- LNhat(t) stays EXACTLY (0,0,1) (an algebraic identity, not
#     just numerically small), and phase(t)/v(t) match an independent, direct re-implementation of
#     the same v-power-series domega/dt formula (bypassing _psph_core's cross-product machinery
#     entirely) to RK4-and-floating-point precision.
#  6. v0 = cbrt(pi*M_sec*f_ref) reproduces real LALSuite's own initial V exactly (see
#     PrecessingSpinPNHam.jl's docstring's "Initial conditions" section for the check this
#     mirrors).
#  7. E1 self-consistency: E1(t) stays exactly perpendicular to LNhat(t), and the (alpha,beta,
#     epsilon) reconstruction in _psph_angles is self-consistent (reproduces E1(t) when the same
#     rotation composition is re-applied forward).
#  8. Finite-difference smoothness sanity, complementing the Enzyme.gradient validation in
#     `precessingspinpnham_enzyme_ka_tests.jl`.

using LinearAlgebra: cross
using StaticArrays: SVector

@testset "PrecessingSpinPNHam construction and evolve basic sanity" begin
    model = PrecessingSpinPNHam()
    @test model isa PrecessingSpinPNHam

    m1, m2 = 30.0, 20.0
    chi1x, chi1y, chi1z = 0.3, 0.1, 0.2
    chi2x, chi2y, chi2z = -0.1, 0.2, 0.4
    dL, inclination, phiRef = 1.0, 0.8, 1.2
    dt_default = 1.0 / 4096.0
    t = (0:20:200) .* dt_default

    hp, hc = evolve(model, t, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dL, inclination, phiRef)
    @test hp isa Vector{ComplexF64}
    @test hc isa Vector{ComplexF64}
    @test length(hp) == length(t)
    @test length(hc) == length(t)
    @test all(isfinite, real(hp)) && all(isfinite, imag(hp))
    @test all(isfinite, real(hc)) && all(isfinite, imag(hc))
    @test all(>(0), abs.(hp) .+ abs.(hc))

    # Scalar t returns a scalar pair, not a vector.
    hp_s, hc_s = evolve(model, 0.02, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dL, inclination, phiRef)
    @test hp_s isa ComplexF64
    @test hc_s isa ComplexF64

    # Non-spin limit is finite and well-behaved too.
    hp0, hc0 = evolve(model, t, m1, m2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, dL, inclination, phiRef)
    @test all(isfinite, real(hp0)) && all(isfinite, real(hc0))

    # Float32 genericity.
    t32 = Float32.(t)
    hp32, hc32 = evolve(model, t32, Float32(m1), Float32(m2),
        Float32(chi1x), Float32(chi1y), Float32(chi1z),
        Float32(chi2x), Float32(chi2y), Float32(chi2z),
        Float32(dL), Float32(inclination), Float32(phiRef))
    @test eltype(hp32) === ComplexF32
    @test eltype(hc32) === ComplexF32
    @test all(isfinite, real(hp32)) && all(isfinite, real(hc32))
end

@testset "validate_precessingspinpnham_params" begin
    @test validate_precessingspinpnham_params(30.0, 20.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3) === nothing
    @test validate_precessingspinpnham_params(30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; f_ref = 20.0) === nothing
    @test_throws ArgumentError validate_precessingspinpnham_params(20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # m1 < m2
    @test_throws ArgumentError validate_precessingspinpnham_params(30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # m2 must be > 0
    @test_throws ArgumentError validate_precessingspinpnham_params(30.0, 20.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)  # |chi1| > 1
    @test_throws ArgumentError validate_precessingspinpnham_params(30.0, 20.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)  # |chi2| > 1
    @test_throws ArgumentError validate_precessingspinpnham_params(30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; f_ref = -1.0)
end

@testset "PNFlux vs TaylorT4wdot coefficient-family resolution" begin
    # AlignedSpinPNHam._asph_flux_coeffs ports XLALSimInspiralPNFlux_3PNSOCoeff(x) = -3/2-5/(4x)
    # (1.5PN spin-orbit flux). PrecessingSpinPNHam ports XLALSimInspiralTaylorT4wdot_3PNSOCoeff(x)
    # = -19/6-25/(4x) directly (used in domega/dt above). The standard energy-balance re-expansion
    # identity (derived in PrecessingSpinPNHam.jl's own docstring): at the LEADING (1.5PN) spin-
    # orbit order, domega/dt's SO coefficient equals f3SO(x) - (5/2)*e3SO(x), with e3SO =
    # XLALSimInspiralPNEnergy_3PNSOCoeff(x) = 2/3+2/x (AlignedSpinPNHam's own E3S1O). Confirmed here
    # this equals TaylorT4wdot_3PNSOCoeff(x) EXACTLY (to machine precision): the two coefficient families are NOT in conflict.
    f3SO(x) = -3.0 / 2.0 - 5.0 / (4.0 * x)
    e3SO(x) = 2.0 / 3.0 + 2.0 / x
    wdot3SO(x) = -19.0 / 6.0 - 25.0 / (4.0 * x)
    for x in (0.3, 0.5, 0.7, 0.9, 1.0)
        lhs = f3SO(x) - 2.5 * e3SO(x)
        @test isapprox(lhs, wdot3SO(x); rtol = 1e-12)
        @test isapprox(lhs, GravitationalWaveforms._psph_coeffs(PrecessingSpinPNHam(), x*50.0, (1-x)*50.0,
            0.0,0.0,0.0, 0.0,0.0,0.0).wdot3S1O; rtol = 1e-12)
    end
end

@testset "Cross product allocation-free confirmation" begin
    a = SVector(1.0, 2.0, 3.0)
    b = SVector(4.0, 5.0, 6.0)
    @test cross(a, b) == SVector(-3.0, 6.0, -3.0)
    @test (@allocated cross(a, b)) == 0

    a32 = SVector(1.0f0, 2.0f0, 3.0f0)
    b32 = SVector(4.0f0, 5.0f0, 6.0f0)
    @test cross(a32, b32) isa SVector{3, Float32}
    @test (@allocated cross(a32, b32)) == 0
end

# Independent re-implementation of the aligned-spin-limit domega/dt series (bypassing
# _psph_core's cross-product machinery entirely) -- used only to cross-check PrecessingSpinPNHam's
# own phase(t)/v(t) output in that limit.
function _ref_aligned_domega(v, eta, wdotcoeff2, wdotcoeff3, wdot3S1O, wdot3S2O, chi1, chi2)
    v2 = v * v
    v3 = v2 * v
    v11 = v^11
    wdotnewt = 96.0 / 5.0 * eta
    wspin3 = wdot3S1O * chi1 + wdot3S2O * chi2
    return wdotnewt * v11 * (1.0 + v2 * wdotcoeff2 + v3 * (wdotcoeff3 + wspin3))
end

function _ref_aligned_rk4_integrate(v0, dt, nsteps, eta, wdotcoeff2, wdotcoeff3, wdot3S1O, wdot3S2O, chi1, chi2)
    phase, v = 0.0, v0
    rhs(v) = (v^3, _ref_aligned_domega(v, eta, wdotcoeff2, wdotcoeff3, wdot3S1O, wdot3S2O, chi1, chi2) / (3 * v^2))
    for _ in 1:nsteps
        k1 = rhs(v)
        k2 = rhs(v + (dt/2)*k1[2])
        k3 = rhs(v + (dt/2)*k2[2])
        k4 = rhs(v + dt*k3[2])
        phase += (dt/6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        v += (dt/6) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    end
    return phase, v
end

@testset "Aligned-spin limit" begin
    model = PrecessingSpinPNHam()
    m1, m2 = 30.0, 20.0
    chi1, chi2 = 0.5, -0.3
    dt = 1.0 / 4096.0
    coeffs = GravitationalWaveforms._psph_coeffs(model, m1, m2, 0.0, 0.0, chi1, 0.0, 0.0, chi2; f_ref = 20.0)

    for t in (0.0, 0.01, 0.03, 0.05, 0.1)
        st = GravitationalWaveforms._psph_state_scalar(coeffs, t, dt)
        # LNhat(t) stays EXACTLY (0,0,1) -- algebraic identity (see PrecessingSpinPNHam.jl's
        # docstring's "Aligned-spin limit" section), not merely numerically small.
        @test st.LNhx == 0.0
        @test st.LNhy == 0.0
        @test st.LNhz == 1.0
        # S1, S2 never change (dS_i/dt = c_i*v^5*(LNhat x S_i) = 0 exactly when S_i || LNhat).
        @test st.S1x == 0.0 && st.S1y == 0.0
        @test st.S2x == 0.0 && st.S2y == 0.0

        nsteps = round(Int, t / dt)
        dt_hat = dt / coeffs.M_sec
        # _psph_core's wspin3 uses LNhat.S_i with the MASS-REDUCED spin S_i = X_i^2*chi_i (matching
        # LALSimInspiralSpinTaylor.c's own internal convention, confirmed directly against real
        # LALSuite output) -- so the reference series must use the same reduced spin, not
        # raw chi_i, to be a genuine apples-to-apples independent re-implementation.
        ref_phase, ref_v = _ref_aligned_rk4_integrate(coeffs.v0, dt_hat, nsteps, coeffs.eta,
            coeffs.wdotcoeff2, coeffs.wdotcoeff3, coeffs.wdot3S1O, coeffs.wdot3S2O,
            coeffs.X1^2 * chi1, coeffs.X2^2 * chi2)
        @test isapprox(st.v, ref_v; rtol = 1e-12)
        @test isapprox(st.phase, ref_phase; rtol = 1e-12, atol = 1e-12)
    end

    # v(t) increases monotonically (chirping) over this safe range, and differs from the
    # non-spin case (spin has a real, non-degenerate effect on this model's own dynamics too --
    # same style of check as AlignedSpinPNHam's "Spin dependence" testset).
    ts = (0.0:0.01:0.1)
    vs_spin = [GravitationalWaveforms._psph_state_scalar(coeffs, t, dt).v for t in ts]
    coeffs0 = GravitationalWaveforms._psph_coeffs(model, m1, m2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; f_ref = 20.0)
    vs_nonspin = [GravitationalWaveforms._psph_state_scalar(coeffs0, t, dt).v for t in ts]
    @test issorted(vs_spin)
    @test issorted(vs_nonspin)
    @test vs_spin != vs_nonspin
end

@testset "v0 formula matches LALSuite's own initial V" begin
    # Direct check (recorded once, offline, against real lalsimulation -- see
    # PrecessingSpinPNHam.jl's docstring's "Initial conditions" section): m1=30, m2=20 Msun,
    # fStart=20 Hz, LALSuite's SimInspiralSpinTaylorPNEvolveOrbit's own first output sample
    # V[0] = 0.24919144739758323 exactly, both sides using LAL_MTSUN_SI-equivalent constants.
    coeffs = GravitationalWaveforms._psph_coeffs(PrecessingSpinPNHam(), 30.0, 20.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0; f_ref = 20.0, GMsun_over_c3 = 4.925490947641267e-06)
    @test isapprox(coeffs.v0, 0.24919144739758323; rtol = 1e-12)
end

@testset "E1 companion self-consistency (evolve's epsilon reconstruction)" begin
    model = PrecessingSpinPNHam()
    m1, m2 = 20.0, 15.0
    chi1x, chi1y, chi1z = 0.4, 0.2, 0.1
    chi2x, chi2y, chi2z = -0.2, 0.3, -0.1
    coeffs = GravitationalWaveforms._psph_coeffs(model, m1, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z; f_ref = 20.0)
    dt = 1.0 / 4096.0

    for t in (0.01, 0.03, 0.05)
        state, E1x, E1y, E1z = GravitationalWaveforms._psph_state_with_E1_scalar(coeffs, t, dt)
        # E1(t) stays exactly (to floating-point/RK4 precision) perpendicular to LNhat(t).
        dotp = state.LNhx * E1x + state.LNhy * E1y + state.LNhz * E1z
        @test abs(dotp) < 1e-8
        # |E1(t)| stays close to 1 (same drift-characterization spirit as LNhat's own unit-norm
        # check -- both driven by the same Om, same construction).
        @test isapprox(E1x^2 + E1y^2 + E1z^2, 1.0; atol = 1e-6)

        Nx, Ny, Nz = 0.0, 0.0, 1.0  # arbitrary fixed line-of-sight for this self-consistency check
        alpha, cos_beta, epsilon, thetaJN = GravitationalWaveforms._psph_angles(state, coeffs, E1x, E1y, E1z, Nx, Ny, Nz)
        beta = acos(clamp(cos_beta, -1.0, 1.0))

        # Self-consistency: re-applying Rz(phiJ)Ry(thetaJ)Rz(alpha)Ry(beta) to (1,0,0) should
        # reproduce E1(t) (this checks _psph_angles' own rotation composition is invertible/
        # self-consistent, not that the sign convention matches an external ground truth -- see
        # evolve's docstring for that caveat).
        L = coeffs.eta / state.v
        Jx = L*state.LNhx + state.S1x + state.S2x
        Jy = L*state.LNhy + state.S1y + state.S2y
        Jz = L*state.LNhz + state.S1z + state.S2z
        Jnorm = sqrt(Jx^2+Jy^2+Jz^2)
        phiJ = atan(Jy/Jnorm, Jx/Jnorm)
        thetaJ = acos(clamp(Jz/Jnorm, -1.0, 1.0))

        ex1, ey1, ez1 = GravitationalWaveforms._xp_rotate_z(alpha, 1.0, 0.0, 0.0)
        ex2, ey2, ez2 = GravitationalWaveforms._xp_rotate_y(beta, ex1, ey1, ez1)
        ex3, ey3, ez3 = GravitationalWaveforms._xp_rotate_z(epsilon, ex2, ey2, ez2)
        # Undo the epsilon rotation direction used in _psph_angles's own forward construction:
        # _psph_angles extracts epsilon via rotate_z(-alpha) then rotate_y(-beta) applied to E1
        # (already in the Jhat frame), so the forward map is rotate_y(beta) then rotate_z(alpha)
        # applied to Rz(epsilon)*(1,0,0), then finally back into the global frame via
        # rotate_y(thetaJ) then rotate_z(phiJ).
        fx1, fy1, fz1 = GravitationalWaveforms._xp_rotate_z(epsilon, 1.0, 0.0, 0.0)
        fx2, fy2, fz2 = GravitationalWaveforms._xp_rotate_y(beta, fx1, fy1, fz1)
        fx3, fy3, fz3 = GravitationalWaveforms._xp_rotate_z(alpha, fx2, fy2, fz2)
        gx1, gy1, gz1 = GravitationalWaveforms._xp_rotate_y(thetaJ, fx3, fy3, fz3)
        gx2, gy2, gz2 = GravitationalWaveforms._xp_rotate_z(phiJ, gx1, gy1, gz1)
        @test isapprox(gx2, E1x; atol = 1e-6)
        @test isapprox(gy2, E1y; atol = 1e-6)
        @test isapprox(gz2, E1z; atol = 1e-6)
    end
end

@testset "Finite-difference smoothness sanity" begin
    model = PrecessingSpinPNHam()
    m1, m2 = 30.0, 20.0
    chi1x, chi1y, chi1z = 0.3, 0.1, 0.2
    chi2x, chi2y, chi2z = -0.1, 0.2, 0.4
    dL, inclination, phiRef = 1.0, 0.8, 1.2
    t = 0.03

    f(m1p) = real(evolve(model, t, m1p, m2, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, dL, inclination, phiRef)[1])
    for h_rel in (1e-5, 1e-6)
        h = h_rel * m1
        d = (f(m1 + h) - f(m1 - h)) / (2h)
        @test isfinite(d)
    end
    d1 = (f(m1 + 1e-5*m1) - f(m1 - 1e-5*m1)) / (2e-5*m1)
    d2 = (f(m1 + 1e-6*m1) - f(m1 - 1e-6*m1)) / (2e-6*m1)
    @test isapprox(d1, d2; rtol = 5e-2)

    g(chi1zp) = real(evolve(model, t, m1, m2, chi1x, chi1y, chi1zp, chi2x, chi2y, chi2z, dL, inclination, phiRef)[1])
    for h in (1e-5, 1e-6)
        d = (g(chi1z + h) - g(chi1z - h)) / (2h)
        @test isfinite(d)
    end
end
