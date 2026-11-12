# Unit tests for GravitationalWaveforms.ODEToy.
#
# ODEToy is explicit scaffolding -- a 0PN, non-spinning, radiation-reaction-only toy model with
# no PN corrections and no LALSuite equivalent worth comparing against (LALSuite's simplest
# generators are already at least Newtonian-restricted-amplitude with proper PN phasing, not a
# bare 0PN two-variable ODE). Its job is to validate the StaticArrays + fixed-step-RK4 +
# Enzyme/KernelAbstractions architecture pattern (see test/odetoy_enzyme_ka_tests.jl for that),
# not to be a physically trustworthy waveform -- so validation here is: (1) basic Phi/Ampl
# dispatch sanity + Float32 genericity, matching every other model in this package, (2) a chirp
# sanity check (frequency must increase over time), and (3) an independent closed-form
# cross-check of the RK4 integrator itself against the ODE's own analytic solution (see
# `odetoy_omega_analytic`'s docstring for the derivation).

@testset "ODEToy construction and state arithmetic" begin
    model = ODEToy()
    @test model isa ODEToy

    # `ODEToyState`'s `+`/scalar-`*` come from `StaticArrays.jl`'s `FieldVector` machinery (see
    # that struct's docstring) -- confirm directly, since an RK4 step is built entirely out of
    # this algebra.
    a = ODEToyState(1.0, 2.0)
    b = ODEToyState(0.5, 0.25)
    @test a + b == ODEToyState(1.5, 2.25)
    @test 2 * a == ODEToyState(2.0, 4.0)
    @test a * 2 == ODEToyState(2.0, 4.0)
    @test (a + b) isa ODEToyState{Float64}
    @test (2 * a) isa ODEToyState{Float64}
end

@testset "ODEToy Phi/Ampl basic sanity" begin
    model = ODEToy()

    # mc = chirp mass (Msun); omega0 corresponds to a 20 Hz GW frequency at coalescence-relative
    # step 0 (f_gw = omega0/pi, see ODEToy's docstring's Phi=2*phi convention), matching this
    # ecosystem's usual f_min baseline. dt matches TaylorT3/data-conditioning's established
    # deltaT. This (mc, omega0) pair reaches its formal 0PN coalescence at t ~= 1.03s (computed
    # directly from `odetoy_omega_analytic`'s underlying formula), so step counts here are kept
    # well below that (nsteps*dt <~ 0.6s) to stay inside the model's valid domain.
    mc, omega0, dt = 25.0, pi * 20.0, 1.0 / 4096.0

    ns = collect(0:400:2400)  # up to t = 2400*dt ~= 0.586s, comfortably < tc ~= 1.03s
    phi = Phi(model, ns, mc, omega0; dt = dt)
    ampl = Ampl(model, ns, mc, omega0, 1.0; dt = dt)

    @test phi isa Vector{Float64}
    @test ampl isa Vector{Float64}
    @test length(phi) == length(ns)
    @test length(ampl) == length(ns)
    @test all(isfinite, phi)
    @test all(isfinite, ampl)
    @test all(>(0), ampl)
    @test phi[1] == 0.0  # n=0 => phase=0 by construction

    # Chirp sanity: GW phase must increase monotonically with step count (omega0 > 0 always
    # drives dphi/dt = omega > 0), and the instantaneous GW frequency (from consecutive-step
    # phase differences) must also increase monotonically -- the defining behavior of a chirp,
    # driven purely by the radiation-reaction term, independent of any closed-form comparison.
    @test issorted(phi)
    f_gw = diff(phi) ./ diff(ns .* dt) ./ (2pi)
    @test all(>(0), f_gw)
    @test issorted(f_gw)

    # Amplitude (the explicitly-non-physical placeholder -- see Ampl's docstring) is at least a
    # smooth, monotonically increasing function of the chirp, matching its documented intent.
    @test issorted(ampl)

    # Scalar step-count input returns a scalar, not a vector. 800 is deliberately one of the
    # values already in `ns` above, so its result can be cross-checked against the array call.
    phi_scalar = Phi(model, 800, mc, omega0; dt = dt)
    ampl_scalar = Ampl(model, 800, mc, omega0, 1.0; dt = dt)
    @test phi_scalar isa Float64
    @test ampl_scalar isa Float64
    @test phi_scalar == phi[findfirst(==(800), ns)]

    # Float32 genericity (the entire reason TypeGenericity.jl's @Tlit macro exists): output type
    # must track input type, not silently promote back to Float64. `n` (step counts) stays
    # `Integer` regardless -- only the physical quantities (mc, omega0, dt, dL, phiRef) carry the
    # working type.
    phi32 = Phi(model, ns, Float32(mc), Float32(omega0); dt = Float32(dt))
    ampl32 = Ampl(model, ns, Float32(mc), Float32(omega0), Float32(1.0); dt = Float32(dt))
    @test eltype(phi32) === Float32
    @test eltype(ampl32) === Float32
    @test all(isfinite, phi32)
    @test all(isfinite, ampl32)

    phi32_scalar = Phi(model, 1000, Float32(mc), Float32(omega0); dt = Float32(dt))
    @test phi32_scalar isa Float32
end

@testset "ODEToy RK4 integrator vs. closed-form 0PN analytic solution" begin
    # Independent correctness check on `odetoy_rk4_integrate` itself (nothing to do with
    # Enzyme/KernelAbstractions): the 0PN ODE has a closed-form solution (see
    # `odetoy_omega_analytic`'s docstring), so a correct fixed-step RK4 implementation should
    # match it to within ordinary RK4 truncation error (tiny, for a step this small relative to
    # the timescale over which omega changes) -- not just "look qualitatively similar."
    mc, omega0, dt = 25.0, pi * 20.0, 1.0 / 4096.0
    Mc_sec = mc * GravitationalWaveforms.GMsun_over_c3

    for nsteps in (10, 100, 500, 1000, 2000, 3500)  # up to t ~= 0.854s, still < tc ~= 1.03s
        t = nsteps * dt
        state0 = ODEToyState(0.0, omega0)
        final = odetoy_rk4_integrate(state0, dt, nsteps, Mc_sec)
        analytic = odetoy_omega_analytic(t, omega0, Mc_sec)
        @test isapprox(final.omega, analytic; rtol = 1e-8)
    end
end

@testset "validate_odetoy_params" begin
    @test validate_odetoy_params(25.0, pi * 20.0) === nothing
    @test validate_odetoy_params(25.0, pi * 20.0; dt = 1.0 / 4096.0) === nothing
    @test_throws ArgumentError validate_odetoy_params(0.0, pi * 20.0)
    @test_throws ArgumentError validate_odetoy_params(-1.0, pi * 20.0)
    @test_throws ArgumentError validate_odetoy_params(25.0, 0.0)
    @test_throws ArgumentError validate_odetoy_params(25.0, -1.0)
    @test_throws ArgumentError validate_odetoy_params(25.0, pi * 20.0; dt = -1.0)
end
