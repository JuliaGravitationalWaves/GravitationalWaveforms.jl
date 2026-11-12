# Unit tests for GravitationalWaveforms.TaylorT3. Basic sanity
# (construction, output shape/type, Float32 genericity via @Tlit) plus a handful of
# cross-validated regression values against real `lalsimulation` , baked in as
# near-exact assertions.

@testset "TaylorT3 construction and Phi/Ampl basic sanity" begin
    model = TaylorT3()
    @test model isa TaylorT3

    # m1=30, m2=25 Msun (this ecosystem's baseline). Times span [-0.9, -0.000244] s, matching
    # real lalsimulation's own valid output range for these masses/f_min=20Hz/deltaT=1/4096 --
    # confirmed via cross-validation that
    # GravitationalWaveforms.TaylorT3's `t0_shift` correction (see TaylorT3.jl's docstring) makes the
    # closed-form frequency series stay monotonically increasing across LAL's entire actual
    # output domain, right up to LAL's own last sample (one deltaT before its "t=0" reference
    # point) -- not just a conservative early cutoff.
    mc, eta, dL = 23.821232072402484, 0.24793388429752067, 0.4
    t = collect(range(-0.9, -0.000244140625, length = 50))

    phi = Phi(model, t, mc, eta)
    ampl = Ampl(model, t, mc, eta, dL)

    @test phi isa Vector{Float64}
    @test ampl isa Vector{Float64}
    @test length(phi) == length(t)
    @test length(ampl) == length(t)
    @test all(isfinite, phi)
    @test all(isfinite, ampl)
    @test all(>(0), ampl)

    # GW phase (2x orbital) must increase monotonically toward coalescence (t -> 0^-).
    @test issorted(phi)

    # Instantaneous GW frequency (via finite difference of Phi) must increase monotonically
    # toward coalescence -- a basic chirp sanity check independent of any LAL cross-validation.
    f_gw = diff(phi) ./ diff(t) ./ (2pi)
    @test all(>(0), f_gw)
    @test issorted(f_gw)

    # Scalar time input returns a scalar, not a vector.
    phi_scalar = Phi(model, -0.5, mc, eta)
    ampl_scalar = Ampl(model, -0.5, mc, eta, dL)
    @test phi_scalar isa Float64
    @test ampl_scalar isa Float64

    # Float32 genericity (the entire reason TypeGenericity.jl's @Tlit macro exists): output
    # type must track input type, not silently promote back to Float64.
    t32 = Float32.(t)
    phi32 = Phi(model, t32, Float32(mc), Float32(eta))
    ampl32 = Ampl(model, t32, Float32(mc), Float32(eta), Float32(dL))
    @test eltype(phi32) === Float32
    @test eltype(ampl32) === Float32
    @test all(isfinite, phi32)
    @test all(isfinite, ampl32)

    phi32_scalar = Phi(model, Float32(-0.5), Float32(mc), Float32(eta))
    @test phi32_scalar isa Float32
end

@testset "validate_taylort3_params" begin
    @test validate_taylort3_params(0.24) === nothing
    @test validate_taylort3_params(0.25) === nothing
    @test_throws ArgumentError validate_taylort3_params(0.0)           # eta must be > 0
    @test_throws ArgumentError validate_taylort3_params(0.26)          # eta must be <= 0.25
    @test_throws ArgumentError validate_taylort3_params(0.2; dL = -1.0) # dL must be > 0
    @test validate_taylort3_params(0.2; dL = 1.0) === nothing
end

@testset "TaylorT3 regression vs lalsimulation (cross-validation)" begin
    # Baseline parameters: m1=30, m2=25 Msun (non-spinning), dL=0.4 Gpc, f_min=20 Hz,
    # deltaT=1/4096 s -- this ecosystem's standard baseline (TaylorT3 is non-spinning, so chi1/
    # chi2 are dropped relative to the IMRPhenomD/IMRPhenomXAS baseline).
    mc = 23.821232072402484
    eta = 0.24793388429752067
    dL = 0.4
    f_min = 20.0
    deltaT = 1.0 / 4096.0

    # LAL's own physical constants (G_SI, MSUN_SI, C_SI, PC_SI as of the lalsuite version used
    # for cross-validation), reproduced here (rather than this package's own module-level
    # `GMsun_over_c3`/`GMsun_over_c2_Gpc` defaults) so these regression values are pinned to the
    # exact constants the comparison actually used.
    GMsun_over_c3 = 4.925490947641267e-6
    GMsun_over_c2_Gpc = GMsun_over_c3 * 299792458.0 / (3.0856775814913674e16 * 1e9)

    # Times span the full range LAL's own output covers for this system (t=-0.9 to the last
    # sample at t=-deltaT=-0.000244140625, one sample before LAL's "t=0" reference point).
    t = [-0.9, -0.5, -0.2, -0.1, -0.000244140625]

    phi = Phi(TaylorT3(), t, mc, eta; f_min = f_min, deltaT = deltaT, GMsun_over_c3 = GMsun_over_c3)
    ampl = Ampl(TaylorT3(), t, mc, eta, dL; f_min = f_min, deltaT = deltaT,
        GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc)

    @test phi isa Vector{Float64}
    @test ampl isa Vector{Float64}
    @test all(isfinite, phi)
    @test all(isfinite, ampl)

    # Cross-validated against
    # real `lalsimulation`, TaylorT3 approximant, ampO=0/phaseO=7): the *current* GravitationalWaveforms.jl
    # output at these exact parameters, confirmed correct to amplitude median rel. diff
    # 1.88e-10 and instantaneous-GW-frequency median rel. diff 1.26e-4 (the latter dominated by
    # the finite-difference truncation error intrinsic to extracting frequency from LAL's raw
    # hp/hc, not a phasing disagreement -- see that script's header/trailing comment) against
    # LAL's real output on its own full time grid. Baked in here as an exact regression lock
    # (bit-for-bit `==`), not re-derived from LAL on every test run.
    @test phi == [
        -142.7233356039616, -87.96029365060825, -37.692393904219145,
        -17.419202850675926, 6.144023279865612,
    ]
    @test ampl == [
        4.32602226509379e-22, 4.870386050304185e-22, 5.697381402235874e-22,
        6.216834700204823e-22, 6.896031277113842e-22,
    ]
end
