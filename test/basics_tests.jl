using Test, GravitationalWaveforms

@testset "basic sanity: construction, Phi/Ampl, parameter validation" begin
    @testset "IMRPhenomD construction and Phi/Ampl basic sanity" begin
        model = IMRPhenomD()
        @test model isa IMRPhenomD

        mc, eta, chi1, chi2, dL = 25.0, 0.24, 0.1, -0.05, 1.0
        f = collect(range(20.0, 1024.0, length = 50))

        phi = Phi(model, f, mc, eta, chi1, chi2)
        ampl = Ampl(model, f, mc, eta, chi1, chi2, dL)

        @test phi isa Vector{Float64}
        @test ampl isa Vector{Float64}
        @test length(phi) == length(f)
        @test length(ampl) == length(f)
        @test all(isfinite, phi)
        @test all(isfinite, ampl)

        # Scalar frequency input returns a scalar, not a vector.
        phi_scalar = Phi(model, 100.0, mc, eta, chi1, chi2)
        @test phi_scalar isa Float64

        # Float32 genericity (the entire reason TypeGenericity.jl's @Tlit macro exists): output
        # type must track input type, not silently promote back to Float64.
        f32 = Float32.(f)
        phi32 = Phi(model, f32, Float32(mc), Float32(eta), Float32(chi1), Float32(chi2))
        ampl32 = Ampl(model, f32, Float32(mc), Float32(eta), Float32(chi1), Float32(chi2), Float32(dL))
        @test eltype(phi32) === Float32
        @test eltype(ampl32) === Float32
    end

    @testset "IMRPhenomXAS construction and Phi/Ampl basic sanity" begin
        model = IMRPhenomXAS()
        @test model isa IMRPhenomXAS

        mc, eta, chi1, chi2, dL = 25.0, 0.24, 0.1, -0.05, 1.0
        f = collect(range(20.0, 1024.0, length = 50))

        phi = Phi(model, f, mc, eta, chi1, chi2)
        ampl = Ampl(model, f, mc, eta, chi1, chi2, dL)

        @test phi isa Vector{Float64}
        @test ampl isa Vector{Float64}
        @test length(phi) == length(f)
        @test length(ampl) == length(f)
        @test all(isfinite, phi)
        @test all(isfinite, ampl)

        phi_scalar = Phi(model, 100.0, mc, eta, chi1, chi2)
        @test phi_scalar isa Float64

        f32 = Float32.(f)
        phi32 = Phi(model, f32, Float32(mc), Float32(eta), Float32(chi1), Float32(chi2))
        ampl32 = Ampl(model, f32, Float32(mc), Float32(eta), Float32(chi1), Float32(chi2), Float32(dL))
        @test eltype(phi32) === Float32
        @test eltype(ampl32) === Float32
    end

    @testset "validate_phenomd_params" begin
        @test validate_phenomd_params(0.24, 0.1, -0.2) === nothing
        @test validate_phenomd_params(0.25, 0.0, 0.0) === nothing
        @test_throws ArgumentError validate_phenomd_params(0.0, 0.0, 0.0)          # eta must be > 0
        @test_throws ArgumentError validate_phenomd_params(0.26, 0.0, 0.0)         # eta must be <= 0.25
        @test_throws ArgumentError validate_phenomd_params(0.2, 1.5, 0.0)          # chi1 out of [-1,1]
        @test_throws ArgumentError validate_phenomd_params(0.2, 0.0, -1.5)         # chi2 out of [-1,1]
        @test_throws ArgumentError validate_phenomd_params(0.2, 0.0, 0.0; dL = -1.0) # dL must be > 0
        @test validate_phenomd_params(0.2, 0.0, 0.0; dL = 1.0) === nothing
    end

    @testset "validate_phenomxas_params" begin
        @test validate_phenomxas_params(0.24, 0.1, -0.2) === nothing
        @test validate_phenomxas_params(0.25, 0.0, 0.0) === nothing
        @test_throws ArgumentError validate_phenomxas_params(0.0, 0.0, 0.0)
        @test_throws ArgumentError validate_phenomxas_params(0.26, 0.0, 0.0)
        @test_throws ArgumentError validate_phenomxas_params(0.2, 1.5, 0.0)
        @test_throws ArgumentError validate_phenomxas_params(0.2, 0.0, -1.5)
        @test_throws ArgumentError validate_phenomxas_params(0.2, 0.0, 0.0; dL = -1.0)
        @test validate_phenomxas_params(0.2, 0.0, 0.0; dL = 1.0) === nothing
    end

    @testset "TaylorF2 construction and Phi/Ampl basic sanity" begin
        model = TaylorF2()
        @test model isa TaylorF2

        mc, eta, chi1, chi2, dL = 1.17, 0.2497, 0.05, -0.02, 0.001
        f = collect(range(20.0, 1500.0, length = 50))

        phi = Phi(model, f, mc, eta, chi1, chi2)
        ampl = Ampl(model, f, mc, eta, chi1, chi2, dL)

        @test phi isa Vector{Float64}
        @test ampl isa Vector{Float64}
        @test length(phi) == length(f)
        @test length(ampl) == length(f)
        @test all(isfinite, phi)
        @test all(isfinite, ampl)
        @test all(>=(0.0), ampl)

        phi_scalar = Phi(model, 100.0, mc, eta, chi1, chi2)
        @test phi_scalar isa Float64

        f32 = Float32.(f)
        phi32 = Phi(model, f32, Float32(mc), Float32(eta), Float32(chi1), Float32(chi2))
        ampl32 = Ampl(model, f32, Float32(mc), Float32(eta), Float32(chi1), Float32(chi2), Float32(dL))
        @test eltype(phi32) === Float32
        @test eltype(ampl32) === Float32

        # Tidal terms change the phase and stay finite; lambda1 = lambda2 = 0 is the default.
        phi_tidal = Phi(model, f, mc, eta, chi1, chi2; lambda1 = 400.0, lambda2 = 600.0)
        @test all(isfinite, phi_tidal)
        @test phi_tidal != phi
    end

    @testset "validate_taylorf2_params" begin
        @test validate_taylorf2_params(0.24, 0.1, -0.2) === nothing
        @test validate_taylorf2_params(0.25, 0.0, 0.0; lambda1 = 400.0, lambda2 = 0.0) === nothing
        @test_throws ArgumentError validate_taylorf2_params(0.0, 0.0, 0.0)
        @test_throws ArgumentError validate_taylorf2_params(0.26, 0.0, 0.0)
        @test_throws ArgumentError validate_taylorf2_params(0.2, 1.5, 0.0)
        @test_throws ArgumentError validate_taylorf2_params(0.2, 0.0, -1.5)
        @test_throws ArgumentError validate_taylorf2_params(0.2, 0.0, 0.0; lambda1 = -1.0)
        @test_throws ArgumentError validate_taylorf2_params(0.2, 0.0, 0.0; lambda2 = -1.0)
        @test_throws ArgumentError validate_taylorf2_params(0.2, 0.0, 0.0; dL = -1.0)
    end

    @testset "SineGaussian hphc_sine_gaussian basic sanity" begin
        model = SineGaussian()
        @test model isa SineGaussian

        t = collect(range(-0.05, 0.05, length = 200))
        hp, hc = hphc_sine_gaussian(model, t, 10.0, 100.0, 1e-21, 0.3, 0.5)
        @test length(hp) == length(t)
        @test length(hc) == length(t)
        @test all(isfinite, hp)
        @test all(isfinite, hc)

        # Scalar t returns a scalar pair.
        hp_s, hc_s = hphc_sine_gaussian(model, 0.0, 10.0, 100.0, 1e-21, 0.3, 0.5)
        @test hp_s isa Float64
        @test hc_s isa Float64

        # eccentricity = 0 (circular): plus/cross are comparable in scale (a coarse sanity
        # check, not an exact peak match -- the two carriers are 90 degrees out of phase, so
        # their discretely-sampled peaks fall at different t and needn't agree to high
        # precision on a modest grid).
        hp0, hc0 = hphc_sine_gaussian(model, t, 10.0, 100.0, 1e-21, 0.3, 0.0)
        @test maximum(abs, hp0) ≈ maximum(abs, hc0) rtol=0.1

        # eccentricity = 1 (linear): cross polarization vanishes.
        hp1, hc1 = hphc_sine_gaussian(model, t, 10.0, 100.0, 1e-21, 0.3, 1.0)
        @test all(iszero, hc1)

        f32 = Float32.(t)
        hp32, hc32 = hphc_sine_gaussian(model, f32, 10.0f0, 100.0f0, 1.0f-21, 0.3f0, 0.5f0)
        @test eltype(hp32) === Float32
        @test eltype(hc32) === Float32
    end
end
