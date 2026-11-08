using Test
using GravitationalWaveforms

# `_xhm_qnmfit_f{ring,damp}{21,33,32,44}`/`_xhm_fring_fdamp` are intentionally unexported
# (matching this package's `_xas_fRD_fdamp` internal convention -- see IMRPhenomXHM_qnm.jl's
# module docstring), so they're accessed qualified.
const _fring21 = GravitationalWaveforms._xhm_qnmfit_fring21
const _fdamp21 = GravitationalWaveforms._xhm_qnmfit_fdamp21
const _fring33 = GravitationalWaveforms._xhm_qnmfit_fring33
const _fdamp33 = GravitationalWaveforms._xhm_qnmfit_fdamp33
const _fring32 = GravitationalWaveforms._xhm_qnmfit_fring32
const _fdamp32 = GravitationalWaveforms._xhm_qnmfit_fdamp32
const _fring44 = GravitationalWaveforms._xhm_qnmfit_fring44
const _fdamp44 = GravitationalWaveforms._xhm_qnmfit_fdamp44
const _fring_fdamp = GravitationalWaveforms._xhm_fring_fdamp

const _ALL_RAW_FITS = (
    ("fring21", _fring21), ("fdamp21", _fdamp21),
    ("fring33", _fring33), ("fdamp33", _fdamp33),
    ("fring32", _fring32), ("fdamp32", _fdamp32),
    ("fring44", _fring44), ("fdamp44", _fdamp44),
)

@testset "IMRPhenomXHM_qnm (higher-mode QNM fits)" begin

    @testset "Zero-spin exact literal values (byte-for-byte transcription check)" begin
        # Each fit's a=0 value reduces to exactly its numerator's leading coefficient (denominator
        # is 1.0 at a=0 for every one of these fits) -- spot-checks the literal constants directly
        # against LALSimIMRPhenomXHM_qnm.c's source lines 30-167.
        @test _fring21(0.0) == 0.059471695665734674
        @test _fdamp21(0.0) == 2.0696914454467294 / 146.1779212636481
        @test _fring33(0.0) == 0.09540436245212061
        @test _fdamp33(0.0) == 0.014754148319335946
        @test _fring32(0.0) == 0.09540436245212061
        @test _fdamp32(0.0) == 0.014754148319335946
        @test _fring44(0.0) == 0.1287821193485683
        @test _fdamp44(0.0) == 0.014986847152355699
    end

    @testset "l=3 zero-spin degeneracy: fring33(0)==fring32(0), fdamp33(0)==fdamp32(0)" begin
        # Schwarzschild (zero-spin) QNM frequencies depend only on l, not m -- see
        # IMRPhenomXHM_qnm.jl's module docstring for the full explanation. This is an expected
        # physical fact, not a coincidence, and holding exactly (not just approximately) is a
        # useful cross-check that all four l=3 constants were transcribed correctly.
        @test _fring33(0.0) == _fring32(0.0)
        @test _fdamp33(0.0) == _fdamp32(0.0)
    end

    @testset "Finite and positive across the full physical spin domain" begin
        for a in range(-0.9999, 0.9999, length = 81)
            for (name, f) in _ALL_RAW_FITS
                val = f(a)
                @test isfinite(val)
                @test val > 0
            end
        end
    end

    @testset "Float32/Float64 genericity" begin
        for a64 in (-0.7, -0.2, 0.0, 0.3, 0.9)
            a32 = Float32(a64)
            for (name, f) in _ALL_RAW_FITS
                v64 = f(a64)
                v32 = f(a32)
                @test v64 isa Float64
                @test v32 isa Float32
                @test isapprox(Float64(v32), v64; rtol = 1e-4)
            end
        end
    end

    @testset "_xhm_fring_fdamp wrapper: division by (1-Erad), matches raw fit / Mfinal" begin
        a, Erad = 0.68, 0.045
        Mfinal = 1.0 - Erad
        for (l, m, fring_raw, fdamp_raw) in (
            (2, 1, _fring21, _fdamp21),
            (3, 3, _fring33, _fdamp33),
            (3, 2, _fring32, _fdamp32),
            (4, 4, _fring44, _fdamp44),
        )
            fr, fd = _fring_fdamp(l, m, a, Erad)
            @test fr == fring_raw(a) / Mfinal
            @test fd == fdamp_raw(a) / Mfinal
        end

        # Float32 genericity through the wrapper too.
        fr32, fd32 = _fring_fdamp(2, 1, Float32(a), Float32(Erad))
        @test fr32 isa Float32
        @test fd32 isa Float32
    end

    @testset "_xhm_fring_fdamp rejects unsupported (l,m) (e.g. the dominant (2,2) mode, out of scope here)" begin
        @test_throws ArgumentError _fring_fdamp(2, 2, 0.5, 0.05)
        @test_throws ArgumentError _fring_fdamp(5, 5, 0.5, 0.05)
    end

    @testset "Consistency across a mass-ratio/aligned-spin grid via _xas_a_Erad (physically relevant range)" begin
        # Spans near-equal-mass to GW190814-like (~1:9) mass ratios and moderate-to-high aligned
        # spins, reusing IMRPhenomXAS's own final-spin/radiated-energy fit exactly as
        # LALSimIMRPhenomXHM_internals.c does (it reuses the (2,2)-mode wf22->afinal/Mfinal for
        # every higher mode rather than refitting per (l,m); see IMRPhenomXHM_qnm.jl's docstring).
        for (m1, m2, chi1, chi2) in (
            (30.0, 30.0, 0.0, 0.0),
            (30.0, 25.0, 0.3, -0.2),
            (50.0, 10.0, 0.5, 0.1),
            (63.0, 7.0, 0.6, 0.2),      # ~1:9 mass ratio, GW190814-like
            (63.0, 7.0, -0.4, 0.8),
        )
            M = m1 + m2
            eta = m1 * m2 / M^2
            Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : 0.0
            mm1 = 0.5 * (1.0 + Seta)
            mm2 = 0.5 * (1.0 - Seta)
            StotR = (mm1^2 * chi1 + mm2^2 * chi2) / (mm1^2 + mm2^2)
            dchi = chi1 - chi2

            a, Erad = GravitationalWaveforms._xas_a_Erad(eta, Seta, dchi, StotR)
            @test abs(a) < 1.0
            @test 0.0 < Erad < 1.0

            for (l, m) in ((2, 1), (3, 3), (3, 2), (4, 4))
                fr, fd = _fring_fdamp(l, m, a, Erad)
                @test isfinite(fr) && fr > 0
                @test isfinite(fd) && fd > 0
            end
        end
    end

end
