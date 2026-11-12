using Test
using GravitationalWaveforms

# `_xhm_*` intermediate functions are intentionally unexported (matching this package's
# `_xas_*`/`_xhm_qnmfit_*`/inspiral `_xhm_*` internal convention), so they're accessed qualified.
const W = GravitationalWaveforms

const MODES = ((2, 1), (3, 3), (3, 2), (4, 4))
const MODES_NO32 = ((2, 1), (3, 3), (4, 4))  # (3,2) phase needs IMRPhenomXHM.jl, see module docstring

# Small helper mirroring this package's `chi1,chi2 -> STotR/chiPNHat/delta` convention (matching
# IMRPhenomXAS.jl's own inline `StotR = (mm1^2*chi1+mm2^2*chi2)/(mm1^2+mm2^2)` computation, and
# IMRPhenomXHM_inspiral.jl's chiPNHat formula).
function _spin_params(eta, chi1, chi2)
    Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : 0.0
    mm1 = 0.5 * (1.0 + Seta)
    mm2 = 0.5 * (1.0 - Seta)
    chi_eff = mm1 * chi1 + mm2 * chi2
    StotR = (mm1 * mm1 * chi1 + mm2 * mm2 * chi2) / (mm1 * mm1 + mm2 * mm2)
    chiPNHat = (chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)) / (1.0 - (76.0 * eta / 113.0))
    chidiff = 0.5 * (chi1 - chi2)
    return Seta, StotR, chiPNHat, chidiff
end

const CONFIGS = (
    (30.0, 30.0, 0.0, 0.0),
    (30.0, 25.0, 0.3, -0.2),
    (50.0, 10.0, 0.5, 0.1),
    (63.0, 7.0, 0.6, 0.2),
    (63.0, 7.0, -0.4, 0.65),
)

@testset "IMRPhenomXHM_intermediate (higher-mode intermediate amplitude/phase ansatze)" begin

    @testset "Amplitude collocation-point fits (int1-int4): finite across a mass-ratio/spin grid" begin
        for eta in (0.05, 0.1, 0.16, 0.2, 0.2499)
            Seta = sqrt(1.0 - 4.0 * eta)
            for S in (-0.8, -0.2, 0.0, 0.3, 0.8), chidiff in (-0.4, 0.0, 0.5)
                for (l, m) in MODES
                    int1, int2, int3, int4 = W._xhm_inter_amp_collocation_values(l, m, eta, Seta, S, S, chidiff)
                    @test all(isfinite, (int1, int2, int3, int4))
                    @test all(>=(0), (int1, int2, int3, int4))  # each is `abs(...)` by construction
                end
            end
        end
    end

    @testset "Amplitude coefficient solve + ansatz reproduces the boundary constraints (the numerical-stability regression test)" begin
        # This is the core regression test for this file's own documented numerical-stability fix
        # (module docstring: raw-`f` Vandermonde at N=8 is catastrophically ill-conditioned;
        # centered-and-scaled rescaling brings it back to near machine precision). F4/V4/d4 here are
        # synthetic stand-ins for the ringdown boundary -- this test checks
        # internal self-consistency (the solved polynomial reproduces every one of its own fitting
        # constraints), NOT physical correctness of those particular numbers (that's the
        # LALSuite comparison's job).
        for (m1, m2, chi1, chi2) in CONFIGS
            M = m1 + m2
            eta = m1 * m2 / M^2
            Seta, StotR, chiPNHat, chidiff = _spin_params(eta, chi1, chi2)
            for (l, m) in MODES
                insp = W._xhm_insp_amp_coefficients(l, m, eta, chi1, chi2)
                F1 = insp.fcutInsp
                V1 = W._xhm_insp_amp_ansatz(F1, insp)
                d1 = W._xhm_insp_amp_nd_ansatz(F1, insp)
                F4 = F1 * 3.5   # a plausible F4/F1 ratio for this region, see module docstring
                V4 = V1 * 0.5
                d4 = -abs(V1) * 2.0

                c, fmid, invfhalf = W._xhm_inter_amp_coefficients_from_inspiral(l, m, eta, Seta, StotR, chiPNHat, chidiff, insp, F4, V4, d4)
                @test all(isfinite, c) && isfinite(fmid) && isfinite(invfhalf)

                val1 = W._xhm_inter_amp_ansatz(F1, c, fmid, invfhalf)
                val4 = W._xhm_inter_amp_ansatz(F4, c, fmid, invfhalf)
                # Odd-m modes at equal mass/zero spin can make V1 itself ~0 (LALSuite's
                # Ampzero degeneracy); use a loose absolute floor alongside the relative
                # tolerance so that degenerate case doesn't spuriously fail.
                @test isapprox(val1, V1; rtol = 1e-9, atol = 1e-12)
                @test isapprox(val4, V4; rtol = 1e-9, atol = 1e-12)

                # Derivative-at-F4 constraint, checked via central finite difference of the ansatz
                # itself (independent of how d4 was fed into the linear system).
                h = 1e-6 * F4
                fd4 = (W._xhm_inter_amp_ansatz(F4 + h, c, fmid, invfhalf) - W._xhm_inter_amp_ansatz(F4 - h, c, fmid, invfhalf)) / (2h)
                @test isapprox(fd4, d4; rtol = 1e-4, atol = 1e-8)
            end
        end
    end

    @testset "Float32/Float64 genericity (amplitude)" begin
        eta64, chi1_64, chi2_64 = 0.18, 0.25, -0.15
        eta32, chi1_32, chi2_32 = Float32(eta64), Float32(chi1_64), Float32(chi2_64)
        Seta64, StotR64, chiPNHat64, chidiff64 = _spin_params(eta64, chi1_64, chi2_64)
        Seta32, StotR32, chiPNHat32, chidiff32 = Float32.(_spin_params(eta32, chi1_32, chi2_32))
        for (l, m) in MODES
            insp64 = W._xhm_insp_amp_coefficients(l, m, eta64, chi1_64, chi2_64)
            insp32 = W._xhm_insp_amp_coefficients(l, m, eta32, chi1_32, chi2_32)
            F1_64 = insp64.fcutInsp
            F1_32 = insp32.fcutInsp
            F4_64 = F1_64 * 3.5
            F4_32 = F1_32 * 3.5f0
            V4_64 = 0.4
            V4_32 = 0.4f0
            d4_64 = -1.0
            d4_32 = -1.0f0

            c64, fmid64, invfhalf64 = W._xhm_inter_amp_coefficients_from_inspiral(l, m, eta64, Seta64, StotR64, chiPNHat64, chidiff64, insp64, F4_64, V4_64, d4_64)
            c32, fmid32, invfhalf32 = W._xhm_inter_amp_coefficients_from_inspiral(l, m, eta32, Seta32, StotR32, chiPNHat32, chidiff32, insp32, F4_32, V4_32, d4_32)
            @test c64 isa NTuple{8,Float64}
            @test c32 isa NTuple{8,Float32}
            @test fmid64 isa Float64 && fmid32 isa Float32

            v64 = W._xhm_inter_amp_ansatz(0.5 * (F1_64 + F4_64), c64, fmid64, invfhalf64)
            v32 = W._xhm_inter_amp_ansatz(Float32(0.5) * (F1_32 + F4_32), c32, fmid32, invfhalf32)
            @test v64 isa Float64
            @test v32 isa Float32
            @test isapprox(Float64(v32), v64; rtol = 1e-2)
        end
    end

    @testset "_xhm_inter_amp_collocation_values / _xhm_inter_amp_coefficients reject unsupported (l,m)" begin
        @test_throws ArgumentError W._xhm_inter_amp_collocation_values(2, 2, 0.2, 0.5, 0.1, 0.1, 0.05)
        @test_throws ArgumentError W._xhm_inter_amp_coefficients(2, 2, 0.2, 0.5, 0.1, 0.1, 0.05, 0.02, 1.0, -10.0, 0.08, 0.4, -1.0)
    end

    @testset "Phase collocation-point fits (p1-p6): finite across a mass-ratio/spin grid" begin
        for eta in (0.05, 0.1, 0.16, 0.2, 0.2499)
            for S in (-0.8, -0.2, 0.0, 0.3, 0.8), chi1 in (-0.5, 0.2, 0.7), chi2 in (-0.3, 0.1)
                for (l, m) in MODES
                    p = W._xhm_inter_phase_collocation_values(l, m, eta, S, chi1, chi2)
                    @test length(p) == 6
                    @test all(isfinite, p)
                end
            end
        end
    end

    @testset "Phase coefficient solve: finite, and dphidf/ansatzint are consistent derivative/antiderivative pairs" begin
        afinal, Erad = 0.68, 0.045
        for (m1, m2, chi1, chi2) in CONFIGS
            M = m1 + m2
            eta = m1 * m2 / M^2
            Seta, StotR, chiPNHat, chidiff = _spin_params(eta, chi1, chi2)
            for (l, m) in MODES_NO32
                if l == 2 && m == 1 && StotR >= 0.8
                    continue  # documented unported near-extremal-spin (2,1) override, see module docstring
                end
                fring, fdamp = W._xhm_fring_fdamp(l, m, afinal, Erad)
                emm = m == 1 ? 1.0 : m == 2 ? 2.0 : m == 3 ? 3.0 : 4.0
                fMECO22 = W._xas_fMECO(eta, chiPNHat, chi1 - chi2, Seta)
                fMECOlm = fMECO22 * emm * 0.5

                c = W._xhm_inter_phase_coefficients(l, m, eta, StotR, chi1, chi2, fMECOlm, fring, fdamp)
                @test length(c) == 5
                @test all(isfinite, c)

                _, f1, f2, f3, f4, f5, fPhaseMatchIN, fPhaseMatchIM = W._xhm_inter_phase_colloc_freqs(l, m, fMECOlm, fring, fdamp, eta)
                @test fPhaseMatchIM == fring - fdamp
                @test fPhaseMatchIN == fMECOlm

                for ftest in (0.3, 0.5, 0.7) .* (fPhaseMatchIM - fPhaseMatchIN) .+ fPhaseMatchIN
                    dphi = W._xhm_inter_phase_dphidf(ftest, c, fring, fdamp)
                    @test isfinite(dphi)
                    h = 1e-6 * ftest
                    phi_p = W._xhm_inter_phase_ansatzint(ftest + h, c, fring, fdamp)
                    phi_m = W._xhm_inter_phase_ansatzint(ftest - h, c, fring, fdamp)
                    fd = (phi_p - phi_m) / (2h)
                    @test isapprox(fd, dphi; rtol = 1e-6)
                end
            end
        end
    end

    @testset "Phase gluing constants (_xhm_inter_phase_glue): finite and reproduce continuity by construction" begin
        afinal, Erad = 0.68, 0.045
        eta, chi1, chi2 = 0.2, 0.3, -0.1
        Seta, StotR, chiPNHat, chidiff = _spin_params(eta, chi1, chi2)
        for (l, m) in MODES_NO32
            fring, fdamp = W._xhm_fring_fdamp(l, m, afinal, Erad)
            emm = m == 1 ? 1.0 : m == 2 ? 2.0 : m == 3 ? 3.0 : 4.0
            fMECOlm = W._xas_fMECO(eta, chiPNHat, chi1 - chi2, Seta) * emm * 0.5
            c = W._xhm_inter_phase_coefficients(l, m, eta, StotR, chi1, chi2, fMECOlm, fring, fdamp)
            _, _, _, _, _, _, fPhaseMatchIN, fPhaseMatchIM = W._xhm_inter_phase_colloc_freqs(l, m, fMECOlm, fring, fdamp, eta)

            # Synthetic inspiral/ringdown boundary values (see module
            # docstring); just checking finiteness/well-definedness of the glue arithmetic here.
            insp_dphi, insp_phi = 3000.0, 10.0
            dphi0RD, phi0RD = 3200.0, 12.0
            C1INSP, CINSP, C1RD, CRD = W._xhm_inter_phase_glue(c, fring, fdamp, fPhaseMatchIN, insp_dphi, insp_phi, fPhaseMatchIM, dphi0RD, phi0RD)
            @test all(isfinite, (C1INSP, CINSP, C1RD, CRD))

            # By construction, the glued phase's derivative at fPhaseMatchIN equals the inspiral's
            # own derivative there (that's what C1INSP is solved to cancel).
            glued_dphi_at_IN = W._xhm_inter_phase_dphidf(fPhaseMatchIN, c, fring, fdamp) - C1INSP
            @test isapprox(glued_dphi_at_IN, insp_dphi; rtol = 1e-10)
            glued_dphi_at_RD = W._xhm_inter_phase_dphidf(fPhaseMatchIM, c, fring, fdamp) - C1RD
            @test isapprox(glued_dphi_at_RD, dphi0RD; rtol = 1e-10)
        end
    end

    @testset "(3,2) mode: phase reconstruction throws (needs IMRPhenomXHM.jl mode-mixing), amplitude does not" begin
        eta, chi1, chi2 = 0.2, 0.3, -0.1
        Seta, StotR, chiPNHat, chidiff = _spin_params(eta, chi1, chi2)
        @test_throws ArgumentError W._xhm_inter_phase_coefficients(3, 2, eta, StotR, chi1, chi2, 0.02, 0.1, 0.01)
        @test_throws ArgumentError W._xhm_inter_phase_colloc_freqs(3, 2, 0.02, 0.1, 0.01, eta)
        @test_throws ArgumentError W._xhm_inter_phase_cpoints_indices(3, 2, eta, StotR)

        # Amplitude support for (3,2) does NOT need IMRPhenomXHM.jl (the ringdown boundary is caller-supplied
        # regardless of mode, see module docstring) -- this must NOT throw.
        insp = W._xhm_insp_amp_coefficients(3, 2, eta, chi1, chi2)
        F1 = insp.fcutInsp
        V1 = W._xhm_insp_amp_ansatz(F1, insp)
        c, fmid, invfhalf = W._xhm_inter_amp_coefficients_from_inspiral(3, 2, eta, Seta, StotR, chiPNHat, chidiff, insp, F1 * 3.5, V1 * 0.4, -1.0)
        @test all(isfinite, c)
    end

    @testset "(2,1) mode with StotR >= 0.8: high-spin override collocation indices; needs the (2,2) coefficients" begin
        # the override (`_internals.c` 2076-2093) was ported on fix/xhm-fd-steps (c5e05f4): indices (1,2,3,5,6), and
        # the reconstruction needs coeffs22 (passed from `_xhm_mode_setup`); without them it still refuses.
        eta, chi1, chi2 = 0.2, 0.85, 0.85  # equal spins => StotR == chi1 == chi2 regardless of mass ratio
        Seta, StotR, chiPNHat, chidiff = _spin_params(eta, chi1, chi2)
        @test StotR >= 0.8
        @test W._xhm_inter_phase_cpoints_indices(2, 1, eta, StotR) == (1, 2, 3, 5, 6)
        @test_throws ArgumentError W._xhm_inter_phase_coefficients(2, 1, eta, StotR, chi1, chi2, 0.02, 0.08, 0.01)
    end

    @testset "_xhm_inter_phase_cpoints_indices: selection branches return 5 distinct valid indices" begin
        for eta in (0.02, 0.1, 0.25), StotR in (-0.5, 0.0, 0.5)
            for (l, m) in ((2, 1), (3, 3), (4, 4))
                if l == 2 && m == 1 && StotR >= 0.8
                    continue
                end
                idx = W._xhm_inter_phase_cpoints_indices(l, m, eta, StotR)
                @test length(idx) == 5
                @test length(unique(idx)) == 5
                @test all(i -> 1 <= i <= 6, idx)
            end
        end
    end
end
