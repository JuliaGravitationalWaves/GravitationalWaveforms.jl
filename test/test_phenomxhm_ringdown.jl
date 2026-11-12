using Test
using GravitationalWaveforms

# `_xhm_*` ringdown functions are intentionally unexported (matching this package's
# `_xas_*`/`_xhm_qnmfit_*`/inspiral/intermediate `_xhm_*` internal convention), so they're
# accessed qualified.
const W = GravitationalWaveforms

const MODES_NOMIX = ((2, 1), (3, 3), (4, 4))  # (3,2) needs IMRPhenomXHM.jl, see module docstring

# Mirrors IMRPhenomXHM_intermediate.jl's/IMRPhenomXHM_inspiral.jl's own test helper.
function _rd_spin_params(eta, chi1, chi2)
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

@testset "IMRPhenomXHM_ringdown (higher-mode ringdown amplitude/phase ansatze)" begin

    @testset "Raw amplitude fits (rdcp1-3 for 21/33/44): finite across a mass-ratio/spin grid" begin
        for eta in (0.05, 0.1, 0.16, 0.2, 0.2499)
            Seta = sqrt(1.0 - 4.0 * eta)
            for S in (-0.8, -0.2, 0.0, 0.3, 0.8), chidiff in (-0.4, 0.0, 0.5)
                @test isfinite(W._xhm_rd_amp21_rdcp1(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp21_rdcp2(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp21_rdcp3(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp33_rdcp1(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp33_rdcp2(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp33_rdcp3(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp44_rdcp1(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp44_rdcp2(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp44_rdcp3(eta, Seta, S, chidiff))
            end
        end
    end

    @testset "Raw (3,2) amplitude fits: finite across a mass-ratio/spin grid" begin
        for eta in (0.05, 0.1, 0.16, 0.2, 0.2499)
            Seta = sqrt(1.0 - 4.0 * eta)
            for S in (-0.8, -0.2, 0.0, 0.3, 0.8), chidiff in (-0.4, 0.0, 0.5)
                @test isfinite(W._xhm_rd_amp32_alambda(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp32_lambda(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp32_sigma(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp32_rdaux1(eta, Seta, S, chidiff))
                @test isfinite(W._xhm_rd_amp32_rdaux2(eta, Seta, S, chidiff))
            end
        end
    end

    @testset "Raw phase fits (alpha2/alphaL, (3,2)'s spheroidal fits): finite across a grid" begin
        for eta in (0.05, 0.1, 0.16, 0.2, 0.2499)
            for S in (-0.8, -0.2, 0.0, 0.3, 0.8), chi1 in (-0.5, 0.2, 0.7), chi2 in (-0.3, 0.1)
                dchi = chi1 - chi2
                @test isfinite(W._xhm_rd_phase_alpha2(eta, S, dchi))
                @test isfinite(W._xhm_rd_phase_alphaL(eta, S, chi1, chi2))
                @test isfinite(W._xhm_rd_phase32_timeshift(eta, S, dchi))
                @test isfinite(W._xhm_rd_phase32_phaseshift(eta, S, chi1, chi2))
                @test isfinite(W._xhm_rd_phase32_p1(eta, S, dchi, chi1, chi2))
                @test isfinite(W._xhm_rd_phase32_p2(eta, S, dchi))
                @test isfinite(W._xhm_rd_phase32_p3(eta, S, dchi))
                @test isfinite(W._xhm_rd_phase32_p4(eta, S, dchi))
            end
        end
    end

    @testset "Amplitude ansatz: continuous at the falloff boundary, and analytic derivative matches finite differences" begin
        for (m1, m2, chi1, chi2) in CONFIGS
            M = m1 + m2
            eta = m1 * m2 / M^2
            Seta, StotR, chiPNHat, chidiff = _rd_spin_params(eta, chi1, chi2)
            dchi = chi1 - chi2
            a, Erad = W._xas_a_Erad(eta, Seta, dchi, StotR)
            for (l, m) in MODES_NOMIX
                # Equal-mass, equal-spin, odd-m is the known "Ampzero" degeneracy (documented in
                # the inspiral and intermediate modules: the odd-m mode's amplitude is exactly zero by
                # symmetry there). Every `rdcp1-3` fit for (2,1)/(3,3) is proportional to `delta`
                # and/or `chidiff`, both exactly 0 here, so `rdcp1==rdcp2==rdcp3==0` and the
                # `deno`/`c2=sqrt(c0/(rdcp2*fdamp))` formulas hit an intrinsic 0/0 -- not a bug in
                # this port, LALSuite's own full pipeline special-cases this via a separate
                # `Ampzero` flag that bypasses the ansatz entirely. Skipped here for the same
                # reason the inspiral and intermediate tests treat it as a documented degeneracy, not a
                # correctness target for this narrow ansatz-only unit test.
                isodd(m) && iszero(Seta) && chi1 == chi2 && continue

                fring, fdamp = W._xhm_fring_fdamp(l, m, a, Erad)
                c = W._xhm_rd_amp_coefficients(l, m, eta, Seta, StotR, chiPNHat, chidiff, fring, fdamp)
                @test isfinite(c.c0) && isfinite(c.c1) && isfinite(c.c2) && isfinite(c.c3) && isfinite(c.c4)
                @test c.falloff ≈ fring + 2 * fdamp

                # Continuity of the ansatz VALUE across the falloff boundary: by construction, c3
                # was seeded to exactly the Lorentzian branch's own value there, so the ansatz
                # (which takes the falloff/exponential branch at f == c.falloff) must agree with
                # the Lorentzian branch's value at that same frequency.
                v_at_boundary = W._xhm_rd_amp_ansatz(c.falloff, c)  # exactly at the boundary: falloff branch
                v_lorentzian_at_falloff = W._xhm_rd_amp_lorentzian(c.falloff, c.c0, c.c1, c.c2, c.fring, c.fdamp)
                @test isapprox(v_at_boundary, v_lorentzian_at_falloff; rtol = 1e-12)

                # Analytic derivative vs central finite difference, at a handful of frequencies
                # spanning the Lorentzian region (below falloff).
                for f in (fring - 0.5 * fdamp, fring, fring + fdamp, c.falloff - 0.1 * fdamp)
                    h = 1e-6 * max(abs(f), 1.0)
                    dv_analytic = W._xhm_rd_amp_dansatz(f, c)
                    dv_fd = (W._xhm_rd_amp_lorentzian(f + h, c.c0, c.c1, c.c2, c.fring, c.fdamp) -
                             W._xhm_rd_amp_lorentzian(f - h, c.c0, c.c1, c.c2, c.fring, c.fdamp)) / (2h)
                    @test isapprox(dv_analytic, dv_fd; rtol = 1e-6, atol = 1e-10)
                end

                # Ansatz stays finite and positive-ish (it's an amplitude, up to the fabs() already
                # baked into the collocation-point fits) across a wide band including the falloff
                # exponential-decay regime.
                for f in range(fring - 2 * fdamp, fring + 8 * fdamp, length = 25)
                    f <= 0 && continue
                    v = W._xhm_rd_amp_ansatz(f, c)
                    @test isfinite(v)
                end
            end
        end
    end

    @testset "Phase ansatz: alpha0==0, and dphidf/ansatzint/derivansatz are consistent derivative pairs" begin
        for (m1, m2, chi1, chi2) in CONFIGS
            M = m1 + m2
            eta = m1 * m2 / M^2
            Seta, StotR, chiPNHat, chidiff = _rd_spin_params(eta, chi1, chi2)
            dchi = chi1 - chi2
            a, Erad = W._xas_a_Erad(eta, Seta, dchi, StotR)
            for (l, m) in MODES_NOMIX
                fring, fdamp = W._xhm_fring_fdamp(l, m, a, Erad)
                pc = W._xhm_rd_phase_coefficients(l, m, eta, StotR, chi1, chi2, fring)
                @test pc.alpha0 == 0
                @test isfinite(pc.alpha2) && isfinite(pc.alphaL)

                for f in (fring - 0.5 * fdamp, fring, fring + fdamp, fring + 3 * fdamp)
                    h = 1e-6 * max(abs(f), 1.0)
                    # dphidf is the derivative of ansatzint
                    dphi_fd = (W._xhm_rd_phase_ansatzint(f + h, pc, fring, fdamp) -
                               W._xhm_rd_phase_ansatzint(f - h, pc, fring, fdamp)) / (2h)
                    dphi_direct = W._xhm_rd_phase_dphidf(f, pc, fring, fdamp)
                    @test isapprox(dphi_direct, dphi_fd; rtol = 1e-6, atol = 1e-10)

                    # derivansatz is the derivative of dphidf
                    d2phi_fd = (W._xhm_rd_phase_dphidf(f + h, pc, fring, fdamp) -
                                W._xhm_rd_phase_dphidf(f - h, pc, fring, fdamp)) / (2h)
                    d2phi_direct = W._xhm_rd_phase_derivansatz(f, pc, fring, fdamp)
                    @test isapprox(d2phi_direct, d2phi_fd; rtol = 1e-6, atol = 1e-8)
                end
            end
        end
    end

    @testset "wlm rescaling: l==m gives wlm=2, l!=m gives wlm=m/3 (checked via alpha2's known linear scaling)" begin
        # alpha2 = (1/fring^2) * wlm * alpha2fit(eta,StotR,dchi) -- so for two different fring values
        # (holding eta,StotR,dchi,l,m fixed), alpha2*fring^2 should be constant, and it should scale
        # by the wlm ratio between (2,1) [wlm=1/3] and (4,4) [wlm=2].
        eta, chi1, chi2 = 0.2, 0.3, -0.15
        Seta, StotR, chiPNHat, chidiff = _rd_spin_params(eta, chi1, chi2)
        fring_dummy = 0.1
        pc21 = W._xhm_rd_phase_coefficients(2, 1, eta, StotR, chi1, chi2, fring_dummy)
        pc44 = W._xhm_rd_phase_coefficients(4, 4, eta, StotR, chi1, chi2, fring_dummy)
        # wlm(2,1) = 1/3, wlm(4,4) = 2 -- ratio 6, and alpha2fit is identical (same eta,StotR,dchi)
        # since it's mode-independent.
        ratio = (pc44.alpha2) / (pc21.alpha2)
        @test isapprox(ratio, 6.0; rtol = 1e-10)
    end

    @testset "_xhm_rd_amp_coefficients / _xhm_rd_phase_coefficients reject (3,2) and unsupported (l,m)" begin
        @test_throws ArgumentError W._xhm_rd_amp_coefficients(3, 2, 0.2, 0.5, 0.1, 0.1, 0.05, 0.08, 0.02)
        @test_throws ArgumentError W._xhm_rd_amp_coefficients(2, 2, 0.2, 0.5, 0.1, 0.1, 0.05, 0.08, 0.02)
        @test_throws ArgumentError W._xhm_rd_phase_coefficients(3, 2, 0.2, 0.1, 0.3, -0.15, 0.08)
        @test_throws ArgumentError W._xhm_rd_phase_coefficients(2, 2, 0.2, 0.1, 0.3, -0.15, 0.08)
    end

    @testset "Float32/Float64 genericity" begin
        eta64, chi1_64, chi2_64 = 0.18, 0.25, -0.15
        eta32, chi1_32, chi2_32 = Float32(eta64), Float32(chi1_64), Float32(chi2_64)
        Seta64, StotR64, chiPNHat64, chidiff64 = _rd_spin_params(eta64, chi1_64, chi2_64)
        Seta32, StotR32, chiPNHat32, chidiff32 = Float32.(_rd_spin_params(eta32, chi1_32, chi2_32))
        dchi64 = chi1_64 - chi2_64
        dchi32 = chi1_32 - chi2_32
        a64, Erad64 = W._xas_a_Erad(eta64, Seta64, dchi64, StotR64)
        a32, Erad32 = W._xas_a_Erad(eta32, Seta32, dchi32, StotR32)

        for (l, m) in MODES_NOMIX
            fring64, fdamp64 = W._xhm_fring_fdamp(l, m, a64, Erad64)
            fring32, fdamp32 = W._xhm_fring_fdamp(l, m, a32, Erad32)

            c64 = W._xhm_rd_amp_coefficients(l, m, eta64, Seta64, StotR64, chiPNHat64, chidiff64, fring64, fdamp64)
            c32 = W._xhm_rd_amp_coefficients(l, m, eta32, Seta32, StotR32, chiPNHat32, chidiff32, fring32, fdamp32)
            @test c64.c0 isa Float64
            @test c32.c0 isa Float32
            v64 = W._xhm_rd_amp_ansatz(fring64, c64)
            v32 = W._xhm_rd_amp_ansatz(fring32, c32)
            @test v64 isa Float64
            @test v32 isa Float32
            @test isapprox(Float64(v32), v64; rtol = 1e-3)

            pc64 = W._xhm_rd_phase_coefficients(l, m, eta64, StotR64, chi1_64, chi2_64, fring64)
            pc32 = W._xhm_rd_phase_coefficients(l, m, eta32, StotR32, chi1_32, chi2_32, fring32)
            @test pc64.alpha2 isa Float64
            @test pc32.alpha2 isa Float32
            dphi64 = W._xhm_rd_phase_dphidf(fring64, pc64, fring64, fdamp64)
            dphi32 = W._xhm_rd_phase_dphidf(fring32, pc32, fring32, fdamp32)
            @test dphi64 isa Float64
            @test dphi32 isa Float32
            @test isapprox(Float64(dphi32), dphi64; rtol = 1e-3)
        end
    end
end
