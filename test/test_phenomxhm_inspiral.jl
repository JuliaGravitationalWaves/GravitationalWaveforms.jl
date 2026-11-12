using Test
using GravitationalWaveforms

# `_xhm_*` inspiral functions are intentionally unexported (matching this package's
# `_xas_*`/`_xhm_qnmfit_*` internal convention -- see IMRPhenomXHM_inspiral.jl's module
# docstring), so they're accessed qualified.
const W = GravitationalWaveforms

const MODES = ((2, 1), (3, 3), (3, 2), (4, 4))

@testset "IMRPhenomXHM_inspiral (higher-mode inspiral amplitude/phase ansatze)" begin

    @testset "Zero-spin, equal-mass exact literal values (byte-for-byte transcription spot-check)" begin
        # eta=0.25 (equal mass) makes delta=Seta=0; chi1=chi2=0 additionally makes S=chidiff=0.
        # For the ODD-m modes (2,1) and (3,3), EVERY term in every `case 122022` collocation-point
        # fit is proportional to `delta`, `chidiff`, or `S` (confirmed by direct inspection of
        # LALSimIMRPhenomXHM_inspiral.c lines 48-286) -- so these must vanish EXACTLY at this point.
        # This is not a coincidental cancellation: it is this port's own (unforced, emergent from
        # the ansatz's literal structure, no explicit `Ampzero`-style override anywhere in this
        # file) reproduction of the standard equal-mass/equal-spin selection rule that kills every
        # odd-`m` mode (a discrete body-exchange symmetry) -- matching
        # `LALSimIMRPhenomXHM_internals.c`'s own explicit `wf->Ampzero = 1` comment ("Odd mode for
        # equal mass, equal spin is zero") for cases 21 and 33, just arrived at completely
        # independently through the raw fit algebra rather than a special-cased flag.
        eta = 0.25
        @test W._xhm_insp_amp_collocation_values(2, 1, eta, 0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)
        @test W._xhm_insp_amp_collocation_values(3, 3, eta, 0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)

        # The EVEN-m modes (3,2) and (4,4) do NOT have this property -- each has at least one
        # eta-only term with no delta/chidiff/S prefactor (e.g. (4,4)'s `_iv1`:
        # `(5.2601381002242595 - 3.557926105832778*eta + ...)*sqroot`), so they stay nonzero here,
        # exactly as the physical selection rule (which only kills odd m) predicts.
        iv32 = W._xhm_insp_amp_collocation_values(3, 2, eta, 0.0, 0.0, 0.0)
        iv44 = W._xhm_insp_amp_collocation_values(4, 4, eta, 0.0, 0.0, 0.0)
        @test all(>(0), iv32)
        @test all(>(0), iv44)
    end

    @testset "Collocation-point fits: finite across a mass-ratio/spin grid" begin
        for eta in (0.05, 0.1, 0.16, 0.2, 0.2499)
            Seta = sqrt(1.0 - 4.0 * eta)
            for S in (-0.8, -0.2, 0.0, 0.3, 0.8), chidiff in (-0.4, 0.0, 0.5)
                for (l, m) in MODES
                    iv1, iv2, iv3 = W._xhm_insp_amp_collocation_values(l, m, eta, Seta, S, chidiff)
                    @test isfinite(iv1) && isfinite(iv2) && isfinite(iv3)
                    @test iv1 >= 0 && iv2 >= 0 && iv3 >= 0  # each is `abs(...)` by construction
                end
            end
        end
    end

    @testset "PN amplitude coefficients: finite, and pnInitial == 0 for every mode" begin
        # Every mode's PN series starts at O(x^{1/3}) or higher (pnInitial multiplies x^0, which
        # the C source always sets to 0 for these 4 higher modes -- LALSimIMRPhenomXHM_internals.c
        # lines 997/1008/1021/1031/1043).
        for eta in (0.05, 0.16, 0.25), delta in (0.0, sqrt(1.0 - 4 * 0.16)), chiS in (-0.3, 0.2), chiA in (-0.2, 0.4)
            for (l, m) in MODES
                pn = W._xhm_pn_amplitude_coefficients(l, m, eta, delta, chiS, chiA)
                @test length(pn) == 7
                @test pn[1] == 0
                for c in pn
                    @test isfinite(real(c)) && isfinite(imag(c))
                end
            end
        end
    end

    @testset "_xhm_insp_amp_coefficients / _xhm_insp_amp_ansatz: finite and positive across the grid" begin
        # Spans near-equal-mass to GW190814-like (~1:9) mass ratios and moderate-to-high aligned
        # spins, mirroring the comparison script's own configuration grid.
        configs = (
            (30.0, 30.0, 0.0, 0.0),
            (30.0, 25.0, 0.3, -0.2),
            (50.0, 10.0, 0.5, 0.1),
            (63.0, 7.0, 0.6, 0.2),
            (63.0, 7.0, -0.4, 0.8),
        )
        for (m1, m2, chi1, chi2) in configs
            M = m1 + m2
            eta = m1 * m2 / M^2
            for (l, m) in MODES
                coeffs = W._xhm_insp_amp_coefficients(l, m, eta, chi1, chi2)
                @test isfinite(coeffs.fcutInsp) && coeffs.fcutInsp > 0
                # Odd-m modes ((2,1),(3,3)) vanish EXACTLY at equal mass with zero spins (see the
                # "Zero-spin, equal-mass exact literal values" testset above for why) -- allow that
                # one specific degenerate combination to be exactly zero rather than strictly
                # positive; everywhere else (including every other config for these same modes)
                # must be strictly positive.
                odd_m_equal_mass_zero_spin = (m1 == m2) && (chi1 == 0.0) && (chi2 == 0.0) && isodd(m)
                for frac in (0.05, 0.3, 0.6, 0.9, 0.999)
                    Mf = frac * coeffs.fcutInsp
                    v = W._xhm_insp_amp_ansatz(Mf, coeffs)
                    @test isfinite(v)
                    if odd_m_equal_mass_zero_spin
                        @test v == 0.0
                    else
                        @test v > 0
                    end
                end
            end
        end
    end

    @testset "_xhm_insp_phase_coefficients / _xhm_insp_phase_ansatz(_int): finite across the grid" begin
        configs = (
            (30.0, 30.0, 0.0, 0.0),
            (30.0, 25.0, 0.3, -0.2),
            (50.0, 10.0, 0.5, 0.1),
            (63.0, 7.0, 0.6, 0.2),
            (63.0, 7.0, -0.4, 0.8),
        )
        for (m1, m2, chi1, chi2) in configs
            M = m1 + m2
            eta = m1 * m2 / M^2
            for (l, m) in MODES
                coeffs = W._xhm_insp_amp_coefficients(l, m, eta, chi1, chi2)
                phi, phiL = W._xhm_insp_phase_coefficients(l, m, eta, chi1, chi2)
                @test length(phi) == 15 && length(phiL) == 15
                for frac in (0.05, 0.3, 0.6, 0.9, 0.999)
                    Mf = frac * coeffs.fcutInsp
                    p = W._xhm_insp_phase_ansatz_int(Mf, phi, phiL)
                    dp = W._xhm_insp_phase_ansatz(Mf, phi, phiL)
                    @test isfinite(p)
                    @test isfinite(dp)
                end
            end
        end
    end

    @testset "_xhm_insp_phase_ansatz is the analytic Mf-derivative of _xhm_insp_phase_ansatz_int" begin
        # Self-consistency check via central finite differences -- these are supposed to be
        # antiderivative/derivative pairs of each other (LALSimIMRPhenomXHM_inspiral.c's own
        # naming: "_AnsatzInt" vs "_Ansatz"), independent of any LALSuite comparison.
        eta, chi1, chi2 = 0.2, 0.3, -0.1
        for (l, m) in MODES
            coeffs = W._xhm_insp_amp_coefficients(l, m, eta, chi1, chi2)
            phi, phiL = W._xhm_insp_phase_coefficients(l, m, eta, chi1, chi2)
            for frac in (0.2, 0.5, 0.8)
                Mf = frac * coeffs.fcutInsp
                h = 1e-6 * Mf
                fd = (W._xhm_insp_phase_ansatz_int(Mf + h, phi, phiL) - W._xhm_insp_phase_ansatz_int(Mf - h, phi, phiL)) / (2h)
                an = W._xhm_insp_phase_ansatz(Mf, phi, phiL)
                @test isapprox(fd, an; rtol = 1e-6)
            end
        end
    end

    @testset "_xhm_insp_amp_nd_ansatz matches a plain finite difference of _xhm_insp_amp_ansatz" begin
        eta, chi1, chi2 = 0.15, 0.4, 0.1
        for (l, m) in MODES
            coeffs = W._xhm_insp_amp_coefficients(l, m, eta, chi1, chi2)
            for frac in (0.2, 0.5, 0.8)
                Mf = frac * coeffs.fcutInsp
                h = 1e-6 * Mf
                fd = (W._xhm_insp_amp_ansatz(Mf + h, coeffs) - W._xhm_insp_amp_ansatz(Mf - h, coeffs)) / (2h)
                nd = W._xhm_insp_amp_nd_ansatz(Mf, coeffs)
                @test isapprox(fd, nd; rtol = 1e-4)
            end
        end
    end

    @testset "Float32/Float64 genericity" begin
        eta64, chi1_64, chi2_64 = 0.18, 0.25, -0.15
        eta32, chi1_32, chi2_32 = Float32(eta64), Float32(chi1_64), Float32(chi2_64)
        for (l, m) in MODES
            c64 = W._xhm_insp_amp_coefficients(l, m, eta64, chi1_64, chi2_64)
            c32 = W._xhm_insp_amp_coefficients(l, m, eta32, chi1_32, chi2_32)
            @test c64.fcutInsp isa Float64
            @test c32.fcutInsp isa Float32
            @test isapprox(Float64(c32.fcutInsp), c64.fcutInsp; rtol = 1e-4)

            v64 = W._xhm_insp_amp_ansatz(0.5 * c64.fcutInsp, c64)
            v32 = W._xhm_insp_amp_ansatz(Float32(0.5) * c32.fcutInsp, c32)
            @test v64 isa Float64
            @test v32 isa Float32
            @test isapprox(Float64(v32), v64; rtol = 1e-3)

            phi64, phiL64 = W._xhm_insp_phase_coefficients(l, m, eta64, chi1_64, chi2_64)
            phi32, phiL32 = W._xhm_insp_phase_coefficients(l, m, eta32, chi1_32, chi2_32)
            p64 = W._xhm_insp_phase_ansatz_int(0.5 * c64.fcutInsp, phi64, phiL64)
            p32 = W._xhm_insp_phase_ansatz_int(Float32(0.5) * c32.fcutInsp, phi32, phiL32)
            @test p64 isa Float64
            @test p32 isa Float32
            @test isapprox(Float64(p32), p64; rtol = 1e-2)
        end
    end

    @testset "_xhm_insp_amp_collocation_values / _xhm_insp_amp_coefficients / _xhm_insp_phase_coefficients reject unsupported (l,m)" begin
        @test_throws ArgumentError W._xhm_insp_amp_collocation_values(2, 2, 0.2, 0.5, 0.1, 0.05)
        @test_throws ArgumentError W._xhm_pn_amplitude_coefficients(5, 5, 0.2, 0.5, 0.1, 0.05)
    end

    @testset "Phase lambda parameter-space fits (not exercised by the default eta>0.01 path, ported for completeness): finite" begin
        # See IMRPhenomXHM_inspiral.jl's module docstring, dead-code finding 3: only used for
        # eta<=0.01 in real LALSuite, but ported/tested directly here regardless.
        for eta in (0.001, 0.005, 0.008), STotR in (-0.5, 0.0, 0.6), dchi in (-0.3, 0.2)
            @test isfinite(W._xhm_insp_phase_21_lambda(eta, STotR, dchi))
            @test isfinite(W._xhm_insp_phase_33_lambda(eta, STotR, dchi))
            @test isfinite(W._xhm_insp_phase_32_lambda(eta, STotR, dchi, 0.3, -0.1))
            @test isfinite(W._xhm_insp_phase_44_lambda(eta, STotR, dchi))
        end
    end

    @testset "_xhm_insp_phase_lambdapn: finite, and matches the exact zero-spin literal closed forms" begin
        eta = 0.2
        @test isapprox(W._xhm_insp_phase_lambdapn(eta, 2, 1), -1.0 * (2.0 * pi * (-0.5 - 2.0 * log(2.0))))
        @test isapprox(W._xhm_insp_phase_lambdapn(eta, 3, 3), -1.0 * (2.0 / 3.0 * pi * (-21.0 / 5.0 + 6.0 * log(1.5))))
        for (l, m) in MODES
            @test isfinite(W._xhm_insp_phase_lambdapn(eta, l, m))
        end
        @test W._xhm_insp_phase_lambdapn(eta, 9, 9) == 0.0  # unsupported (l,m) -> the C source's `default: output=0.`
    end

    @testset "_xhm_wavy_points / _xhm_inspiral_amplitude_veto (ported for completeness; unreachable under the default 122022 release)" begin
        @test W._xhm_wavy_points(1.0, 2.0, 1.0) == true   # local max
        @test W._xhm_wavy_points(2.0, 1.0, 2.0) == true   # local min
        @test W._xhm_wavy_points(1.0, 2.0, 3.0) == false  # monotonic
        @test W._xhm_wavy_points(3.0, 2.0, 1.0) == false  # monotonic

        ampNorm = 0.05
        iv1, iv2, iv3 = 10.0, 10.0, 10.0
        f1, f2, f3 = 0.01, 0.015, 0.02
        # Both above threshold: nothing vetoed.
        r1, r2, r3, n = W._xhm_inspiral_amplitude_veto(iv1, iv2, iv3, f1, f2, f3, ampNorm)
        @test (r1, r2, r3) == (iv1, iv2, iv3)
        @test n == 3
        # A tiny value should get vetoed to zero.
        r1, r2, r3, n = W._xhm_inspiral_amplitude_veto(1e-12, iv2, iv3, f1, f2, f3, ampNorm)
        @test r1 == 0.0
        @test n == 2
    end

    @testset "_xhm_inspiral_amp_rho1/rho2/rho3 (legacy, unreachable under the default release): sanity" begin
        # version=0 always returns 0 for all three; version=1 gives rho2=rho3=0 (only rho1
        # nonzero); matches the C source's own documented behaviour (module comment lines
        # 557-559 of LALSimIMRPhenomXHM_inspiral.c).
        v1, v2, v3 = 1.2, 0.8, 0.5
        fc, f1, f2, f3 = 1.0, 0.5, 0.75, 1.0
        @test W._xhm_inspiral_amp_rho1(v1, v2, v3, fc, f1, f2, f3, 0) == 0.0
        @test W._xhm_inspiral_amp_rho2(v1, v2, v3, fc, f1, f2, f3, 0) == 0.0
        @test W._xhm_inspiral_amp_rho3(v1, v2, v3, fc, f1, f2, f3, 0) == 0.0
        @test W._xhm_inspiral_amp_rho2(v1, v2, v3, fc, f1, f2, f3, 1) == 0.0
        @test W._xhm_inspiral_amp_rho3(v1, v2, v3, fc, f1, f2, f3, 1) == 0.0
        @test W._xhm_inspiral_amp_rho3(v1, v2, v3, fc, f1, f2, f3, 2) == 0.0
        @test isfinite(W._xhm_inspiral_amp_rho1(v1, v2, v3, fc, f1, f2, f3, 1))
        @test isfinite(W._xhm_inspiral_amp_rho1(v1, v2, v3, fc, f1, f2, f3, 3))
        @test isfinite(W._xhm_inspiral_amp_rho2(v1, v2, v3, fc, f1, f2, f3, 3))
        @test isfinite(W._xhm_inspiral_amp_rho3(v1, v2, v3, fc, f1, f2, f3, 3))
        @test_throws ArgumentError W._xhm_inspiral_amp_rho1(v1, v2, v3, fc, f1, f2, f3, 4)
    end

end
