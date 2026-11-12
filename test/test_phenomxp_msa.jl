# IMRPhenomXP/XPHM MSA precession angles (PrecVersion 223). References from LALSuite 6.2.0
# (`lalsimulation.SimIMRPhenomXPMSAAngles`, PrecVersion 223, f_ref 20 Hz) and scipy (`ellipkinc`, `ellipj`).
# A 160-configuration comparison (spins up to 0.99, m' = 1..4) agreed to 8e-10 rad in alpha / gamma and 1e-13 in
# cos(beta); full XPHM waveforms at 100 GWTC-2.1 posterior samples to a median relative difference of 1.3e-4 (LAL's
# own angle multibanding), the same level as the 102 (NNLO) port.
using Test, GravitationalWaveforms
using GravitationalWaveforms: _xp_msa_setup, _xp_msa_angles, _ellint_F, _jacobi_sn, _xp_precession_setup, _xp_euler_angles

const _MSA_GMSUN = 4.925490947641267e-6        # LAL MSUN_SI * G_SI / C_SI^3
const _MSA_F = [20.0, 35.0, 60.0, 120.0, 250.0]
# ((m1, m2, chi1, chi2, inclination, m'), alpha, gamma, cos beta) from SimIMRPhenomXPMSAAngles
const _MSA_REF = [
    ((36.0, 29.0, (0.3, -0.2, 0.1), (-0.4, 0.3, -0.2), 0.6, 2),
     [1.4220980140252026, 5.4623471729668305, 7.664921953839209, 8.957173145631884, 9.219072767074568],
     [3.2725187809660974, -0.7513000970928347, -2.8969923821181043, -4.108890430760763, -4.317972284129649],
     [0.999935175085184, 0.9961709798742859, 0.9902628644627802, 0.9882825840969524, 0.9969780160185313]),
    ((36.0, 29.0, (0.3, -0.2, 0.1), (-0.4, 0.3, -0.2), 0.6, 3),
     [-1.4207592858541176, 2.7214046490097865, 6.139682736919772, 8.364493991808938, 9.182305267518478],
     [6.111819085009851, 1.9726593638986856, -1.4171201328022764, -3.5613943295671993, -4.304972006122514],
     [0.9953641786374299, 0.9996639164460847, 0.9946274811167345, 0.9883127891890644, 0.9910351209488045]),
    ((50.0, 12.0, (0.55, 0.1, -0.3), (0.05, -0.6, 0.4), 2.1, 2),
     [1.9926018137294612, 4.4809303275546775, 5.707708370407415, 6.274036077922428, 6.1353556507030085],
     [3.0237787556430002, 1.2775718043515396, 0.5380771489978997, 0.22969833804828266, 0.18708719852541167],
     [0.7993113356912174, 0.7552204322351191, 0.7062134198454698, 0.673193437870963, 0.7831720988199188]),
    ((50.0, 12.0, (0.55, 0.1, -0.3), (0.05, -0.6, 0.4), 2.1, 3),
     [-0.8030335013292862, 2.900027590515405, 4.788677743324824, 5.976499322401564, 6.35055639074854],
     [5.211859583036623, 2.3674085457355365, 1.0688099964435451, 0.3725578543593717, 0.19375888489398063],
     [0.8235993423419973, 0.786943358034328, 0.7568311869738708, 0.7168345778187627, 0.71137531459155]),
]

@testset "IMRPhenomXP MSA angles (PrecVersion 223)" begin
    @testset "elliptic functions vs scipy" begin
        # F(phi, k) (GSL modulus convention) = scipy ellipkinc(phi, m = k^2)
        for ((phi, k), ref) in zip(((0.3, 0.2), (1.2, 0.7), (-0.9, 0.95)),
                                   (0.3001770675788562, 1.3370041295571116, -1.0316386650814247))
            @test _ellint_F(phi, k) ≈ ref rtol = 1e-13
        end
        for ((u, m), ref) in zip(((0.4, 0.1), (2.3, 0.6), (-1.7, 0.99), (5.0, 0.3)),
                                 (0.3884672940502442, 0.97494171343052, -0.9372038834473152, -0.9929697113060251))
            @test _jacobi_sn(u, m) ≈ ref rtol = 1e-12
        end
        @test _jacobi_sn(0.7, 0.0) ≈ sin(0.7)
        @test _jacobi_sn(0.7, 1.0) ≈ tanh(0.7)
    end

    @testset "angles vs LAL SimIMRPhenomXPMSAAngles" begin
        for ((m1, m2, c1, c2, incl, mp), la, lg, lc) in _MSA_REF
            M = m1 + m2
            P = _xp_msa_setup(m1 / M, m2 / M, c1..., c2..., cbrt(pi * M * _MSA_GMSUN * 20.0))
            @test P.ok
            A = [_xp_msa_angles(cbrt(pi * M * _MSA_GMSUN * f * 2 / mp), P) for f in _MSA_F]
            al = first.(A); ze = getindex.(A, 2); cb = last.(A)
            # LAL returns alpha = phi_z - alpha_offset, gamma = -(zeta - epsilon_offset): compare differences across
            # frequency (the offsets, which also depend on the frame, drop out); cos(beta) directly
            @test (al .- al[1]) ≈ (la .- la[1]) atol = 1e-8
            @test (ze .- ze[1]) ≈ -(lg .- lg[1]) atol = 1e-8
            @test cb ≈ lc atol = 1e-12
        end
    end

    @testset "phiJ is exact at tiny in-plane spin" begin
        # A guard meant for exactly zero in-plane spin once also replaced phiJ up to J0perp = 1e-6: 3.5% from LAL.
        for v in (Val(102), Val(223)), e in (1e-9, 1e-7, 1e-6)
            s = _xp_precession_setup(36.0, 29.0, e, 0.5e, 0.1, -0.7e, 0.3e, -0.2, 20.0, 0.6, 1.1; prec_version = v)
            @test s.phiJ == atan(s.J0y, s.J0x)
        end
        s0 = _xp_precession_setup(36.0, 29.0, 0.0, 0.0, 0.1, 0.0, 0.0, -0.2, 20.0, 0.6, 1.1; prec_version = Val(223))
        @test s0.phiJ == 0.0
    end

    @testset "version-aware setup" begin
        args = (36.0, 29.0, 0.3, -0.2, 0.1, -0.4, 0.3, -0.2, 20.0, 0.6, 1.1)
        s102 = _xp_precession_setup(args...; prec_version = Val(102))
        s223 = _xp_precession_setup(args...; prec_version = Val(223))
        @test !hasfield(typeof(s102), :msa)
        @test s223.msa_ok
        # the MSA alpha, epsilon equal alpha0, epsilon0 at f_ref (m' = 2): the offsets are the angles at v_ref
        v_ref = cbrt(pi * 65.0 * GravitationalWaveforms.GMsun_over_c3 * 20.0)
        cb, al, ep = _xp_euler_angles(v_ref, s223)
        @test al ≈ s223.alpha0 atol = 1e-12
        @test ep ≈ s223.epsilon0 atol = 1e-12
        # 102 angles unchanged by the refactor
        cb2, al2, ep2 = _xp_euler_angles(v_ref, s102)
        @test cb2 == GravitationalWaveforms._xp_cos_beta(v_ref, s102)
        @test isfinite(s223.final_spin) && 0 < s223.final_spin < 1
        # default model = LAL's default 223
        @test IMRPhenomXPHM() isa IMRPhenomXPHM{223}
        @test IMRPhenomXPHM(102) isa IMRPhenomXPHM{102}
        @test_throws ArgumentError IMRPhenomXPHM(320)
    end
end
