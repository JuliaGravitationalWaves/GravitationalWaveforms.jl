# The SEOBNRv5 / SEOBNRv5PHM effective-one-body Hamiltonian: the single source from which gen/symbolic/generate.jl
# generates the precessing `_seobv5phm_ham_call`, the aligned-spin `_seobv5_potentials` (this function with the in-plane
# spin components set to zero) and every derivative of both models (src/GravitationalWaveforms/SEOBNRv5_ham_symbolic_generated.jl).
#
# pyseobnr 0.3.7: `Ham_AvgS2precess_simple_cython_PA_AD._call` (SEOBNRv5PHM, orbit-averaged in-plane spins), which for
# spins along L reduces to `Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C.evaluate_H` (SEOBNRv5HM); the aligned-spin lines
# below are that function's, verbatim (flags and the d5 = 0 placeholder included).
#
# Units M = 1; r, L = p_phi in units of M; prst = p_r* = xi p_r (tortoise radial momentum). chix_i, chiy_i, chiz_i: the
# dimensionless spin vectors; chi_i their projections on LN_hat, chi_Li those on L (all equal to chiz_i for aligned
# spins). X_i = m_i / M, delta = X_1 - X_2, nu = X_1 X_2; a6, dSO the calibration parameters.
#
# Format (what the generator evaluates on symbolic values): straight-line assignments, one per line, using + - * / ^,
# log and sqrt; no control flow. Domain guards (sqrt / log of a non-positive argument outside the physical domain)
# are the emitter's choice per model, not written here. Edit this file, then rerun gen/symbolic/generate.jl.
function seobnrv5phm_hamiltonian(r, prst, L, chix1, chiy1, chiz1, chix2, chiy2, chiz2, chi_1, chi_2, chi_L1, chi_L2, nu,
                                 X_1, X_2, delta, a6, dSO)
    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r5 = r * r4

    L2 = L * L
    lr = log(r)

    nu2 = nu * nu
    nu3 = nu2 * nu
    nu4 = nu3 * nu

    prst2 = prst * prst
    prst4 = prst2 * prst2
    prst6 = prst4 * prst2
    prst8 = prst6 * prst2

    # ---- non-spinning potentials: A (Apm, 5PN with the a6 calibration) and D (Dbpm), Pade-resummed
    d5 = 0.0

    Dbpm = r * (6730497718123.02 * nu3 + 22295347200.0 * nu2 * d5 + 133772083200.0 * nu2 * r2 + 1822680546449.21 * nu2 * r + 80059249540278.2 * nu2 + 22295347200.0 * nu * d5 * r - 193226342400.0 * nu * d5 + 2589101062873.81 * nu * r2 + 10611661054566.2 * nu * r - 12049908701745.2 * nu + 5107745331375.71 * r2 - 326837426.241486 * r * (14700.0 * nu + 42911.0) - 39476764256925.6 * r - (-5041721180160.0 * nu2 - 25392914995744.3 * nu - 879923036160.0 * r2 - 283115520.0 * r * (14700.0 * nu + 42911.0) + 104186110149937.0) * lr + 5787938193408.0 * lr^2 + 275059053208689.0) / (55296.0 * nu * (14515200.0 * nu3 - 42636451.6032331 * nu2 - 7680.0 * nu * (315.0 * d5 + 890888.810272497) + 4331361844.61149 * nu + 1002013764.01019) - 967680.0 * r3 * (-138240.0 * nu2 - 2675575.66847905 * nu - 5278341.3229329) - 9216.0 * r2 * (-197773496.793534 * nu2 - 7680.0 * nu * (315.0 * d5 + 405152.309729121) + 2481453539.84635 * nu + 5805304367.87913) + r * (5927865218923.02 * nu3 + 70778880.0 * nu2 * (315.0 * d5 + 2561145.80918574) - 138141470005001.0 * nu2 - 4718592.0 * nu * (40950.0 * d5 + 86207832.4415642) + 450172889755120.0 * nu + 86618264430493.3 * (1.0 - 0.496948781616935 * nu)^2 + 188440788778196.0) + 5787938193408.0 * r * lr^2 + (-1698693120.0 * nu * (11592.0 * nu + 69847.0) + 879923036160.0 * r3 + 283115520.0 * r2 * (14700.0 * nu + 42911.0) + 49152.0 * r * (102574080.0 * nu2 + 409207698.136075 * nu - 2119671837.36038)) * lr)

    Apm = 7680.0 * r4 * (-5416406.59541186 * nu2 + 28.0 * nu * (1920.0 * a6 + 733955.307463037) + 2048.0 * nu * (756.0 * nu + 336.0 * r + 407.0) * lr - 7.0 * r * (-185763.092693281 * nu2 + 938918.400156317 * nu - 245760.0) - 3440640.0) / (241555486248.807 * nu4 + 1120.0 * nu3 * (-17833256.898555 * r2 - 163683964.822551 * r - 1188987459.03162) + 7.0 * nu2 * (-39321600.0 * a6 * (3.0 * r + 59.0) + 745857848.115604 * a6 + 1426660551.8844 * r5 - 3089250703.76879 * r4 - 6178501407.53758 * r3 + 2064783811.32587 * r2 + 122635399361.987 * r + 276057889687.011) + 67645734912.0 * nu2 * lr^2 + 53760.0 * nu * (7680.0 * a6 * (r4 + 2.0 * r3 + 4.0 * r2 + 8.0 * r + 16.0) + 128.0 * r * (-6852.34813868015 * r4 + 4264.6962773603 * r3 + 8529.39255472061 * r2 + 13218.7851094412 * r - 33722.4297811176) + 113485.217444961 * r * (-r4 + 2.0 * r3 + 4.0 * r2 + 8.0 * r + 16.0) + 148.04406601634 * r * (349.0 * r4 + 1926.0 * r3 + 3852.0 * r2 + 7704.0 * r + 36400.0)) + 32768.0 * nu * (-1882456.23663972 * nu2 - 38842241.4769507 * nu + 161280.0 * r5 + 480.0 * r4 * (756.0 * nu + 1079.0) + 960.0 * r3 * (756.0 * nu + 1079.0) + 1920.0 * r2 * (588.0 * nu + 1079.0) + 240.0 * r * (-3024.0 * nu2 - 7466.27061066206 * nu + 17264.0) + 13447680.0) * lr + 13212057600.0 * r5)

    # ---- spin combinations: a_+ = X_1 chi_1 + X_2 chi_2, a_- = X_1 chi_1 - X_2 chi_2 as vectors (squares of the full
    # spins: the orbit average of the in-plane parts), their projections on LN_hat (apz, amz) and on L (lap, lam)
    t1 = chix1 * chix1 + chiy1 * chiy1 + chiz1 * chiz1
    t2 = chix2 * chix2 + chiy2 * chiy2 + chiz2 * chiz2
    chi1chi2 = chix1 * chix2 + chiy1 * chiy2 + chiz1 * chiz2
    ap2 = X_1 * X_1 * t1 + 2.0 * X_1 * X_2 * chi1chi2 + X_2 * X_2 * t2
    am2 = X_1 * X_1 * t1 - 2.0 * X_1 * X_2 * chi1chi2 + X_2 * X_2 * t2
    apam = X_1 * X_1 * t1 - X_2 * X_2 * t2
    apamd = apam * delta
    apz = X_1 * chi_1 + X_2 * chi_2
    amz = X_1 * chi_1 - X_2 * chi_2
    nap2 = 0.5 * (ap2 - apz * apz)          # in-plane parts (zero for aligned spins)
    nam2 = 0.5 * (am2 - amz * amz)
    napnam = 0.5 * (apam - apz * amz)
    lap = X_1 * chi_L1 + X_2 * chi_L2
    lam = X_1 * chi_L1 - X_2 * chi_L2

    # ---- tortoise factor xi = dr*/dr
    xi = sqrt(Dbpm) * r2 * (Apm + ap2 / r2) / (ap2 + r2)

    # ---- Q: spin part ~ p_r^4 = (prst / xi)^4, non-spinning part in prst
    flagNLOSS2 = 1.0
    pr4 = (prst / xi)^4
    QSalign2 = flagNLOSS2 * pr4 * (-0.46875 * am2 * (4.0 * nu2 - 5.0 * nu + 1.0) - 0.15625 * ap2 * (32.0 * nu2 - 33.0 * nu - 5.0) + 0.3125 * apamd * (18.0 * nu - 1.0)) / r3

    flagQPN55 = 1.0
    flagQPN5 = 1.0
    flagQPN4 = 1.0

    Qpm = flagQPN4 * (0.121954868780449 * nu * prst8 / r + prst6 * (6.0 * nu3 - 5.4 * nu2 - 2.78300763695006 * nu) / r2 + prst4 * (10.0 * nu3 - 131.0 * nu2 + 92.7110442849544 * nu) / r3) + flagQPN5 * (prst8 * (-6.0 * nu4 + 3.42857142857143 * nu3 + 3.33842023648322 * nu2 + 1.38977750996128 * nu) / r2 + prst6 * (-14.0 * nu4 + 188.0 * nu3 - 89.5298327361234 * nu2 - 33.9782122170436 * nu) / r3 + prst4 * (602.318540416564 * nu3 + nu2 * (118.4 * lr - 1796.13660498019) + nu * (452.542166996693 - 51.6952380952381 * lr)) / r4) + flagQPN55 * (1.48275342024365 * nu * prst8 / r^2.5 - 11.3175085791863 * nu * prst6 / r^3.5 + 147.443752990146 * nu * prst4 / r^4.5) + prst4 * (-6.0 * nu2 + 8.0 * nu) / r2

    Qq = QSalign2 + Qpm

    # ---- D-bar: Bnp (non-spinning + aligned spin), Bnpa (spin, p_phi^2 part), Bp (in-plane spin, SEOBNRv5PHM)
    Bnpa = -r * (r + 2.0) / (ap2 * r * (r + 2.0) + r4)

    flagNLOSS = 1.0
    BnpSalign2 = flagNLOSS * (0.1875 * am2 * (4.0 * nu - 1.0) + ap2 * (3.0 * nu + 2.8125) - 2.625 * apamd) / r3 + flagNLOSS2 * (0.015625 * am2 * (4.0 * nu2 + 115.0 * nu - 37.0) + 0.015625 * ap2 * (-1171.0 * nu - 861.0) + 0.03125 * apamd * (26.0 * nu + 449.0)) / r4

    Bnp = Apm * Dbpm + BnpSalign2 + ap2 / r2 - 1.0

    BpSprec2 = -nap2 / r2 + (nam2 * (0.1875 - 0.75 * nu) + nap2 * (-1.75 * nu - 0.9375) + napnam * (0.75 - 1.5 * X_2)) / r3 + (-0.125 * delta * napnam * (98.0 * nu + 43.0) + 0.015625 * nam2 * (152.0 * nu2 - 1090.0 * nu + 219.0) + 0.00520833333333333 * nap2 * (264.0 * nu2 - 1610.0 * nu + 375.0)) / r4

    Bp = 1.0 + BpSprec2

    # ---- A
    ASalignCal2 = 0.0
    ASalign2 = flagNLOSS * (0.125 * am2 * (4.0 * nu + 1.0) + 1.125 * ap2 - 1.25 * apamd) / r4 + flagNLOSS2 * (0.046875 * am2 * (28.0 * nu2 - 27.0 * nu - 3.0) - 0.390625 * ap2 * (7.0 * nu + 9.0) - 1.21875 * apamd * (2.0 * nu - 3.0)) / r^5.0

    ASprec2 = 2.0 * nap2 / r3 + (4.125 * delta * napnam + 0.125 * nam2 * (-4.0 * nu - 3.0) + 0.25 * nap2 * (7.0 * nu - 31.0)) / r4 + (0.25 * delta * napnam * (68.0 * nu - 1.0) + 0.015625 * nam2 * (-328.0 * nu2 + 1166.0 * nu - 171.0) + 0.00520833333333333 * nap2 * (-264.0 * nu2 + 2870.0 * nu + 561.0)) / r5

    A = (ASalign2 + ASalignCal2 + ASprec2 + Apm + ap2 / r2) / (ap2 * (1.0 + 2.0 / r) / r2 + 1.0)

    # ---- even part
    Heven = sqrt(A * (Bnpa * L2 * lap^2 / r2 + Bp * L2 / r2 + Qq + prst2 * (Bnp + 1.0) / xi^2 + 1.0))

    # ---- odd (spin-orbit) part
    Ga3 = L * lam * (-0.25 * L2 * delta * nap2 / r3 + (0.0416666666666667 * ap2 * delta + nap2 * (0.416666666666667 - 0.833333333333333 * X_2)) / r2) + L * lap * (L2 * (-0.25 * nap2 + napnam * (0.5 - X_2)) / r3 + (-0.25 * ap2 + 0.208333333333333 * apamd - 1.66666666666667 * delta * napnam - 0.75 * nap2) / r2)

    SOcalib = L * nu * dSO * lap / r3

    flagNLOSO2 = 1.0
    flagNLOSO = 1.0

    gam = flagNLOSO * (L2 * (0.46875 - 0.28125 * nu) / r2 + (0.34375 * nu + 0.09375) / r) + flagNLOSO2 * (L2 * L2 * (0.29296875 * nu2 - 0.3515625 * nu - 0.41015625) / r4 + L2 * (-0.798177083333333 * nu2 - 0.2734375 * nu - 0.23046875) / r3 + (0.536458333333333 * nu2 - 0.03125 * nu + 0.078125) / r2) + 0.25

    gap = flagNLOSO * (L2 * (-1.40625 * nu - 0.46875) / r2 + (0.71875 * nu - 0.09375) / r) + flagNLOSO2 * (L2 * L2 * (1.34765625 * nu2 + 0.5859375 * nu + 0.41015625) / r4 + L2 * (-2.07161458333333 * nu2 - 2.0859375 * nu + 0.23046875) / r3 + (0.567708333333333 * nu2 - 5.53125 * nu - 0.078125) / r2) + 1.75

    Hodd = (Ga3 + L * delta * gam * lam + L * gap * lap + SOcalib) / (ap2 * (r + 2.0) + r3)

    Heff = Heven + Hodd

    H = sqrt(1.0 + 2.0 * nu * (Heff - 1.0)) / nu

    return (H, xi, A, Bnp, Bnpa, Qq, Heven, Hodd, Bp)
end
