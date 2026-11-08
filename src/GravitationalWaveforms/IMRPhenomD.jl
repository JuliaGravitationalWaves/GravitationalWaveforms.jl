export IMRPhenomD, Phi, Ampl, validate_phenomd_params

# Solar-mass time and length scales, LAL 7.7.0 (`lal/lib/std/LALConstants.h`, read from
# `lal.MTSUN_SI` / `lal.MRSUN_SI / (1e9 * lal.PC_SI)`): G Msun / c^3 in seconds, G Msun / c^2 in Gpc.
const GMsun_over_c3 = 4.925490947641267e-6
const GMsun_over_c2_Gpc = 4.7854158415878415e-23

struct IMRPhenomD end

struct TF2coeffsStructure{T}
    zero::T
    one::T
    two::T
    three::T
    four::T
    five::T
    six::T
    six_log::T
    five_log::T
    seven::T
    seven_thirds::T
    eight_thirds::T
end

struct PhiInspcoeffsStructure{T}
    initial_phasing::T
    two_thirds::T
    third::T
    third_log::T
    log::T
    min_third::T
    min_two_thirds::T
    min_one::T
    min_four_thirds::T
    min_five_thirds::T
    one::T
    four_thirds::T
    five_thirds::T
    two::T
end

# `@Tlit` (and its helper functions `_tlit_rewrite`/`_tlit_cast`/`_DOT_MACRO_NAMES`) used to be
# defined here; it now lives in TypeGenericity.jl (included by Waveforms.jl before this file,
# into the same module scope) since IMRPhenomXAS.jl needs it too. See that file's docstring for
# the full explanation, including the `@.`-interaction subtlety.

"""
    validate_phenomd_params(eta, chi1, chi2; dL = nothing)

Host-side sanity check for IMRPhenomD parameter domains: `0 < eta <= 0.25`, `-1 <= chi1,chi2 <= 1`,
and (if `dL` is given) `dL > 0`. Throws `ArgumentError` on violation, returns `nothing`
otherwise.

This is intentionally **not** called by `Phi`/`Ampl` themselves. Those are the hot path —
invoked once per walker per NUTS step, ultimately from inside a KernelAbstractions GPU kernel
body and under Enzyme autodiff — and a host-only `@assert` with string interpolation is not
GPU-kernel-compatible, nor is it useful to re-validate the same bounds on every single
evaluation when they should instead be enforced once, where a proposal/walker state is
constructed (sampler or prior level). Call `validate_phenomd_params` explicitly wherever that
one-time host-side check is wanted (tests, sampler setup, interactive debugging).
"""
function validate_phenomd_params(eta, chi1, chi2; dL = nothing)
    0.0 < eta <= 0.25 || throw(ArgumentError("Invalid eta = $eta: must be in (0, 0.25]"))
    -1.0 <= chi1 <= 1.0 || throw(ArgumentError("Invalid chi1 = $chi1: spin must be in [-1, 1]"))
    -1.0 <= chi2 <= 1.0 || throw(ArgumentError("Invalid chi2 = $chi2: spin must be in [-1, 1]"))
    if dL !== nothing
        dL > 0.0 || throw(ArgumentError("Invalid dL = $dL: luminosity distance must be positive (and must be in Gpc, see Ampl's docstring)"))
    end
    return nothing
end

@inline function _finalspin(model::IMRPhenomD, eta, chi1, chi2)
    T = promote_type(typeof(eta), typeof(chi1), typeof(chi2))
    @Tlit T begin
        Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s = m1 * m1 * chi1 + m2 * m2 * chi2
        af1 = eta * (3.4641016151377544 - 4.399247300629289 * eta + 9.397292189321194 * eta^2 - 13.180949901606242 * eta^3)
        af2 = eta * (s * ((1.0 / eta - 0.0850917821418767 - 5.837029316602263 * eta) + (0.1014665242971878 - 2.0967746996832157 * eta) * s))
        af3 = eta * (s * ((-1.3546806617824356 + 4.108962025369336 * eta) * s^2 + (-0.8676969352555539 + 2.064046835273906 * eta) * s^3))
        return af1 + af2 + af3
    end
end

@inline function _radiatednrg(model::IMRPhenomD, eta, chi1, chi2)
    T = promote_type(typeof(eta), typeof(chi1), typeof(chi2))
    @Tlit T begin
        Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
        m1 = 0.5 * (1.0 + Seta)
        m2 = 0.5 * (1.0 - Seta)
        s = (m1 * m1 * chi1 + m2 * m2 * chi2) / (m1 * m1 + m2 * m2)
        EradNS = eta * (0.055974469826360077 + 0.5809510763115132 * eta - 0.9606726679372312 * eta^2 + 3.352411249771192 * eta^3)
        return (EradNS * (1.0 + (-0.0030302335878845507 - 2.0066110851351073 * eta + 7.7050567802399215 * eta^2) * s)) /
               (1.0 + (-0.6714403054720589 - 1.4756929437702908 * eta + 7.304676214885011 * eta^2) * s)
    end
end

"""
    _phi_coeffs(model::IMRPhenomD, mc, eta, chi1, chi2; kwargs...)

Frequency-independent part of the IMRPhenomD phase calculation, factored out of `Phi` so it can be
computed once per waveform (once per walker) rather than once per frequency bin. Returns a
`NamedTuple` holding everything [`_phi_scalar`](@ref) needs to evaluate the phase at a single
frequency; none of these fields depend on frequency.

This split exists for two reasons:
1. It is the architecturally correct design for a per-GPU-thread kernel: recomputing this whole
   block of PN/phenomenological-coefficient arithmetic for every frequency bin (instead of once
   per walker) would be a severe, avoidable performance regression on real waveforms with
   thousands of frequency bins.
2. Empirically, it is *required* for `Enzyme.autodiff` to complete on a KernelAbstractions
   kernel that evaluates the phase: a `@kernel` body calling the old vector-in/vector-out `Phi`
   (which internally `@.`-broadcasts and allocates a fresh output array per call) caused
   `Enzyme.Reverse` compilation to hang indefinitely (still running after 5 minutes, confirmed
   twice, at a tiny problem size: `ntime=16`, `nwalkers=2`). The same kernel body calling
   `_phi_scalar` directly in a non-allocating per-frequency loop differentiates successfully in
   a few minutes -- consistent with this codebase's other documented Enzyme-vs-heap-allocation
   findings (see `@Tlit`'s docstring in `TypeGenericity.jl`).
"""
@inline function _phi_coeffs(model::IMRPhenomD,
    mc,
    eta,
    chi1,
    chi2;
    fInsJoin_PHI = 0.018,
    fcutPar = 0.2,
    t0 = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
    phiRef = zero(promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2))),
    GMsun_over_c3 = GMsun_over_c3,
)
    # No @assert here: this is the hot path (per-walker GPU kernel call / Enzyme target).
    # Parameter-domain checks belong once, at the sampler/prior level, not on every waveform
    # evaluation — call `validate_phenomd_params` explicitly for host-side sanity checks.
    T = promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2), typeof(t0), typeof(phiRef))

    # Keyword-derived scalars aren't reached by the `@Tlit` rewrite below (it only rewrites
    # literal numbers written directly in this file's source, not values flowing in through
    # arguments) — cast them explicitly so a Float32 call that leaves them at their Float64
    # defaults doesn't silently promote the whole computation back to Float64.
    fInsJoin_PHI = T(fInsJoin_PHI)
    fcutPar = T(fcutPar)
    GMsun_over_c3 = T(GMsun_over_c3)
    t0 = T(t0)
    phiRef = T(phiRef)

@Tlit T begin
    fInsJoin = fInsJoin_PHI

    M = mc / (eta^(0.6))
    eta2 = eta * eta
    etaInv = one(eta) / eta

    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
    m1ByM = 0.5 * (1.0 + Seta)
    m2ByM = 0.5 * (1.0 - Seta)
    chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
    xi = -1.0 + chiPN
    xi2 = xi * xi
    aeff = _finalspin(model, eta, chi1, chi2)
    Erad = _radiatednrg(model, eta, chi1, chi2)

    fring = ((0.05947169566573468 - 0.14989771215394762 * aeff + 0.09535606290986028 * aeff * aeff + 0.02260924869042963 * aeff * aeff * aeff - 0.02501704155363241 * aeff * aeff * aeff * aeff - 0.005852438240997211 * (aeff^5) + 0.0027489038393367993 * (aeff^6) + 0.0005821983163192694 * (aeff^7)) / (1 - 2.8570126619966296 * aeff + 2.373335413978394 * aeff * aeff - 0.6036964688511505 * aeff * aeff * aeff * aeff + 0.0873798215084077 * (aeff^6))) / (1.0 - Erad)
    fdamp = ((0.014158792290965177 - 0.036989395871554566 * aeff + 0.026822526296575368 * aeff * aeff + 0.0008490933750566702 * aeff * aeff * aeff - 0.004843996907020524 * aeff * aeff * aeff * aeff - 0.00014745235759327472 * (aeff^5) + 0.0001504546201236794 * (aeff^6)) / (1 - 2.5900842798681376 * aeff + 1.8952576220623967 * aeff * aeff - 0.31416610693042507 * aeff * aeff * aeff * aeff + 0.009002719412407133 * (aeff^6))) / (1.0 - Erad)

    sigma1 =
        2096.551999295543 +
        1463.7493168261553 * eta +
        (
            1312.5493286098522 + 18307.330017082117 * eta - 43534.1440746107 * eta2 +
            (-833.2889543511114 + 32047.31997183187 * eta - 108609.45037520859 * eta2) *
            xi +
            (452.25136398112204 + 8353.439546391714 * eta - 44531.3250037322 * eta2) * xi2
        ) * xi
    sigma2 =
        -10114.056472621156 - 44631.01109458185 * eta +
        (
            -6541.308761668722 - 266959.23419307504 * eta +
            686328.3229317984 * eta2 +
            (3405.6372187679685 - 437507.7208209015 * eta + 1.6318171307344697e6 * eta2) *
            xi +
            (-7462.648563007646 - 114585.25177151319 * eta + 674402.4689098676 * eta2) * xi2
        ) * xi
    sigma3 =
        22933.658273436497 +
        230960.00814979506 * eta +
        (
            14961.083974183695 + 1.1940181342318142e6 * eta - 3.1042239693052764e6 * eta2 +
            (-3038.166617199259 + 1.8720322849093592e6 * eta - 7.309145012085539e6 * eta2) *
            xi +
            (42738.22871475411 + 467502.018616601 * eta - 3.064853498512499e6 * eta2) * xi2
        ) * xi
    sigma4 =
        -14621.71522218357 - 377812.8579387104 * eta +
        (
            -9608.682631509726 - 1.7108925257424056e6 * eta +
            4.332924601416521e6 * eta2 +
            (
                -22366.683262266528 - 2.5019716386377467e6 * eta +
                1.0274495902259542e7 * eta2
            ) * xi +
            (-85360.30079034246 - 570025.3441737515 * eta + 4.396844346849777e6 * eta2) *
            xi2
        ) * xi

    beta1 =
        97.89747327985583 - 42.659730877489224 * eta +
        (
            153.48421037904913 - 1417.0620760768954 * eta +
            2752.8614143665027 * eta2 +
            (138.7406469558649 - 1433.6585075135881 * eta + 2857.7418952430758 * eta2) *
            xi +
            (41.025109467376126 - 423.680737974639 * eta + 850.3594335637173 * eta2) * xi2
        ) * xi
    beta2 =
        -3.282701958759534 - 9.051384468245866 * eta +
        (
            -12.415449742258042 + 55.4716447709787 * eta - 106.05109938966335 * eta2 +
            (-11.953044553690658 + 76.80704618365418 * eta - 155.33172948098394 * eta2) *
            xi +
            (-3.4129261592393263 + 25.572377569952536 * eta - 54.408036707740465 * eta2) * xi2
        ) * xi
    beta3 =
        -0.000025156429818799565 +
        0.000019750256942201327 * eta +
        (
            -0.000018370671469295915 +
            0.000021886317041311973 * eta +
            0.00008250240316860033 * eta2 +
            (
                7.157371250566708e-6 - 0.000055780000112270685 * eta +
                0.00019142082884072178 * eta2
            ) * xi +
            (
                5.447166261464217e-6 - 0.00003220610095021982 * eta +
                0.00007974016714984341 * eta2
            ) * xi2
        ) * xi

    alpha1 =
        43.31514709695348 +
        638.6332679188081 * eta +
        (
            -32.85768747216059 + 2415.8938269370315 * eta - 5766.875169379177 * eta2 +
            (-61.85459307173841 + 2953.967762459948 * eta - 8986.29057591497 * eta2) * xi +
            (-21.571435779762044 + 981.2158224673428 * eta - 3239.5664895930286 * eta2) *
            xi2
        ) * xi
    alpha2 =
        -0.07020209449091723 - 0.16269798450687084 * eta +
        (
            -0.1872514685185499 + 1.138313650449945 * eta - 2.8334196304430046 * eta2 +
            (-0.17137955686840617 + 1.7197549338119527 * eta - 4.539717638261272 * eta2) * xi +
            (-0.049983437357548705 + 0.6062072042325992 * eta - 1.682769630644546 * eta2) * xi2
        ) * xi
    alpha3 =
        9.5988072383479 - 397.05438595557433 * eta +
        (
            16.202126189517813 - 1574.8286986717037 * eta +
            3600.3410843831093 * eta2 +
            (27.092429659075467 - 1786.482357315139 * eta + 5152.919378666511 * eta2) * xi +
            (11.175710130033895 - 577.7999423177481 * eta + 1808.730762932043 * eta2) * xi2
        ) * xi
    alpha4 =
        -0.02989487384493607 +
        1.4022106448583738 * eta +
        (
            -0.07356049468633846 +
            0.8337006542278661 * eta +
            0.2240008282397391 * eta2 +
            (-0.055202870001177226 + 0.5667186343606578 * eta + 0.71869313615176228 * eta2) * xi +
            (
                -0.015507437354325743 +
                0.15750322779277187 * eta +
                0.21076815715176228 * eta2
            ) * xi2
        ) * xi
    alpha5 =
        0.9974408278363099 - 0.007884449714907203 * eta +
        (
            -0.059046901195591035 + 1.3958712396764088 * eta - 4.516631601676276 * eta2 +
            (-0.05585343136869692 + 1.7516580039343603 * eta - 5.990208965347804 * eta2) * xi +
            (-0.017945336522161195 + 0.5965097794825992 * eta - 2.060931148018704 * eta2) * xi2
        ) * xi

    TF2OverallAmpl = 3 / (128.0 * eta)
    TF2_5coeff_tmp =
        38645.0 * pi / 756.0 - 65.0 * pi * eta / 9.0 - (
            (732985.0 / 2268.0 - 24260.0 * eta / 81.0 - 340.0 * eta2 / 9.0) * chi_s +
            (732985.0 / 2268.0 + 140.0 * eta / 9.0) * Seta * chi_a
        )
    TF2_6coeff_tmp =
        11583.231236531 / 4.694215680 - 640.0 / 3.0 * (pi^2) -
        684.8 / 2.1 * 0.5772156649015 +
        eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * (pi^2)) +
        eta2 * 76.055 / 1.728 - eta2 * eta * 127.825 / 1.296 - log(4.0) * 684.8 / 2.1 +
        pi * chi1 * m1ByM * (1490.0 / 3.0 + m1ByM * 260.0) +
        pi * chi2 * m2ByM * (1490.0 / 3.0 + m2ByM * 260.0) +
        (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1 * chi2 +
        (4703.5 / 8.4 + 2935.0 / 6.0 * m1ByM - 120.0 * m1ByM^2) *
        m1ByM^2 * chi1 * chi1 +
        (-4108.25 / 6.72 - 108.5 / 1.2 * m1ByM + 125.5 / 3.6 * m1ByM^2) *
        m1ByM^2 * chi1 * chi1 +
        (4703.5 / 8.4 + 2935.0 / 6.0 * m2ByM - 120.0 * m2ByM^2) *
        m2ByM^2 * chi2 * chi2 +
        (-4108.25 / 6.72 - 108.5 / 1.2 * m2ByM + 125.5 / 3.6 * m2ByM^2) *
        m2ByM^2 * chi2 * chi2

    TF2coeffs = TF2coeffsStructure(
        1.0,
        0.0,
        3715.0 / 756.0 + (55.0 * eta) / 9.0,
        -16.0 * pi +
        (113.0 * Seta * chi_a) / 3.0 +
        (113.0 / 3.0 - (76.0 * eta) / 3.0) * chi_s,
        5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta2) / 72.0 +
        247.0 / 4.8 * eta * chi1 * chi2 - 721.0 / 4.8 * eta * chi1 * chi2 +
        (-720.0 / 9.6 * 1.0 + 1.0 / 9.6) * m1ByM^2 * chi1 * chi1 +
        (-720.0 / 9.6 * 1.0 + 1.0 / 9.6) * m2ByM^2 * chi2 * chi2 +
        (240.0 / 9.6 * 1.0 - 7.0 / 9.6) * m1ByM^2 * chi1 * chi1 +
        (240.0 / 9.6 * 1.0 - 7.0 / 9.6) * m2ByM^2 * chi2 * chi2,
        TF2_5coeff_tmp,
        TF2_6coeff_tmp - (
            (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * chi1 * chi2 +
            ((4703.5 / 8.4 + 2935.0 / 6.0 * m1ByM - 120.0 * m1ByM^2) +
            (-4108.25 / 6.72 - 108.5 / 1.2 * m1ByM + 125.5 / 3.6 * m1ByM^2)) *
            m1ByM * m1ByM * chi1 * chi1 +
            ((4703.5 / 8.4 + 2935.0 / 6.0 * m2ByM - 120.0 * m2ByM^2) +
            (-4108.25 / 6.72 - 108.5 / 1.2 * m2ByM + 125.5 / 3.6 * m2ByM^2)) *
            m2ByM * m2ByM * chi2 * chi2
        ),
        -6848.0 / 21.0,
        TF2_5coeff_tmp * 3.0,
        77096675.0 * pi / 254016.0 + 378515.0 * pi * eta / 1512.0 -
        74045.0 * pi * eta2 / 756.0 +
        (
            -25150083775.0 / 3048192.0 + 10566655595.0 * eta / 762048.0 -
            1042165.0 * eta2 / 3024.0 + 5345.0 * eta2 * eta / 36.0
        ) * chi_s +
        Seta * (
            (
                -25150083775.0 / 3048192.0 + 26804935.0 * eta / 6048.0 -
                1985.0 * eta2 / 48.0
            ) * chi_a
        ),
        0.0,
        0.0,
    )

    PhiInspcoeffs = PhiInspcoeffsStructure(
        TF2coeffs.five * TF2OverallAmpl,
        TF2coeffs.seven * TF2OverallAmpl * (pi^(2.0 / 3.0)),
        TF2coeffs.six * TF2OverallAmpl * (pi^(1.0 / 3.0)),
        TF2coeffs.six_log * TF2OverallAmpl * (pi^(1.0 / 3.0)),
        TF2coeffs.five_log * TF2OverallAmpl,
        TF2coeffs.four * TF2OverallAmpl * (pi^(-1.0 / 3.0)),
        TF2coeffs.three * TF2OverallAmpl * (pi^(-2.0 / 3.0)),
        TF2coeffs.two * TF2OverallAmpl / pi,
        TF2coeffs.one * TF2OverallAmpl * (pi^(-4.0 / 3.0)),
        TF2coeffs.zero * TF2OverallAmpl * (pi^(-5.0 / 3.0)),
        sigma1,
        sigma2 * 0.75,
        sigma3 * 0.6,
        sigma4 * 0.5,
    )

    fMRDJoin = 0.5 * fring

    DPhiIns =
        (
            2.0 * TF2coeffs.seven * TF2OverallAmpl * ((pi * fInsJoin)^(7.0 / 3.0)) +
            (
                TF2coeffs.six * TF2OverallAmpl +
                TF2coeffs.six_log * TF2OverallAmpl * (1.0 + log(pi * fInsJoin) / 3.0)
            ) * ((pi * fInsJoin)^(2.0)) +
            TF2coeffs.five_log * TF2OverallAmpl * ((pi * fInsJoin)^(5.0 / 3.0)) -
            TF2coeffs.four * TF2OverallAmpl * ((pi * fInsJoin)^(4.0 / 3.0)) -
            2.0 * TF2coeffs.three * TF2OverallAmpl * (pi * fInsJoin) -
            3.0 * TF2coeffs.two * TF2OverallAmpl * ((pi * fInsJoin)^(2.0 / 3.0)) -
            4.0 * TF2coeffs.one * TF2OverallAmpl * ((pi * fInsJoin)^(1.0 / 3.0)) -
            5.0 * TF2coeffs.zero * TF2OverallAmpl
        ) * pi / (3.0 * ((pi * fInsJoin)^(8.0 / 3.0))) +
        (
            sigma1 +
            sigma2 * (fInsJoin^(1.0 / 3.0)) +
            sigma3 * (fInsJoin^(2.0 / 3.0)) +
            sigma4 * fInsJoin
        ) * etaInv

    DPhiInt = (beta1 + beta3 / (fInsJoin^4) + beta2 / fInsJoin) * etaInv
    C2Int = DPhiIns - DPhiInt

    PhiInsJoin =
        PhiInspcoeffs.initial_phasing +
        PhiInspcoeffs.two_thirds * (fInsJoin^(2.0 / 3.0)) +
        PhiInspcoeffs.third * (fInsJoin^(1.0 / 3.0)) +
        PhiInspcoeffs.third_log * (fInsJoin^(1.0 / 3.0)) * log(pi * fInsJoin) / 3.0 +
        PhiInspcoeffs.log * log(pi * fInsJoin) / 3.0 +
        PhiInspcoeffs.min_third * (fInsJoin^(-1.0 / 3.0)) +
        PhiInspcoeffs.min_two_thirds * (fInsJoin^(-2.0 / 3.0)) +
        PhiInspcoeffs.min_one / fInsJoin +
        PhiInspcoeffs.min_four_thirds * (fInsJoin^(-4.0 / 3.0)) +
        PhiInspcoeffs.min_five_thirds * (fInsJoin^(-5.0 / 3.0)) +
        (
            PhiInspcoeffs.one * fInsJoin +
            PhiInspcoeffs.four_thirds * (fInsJoin^(4.0 / 3.0)) +
            PhiInspcoeffs.five_thirds * (fInsJoin^(5.0 / 3.0)) +
            PhiInspcoeffs.two * fInsJoin * fInsJoin
        ) * etaInv

    PhiIntJoin =
        beta1 * fInsJoin - beta3 / (3.0 * fInsJoin * fInsJoin * fInsJoin) +
        beta2 * log(fInsJoin)

    C1Int = PhiInsJoin - PhiIntJoin * etaInv - C2Int * fInsJoin

    PhiIntTempVal =
        (beta1 * fMRDJoin - beta3 / (3.0 * fMRDJoin * fMRDJoin * fMRDJoin) + beta2 * log(fMRDJoin)) * etaInv +
        C1Int + C2Int * fMRDJoin
    DPhiIntTempVal = C2Int + (beta1 + beta3 / (fMRDJoin^4) + beta2 / fMRDJoin) * etaInv
    DPhiMRDVal =
        (
            alpha1 +
            alpha2 / (fMRDJoin * fMRDJoin) +
            alpha3 / (fMRDJoin^(0.25)) +
            alpha4 / (
                fdamp * (
                    1.0 +
                    (fMRDJoin - alpha5 * fring) * (fMRDJoin - alpha5 * fring) / (fdamp * fdamp)
                )
            )
        ) * etaInv
    PhiMRJoinTemp =
        -(alpha2 / fMRDJoin) +
        (4.0 / 3.0) * (alpha3 * (fMRDJoin^(0.75))) +
        alpha1 * fMRDJoin +
        alpha4 * atan((fMRDJoin - alpha5 * fring) / fdamp)

    C2MRD = DPhiIntTempVal - DPhiMRDVal
    C1MRD = PhiIntTempVal - PhiMRJoinTemp * etaInv - C2MRD * fMRDJoin

    return (
        T = T, fInsJoin = fInsJoin, fMRDJoin = fMRDJoin, fcutPar = fcutPar,
        t0 = t0, phiRef = phiRef, M = M, GMsun_over_c3 = GMsun_over_c3, etaInv = etaInv,
        PhiInspcoeffs = PhiInspcoeffs,
        beta1 = beta1, beta2 = beta2, beta3 = beta3, C1Int = C1Int, C2Int = C2Int,
        alpha1 = alpha1, alpha2 = alpha2, alpha3 = alpha3, alpha4 = alpha4, alpha5 = alpha5,
        fring = fring, fdamp = fdamp, C1MRD = C1MRD, C2MRD = C2MRD,
    )
end # @Tlit T
end

"""
    _phi_scalar(coeffs, f)

Non-allocating IMRPhenomD phase evaluation at a single frequency `f` (not a vector), using the
frequency-independent coefficients from [`_phi_coeffs`](@ref). This is the scalar core that
`Phi` broadcasts over an `AbstractVector` of frequencies (preserving the existing array-in/
array-out API for all current callers), and that the fused KernelAbstractions likelihood kernel
calls directly in a per-thread frequency loop with zero array allocation.
"""
@inline function _phi_scalar(coeffs, f)
    T = coeffs.T
@Tlit T begin
    f_safe = max(f, eps(T))
    fgrid = coeffs.M * coeffs.GMsun_over_c3 * f_safe

    phis = ifelse(
        fgrid < coeffs.fInsJoin,
        coeffs.PhiInspcoeffs.initial_phasing +
        coeffs.PhiInspcoeffs.two_thirds * fgrid^(2.0 / 3.0) +
        coeffs.PhiInspcoeffs.third * fgrid^(1.0 / 3.0) +
        coeffs.PhiInspcoeffs.third_log * fgrid^(1.0 / 3.0) * log(pi * fgrid) / 3.0 +
        coeffs.PhiInspcoeffs.log * log(pi * fgrid) / 3.0 +
        coeffs.PhiInspcoeffs.min_third * fgrid^(-1.0 / 3.0) +
        coeffs.PhiInspcoeffs.min_two_thirds * fgrid^(-2.0 / 3.0) +
        coeffs.PhiInspcoeffs.min_one / fgrid +
        coeffs.PhiInspcoeffs.min_four_thirds * fgrid^(-4.0 / 3.0) +
        coeffs.PhiInspcoeffs.min_five_thirds * fgrid^(-5.0 / 3.0) +
        (
            coeffs.PhiInspcoeffs.one * fgrid +
            coeffs.PhiInspcoeffs.four_thirds * fgrid^(4.0 / 3.0) +
            coeffs.PhiInspcoeffs.five_thirds * fgrid^(5.0 / 3.0) +
            coeffs.PhiInspcoeffs.two * fgrid * fgrid
        ) * coeffs.etaInv,
        ifelse(
            fgrid < coeffs.fMRDJoin,
            (coeffs.beta1 * fgrid - coeffs.beta3 / (3.0 * fgrid * fgrid * fgrid) + coeffs.beta2 * log(fgrid)) * coeffs.etaInv + coeffs.C1Int + coeffs.C2Int * fgrid,
            ifelse(
                fgrid < coeffs.fcutPar,
                (
                    -(coeffs.alpha2 / fgrid) +
                    (4.0 / 3.0) * (coeffs.alpha3 * (fgrid^(3.0 / 4.0))) +
                    coeffs.alpha1 * fgrid +
                    coeffs.alpha4 * atan((fgrid - coeffs.alpha5 * coeffs.fring) / coeffs.fdamp)
                ) * coeffs.etaInv +
                coeffs.C1MRD +
                coeffs.C2MRD * fgrid,
                zero(T),
            ),
        ),
    )

    return phis + ifelse(fgrid < coeffs.fcutPar, -coeffs.t0 * (fgrid - zero(T)) - coeffs.phiRef, zero(T))
end # @Tlit T
end

"""
    Phi(model::IMRPhenomD, f, mc, eta, chi1, chi2; kwargs...)

IMRPhenomD frequency-domain phase. `f` may be a scalar (returns a scalar, non-allocating) or an
`AbstractVector` (returns a same-length `Vector`) -- computes the frequency-independent
coefficients once via [`_phi_coeffs`](@ref), then evaluates [`_phi_scalar`](@ref) at each
frequency. Callers evaluating many frequencies for one set of parameters where per-call
allocation matters (e.g. inside a GPU kernel) should call `_phi_coeffs` once and `_phi_scalar`
per point directly instead of going through this array-oriented wrapper (tests and the LALSuite
comparison scripts use it).

Implementation note: this deliberately uses an explicit `similar`-then-loop instead of
broadcasting `_phi_scalar` over `f` (e.g. `_phi_scalar.(Ref(coeffs), f)`). The broadcast form is
numerically identical but confuses Enzyme's static activity analysis when this array-out `Phi`
is itself a step inside a larger differentiated call graph (observed as an
`EnzymeRuntimeActivityError` — "Mismatched activity" — when GravitationalWaveBayes.jl differentiated a batched
generator built on it); the plain loop does not trigger it.

No parameter-domain assertions run here — this is the hot path. See `validate_phenomd_params`
for an opt-in host-side check.
"""
function Phi(model::IMRPhenomD,
    f,
    mc,
    eta,
    chi1,
    chi2;
    fInsJoin_PHI = 0.018,
    fcutPar = 0.2,
    t0 = zero(eltype(f)),
    phiRef = zero(eltype(f)),
    GMsun_over_c3 = GMsun_over_c3,
)
    coeffs = _phi_coeffs(model, mc, eta, chi1, chi2;
        fInsJoin_PHI = fInsJoin_PHI, fcutPar = fcutPar, t0 = t0, phiRef = phiRef,
        GMsun_over_c3 = GMsun_over_c3)
    if f isa AbstractArray
        out = similar(f, coeffs.T)
        @inbounds for i in eachindex(f)
            out[i] = _phi_scalar(coeffs, f[i])
        end
        return out
    else
        return _phi_scalar(coeffs, f)
    end
end

"""
    _ampl_coeffs(model::IMRPhenomD, mc, eta, chi1, chi2, dL; kwargs...)

Frequency-independent part of the IMRPhenomD amplitude calculation, factored out of `Ampl` for the
same reasons as [`_phi_coeffs`](@ref) (see its docstring): computing this block once per
waveform instead of once per frequency bin is both the correct design for a per-GPU-thread
kernel and empirically required for `Enzyme.autodiff` to complete on a KernelAbstractions
kernel that evaluates the amplitude (the array-allocating `Ampl` caused `Enzyme.Reverse` to hang
indefinitely inside a `@kernel` body; the non-allocating scalar core does not).
"""
@inline function _ampl_coeffs(model::IMRPhenomD,
    mc,
    eta,
    chi1,
    chi2,
    dL;
    fcutPar = 0.2,
    fInsJoin_Ampl = 0.014,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    # No @assert here: this is the hot path (per-walker GPU kernel call / Enzyme target).
    # Call `validate_phenomd_params(eta, chi1, chi2; dL=dL)` explicitly for host-side checks.
    T = promote_type(typeof(mc), typeof(eta), typeof(chi1), typeof(chi2), typeof(dL))

    # See the matching comment in _phi_coeffs(): keyword-derived scalars need an explicit cast
    # since @Tlit only rewrites literals written directly in this file's source.
    fcutPar = T(fcutPar)
    fInsJoin_Ampl = T(fInsJoin_Ampl)
    GMsun_over_c3 = T(GMsun_over_c3)
    GMsun_over_c2_Gpc = T(GMsun_over_c2_Gpc)

@Tlit T begin
    M = mc / (eta^(3.0 / 5.0))
    eta2 = eta * eta
    chi12 = chi1 * chi1
    chi22 = chi2 * chi2

    Seta = eta < 0.25 ? sqrt(1.0 - 4.0 * eta) : zero(eta)
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    chiPN = (chi_s * (1.0 - eta * 76.0 / 113.0) + Seta * chi_a)
    xi = -1.0 + chiPN
    xi2 = xi * xi
    aeff = _finalspin(model, eta, chi1, chi2)
    Erad = _radiatednrg(model, eta, chi1, chi2)

    fring = ((0.05947169566573468 - 0.14989771215394762 * aeff + 0.09535606290986028 * aeff * aeff + 0.02260924869042963 * aeff * aeff * aeff - 0.02501704155363241 * aeff * aeff * aeff * aeff - 0.005852438240997211 * (aeff^5) + 0.0027489038393367993 * (aeff^6) + 0.0005821983163192694 * (aeff^7)) / (1 - 2.8570126619966296 * aeff + 2.373335413978394 * aeff * aeff - 0.6036964688511505 * aeff * aeff * aeff * aeff + 0.0873798215084077 * (aeff^6))) / (1.0 - Erad)
    fdamp = ((0.014158792290965177 - 0.036989395871554566 * aeff + 0.026822526296575368 * aeff * aeff + 0.0008490933750566702 * aeff * aeff * aeff - 0.004843996907020524 * aeff * aeff * aeff * aeff - 0.00014745235759327472 * (aeff^5) + 0.0001504546201236794 * (aeff^6)) / (1 - 2.5900842798681376 * aeff + 1.8952576220623967 * aeff * aeff - 0.31416610693042507 * aeff * aeff * aeff * aeff + 0.009002719412407133 * (aeff^6))) / (1.0 - Erad)

    gamma1 =
        0.006927402739328343 +
        0.03020474290328911 * eta +
        (
            0.006308024337706171 - 0.12074130661131138 * eta +
            0.26271598905781324 * eta2 +
            (
                0.0034151773647198794 - 0.10779338611188374 * eta +
                0.27098966966891747 * eta2
            ) * xi +
            (
                0.0007374185938559283 - 0.02749621038376281 * eta +
                0.0733150789135702 * eta2
            ) * xi2
        ) * xi
    gamma2 =
        1.010344404799477 +
        0.0008993122007234548 * eta +
        (
            0.283949116804459 - 4.049752962958005 * eta +
            13.207828172665366 * eta2 +
            (0.10396278486805426 - 7.025059158961947 * eta + 24.784892370130475 * eta2) *
            xi +
            (0.03093202475605892 - 2.6924023896851663 * eta + 9.609374464684983 * eta2) *
            xi2
        ) * xi
    gamma3 =
        1.3081615607036106 - 0.005537729694807678 * eta +
        (
            -0.06782917938621007 - 0.6689834970767117 * eta +
            3.403147969668083 * eta2 +
            (-0.05296577374411866 - 0.9923793203111362 * eta + 4.820681208440587 * eta2) *
            xi +
            (
                -0.006134139870393713 - 0.38429253308696365 * eta +
                1.7561754421985984 * eta2
            ) * xi2
        ) * xi
    if gamma2 >= 1.0
        fpeak = abs(fring - (fdamp * gamma3) / gamma2)
    else
        fpeak = fring + (fdamp * (-1.0 + sqrt(1.0 - gamma2 * gamma2)) * gamma3) / gamma2
    end

    rho1 =
        3931.8979897196696 - 17395.758706812805 * eta +
        (
            3132.375545898835 + 343965.86092361377 * eta - 1.2162565819981997e6 * eta2 +
            (-70698.00600428853 + 1.383907177859705e6 * eta - 3.9662761890979446e6 * eta2) *
            xi +
            (-60017.52423652596 + 803515.1181825735 * eta - 2.091710365941658e6 * eta2) *
            xi2
        ) * xi
    rho2 =
        -40105.47653771657 +
        112253.0169706701 * eta +
        (
            23561.696065836168 - 3.476180699403351e6 * eta +
            1.137593670849482e7 * eta2 +
            (754313.1127166454 - 1.308476044625268e7 * eta + 3.6444584583928134e7 * eta2) *
            xi +
            (596226.612472288 - 7.42779011435644e6 * eta + 1.8928977514040343e7 * eta2) *
            xi2
        ) * xi
    rho3 =
        83208.35471266537 - 191237.7264145924 * eta +
        (
            -210916.2454782992 + 8.71797508352568e6 * eta - 2.6914942420669552e7 * eta2 +
            (-1.9889806527362722e6 + 3.0888029960154563e7 * eta - 8.390870279256162e7 * eta2) *
            xi +
            (-1.4535031953446497e6 + 1.706352899082216e7 * eta - 4.274865973112091e7 * eta2) *
            xi2
        ) * xi

    f1Interm = fInsJoin_Ampl
    f3Interm = fpeak
    dfInterm = 0.5 * (f3Interm - f1Interm)
    f2Interm = f1Interm + dfInterm

    amp0 = sqrt(2.0 * eta / 3.0) * (pi^(-1.0 / 6.0))
    # Acoeffs is a keyword-labeled NamedTuple (not a positional struct) specifically so that
    # field order can never again silently diverge from field NAME -- a previous version used a
    # positional `AcoeffsStructure(...)` constructor whose argument order didn't match its own
    # struct's declared field order, silently swapping six of the eight PN amplitude
    # coefficients (e.g. the O(1) linear-in-chi "one" coefficient was replaced by the O(1000s)
    # ringdown coefficient "rho1"), which is what caused the ~18-45x real-data amplitude scale
    # bug found while cross-validating against LALSuite (see
    # gwbayes_amplitude_scale_bug_critical memory). Field names/values match gwfast's Acoeffs
    # dict one-to-one.
    Acoeffs = (
        two_thirds = ((-969.0 + 1804.0 * eta) * (pi^(2.0 / 3.0))) / 672.0,
        one = (
            (
                chi1 * (81.0 * (1.0 + Seta) - 44.0 * eta) +
                chi2 * (81.0 - 81.0 * Seta - 44.0 * eta)
            ) * pi
        ) / 48.0,
        four_thirds = (
            (
                -27312085.0 - 10287648.0 * chi22 - 10287648.0 * chi12 * (1.0 + Seta) +
                10287648.0 * chi22 * Seta +
                24.0 *
                (
                    -1975055.0 + 857304.0 * chi12 - 994896.0 * chi1 * chi2 +
                    857304.0 * chi22
                ) *
                eta +
                35371056.0 * eta2
            ) * (pi^(4.0 / 3.0))
        ) / 8.128512e6,
        five_thirds = (
            (pi^(5.0 / 3.0)) * (
                chi2 * (
                    -285197.0 * (-1.0 + Seta) + 4.0 * (-91902.0 + 1579.0 * Seta) * eta -
                    35632.0 * eta2
                ) +
                chi1 * (
                    285197.0 * (1.0 + Seta) - 4.0 * (91902.0 + 1579.0 * Seta) * eta -
                    35632.0 * eta2
                ) +
                42840.0 * (-1.0 + 4.0 * eta) * pi
            )
        ) / 32256.0,
        two = -(
            (pi^2) * (
                -336.0 *
                (
                    -3248849057.0 + 2943675504.0 * chi12 - 3339284256.0 * chi1 * chi2 +
                    2943675504.0 * chi22
                ) *
                eta2 - 324322727232.0 * eta2 * eta -
                7.0 * (
                    -177520268561.0 +
                    107414046432.0 * chi22 +
                    107414046432.0 * chi12 * (1.0 + Seta) - 107414046432.0 * chi22 * Seta +
                    11087290368.0 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) * pi
                ) +
                12.0 *
                eta *
                (
                    -545384828789.0 - 176491177632.0 * chi1 * chi2 +
                    202603761360.0 * chi22 +
                    77616.0 * chi12 * (2610335.0 + 995766.0 * Seta) -
                    77287373856.0 * chi22 * Seta +
                    5841690624.0 * (chi1 + chi2) * pi +
                    21384760320.0 * (pi^2)
                )
            )
        ) / 6.0085960704e10,
        seven_thirds = rho1,
        eight_thirds = rho2,
        three = rho3,
    )

    v1 =
        1.0 +
        (f1Interm^(2.0 / 3.0)) * Acoeffs.two_thirds +
        (f1Interm^(4.0 / 3.0)) * Acoeffs.four_thirds +
        (f1Interm^(5.0 / 3.0)) * Acoeffs.five_thirds +
        (f1Interm^(7.0 / 3.0)) * Acoeffs.seven_thirds +
        (f1Interm^(8.0 / 3.0)) * Acoeffs.eight_thirds +
        f1Interm *
        (Acoeffs.one + f1Interm * Acoeffs.two + f1Interm * f1Interm * Acoeffs.three)
    d1 =
        ((-969.0 + 1804.0 * eta) * (pi^(2.0 / 3.0))) / (1008.0 * (f1Interm^(1.0 / 3.0))) +
        (
            (
                chi1 * (81.0 * (1.0 + Seta) - 44.0 * eta) +
                chi2 * (81.0 - 81.0 * Seta - 44.0 * eta)
            ) * pi
        ) / 48.0 +
        (
            (
                -27312085.0 - 10287648.0 * chi22 - 10287648.0 * chi12 * (1.0 + Seta) +
                10287648.0 * chi22 * Seta +
                24.0 *
                (
                    -1975055.0 + 857304.0 * chi12 - 994896.0 * chi1 * chi2 +
                    857304.0 * chi22
                ) *
                eta +
                35371056.0 * eta2
            ) *
            (f1Interm^(1.0 / 3.0)) *
            (pi^(4.0 / 3.0))
        ) / 6.096384e6 +
        (
            5.0 *
            (f1Interm^(2.0 / 3.0)) *
            (pi^(5.0 / 3.0)) *
            (
                chi2 * (
                    -285197.0 * (-1.0 + Seta) + 4.0 * (-91902.0 + 1579.0 * Seta) * eta -
                    35632.0 * eta2
                ) +
                chi1 * (
                    285197.0 * (1.0 + Seta) - 4.0 * (91902.0 + 1579.0 * Seta) * eta -
                    35632.0 * eta2
                ) +
                42840.0 * (-1.0 + 4.0 * eta) * pi
            )
        ) / 96768.0 -
        (
            f1Interm *
            (pi^2) *
            (
                -336.0 *
                (
                    -3248849057.0 + 2943675504.0 * chi12 - 3339284256.0 * chi1 * chi2 +
                    2943675504.0 * chi22
                ) *
                eta2 - 324322727232.0 * eta2 * eta -
                7.0 * (
                    -177520268561.0 +
                    107414046432.0 * chi22 +
                    107414046432.0 * chi12 * (1.0 + Seta) - 107414046432.0 * chi22 * Seta +
                    11087290368.0 * (chi1 + chi2 + chi1 * Seta - chi2 * Seta) * pi
                ) +
                12.0 *
                eta *
                (
                    -545384828789.0 - 176491177632.0 * chi1 * chi2 +
                    202603761360.0 * chi22 +
                    77616.0 * chi12 * (2610335.0 + 995766.0 * Seta) -
                    77287373856.0 * chi22 * Seta +
                    5841690624.0 * (chi1 + chi2) * pi +
                    21384760320.0 * (pi^2)
                )
            )
        ) / 3.0042980352e10 +
        (7.0 / 3.0) * (f1Interm^(4.0 / 3.0)) * rho1 +
        (8.0 / 3.0) * (f1Interm^(5.0 / 3.0)) * rho2 +
        3.0 * (f1Interm * f1Interm) * rho3
    v3 =
        exp(-(f3Interm - fring) * gamma2 / (fdamp * gamma3)) * (fdamp * gamma3 * gamma1) /
        ((f3Interm - fring) * (f3Interm - fring) + fdamp * gamma3 * fdamp * gamma3)
    d2 =
        (
            (-2.0 * fdamp * (f3Interm - fring) * gamma3 * gamma1) /
            ((f3Interm - fring) * (f3Interm - fring) + fdamp * gamma3 * fdamp * gamma3) -
            (gamma2 * gamma1)
        ) / (
            exp((f3Interm - fring) * gamma2 / (fdamp * gamma3)) *
            ((f3Interm - fring) * (f3Interm - fring) + fdamp * gamma3 * fdamp * gamma3)
        )
    v2 =
        0.8149838730507785 +
        2.5747553517454658 * eta +
        (
            1.1610198035496786 - 2.3627771785551537 * eta +
            6.771038707057573 * eta2 +
            (0.7570782938606834 - 2.7256896890432474 * eta + 7.1140380319149965 * eta2) *
            xi +
            (0.1766934149293479 - 0.7978690983168183 * eta + 2.1162391502005153 * eta2) * xi2
        ) * xi
    f1 = f1Interm
    f2 = f2Interm
    f3 = f3Interm
    f12 = f1Interm * f1Interm
    f13 = f1Interm * f12
    f14 = f1Interm * f13
    f15 = f1Interm * f14
    f22 = f2Interm * f2Interm
    f23 = f2Interm * f22
    f24 = f2Interm * f23
    f32 = f3Interm * f3Interm
    f33 = f3Interm * f32
    f34 = f3Interm * f33
    f35 = f3Interm * f34

    delta0 = -(
        (
            d2 * f15 * f22 * f3 - 2.0 * d2 * f14 * f23 * f3 + d2 * f13 * f24 * f3 -
            d2 * f15 * f2 * f32 + d2 * f14 * f22 * f32 - d1 * f13 * f23 * f32 +
            d2 * f13 * f23 * f32 +
            d1 * f12 * f24 * f32 - d2 * f12 * f24 * f32 +
            d2 * f14 * f2 * f33 +
            2.0 * d1 * f13 * f22 * f33 - 2.0 * d2 * f13 * f22 * f33 -
            d1 * f12 * f23 * f33 + d2 * f12 * f23 * f33 - d1 * f1 * f24 * f33 -
            d1 * f13 * f2 * f34 - d1 * f12 * f22 * f34 +
            2.0 * d1 * f1 * f23 * f34 +
            d1 * f12 * f2 * f35 - d1 * f1 * f22 * f35 + 4.0 * f12 * f23 * f32 * v1 -
            3.0 * f1 * f24 * f32 * v1 - 8.0 * f12 * f22 * f33 * v1 +
            4.0 * f1 * f23 * f33 * v1 +
            f24 * f33 * v1 +
            4.0 * f12 * f2 * f34 * v1 +
            f1 * f22 * f34 * v1 - 2.0 * f23 * f34 * v1 - 2.0 * f1 * f2 * f35 * v1 +
            f22 * f35 * v1 - f15 * f32 * v2 + 3.0 * f14 * f33 * v2 -
            3.0 * f13 * f34 * v2 + f12 * f35 * v2 - f15 * f22 * v3 +
            2.0 * f14 * f23 * v3 - f13 * f24 * v3 + 2.0 * f15 * f2 * f3 * v3 -
            f14 * f22 * f3 * v3 - 4.0 * f13 * f23 * f3 * v3 +
            3.0 * f12 * f24 * f3 * v3 - 4.0 * f14 * f2 * f32 * v3 +
            8.0 * f13 * f22 * f32 * v3 - 4.0 * f12 * f23 * f32 * v3
        ) / (
            (f1 - f2) *
            (f1 - f2) *
            (f1 - f3) *
            (f1 - f3) *
            (f1 - f3) *
            (f3 - f2) *
            (f3 - f2)
        )
    )
    delta1 = -(
        (
            -(d2 * f15 * f22) + 2.0 * d2 * f14 * f23 - d2 * f13 * f24 -
            d2 * f14 * f22 * f3 +
            2.0 * d1 * f13 * f23 * f3 +
            2.0 * d2 * f13 * f23 * f3 - 2.0 * d1 * f12 * f24 * f3 - d2 * f12 * f24 * f3 +
            d2 * f15 * f32 - 3.0 * d1 * f13 * f22 * f32 - d2 * f13 * f22 * f32 +
            2.0 * d1 * f12 * f23 * f32 - 2.0 * d2 * f12 * f23 * f32 +
            d1 * f1 * f24 * f32 +
            2.0 * d2 * f1 * f24 * f32 - d2 * f14 * f33 +
            d1 * f12 * f22 * f33 +
            3.0 * d2 * f12 * f22 * f33 - 2.0 * d1 * f1 * f23 * f33 -
            2.0 * d2 * f1 * f23 * f33 +
            d1 * f24 * f33 +
            d1 * f13 * f34 +
            d1 * f1 * f22 * f34 - 2.0 * d1 * f23 * f34 - d1 * f12 * f35 + d1 * f22 * f35 -
            8.0 * f12 * f23 * f3 * v1 +
            6.0 * f1 * f24 * f3 * v1 +
            12.0 * f12 * f22 * f32 * v1 - 8.0 * f1 * f23 * f32 * v1 - 4.0 * f12 * f34 * v1 +
            2.0 * f1 * f35 * v1 +
            2.0 * f15 * f3 * v2 - 4.0 * f14 * f32 * v2 + 4.0 * f12 * f34 * v2 -
            2.0 * f1 * f35 * v2 - 2.0 * f15 * f3 * v3 + 8.0 * f12 * f23 * f3 * v3 -
            6.0 * f1 * f24 * f3 * v3 + 4.0 * f14 * f32 * v3 - 12.0 * f12 * f22 * f32 * v3 +
            8.0 * f1 * f23 * f32 * v3
        ) / (
            (f1 - f2) *
            (f1 - f2) *
            (f1 - f3) *
            (f1 - f3) *
            (f1 - f3) *
            (f3 - f2) *
            (f3 - f2)
        )
    )
    delta2 = -(
        (
            d2 * f15 * f2 - d1 * f13 * f23 - 3.0 * d2 * f13 * f23 +
            d1 * f12 * f24 +
            2.0 * d2 * f12 * f24 - d2 * f15 * f3 + d2 * f14 * f2 * f3 -
            d1 * f12 * f23 * f3 + d2 * f12 * f23 * f3 +
            d1 * f1 * f24 * f3 - d2 * f1 * f24 * f3 - d2 * f14 * f32 +
            3.0 * d1 * f13 * f2 * f32 +
            d2 * f13 * f2 * f32 - d1 * f1 * f23 * f32 + d2 * f1 * f23 * f32 -
            2.0 * d1 * f24 * f32 - d2 * f24 * f32 - 2.0 * d1 * f13 * f33 +
            2.0 * d2 * f13 * f33 - d1 * f12 * f2 * f33 - 3.0 * d2 * f12 * f2 * f33 +
            3.0 * d1 * f23 * f33 +
            d2 * f23 * f33 +
            d1 * f12 * f34 - d1 * f1 * f2 * f34 + d1 * f1 * f35 - d1 * f2 * f35 +
            4.0 * f12 * f23 * v1 - 3.0 * f1 * f24 * v1 + 4.0 * f1 * f23 * f3 * v1 -
            3.0 * f24 * f3 * v1 - 12.0 * f12 * f2 * f32 * v1 +
            4.0 * f23 * f32 * v1 +
            8.0 * f12 * f33 * v1 - f1 * f34 * v1 - f35 * v1 - f15 * v2 - f14 * f3 * v2 +
            8.0 * f13 * f32 * v2 - 8.0 * f12 * f33 * v2 +
            f1 * f34 * v2 +
            f35 * v2 +
            f15 * v3 - 4.0 * f12 * f23 * v3 + 3.0 * f1 * f24 * v3 +
            f14 * f3 * v3 - 4.0 * f1 * f23 * f3 * v3 + 3.0 * f24 * f3 * v3 -
            8.0 * f13 * f32 * v3 + 12.0 * f12 * f2 * f32 * v3 - 4.0 * f23 * f32 * v3
        ) / (
            (f1 - f2) *
            (f1 - f2) *
            (f1 - f3) *
            (f1 - f3) *
            (f1 - f3) *
            (f3 - f2) *
            (f3 - f2)
        )
    )
    delta3 = -(
        (
            -2.0 * d2 * f14 * f2 + d1 * f13 * f22 + 3.0 * d2 * f13 * f22 - d1 * f1 * f24 -
            d2 * f1 * f24 + 2.0 * d2 * f14 * f3 - 2.0 * d1 * f13 * f2 * f3 -
            2.0 * d2 * f13 * f2 * f3 + d1 * f12 * f22 * f3 - d2 * f12 * f22 * f3 +
            d1 * f24 * f3 +
            d2 * f24 * f3 +
            d1 * f13 * f32 - d2 * f13 * f32 - 2.0 * d1 * f12 * f2 * f32 +
            2.0 * d2 * f12 * f2 * f32 +
            d1 * f1 * f22 * f32 - d2 * f1 * f22 * f32 + d1 * f12 * f33 -
            d2 * f12 * f33 +
            2.0 * d1 * f1 * f2 * f33 +
            2.0 * d2 * f1 * f2 * f33 - 3.0 * d1 * f22 * f33 - d2 * f22 * f33 -
            2.0 * d1 * f1 * f34 + 2.0 * d1 * f2 * f34 - 4.0 * f12 * f22 * v1 +
            2.0 * f24 * v1 +
            8.0 * f12 * f2 * f3 * v1 - 4.0 * f1 * f22 * f3 * v1 - 4.0 * f12 * f32 * v1 +
            8.0 * f1 * f2 * f32 * v1 - 4.0 * f22 * f32 * v1 - 4.0 * f1 * f33 * v1 +
            2.0 * f34 * v1 +
            2.0 * f14 * v2 - 4.0 * f13 * f3 * v2 + 4.0 * f1 * f33 * v2 - 2.0 * f34 * v2 -
            2.0 * f14 * v3 + 4.0 * f12 * f22 * v3 - 2.0 * f24 * v3 + 4.0 * f13 * f3 * v3 -
            8.0 * f12 * f2 * f3 * v3 +
            4.0 * f1 * f22 * f3 * v3 +
            4.0 * f12 * f32 * v3 - 8.0 * f1 * f2 * f32 * v3 + 4.0 * f22 * f32 * v3
        ) / (
            (f1 - f2) *
            (f1 - f2) *
            (f1 - f3) *
            (f1 - f3) *
            (f1 - f3) *
            (f3 - f2) *
            (f3 - f2)
        )
    )
    delta4 = -(
        (
            d2 * f13 * f2 - d1 * f12 * f22 - 2.0 * d2 * f12 * f22 +
            d1 * f1 * f23 +
            d2 * f1 * f23 - d2 * f13 * f3 +
            2.0 * d1 * f12 * f2 * f3 +
            d2 * f12 * f2 * f3 - d1 * f1 * f22 * f3 + d2 * f1 * f22 * f3 -
            d1 * f23 * f3 - d2 * f23 * f3 - d1 * f12 * f32 + d2 * f12 * f32 -
            d1 * f1 * f2 * f32 - 2.0 * d2 * f1 * f2 * f32 +
            2.0 * d1 * f22 * f32 +
            d2 * f22 * f32 +
            d1 * f1 * f33 - d1 * f2 * f33 + 3.0 * f1 * f22 * v1 - 2.0 * f23 * v1 -
            6.0 * f1 * f2 * f3 * v1 +
            3.0 * f22 * f3 * v1 +
            3.0 * f1 * f32 * v1 - f33 * v1 - f13 * v2 + 3.0 * f12 * f3 * v2 -
            3.0 * f1 * f32 * v2 +
            f33 * v2 +
            f13 * v3 - 3.0 * f1 * f22 * v3 + 2.0 * f23 * v3 - 3.0 * f12 * f3 * v3 +
            6.0 * f1 * f2 * f3 * v3 - 3.0 * f22 * f3 * v3
        ) / (
            (f1 - f2) *
            (f1 - f2) *
            (f1 - f3) *
            (f1 - f3) *
            (f1 - f3) *
            (f3 - f2) *
            (f3 - f2)
        )
    )

    Overallamp =
        2.0 * sqrt(5.0 / (64.0 * pi)) * M * GMsun_over_c2_Gpc * M * GMsun_over_c3 / dL

    return (
        T = T, M = M, GMsun_over_c3 = GMsun_over_c3, fInsJoin_Ampl = fInsJoin_Ampl,
        fpeak = fpeak, fcutPar = fcutPar, Acoeffs = Acoeffs,
        delta0 = delta0, delta1 = delta1, delta2 = delta2, delta3 = delta3, delta4 = delta4,
        Overallamp = Overallamp, amp0 = amp0,
        fring = fring, fdamp = fdamp, gamma1 = gamma1, gamma2 = gamma2, gamma3 = gamma3,
    )
end # @Tlit T
end

"""
    _ampl_scalar(coeffs, f)

Non-allocating IMRPhenomD amplitude evaluation at a single frequency `f` (not a vector), using the
frequency-independent coefficients from [`_ampl_coeffs`](@ref). This is the scalar core that
`Ampl` broadcasts over an `AbstractVector` of frequencies (preserving the existing array-in/
array-out API for all current callers), and that the fused KernelAbstractions likelihood kernel
calls directly in a per-thread frequency loop with zero array allocation.
"""
@inline function _ampl_scalar(coeffs, f)
    T = coeffs.T
@Tlit T begin
    f_safe = max(f, eps(T))
    fgrid = coeffs.M * coeffs.GMsun_over_c3 * f_safe

    amplitudeIMR = ifelse(
        fgrid < coeffs.fInsJoin_Ampl,
        1.0 +
        (fgrid^(2.0 / 3.0)) * coeffs.Acoeffs.two_thirds +
        (fgrid^(4.0 / 3.0)) * coeffs.Acoeffs.four_thirds +
        (fgrid^(5.0 / 3.0)) * coeffs.Acoeffs.five_thirds +
        (fgrid^(7.0 / 3.0)) * coeffs.Acoeffs.seven_thirds +
        (fgrid^(8.0 / 3.0)) * coeffs.Acoeffs.eight_thirds +
        fgrid * (coeffs.Acoeffs.one + fgrid * coeffs.Acoeffs.two + fgrid * fgrid * coeffs.Acoeffs.three),
        ifelse(
            fgrid < coeffs.fpeak,
            coeffs.delta0 +
            fgrid * coeffs.delta1 +
            fgrid * fgrid * (coeffs.delta2 + fgrid * coeffs.delta3 + fgrid * fgrid * coeffs.delta4),
            ifelse(
                fgrid < coeffs.fcutPar,
                exp(-(fgrid - coeffs.fring) * coeffs.gamma2 / (coeffs.fdamp * coeffs.gamma3)) *
                (coeffs.fdamp * coeffs.gamma3 * coeffs.gamma1) /
                ((fgrid - coeffs.fring) * (fgrid - coeffs.fring) + coeffs.fdamp * coeffs.gamma3 * coeffs.fdamp * coeffs.gamma3),
                zero(T),
            ),
        ),
    )
    return coeffs.Overallamp * coeffs.amp0 * (fgrid^(-7.0 / 6.0)) * amplitudeIMR
end # @Tlit T
end

"""
    Ampl(model::IMRPhenomD, f, mc, eta, chi1, chi2, dL; kwargs...)

IMRPhenomD frequency-domain amplitude (SPA-style normalization; the (2,2)-mode-only,
face-on convention used by `gwfast`/this comparison harness, not LAL's full-polarization
`hplus`/`hcross`). `f` may be a scalar (returns a scalar, non-allocating) or an
`AbstractVector` (returns a same-length `Vector`) -- computes the frequency-independent
coefficients once via [`_ampl_coeffs`](@ref), then evaluates [`_ampl_scalar`](@ref) at each
frequency. Callers evaluating many frequencies for one set of parameters where per-call
allocation matters (e.g. inside a GPU kernel) should call `_ampl_coeffs` once and `_ampl_scalar`
per point directly instead of going through this array-oriented wrapper.

Implementation note: this deliberately uses an explicit `similar`-then-loop instead of
broadcasting `_ampl_scalar` over `f` (e.g. `_ampl_scalar.(Ref(coeffs), f)`). The broadcast form
is numerically identical but confuses Enzyme's static activity analysis when this array-out
`Ampl` is itself a step inside a larger differentiated call graph (observed as an
`EnzymeRuntimeActivityError` — "Mismatched activity" — when GravitationalWaveBayes.jl differentiated a batched
generator built on it); the plain loop does not trigger it.

# Units
- `f`: geometric frequency (same convention as `mc`; internally rescaled to `M*f` via
  `GMsun_over_c3`).
- `mc`: chirp mass in solar masses.
- `dL`: luminosity distance **in Gpc**, not Mpc. `Overallamp` multiplies by
  `GMsun_over_c2_Gpc` (`G*Msun/c^2` expressed in Gpc) and divides by `dL`, so `dL` must
  already be in Gpc for those units to cancel correctly — passing a value in Mpc would be
  off by a factor of 1000. This was verified against LALSuite's reference `IMRPhenomD` at
  `dL = 0.4 Gpc`.

No parameter-domain assertions run here — this is the hot path. See `validate_phenomd_params`
for an opt-in host-side check.
"""
function Ampl(model::IMRPhenomD,
    f,
    mc,
    eta,
    chi1,
    chi2,
    dL;
    fcutPar = 0.2,
    fInsJoin_Ampl = 0.014,
    GMsun_over_c3 = GMsun_over_c3,
    GMsun_over_c2_Gpc = GMsun_over_c2_Gpc,
)
    coeffs = _ampl_coeffs(model, mc, eta, chi1, chi2, dL;
        fcutPar = fcutPar, fInsJoin_Ampl = fInsJoin_Ampl,
        GMsun_over_c3 = GMsun_over_c3, GMsun_over_c2_Gpc = GMsun_over_c2_Gpc)
    if f isa AbstractArray
        out = similar(f, coeffs.T)
        @inbounds for i in eachindex(f)
            out[i] = _ampl_scalar(coeffs, f[i])
        end
        return out
    else
        return _ampl_scalar(coeffs, f)
    end
end
