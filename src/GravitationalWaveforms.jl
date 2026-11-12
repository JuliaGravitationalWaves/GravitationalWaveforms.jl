module GravitationalWaveforms

# Dependency order matters: TypeGenericity.jl defines the `@Tlit` macro that both IMRPhenomD.jl and
# IMRPhenomXAS.jl rely on for Float32/Float64 genericity, so it must be included first.
include("GravitationalWaveforms/TypeGenericity.jl")
include("GravitationalWaveforms/TidalUtils.jl")
include("GravitationalWaveforms/IMRPhenomD.jl")
include("GravitationalWaveforms/IMRPhenomXAS.jl")
include("GravitationalWaveforms/IMRPhenomXHM_qnm.jl")
include("GravitationalWaveforms/IMRPhenomXHM_inspiral.jl")
include("GravitationalWaveforms/IMRPhenomXHM_intermediate.jl")
include("GravitationalWaveforms/IMRPhenomXHM_ringdown.jl")
include("GravitationalWaveforms/IMRPhenomXPrecession.jl")
include("GravitationalWaveforms/IMRPhenomXPrecessionMSA.jl")
include("GravitationalWaveforms/IMRPhenomXHM.jl")
# IMRPhenomXPHM.jl combines IMRPhenomXPrecession.jl's NNLO Euler angles with IMRPhenomXHM.jl's
# aligned-spin mode content, so it must come after both.
include("GravitationalWaveforms/IMRPhenomXPHM.jl")
include("GravitationalWaveforms/TaylorT3.jl")
include("GravitationalWaveforms/TaylorF2.jl")
include("GravitationalWaveforms/SineGaussian.jl")
include("GravitationalWaveforms/ODEToy.jl")
include("GravitationalWaveforms/AlignedSpinPNHam.jl")
include("GravitationalWaveforms/AlignedSpinPNHamF2.jl")
include("GravitationalWaveforms/PrecessingSpinPNHam.jl")
include("GravitationalWaveforms/SEOBNRv5HamParams.jl")
include("GravitationalWaveforms/SEOBNRv5AlignedSpin.jl")
include("GravitationalWaveforms/SEOBNRv5_flux_extra.jl")
include("GravitationalWaveforms/SEOBNRv5AlignedSpinPA.jl")
include("GravitationalWaveforms/SEOBNRv5AlignedSpinPAFixed.jl")
include("GravitationalWaveforms/SEOBNRv5AlignedSpinF2.jl")
# Shared aligned-spin EOB IMR pipeline (PA inspiral + RK4 plunge, NQC, merger-ringdown, SPA + FFT tables).
include("GravitationalWaveforms/EOBIMR.jl")
# The shared IMR pipeline driven by the SEOBNRv5 dynamics.
include("GravitationalWaveforms/SEOBNRv5IMRAlignedSpinF2.jl")
include("GravitationalWaveforms/SEOBNRv5HM_fits.jl")
include("GravitationalWaveforms/SEOBNRv5HM_qnm.jl")
include("GravitationalWaveforms/SEOBNRv5HM.jl")
include("GravitationalWaveforms/SEOBNRv5PHM_pn.jl")
include("GravitationalWaveforms/SEOBNRv5_ham_symbolic_generated.jl")
include("GravitationalWaveforms/SEOBNRv5_ham_symbolic.jl")
include("GravitationalWaveforms/SEOBNRv5PHM_dynamics.jl")
include("GravitationalWaveforms/SEOBNRv5PHM.jl")
include("GravitationalWaveforms/SEOBNRv5PHM_asym.jl")
include("GravitationalWaveforms/SEOBNRv5PHM_fd.jl")
include("GravitationalWaveforms/GravitationalWaveDataInterop.jl")

export IMRPhenomD, IMRPhenomXAS, IMRPhenomXP, IMRPhenomXHM, IMRPhenomXPHM, TaylorT3, TaylorF2,
    SineGaussian, ODEToy, ODEToyState, AlignedSpinPNHam, AlignedSpinPNHamState,
    AlignedSpinPNHamF2,
    PrecessingSpinPNHam, PrecessingSpinPNHamState, SEOBNRv5AlignedSpin, SEOBNRv5AlignedSpinState,
    SEOBNRv5AlignedSpinF2, SEOBNRv5IMRAlignedSpinF2,
    Phi, Ampl, evolve,
    twist_up22, h_lm, hphc, hphc_sine_gaussian, frequency_series, time_series,
    validate_phenomd_params, validate_phenomxas_params, validate_phenomxp_precession_params,
    validate_taylort3_params, validate_taylorf2_params, validate_odetoy_params,
    validate_alignedspinpnham_params, validate_alignedspinpnhamf2_params,
    validate_precessingspinpnham_params,
    validate_seobnrv5alignedspin_params,
    odetoy_rk4_integrate, odetoy_omega_analytic, alignedspinpnham_rk4_integrate,
    precessingspinpnham_rk4_integrate, seobnrv5alignedspin_rk4_integrate

end # module GravitationalWaveforms
