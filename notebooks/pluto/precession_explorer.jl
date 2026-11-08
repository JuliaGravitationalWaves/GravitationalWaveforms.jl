### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 6b2f0001-0001-4001-8002-000000000001
begin
    import Pkg
    Pkg.activate(@__DIR__)
end

# ╔═╡ 6b2f0001-0001-4001-8002-000000000002
using GravitationalWaveforms, Plots, PlutoUI

# ╔═╡ 6b2f0001-0001-4001-8002-000000000003
md"""
# Precession Explorer

`IMRPhenomXP` twists up the aligned-spin `(2,2)` mode using the component of each spin lying
IN the orbital plane, not just the aligned (`z`) component `IMRPhenomD`/`IMRPhenomXAS` see. Tilt
the primary spin away from the orbital angular momentum and watch the precession-induced
amplitude/phase modulation appear against the non-precessing (`tilt = 0`) baseline. Mirrors
`docs/literate/04_precession.jl`'s static tutorial, parametrized here as (magnitude, tilt,
azimuth) instead of raw `(chi1x, chi1y, chi1z)` — the physically intuitive knobs for "how much
precession."
"""

# ╔═╡ 6b2f0001-0001-4001-8002-000000000004
gr()

# ╔═╡ 6b2f0001-0001-4001-8002-000000000005
md"""
## System (GW150914-like masses, non-spinning secondary)
"""

# ╔═╡ 6b2f0001-0001-4001-8002-000000000006
@bind chi1_mag Slider(0.0:0.05:0.99; default=0.7, show_value=true)

# ╔═╡ 6b2f0001-0001-4001-8002-000000000007
@bind tilt1_deg Slider(0.0:5.0:90.0; default=45.0, show_value=true)

# ╔═╡ 6b2f0001-0001-4001-8002-000000000008
@bind phi1_deg Slider(0.0:10.0:350.0; default=0.0, show_value=true)

# ╔═╡ 6b2f0001-0001-4001-8002-000000000009
@bind inclination_deg Slider(0.0:5.0:180.0; default=60.0, show_value=true)

# ╔═╡ 6b2f0001-0001-4001-8002-00000000000a
md"""
`chi1_mag` — dimensionless spin magnitude of the primary. `tilt1_deg` — angle between that spin
and the orbital angular momentum at `f_ref` (0° = aligned, no precession; 90° = spin lies entirely
in the orbital plane, maximal precession at fixed magnitude). `phi1_deg` — azimuthal orientation
of the in-plane component. `inclination_deg` — line-of-sight inclination (`theta_jn` at `f_ref`).
"""

# ╔═╡ 6b2f0001-0001-4001-8002-00000000000b
begin
    m1, m2 = 36.0, 29.0
    dL = 0.41
    f_ref = 20.0
    phiRef = 0.0
    freqs = collect(20.0:0.5:512.0)
    inclination = inclination_deg * pi / 180

    tilt1 = tilt1_deg * pi / 180
    phi1 = phi1_deg * pi / 180
    chi1x = chi1_mag * sin(tilt1) * cos(phi1)
    chi1y = chi1_mag * sin(tilt1) * sin(phi1)
    chi1z = chi1_mag * cos(tilt1)
    chi2x, chi2y, chi2z = 0.0, 0.0, 0.0

    hp_prec, hc_prec = twist_up22(IMRPhenomXP(), freqs, m1, m2, chi1x, chi1y, chi1z,
        chi2x, chi2y, chi2z, f_ref, inclination, phiRef, dL)
    hp_aligned, hc_aligned = twist_up22(IMRPhenomXP(), freqs, m1, m2, 0.0, 0.0, chi1z,
        0.0, 0.0, 0.0, f_ref, inclination, phiRef, dL)
    nothing
end

# ╔═╡ 6b2f0001-0001-4001-8002-00000000000c
plot(freqs, [abs.(hp_prec) abs.(hp_aligned)]; xscale=:log10, yscale=:log10,
    label=["precessing (tilt=$(tilt1_deg)°)" "aligned-spin baseline (same chi1z, tilt=0°)"],
    xlabel="frequency (Hz)", ylabel="|h+|",
    title="Precession-induced amplitude modulation")

# ╔═╡ 6b2f0001-0001-4001-8002-00000000000d
md"""
The two curves share the same aligned-spin component `chi1z = chi1_mag*cos(tilt1)`; the only
difference is whether the remaining spin magnitude sits in-plane (precessing) or is absent
entirely (aligned baseline). At `tilt1 = 0°` the two curves are identical, by construction. As
`tilt1` grows, the precessing curve picks up the modulation `twist_up22` applies via `_xp_apply_zeta_rotation`
(see the Precession tutorial for the full derivation) — most visible as a slowly-varying ripple on
top of the same underlying chirp, strongest for large in-plane spin (`tilt1` near 90°) and large
inclination (edge-on systems see more of the orbital-plane geometry).
"""

# ╔═╡ Cell order:
# ╟─6b2f0001-0001-4001-8002-000000000003
# ╠═6b2f0001-0001-4001-8002-000000000001
# ╠═6b2f0001-0001-4001-8002-000000000002
# ╠═6b2f0001-0001-4001-8002-000000000004
# ╟─6b2f0001-0001-4001-8002-000000000005
# ╟─6b2f0001-0001-4001-8002-000000000006
# ╟─6b2f0001-0001-4001-8002-000000000007
# ╟─6b2f0001-0001-4001-8002-000000000008
# ╟─6b2f0001-0001-4001-8002-000000000009
# ╟─6b2f0001-0001-4001-8002-00000000000a
# ╠═6b2f0001-0001-4001-8002-00000000000b
# ╠═6b2f0001-0001-4001-8002-00000000000c
# ╟─6b2f0001-0001-4001-8002-00000000000d
