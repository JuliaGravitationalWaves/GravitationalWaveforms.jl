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

# ╔═╡ 6a1f0001-0001-4001-8001-000000000001
begin
    import Pkg
    Pkg.activate(@__DIR__)
end

# ╔═╡ 6a1f0001-0001-4001-8001-000000000002
using GravitationalWaveforms, Plots, PlutoUI

# ╔═╡ 6a1f0001-0001-4001-8001-000000000003
md"""
# Waveform Explorer

Pick a model, drag the sliders, watch the frequency-domain amplitude/phase (or `h+` for the
higher-mode/precessing models, which don't have a single scalar amplitude) update live. Mirrors
`docs/literate/08_waveform_gallery_capstone.jl`'s static side-by-side comparison, but lets you
move continuously through parameter space on one model at a time instead of a single fixed
GW150914-like system.
"""

# ╔═╡ 6a1f0001-0001-4001-8001-000000000004
gr()

# ╔═╡ 6a1f0001-0001-4001-8001-000000000005
md"""
## Model
"""

# ╔═╡ 6a1f0001-0001-4001-8001-000000000006
@bind model_name Select(["IMRPhenomD", "IMRPhenomXAS", "IMRPhenomXHM", "IMRPhenomXP", "IMRPhenomXPHM"]; default="IMRPhenomXAS")

# ╔═╡ 6a1f0001-0001-4001-8001-000000000007
md"""
## Masses and aligned spin
"""

# ╔═╡ 6a1f0001-0001-4001-8001-000000000008
@bind m1 Slider(5.0:1.0:80.0; default=36.0, show_value=true)

# ╔═╡ 6a1f0001-0001-4001-8001-000000000009
@bind m2 Slider(5.0:1.0:80.0; default=29.0, show_value=true)

# ╔═╡ 6a1f0001-0001-4001-8001-00000000000a
@bind chi1 Slider(-0.99:0.01:0.99; default=0.0, show_value=true)

# ╔═╡ 6a1f0001-0001-4001-8001-00000000000b
@bind chi2 Slider(-0.99:0.01:0.99; default=0.0, show_value=true)

# ╔═╡ 6a1f0001-0001-4001-8001-00000000000c
md"""
## Extrinsic parameters
"""

# ╔═╡ 6a1f0001-0001-4001-8001-00000000000d
@bind dL Slider(0.1:0.05:5.0; default=0.41, show_value=true)

# ╔═╡ 6a1f0001-0001-4001-8001-00000000000e
@bind inclination Slider(0.0:(pi/32):pi; default=pi/4, show_value=true)

# ╔═╡ 6a1f0001-0001-4001-8001-00000000000f
md"""
`m2` is clamped to `min(m1, m2)` below so `m1 >= m2` always holds (several models assert this).
For the precessing models (`IMRPhenomXP`/`IMRPhenomXPHM`) the sliders above set the ALIGNED
(`z`) spin component; the in-plane components are fixed at a modest, illustrative
`(chi1x, chi1y) = (0.3, 0.0)`, `(chi2x, chi2y) = (0.0, 0.0)` so the explorer stays to a
manageable number of sliders — see the Precession tutorial for a dedicated in-plane-spin
explorer.
"""

# ╔═╡ 6a1f0001-0001-4001-8001-000000000010
begin
    m1c, m2c = max(m1, m2), min(m1, m2)
    mc = (m1c * m2c)^(3 / 5) / (m1c + m2c)^(1 / 5)
    eta = (m1c * m2c) / (m1c + m2c)^2
    f_ref = 20.0
    phiRef = 0.0
    freqs = collect(20.0:0.5:512.0)
    nothing
end

# ╔═╡ 6a1f0001-0001-4001-8001-000000000011
begin
    if model_name == "IMRPhenomD"
        amp = Ampl(IMRPhenomD(), freqs, mc, eta, chi1, chi2, dL)
        curve_label = "|h| (IMRPhenomD, (2,2) only)"
        curve = amp
    elseif model_name == "IMRPhenomXAS"
        amp = Ampl(IMRPhenomXAS(), freqs, mc, eta, chi1, chi2, dL)
        curve_label = "|h| (IMRPhenomXAS, (2,2) only)"
        curve = amp
    elseif model_name == "IMRPhenomXHM"
        hp, hc = hphc(IMRPhenomXHM(), freqs, m1c, m2c, chi1, chi2, dL, inclination, phiRef; f_ref=f_ref)
        curve_label = "|h+| (IMRPhenomXHM, all modes)"
        curve = abs.(hp)
    elseif model_name == "IMRPhenomXP"
        hp, hc = twist_up22(IMRPhenomXP(), freqs, m1c, m2c, 0.3, 0.0, chi1, 0.0, 0.0, chi2,
            f_ref, inclination, phiRef, dL)
        curve_label = "|h+| (IMRPhenomXP, precessing (2,2))"
        curve = abs.(hp)
    else # IMRPhenomXPHM
        hp, hc = hphc(IMRPhenomXPHM(), freqs, m1c, m2c, 0.3, 0.0, chi1, 0.0, 0.0, chi2,
            f_ref, inclination, phiRef, dL)
        curve_label = "|h+| (IMRPhenomXPHM, precessing + HM)"
        curve = abs.(hp)
    end
    plot(freqs, curve; xscale=:log10, yscale=:log10, label=curve_label,
        xlabel="frequency (Hz)", ylabel="amplitude",
        title="$(model_name): m1=$(round(m1c,digits=1)), m2=$(round(m2c,digits=1)), " *
              "chi1=$(chi1), chi2=$(chi2), dL=$(dL) Gpc")
end

# ╔═╡ Cell order:
# ╟─6a1f0001-0001-4001-8001-000000000003
# ╠═6a1f0001-0001-4001-8001-000000000001
# ╠═6a1f0001-0001-4001-8001-000000000002
# ╠═6a1f0001-0001-4001-8001-000000000004
# ╟─6a1f0001-0001-4001-8001-000000000005
# ╟─6a1f0001-0001-4001-8001-000000000006
# ╟─6a1f0001-0001-4001-8001-000000000007
# ╟─6a1f0001-0001-4001-8001-000000000008
# ╟─6a1f0001-0001-4001-8001-000000000009
# ╟─6a1f0001-0001-4001-8001-00000000000a
# ╟─6a1f0001-0001-4001-8001-00000000000b
# ╟─6a1f0001-0001-4001-8001-00000000000c
# ╟─6a1f0001-0001-4001-8001-00000000000d
# ╟─6a1f0001-0001-4001-8001-00000000000e
# ╟─6a1f0001-0001-4001-8001-00000000000f
# ╠═6a1f0001-0001-4001-8001-000000000010
# ╠═6a1f0001-0001-4001-8001-000000000011
