using Documenter
using Literate
using GravitationalWaveforms

const DOCS_SRC = joinpath(@__DIR__, "src")
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const GENERATED_DIR = joinpath(DOCS_SRC, "generated")

# Weave every docs/literate/*.jl tutorial into docs/src/generated/*.md (build artifact,
# gitignored, mirroring GravitationalWaveData.jl's own docs setup). Ordered explicitly (rather than a directory listing) so the sidebar order in `pages`
# below matches; add new topics to BOTH this list and `pages`.
const LITERATE_TOPICS = [
    "01_getting_started",
    "02_frequency_domain_models",
    "03_higher_order_modes",
    "04_precession",
    "05_imrphenomxphm",
    "06_time_domain_taylort3",
    "07_odetoy_eob",
    "08_waveform_gallery_capstone",
    "09_collocation_fitting",
]

# docs/CONVENTIONS.md (repo root of the docs/ tree) is the single source of truth; copy it in
# verbatim rather than maintaining a second copy under docs/src/ — mirrors GravitationalWaveData.jl's own
# docs/make.jl pattern exactly.
cp(joinpath(@__DIR__, "CONVENTIONS.md"), joinpath(DOCS_SRC, "conventions.md"); force=true)

mkpath(GENERATED_DIR)
for topic in LITERATE_TOPICS
    src = joinpath(LITERATE_DIR, "$(topic).jl")
    isfile(src) || continue  # tolerate topics not yet written during incremental rollout
    Literate.markdown(src, GENERATED_DIR; documenter=true)
end

generated_pages = [
    "Getting Started" => "generated/01_getting_started.md",
    "Frequency-Domain Models: D, XAS" => "generated/02_frequency_domain_models.md",
    "Higher-Order Modes: XHM" => "generated/03_higher_order_modes.md",
    "Precession: XP" => "generated/04_precession.md",
    "Precessing + HM: XPHM" => "generated/05_imrphenomxphm.md",
    "Time-Domain PN: TaylorT3" => "generated/06_time_domain_taylort3.md",
    "ODE-Integrated Toy Model" => "generated/07_odetoy_eob.md",
    "Capstone: Waveform Gallery" => "generated/08_waveform_gallery_capstone.md",
    "Collocation-Point Coefficient Fitting" => "generated/09_collocation_fitting.md",
]
# Only list pages that actually got generated, so `make.jl` stays runnable while incremental
# rollout is in progress.
generated_pages = filter(p -> isfile(joinpath(DOCS_SRC, p.second)), generated_pages)

DocMeta.setdocmeta!(GravitationalWaveforms, :DocTestSetup, :(using GravitationalWaveforms); recursive=true)

makedocs(;
    sitename="GravitationalWaveforms.jl",
    modules=[GravitationalWaveforms],
    authors="Paul Hopkins <paul.hopkins256@gmail.com>",
    format=Documenter.HTML(;
        canonical="https://juliagravitationalwaves.github.io/GravitationalWaveforms.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => generated_pages,
        "Conventions" => "conventions.md",
        "API Reference" => "api.md",
    ],
    checkdocs=:exports,
    warnonly=[:missing_docs, :cross_references],
)

deploydocs(;
    repo="github.com/JuliaGravitationalWaves/GravitationalWaveforms.jl",
    devbranch="main",
)
