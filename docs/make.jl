using Documenter
using Literate
using GravitationalWaveforms

const DOCS_SRC = joinpath(@__DIR__, "src")
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const GENERATED_DIR = joinpath(DOCS_SRC, "generated")

# Weave the docs/literate/*.jl tutorials into docs/src/generated/*.md (gitignored). Add new topics
# to both this list and `pages`.
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
    "10_seobnrv5",
]

# docs/CONVENTIONS.md is the single source of truth; copy it in rather than keeping a second copy.
cp(joinpath(@__DIR__, "CONVENTIONS.md"), joinpath(DOCS_SRC, "conventions.md"); force=true)

mkpath(GENERATED_DIR)
for topic in LITERATE_TOPICS
    Literate.markdown(joinpath(LITERATE_DIR, "$(topic).jl"), GENERATED_DIR; documenter=true)
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
    "Effective-One-Body: SEOBNRv5" => "generated/10_seobnrv5.md",
]

DocMeta.setdocmeta!(GravitationalWaveforms, :DocTestSetup, :(using GravitationalWaveforms); recursive=true)

makedocs(;
    sitename="GravitationalWaveforms.jl",
    modules=[GravitationalWaveforms],
    authors="Paul Hopkins <paul.hopkins256@gmail.com>",
    format=Documenter.HTML(;
        canonical="https://juliagravitationalwaves.github.io/GravitationalWaveforms.jl",
        edit_link="main",
        assets=String[],
        size_threshold_ignore=["api.md"],
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
