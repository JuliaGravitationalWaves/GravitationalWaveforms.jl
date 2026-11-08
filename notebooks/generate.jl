using Literate

# Regenerates every notebooks/*.ipynb from its docs/literate/*.jl source.
# Run with: julia --project=docs notebooks/generate.jl
# (Notebooks ARE committed, unlike docs/src/generated/.)

const ROOT = joinpath(@__DIR__, "..")
const LITERATE_DIR = joinpath(ROOT, "docs", "literate")
const NOTEBOOK_DIR = @__DIR__

for file in sort(readdir(LITERATE_DIR))
    endswith(file, ".jl") || continue
    Literate.notebook(joinpath(LITERATE_DIR, file), NOTEBOOK_DIR; execute=false)
end
