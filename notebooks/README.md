# GravitationalWaveforms.jl notebooks

- `*.ipynb` — Jupyter notebooks, one per docs tutorial, generated from `docs/literate/*.jl` via
  `julia --project=docs generate.jl`. Do not hand-edit; edit the Literate source under
  `docs/literate/` instead and regenerate. Requires an `IJulia` kernel to run (not a dependency of
  this package itself — install it in your own global environment: `using Pkg; Pkg.add("IJulia")`).
- `pluto/*.jl` — standalone interactive [Pluto.jl](https://plutojl.org/) notebooks (not generated;
  edited directly). Open with `using Pluto; Pluto.run()` then open the file.
