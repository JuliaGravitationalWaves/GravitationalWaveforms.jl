using Test
using GravitationalWaveforms
using Aqua

# Test selection: `Pkg.test(test_args = ["seobnrv5phm", "basics_tests.jl"])` runs only the files whose names
# contain an arg (an arg ending in .jl must match exactly; "aqua" = the Aqua checks). Each file prints
# `[test-time] <file> <s>`. Parallel full runs: `julia +1.13.2 test/run_parallel.jl -j 3` (see there).
const SELECT = ARGS
selected(file) = isempty(SELECT) || any(a -> endswith(a, ".jl") ? a == file : occursin(a, file), SELECT)

function test_include(file)
    selected(file) || return
    t = @elapsed include(file)
    println("[test-time] $file $(round(t; digits=1))")
end

if selected("aqua")
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(GravitationalWaveforms)
    end
end

@testset "GravitationalWaveforms.jl" begin
    test_include("basics_tests.jl")
    test_include("xas_linsolve_enzyme_tests.jl")
    test_include("test_precession_frame.jl")
    test_include("test_twisting_up.jl")
    test_include("phenomxprecession_enzyme_ka_tests.jl")
    test_include("test_phenomxhm_qnm.jl")
    test_include("test_phenomxhm_inspiral.jl")
    test_include("test_phenomxhm_intermediate.jl")
    test_include("test_phenomxhm_ringdown.jl")
    test_include("test_phenomxhm_full.jl")
    test_include("test_phenomxphm.jl")
    test_include("test_phenomxp_msa.jl")
    test_include("taylort3_tests.jl")
    test_include("odetoy_tests.jl")
    test_include("odetoy_enzyme_ka_tests.jl")
    test_include("alignedspinpnham_tests.jl")
    test_include("alignedspinpnham_enzyme_ka_tests.jl")
    test_include("alignedspinpnhamf2_tests.jl")
    test_include("precessingspinpnham_tests.jl")
    test_include("precessingspinpnham_enzyme_ka_tests.jl")
    test_include("seobnrv5alignedspin_tests.jl")
    test_include("seobnrv5alignedspin_pa_tests.jl")
    test_include("seobnrv5alignedspin_enzyme_ka_tests.jl")
    test_include("seobnrv5alignedspin_pa_fixed_tests.jl")
    test_include("seobnrv5alignedspinf2_tests.jl")
    test_include("dense_to_freq_edge_tests.jl")
    test_include("seobnrv5alignedspinf2_enzyme_ka_tests.jl")
    test_include("seobnrv5alignedspinf2_ka_cuda_tests.jl")
    test_include("seobnrv5hm_tests.jl")
    test_include("seobnrv5phm_pn_tests.jl")
    test_include("seobnrv5phm_dyn_tests.jl")
    test_include("seobnrv5phm_wf_tests.jl")
    test_include("seobnrv5phm_fd_tests.jl")
    test_include("seobnrv5_ham_symbolic_tests.jl")
end
