# Parallel test run: the runtests.jl files split over N julia processes, each `Pkg.test(test_args = <its files>)`.
#   julia +1.13.2 test/run_parallel.jl [-j N] [--slow] [filter ...]
# -j N     processes (default 3: ~2-3 GB each, 14 GB laptop); --slow adds the slow_include files; filters select files
#          as in runtests.jl (substring, or exact name if it ends in .jl). Files are balanced over the processes with
#          test/timings.toml (seconds per file, measured: rewritten from every run's `[test-time]` lines; new files 120 s).
# Logs: <tempdir>/<package>_tests/w<i>.log. Exit code 1 if any process failed. (Same file in GravitationalWaveforms and GravitationalWaveBayes.)
using TOML

const TESTDIR = @__DIR__
const PKG = dirname(TESTDIR)
nproc, slow, filters = 3, false, String[]
let i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "-j"; global nproc = parse(Int, ARGS[i += 1])
        elseif a == "--slow"; global slow = true
        else push!(filters, a) end
        i += 1
    end
end
matches(file) = isempty(filters) || any(a -> endswith(a, ".jl") ? a == file : occursin(a, file), filters)

files = ["aqua"]
for line in eachline(joinpath(TESTDIR, "runtests.jl"))
    m = match(r"^\s*(test_include|slow_include)\(\"([^\"]+)\"", line)
    m === nothing && continue
    (m[1] == "test_include" || slow) && push!(files, m[2])
end
filter!(matches, files)
isempty(files) && error("no test files match $(filters)")

tfile = joinpath(TESTDIR, "timings.toml")
timings = isfile(tfile) ? TOML.parsefile(tfile) : Dict{String,Any}()
est(f) = Float64(get(timings, f, 120.0))
# longest-processing-time-first: each file to the currently lightest process
nproc = min(nproc, length(files))
buckets = [String[] for _ in 1:nproc]; load = zeros(nproc)
for f in sort(files; by = est, rev = true)
    k = argmin(load); push!(buckets[k], f); load[k] += est(f)
end

logdir = mkpath(joinpath(tempdir(), TOML.parsefile(joinpath(PKG, "Project.toml"))["name"] * "_tests"))
println("$(length(files)) files on $nproc processes (estimated $(round(Int, maximum(load) / 60)) min); logs in $logdir")
t0 = time()
testcmd(args) = `$(Base.julia_cmd()) --startup-file=no --project=$PKG -e "using Pkg; Pkg.test(test_args = ARGS)" $args`
# warm-up: one Pkg.test selecting no file precompiles the test environment, so the processes do not race for the same
# cache files (on Windows that fails with "Error opening package file ... .dll")
success(pipeline(testcmd(["__warmup__"]); stdout = joinpath(logdir, "warmup.log"), stderr = joinpath(logdir, "warmup.log"))) ||
    (println("warm-up failed: ", joinpath(logdir, "warmup.log")); exit(1))
println("warm-up done after $(round(Int, (time() - t0) / 60)) min")
ok = fill(false, nproc)
@sync for k in 1:nproc
    @async begin
        sleep(15 * (k - 1))
        cmd = testcmd(buckets[k])
        ok[k] = success(pipeline(cmd; stdout = joinpath(logdir, "w$k.log"), stderr = joinpath(logdir, "w$k.log")))
        println("process $k: $(ok[k] ? "passed" : "FAILED") after $(round(Int, (time() - t0) / 60)) min ($(join(buckets[k], ", ")))")
    end
end

for k in 1:nproc, line in eachline(joinpath(logdir, "w$k.log"))
    m = match(r"^\[test-time\] (\S+) ([0-9.]+)$", line)
    m === nothing || (timings[m[1]] = parse(Float64, m[2]))
end
open(io -> TOML.print(io, timings; sorted = true), tfile, "w")
for k in findall(!, ok)
    println("\n--- process $k log tail ($(joinpath(logdir, "w$k.log"))) ---")
    foreach(println, last(readlines(joinpath(logdir, "w$k.log")), 30))
end
println("wall $(round((time() - t0) / 60; digits = 1)) min: ", all(ok) ? "ALL PASSED" : "FAILURES in processes $(findall(!, ok))")
exit(all(ok) ? 0 : 1)
