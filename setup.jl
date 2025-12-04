using Pkg

manifest_paths = [
    "PlutoStartup.jl"
]

for path in manifest_paths
    base_path = joinpath(@__DIR__, path)
    if !isfile(joinpath(base_path, "Manifest.toml"))
        Pkg.activate(base_path)
        Pkg.instantiate()
        Pkg.precompile()
    end
end

mkpath(joinpath(@__DIR__, "setup_complete"))
