julia --startup-file no --threads auto -e 'using Pkg; !isdir(joinpath(@__DIR__, "setup_complete")) && include(joinpath(@__DIR__, "setup.jl")); Pkg.activate("PlutoStartup"); cd(joinpath(@__DIR__, "MultiAgentRL", "src", "notebooks")); using PlutoStartup'

