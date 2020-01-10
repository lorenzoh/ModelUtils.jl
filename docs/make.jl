using Documenter
using ModelUtils


makedocs(
    sitename="ModelUtils.jl documentation",
    format=Documenter.HTML(),
    modules=[ModelUtils]
)

deploydocs(
    repo="github.com/lorenzoh/ModelUtils.jl.git"
)
