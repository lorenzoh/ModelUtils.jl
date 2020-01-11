using Documenter
using ModelUtils
using Flux

makedocs(
    sitename="ModelUtils.jl documentation",
    format=Documenter.HTML(),
    modules=[ModelUtils]
)

deploydocs(
    repo="github.com/lorenzoh/ModelUtils.jl.git"
)
