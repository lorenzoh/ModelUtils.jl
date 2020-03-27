using ModelUtils
using Documenter

makedocs(;
    modules=[ModelUtils],
    authors="lorenzoh <lorenz.ohly@gmail.com>",
    repo="https://github.com/lorenzoh/ModelUtils.jl/blob/{commit}{path}#L{line}",
    sitename="ModelUtils.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://lorenzoh.github.io/ModelUtils.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/lorenzoh/ModelUtils.jl",
)
