"""
ModelUtils.jl provides functionality for analyzing and manipulating
Flux.jl models.
"""
module ModelUtils

include("./model.jl")
include("./utils.jl")
include("./map.jl")
include("./hooks.jl")
include("./introspection.jl")
include("./attributes.jl")
include("./initialization.jl")
include("./plot.jl")

export Model,
    Hook,
    Initialization,
    IterLayers,
    IterModels,
    addhook,
    attrtype,
    backwardstatshook,
    children,
    forwardstatshook,
    gradientsonebatch,
    gradientsdict,
    hooksaveactivations,
    hooksavegrads,
    init!,
    initmodel!,
    init_kaiming_normal,
    init_kaiming_uniform,
    mapmodel,
    paramsdict,
    plotlayers,
    printmodel,
    removehooks
end # module
