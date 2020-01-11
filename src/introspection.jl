using AbstractTrees
using Flux
using Zygote
using Zygote: Grads

"""
    paramsdict(model::Model) -> Dict{Symbol, Array}

Return a Dict with mapping parameter name => parameter value.

Useful for inspecting the distribution of parameters.
"""
function paramsdict(model::Model; filterfn = _ -> true)
    ps = Dict()
    for child::Model in IterModels(model)
        if filterfn(child.layer)
            pslayer = Dict(field => getfield(child.layer, field) for field in child.paramattrs)
            for (name, p) in pslayer
                if !haskey(ps, name)
                    ps[name] = []
                end
                push!(ps[name], p)
            end
        end
    end
    return ps
end


"""
    gradientsdict(model::Model, gs::Grads) -> Dict{Symbol, Array}

Return a Dict with mapping parameter name => gradient value

Useful for inspecting the distribution of gradients.
"""
function gradientsdict(model::Model, gs::Grads; filterfn = _ -> true)
    return Dict(
        name => [gs[p] for p in ps_]
        for (name, ps_) in paramsdict(model, filterfn = filterfn)
    )
end


"""
    layerstats(model, lossfn, batch, [filter_fn])

Return the names, activations and gradients of every layer
where `filter_fn(layer) == true`.

Gradients and activations are calculated on `batch` and with `lossfn`
"""
function layerstats(model, lossfn, batch; filter_fn = l -> true)
    x, y = batch
    hmodel = addhook(model, forward = hooksaveactivations, backward = hooksavegrads, filter_fn = filter_fn)

    gradient(params(hmodel)) do
        return lossfn(hmodel(x), y)
    end

    names = []
    acts = []
    gs = []

    for layer in IterLayers(hmodel)
        if layer isa Hook
            push!(names, string(layer.layer))
            push!(acts, layer.state["activation"])
            push!(gs, layer.state["gradient"])
        end
    end

    return names, acts, gs
end
