using Flux
using Flux: trainable

using AbstractTrees
using AbstractTrees: children

"""
    Model(layer)

Wrapper for Flux layer that recursively wraps children layers.

Use `model.layer` to access wrapped layer

Use `children(model)` to access children wrappers
"""
struct Model{T}
    layer::T
    children::AbstractVector{Model}
    paramattrs::AbstractVector{Symbol}
    settingattrs::AbstractVector{Symbol}
    childrenattrs::AbstractVector{Symbol}
    stateattrs::AbstractVector{Symbol}
end

Model(layer::T, args...) where T = Model{T}(layer, args...)

(model::Model)(x) = model.layer(x)
Flux.trainable(model::Model) = (model.layer,)


totuple(t::Tuple) = t
totuple(t::NTuple) = t
totuple(x) = (x,)


function Model(layer::T) where T
    fields = Dict(
        ParamAttr => [],
        SettingAttr => [],
        ChildAttr => [],
        StateAttr => [],
    )
    for name in fieldnames(T)
        value = getfield(layer, name)
        ft = attrtype(layer, Val(name), value)
        push!(fields[ft], name)
    end

    chs = Model[]
    for field in fields[ChildAttr]
        for child in totuple(getfield(layer, field))
            push!(chs, Model(child))
        end
    end

    return Model{T}(
        layer,
        chs,
        fields[ParamAttr],
        fields[SettingAttr],
        fields[ChildAttr],
        fields[StateAttr],
    )
end



# Iteration

AbstractTrees.children(model::Model) = model.children

"""
    IterModels(model::Model)

Pre-order (parents before children) iterator over all `Model`s.
If you want to iterate over the wrapped layers instead, use
[`IterLayers`](@ref).

Example
```julia
> model = Model(Chain(Conv((3, 3), 3 => 16), BatchNorm(16)))
> collect(IterModels(model))

[Model{Chain{...}}, Model{Conv{...}}, Model{BatchNorm{...}}]
```
"""
IterModels(model::Model) = PreOrderDFS(model)

"""
    IterLayers(model::Model)

Pre-order (parents before children) iterator over all wrapped layers.

Example
```julia
> model = Model(Chain(Conv((3, 3), 3 => 16), BatchNorm(16)))
> collect(IterModels(model))

[Chain{...}, Conv{...}, BatchNorm{...}]
```
"""
IterLayers(model::Model) = (m.layer for m in IterModels(model))
IterLayers(model) = (m.layer for m in IterModels(Model(model)))

numleaves(tree) = filter(m -> m !== tree, collect(Leaves(tree))) |> length

# Printing

function Base.show(io::IO, li::Model{T}) where T
    s = "$(T.name)"
    if length(li.paramattrs) > 0
        s = string(s, ", params $(Tuple(li.paramattrs))")
    end
    if length(li.settingattrs) > 0
        s = string(s, ", settings $(Tuple(li.settingattrs))")
    end
    if length(li.childrenattrs) > 0
        nch = length(li.children)
        s = string(s, ", $(nch) direct children")
    end
    print(io, s)
end

printmodel(model::Model) = print_tree(model)
