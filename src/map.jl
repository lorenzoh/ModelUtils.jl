"""
    mapmodel(f, model::M, [constructor])

Applies `f` to every layer, resulting in a new model.

    !!!warning

The model layer structs are copied in the process, but the
fields are not, hence the resulting model will have the
same parameters!

    !!!warning

The copying is done by first recursively applying `mapmodel` to
all children fields and then attempting to create a new object
of type `M` with the default outer constructor `M(fields...)`.

If your custom type does not have a constructor of that form,
like e.g. `Flux.Chain`, you will have to extend this function
with a signature like

    `mapmodel(f, model::MyType, (fields) -> ...)`

where the last argument results in a `MyType(fields...)`
"""
function mapmodel(f, model::M, constructor = M) where M
    fields = []
    for (name, T) in zip(fieldnames(M), fieldtypes(M))
        value::T = getfield(model, name)
        if attrtype(model, Val(name), value) == ModelUtils.ChildAttr
            push!(fields, mapmodel(f, value))
        else
            push!(fields, value)
        end
    end
    return f(constructor(fields...))
end

mapmodel(f, t::T) where T<:Tuple = (x -> mapmodel(f, x)).(t)
mapmodel(f, g::Function) = g
mapmodel(f, model::C) where C<:Chain = mapmodel(f, model, (layers) -> Chain(layers...))
mapmodel(f, model::M) where M<:Model = Model(mapmodel(f, model.layer))
