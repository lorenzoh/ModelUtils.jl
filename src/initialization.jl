using Flux


"""
    init!(a, initializer)

Initialize an array `a` with `initializer`.

Example

```julia
layer = Conv((3, 3), 3 => 16)
init!(layer.weight, init_kaiming_normal)
```
"""
init!(a::AbstractArray{T}, initializer) where T = copy!(a, initializer(T, size(a)...))

# initializers

"""
    init_kaiming_uniform(T, dims...; gain=sqrt(2))

Kaiming uniform initializer
"""
function init_kaiming_uniform(T, dims...; gain=sqrt(2))
  fan_in = length(dims) <= 2 ? dims[end] : div(*(dims...), dims[end])
  bound = sqrt(3.0) * gain / sqrt(fan_in) |> Float32
  return rand(Float32, dims...) .* (2bound) .+ -bound
end
init_kaiming_uniform(dims...; gain = sqrt(2)) = init_kaiming_uniform(Float32, dims...; gain = gain)

"""
    init_normal_uniform(T, dims...; gain=sqrt(2))

Kaiming normal initializer
"""
function init_kaiming_normal(T, dims...; gain=sqrt(2))
  fan_in = length(dims) <= 2 ? dims[end] : div(*(dims...), dims[end])
  std = gain / sqrt(fan_in) |> T
  return randn(T, dims...) .* std
end
init_kaiming_normal(dims...; gain = sqrt(2)) = init_kaiming_normal(Float32, dims...; gain = gain)

init_zeros = Flux.zeros


"""
    Initialization(layertype, field, initializer)

Describes an initialization for field `field` of all layers of type
`layertype` to be initialized with `initializer`.

Use [`initmodel!`](@ref) to apply to a model.

## Example

`Initialization(Conv, :bias, init_zeros)` -> initialize the bias of all
Conv layers with zeros.
"""
struct Initialization
    layertype::Type
    field::Symbol
    initializer
end

"""
    initmodel!(model, inits::Vector{Initialization})

Applies `inits` to `model`.

See [`Initialization`](@ref).

## Example

```julia
model = Chain(Dense(10, 10), Dense(10, 10))
initmodel!(model, [
    Initialization(Dense, :W, init_kaiming_normal),
    Initialization(Dense, :bias, init_zeros)
])
```
"""
function initmodel!(model, inits::AbstractVector{Initialization})
    for layer in IterLayers(model)
        for init in inits
            if (layer isa init.layertype) && hasfield(typeof(layer), init.field)
                init!(getfield(layer, init.field), init.initializer)
            end
        end
    end
    return model
end
