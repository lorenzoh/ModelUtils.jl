using Flux


init!(a::AbstractArray{T}, initfn) where T = copy!(a, initfn(T, size(a)...))

# initializers

function init_kaiming_uniform(T, dims...; gain=sqrt(2))
  fan_in = length(dims) <= 2 ? dims[end] : div(*(dims...), dims[end])
  bound = sqrt(3.0) * gain / sqrt(fan_in) |> Float32
  return rand(Float32, dims...) .* (2bound) .+ -bound
end
init_kaiming_uniform(dims...; gain = sqrt(2)) = init_kaiming_uniform(Float32, dims...; gain = gain)

function init_kaiming_normal(T, dims...; gain=sqrt(2))
  fan_in = length(dims) <= 2 ? dims[end] : div(*(dims...), dims[end])
  std = gain / sqrt(fan_in) |> T
  return randn(T, dims...) .* std
end
init_kaiming_normal(dims...; gain = sqrt(2)) = init_kaiming_normal(Float32, dims...; gain = gain)

init_zeros = Flux.zeros


struct Initialization
    layertype
    field::Symbol
    initfn
end

function initmodel!(model, inits::AbstractVector{Initialization})
    for layer in IterLayers(model)
        for init in inits
            if (layer isa init.layertype) && hasfield(typeof(layer), init.field)
                init!(getfield(layer, init.field), init.initfn)
            end
        end
    end
    return model
end
