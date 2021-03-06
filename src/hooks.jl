using Flux
import Flux: trainable
using Zygote: hook
using Statistics: mean, std

"""
    Hook(layer, [forward, backward, state = Dict()])

Wraps a Flux layer, and holds a Dict with state (`Hook.state`)

After the forward pass of the wrapped layer calls `Hook.forward(state, layeroutput)`
After the gradient calculation of the wrapped layer calls `Hook.backward(state, layergradient)`

For adding hooks to existing models, see [`addhook`](@ref)
"""
struct Hook
    layer
    forward
    backward
    state
end

Hook(layer; forward = const_, backward = const_, state = Dict()) = Hook(
        layer, forward, backward, state)
const_(_, x) = x


(hook::Hook)(x) = hook.forward(hook.state, Zygote.hook(x̄ -> hook.backward(hook.state, x̄), hook.layer(x)))
trainable(hook::Hook) = (hook.layer,)

"""
    addhook(model; [forward, backward, filter_fn])

Add a [`Hook`](@ref) with callbacks `forward` and `backward` to all layers
of `model` where `filter_fn(layer) == true`.

See [`Hook`](@ref)
"""
function addhook(model; forward = const_, backward = const_, filter_fn = _ -> true)
    mapmodel(model) do layer
        filter_fn(layer) ? Hook(layer, forward = forward, backward = backward) : layer
    end
end

function removehooks(model)
    mapmodel(l -> l isa Hook ? l.layer : l, model)
end

# hook functions

function forwardstatshook(state::Dict, output)
    state["activation_mean"] = mean(output)
    state["activation_std"] = std(output)
    return output
end

function hooksaveactivations(state::Dict, output)
    state["activation"] = output
    return output
end

function hooksavegrads(state::Dict, gradients)
    state["gradient"] = gradients
    return gradients
end


function backwardstatshook(state::Dict, gradients)
    state["gradient_mean"] = mean(gradients)
    state["gradient_std"] = std(gradients)
    return gradients
end
