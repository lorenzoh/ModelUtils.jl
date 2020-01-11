# Hooks

Hooks are a feature that lets you add functions to a layer that are called on either the forward or the backward pass.

Since Flux doesn't make any assumptions about your layers except them being callable as functions, Hooks are implemented as a wrapper layer, [`Hook`](@ref).

In most cases you will have an existing model that you want to add a hook to, and [`addhook`](@ref) does just that:

```@example hooks
using Flux, ModelUtils # hide
model = Chain(Conv((3, 3), 3 => 16, relu), Conv((3, 3), 16 => 1, relu))
hmodel = addhook(
    model,
    forward = (state, activation) -> println(summary(activation)),
    backward = (state, gradient) -> println(summary(gradient)),
    filter_fn = (layer) -> layer isa Conv
)
printmodel(hmodel)
```

