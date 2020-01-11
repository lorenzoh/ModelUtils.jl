# Initialization


```@meta
CurrentModule = ModelUtils
```

`ModelUtils.jl` makes it easy to selectively initialize weights in your model even after it is created.

The simplest case of initialization is when all layers of the same kind are initialized the same. We can represent that with an [`Initialization`](@ref) object.

For example, `Initialization(Dense, :W, Flux.zeros)` means, for all `Dense` layers `dense`, initialize `dense.W` with initializer `Flux.zeros`.

These initalizations can be applied with [`initmodel!`](@ref).

## Example

Let's look at a real example where custom initialization is necessary.

Flux's default initialization for a `Conv` layer's `weight` field is "Glorot/Xavier uniform". However, almost always one will use ReLU activations for CNNs, and it is known that "Kaiming/He" initialization works better for ReLU activations.

That is why `ModelUtils.jl` exports [`init_kaiming_normal`](@ref) and [`init_kaiming_uniform`](@ref).

Let's see how we can reinitialize a small CNN with Kaiming initialization instead of Glorot.

```@example
using Flux, ModelUtils # hide
cnn = Chain(Conv((3, 3), 3 => 16, relu), Conv((3, 3), 16 => 16))
inits = [Initialization(Conv, :weight, init_kaiming_normal)]
initmodel!(cnn, inits)
```

## Custom initializations

If you want more control over which layers are initialized how, use [`IterLayers`](@ref) to iterate through your layers and initialize them one-by-one with [`init!`](@ref)



## Initialization reference

### Helpers

- [`init!`](@ref)
- [`initmodel!`](@ref)

### Initializers

- [`init_kaiming_normal`](@ref)
- [`init_kaiming_uniform`](@ref)