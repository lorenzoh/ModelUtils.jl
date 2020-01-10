# Initialization


```@meta
CurrentModule = ModelUtils
```

`ModelUtils.jl` makes it easy to selectively initialize weights in your model even after it is created.

Flux's default initialization for a `Conv` layer's `weight` field is "Glorot/Xavier uniform". However, almost always one will use ReLU activations for CNNs, and it is known that "Kaiming/He" initialization works better for ReLU activations.

That is why `ModelUtils.jl` exports [`init_kaiming_normal`](@ref) and [`init_kaiming_uniform`](@ref).

Let's see how we can reinitialize a small CNN with Kaiming initialization instead of Glorot.

```@example
using Flux, ModelUtils # hide
cnn = Chain(Conv((3, 3), 3 => 16, relu), Conv((3, 3), 16 => 16))
inits = [Initialization(Conv, :weight, init_kaiming_normal)]
initmodel!(cnn, inits)
```