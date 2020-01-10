# Getting started

!!! note

    This documentation assumes familiarity with Flux.jl and
    deep learning models in general.

The most important type ModelUtils.jl defines is [`Model`](@ref).

It is a tree structure that wraps any Flux or custom layer,
recursively wrapping all children layers too.

Let's start with a simple example model:

```julia
using Flux: BatchNorm, Conv, Chain

chain = Chain(Conv((3, 3), 3 => 16), BatchNorm(16))
```

This pure-Flux model has an implicit tree structure, namely:

- Chain
    - Conv
    - BatchNorm

Deep learning models often get very large and have multiple levels of hierarchy in their layers, so making this tree structure explicit helps to visualize and inspect a model.

We can do this with the [`Model`](@ref) constructor:

```julia
using ModelUtils: Model, printmodel

model = Model(chain)
```

This enables a lot of functionality, for example `printmodel(model)` outputs:

```
Chain, 2 direct children
├─ Conv, params (:weight, :bias), settings (:σ, :stride, :pad, :dilation)
└─ BatchNorm, params (:β, :γ), settings (:λ, :ϵ, :momentum)
```

You can see that the tree structure is indicated and that we get some additional information about each wrapped layer's fields.

If you want to customize this behavior, see [Custom models](@ref).

## Iteration

We can also iterate over a `Model` in 2 ways:

The first way is to iterate over the `Model` wrappers with [`IterModels`](@ref):

```@example 1
using Flux, ModelUtils # hide
```

```@example 1
model = Model(Chain(Conv((3, 3), 3 => 16), BatchNorm(16)))
collect(IterModels(model))
```

But it is also possible to iterate over the wrapped layers directly with [`IterLayers`](@ref):

```@example 1
model = Model(Chain(Conv((3, 3), 3 => 16), BatchNorm(16)))
collect(IterLayers(model))
```