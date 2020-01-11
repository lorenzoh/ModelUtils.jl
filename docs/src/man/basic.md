# Getting started

!!! note

    This documentation assumes familiarity with Flux.jl and
    deep learning models in general.

The most important type ModelUtils.jl defines is [`Model`](@ref).

It is a tree structure that wraps any Flux or custom layer,
recursively wrapping all children layers too.

To see what that is useful for, let's look at a simple Flux model:

```@example 1
using Flux: BatchNorm, Conv, Chain

chain = Chain(Conv((3, 3), 3 => 16), BatchNorm(16))
nothing # hide
```

This model has an implicit tree structure, namely:

- Chain
    - Conv
    - BatchNorm

Deep learning models often get very large and have multiple levels of hierarchy in their layers, so making this tree structure explicit helps to visualize and inspect a model.

We can do this with the [`Model`](@ref) constructor:

```@example 1
using ModelUtils: Model, printmodel

model = Model(chain)
```

Note that [`Model`](@ref) can be called like any layer, and simply passes the input to the wrapped layer:

```@example 1
x = randn(8, 8, 3, 1)
@assert chain(x) ≈ model(x) 
```

Now we can show the model's tree structure with [`printmodel`](@ref):

```@example 1
printmodel(model)
```

You can see that the tree structure is indicated and that we get some additional information about each wrapped layer's fields.

!!! note

    Many functions in `ModelUtils.jl` will work with unwrapped Flux models, but will wrap and unwrap the model in the function call. Since constructing the tree takes some time and memory, it is more performant to keep your model wrapped in a [`Model`](@ref).

!!! note

    `ModelTools.jl` will try to automatically detect which fields are trainable parameters, settings, or children layers.
    This should work for most custom layers, too, but if the result is not what you expect you can customize the behavior, see [Custom models](@ref).

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